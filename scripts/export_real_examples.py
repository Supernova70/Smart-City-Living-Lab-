from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from app.deep_model import build_model
from app.model_service import ISSUE_TYPES
from scripts.real_data import load_manifest


def load_checkpoint(path: Path, device: torch.device):
    artifact = torch.load(path, map_location=device, weights_only=False)
    if artifact["issue_types"] != ISSUE_TYPES:
        raise ValueError("Checkpoint issue order does not match application schema")
    model = build_model(artifact["architecture"], len(ISSUE_TYPES), pretrained=False)
    model.load_state_dict(artifact["state_dict"])
    model.to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize((artifact["image_size"], artifact["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                tuple(artifact["normalization"]["mean"]),
                tuple(artifact["normalization"]["std"]),
            ),
        ]
    )
    return artifact, model, transform


@torch.inference_mode()
def predict(image: Image.Image, model, transform, device: torch.device):
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        logits, quality = model(tensor)
    return torch.sigmoid(logits)[0].cpu().tolist(), float(quality[0].cpu()) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Export auditable held-out real-image predictions.")
    parser.add_argument("--manifest", type=Path, default=Path("datasets/manifests/real_dataset.json"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/image_quality_model.pt"))
    parser.add_argument("--output", type=Path, default=Path("real_examples"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact, model, transform = load_checkpoint(args.model, device)
    test_records = [record for record in load_manifest(args.manifest) if record.split == "test"]
    selected = []
    acceptable = sorted(
        (record for record in test_records if record.quality_mask and not any(record.issues)),
        key=lambda record: record.quality_score,
        reverse=True,
    )
    if acceptable:
        selected.append(("acceptable", acceptable[0]))
    for index, issue in enumerate(ISSUE_TYPES):
        examples = [record for record in test_records if record.issue_mask[index] and record.issues[index]]
        if examples:
            selected.append((issue, examples[0]))

    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for label, record in selected:
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        probabilities, quality = predict(image, model, transform, device)
        preview = image.copy()
        preview.thumbnail((960, 960), Image.Resampling.LANCZOS)
        output_path = args.output / f"{len(results) + 1:02d}_{label}.jpg"
        preview.save(output_path, "JPEG", quality=90, optimize=True)
        predicted_issues = [
            issue
            for issue, probability in zip(ISSUE_TYPES, probabilities)
            if probability >= float(artifact["thresholds"][issue])
        ]
        results.append(
            {
                "preview": output_path.name,
                "source_dataset": record.source,
                "source_filename": Path(record.path).name,
                "target_issues": [issue for issue, target in zip(ISSUE_TYPES, record.issues) if target],
                "target_quality": record.quality_score if record.quality_mask else None,
                "predicted_issues": predicted_issues,
                "predicted_quality": round(quality, 2),
                "probabilities": {
                    issue: round(float(probability), 4)
                    for issue, probability in zip(ISSUE_TYPES, probabilities)
                },
            }
        )
    (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Exported {len(results)} held-out real examples to {args.output}")


if __name__ == "__main__":
    main()
