from __future__ import annotations

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from app.deep_model import SUPPORTED_ARCHITECTURES, build_model
from app.model_service import ISSUE_TYPES
from scripts.real_data import MultiTaskImageDataset, RealRecord, load_manifest


ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            # Preserve the full frame: localized MVTec anomalies can disappear
            # under random crops and would create incorrect positive labels.
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def make_loaders(
    records: list[RealRecord], image_size: int, batch_size: int, workers: int
) -> dict[str, DataLoader]:
    train_transform, evaluation_transform = make_transforms(image_size)
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "validation", "test"):
        split_records = [record for record in records if record.split == split]
        if not split_records:
            raise ValueError(f"Manifest has no {split} records")
        dataset = MultiTaskImageDataset(
            split_records,
            train_transform if split == "train" else evaluation_transform,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
            drop_last=split == "train",
        )
    return loaders


def positive_weights(records: list[RealRecord], device: torch.device) -> torch.Tensor:
    values = []
    for index in range(len(ISSUE_TYPES)):
        observed = [record.issues[index] for record in records if record.issue_mask[index]]
        positives = sum(observed)
        negatives = len(observed) - positives
        values.append(min(20.0, max(1.0, negatives / max(1, positives))))
    return torch.tensor(values, dtype=torch.float32, device=device)


def masked_losses(
    logits: torch.Tensor,
    quality: torch.Tensor,
    batch: dict[str, torch.Tensor | list[str]],
    issue_loss: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    issue_targets = batch["issues"]
    issue_mask = batch["issue_mask"]
    quality_targets = batch["quality"]
    quality_mask = batch["quality_mask"]
    assert isinstance(issue_targets, torch.Tensor)
    assert isinstance(issue_mask, torch.Tensor)
    assert isinstance(quality_targets, torch.Tensor)
    assert isinstance(quality_mask, torch.Tensor)

    raw_issue = issue_loss(logits, issue_targets)
    classification = (raw_issue * issue_mask).sum() / issue_mask.sum().clamp_min(1.0)
    raw_quality = nn.functional.smooth_l1_loss(quality, quality_targets, reduction="none", beta=0.08)
    regression = (raw_quality * quality_mask).sum() / quality_mask.sum().clamp_min(1.0)
    return classification + 1.5 * regression, classification, regression


def train_epoch(model, loader, optimizer, scaler, device, issue_loss, epoch: int) -> dict[str, float]:
    model.train()
    totals = np.zeros(3, dtype=np.float64)
    count = 0
    progress = tqdm(loader, desc=f"epoch {epoch:02d} train", leave=False)
    for batch in progress:
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        for key in ("issues", "issue_mask", "quality", "quality_mask"):
            batch[key] = batch[key].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, quality = model(images)
            loss, classification, regression = masked_losses(logits, quality, batch, issue_loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        totals += [loss.item(), classification.item(), regression.item()]
        count += 1
        progress.set_postfix(loss=f"{totals[0] / count:.4f}")
    return {"loss": totals[0] / count, "classification_loss": totals[1] / count, "quality_loss": totals[2] / count}


@torch.inference_mode()
def collect_predictions(model, loader, device) -> dict[str, np.ndarray | list[str]]:
    model.eval()
    collected: dict[str, list] = {
        "probabilities": [], "issues": [], "issue_mask": [],
        "quality": [], "quality_target": [], "quality_mask": [], "paths": [], "sources": [],
    }
    for batch in tqdm(loader, desc="evaluate", leave=False):
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits, quality = model(images)
        collected["probabilities"].append(torch.sigmoid(logits).cpu().numpy())
        collected["issues"].append(batch["issues"].numpy())
        collected["issue_mask"].append(batch["issue_mask"].numpy())
        collected["quality"].append((quality.cpu().numpy() * 100.0))
        collected["quality_target"].append((batch["quality"].numpy() * 100.0))
        collected["quality_mask"].append(batch["quality_mask"].numpy())
        collected["paths"].extend(batch["path"])
        collected["sources"].extend(batch["source"])
    return {
        **{key: np.concatenate(value) for key, value in collected.items() if key not in {"paths", "sources"}},
        "paths": collected["paths"],
        "sources": collected["sources"],
    }


def tune_thresholds(predictions: dict[str, np.ndarray | list[str]]) -> dict[str, float]:
    probabilities = predictions["probabilities"]
    targets = predictions["issues"]
    masks = predictions["issue_mask"]
    assert isinstance(probabilities, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert isinstance(masks, np.ndarray)
    thresholds: dict[str, float] = {}
    for index, issue in enumerate(ISSUE_TYPES):
        observed = masks[:, index].astype(bool)
        if observed.sum() == 0 or np.unique(targets[observed, index]).size < 2:
            thresholds[issue] = 0.5
            continue
        candidates = np.arange(0.15, 0.86, 0.025)
        scores = [f1_score(targets[observed, index], probabilities[observed, index] >= value) for value in candidates]
        thresholds[issue] = round(float(candidates[int(np.argmax(scores))]), 3)
    return thresholds


def calculate_metrics(
    predictions: dict[str, np.ndarray | list[str]], thresholds: dict[str, float]
) -> dict[str, object]:
    probabilities = predictions["probabilities"]
    targets = predictions["issues"]
    masks = predictions["issue_mask"]
    assert isinstance(probabilities, np.ndarray)
    assert isinstance(targets, np.ndarray)
    assert isinstance(masks, np.ndarray)
    per_issue: dict[str, object] = {}
    f1_values = []
    for index, issue in enumerate(ISSUE_TYPES):
        observed = masks[:, index].astype(bool)
        if observed.sum() == 0:
            per_issue[issue] = {"support": 0, "available": False}
            continue
        truth = targets[observed, index].astype(int)
        scores = probabilities[observed, index]
        predicted = (scores >= thresholds[issue]).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
        auc = roc_auc_score(truth, scores) if np.unique(truth).size == 2 else None
        per_issue[issue] = {
            "available": True,
            "observed": int(observed.sum()),
            "positive_support": int(truth.sum()),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
            "roc_auc": round(float(auc), 4) if auc is not None else None,
            "threshold": thresholds[issue],
        }
        if np.unique(truth).size == 2:
            f1_values.append(float(f1))

    quality_mask = predictions["quality_mask"].astype(bool)
    quality_target = predictions["quality_target"][quality_mask]
    quality_prediction = predictions["quality"][quality_mask]
    quality_mae = float(mean_absolute_error(quality_target, quality_prediction))
    quality_rmse = float(math.sqrt(mean_squared_error(quality_target, quality_prediction)))
    macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
    selection_score = 0.70 * macro_f1 + 0.30 * max(0.0, 1.0 - quality_mae / 100.0)
    return {
        "macro_f1": round(macro_f1, 4),
        "quality_mae": round(quality_mae, 3),
        "quality_rmse": round(quality_rmse, 3),
        "selection_score": round(selection_score, 5),
        "per_issue": per_issue,
    }


def failure_cases(
    predictions: dict[str, np.ndarray | list[str]], thresholds: dict[str, float], limit: int = 12
) -> list[dict[str, object]]:
    paths = predictions["paths"]
    sources = predictions["sources"]
    probabilities = predictions["probabilities"]
    targets = predictions["issues"]
    masks = predictions["issue_mask"]
    quality = predictions["quality"]
    quality_target = predictions["quality_target"]
    quality_mask = predictions["quality_mask"].astype(bool)
    assert isinstance(paths, list) and isinstance(sources, list)
    assert isinstance(probabilities, np.ndarray)
    cases: list[dict[str, object]] = []

    quality_indices = np.where(quality_mask)[0]
    if len(quality_indices):
        errors = np.abs(quality[quality_indices] - quality_target[quality_indices])
        for index in quality_indices[np.argsort(errors)[-4:][::-1]]:
            cases.append(
                {
                    "kind": "quality_regression_error",
                    "file": Path(paths[index]).name,
                    "source": sources[index],
                    "target": round(float(quality_target[index]), 2),
                    "prediction": round(float(quality[index]), 2),
                    "absolute_error": round(float(abs(quality[index] - quality_target[index])), 2),
                }
            )

    for issue_index, issue in enumerate(ISSUE_TYPES):
        observed = masks[:, issue_index].astype(bool)
        predicted = probabilities[:, issue_index] >= thresholds[issue]
        wrong = np.where(observed & (predicted != targets[:, issue_index].astype(bool)))[0]
        if len(wrong):
            uncertainty = np.abs(probabilities[wrong, issue_index] - thresholds[issue])
            index = wrong[int(np.argmax(uncertainty))]
            cases.append(
                {
                    "kind": "issue_misclassification",
                    "issue": issue,
                    "file": Path(paths[index]).name,
                    "source": sources[index],
                    "target": int(targets[index, issue_index]),
                    "probability": round(float(probabilities[index, issue_index]), 4),
                    "threshold": thresholds[issue],
                }
            )
    return cases[:limit]


def save_checkpoint(path: Path, model, architecture: str, thresholds, metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "architecture": architecture,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "issue_types": ISSUE_TYPES,
            "thresholds": thresholds,
            "image_size": metadata["image_size"],
            "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            "model_name": f"real_data_{architecture}_multitask",
            "model_version": "2.0.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        },
        path,
    )


def train_candidate(args, architecture: str, records: list[RealRecord], device: torch.device):
    loaders = make_loaders(records, args.image_size, args.batch_size, args.workers)
    model = build_model(architecture, len(ISSUE_TYPES), pretrained=True).to(
        device, memory_format=torch.channels_last
    )
    if args.init_checkpoint and not args.resume:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        if initial.get("architecture") != architecture:
            raise ValueError(
                f"Initial checkpoint architecture {initial.get('architecture')} does not match {architecture}"
            )
        model.load_state_dict(initial["state_dict"])
        print(f"Initialized {architecture} from {args.init_checkpoint}")
    train_records = [record for record in records if record.split == "train"]
    issue_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weights(train_records, device), reduction="none")
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    history = []
    best_score = -1.0
    best_state = None
    best_thresholds = None
    best_validation = None
    stale_epochs = 0
    latest_path = args.checkpoint_dir / f"{architecture}_latest.pt"
    resume_state = None
    if args.resume and latest_path.exists():
        resume_state = torch.load(latest_path, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_state["state_dict"])
        history = resume_state["history"]
        best_score = resume_state["best_score"]
        best_state = resume_state["best_state"]
        best_thresholds = resume_state["best_thresholds"]
        best_validation = resume_state["best_validation"]
        stale_epochs = resume_state["stale_epochs"]
        start_epoch = int(resume_state["epoch"]) + 1
        print(f"Resuming {architecture} at epoch {start_epoch}")
    else:
        start_epoch = 1

    model.freeze_backbone(start_epoch <= args.freeze_epochs)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.head_lr if start_epoch <= args.freeze_epochs else args.learning_rate,
        weight_decay=1e-4,
    )
    if resume_state is not None and start_epoch != args.freeze_epochs + 1:
        optimizer.load_state_dict(resume_state["optimizer_state"])
        scaler.load_state_dict(resume_state["scaler_state"])

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.freeze_epochs + 1 and args.freeze_epochs > 0:
            model.freeze_backbone(False)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=1e-4
            )
        learning_rate = float(optimizer.param_groups[0]["lr"])
        started = time.perf_counter()
        train_metrics = train_epoch(model, loaders["train"], optimizer, scaler, device, issue_loss, epoch)
        validation_predictions = collect_predictions(model, loaders["validation"], device)
        thresholds = tune_thresholds(validation_predictions)
        validation_metrics = calculate_metrics(validation_predictions, thresholds)
        row = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 1),
            "learning_rate": learning_rate,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(row)
        print(json.dumps(row))
        score = float(validation_metrics["selection_score"])
        if score > best_score + 1e-4:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_thresholds = thresholds
            best_validation = validation_metrics
            stale_epochs = 0
        else:
            stale_epochs += 1
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "history": history,
                "best_score": best_score,
                "best_state": best_state,
                "best_thresholds": best_thresholds,
                "best_validation": best_validation,
                "stale_epochs": stale_epochs,
            },
            latest_path,
        )
        if epoch > args.freeze_epochs and stale_epochs >= args.patience:
            print(f"Early stopping {architecture} after epoch {epoch}")
            break

    assert best_state is not None and best_thresholds is not None
    model.load_state_dict(best_state)
    test_predictions = collect_predictions(model, loaders["test"], device)
    test_metrics = calculate_metrics(test_predictions, best_thresholds)
    return model, best_thresholds, best_validation, test_metrics, history, failure_cases(test_predictions, best_thresholds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select a real-data image quality model.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "datasets/manifests/real_dataset.json")
    parser.add_argument("--architectures", nargs="+", choices=SUPPORTED_ARCHITECTURES, default=["mobilenet_v3_small"])
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--head-lr", type=float, default=8e-4)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "artifacts/checkpoints")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/image_quality_model.pt")
    parser.add_argument("--metrics", type=Path, default=ROOT / "artifacts/real_metrics.json")
    args = parser.parse_args()

    seed_everything(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use a CUDA PyTorch wheel or run this script on Colab GPU.")
    device = torch.device("cuda")
    records = load_manifest(args.manifest)
    print(f"Device: {torch.cuda.get_device_name(0)}; records: {len(records)}")

    candidates = []
    best = None
    for architecture in args.architectures:
        model, thresholds, validation, test, history, failures = train_candidate(args, architecture, records, device)
        result = {
            "architecture": architecture,
            "validation": validation,
            "test": test,
            "thresholds": thresholds,
            "history": history,
            "failure_cases": failures,
        }
        candidates.append(result)
        if best is None or validation["selection_score"] > best[1]["selection_score"]:
            best = (model, validation, result)

    assert best is not None
    best_model, _, best_result = best
    split_counts = {split: sum(record.split == split for record in records) for split in ("train", "validation", "test")}
    source_counts = {source: sum(record.source == source for record in records) for source in sorted({r.source for r in records})}
    metadata = {
        "seed": args.seed,
        "image_size": args.image_size,
        "training_sources": source_counts,
        "split_counts": split_counts,
        "validation_metrics": best_result["validation"],
        "test_metrics": best_result["test"],
    }
    save_checkpoint(args.output, best_model, best_result["architecture"], best_result["thresholds"], metadata)
    metrics = {
        "selected_architecture": best_result["architecture"],
        "device": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "manifest": str(args.manifest),
        "dataset": {"records": len(records), "sources": source_counts, "splits": split_counts},
        "candidates": candidates,
        "selected_validation": best_result["validation"],
        "selected_test": best_result["test"],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Selected {best_result['architecture']} -> {args.output}")
    print(json.dumps(best_result["test"], indent=2))


if __name__ == "__main__":
    main()
