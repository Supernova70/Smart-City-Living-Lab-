from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset

from app.model_service import ISSUE_TYPES


KADID_DISTORTION_TO_ISSUE: dict[int, str] = {
    1: "blur",
    2: "blur",
    3: "blur",
    9: "severe_degradation",
    10: "severe_degradation",
    11: "noise",
    12: "noise",
    13: "noise",
    14: "noise",
    15: "noise",
    16: "overexposure",
    17: "underexposure",
    19: "severe_degradation",
    20: "severe_degradation",
    21: "severe_degradation",
    22: "severe_degradation",
    23: "severe_degradation",
}


@dataclass(frozen=True)
class RealRecord:
    path: str
    source: str
    split: str
    group: str
    quality_score: float
    quality_mask: int
    issues: tuple[int, ...]
    issue_mask: tuple[int, ...]
    metadata: dict[str, str | int | float]


def _stable_bucket(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 100


def _split_for_group(group: str, seed: int, train_percent: int = 70, val_percent: int = 15) -> str:
    bucket = _stable_bucket(group, seed)
    if bucket < train_percent:
        return "train"
    if bucket < train_percent + val_percent:
        return "validation"
    return "test"


def _find_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str:
    normalized = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        for normalized_name, original in normalized.items():
            if candidate in normalized_name:
                return original
    raise ValueError(f"Missing one of columns {candidates}; found {list(columns)}")


def build_kadid_records(root: Path, seed: int = 42) -> list[RealRecord]:
    csv_paths = list(root.rglob("dmos.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No dmos.csv found under {root}")
    with csv_paths[0].open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        image_col = _find_column(columns, ("dist_img", "image", "filename", "name"))
        score_col = _find_column(columns, ("dmos", "mos", "score"))
        rows = list(reader)

    images = {path.name.lower(): path for path in root.rglob("*.png")}
    records: list[RealRecord] = []
    for row in rows:
        filename = Path(row[image_col].strip()).name
        image_path = images.get(filename.lower())
        match = re.fullmatch(r"I(\d{2})_(\d{2})_(\d{2})\.png", filename, re.IGNORECASE)
        if image_path is None or match is None:
            continue
        reference_id, distortion_id, level = (int(value) for value in match.groups())
        raw_mos = float(row[score_col])
        quality = 25.0 * (raw_mos - 1.0) if 1.0 <= raw_mos <= 5.0 else raw_mos
        positive_issue = KADID_DISTORTION_TO_ISSUE.get(distortion_id)
        issues = tuple(int(issue == positive_issue) for issue in ISSUE_TYPES)
        issue_mask = tuple(int(issue != "potential_defect") for issue in ISSUE_TYPES)
        group = f"kadid-reference-{reference_id:02d}"
        records.append(
            RealRecord(
                path=str(image_path.resolve()),
                source="kadid10k",
                split=_split_for_group(group, seed),
                group=group,
                quality_score=max(0.0, min(100.0, quality)),
                quality_mask=1,
                issues=issues,
                issue_mask=issue_mask,
                metadata={"filename": filename, "distortion_id": distortion_id, "level": level},
            )
        )
    if len(records) < 10_000:
        raise ValueError(f"KADID validation failed: expected at least 10,000 labeled images, found {len(records)}")
    return records


def _mvtec_parts(path: Path, root: Path) -> tuple[str, str, str] | None:
    parts = path.relative_to(root).parts
    for index, part in enumerate(parts):
        if part in {"train", "test"} and index >= 1 and index + 1 < len(parts):
            return parts[index - 1], part, parts[index + 1]
    return None


def build_mvtec_records(root: Path, seed: int = 42) -> list[RealRecord]:
    fiftyone_metadata = list(root.rglob("samples.json"))
    if fiftyone_metadata:
        payload = json.loads(fiftyone_metadata[0].read_text(encoding="utf-8"))
        parsed = []
        repo_root = fiftyone_metadata[0].parent
        for sample in payload.get("samples", []):
            image_path = repo_root / sample["filepath"]
            if not image_path.exists():
                continue
            category = sample.get("category", {}).get("label", "unknown")
            defect_type = sample.get("defect", {}).get("label", "unknown")
            official_split = str(sample.get("split", "test"))
            parsed.append((image_path, category, official_split, defect_type))
        return _records_from_mvtec_parsed(parsed, seed)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    candidates = [path for path in root.rglob("*") if path.suffix.lower() in image_extensions]
    parsed: list[tuple[Path, str, str, str]] = []
    for path in candidates:
        parts = _mvtec_parts(path, root)
        if parts and "ground_truth" not in path.parts:
            parsed.append((path, *parts))
    if len(parsed) < 5_000:
        raise ValueError(f"MVTec validation failed: expected at least 5,000 images, found {len(parsed)}")

    return _records_from_mvtec_parsed(parsed, seed)


def _records_from_mvtec_parsed(
    parsed: list[tuple[Path, str, str, str]], seed: int
) -> list[RealRecord]:
    if len(parsed) < 5_000:
        raise ValueError(f"MVTec validation failed: expected at least 5,000 images, found {len(parsed)}")

    records: list[RealRecord] = []
    defect_index = ISSUE_TYPES.index("potential_defect")
    for path, category, official_split, defect_type in parsed:
        is_defect = int(official_split == "test" and defect_type != "good")
        # The official training normals remain in training. Labeled official test
        # images are deterministically divided for supervised validation/testing.
        if official_split == "train":
            split = "train"
        else:
            split = _split_for_group(f"{category}:{defect_type}:{path.stem}", seed, 50, 25)
        targets = [0] * len(ISSUE_TYPES)
        targets[defect_index] = is_defect
        mask = [0] * len(ISSUE_TYPES)
        mask[defect_index] = 1
        records.append(
            RealRecord(
                path=str(path.resolve()),
                source="mvtec_ad",
                split=split,
                group=f"mvtec:{category}:{path.stem}",
                quality_score=0.0,
                quality_mask=0,
                issues=tuple(targets),
                issue_mask=tuple(mask),
                metadata={
                    "category": category,
                    "official_split": official_split,
                    "defect_type": defect_type,
                },
            )
        )
    return records


def write_manifest(records: list[RealRecord], output: Path, seed: int) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "issue_types": ISSUE_TYPES,
        "record_count": len(records),
        "split_counts": dict(Counter(record.split for record in records)),
        "source_counts": dict(Counter(record.source for record in records)),
        "records": [asdict(record) for record in records],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_manifest(path: Path) -> list[RealRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("issue_types") != ISSUE_TYPES:
        raise ValueError("Manifest issue order does not match application issue order")
    return [
        RealRecord(
            **{
                **row,
                "issues": tuple(row["issues"]),
                "issue_mask": tuple(row["issue_mask"]),
            }
        )
        for row in payload["records"]
    ]


class MultiTaskImageDataset(Dataset):
    def __init__(self, records: list[RealRecord], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records[index]
        with Image.open(record.path) as source:
            image = source.convert("RGB")
        return {
            "image": self.transform(image),
            "issues": torch.tensor(record.issues, dtype=torch.float32),
            "issue_mask": torch.tensor(record.issue_mask, dtype=torch.float32),
            "quality": torch.tensor(record.quality_score / 100.0, dtype=torch.float32),
            "quality_mask": torch.tensor(record.quality_mask, dtype=torch.float32),
            "path": record.path,
            "source": record.source,
        }
