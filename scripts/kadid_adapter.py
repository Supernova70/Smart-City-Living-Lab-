from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DISTORTION_TO_ISSUES: dict[int, list[str]] = {
    1: ["blur"],
    2: ["blur"],
    3: ["blur"],
    9: ["severe_degradation"],
    10: ["severe_degradation"],
    11: ["noise"],
    12: ["noise"],
    13: ["noise"],
    14: ["noise"],
    15: ["noise"],
    16: ["overexposure"],
    17: ["underexposure"],
    19: ["severe_degradation"],
    20: ["potential_defect"],
    21: ["severe_degradation"],
    22: ["severe_degradation"],
    23: ["potential_defect"],
}


def distortion_id(filename: str) -> int | None:
    match = re.fullmatch(r"I\d{2}_(\d{2})_\d{2}\.png", Path(filename).name, re.IGNORECASE)
    return int(match.group(1)) if match else None


def reference_id(filename: str) -> str | None:
    match = re.match(r"(I\d{2})", Path(filename).name, re.IGNORECASE)
    return match.group(1).upper() if match else None


def _select_column(columns: list[str], candidates: tuple[str, ...]) -> str:
    normalized = {column.lower().strip(): column for column in columns}
    for candidate in candidates:
        for normalized_name, original in normalized.items():
            if candidate in normalized_name:
                return original
    raise ValueError(f"Could not find one of {candidates} in CSV columns {columns}")


def load_records(root: Path) -> list[dict[str, Any]]:
    csv_candidates = list(root.rglob("dmos.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No dmos.csv found below {root}")
    csv_path = csv_candidates[0]

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        filename_column = _select_column(columns, ("dist_img", "image", "filename", "name"))
        score_column = _select_column(columns, ("dmos", "mos", "score"))
        rows = list(reader)

    image_lookup = {path.name.lower(): path for path in root.rglob("*.png")}
    records: list[dict[str, Any]] = []
    for row in rows:
        filename = row[filename_column].strip()
        image_path = image_lookup.get(Path(filename).name.lower())
        if not image_path:
            continue
        distortion = distortion_id(filename)
        raw_score = float(row[score_column])
        score = 25.0 * (raw_score - 1.0) if 1.0 <= raw_score <= 5.0 else raw_score
        records.append(
            {
                "path": str(image_path),
                "filename": filename,
                "reference_id": reference_id(filename),
                "distortion_id": distortion,
                "issues": DISTORTION_TO_ISSUES.get(distortion or -1, ["severe_degradation"]),
                "quality_score": round(max(0.0, min(100.0, score)), 3),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and index a downloaded KADID-10k dataset.")
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/processed/kadid_manifest.json"))
    args = parser.parse_args()
    records = load_records(args.root)
    counts = Counter(issue for record in records for issue in record["issues"])
    manifest = {"record_count": len(records), "issue_counts": counts, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Indexed {len(records)} KADID images -> {args.output}")


if __name__ == "__main__":
    main()

