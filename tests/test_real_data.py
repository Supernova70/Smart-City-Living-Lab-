from __future__ import annotations

from pathlib import Path

from app.model_service import ISSUE_TYPES
from scripts.real_data import KADID_DISTORTION_TO_ISSUE, _records_from_mvtec_parsed, _split_for_group


def test_group_split_is_deterministic_and_seeded() -> None:
    first = [_split_for_group(f"reference-{index}", 42) for index in range(100)]
    second = [_split_for_group(f"reference-{index}", 42) for index in range(100)]
    changed = [_split_for_group(f"reference-{index}", 43) for index in range(100)]
    assert first == second
    assert first != changed
    assert {"train", "validation", "test"}.issubset(first)


def test_kadid_mapping_never_claims_a_real_defect() -> None:
    assert KADID_DISTORTION_TO_ISSUE[1] == "blur"
    assert KADID_DISTORTION_TO_ISSUE[16] == "overexposure"
    assert "potential_defect" not in KADID_DISTORTION_TO_ISSUE.values()


def test_mvtec_only_supervises_the_defect_head() -> None:
    parsed = []
    for index in range(5000):
        split = "train" if index < 3500 else "test"
        defect = "good" if index % 3 else "scratch"
        parsed.append((Path(f"image-{index}.png"), "bottle", split, defect))
    records = _records_from_mvtec_parsed(parsed, seed=42)
    defect_index = ISSUE_TYPES.index("potential_defect")
    assert len(records) == 5000
    assert all(sum(record.issue_mask) == 1 for record in records)
    assert all(record.issue_mask[defect_index] == 1 for record in records)
    assert all(record.quality_mask == 0 for record in records)
