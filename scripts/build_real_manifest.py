from __future__ import annotations

import argparse
from pathlib import Path

from scripts.real_data import build_kadid_records, build_mvtec_records, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe manifests for real IQA datasets.")
    parser.add_argument("--kadid-root", type=Path, required=True)
    parser.add_argument("--mvtec-root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/manifests/real_dataset.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = build_kadid_records(args.kadid_root, args.seed)
    if args.mvtec_root:
        records.extend(build_mvtec_records(args.mvtec_root, args.seed))
    result = write_manifest(records, args.output, args.seed)
    print(f"Wrote {result['record_count']} real records to {args.output}")
    print(f"Splits: {result['split_counts']}")
    print(f"Sources: {result['source_counts']}")


if __name__ == "__main__":
    main()
