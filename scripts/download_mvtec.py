from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the licensed MVTec AD mirror reproducibly.")
    parser.add_argument("--output", type=Path, default=Path("datasets/mvtec_ad"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--use-xet",
        action="store_true",
        help="Use Xet transfer. Anonymous Xet requests may receive HTTP 429; regular HTTP is the safe default.",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    if not args.use_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    from huggingface_hub import snapshot_download

    args.output.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="Voxel51/mvtec-ad",
        repo_type="dataset",
        local_dir=args.output,
        allow_patterns=["data/**", "samples.json", "README.md", "license.txt", "metadata.json"],
        max_workers=args.workers,
    )
    print(f"MVTec AD downloaded to {path}")


if __name__ == "__main__":
    main()
