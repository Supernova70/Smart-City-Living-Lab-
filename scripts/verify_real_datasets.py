from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify downloaded real training datasets.")
    parser.add_argument("--kadid-archive", type=Path, default=Path("datasets/kadid10k.zip"))
    parser.add_argument("--kadid-root", type=Path, default=Path("datasets/kadid10k"))
    parser.add_argument("--mvtec-root", type=Path, default=Path("datasets/mvtec_ad"))
    parser.add_argument("--output", type=Path, default=Path("datasets/manifests/integrity.json"))
    args = parser.parse_args()

    all_kadid_images = list(args.kadid_root.rglob("*.png"))
    kadid_images = [
        path for path in all_kadid_images
        if re.fullmatch(r"I\d{2}_\d{2}_\d{2}\.png", path.name, re.IGNORECASE)
    ]
    kadid_pristine = [
        path for path in all_kadid_images
        if re.fullmatch(r"I\d{2}\.png", path.name, re.IGNORECASE)
    ]
    kadid_csv = list(args.kadid_root.rglob("dmos.csv"))
    mvtec_metadata = list(args.mvtec_root.rglob("samples.json"))
    mvtec_images = list(args.mvtec_root.rglob("*.png"))
    if len(kadid_images) != 10_125 or len(kadid_pristine) != 81 or not kadid_csv:
        raise ValueError(
            f"KADID integrity failed: {len(kadid_images)} distorted, "
            f"{len(kadid_pristine)} pristine, {len(kadid_csv)} label files"
        )
    if len(mvtec_images) != 5_354 or not mvtec_metadata:
        raise ValueError(f"MVTec integrity failed: {len(mvtec_images)} PNGs, {len(mvtec_metadata)} metadata files")

    payload = {
        "kadid10k": {
            "archive_bytes": args.kadid_archive.stat().st_size,
            "archive_sha256": sha256(args.kadid_archive),
            "distorted_png_count": len(kadid_images),
            "pristine_png_count": len(kadid_pristine),
            "ignored_metadata_png_count": len(all_kadid_images) - len(kadid_images) - len(kadid_pristine),
            "label_file": str(kadid_csv[0]),
        },
        "mvtec_ad": {
            "png_count": len(mvtec_images),
            "samples_metadata_sha256": sha256(mvtec_metadata[0]),
            "samples_metadata": str(mvtec_metadata[0]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
