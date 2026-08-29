from __future__ import annotations

from pathlib import Path

import numpy as np

from app.model_service import ISSUE_TYPES
from scripts.train_model import apply_degradation, generate_base_image


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    output = ROOT / "sample_images"
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    base = generate_base_image(rng, size=512)
    base.save(output / "01_acceptable.jpg", quality=94)

    for index, issue in enumerate(ISSUE_TYPES, start=2):
        degraded = apply_degradation(base, issue, 0.82, rng)
        degraded.save(output / f"{index:02d}_{issue}.jpg", quality=92)
    print(f"Generated {1 + len(ISSUE_TYPES)} sample images in {output}")


if __name__ == "__main__":
    main()

