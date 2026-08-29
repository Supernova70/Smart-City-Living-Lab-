from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit

from app.features import FEATURE_NAMES, extract_features, feature_vector
from app.model_service import ISSUE_TYPES


ROOT = Path(__file__).resolve().parent.parent


def generate_base_image(rng: np.random.Generator, size: int = 256) -> Image.Image:
    x = np.linspace(0, 1, size, dtype=np.float32)
    y = np.linspace(0, 1, size, dtype=np.float32)[:, None]
    color_a = rng.uniform(0.08, 0.72, 3)
    color_b = rng.uniform(0.25, 0.95, 3)
    blend = np.clip((rng.uniform(0.2, 0.8) * x[None, :] + rng.uniform(0.2, 0.8) * y), 0, 1)
    canvas = color_a[None, None, :] * (1 - blend[:, :, None]) + color_b[None, None, :] * blend[:, :, None]
    texture = rng.normal(0, rng.uniform(0.005, 0.025), canvas.shape)
    canvas = np.clip(canvas + texture, 0, 1)
    image = Image.fromarray((canvas * 255).astype(np.uint8), "RGB")

    draw = ImageDraw.Draw(image, "RGBA")
    for _ in range(int(rng.integers(8, 24))):
        x1, y1 = rng.integers(0, size - 20, 2)
        width, height = rng.integers(12, 90, 2)
        x2, y2 = min(size - 1, x1 + width), min(size - 1, y1 + height)
        color = tuple(int(v) for v in rng.integers(0, 256, 3)) + (int(rng.integers(70, 210)),)
        if rng.random() < 0.5:
            draw.ellipse((int(x1), int(y1), int(x2), int(y2)), fill=color)
        else:
            draw.rectangle((int(x1), int(y1), int(x2), int(y2)), fill=color)
    for _ in range(int(rng.integers(4, 12))):
        points = [tuple(map(int, rng.integers(0, size, 2))) for _ in range(3)]
        draw.line(points, fill=tuple(int(v) for v in rng.integers(0, 256, 3)) + (180,), width=int(rng.integers(1, 5)))
    return image


def apply_degradation(
    image: Image.Image, issue: str, severity: float, rng: np.random.Generator
) -> Image.Image:
    severity = float(np.clip(severity, 0.15, 1.0))
    if issue == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=0.8 + 5.2 * severity))
    if issue == "underexposure":
        return ImageEnhance.Brightness(image).enhance(0.72 - 0.58 * severity)
    if issue == "overexposure":
        bright = ImageEnhance.Brightness(image).enhance(1.0 + 1.8 * severity)
        return Image.blend(bright, Image.new("RGB", image.size, "white"), 0.08 + 0.30 * severity)
    if issue == "noise":
        array = np.asarray(image, dtype=np.float32)
        sigma = 8.0 + 48.0 * severity
        noisy = np.clip(array + rng.normal(0, sigma, array.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, "RGB")
    if issue == "severe_degradation":
        small = max(12, int(image.width * (0.32 - 0.25 * severity)))
        pixelated = image.resize((small, small), Image.Resampling.BILINEAR).resize(
            image.size, Image.Resampling.NEAREST
        )
        stream = io.BytesIO()
        pixelated.save(stream, format="JPEG", quality=max(5, int(38 - 30 * severity)))
        return Image.open(io.BytesIO(stream.getvalue())).convert("RGB")
    if issue == "potential_defect":
        result = image.copy()
        draw = ImageDraw.Draw(result, "RGBA")
        x1 = int(rng.integers(20, image.width - 70))
        y1 = int(rng.integers(20, image.height - 70))
        length = int(35 + 100 * severity)
        color = (20, 15, 18, int(150 + 100 * severity))
        draw.line((x1, y1, min(image.width - 10, x1 + length), min(image.height - 10, y1 + int(rng.integers(-25, 26)))), fill=color, width=int(3 + 12 * severity))
        if rng.random() < 0.6:
            patch_size = int(12 + 42 * severity)
            draw.rectangle((x1, y1, x1 + patch_size, y1 + patch_size), fill=(220, 35, 60, 100))
        return result
    raise ValueError(f"Unknown issue type: {issue}")


def build_dataset(base_count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_rows: list[np.ndarray] = []
    y_issues: list[list[int]] = []
    y_scores: list[float] = []
    groups: list[int] = []

    for base_id in range(base_count):
        base = generate_base_image(rng)
        samples: list[tuple[Image.Image, list[str], float]] = [(base, [], 100.0)]
        for issue in ISSUE_TYPES:
            severity = float(rng.uniform(0.35, 1.0))
            samples.append((apply_degradation(base, issue, severity, rng), [issue], 100 - 62 * severity))

        for _ in range(2):
            first, second = rng.choice(ISSUE_TYPES, size=2, replace=False)
            severity_a, severity_b = rng.uniform(0.3, 0.9, 2)
            mixed = apply_degradation(base, str(first), float(severity_a), rng)
            mixed = apply_degradation(mixed, str(second), float(severity_b), rng)
            score = 100 - 42 * float(severity_a) - 42 * float(severity_b)
            samples.append((mixed, [str(first), str(second)], max(5.0, score)))

        for sample, labels, score in samples:
            stats = extract_features(sample)
            x_rows.append(feature_vector(stats))
            y_issues.append([int(issue in labels) for issue in ISSUE_TYPES])
            y_scores.append(score)
            groups.append(base_id)

    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_issues, dtype=np.int8),
        np.asarray(y_scores, dtype=np.float32),
        np.asarray(groups, dtype=np.int32),
    )


def grouped_split(groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(groups))
    split = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, remaining_idx = next(split.split(indices, groups=groups))
    second = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed + 1)
    val_local, test_local = next(second.split(remaining_idx, groups=groups[remaining_idx]))
    return train_idx, remaining_idx[val_local], remaining_idx[test_local]


def evaluate(
    issue_model: RandomForestClassifier,
    score_model: RandomForestRegressor,
    x: np.ndarray,
    y_issues: np.ndarray,
    y_scores: np.ndarray,
) -> dict[str, object]:
    predicted = issue_model.predict(x)
    score_predictions = np.clip(score_model.predict(x), 0, 100)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_issues, predicted, average=None, zero_division=0
    )
    per_issue = {
        issue: {
            "precision": round(float(precision[index]), 4),
            "recall": round(float(recall[index]), 4),
            "f1": round(float(f1[index]), 4),
            "support": int(support[index]),
        }
        for index, issue in enumerate(ISSUE_TYPES)
    }
    return {
        "exact_match_accuracy": round(float(accuracy_score(y_issues, predicted)), 4),
        "macro_f1": round(float(f1_score(y_issues, predicted, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(y_issues, predicted, average="micro", zero_division=0)), 4),
        "quality_mae": round(float(mean_absolute_error(y_scores, score_predictions)), 3),
        "quality_rmse": round(float(mean_squared_error(y_scores, score_predictions) ** 0.5), 3),
        "per_issue": per_issue,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the deadline-safe image-quality ML model.")
    parser.add_argument("--base-count", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "image_quality_model.joblib")
    parser.add_argument("--metrics", type=Path, default=ROOT / "artifacts" / "metrics.json")
    args = parser.parse_args()

    x, y_issues, y_scores, groups = build_dataset(args.base_count, args.seed)
    train_idx, val_idx, test_idx = grouped_split(groups, args.seed)

    issue_model = RandomForestClassifier(
        n_estimators=220,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,
        random_state=args.seed,
    )
    score_model = RandomForestRegressor(
        n_estimators=220,
        max_depth=16,
        min_samples_leaf=2,
        n_jobs=1,
        random_state=args.seed,
    )
    issue_model.fit(x[train_idx], y_issues[train_idx])
    score_model.fit(x[train_idx], y_scores[train_idx])

    metrics = {
        "model_name": "hybrid_cv_random_forest",
        "model_version": "1.0.0",
        "training_source": "controlled synthetic degradations from procedurally generated clean images",
        "seed": args.seed,
        "base_image_count": args.base_count,
        "sample_count": int(len(x)),
        "split_samples": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "validation": evaluate(issue_model, score_model, x[val_idx], y_issues[val_idx], y_scores[val_idx]),
        "test": evaluate(issue_model, score_model, x[test_idx], y_issues[test_idx], y_scores[test_idx]),
        "limitations": [
            "Training images are procedurally generated and do not represent all natural-image content.",
            "Potential defect is trained as a local anomaly proxy, not product-specific defect certification.",
            "KADID-10k training is recommended when deadline and download time permit.",
        ],
    }
    thresholds = {issue: 0.45 for issue in ISSUE_TYPES}
    # The local-anomaly class is visually diverse and had lower recall at 0.5;
    # use a documented recall-oriented threshold for screening.
    thresholds["potential_defect"] = 0.30
    artifact = {
        "issue_model": issue_model,
        "score_model": score_model,
        "feature_names": FEATURE_NAMES,
        "issue_types": ISSUE_TYPES,
        "thresholds": thresholds,
        "model_name": "hybrid_cv_random_forest",
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output, compress=3)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
