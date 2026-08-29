from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

FEATURE_NAMES = [
    "brightness_mean",
    "brightness_std",
    "luminance_p01",
    "luminance_p05",
    "luminance_p50",
    "luminance_p95",
    "luminance_p99",
    "dark_pixel_ratio",
    "highlight_clip_ratio",
    "contrast_rms",
    "dynamic_range",
    "laplacian_variance",
    "tenengrad",
    "noise_estimate",
    "saturation_mean",
    "saturation_std",
    "low_saturation_ratio",
    "high_saturation_ratio",
    "entropy",
    "blockiness",
    "edge_density",
]


@dataclass
class DecodedImage:
    image: Image.Image
    format: str
    mime_type: str
    width: int
    height: int
    file_size_bytes: int


class ImageValidationError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def decode_image(
    data: bytes,
    filename: str,
    content_type: str | None,
    *,
    max_upload_mb: int = 10,
    max_image_pixels: int = 40_000_000,
) -> DecodedImage:
    if not data:
        raise ImageValidationError("EMPTY_FILE", "The uploaded file is empty.", 400)
    if len(data) > max_upload_mb * 1024 * 1024:
        raise ImageValidationError(
            "FILE_TOO_LARGE", f"The image exceeds the {max_upload_mb} MB upload limit.", 413
        )

    extension = Path(filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageValidationError(
            "UNSUPPORTED_EXTENSION", "Supported file extensions are JPEG, PNG, and WebP.", 415
        )
    if content_type and content_type.lower() not in set(ALLOWED_FORMATS.values()) | {
        "application/octet-stream"
    }:
        raise ImageValidationError(
            "UNSUPPORTED_MEDIA_TYPE", "Supported media types are image/jpeg, image/png, and image/webp.", 415
        )

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = max_image_pixels
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = (probe.format or "").upper()
            if image_format not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    "UNSUPPORTED_IMAGE_FORMAT", "The decoded image is not JPEG, PNG, or WebP.", 415
                )
            if getattr(probe, "n_frames", 1) != 1:
                raise ImageValidationError(
                    "ANIMATED_IMAGE_NOT_SUPPORTED", "Animated or multi-frame images are not supported."
                )
            probe.verify()

        with Image.open(io.BytesIO(data)) as source:
            source.load()
            width, height = source.size
            if width < 64 or height < 64:
                raise ImageValidationError(
                    "IMAGE_TOO_SMALL", "Image dimensions must be at least 64 x 64 pixels."
                )
            if width * height > max_image_pixels:
                raise ImageValidationError(
                    "IMAGE_TOO_LARGE", "Decoded image dimensions exceed the configured safety limit.", 413
                )
            image = ImageOps.exif_transpose(source).convert("RGB")
    except ImageValidationError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageValidationError(
            "IMAGE_TOO_LARGE", "Decoded image dimensions exceed the configured safety limit.", 413
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            "UNREADABLE_IMAGE", "The uploaded file could not be decoded as a supported image."
        ) from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    expected_mime = ALLOWED_FORMATS[image_format]
    return DecodedImage(
        image=image,
        format=image_format,
        mime_type=expected_mime,
        width=image.width,
        height=image.height,
        file_size_bytes=len(data),
    )


def _box_blur_3x3(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="reflect")
    result = np.zeros_like(gray, dtype=np.float32)
    for row in range(3):
        for col in range(3):
            result += padded[row : row + gray.shape[0], col : col + gray.shape[1]]
    return result / 9.0


def _entropy(gray: np.ndarray) -> float:
    histogram, _ = np.histogram(gray, bins=64, range=(0.0, 1.0))
    probabilities = histogram.astype(np.float64)
    probabilities /= max(probabilities.sum(), 1.0)
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum() / math.log2(64))


def _blockiness(gray: np.ndarray) -> float:
    vertical = np.abs(np.diff(gray, axis=1))
    horizontal = np.abs(np.diff(gray, axis=0))
    boundary_v = vertical[:, 7::8]
    boundary_h = horizontal[7::8, :]
    all_mean = (float(vertical.mean()) + float(horizontal.mean())) / 2.0
    if boundary_v.size == 0 or boundary_h.size == 0:
        return 0.0
    boundary_mean = (float(boundary_v.mean()) + float(boundary_h.mean())) / 2.0
    return max(0.0, boundary_mean - all_mean)


def extract_features(image: Image.Image) -> dict[str, float]:
    working = image.convert("RGB")
    if max(working.size) > 1024:
        working.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    rgb = np.asarray(working, dtype=np.float32) / 255.0
    gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

    p01, p05, p50, p95, p99 = np.percentile(gray, [1, 5, 50, 95, 99])
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gx_core = gx[:-1, :]
    gy_core = gy[:, :-1]
    gradient_energy = gx_core * gx_core + gy_core * gy_core

    core = gray[1:-1, 1:-1]
    laplacian = (
        gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
        - 4.0 * core
    )

    local_mean = _box_blur_3x3(gray)
    residual = gray - local_mean
    residual_median = np.median(residual)
    noise_estimate = float(np.median(np.abs(residual - residual_median)) / 0.6745)

    rgb_max = rgb.max(axis=2)
    rgb_min = rgb.min(axis=2)
    saturation = np.divide(
        rgb_max - rgb_min,
        np.maximum(rgb_max, 1e-6),
        out=np.zeros_like(rgb_max),
        where=rgb_max > 1e-6,
    )

    features = {
        "brightness_mean": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "luminance_p01": float(p01),
        "luminance_p05": float(p05),
        "luminance_p50": float(p50),
        "luminance_p95": float(p95),
        "luminance_p99": float(p99),
        "dark_pixel_ratio": float((gray < 0.10).mean()),
        "highlight_clip_ratio": float((gray > 0.90).mean()),
        "contrast_rms": float(gray.std()),
        "dynamic_range": float(p95 - p05),
        "laplacian_variance": float(np.var(laplacian)),
        "tenengrad": float(gradient_energy.mean()),
        "noise_estimate": noise_estimate,
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "low_saturation_ratio": float((saturation < 0.08).mean()),
        "high_saturation_ratio": float((saturation > 0.90).mean()),
        "entropy": _entropy(gray),
        "blockiness": _blockiness(gray),
        "edge_density": float((gradient_energy > 0.02).mean()),
    }
    return {name: round(features[name], 6) for name in FEATURE_NAMES}


def feature_vector(features: dict[str, float]) -> np.ndarray:
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float32)


def issue_evidence(issue_type: str, stats: dict[str, float]) -> list[str]:
    evidence: dict[str, list[str]] = {
        "blur": [
            f"Sharpness response is {stats['laplacian_variance']:.4f}; lower values indicate fewer fine edges.",
            f"Edge density is {stats['edge_density'] * 100:.1f}% of analyzed pixels.",
        ],
        "underexposure": [
            f"Mean luminance is {stats['brightness_mean']:.2f} on a 0-1 scale.",
            f"{stats['dark_pixel_ratio'] * 100:.1f}% of pixels are near black.",
        ],
        "overexposure": [
            f"Mean luminance is {stats['brightness_mean']:.2f} on a 0-1 scale.",
            f"{stats['highlight_clip_ratio'] * 100:.1f}% of pixels are near white.",
        ],
        "noise": [
            f"High-frequency residual noise estimate is {stats['noise_estimate']:.4f}.",
            f"Image entropy is {stats['entropy']:.2f} on a normalized 0-1 scale.",
        ],
        "severe_degradation": [
            f"Block discontinuity score is {stats['blockiness']:.4f}.",
            f"Usable luminance range is {stats['dynamic_range']:.2f}.",
        ],
        "potential_defect": [
            "The learned model found a local irregularity pattern similar to its synthetic anomaly examples.",
            f"Edge density is {stats['edge_density'] * 100:.1f}% and contrast is {stats['contrast_rms']:.2f}.",
        ],
    }
    return evidence.get(issue_type, ["The learned model detected an unusual quality pattern."])


def public_statistics(decoded: DecodedImage, features: dict[str, float]) -> dict[str, Any]:
    return {
        "width": decoded.width,
        "height": decoded.height,
        "format": decoded.format,
        "file_size_bytes": decoded.file_size_bytes,
        **features,
    }

