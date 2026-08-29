from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageFilter

from app.features import ImageValidationError, decode_image, extract_features


def _png_bytes(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_decode_rejects_empty_and_fake_images() -> None:
    with pytest.raises(ImageValidationError) as empty:
        decode_image(b"", "empty.png", "image/png")
    assert empty.value.code == "EMPTY_FILE"

    with pytest.raises(ImageValidationError) as fake:
        decode_image(b"not an image", "fake.png", "image/png")
    assert fake.value.code == "UNREADABLE_IMAGE"


def test_decode_valid_png() -> None:
    data = _png_bytes(Image.new("RGB", (96, 64), "navy"))
    decoded = decode_image(data, "valid.png", "image/png")
    assert decoded.width == 96
    assert decoded.height == 64
    assert decoded.format == "PNG"


def test_exposure_features_separate_black_and_white() -> None:
    dark = extract_features(Image.new("RGB", (128, 128), (2, 2, 2)))
    bright = extract_features(Image.new("RGB", (128, 128), (253, 253, 253)))
    assert dark["brightness_mean"] < 0.02
    assert dark["dark_pixel_ratio"] > 0.95
    assert bright["brightness_mean"] > 0.98
    assert bright["highlight_clip_ratio"] > 0.95


def test_blur_reduces_sharpness() -> None:
    pattern = np.indices((256, 256)).sum(axis=0) % 2
    rgb = np.repeat((pattern * 255).astype(np.uint8)[:, :, None], 3, axis=2)
    sharp = Image.fromarray(rgb, "RGB")
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=4))
    sharp_stats = extract_features(sharp)
    blurred_stats = extract_features(blurred)
    assert blurred_stats["laplacian_variance"] < sharp_stats["laplacian_variance"]
    assert blurred_stats["tenengrad"] < sharp_stats["tenengrad"]


def test_noise_increases_residual_estimate() -> None:
    rng = np.random.default_rng(42)
    clean_array = np.full((192, 192, 3), 128, dtype=np.uint8)
    noisy_array = np.clip(clean_array.astype(np.int16) + rng.normal(0, 35, clean_array.shape), 0, 255).astype(np.uint8)
    clean = extract_features(Image.fromarray(clean_array, "RGB"))
    noisy = extract_features(Image.fromarray(noisy_array, "RGB"))
    assert noisy["noise_estimate"] > clean["noise_estimate"] + 0.02

