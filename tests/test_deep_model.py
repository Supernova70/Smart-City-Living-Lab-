"""Tests for the deep-model checkpoint, preprocessing, and ModelService integration.

These tests run against artifacts/image_quality_model.pt once Phase E integration
is complete. They are skipped if the checkpoint does not yet exist so that the
existing baseline test suite continues to pass during training.
"""
from __future__ import annotations

import io
import math
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from app.config import settings
from app.model_service import ISSUE_TYPES, ModelService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PT_MODEL = Path("artifacts/image_quality_model.pt")
_JOBLIB_MODEL = Path("artifacts/image_quality_model.joblib")

skip_no_pt = pytest.mark.skipif(
    not _PT_MODEL.exists(),
    reason="artifacts/image_quality_model.pt not yet produced",
)


def _solid_image(color=(128, 128, 128), size=(224, 224)) -> Image.Image:
    return Image.new("RGB", size, color)


def _solid_bytes(color=(128, 128, 128), size=(224, 224), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    _solid_image(color, size).save(buf, format=fmt)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Checkpoint structure
# ---------------------------------------------------------------------------


@skip_no_pt
def test_checkpoint_required_keys_present() -> None:
    import torch

    artifact = torch.load(_PT_MODEL, map_location="cpu", weights_only=False)
    required = {
        "format_version", "architecture", "state_dict", "issue_types",
        "thresholds", "image_size", "normalization", "model_name", "model_version",
    }
    missing = required - set(artifact)
    assert not missing, f"Checkpoint missing keys: {missing}"


@skip_no_pt
def test_checkpoint_issue_types_match_application() -> None:
    import torch

    artifact = torch.load(_PT_MODEL, map_location="cpu", weights_only=False)
    assert artifact["issue_types"] == ISSUE_TYPES, (
        f"Checkpoint issue_types {artifact['issue_types']} != app {ISSUE_TYPES}"
    )


@skip_no_pt
def test_checkpoint_architecture_is_supported() -> None:
    import torch
    from app.deep_model import SUPPORTED_ARCHITECTURES

    artifact = torch.load(_PT_MODEL, map_location="cpu", weights_only=False)
    assert artifact["architecture"] in SUPPORTED_ARCHITECTURES


@skip_no_pt
def test_checkpoint_thresholds_all_issue_types() -> None:
    import torch

    artifact = torch.load(_PT_MODEL, map_location="cpu", weights_only=False)
    thresholds = artifact["thresholds"]
    for issue in ISSUE_TYPES:
        assert issue in thresholds, f"Missing threshold for {issue}"
        assert 0.0 <= thresholds[issue] <= 1.0, f"Threshold out of range: {issue}={thresholds[issue]}"


# ---------------------------------------------------------------------------
# ModelService loading and inference
# ---------------------------------------------------------------------------


@skip_no_pt
def test_model_service_loads_pt_checkpoint() -> None:
    svc = ModelService(_PT_MODEL)
    svc.load()
    assert svc.ready
    assert svc.model_name != "unavailable"
    assert svc.model_version != "unavailable"


@skip_no_pt
def test_model_service_predict_valid_ranges() -> None:
    svc = ModelService(_PT_MODEL)
    svc.load()
    image = _solid_image()
    result = svc.predict(image=image)
    assert 0.0 <= result["quality_score"] <= 100.0
    probs = result["issue_probabilities"]
    assert list(probs.keys()) == ISSUE_TYPES
    for issue, prob in probs.items():
        assert 0.0 <= prob <= 1.0, f"{issue} probability {prob} out of [0,1]"


@skip_no_pt
def test_model_service_predict_all_issue_keys_present() -> None:
    svc = ModelService(_PT_MODEL)
    svc.load()
    result = svc.predict(image=_solid_image())
    assert set(result["issue_probabilities"].keys()) == set(ISSUE_TYPES)
    assert set(result["thresholds"].keys()) == set(ISSUE_TYPES)


@skip_no_pt
def test_model_service_deterministic_inference() -> None:
    """Same image should produce identical prediction when called twice."""
    svc = ModelService(_PT_MODEL)
    svc.load()
    image = _solid_image(color=(100, 150, 200))
    r1 = svc.predict(image=image)
    r2 = svc.predict(image=image)
    assert r1["quality_score"] == r2["quality_score"]
    assert r1["issue_probabilities"] == r2["issue_probabilities"]


@skip_no_pt
def test_model_service_different_images_differ() -> None:
    """Dark vs bright image should produce different predictions."""
    svc = ModelService(_PT_MODEL)
    svc.load()
    dark = svc.predict(image=_solid_image(color=(10, 10, 10)))
    bright = svc.predict(image=_solid_image(color=(245, 245, 245)))
    assert (
        dark["quality_score"] != bright["quality_score"]
        or dark["issue_probabilities"] != bright["issue_probabilities"]
    )


# ---------------------------------------------------------------------------
# Preprocessing compatibility
# ---------------------------------------------------------------------------


@skip_no_pt
def test_preprocessing_matches_training_normalization() -> None:
    """Verify that the checkpoint records the expected ImageNet normalization."""
    import torch

    artifact = torch.load(_PT_MODEL, map_location="cpu", weights_only=False)
    norm = artifact["normalization"]
    expected_mean = (0.485, 0.456, 0.406)
    expected_std = (0.229, 0.224, 0.225)
    for got, exp in zip(norm["mean"], expected_mean):
        assert abs(got - exp) < 1e-4
    for got, exp in zip(norm["std"], expected_std):
        assert abs(got - exp) < 1e-4


# ---------------------------------------------------------------------------
# API integration with .pt model
# ---------------------------------------------------------------------------


@skip_no_pt
def test_api_accepts_upload_with_pt_model(tmp_path) -> None:
    """Full API flow using the deep model checkpoint."""
    from fastapi.testclient import TestClient
    from app.main import create_app

    active_settings = replace(
        settings,
        database_path=tmp_path / "quality.db",
        upload_dir=tmp_path / "uploads",
        model_path=_PT_MODEL.resolve(),
    )
    with TestClient(create_app(active_settings)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model"] == "ready"

        response = client.post(
            "/api/v1/analyses",
            files={"file": ("test.png", _solid_bytes(), "image/png")},
        )
        assert response.status_code == 201
        result = response.json()
        assert 0 <= result["quality_score"] <= 100
        assert result["quality_label"] in {"ACCEPTABLE", "DEGRADED", "POTENTIALLY_DEFECTIVE"}
        assert "issues" in result
        assert "statistics" in result


# ---------------------------------------------------------------------------
# Latency measurement (informational, not a hard gate)
# ---------------------------------------------------------------------------


@skip_no_pt
def test_inference_latency_cpu(capsys) -> None:
    """Measure single-image CPU inference latency (informational)."""
    import time

    svc = ModelService(_PT_MODEL)
    svc.load()
    image = _solid_image()
    # Warm up
    svc.predict(image=image)
    iterations = 5
    start = time.perf_counter()
    for _ in range(iterations):
        svc.predict(image=image)
    elapsed_ms = (time.perf_counter() - start) / iterations * 1000
    with capsys.disabled():
        print(f"\n[INFO] CPU inference latency (avg over {iterations}): {elapsed_ms:.1f} ms")
    # Sanity: 10 seconds per image on CPU would be unreasonably slow
    assert elapsed_ms < 10_000, f"Inference too slow: {elapsed_ms:.0f} ms"
