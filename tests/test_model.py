from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.features import extract_features
from app.model_service import ISSUE_TYPES, ModelService
from scripts.kadid_adapter import distortion_id, reference_id


def test_saved_model_loads_and_predicts_ranges() -> None:
    model = ModelService(Path("artifacts/image_quality_model.joblib"))
    model.load()
    result = model.predict(extract_features(Image.new("RGB", (128, 128), "gray")))
    assert 0 <= result["quality_score"] <= 100
    assert list(result["issue_probabilities"]) == ISSUE_TYPES
    assert all(0 <= probability <= 1 for probability in result["issue_probabilities"].values())


def test_kadid_filename_mapping() -> None:
    assert reference_id("I07_16_04.png") == "I07"
    assert distortion_id("I07_16_04.png") == 16
    assert distortion_id("not-kadid.png") is None

