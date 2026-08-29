from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app


def test_demonstration_samples_cover_every_required_condition() -> None:
    runtime = Path("test-output/sample-coverage")
    if runtime.exists():
        shutil.rmtree(runtime)
    active_settings = replace(
        settings,
        database_path=runtime / "quality.db",
        upload_dir=runtime / "uploads",
        model_path=Path("artifacts/image_quality_model.joblib").resolve(),
    )
    expected = {
        "01_acceptable.jpg": None,
        "02_blur.jpg": "blur",
        "03_underexposure.jpg": "underexposure",
        "04_overexposure.jpg": "overexposure",
        "05_noise.jpg": "noise",
        "06_severe_degradation.jpg": "severe_degradation",
        "07_potential_defect.jpg": "potential_defect",
    }
    with TestClient(create_app(active_settings)) as client:
        for filename, expected_issue in expected.items():
            path = Path("sample_images") / filename
            with path.open("rb") as stream:
                response = client.post(
                    "/api/v1/analyses",
                    files={"file": (filename, stream, "image/jpeg")},
                )
            assert response.status_code == 201, filename
            result = response.json()
            detected = {issue["type"] for issue in result["issues"]}
            if expected_issue is None:
                assert result["quality_label"] == "ACCEPTABLE"
                assert not detected
            else:
                assert expected_issue in detected, (filename, detected)

