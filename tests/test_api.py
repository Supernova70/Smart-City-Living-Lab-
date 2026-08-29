from __future__ import annotations

import io
import shutil
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import settings
from app.main import create_app


def _runtime(name: str) -> Path:
    path = Path("test-output") / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def _test_image() -> bytes:
    image = Image.new("RGB", (192, 128), (70, 110, 155))
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 20, 100, 90), fill=(230, 180, 45))
    draw.line((0, 120, 190, 5), fill=(255, 255, 255), width=4)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _settings(tmp_path: Path):
    return replace(
        settings,
        database_path=tmp_path / "quality.db",
        upload_dir=tmp_path / "uploads",
        model_path=Path("artifacts/image_quality_model.joblib").resolve(),
    )


def test_complete_analysis_and_history_flow() -> None:
    active_settings = _settings(_runtime("complete-flow"))
    with TestClient(create_app(active_settings)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model"] == "ready"

        response = client.post(
            "/api/v1/analyses",
            files={"file": ("sample.png", _test_image(), "image/png")},
        )
        assert response.status_code == 201
        result = response.json()
        assert 0 <= result["quality_score"] <= 100
        assert result["quality_label"] in {"ACCEPTABLE", "DEGRADED", "POTENTIALLY_DEFECTIVE"}
        assert "statistics" in result
        assert "stored_filename" not in result
        analysis_id = result["id"]

        detail = client.get(f"/api/v1/analyses/{analysis_id}")
        assert detail.status_code == 200
        history = client.get("/api/v1/analyses?limit=10&offset=0")
        assert history.status_code == 200
        assert history.json()["total"] == 1
        preview = client.get(f"/api/v1/analyses/{analysis_id}/image")
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/jpeg"

    with TestClient(create_app(active_settings)) as restarted:
        history_after_restart = restarted.get("/api/v1/analyses")
        assert history_after_restart.json()["total"] == 1


def test_invalid_uploads_return_structured_errors() -> None:
    with TestClient(create_app(_settings(_runtime("invalid-uploads")))) as client:
        fake = client.post(
            "/api/v1/analyses",
            files={"file": ("fake.png", b"not an image", "image/png")},
        )
        assert fake.status_code == 422
        assert fake.json()["error"]["code"] == "UNREADABLE_IMAGE"
        assert fake.json()["error"]["request_id"]

        unsupported = client.post(
            "/api/v1/analyses",
            files={"file": ("sample.gif", b"GIF89a", "image/gif")},
        )
        assert unsupported.status_code == 415
        assert unsupported.json()["error"]["code"] in {
            "UNSUPPORTED_EXTENSION",
            "UNSUPPORTED_MEDIA_TYPE",
        }


def test_missing_record_and_bad_pagination() -> None:
    with TestClient(create_app(_settings(_runtime("missing-record")))) as client:
        missing = client.get("/api/v1/analyses/missing")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "ANALYSIS_NOT_FOUND"

        invalid = client.get("/api/v1/analyses?limit=0")
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "INVALID_REQUEST"
