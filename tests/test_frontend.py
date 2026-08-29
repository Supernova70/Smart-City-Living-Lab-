from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app


def test_frontend_and_assets_are_served() -> None:
    runtime = Path("test-output/frontend")
    active_settings = replace(
        settings,
        database_path=runtime / "quality.db",
        upload_dir=runtime / "uploads",
        model_path=Path("artifacts/image_quality_model.joblib").resolve(),
    )
    with TestClient(create_app(active_settings)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "ImageGuard AI" in page.text
        assert 'id="dropZone"' in page.text
        assert 'id="resultSection"' in page.text
        assert 'id="historyList"' in page.text
        assert 'aria-live="polite"' in page.text

        styles = client.get("/static/styles.css")
        script = client.get("/static/app.js")
        assert styles.status_code == 200
        assert script.status_code == 200
        assert "renderResult" in script.text

