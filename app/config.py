from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Load .env file if it exists (development convenience)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on system env vars


ROOT_DIR = Path(__file__).resolve().parent.parent


def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT_DIR / value


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))
    database_path: Path = _path_from_env("DATABASE_PATH", "data/quality.db")
    upload_dir: Path = _path_from_env("UPLOAD_DIR", "data/uploads")
    model_path: Path = _path_from_env("MODEL_PATH", "artifacts/image_quality_model.pt")
    metrics_path: Path = _path_from_env("METRICS_PATH", "artifacts/metrics.json")
    static_dir: Path = ROOT_DIR / "app" / "static"
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
    history_page_size: int = int(os.getenv("HISTORY_PAGE_SIZE", "20"))

    def ensure_directories(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)


settings = Settings()

