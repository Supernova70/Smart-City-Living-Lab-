from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    quality_score REAL NOT NULL,
    quality_label TEXT NOT NULL,
    issues_json TEXT NOT NULL,
    statistics_json TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    timing_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_sha256 ON analyses(sha256);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    def ready(self) -> bool:
        try:
            with self.connect() as connection:
                connection.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def insert(self, record: dict[str, Any]) -> None:
        payload = {
            **record,
            "created_at": record.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "issues_json": json.dumps(record["issues"]),
            "statistics_json": json.dumps(record["statistics"]),
            "timing_json": json.dumps(record["timing_ms"]),
        }
        columns = (
            "id", "created_at", "original_filename", "stored_filename", "sha256",
            "mime_type", "width", "height", "file_size_bytes", "quality_score",
            "quality_label", "issues_json", "statistics_json", "model_name",
            "model_version", "timing_json",
        )
        placeholders = ", ".join("?" for _ in columns)
        with self.connect() as connection:
            connection.execute(
                f"INSERT INTO analyses ({', '.join(columns)}) VALUES ({placeholders})",
                tuple(payload[column] for column in columns),
            )

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["issues"] = json.loads(item.pop("issues_json"))
        item["statistics"] = json.loads(item.pop("statistics_json"))
        item["timing_ms"] = json.loads(item.pop("timing_json"))
        item["image_url"] = f"/api/v1/analyses/{item['id']}/image"
        return item

    def get(self, analysis_id: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
            ).fetchone()
        return self._deserialize(row) if row else None

    def list(self, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM analyses ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        return [self._deserialize(row) for row in rows], int(total)

