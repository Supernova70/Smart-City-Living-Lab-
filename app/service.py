from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any

from app.config import Settings
from app.database import Database
from app.features import (
    decode_image,
    extract_features,
    issue_evidence,
    public_statistics,
)
from app.model_service import ISSUE_TYPES, ModelService


def _severity(confidence: float) -> str:
    if confidence >= 0.80:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"


class AnalysisService:
    def __init__(self, settings: Settings, database: Database, model: ModelService):
        self.settings = settings
        self.database = database
        self.model = model

    @staticmethod
    def _apply_explainable_gates(
        probabilities: dict[str, float], stats: dict[str, float]
    ) -> dict[str, float]:
        adjusted = dict(probabilities)
        if stats["brightness_mean"] < 0.18 or stats["dark_pixel_ratio"] > 0.40:
            adjusted["underexposure"] = max(adjusted["underexposure"], 0.92)
        elif stats["brightness_mean"] < 0.28:
            adjusted["underexposure"] = max(adjusted["underexposure"], 0.68)

        if stats["brightness_mean"] > 0.84 or stats["highlight_clip_ratio"] > 0.35:
            adjusted["overexposure"] = max(adjusted["overexposure"], 0.92)
        elif stats["brightness_mean"] > 0.74:
            adjusted["overexposure"] = max(adjusted["overexposure"], 0.68)

        if stats["noise_estimate"] > 0.075:
            adjusted["noise"] = max(adjusted["noise"], 0.88)
        if stats["blockiness"] > 0.055:
            adjusted["severe_degradation"] = max(adjusted["severe_degradation"], 0.82)
        return {key: round(min(1.0, value), 4) for key, value in adjusted.items()}

    @staticmethod
    def public_record(record: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in record.items()
            if key not in {"stored_filename", "sha256"}
        }

    def analyze(self, data: bytes, filename: str, content_type: str | None) -> dict[str, Any]:
        started = time.perf_counter()
        decode_started = time.perf_counter()
        decoded = decode_image(
            data,
            filename,
            content_type,
            max_upload_mb=self.settings.max_upload_mb,
            max_image_pixels=self.settings.max_image_pixels,
        )
        decode_ms = (time.perf_counter() - decode_started) * 1000

        feature_started = time.perf_counter()
        features = extract_features(decoded.image)
        statistics = public_statistics(decoded, features)
        feature_ms = (time.perf_counter() - feature_started) * 1000

        inference_started = time.perf_counter()
        prediction = self.model.predict(features, image=decoded.image)
        probabilities = self._apply_explainable_gates(prediction["issue_probabilities"], features)
        thresholds = prediction["thresholds"]
        issues = []
        for issue_type in ISSUE_TYPES:
            confidence = probabilities[issue_type]
            detected = confidence >= float(thresholds[issue_type])
            if detected:
                issues.append(
                    {
                        "type": issue_type,
                        "detected": True,
                        "severity": _severity(confidence),
                        "confidence": confidence,
                        "model_probability": prediction["issue_probabilities"][issue_type],
                        "evidence": issue_evidence(issue_type, features),
                    }
                )
        issues.sort(key=lambda item: item["confidence"], reverse=True)

        score = float(prediction["quality_score"])
        if issues:
            maximum_confidence = max(issue["confidence"] for issue in issues)
            score = min(score, 100.0 - 55.0 * maximum_confidence)
        score = round(max(0.0, min(100.0, score)), 1)

        defect_confidence = probabilities["potential_defect"]
        severe_confidence = probabilities["severe_degradation"]
        if defect_confidence >= thresholds["potential_defect"] or severe_confidence >= 0.75:
            label = "POTENTIALLY_DEFECTIVE"
        elif score < 70 or any(issue["severity"] in {"medium", "high"} for issue in issues):
            label = "DEGRADED"
        else:
            label = "ACCEPTABLE"
        inference_ms = (time.perf_counter() - inference_started) * 1000

        analysis_id = str(uuid.uuid4())
        stored_filename = f"{analysis_id}.jpg"
        stored_path = self.settings.upload_dir / stored_filename
        decoded.image.save(stored_path, format="JPEG", quality=90, optimize=True)

        total_ms = (time.perf_counter() - started) * 1000
        record = {
            "id": analysis_id,
            "original_filename": Path(filename).name[:255] or "upload",
            "stored_filename": stored_filename,
            "sha256": hashlib.sha256(data).hexdigest(),
            "mime_type": decoded.mime_type,
            "width": decoded.width,
            "height": decoded.height,
            "file_size_bytes": decoded.file_size_bytes,
            "quality_score": score,
            "quality_label": label,
            "issues": issues,
            "statistics": statistics,
            "model_name": prediction["model_name"],
            "model_version": prediction["model_version"],
            "timing_ms": {
                "decode": round(decode_ms, 2),
                "features": round(feature_ms, 2),
                "inference": round(inference_ms, 2),
                "total": round(total_ms, 2),
            },
        }
        self.database.insert(record)
        saved = self.database.get(analysis_id)
        if not saved:
            raise RuntimeError("Analysis was not available after database insertion.")
        return self.public_record(saved)

    def get(self, analysis_id: str, *, public: bool = True) -> dict[str, Any] | None:
        record = self.database.get(analysis_id)
        return self.public_record(record) if record and public else record

    def history(self, limit: int, offset: int) -> dict[str, Any]:
        records, total = self.database.list(limit, offset)
        return {
            "items": [self.public_record(record) for record in records],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

