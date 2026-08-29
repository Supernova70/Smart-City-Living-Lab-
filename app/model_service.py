from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.features import FEATURE_NAMES, feature_vector


ISSUE_TYPES = [
    "blur",
    "underexposure",
    "overexposure",
    "noise",
    "severe_degradation",
    "potential_defect",
]


class ModelNotReadyError(RuntimeError):
    pass


class ModelService:
    """Loads and runs the image quality model.

    Supports two artifact formats transparently, detected by file extension:
      - ``.joblib``: legacy Random Forest baseline (feature-vector input).
      - ``.pt``:     PyTorch multi-task deep model (PIL Image input).
    """

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.artifact: dict[str, Any] | None = None
        self._deep_model = None       # torch.nn.Module when .pt is loaded
        self._deep_transform = None   # torchvision transform pipeline
        self._device = None           # torch.device
        self._is_deep: bool = False

    @property
    def ready(self) -> bool:
        return self.artifact is not None

    @property
    def model_name(self) -> str:
        return str(self.artifact.get("model_name", "unknown")) if self.artifact else "unavailable"

    @property
    def model_version(self) -> str:
        return str(self.artifact.get("model_version", "unknown")) if self.artifact else "unavailable"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        if not self.model_path.exists():
            raise ModelNotReadyError(
                f"Model artifact not found at {self.model_path}. "
                "Run `python -m scripts.train_model` or point MODEL_PATH at an existing artifact."
            )
        if self.model_path.suffix.lower() == ".pt":
            self._load_deep()
        else:
            self._load_legacy()

    def _load_legacy(self) -> None:
        import joblib

        artifact = joblib.load(self.model_path)
        required = {"issue_model", "score_model", "feature_names", "issue_types", "thresholds"}
        missing = required - set(artifact)
        if missing:
            raise ModelNotReadyError(f"Model artifact is missing keys: {sorted(missing)}")
        if list(artifact["feature_names"]) != FEATURE_NAMES:
            raise ModelNotReadyError("Model feature order does not match the application feature pipeline.")
        if list(artifact["issue_types"]) != ISSUE_TYPES:
            raise ModelNotReadyError("Model issue order does not match the application issue schema.")
        self.artifact = artifact
        self._is_deep = False

    def _load_deep(self) -> None:
        import torch
        from torchvision import transforms

        from app.deep_model import build_model

        artifact = torch.load(self.model_path, map_location="cpu", weights_only=False)
        if artifact.get("issue_types") != ISSUE_TYPES:
            raise ModelNotReadyError(
                f"Checkpoint issue_types {artifact.get('issue_types')} != application {ISSUE_TYPES}"
            )
        required = {"architecture", "state_dict", "thresholds", "image_size", "normalization"}
        missing = required - set(artifact)
        if missing:
            raise ModelNotReadyError(f"Deep checkpoint missing keys: {sorted(missing)}")

        # Use CUDA when available; fall back to CPU (e.g. production Docker)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(artifact["architecture"], len(ISSUE_TYPES), pretrained=False)
        model.load_state_dict(artifact["state_dict"])
        model.to(device)
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        model.eval()

        norm = artifact["normalization"]
        transform = transforms.Compose([
            transforms.Resize((artifact["image_size"], artifact["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(tuple(norm["mean"]), tuple(norm["std"])),
        ])

        self.artifact = artifact
        self._deep_model = model
        self._deep_transform = transform
        self._device = device
        self._is_deep = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def _positive_probability(model: Any, probabilities: np.ndarray, index: int) -> float:
        """Extract positive-class probability from a legacy sklearn multi-output model."""
        classes = list(model.classes_[index])
        if 1 not in classes:
            return 1.0 if classes and classes[0] == 1 else 0.0
        return float(probabilities[index][0, classes.index(1)])

    def predict(self, features: dict[str, float] | None = None, *, image=None) -> dict[str, Any]:
        """Run inference and return the standardised prediction dict.

        Args:
            features: Engineered feature dict — required for the legacy .joblib
                      model; passed but not used by the deep model.
            image:    PIL Image — required keyword argument for the .pt deep
                      model; ignored by the legacy model.

        Returns:
            Dict with keys: quality_score, issue_probabilities, thresholds,
            model_name, model_version.
        """
        if not self.artifact:
            raise ModelNotReadyError("The model has not been loaded.")
        if self._is_deep:
            return self._predict_deep(image)
        return self._predict_legacy(features)

    def _predict_legacy(self, features: dict[str, float] | None) -> dict[str, Any]:
        if features is None:
            raise ModelNotReadyError("Legacy model requires a feature dict.")
        vector = feature_vector(features).reshape(1, -1)
        issue_model = self.artifact["issue_model"]
        raw_probabilities = issue_model.predict_proba(vector)
        issue_probabilities = {
            issue: round(self._positive_probability(issue_model, raw_probabilities, index), 4)
            for index, issue in enumerate(ISSUE_TYPES)
        }
        score = float(np.clip(self.artifact["score_model"].predict(vector)[0], 0.0, 100.0))
        return {
            "quality_score": round(score, 1),
            "issue_probabilities": issue_probabilities,
            "thresholds": dict(self.artifact["thresholds"]),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }

    def _predict_deep(self, image) -> dict[str, Any]:
        import torch

        if image is None:
            raise ModelNotReadyError("Deep model requires a PIL Image; pass image=<pil_image>.")
        tensor = self._deep_transform(image.convert("RGB")).unsqueeze(0).to(self._device)
        if self._device.type == "cuda":
            tensor = tensor.to(memory_format=torch.channels_last)

        with torch.inference_mode():
            with torch.autocast(
                device_type=self._device.type,
                dtype=torch.float16,
                enabled=self._device.type == "cuda",
            ):
                logits, quality_t = self._deep_model(tensor)
            probabilities = torch.sigmoid(logits)[0].cpu().float().numpy()
            quality_score = float(quality_t[0].cpu().float()) * 100.0

        issue_probabilities = {
            issue: round(float(probabilities[idx]), 4)
            for idx, issue in enumerate(ISSUE_TYPES)
        }
        return {
            "quality_score": round(float(np.clip(quality_score, 0.0, 100.0)), 1),
            "issue_probabilities": issue_probabilities,
            "thresholds": dict(self.artifact["thresholds"]),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }
