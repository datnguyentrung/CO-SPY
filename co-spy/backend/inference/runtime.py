"""Thin runtime adapter around the repository's CO-SPY detector implementations.

The public interface intentionally stays small: load once, inspect readiness, and
run both detectors over a PIL image.  All model-specific preprocessing remains in
the detector classes already used by ``evaluate.py``.
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from backend.config import (
    FUSION_V3_CHECKPOINTS,
    OPENCLIP_CACHE_DIR,
    LOCAL_OPENCLIP_WEIGHTS,
    ORIGINAL_CHECKPOINTS,
    PROJECT_ROOT,
)

LOGGER = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when a requested prediction cannot use a fully loaded model."""


@dataclass(frozen=True)
class Prediction:
    model_name: str
    ai_probability: float
    processing_time_ms: int
    raw_score: float

    def as_dict(self) -> dict[str, Any]:
        ai_probability = float(self.ai_probability)
        prediction = "AI_GENERATED" if ai_probability > 0.5 else "REAL"
        return {
            "model_name": self.model_name,
            "prediction": prediction,
            "confidence": max(ai_probability, 1 - ai_probability),
            "real_probability": 1 - ai_probability,
            "ai_probability": ai_probability,
            "processing_time_ms": self.processing_time_ms,
            "raw_score": self.raw_score,
        }


def _model_state(payload: Any) -> dict[str, Any]:
    """Extract a model state dict from plain weights or CheckpointManager bundles."""
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload is not a mapping")
    for candidate in (
        payload.get("models", {}).get("model") if isinstance(payload.get("models"), dict) else None,
        payload.get("model"),
        payload.get("state_dict"),
    ):
        if isinstance(candidate, dict):
            return candidate
    # Released branch weights are state dictionaries themselves.
    if payload and all(isinstance(key, str) for key in payload):
        return payload
    raise ValueError("Checkpoint has no recognizable model state")


class CoSpyRuntime:
    """Owns both model graphs for the FastAPI process lifetime."""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.original: Any | None = None
        self.fusion_v3: Any | None = None
        self.errors: dict[str, str] = {}
        self.loading = False
        # CUDA inference is serialized to avoid interleaving two large VAE/CLIP graphs.
        self._inference_lock = threading.Lock()

    @property
    def original_loaded(self) -> bool:
        return self.original is not None

    @property
    def fusion_v3_loaded(self) -> bool:
        return self.fusion_v3 is not None

    @property
    def ready(self) -> bool:
        return self.original_loaded and self.fusion_v3_loaded

    def load(self) -> None:
        # Imports happen only during startup so errors are reported through health,
        # instead of preventing the web server from serving a useful status.
        sys.path.insert(0, str(PROJECT_ROOT))
        self.loading = True
        try:
            LOGGER.info("[MODEL] Device: %s", self.device.type)
            self._load_original()
            self._load_fusion_v3()
            LOGGER.info("[API] Model startup finished.")
        finally:
            self.loading = False

    def _require_files(self, checkpoints: dict[str, Path]) -> None:
        missing = [str(path) for path in checkpoints.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing checkpoint(s): " + ", ".join(missing))

    def _load_original(self) -> None:
        try:
            self._require_files(ORIGINAL_CHECKPOINTS)
            from detectors.sd_v1_4.fusion import CoSpyFusionDetector

            LOGGER.info("[MODEL] Loading CO-SPY Original...")
            model = CoSpyFusionDetector(
                semantic_weights_path=str(ORIGINAL_CHECKPOINTS["semantic"]),
                artifact_weights_path=str(ORIGINAL_CHECKPOINTS["artifact"]),
                fusion_version="v1",
                semantic_pretrained_weights_path=LOCAL_OPENCLIP_WEIGHTS,
                semantic_pretrained_cache_dir=str(OPENCLIP_CACHE_DIR),
            )
            model.load_weights(str(ORIGINAL_CHECKPOINTS["fusion"]))
            self.original = model.to(self.device).eval()
            LOGGER.info("[MODEL] CO-SPY Original ready.")
        except Exception as exc:  # health must expose a startup failure, never fake a result
            self.errors["original"] = str(exc)
            LOGGER.exception("[MODEL] CO-SPY Original failed to load")

    def _load_fusion_v3(self) -> None:
        try:
            self._require_files(FUSION_V3_CHECKPOINTS)
            LOGGER.info("[MODEL] Loading Fusion v3...")
            # CheckpointManager writes full bundles; validating every file here makes
            # truncation/corruption fail at startup rather than halfway through a request.
            from checkpoint_manager import load_torch_checkpoint
            from detectors.sd_v1_4.fusion import CoSpyFusionDetector

            npr_state: dict[str, Any] | None = None
            for name, path in FUSION_V3_CHECKPOINTS.items():
                LOGGER.info("[MODEL] Loading %s checkpoint...", name)
                payload = load_torch_checkpoint(path, map_location="cpu")
                state = _model_state(payload)
                if name == "npr":
                    npr_state = state

            if npr_state is None:
                raise ValueError("NPR checkpoint contains no model state")

            # CoSpyFusionDetector accepts branch *weight files*, while training
            # checkpoints are full CheckpointManager bundles. Materialize only the
            # extracted NPR state to a disposable file for the existing constructor.
            file_descriptor, temporary_path = tempfile.mkstemp(suffix=".pth")
            os.close(file_descriptor)
            try:
                torch.save(npr_state, temporary_path)
                model = CoSpyFusionDetector(
                    semantic_weights_path=str(ORIGINAL_CHECKPOINTS["semantic"]),
                    artifact_weights_path=str(ORIGINAL_CHECKPOINTS["artifact"]),
                    npr_weights_path=temporary_path,
                    fusion_version="v3",
                    semantic_pretrained_weights_path=LOCAL_OPENCLIP_WEIGHTS,
                    semantic_pretrained_cache_dir=str(OPENCLIP_CACHE_DIR),
                )
            finally:
                Path(temporary_path).unlink(missing_ok=True)

            # The fusion bundle contains the complete v3 graph. The original branch
            # files merely satisfy construction before its state is replaced below.
            fusion_payload = load_torch_checkpoint(FUSION_V3_CHECKPOINTS["fusion"], map_location="cpu")
            model.load_state_dict(_model_state(fusion_payload), strict=True)
            self.fusion_v3 = model.to(self.device).eval()
            LOGGER.info("[MODEL] Fusion v3 ready.")
        except Exception as exc:
            self.errors["fusion_v3"] = str(exc)
            LOGGER.exception("[MODEL] Fusion v3 failed to load")

    @staticmethod
    def _prepare_v1(model: Any, image: Image.Image) -> torch.Tensor:
        image_tensor = model.test_transform(image).unsqueeze(0)
        return image_tensor

    @staticmethod
    def _prepare_v3(model: Any, image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
        semantic = model.sem.test_transform(image).unsqueeze(0)
        artifact = model.artifact_view_transform(model.artifact_views)(image).unsqueeze(0)
        return semantic, artifact

    def _predict_original(self, image: Image.Image) -> Prediction:
        assert self.original is not None
        started = time.perf_counter()
        with torch.inference_mode():
            # This is the released v1 forward path: 384px test tensor then the
            # detector's semantic/artifact transforms before fusion.
            tensor = self._prepare_v1(self.original, image).to(self.device)
            logits = self.original(tensor)
            probability = logits.sigmoid().flatten()[0].item()
            raw_score = logits.flatten()[0].item()
        return Prediction("CO-SPY Original", probability, round((time.perf_counter() - started) * 1000), raw_score)

    def _predict_fusion_v3(self, image: Image.Image) -> Prediction:
        assert self.fusion_v3 is not None
        started = time.perf_counter()
        with torch.inference_mode():
            semantic, artifact = self._prepare_v3(self.fusion_v3, image)
            logits = self.fusion_v3((semantic.to(self.device), artifact.to(self.device)))
            probability = logits.sigmoid().flatten()[0].item()
            raw_score = logits.flatten()[0].item()
        return Prediction("CO-SPY Fusion v3", probability, round((time.perf_counter() - started) * 1000), raw_score)

    def predict_both(self, image: Image.Image) -> dict[str, Any]:
        if not self.ready:
            detail = self.errors.get("fusion_v3") or self.errors.get("original") or "Models are still loading"
            raise ModelUnavailableError(detail)
        total_started = time.perf_counter()
        # Sequential execution keeps both graphs resident without doubling peak VRAM.
        with self._inference_lock:
            original = self._predict_original(image)
            fusion_v3 = self._predict_fusion_v3(image)
        return {
            "original": original.as_dict(),
            "fusion_v3": fusion_v3.as_dict(),
            "total_processing_time_ms": round((time.perf_counter() - total_started) * 1000),
            "device": self.device.type,
        }
