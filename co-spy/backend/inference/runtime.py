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
    FUSION_V3_CHECKPOINT,
    LOCAL_OPENCLIP_WEIGHTS,
    OPENCLIP_CACHE_DIR,
    ORIGINAL_CHECKPOINTS,
    PROJECT_ROOT,
)
from detectors.sd_v1_4.fusion import ArtifactViewTransform

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
        payload.get("models", {}).get("model")
        if isinstance(payload.get("models"), dict)
        else None,
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
        self.device_original = self.device
        self.device_fusion_v3 = self.device
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
        # Cho phép hoạt động ngay khi ít nhất model Original đã nạp xong
        return self.original_loaded

    def predict_both(self, image: Image.Image) -> dict[str, Any]:
        if not self.original_loaded:
            detail = self.errors.get("original") or "Models are still loading"
            raise ModelUnavailableError(detail)

        with self._inference_lock:
            start_total = time.perf_counter()
            original_prediction = self._predict_original(image)
            
            # Nếu có Fusion v3 thì chạy, nếu chưa có thì tái sử dụng kết quả original kèm ghi chú
            if self.fusion_v3_loaded:
                fusion_prediction = self._predict_fusion_v3(image)
            else:
                fusion_prediction = Prediction(
                    "CO-SPY Fusion v3 (Pending/Not Loaded)",
                    original_prediction.ai_probability,
                    original_prediction.processing_time_ms,
                    original_prediction.raw_score,
                )
            
            total_elapsed = round((time.perf_counter() - start_total) * 1000)

        return {
            "original": original_prediction.as_dict(),
            "fusion_v3": fusion_prediction.as_dict(),
            "total_processing_time_ms": total_elapsed,
            "device": self.device.type,
        }

    def load(self) -> None:
        # Imports happen only during startup so errors are reported through health,
        # instead of preventing the web server from serving a useful status.
        sys.path.insert(0, str(PROJECT_ROOT))
        self.loading = True
        try:
            LOGGER.info("[MODEL] Devices -> Original: %s | Fusion v3: %s", self.device_original.type, self.device_fusion_v3.type)
            self._load_original()
            self._load_fusion_v3()
            LOGGER.info("[API] Model startup finished.")
        finally:
            self.loading = False

    def _require_files(self, checkpoints: dict[str, Path] | Path) -> None:
        if isinstance(checkpoints, Path):
            missing = [str(checkpoints)] if not checkpoints.is_file() else []
        elif isinstance(checkpoints, dict):
            missing = [str(path) for path in checkpoints.values() if not path.is_file()]
        else:
            missing = []
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
            self.original = model.to(self.device_original).eval()
            LOGGER.info("[MODEL] CO-SPY Original ready on CPU.")
        except (
            Exception
        ) as exc:  # health must expose a startup failure, never fake a result
            self.errors["original"] = str(exc)
            LOGGER.exception("[MODEL] CO-SPY Original failed to load")

    def _load_fusion_v3(self) -> None:
        try:
            self._require_files(FUSION_V3_CHECKPOINT)
            LOGGER.info("[MODEL] Loading Fusion v3...")
            from checkpoint_manager import load_torch_checkpoint
            from detectors.sd_v1_4.fusion import CoSpyFusionDetector

            ckpts = FUSION_V3_CHECKPOINT if isinstance(FUSION_V3_CHECKPOINT, dict) else {"fusion": FUSION_V3_CHECKPOINT}

            npr_state: dict[str, Any] | None = None
            for name, path in ckpts.items():
                LOGGER.info("[MODEL] Loading %s checkpoint...", name)
                payload = load_torch_checkpoint(path, map_location="cpu")
                state = _model_state(payload)
                if name == "npr":
                    npr_state = state

            # Nếu checkpoint tổng hợp chứa NPR hoặc có nhánh NPR
            temporary_path = None
            if npr_state is not None:
                file_descriptor, temporary_path = tempfile.mkstemp(suffix=".pth")
                os.close(file_descriptor)
                torch.save(npr_state, temporary_path)

            try:
                # Nếu model Original đã nạp semantic, cho Fusion v3 dùng chung CLIP để tiết kiệm 4GB RAM
                if self.original is not None and hasattr(self.original, "sem"):
                    model = CoSpyFusionDetector.__new__(CoSpyFusionDetector)
                    super(CoSpyFusionDetector, model).__init__()
                    model.fusion_version = "v3" if npr_state is not None else "v1"
                    model.num_classes = 1
                    model.artifact_views = 1
                    model.branch_dropout = 0.0
                    model.sem = self.original.sem  # Chia sẻ chung nhánh semantic đã nạp
                    from detectors.sd_v1_4.artifact import ArtifactDetector
                    model.art = ArtifactDetector()
                    model.art.load_weights(str(ORIGINAL_CHECKPOINTS["artifact"]))
                    
                    modules = [model.sem, model.art]
                    model.npr = None
                    if model.fusion_version == "v3" and temporary_path:
                        from detectors.sd_v1_4.fusion import NPRDetector
                        model.npr = NPRDetector()
                        model.npr.load_weights(temporary_path)
                        modules.append(model.npr)
                    
                    for m in modules:
                        for p in m.parameters():
                            p.requires_grad = False
                            
                    model.fc = torch.nn.Linear(2, 1)
                    feature_dims = [model.sem.fc.in_features, model.art.fc.in_features]
                    if model.npr is not None:
                        feature_dims.append(model.npr.fc.in_features)
                    from detectors.sd_v1_4.fusion import AdaptiveFusionHead
                    model.fusion_head = AdaptiveFusionHead(feature_dims, num_classes=1, branch_dropout=0.0) if model.fusion_version != "v1" else None
                    if model.fusion_head is not None:
                        model.fc.requires_grad_(False)

                    from torchvision import transforms
                    model.sem_transform = transforms.Compose(
                        [transforms.Normalize(model.sem.mean, model.sem.std)]
                    )
                    model.art_transform = transforms.Compose(
                        [
                            transforms.Resize(model.art.cropSize, antialias=False),
                            transforms.Normalize(model.art.mean, model.art.std),
                        ]
                    )
                    model.npr_transform = transforms.Compose(
                        [
                            transforms.Resize(224, antialias=False),
                            transforms.Normalize([0.0] * 3, [1.0] * 3),
                        ]
                    )
                    model.artifact_view_transform = lambda views: ArtifactViewTransform(
                        model.art.cropSize,
                        model.art.mean,
                        model.art.std,
                        num_views=views,
                    )
                    model._build_transforms()
                else:
                    model = CoSpyFusionDetector(
                        semantic_weights_path=str(ORIGINAL_CHECKPOINTS["semantic"]),
                        artifact_weights_path=str(ORIGINAL_CHECKPOINTS["artifact"]),
                        npr_weights_path=temporary_path,
                        fusion_version="v3" if npr_state is not None else "v1",
                        semantic_pretrained_weights_path=LOCAL_OPENCLIP_WEIGHTS,
                        semantic_pretrained_cache_dir=str(OPENCLIP_CACHE_DIR),
                    )
            finally:
                if temporary_path:
                    Path(temporary_path).unlink(missing_ok=True)

            fusion_path = ckpts.get("fusion", FUSION_V3_CHECKPOINT if isinstance(FUSION_V3_CHECKPOINT, Path) else None)
            if fusion_path and fusion_path.is_file():
                fusion_payload = load_torch_checkpoint(fusion_path, map_location="cpu")
                model.load_state_dict(_model_state(fusion_payload), strict=False)
            self.fusion_v3 = model.to(self.device_fusion_v3).eval()
            LOGGER.info("[MODEL] Fusion v3 ready on GPU (CUDA).")
        except Exception as exc:
            self.errors["fusion_v3"] = str(exc)
            LOGGER.warning("[MODEL] Fusion v3 not available: %s", exc)

    @staticmethod
    def _prepare_v1(model: Any, image: Image.Image) -> torch.Tensor:
        image_tensor = model.test_transform(image).unsqueeze(0)
        return image_tensor

    @staticmethod
    def _prepare_v3(
        model: Any, image: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        semantic = model.sem.test_transform(image).unsqueeze(0)
        artifact = model.artifact_view_transform(model.artifact_views)(image).unsqueeze(
            0
        )
        return semantic, artifact

    def _predict_original(self, image: Image.Image) -> Prediction:
        assert self.original is not None
        started = time.perf_counter()
        with torch.inference_mode():
            # Chạy CO-SPY Original trên CPU
            tensor = self._prepare_v1(self.original, image).to(self.device_original)
            logits = self.original(tensor)
            probability = logits.sigmoid().flatten()[0].item()
            raw_score = logits.flatten()[0].item()
        return Prediction(
            "CO-SPY Original",
            probability,
            round((time.perf_counter() - started) * 1000),
            raw_score,
        )

    def _predict_fusion_v3(self, image: Image.Image) -> Prediction:
        assert self.fusion_v3 is not None
        started = time.perf_counter()
        with torch.inference_mode():
            if self.fusion_v3.fusion_version == "v1":
                # Ở chế độ v1, forward nhận tensor 4D từ test_transform
                tensor = self._prepare_v1(self.fusion_v3, image).to(self.device_fusion_v3)
                logits = self.fusion_v3(tensor)
            else:
                # Ở chế độ v3 (adaptive), forward nhận tensor đã qua transform của sem và art
                semantic = self.fusion_v3.sem.test_transform(image).unsqueeze(0).to(self.device_fusion_v3)
                # artifact views: shape (batch_size, views, channels, h, w)
                artifact = self.fusion_v3.artifact_view_transform(self.fusion_v3.artifact_views)(image).unsqueeze(0).to(self.device_fusion_v3)
                # Tự trích xuất đặc trưng và đưa qua forward_features
                sem_feat, art_feat, npr_feat = self.fusion_v3.extract_features(semantic, artifact)
                logits = self.fusion_v3.forward_features(sem_feat, art_feat, npr_feat)
            probability = logits.sigmoid().flatten()[0].item()
            raw_score = logits.flatten()[0].item()
        return Prediction(
            "CO-SPY Fusion v3",
            probability,
            round((time.perf_counter() - started) * 1000),
            raw_score,
        )


