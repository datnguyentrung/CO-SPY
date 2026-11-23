"""Thin runtime adapter around the repository's CO-SPY detector implementations.

The public interface intentionally stays small: load once, inspect readiness, and
run both detectors over a PIL image.  All model-specific preprocessing remains in
the detector classes already used by ``evaluate.py``.
"""

from __future__ import annotations

import logging
import sys
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
LOGGER = logging.getLogger(__name__)
FORENSIC_VIEW_NAMES = ("center", "high_frequency")
NPR_GUARD_AI_THRESHOLD = 0.95


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


def _prediction_from_logits(model_name: str, logits: torch.Tensor) -> Prediction:
    """Convert one binary branch logit into the public prediction shape."""
    raw_score = logits.flatten()[0].item()
    return Prediction(model_name, logits.sigmoid().flatten()[0].item(), 0, raw_score)


def _apply_npr_guard(
    fusion_prediction: Prediction, npr_by_view: dict[str, Prediction]
) -> tuple[Prediction, dict[str, Any]]:
    """Escalate only when every forensic view independently flags AI."""
    npr_ai_probabilities = {
        view: prediction.ai_probability for view, prediction in npr_by_view.items()
    }
    minimum_npr_ai_probability = min(npr_ai_probabilities.values())
    applied = minimum_npr_ai_probability >= NPR_GUARD_AI_THRESHOLD
    metadata = {
        "views": list(FORENSIC_VIEW_NAMES),
        "npr_ai_probabilities": npr_ai_probabilities,
        "npr_guard_threshold": NPR_GUARD_AI_THRESHOLD,
        "npr_guard_applied": applied,
        "model_prediction": fusion_prediction.as_dict(),
    }
    if not applied:
        return fusion_prediction, metadata
    return (
        Prediction(
            fusion_prediction.model_name,
            minimum_npr_ai_probability,
            fusion_prediction.processing_time_ms,
            fusion_prediction.raw_score,
        ),
        metadata,
    )


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


def _require_fusion_v3_state(state: dict[str, Any]) -> None:
    """Reject checkpoints that cannot drive the adaptive v3 inference path."""
    required_keys = {
        "sem.fc.weight",
        "sem.fc.bias",
        "art.fc.weight",
        "art.fc.bias",
        "npr.fc.weight",
        "npr.fc.bias",
        "fusion_head.classifier.weight",
        "fusion_head.classifier.bias",
    }
    missing = sorted(required_keys.difference(state))
    if missing:
        raise ValueError(
            "Fusion v3 checkpoint is missing required weights: " + ", ".join(missing)
        )

    expected_features = (
        int(state["sem.fc.weight"].shape[1])
        + int(state["sem.fc.weight"].shape[0])
        + int(state["art.fc.weight"].shape[1])
        + int(state["npr.fc.weight"].shape[1])
    )
    classifier_features = int(state["fusion_head.classifier.weight"].shape[1])
    if classifier_features != expected_features:
        raise ValueError(
            "Fusion v3 checkpoint has incompatible feature dimensions: "
            f"classifier expects {classifier_features}, branches provide "
            f"{expected_features}"
        )


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
        return self.original_loaded and self.fusion_v3_loaded

    def predict_both(self, image: Image.Image) -> dict[str, Any]:
        if not self.ready:
            detail = (
                self.errors.get("fusion_v3")
                or self.errors.get("original")
                or "Models are still loading"
            )
            raise ModelUnavailableError(detail)

        with self._inference_lock:
            start_total = time.perf_counter()
            original_prediction = self._predict_original(image)
            fusion_prediction, fusion_branches, forensic = self._predict_fusion_v3(image)
            total_elapsed = round((time.perf_counter() - start_total) * 1000)

        return {
            "original": original_prediction.as_dict(),
            "fusion_v3": {
                **fusion_prediction.as_dict(),
                "branches": {
                    name: prediction.as_dict()
                    for name, prediction in fusion_branches.items()
                },
                "forensic": forensic,
            },
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
            fusion_path = (
                FUSION_V3_CHECKPOINT.get("fusion")
                if isinstance(FUSION_V3_CHECKPOINT, dict)
                else FUSION_V3_CHECKPOINT
            )
            if fusion_path is None:
                raise ValueError("Fusion v3 checkpoint configuration has no fusion file")
            LOGGER.info("[MODEL] Loading checkpoint %s...", fusion_path)
            payload = load_torch_checkpoint(fusion_path, map_location="cpu")
            state = _model_state(payload)
            _require_fusion_v3_state(state)
            model = self._build_fusion_v3_model(state)
            self.fusion_v3 = model.to(self.device_fusion_v3).eval()
            LOGGER.info(
                "[MODEL] Fusion v3 ready on %s with adaptive NPR fusion.",
                self.device_fusion_v3.type.upper(),
            )
        except Exception as exc:
            self.errors["fusion_v3"] = str(exc)
            LOGGER.exception("[MODEL] Fusion v3 failed to load")

    def _build_fusion_v3_model(self, state: dict[str, Any]) -> Any:
        """Build v3 while sharing only the frozen CLIP backbone with Original."""
        if self.original is None or not hasattr(self.original, "sem"):
            raise ModelUnavailableError("CO-SPY Original must load before Fusion v3")

        from detectors.sd_v1_4.artifact import ArtifactDetector
        from detectors.sd_v1_4.fusion import AdaptiveFusionHead, CoSpyFusionDetector
        from detectors.sd_v1_4.npr import NPRDetector
        from detectors.sd_v1_4.semantic import SemanticDetector
        from torchvision import transforms

        sem = SemanticDetector.__new__(SemanticDetector)
        torch.nn.Module.__init__(sem)
        sem.clip = self.original.sem.clip
        sem.fc = torch.nn.Linear(
            int(state["sem.fc.weight"].shape[1]),
            int(state["sem.fc.weight"].shape[0]),
        )
        sem.downscale_prob = 0.0
        sem._build_transforms()

        model = CoSpyFusionDetector.__new__(CoSpyFusionDetector)
        torch.nn.Module.__init__(model)
        model.fusion_version = "v3"
        model.num_classes = 1
        # Evaluate both the center crop and the crop with the strongest local
        # high-frequency energy. This leaves the semantic image untouched.
        model.artifact_views = len(FORENSIC_VIEW_NAMES)
        model.branch_dropout = 0.0
        model.sem = sem
        model.art = ArtifactDetector()
        model.npr = NPRDetector()
        model.fc = torch.nn.Linear(2, 1)

        feature_dims = [
            model.sem.fc.in_features + model.sem.fc.out_features,
            model.art.fc.in_features,
            model.npr.fc.in_features,
        ]
        model.fusion_head = AdaptiveFusionHead(
            feature_dims, num_classes=1, branch_dropout=0.0
        )
        model.fc.requires_grad_(False)
        for module in (model.sem, model.art, model.npr):
            module.requires_grad_(False)

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
        model._build_transforms()

        # The CLIP backbone is frozen during fusion training and shared with the
        # Original detector. Load every v3-specific tensor without mutating that
        # shared backbone, then explicitly verify that nothing else was skipped.
        loadable_state = {
            key: value for key, value in state.items() if not key.startswith("sem.clip.")
        }
        incompatible = model.load_state_dict(loadable_state, strict=False)
        expected_missing = {
            key for key in model.state_dict() if key.startswith("sem.clip.")
        }
        actual_missing = set(incompatible.missing_keys)
        if actual_missing != expected_missing or incompatible.unexpected_keys:
            unexpected = sorted(incompatible.unexpected_keys)
            unaccounted_missing = sorted(actual_missing.difference(expected_missing))
            raise ValueError(
                "Fusion v3 checkpoint/model mismatch; "
                f"missing={unaccounted_missing}, unexpected={unexpected}"
            )
        return model

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

    def _predict_fusion_v3(
        self, image: Image.Image
    ) -> tuple[Prediction, dict[str, Prediction], dict[str, Any]]:
        assert self.fusion_v3 is not None
        if self.fusion_v3.fusion_version != "v3":
            raise RuntimeError("Fusion v3 runtime loaded a non-v3 model")
        started = time.perf_counter()
        with torch.inference_mode():
            semantic, artifact = self._prepare_v3(self.fusion_v3, image)
            semantic = semantic.to(self.device_fusion_v3)
            artifact = artifact.to(self.device_fusion_v3)
            sem_feat, art_feat, _ = self.fusion_v3.extract_features(
                semantic, artifact, include_npr=False
            )
            semantic_logits = sem_feat[:, -self.fusion_v3.sem.fc.out_features :]
            batch_size, num_views = art_feat.shape[:2]
            if num_views != len(FORENSIC_VIEW_NAMES):
                raise RuntimeError("Fusion v3 did not produce the expected forensic views")

            flat_artifact = artifact.reshape(
                batch_size * num_views, *artifact.shape[2:]
            )
            npr_input = self.fusion_v3.npr_transform(flat_artifact)
            npr_feat = self.fusion_v3.npr.artifact_encoder(npr_input).reshape(
                batch_size, num_views, -1
            )

            fusion_by_view: dict[str, Prediction] = {}
            artifact_by_view: dict[str, Prediction] = {}
            npr_by_view: dict[str, Prediction] = {}
            for view_index, view_name in enumerate(FORENSIC_VIEW_NAMES):
                view_artifact = art_feat[:, view_index : view_index + 1]
                view_npr = npr_feat[:, view_index]
                view_logits = self.fusion_v3.forward_features(
                    sem_feat, view_artifact, view_npr
                )
                fusion_by_view[view_name] = _prediction_from_logits(
                    f"CO-SPY Fusion v3 ({view_name})", view_logits
                )
                artifact_by_view[view_name] = _prediction_from_logits(
                    "Artifact-VAE", self.fusion_v3.art.fc(view_artifact)
                )
                npr_by_view[view_name] = _prediction_from_logits(
                    "NPR", self.fusion_v3.npr.fc(view_npr)
                )

            selected_view = max(
                fusion_by_view, key=lambda view: fusion_by_view[view].ai_probability
            )
            selected_fusion = fusion_by_view[selected_view]
            fusion_prediction, forensic = _apply_npr_guard(
                Prediction(
                    "CO-SPY Fusion v3",
                    selected_fusion.ai_probability,
                    0,
                    selected_fusion.raw_score,
                ),
                npr_by_view,
            )
            forensic["selected_fusion_view"] = selected_view
        elapsed = round((time.perf_counter() - started) * 1000)
        fusion_prediction = Prediction(
            fusion_prediction.model_name,
            fusion_prediction.ai_probability,
            elapsed,
            fusion_prediction.raw_score,
        )
        forensic["model_prediction"]["processing_time_ms"] = elapsed
        return (
            fusion_prediction,
            {
                "semantic": _prediction_from_logits("Semantic", semantic_logits),
                "artifact": max(
                    artifact_by_view.values(), key=lambda prediction: prediction.ai_probability
                ),
                "npr": max(
                    npr_by_view.values(), key=lambda prediction: prediction.ai_probability
                ),
            },
            forensic,
        )


