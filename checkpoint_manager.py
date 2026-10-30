"""Crash-safe training checkpoints, auto-resume, and storage reporting.

The manager keeps the training engine independent from the storage location.
Kaggle, Colab, and local runs use the same checkpoints; only the discovery
roots and output directory differ.
"""

import hashlib
import json
import os
import random
import re
import signal
import threading
import time
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = 1
LATEST_NAME = "checkpoint_latest.pth"
BEST_NAME = "checkpoint_best.pth"
RECOVERY_NAME = "checkpoint_recovery.pth"
LEGACY_BEST_NAME = "best_model.pth"
TRAINING_STATE_NAME = "training_state.json"

RESUMABLE_NAMES = (LATEST_NAME, RECOVERY_NAME)
CHECKPOINT_NAMES = RESUMABLE_NAMES + (BEST_NAME,)
DISCOVERY_NAMES = CHECKPOINT_NAMES + (LEGACY_BEST_NAME,)
WEIGHTS_ONLY_NAMES = (BEST_NAME, LEGACY_BEST_NAME)
RECOVERY_BACKUP_PATTERN = re.compile(r"^checkpoint_recovery\.(\d+)\.pth$")

STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_RUNNING = "RUNNING"
STATUS_CHECKPOINTED = "CHECKPOINTED"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_RESUMABLE = "RESUMABLE"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

VALID_STATUSES = {
    STATUS_NOT_STARTED,
    STATUS_RUNNING,
    STATUS_CHECKPOINTED,
    STATUS_INTERRUPTED,
    STATUS_RESUMABLE,
    STATUS_COMPLETED,
    STATUS_FAILED,
}


class CheckpointError(RuntimeError):
    """Base class for controlled checkpoint failures."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when an explicitly requested checkpoint cannot be resumed."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _deep_equal(left: Any, right: Any) -> bool:
    """Compare JSON-like metadata while allowing tuples and lists to match."""
    return _canonical_json(left) == _canonical_json(right)


def _fsync_file(handle) -> None:
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except OSError:
        # Some virtual filesystems (including mounted Drive variants) do not
        # expose fsync. The atomic replace still protects against partial files.
        pass


def atomic_torch_save(
    payload: Any,
    path: Union[str, os.PathLike],
    *,
    before_replace=None,
) -> Path:
    """Write a torch file in the destination directory and atomically replace."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            _fsync_file(handle)
        if before_replace is not None:
            before_replace()
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_json_save(payload: Any, path: Union[str, os.PathLike]) -> Path:
    """Atomically write UTF-8 JSON metadata."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            _fsync_file(handle)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def load_torch_checkpoint(
    path: Union[str, os.PathLike],
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> Any:
    """Load a trusted local checkpoint with a common cross-version interface."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0 does not expose weights_only.
        return torch.load(path, map_location=map_location)


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Mapping[str, Any]]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state and torch.cuda.is_available():
        available = torch.cuda.device_count()
        for device_index, device_state in enumerate(cuda_state[:available]):
            torch.cuda.set_rng_state(device_state, device=device_index)


def _as_mapping(value: Any, label: str) -> MutableMapping[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if value is None:
        return {}
    return {label: value}


def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in module.state_dict().items()
    }


def model_signature(module: torch.nn.Module) -> str:
    shapes = [
        (key, tuple(value.shape), str(value.dtype))
        for key, value in module.state_dict().items()
        if torch.is_tensor(value)
    ]
    return hashlib.sha256(_canonical_json(shapes).encode("utf-8")).hexdigest()


def _dataset_paths(dataset: Any) -> Tuple[Sequence[str], Mapping[str, int]]:
    candidates = (dataset, getattr(dataset, "base", None))
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "total_list"):
            labels = getattr(candidate, "labels_dict", {})
            return list(candidate.total_list), labels
    return [], {}


def dataset_fingerprint(dataset: Any) -> str:
    """Return a path-order-independent dataset fingerprint.

    Basenames plus labels avoid absolute-path differences between Kaggle
    notebook versions while still detecting changed or reordered image sets.
    """
    paths, labels = _dataset_paths(dataset)
    if not paths:
        return "unknown"
    records = sorted(
        f"{'/'.join(Path(path).parts[-3:])}\t{int(labels.get(path, -1))}"
        for path in paths
    )
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.encode("utf-8", errors="surrogateescape"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class StorageLayout:
    environment: str
    working_root: Path
    output_root: Path
    discovery_roots: Tuple[Path, ...]


def _unique_paths(
    paths: Iterable[Union[str, os.PathLike]],
) -> Tuple[Path, ...]:
    seen: Set[str] = set()
    result: List[Path] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return tuple(result)


def detect_storage_layout(
    output_root: Union[str, os.PathLike],
    *,
    input_roots: Iterable[Union[str, os.PathLike]] = (),
    environment: Optional[str] = None,
) -> StorageLayout:
    """Detect Kaggle/Colab/local storage without hardcoding it in Trainer."""
    cwd = Path.cwd().resolve()
    detected = (environment or "").strip().lower()
    if not detected:
        if Path("/kaggle/working").is_dir() or os.environ.get(
            "KAGGLE_KERNEL_RUN_TYPE"
        ):
            detected = "kaggle"
        elif Path("/content").is_dir() or os.environ.get("COLAB_RELEASE_TAG"):
            detected = "colab"
        else:
            detected = "local"

    roots: List[Union[str, os.PathLike]] = [output_root, *input_roots]
    env_roots = os.environ.get("CO_SPY_CHECKPOINT_ROOTS", "")
    roots.extend(item for item in env_roots.split(os.pathsep) if item)

    if detected == "kaggle":
        roots.append("/kaggle/input")
    elif detected == "colab":
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.is_dir():
            project_root = drive_root / "CO-SPY"
            roots.append(project_root if project_root.is_dir() else drive_root)

    return StorageLayout(
        environment=detected,
        working_root=cwd,
        output_root=Path(output_root).expanduser().resolve(),
        discovery_roots=_unique_paths(roots),
    )


def _metadata_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.json")


def _is_discovery_name(name: str) -> bool:
    return name in DISCOVERY_NAMES or bool(RECOVERY_BACKUP_PATTERN.match(name))


@dataclass
class ResumeResult:
    checkpoint_path: Optional[Path] = None
    payload: Optional[Dict[str, Any]] = None
    resumed: bool = False
    weights_only: bool = False
    start_epoch: int = 0
    batch_position: int = 0
    messages: List[str] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)

    @property
    def training_state(self) -> Mapping[str, Any]:
        if not self.payload:
            return {}
        return self.payload.get("training_state", {})


class CheckpointManager:
    """Persist and restore complete, validated training state."""

    def __init__(
        self,
        checkpoint_dir: Union[str, os.PathLike],
        *,
        checkpoint_root: Optional[Union[str, os.PathLike]] = None,
        project_name: str = "CO-SPY",
        experiment_id: Optional[str] = None,
        architecture_id: str,
        model_config: Mapping[str, Any],
        dataset_id: str,
        training_config: Mapping[str, Any],
        optimizer_config: Mapping[str, Any],
        scheduler_config: Mapping[str, Any],
        total_epochs: int,
        resume_mode: str = "auto",
        resume_path: Optional[Union[str, os.PathLike]] = None,
        input_roots: Iterable[Union[str, os.PathLike]] = (),
        environment: Optional[str] = None,
        checkpoint_interval_seconds: float = 900.0,
        session_timeout_seconds: float = 12 * 60 * 60,
        pre_timeout_seconds: float = 15 * 60,
        recovery_retention: int = 1,
    ):
        if resume_mode not in {"auto", "full", "weights", "new"}:
            raise ValueError("resume_mode must be auto, full, weights, or new")
        self.checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_root = Path(checkpoint_root or checkpoint_dir).expanduser().resolve()
        self.project_name = project_name
        self.experiment_id = experiment_id or "default"
        self.architecture_id = architecture_id
        self.model_config = dict(model_config)
        self.dataset_id = dataset_id
        self.training_config = dict(training_config)
        self.optimizer_config = dict(optimizer_config)
        self.scheduler_config = dict(scheduler_config)
        self.total_epochs = int(total_epochs)
        self.resume_mode = resume_mode
        self.resume_path = (
            Path(resume_path).expanduser().resolve() if resume_path else None
        )
        storage_input_roots = list(input_roots)
        if self.checkpoint_root != self.checkpoint_dir:
            storage_input_roots.append(self.checkpoint_root)
        self.storage = detect_storage_layout(
            self.checkpoint_dir,
            input_roots=storage_input_roots,
            environment=environment,
        )
        self.checkpoint_interval_seconds = max(0.0, float(checkpoint_interval_seconds))
        self.session_timeout_seconds = max(0.0, float(session_timeout_seconds))
        self.pre_timeout_seconds = max(0.0, float(pre_timeout_seconds))
        self.recovery_retention = max(1, int(recovery_retention))
        self._last_checkpoint_monotonic = time.monotonic()
        self._last_checkpoint_path: Optional[Path] = None
        self.state_path = self.checkpoint_dir / TRAINING_STATE_NAME
        self.state: Dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "current_epoch": 0,
            "total_epochs": self.total_epochs,
            "global_step": 0,
            "best_metric": None,
            "last_checkpoint": None,
            "checkpoint_timestamp": None,
            "training_status": STATUS_NOT_STARTED,
            "environment": self.storage.environment,
            "model_config": self.model_config,
            "dataset_fingerprint": self.dataset_id,
            "checkpoint_version": CHECKPOINT_FORMAT_VERSION,
            "architecture_id": self.architecture_id,
        }
        self._write_training_state()

    @property
    def compatibility(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "experiment_id": self.experiment_id,
            "architecture_id": self.architecture_id,
            "model_config": self.model_config,
            "dataset_id": self.dataset_id,
            "training_config": self.training_config,
            "optimizer_config": self.optimizer_config,
            "scheduler_config": self.scheduler_config,
        }

    def _base_metadata(self) -> Dict[str, Any]:
        return {
            "project_name": self.project_name,
            "experiment_id": self.experiment_id,
            "architecture_id": self.architecture_id,
            "model_config": self.model_config,
            "dataset_fingerprint": self.dataset_id,
            "training_config": self.training_config,
            "optimizer_config": self.optimizer_config,
            "scheduler_config": self.scheduler_config,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "framework_version": torch.__version__,
            "environment": self.storage.environment,
        }

    def _write_training_state(self) -> None:
        atomic_json_save(self.state, self.state_path)

    def update_training_metadata(self, **values: Any) -> None:
        self.state.update(values)
        self._write_training_state()

    def mark_status(self, status: str, **values: Any) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Unknown training status: {status}")
        self.state["training_status"] = status
        self.state.update(values)
        self.state["checkpoint_timestamp"] = _utc_now()
        self._write_training_state()

    def checkpoint_due(self) -> bool:
        return (
            self.checkpoint_interval_seconds > 0
            and time.monotonic() - self._last_checkpoint_monotonic
            >= self.checkpoint_interval_seconds
        )

    def save_full(
        self,
        *,
        kind: str,
        models: Union[Mapping[str, torch.nn.Module], torch.nn.Module],
        optimizers: Union[
            Mapping[str, torch.optim.Optimizer],
            torch.optim.Optimizer,
        ],
        schedulers: Any = None,
        grad_scaler: Any = None,
        ema: Any = None,
        resume_epoch: int,
        batch_position: int,
        global_step: int,
        best_metric: Optional[Mapping[str, Any]],
        epochs_without_improvement: int = 0,
        last_completed_epoch: Optional[int] = None,
        epoch_completed: bool = False,
        additional_state: Optional[Mapping[str, Any]] = None,
        status: str = STATUS_CHECKPOINTED,
    ) -> Path:
        if kind not in RESUMABLE_NAMES and not RECOVERY_BACKUP_PATTERN.match(kind):
            raise ValueError(f"Unsupported full checkpoint kind: {kind}")
        model_map = _as_mapping(models, "model")
        optimizer_map = _as_mapping(optimizers, "optimizer")
        scheduler_map = _as_mapping(schedulers, "scheduler")
        scaler_map = _as_mapping(grad_scaler, "grad_scaler")
        ema_map = _as_mapping(ema, "ema")
        now = _utc_now()
        training_state = {
            "resume_epoch": int(resume_epoch),
            "batch_position": int(batch_position),
            "global_step": int(global_step),
            "epoch_completed": bool(epoch_completed),
            "last_completed_epoch": last_completed_epoch,
            "best_metric": dict(best_metric) if best_metric else None,
            "epochs_without_improvement": int(epochs_without_improvement),
            "completed_epochs": (
                int(last_completed_epoch) + 1 if last_completed_epoch is not None else 0
            ),
        }
        payload = {
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "kind": "full_training",
            "metadata": {**self._base_metadata(), "saved_at": now},
            "compatibility": self.compatibility,
            "training_state": training_state,
            "model_state_dicts": {
                name: _cpu_state_dict(model) for name, model in model_map.items()
            },
            "optimizer_state_dicts": {
                name: optimizer.state_dict() for name, optimizer in optimizer_map.items()
            },
            "scheduler_state_dicts": {
                name: scheduler.state_dict()
                for name, scheduler in scheduler_map.items()
                if scheduler is not None and hasattr(scheduler, "state_dict")
            },
            "grad_scaler_state_dicts": {
                name: scaler.state_dict()
                for name, scaler in scaler_map.items()
                if scaler is not None and hasattr(scaler, "state_dict")
            },
            "ema_state_dicts": {
                name: value.state_dict() if hasattr(value, "state_dict") else value
                for name, value in ema_map.items()
            },
            "rng_state": capture_rng_state(),
            "additional_state": dict(additional_state or {}),
        }
        destination = self.checkpoint_dir / kind
        sidecar = {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "experiment_id": self.experiment_id,
            "architecture_id": self.architecture_id,
            "dataset_fingerprint": self.dataset_id,
            "resume_epoch": resume_epoch,
            "batch_position": batch_position,
            "global_step": global_step,
            "epoch_completed": epoch_completed,
            "saved_at": now,
        }
        rotate_recovery = None
        if kind == RECOVERY_NAME and self.recovery_retention > 1:
            rotate_recovery = self._rotate_recovery
        atomic_torch_save(
            payload,
            destination,
            before_replace=rotate_recovery,
        )
        atomic_json_save(sidecar, _metadata_path(destination))
        self._last_checkpoint_monotonic = time.monotonic()
        self._last_checkpoint_path = destination
        self.state.update(
            {
                "current_epoch": (
                    int(last_completed_epoch) + 1
                    if last_completed_epoch is not None
                    else int(resume_epoch) + 1
                ),
                "global_step": int(global_step),
                "best_metric": dict(best_metric) if best_metric else None,
                "last_checkpoint": str(destination),
                "checkpoint_timestamp": now,
                "training_status": status,
            }
        )
        self._write_training_state()
        return destination

    def _rotate_recovery(self) -> None:
        base = self.checkpoint_dir / RECOVERY_NAME
        max_backup = self.recovery_retention - 1
        if max_backup <= 0:
            return
        for index in range(max_backup, 1, -1):
            source = self.checkpoint_dir / f"checkpoint_recovery.{index - 1}.pth"
            destination = self.checkpoint_dir / f"checkpoint_recovery.{index}.pth"
            self._move_recovery_file(source, destination)
        self._move_recovery_file(
            base,
            self.checkpoint_dir / "checkpoint_recovery.1.pth",
        )

    @staticmethod
    def _move_recovery_file(source: Path, destination: Path) -> None:
        if not source.is_file():
            return
        destination.unlink(missing_ok=True)
        _metadata_path(destination).unlink(missing_ok=True)
        os.replace(source, destination)
        source_metadata = _metadata_path(source)
        if source_metadata.is_file():
            os.replace(source_metadata, _metadata_path(destination))

    def save_best(
        self,
        model: torch.nn.Module,
        *,
        epoch: int,
        global_step: int,
        metric: Mapping[str, Any],
    ) -> Path:
        destination = self.checkpoint_dir / BEST_NAME
        atomic_torch_save(_cpu_state_dict(model), destination)
        metadata = {
            "kind": "best_weights",
            "experiment_id": self.experiment_id,
            "architecture_id": self.architecture_id,
            "dataset_fingerprint": self.dataset_id,
            "epoch": int(epoch),
            "global_step": int(global_step),
            "metric": dict(metric),
            "saved_at": _utc_now(),
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        }
        atomic_json_save(metadata, _metadata_path(destination))
        self.state.update(
            {
                "best_metric": dict(metric),
                "best_checkpoint": str(destination),
                "checkpoint_timestamp": metadata["saved_at"],
            }
        )
        self._write_training_state()
        return destination

    def _candidate_paths(self) -> List[Path]:
        if self.resume_path:
            return [self.resume_path]
        candidates: Dict[str, Path] = {}
        for root in self.storage.discovery_roots:
            if not root.exists():
                continue
            if root.is_file():
                if _is_discovery_name(root.name):
                    candidates[os.path.normcase(str(root))] = root
                continue
            for current_root, _, files in os.walk(root):
                for name in files:
                    if _is_discovery_name(name):
                        path = Path(current_root) / name
                        candidates[os.path.normcase(str(path))] = path

        def score(path: Path) -> Tuple[int, str, float]:
            metadata: Mapping[str, Any] = {}
            try:
                with _metadata_path(path).open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
            except (OSError, ValueError, TypeError):
                pass
            return (
                int(metadata.get("global_step", -1)),
                str(metadata.get("saved_at", "")),
                path.stat().st_mtime,
            )

        return sorted(candidates.values(), key=score, reverse=True)

    def _validate_payload(self, payload: Any, source: Path) -> None:
        if not isinstance(payload, Mapping):
            raise CheckpointCompatibilityError(f"{source}: payload is not a mapping")
        version = payload.get("format_version")
        if version != CHECKPOINT_FORMAT_VERSION:
            raise CheckpointCompatibilityError(
                f"{source}: unsupported checkpoint format {version}"
            )
        if self.resume_mode == "weights":
            saved_architecture = payload.get("metadata", {}).get("architecture_id")
            if saved_architecture != self.architecture_id:
                raise CheckpointCompatibilityError(
                    f"{source}: model architecture is incompatible"
                )
            return
        saved_compatibility = payload.get("compatibility")
        if not _deep_equal(saved_compatibility, self.compatibility):
            raise CheckpointCompatibilityError(
                f"{source}: experiment, architecture, dataset, or training "
                "configuration is incompatible"
            )

    def find_and_restore(
        self,
        *,
        models: Union[Mapping[str, torch.nn.Module], torch.nn.Module],
        optimizers: Union[
            Mapping[str, torch.optim.Optimizer],
            torch.optim.Optimizer,
        ],
        schedulers: Any = None,
        grad_scaler: Any = None,
        ema: Any = None,
    ) -> ResumeResult:
        result = ResumeResult()
        if self.resume_mode == "new":
            result.messages.append("Resume disabled; starting a new training run.")
            return result

        candidates = self._candidate_paths()
        if not candidates:
            if self.resume_mode == "full":
                raise CheckpointCompatibilityError(
                    "Full resume was requested, but no checkpoint was found."
                )
            result.messages.append("No checkpoint found.")
            return result

        model_map = _as_mapping(models, "model")
        optimizer_map = _as_mapping(optimizers, "optimizer")
        scheduler_map = _as_mapping(schedulers, "scheduler")
        scaler_map = _as_mapping(grad_scaler, "grad_scaler")
        ema_map = _as_mapping(ema, "ema")

        for candidate in candidates:
            try:
                payload = load_torch_checkpoint(candidate)
            except Exception as exc:  # noqa: BLE001 - torch has many corruption errors
                result.rejected.append(f"{candidate}: cannot load checkpoint ({exc})")
                continue

            if candidate.name in WEIGHTS_ONLY_NAMES:
                if self.resume_mode != "weights":
                    result.rejected.append(
                        f"{candidate}: best checkpoint contains weights only"
                    )
                    continue
                try:
                    sidecar_path = _metadata_path(candidate)
                    if sidecar_path.is_file():
                        with sidecar_path.open("r", encoding="utf-8") as handle:
                            sidecar = json.load(handle)
                        expected = {
                            "experiment_id": self.experiment_id,
                            "architecture_id": self.architecture_id,
                            "dataset_fingerprint": self.dataset_id,
                        }
                        for key, value in expected.items():
                            if key in sidecar and sidecar.get(key) != value:
                                raise CheckpointCompatibilityError(
                                    f"{candidate}: incompatible {key}"
                                )
                    if isinstance(payload, Mapping) and "model_state_dicts" in payload:
                        weights = next(iter(payload["model_state_dicts"].values()))
                    else:
                        weights = payload
                    next(iter(model_map.values())).load_state_dict(weights)
                except Exception as exc:  # noqa: BLE001 - mapping can fail in many ways
                    result.rejected.append(f"{candidate}: cannot load weights ({exc})")
                    continue
                result.checkpoint_path = candidate
                result.resumed = True
                result.weights_only = True
                result.messages.append(f"Loaded model weights from {candidate}.")
                return result

            try:
                self._validate_payload(payload, candidate)
                if self.resume_mode == "weights":
                    weights = payload.get("model_state_dicts", {})
                    model_name = next(iter(model_map))
                    model_map[model_name].load_state_dict(weights[model_name])
                    result.checkpoint_path = candidate
                    result.resumed = True
                    result.weights_only = True
                    result.messages.append(
                        f"Loaded model weights only from {candidate}; "
                        "optimizer and epoch progress were not restored."
                    )
                    return result

                for name, model in model_map.items():
                    model.load_state_dict(payload["model_state_dicts"][name])
                for name, optimizer in optimizer_map.items():
                    optimizer.load_state_dict(payload["optimizer_state_dicts"][name])
                for name, scheduler in scheduler_map.items():
                    state = payload.get("scheduler_state_dicts", {}).get(name)
                    if scheduler is not None and state is not None:
                        scheduler.load_state_dict(state)
                for name, scaler in scaler_map.items():
                    state = payload.get("grad_scaler_state_dicts", {}).get(name)
                    if scaler is not None and state is not None:
                        scaler.load_state_dict(state)
                for name, value in ema_map.items():
                    state = payload.get("ema_state_dicts", {}).get(name)
                    if state is not None:
                        if hasattr(value, "load_state_dict"):
                            value.load_state_dict(state)
                        else:
                            ema_map[name] = state
                restore_rng_state(payload.get("rng_state"))
            except Exception as exc:  # noqa: BLE001 - include the exact restore failure
                result.rejected.append(f"{candidate}: {exc}")
                continue

            training_state = payload.get("training_state", {})
            result.checkpoint_path = candidate
            result.payload = dict(payload)
            result.resumed = True
            result.start_epoch = int(training_state.get("resume_epoch", 0))
            result.batch_position = int(training_state.get("batch_position", 0))
            result.messages.append(f"Checkpoint found: {candidate}")
            result.messages.append(
                f"Last completed epoch: {training_state.get('last_completed_epoch')}"
            )
            result.messages.append(
                f"Training will continue at epoch {result.start_epoch + 1}, "
                f"batch {result.batch_position + 1}."
            )
            self.state.update(
                {
                    "last_checkpoint": str(candidate),
                    "training_status": STATUS_RESUMABLE,
                    "current_epoch": result.start_epoch + 1,
                    "global_step": int(training_state.get("global_step", 0)),
                    "best_metric": training_state.get("best_metric"),
                }
            )
            self._write_training_state()
            return result

        result.messages.append("No compatible checkpoint could be restored.")
        if (self.resume_path and result.rejected) or self.resume_mode == "full":
            raise CheckpointCompatibilityError(
                "Full checkpoint resume failed:\n"
                + "\n".join(result.rejected)
            )
        return result

    def report_storage(
        self,
        root: Optional[Union[str, os.PathLike]] = None,
    ) -> str:
        scan_root = Path(root or self.storage.output_root).expanduser().resolve()
        categories = {
            "CHECKPOINT": 0,
            "LOGS": 0,
            "GENERATED SAMPLES": 0,
            "CACHE": 0,
            "OTHER": 0,
        }
        files: List[Tuple[int, Path]] = []
        if scan_root.exists():
            for current_root, _, names in os.walk(scan_root):
                for name in names:
                    path = Path(current_root) / name
                    try:
                        size = path.stat().st_size
                    except OSError:
                        continue
                    lowered = {part.lower() for part in path.parts}
                    suffix = path.suffix.lower()
                    if suffix in {".pth", ".pt", ".ckpt"}:
                        category = "CHECKPOINT"
                    elif suffix in {".log", ".jsonl", ".json"} or (
                        "events.out.tfevents" in name
                    ):
                        category = "LOGS"
                    elif "generated" in lowered or "samples" in lowered:
                        category = "GENERATED SAMPLES"
                    elif suffix in {".npy", ".npz", ".pt", ".pth", ".tmp"} or any(
                        marker in lowered for marker in ("cache", "features")
                    ):
                        category = "CACHE"
                    else:
                        category = "OTHER"
                    categories[category] += size
                    files.append((size, path))

        def human_size(size: int) -> str:
            value = float(size)
            for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
                if value < 1024 or unit == "TiB":
                    return f"{value:.2f} {unit}"
                value /= 1024
            return f"{value:.2f} TiB"

        lines = [f"STORAGE REPORT: {scan_root}"]
        for category in (
            "CHECKPOINT",
            "LOGS",
            "GENERATED SAMPLES",
            "CACHE",
            "OTHER",
        ):
            lines.append(f"{category}: {human_size(categories[category])}")
        total = sum(categories.values())
        lines.append(f"TOTAL: {human_size(total)}")
        lines.append("TOP 20 LARGEST FILES:")
        for size, path in sorted(files, reverse=True)[:20]:
            try:
                display = path.relative_to(scan_root)
            except ValueError:
                display = path
            lines.append(f"{human_size(size):>12}  {display}")
        return "\n".join(lines)


class GracefulStopController:
    """Stop at a safe batch boundary and checkpoint before a runtime deadline."""

    def __init__(
        self,
        *,
        session_timeout_seconds: float,
        pre_timeout_seconds: float,
    ):
        self.session_timeout_seconds = max(0.0, float(session_timeout_seconds))
        self.pre_timeout_seconds = max(0.0, float(pre_timeout_seconds))
        self.started_at = time.monotonic()
        self.stop_requested = False
        self.reason = ""
        self._previous_handlers: Dict[signal.Signals, Any] = {}

    def request_stop(self, reason: str) -> None:
        if not self.stop_requested:
            self.stop_requested = True
            self.reason = reason

    def _handle_signal(self, signum, _frame) -> None:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        self.request_stop(f"received {signal_name}")

    def __enter__(self) -> "GracefulStopController":  # noqa: PYI034, UP037
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                try:
                    self._previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, self._handle_signal)
                except (ValueError, OSError):
                    pass
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        if threading.current_thread() is threading.main_thread():
            for signum, handler in self._previous_handlers.items():
                try:
                    signal.signal(signum, handler)
                except (ValueError, OSError):
                    pass
        self._previous_handlers.clear()

    @property
    def remaining_seconds(self) -> Optional[float]:
        if self.session_timeout_seconds <= 0:
            return None
        deadline = self.started_at + self.session_timeout_seconds
        return max(0.0, deadline - time.monotonic())

    def should_stop(self) -> bool:
        if self.stop_requested:
            return True
        remaining = self.remaining_seconds
        if remaining is not None and remaining <= self.pre_timeout_seconds:
            self.request_stop("approaching configured session timeout")
            return True
        return False


def find_best_weights(
    directory: Union[str, os.PathLike],
) -> Optional[Path]:
    """Find the new best checkpoint while accepting legacy checkpoints."""
    base = Path(directory)
    for name in (BEST_NAME, LEGACY_BEST_NAME):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def is_metric_improved(previous: float, candidate: float) -> bool:
    """Best checkpoints are updated only for a strict metric improvement."""
    return float(candidate) > float(previous)
