"""One-pass feature caching for efficient adaptive-fusion calibration."""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from checkpoint_manager import dataset_fingerprint, model_signature
from dataSets import BranchFeatureDataset, FusionFeatureDataset


class FeatureCacheInterrupted(RuntimeError):
    """Raised after durable partial cache progress has been saved."""


def _atomic_json_save(path: str, payload: dict) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
    os.replace(temporary, path)


def _cache_loader(dataset, *, batch_size, num_workers, device, start_offset=0,
                  prefetch_factor=2):
    """Build only the unfinished part of a sequential cache pass."""
    remaining = torch.utils.data.Subset(dataset, range(start_offset, len(dataset)))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.startswith("cuda"),
    }
    if num_workers:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(remaining, **kwargs)


def _as_fp16_numpy(features: torch.Tensor) -> np.ndarray:
    return features.detach().to(device="cpu", dtype=torch.float16).numpy()


def _open_memmap(path: str, shape, dtype=np.float16):
    return np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=tuple(shape),
    )


def build_fusion_feature_cache(
    model,
    dataset,
    cache_dir: str,
    split: str,
    device: str,
    batch_size: int = 32,
    num_workers: int = 4,
    force: bool = False,
    precision: str = "fp32",
    prefetch_factor: int = 2,
    checkpoint_batches: int = 5,
    should_stop=None,
) -> FusionFeatureDataset:
    """Extract fusion inputs once, atomically publishing only a complete cache."""
    os.makedirs(cache_dir, exist_ok=True)
    sem_path = os.path.join(cache_dir, f"{split}_sem.npy")
    art_path = os.path.join(cache_dir, f"{split}_art.npy")
    npr_path = os.path.join(cache_dir, f"{split}_npr.npy")
    label_path = os.path.join(cache_dir, f"{split}_labels.npy")
    metadata_path = os.path.join(cache_dir, f"{split}_metadata.json")

    has_npr = model.npr is not None
    dataset_id = dataset_fingerprint(dataset)
    architecture_id = model_signature(model)
    expected_metadata = {
        "length": len(dataset),
        "fusion_version": model.fusion_version,
        "semantic_dim": model.sem.fc.in_features,
        "artifact_dim": model.art.fc.in_features,
        "npr_dim": model.npr.fc.in_features if has_npr else None,
        "dataset_fingerprint": dataset_id,
        "architecture_id": architecture_id,
        "dtype": "float16",
    }
    required = [sem_path, art_path, label_path, metadata_path]
    if has_npr:
        required.append(npr_path)
    if not force and all(os.path.exists(path) for path in required):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        # Old complete caches without fingerprints remain readable; new caches
        # additionally reject stale data after a dataset/model change.
        legacy_keys = ("length", "fusion_version", "semantic_dim", "artifact_dim", "npr_dim")
        if all(metadata.get(key) == expected_metadata[key] for key in legacy_keys) and (
            "dataset_fingerprint" not in metadata
            or (
                metadata.get("dataset_fingerprint") == dataset_id
                and metadata.get("architecture_id") == architecture_id
            )
        ):
            return FusionFeatureDataset(
                np.load(sem_path, mmap_mode="r"),
                np.load(art_path, mmap_mode="r"),
                np.load(label_path, mmap_mode="r"),
                np.load(npr_path, mmap_mode="r") if has_npr else None,
            )

    partial_paths = {
        "sem": f"{sem_path}.partial",
        "art": f"{art_path}.partial",
        "labels": f"{label_path}.partial",
    }
    if has_npr:
        partial_paths["npr"] = f"{npr_path}.partial"
    progress_path = os.path.join(cache_dir, f"{split}_fusion.progress.json")
    offset = 0
    if not force and all(os.path.exists(path) for path in (*partial_paths.values(), progress_path)):
        try:
            with open(progress_path, "r", encoding="utf-8") as handle:
                progress = json.load(handle)
            if all(progress.get(key) == value for key, value in expected_metadata.items()):
                offset = max(0, min(int(progress.get("offset", 0)), len(dataset)))
        except (OSError, ValueError, TypeError):
            offset = 0
    if offset:
        sem_cache = np.lib.format.open_memmap(partial_paths["sem"], mode="r+")
        art_cache = np.lib.format.open_memmap(partial_paths["art"], mode="r+")
        labels = np.lib.format.open_memmap(partial_paths["labels"], mode="r+")
        npr_cache = (
            np.lib.format.open_memmap(partial_paths["npr"], mode="r+")
            if has_npr else None
        )
    else:
        sem_cache = _open_memmap(partial_paths["sem"], (len(dataset), model.sem.fc.in_features))
        art_cache = _open_memmap(partial_paths["art"], (len(dataset), model.art.fc.in_features))
        labels = _open_memmap(partial_paths["labels"], (len(dataset),), dtype=np.int8)
        npr_cache = (
            _open_memmap(partial_paths["npr"], (len(dataset), model.npr.fc.in_features))
            if has_npr else None
        )

    model.eval()
    loader = _cache_loader(
        dataset, batch_size=batch_size, num_workers=num_workers, device=device,
        start_offset=offset, prefetch_factor=prefetch_factor,
    )
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_enabled = device_type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    started = time.perf_counter()
    for batch_index, batch in enumerate(tqdm(
        loader, desc=f"Caching fusion features ({split})", initial=offset // batch_size,
        total=(len(dataset) + batch_size - 1) // batch_size,
    )):
        sem_images = batch[0].to(device, non_blocking=True)
        art_images = batch[1].to(device, non_blocking=True)
        batch_labels = batch[-1]
        with torch.inference_mode(), torch.autocast(
            device_type=device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            sem_feat, art_feat, npr_feat = model.extract_features(
                sem_images,
                art_images,
                include_npr=not (has_npr and len(batch) == 4),
            )
            if has_npr and len(batch) == 4:
                npr_images = batch[2].to(device, non_blocking=True)
                npr_feat = model.npr.artifact_encoder(npr_images)

        count = sem_images.shape[0]
        sem_cache[offset : offset + count] = _as_fp16_numpy(sem_feat)
        art_cache[offset : offset + count] = _as_fp16_numpy(art_feat[:, 0])
        if npr_cache is not None:
            npr_cache[offset : offset + count] = _as_fp16_numpy(npr_feat)
        labels[offset : offset + count] = batch_labels.cpu().numpy()
        offset += count
        if (batch_index + 1) % max(1, checkpoint_batches) == 0:
            sem_cache.flush()
            art_cache.flush()
            labels.flush()
            if npr_cache is not None:
                npr_cache.flush()
            _atomic_json_save(progress_path, {**expected_metadata, "offset": offset})
        if should_stop is not None and should_stop():
            sem_cache.flush()
            art_cache.flush()
            labels.flush()
            if npr_cache is not None:
                npr_cache.flush()
            _atomic_json_save(progress_path, {**expected_metadata, "offset": offset})
            raise FeatureCacheInterrupted(
                f"Fusion feature cache interrupted at {offset}/{len(dataset)} samples"
            )

    if offset != len(dataset):
        raise RuntimeError(
            f"Feature cache wrote {offset} of {len(dataset)} samples"
        )

    sem_cache.flush()
    art_cache.flush()
    if npr_cache is not None:
        npr_cache.flush()
    del sem_cache, art_cache, labels, npr_cache
    os.replace(partial_paths["sem"], sem_path)
    os.replace(partial_paths["art"], art_path)
    os.replace(partial_paths["labels"], label_path)
    if has_npr:
        os.replace(partial_paths["npr"], npr_path)
    _atomic_json_save(metadata_path, expected_metadata)
    Path(progress_path).unlink(missing_ok=True)
    elapsed = max(time.perf_counter() - started, 1e-9)
    print(
        f"[FEATURE CACHE] fusion {split}: {len(dataset) / elapsed:.1f} samples/s | "
        f"{elapsed:.1f}s | precision={precision}", flush=True,
    )
    base = getattr(dataset, "base", None)
    if base is not None and hasattr(base, "total_list"):
        manifest_path = os.path.join(cache_dir, f"{split}_paths.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            for path in base.total_list:
                record = {
                    "path": path,
                    "label": int(base.labels_dict[path]),
                    "split": split,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return FusionFeatureDataset(
        np.load(sem_path, mmap_mode="r"),
        np.load(art_path, mmap_mode="r"),
        np.load(label_path, mmap_mode="r"),
        np.load(npr_path, mmap_mode="r") if has_npr else None,
    )


def build_branch_feature_cache(
    model,
    dataset,
    cache_dir: str,
    split: str,
    branch: str,
    device: str,
    batch_size: int = 32,
    num_workers: int = 4,
    force: bool = False,
    precision: str = "fp32",
    prefetch_factor: int = 2,
    checkpoint_batches: int = 5,
    should_stop=None,
) -> BranchFeatureDataset:
    """Extract frozen branch features once, with interruption-safe resume."""
    if branch not in {"artifact", "semantic", "npr"}:
        raise ValueError(f"Unsupported branch feature cache: {branch}")
    if branch != "semantic" and any(
        parameter.requires_grad for parameter in model.artifact_encoder.parameters()
    ):
        raise ValueError(
            f"Cannot cache {branch} outputs while its encoder is trainable"
        )

    os.makedirs(cache_dir, exist_ok=True)
    feature_path = os.path.join(cache_dir, f"{split}_{branch}.npy")
    label_path = os.path.join(cache_dir, f"{split}_labels.npy")
    metadata_path = os.path.join(cache_dir, f"{split}_metadata.json")
    dataset_id = dataset_fingerprint(dataset)
    architecture_id = model_signature(model)
    feature_dim = int(model.fc.in_features)

    if not force and all(
        os.path.exists(path)
        for path in (feature_path, label_path, metadata_path)
    ):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            metadata.get("length") == len(dataset)
            and metadata.get("branch") == branch
            and metadata.get("feature_dim") == feature_dim
            and metadata.get("dataset_fingerprint") == dataset_id
            and metadata.get("architecture_id") == architecture_id
        ):
            features = np.load(feature_path, mmap_mode="r")
            labels = np.load(label_path, mmap_mode="r")
            if (
                features.shape == (len(dataset), feature_dim)
                and labels.shape == (len(dataset),)
            ):
                return BranchFeatureDataset(features, labels)

    partial_feature_path = f"{feature_path}.partial"
    partial_label_path = f"{label_path}.partial"
    progress_path = os.path.join(cache_dir, f"{split}_{branch}.progress.json")
    expected_progress = {
        "length": len(dataset),
        "branch": branch,
        "feature_dim": feature_dim,
        "dataset_fingerprint": dataset_id,
        "architecture_id": architecture_id,
        "dtype": "float16",
    }
    offset = 0
    if not force and all(os.path.exists(path) for path in (
        partial_feature_path, partial_label_path, progress_path
    )):
        try:
            with open(progress_path, "r", encoding="utf-8") as handle:
                progress = json.load(handle)
            if all(progress.get(key) == value for key, value in expected_progress.items()):
                offset = max(0, min(int(progress.get("offset", 0)), len(dataset)))
        except (OSError, ValueError, TypeError):
            offset = 0
    if offset:
        feature_cache = np.lib.format.open_memmap(partial_feature_path, mode="r+")
        labels = np.lib.format.open_memmap(partial_label_path, mode="r+")
    else:
        feature_cache = _open_memmap(partial_feature_path, (len(dataset), feature_dim))
        labels = _open_memmap(partial_label_path, (len(dataset),), dtype=np.int8)

    model.eval()
    loader = _cache_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        start_offset=offset,
        prefetch_factor=prefetch_factor,
    )

    if branch == "semantic":
        encode_fn = (
            getattr(model.clip, "encode_image", None)
            or model.clip.get_image_features
        )
    else:
        encode_fn = model.artifact_encoder

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_enabled = device_type == "cuda" and precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    cache_started = time.perf_counter()
    data_wait = 0.0
    forward_events = []
    progress = tqdm(
        loader,
        desc=f"Caching {branch} features ({split})",
        initial=offset // batch_size,
        total=(len(dataset) + batch_size - 1) // batch_size,
    )
    iterator = iter(progress)
    batch_index = 0
    while True:
        fetch_started = time.perf_counter()
        try:
            images, batch_labels = next(iterator)
        except StopIteration:
            break
        data_wait += time.perf_counter() - fetch_started
        images = images.to(device, non_blocking=True)
        timing_start = timing_end = None
        if device_type == "cuda":
            timing_start = torch.cuda.Event(enable_timing=True)
            timing_end = torch.cuda.Event(enable_timing=True)
            timing_start.record()
        with torch.inference_mode(), torch.autocast(
            device_type=device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            features = encode_fn(images)
        if timing_end is not None:
            timing_end.record()
            forward_events.append((timing_start, timing_end))
        count = images.shape[0]
        feature_cache[offset : offset + count] = _as_fp16_numpy(features)
        labels[offset : offset + count] = batch_labels.numpy()
        offset += count
        batch_index += 1
        if batch_index % max(1, checkpoint_batches) == 0:
            feature_cache.flush()
            labels.flush()
            _atomic_json_save(progress_path, {**expected_progress, "offset": offset})
        if should_stop is not None and should_stop():
            feature_cache.flush()
            labels.flush()
            _atomic_json_save(progress_path, {**expected_progress, "offset": offset})
            raise FeatureCacheInterrupted(
                f"Feature cache interrupted at {offset}/{len(dataset)} samples"
            )

    if offset != len(dataset):
        raise RuntimeError(
            f"Feature cache wrote {offset} of {len(dataset)} samples"
        )

    feature_cache.flush()
    labels.flush()
    if device_type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = sum(
        start.elapsed_time(end) for start, end in forward_events
    ) / 1000
    elapsed = max(time.perf_counter() - cache_started, 1e-9)
    del feature_cache, labels
    os.replace(partial_feature_path, feature_path)
    os.replace(partial_label_path, label_path)
    _atomic_json_save(metadata_path, expected_progress)
    Path(progress_path).unlink(missing_ok=True)

    encoder_name = "CLIP forward" if branch == "semantic" else "VAE/residual forward"
    print(
        f"[FEATURE CACHE] {branch} {split}: {len(dataset) / elapsed:.1f} samples/s | "
        f"data wait {data_wait:.2f}s | {encoder_name} {forward_seconds:.2f}s | "
        f"total {elapsed:.2f}s | precision={precision}", flush=True,
    )

    if hasattr(dataset, "total_list"):
        manifest_path = os.path.join(cache_dir, f"{split}_paths.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            for path in dataset.total_list:
                record = {
                    "path": path,
                    "label": int(dataset.labels_dict[path]),
                    "split": split,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return BranchFeatureDataset(
        np.load(feature_path, mmap_mode="r"),
        np.load(label_path, mmap_mode="r"),
    )
