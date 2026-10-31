"""One-pass feature caching for efficient adaptive-fusion calibration."""

import json
import os

import numpy as np
import torch
from tqdm import tqdm

from checkpoint_manager import dataset_fingerprint, model_signature
from dataSets import BranchFeatureDataset, FusionFeatureDataset


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
) -> FusionFeatureDataset:
    """Extract and persist semantic, artifact, and optional NPR features."""
    os.makedirs(cache_dir, exist_ok=True)
    sem_path = os.path.join(cache_dir, f"{split}_sem.npy")
    art_path = os.path.join(cache_dir, f"{split}_art.npy")
    npr_path = os.path.join(cache_dir, f"{split}_npr.npy")
    label_path = os.path.join(cache_dir, f"{split}_labels.npy")
    metadata_path = os.path.join(cache_dir, f"{split}_metadata.json")

    has_npr = model.npr is not None
    required = [sem_path, art_path, label_path, metadata_path]
    if has_npr:
        required.append(npr_path)
    if not force and all(os.path.exists(path) for path in required):
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if (
            metadata.get("length") == len(dataset)
            and metadata.get("fusion_version") == model.fusion_version
            and metadata.get("semantic_dim") == model.sem.fc.in_features
            and metadata.get("artifact_dim") == model.art.fc.in_features
            and metadata.get("npr_dim")
            == (model.npr.fc.in_features if has_npr else None)
        ):
            return FusionFeatureDataset(
                np.load(sem_path, mmap_mode="r"),
                np.load(art_path, mmap_mode="r"),
                np.load(label_path, mmap_mode="r"),
                np.load(npr_path, mmap_mode="r") if has_npr else None,
            )

    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    sem_cache = _open_memmap(sem_path, (len(dataset), model.sem.fc.in_features))
    art_cache = _open_memmap(art_path, (len(dataset), model.art.fc.in_features))
    npr_cache = (
        _open_memmap(npr_path, (len(dataset), model.npr.fc.in_features))
        if has_npr
        else None
    )
    labels = np.empty(len(dataset), dtype=np.int8)

    offset = 0
    for batch in tqdm(loader, desc=f"Caching fusion features ({split})"):
        sem_images = batch[0].to(device, non_blocking=True)
        art_images = batch[1].to(device, non_blocking=True)
        batch_labels = batch[-1]
        with torch.no_grad():
            sem_feat, art_feat, npr_feat = model.extract_features(
                sem_images,
                art_images,
                include_npr=not (has_npr and len(batch) == 4),
            )
            if has_npr and len(batch) == 4:
                npr_images = batch[2].to(device, non_blocking=True)
                npr_feat = model.npr.artifact_encoder(npr_images)

        count = sem_images.shape[0]
        sem_cache[offset : offset + count] = (
            sem_feat.float().cpu().numpy().astype(np.float16)
        )
        art_cache[offset : offset + count] = (
            art_feat[:, 0].float().cpu().numpy().astype(np.float16)
        )
        if npr_cache is not None:
            npr_cache[offset : offset + count] = (
                npr_feat.float().cpu().numpy().astype(np.float16)
            )
        labels[offset : offset + count] = batch_labels.cpu().numpy()
        offset += count

    if offset != len(dataset):
        raise RuntimeError(
            f"Feature cache wrote {offset} of {len(dataset)} samples"
        )

    sem_cache.flush()
    art_cache.flush()
    if npr_cache is not None:
        npr_cache.flush()
    np.save(label_path, labels)
    metadata = {
        "length": len(dataset),
        "fusion_version": model.fusion_version,
        "semantic_dim": model.sem.fc.in_features,
        "artifact_dim": model.art.fc.in_features,
        "npr_dim": model.npr.fc.in_features if has_npr else None,
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
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
) -> BranchFeatureDataset:
    """Extract frozen branch features once and reuse them on later epochs."""
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

    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
    )
    feature_cache = _open_memmap(
        feature_path,
        (len(dataset), feature_dim),
    )
    labels = np.empty(len(dataset), dtype=np.int8)

    if branch == "semantic":
        encode_fn = (
            getattr(model.clip, "encode_image", None)
            or model.clip.get_image_features
        )
    else:
        encode_fn = model.artifact_encoder

    offset = 0
    for images, batch_labels in tqdm(
        loader,
        desc=f"Caching {branch} features ({split})",
    ):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            features = encode_fn(images)
        count = images.shape[0]
        feature_cache[offset : offset + count] = (
            features.float().cpu().numpy().astype(np.float16)
        )
        labels[offset : offset + count] = batch_labels.cpu().numpy()
        offset += count

    if offset != len(dataset):
        raise RuntimeError(
            f"Feature cache wrote {offset} of {len(dataset)} samples"
        )

    feature_cache.flush()
    np.save(label_path, labels)
    metadata = {
        "length": len(dataset),
        "branch": branch,
        "feature_dim": feature_dim,
        "dataset_fingerprint": dataset_id,
        "architecture_id": architecture_id,
        "dtype": "float16",
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

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
