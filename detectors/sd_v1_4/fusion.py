"""CO-SPY fusion detectors with backward-compatible v1 and adaptive v2/v3."""

from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .artifact import AlignedCrop, ArtifactDetector
from .base import BaseDetector
from .fusion_head import AdaptiveFusionHead
from .npr import NPRDetector
from .semantic import SemanticDetector


class HighFrequencyCrop:
    """Select an aligned crop with the strongest local edge energy.

    The score is a DCT-like high-pass proxy: horizontal and vertical first
    differences followed by local aggregation. It is inexpensive and does not
    require OpenCV at evaluation time.
    """

    def __init__(self, size: int = 224, grid: int = 32):
        self.size = size
        self.grid = max(8, grid)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = _ensure_min_size(image, self.size)
        width, height = image.size
        max_x = width - self.size
        max_y = height - self.size
        if max_x == 0 and max_y == 0:
            return image.copy()

        gray = image.convert("L")
        gray.thumbnail((256, 256), Image.Resampling.BILINEAR)
        scores = np.asarray(gray, dtype=np.float32)
        dy = np.abs(np.diff(scores, axis=0, append=scores[-1:, :]))
        dx = np.abs(np.diff(scores, axis=1, append=scores[:, -1:]))
        energy = dx + dy

        patch = max(8, min(self.grid, scores.shape[0], scores.shape[1]))
        integral = np.pad(energy, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        local = (
            integral[patch:, patch:]
            - integral[:-patch, patch:]
            - integral[patch:, :-patch]
            + integral[:-patch, :-patch]
        )
        best_y, best_x = np.unravel_index(int(np.argmax(local)), local.shape)

        thumb_w, thumb_h = gray.size
        x = round(best_x / max(1, thumb_w - patch) * max_x)
        y = round(best_y / max(1, thumb_h - patch) * max_y)
        x = min(max(0, x // 8 * 8), max_x // 8 * 8)
        y = min(max(0, y // 8 * 8), max_y // 8 * 8)
        return image.crop((x, y, x + self.size, y + self.size))


def _ensure_min_size(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if min(width, height) >= size:
        return image
    scale = size / min(width, height)
    return image.resize(
        (max(size, round(width * scale)), max(size, round(height * scale))),
        Image.Resampling.BICUBIC,
    )


class ArtifactViewTransform:
    """Return one or more normalized artifact views as a stacked tensor."""

    def __init__(
        self,
        size: int,
        mean: Sequence[float],
        std: Sequence[float],
        num_views: int = 1,
    ):
        if num_views not in (1, 2):
            raise ValueError("artifact_views must be 1 or 2")
        self.size = size
        self.num_views = num_views
        self.center = AlignedCrop(size, train=False)
        self.high_frequency = HighFrequencyCrop(size)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def _tensor(self, image: Image.Image) -> torch.Tensor:
        return self.normalize(self.to_tensor(image))

    def __call__(self, image: Image.Image) -> torch.Tensor:
        center = self.center(image)
        crops = [center]
        if self.num_views == 2:
            crops.append(self.high_frequency(image))
        return torch.stack([self._tensor(crop) for crop in crops], dim=0)


class CoSpyFusionDetector(BaseDetector):
    """Fuse frozen CO-SPY feature streams.

    v1 preserves the released two-logit checkpoint format. v2 learns regulators
    over the full semantic/artifact features. v3 adds an optional NPR stream.
    """

    def __init__(
        self,
        semantic_weights_path: str,
        artifact_weights_path: str,
        npr_weights_path: Optional[str] = None,
        num_classes: int = 1,
        fusion_version: str = "v1",
        branch_dropout: float = 0.15,
        artifact_views: int = 1,
        semantic_downscale_prob: float = 0.0,
    ):
        super().__init__()
        if fusion_version not in {"v1", "v2", "v3", "v3_npr"}:
            raise ValueError(f"Unknown fusion version: {fusion_version}")
        self.fusion_version = "v3" if fusion_version == "v3_npr" else fusion_version
        self.num_classes = num_classes
        self.artifact_views = artifact_views
        self.branch_dropout = branch_dropout

        self.sem = SemanticDetector(downscale_prob=semantic_downscale_prob)
        self.sem.load_weights(semantic_weights_path)
        self.art = ArtifactDetector()
        self.art.load_weights(artifact_weights_path)

        modules = [self.sem, self.art]
        self.npr = None
        if self.fusion_version == "v3":
            if not npr_weights_path:
                raise ValueError("v3 fusion requires npr_weights_path")
            self.npr = NPRDetector()
            self.npr.load_weights(npr_weights_path)
            modules.append(self.npr)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        self.fc = torch.nn.Linear(2, num_classes)
        feature_dims = [self.sem.fc.in_features, self.art.fc.in_features]
        if self.npr is not None:
            feature_dims.append(self.npr.fc.in_features)
        self.fusion_head = (
            AdaptiveFusionHead(
                feature_dims,
                num_classes=num_classes,
                branch_dropout=branch_dropout,
            )
            if self.fusion_version != "v1"
            else None
        )
        if self.fusion_head is not None:
            self.fc.requires_grad_(False)

        self.sem_transform = transforms.Compose(
            [transforms.Normalize(self.sem.mean, self.sem.std)]
        )
        self.art_transform = transforms.Compose(
            [
                transforms.Resize(self.art.cropSize, antialias=False),
                transforms.Normalize(self.art.mean, self.art.std),
            ]
        )
        self.npr_transform = transforms.Compose(
            [
                transforms.Resize(224, antialias=False),
                transforms.Normalize([0.0] * 3, [1.0] * 3),
            ]
        )
        self._build_transforms()

    def _build_transforms(self):
        self.loadSize = 384
        self.cropSize = 384
        self.blur_prob = 0.0
        self.blur_sig = [0.0, 3.0]
        self.jpg_prob = 0.5
        self.jpg_method = ["cv2", "pil"]
        self.jpg_qual = list(range(70, 96))

        random_crop = transforms.RandomCrop(self.cropSize)
        center_crop = transforms.CenterCrop(self.cropSize)
        flip_func = transforms.RandomHorizontalFlip()
        rz_func = transforms.Resize(self.loadSize)

        self.train_transform = transforms.Compose(
            [
                flip_func,
                rz_func,
                random_crop,
                transforms.ToTensor(),
            ]
        )
        self.val_transform = transforms.Compose(
            [rz_func, center_crop, transforms.ToTensor()]
        )
        self.test_transform = self.val_transform

    def train(self, mode: bool = True):
        super().train(mode)
        self.sem.eval()
        self.art.eval()
        if self.npr is not None:
            self.npr.eval()
        return self

    def artifact_view_transform(self, num_views: int) -> ArtifactViewTransform:
        return ArtifactViewTransform(
            self.art.cropSize,
            self.art.mean,
            self.art.std,
            num_views=num_views,
        )

    @staticmethod
    def _prepare_artifact_views(x_art: torch.Tensor) -> torch.Tensor:
        if x_art.ndim == 2:
            return x_art.unsqueeze(1)
        if x_art.ndim == 4:
            return x_art.unsqueeze(1)
        if x_art.ndim == 5:
            return x_art
        raise ValueError(
            f"Artifact views/features must be 2-D, 4-D, or 5-D, "
            f"got {tuple(x_art.shape)}"
        )

    def extract_features(
        self,
        x_sem: torch.Tensor,
        x_art: torch.Tensor,
        include_npr: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract features from branch-ready (already normalized) inputs."""
        x_art = self._prepare_artifact_views(x_art)

        batch_size, num_views = x_art.shape[:2]
        with torch.no_grad():
            sem_feat = self.sem.clip.encode_image(x_sem)
            flat_art = x_art.reshape(
                batch_size * num_views, *x_art.shape[2:]
            )
            art_feat = self.art.artifact_encoder(flat_art)
            art_feat = art_feat.reshape(batch_size, num_views, -1)
            npr_feat = None
            if self.npr is not None and include_npr:
                npr_input = self.npr_transform(x_art[:, 0])
                npr_feat = self.npr.artifact_encoder(npr_input)
        return sem_feat, art_feat, npr_feat

    def forward_features(
        self,
        sem_feat: torch.Tensor,
        art_feat: torch.Tensor,
        npr_feat: Optional[torch.Tensor] = None,
        return_feat: bool = False,
    ):
        if self.fusion_head is None:
            raise RuntimeError("forward_features is only valid for adaptive fusion")
        art_feat = self._prepare_artifact_views(art_feat)
        num_views = art_feat.shape[1]
        sem_feat = sem_feat.float()
        art_feat = art_feat.float()
        npr_feat = npr_feat.float() if npr_feat is not None else None

        view_logits = []
        last_fused = None
        for view_index in range(num_views):
            streams = [sem_feat, art_feat[:, view_index]]
            if self.npr is not None:
                if npr_feat is None:
                    raise ValueError("v3 fusion requires NPR features")
                streams.append(npr_feat)
            logits, info = self.fusion_head.forward_streams(
                streams, return_info=True
            )
            view_logits.append(logits)
            last_fused = info["features"]
        out = torch.stack(view_logits, dim=1).mean(dim=1)
        if return_feat:
            return last_fused, out
        return out

    def forward(self, x, return_feat: bool = False):
        if self.fusion_version == "v1":
            if isinstance(x, (tuple, list)):
                x_sem, x_art = x[:2]
            else:
                x_sem, x_art = x, x
            sem_logit = self.sem(self.sem_transform(x_sem))
            art_logit = self.art(self.art_transform(x_art))
            feat = torch.cat([sem_logit, art_logit], dim=1)
            keep = torch.ones_like(feat)
            if self.training and self.fc.training and self.branch_dropout > 0:
                drop = torch.rand(feat.shape[0], device=feat.device) < (
                    self.branch_dropout
                )
                stream = torch.randint(0, 2, (feat.shape[0],), device=feat.device)
                for index in range(2):
                    keep[:, index] = torch.logical_not(
                        torch.logical_and(drop, stream == index)
                    ).to(feat.dtype)
                feat = feat * keep
            out = self.fc(feat)
            if return_feat:
                return feat, out
            return out

        if isinstance(x, (tuple, list)):
            if len(x) not in (2, 3):
                raise ValueError("Adaptive fusion expects (semantic, artifact) views")
            x_sem, x_art = x[:2]
        else:
            x_sem = self.sem_transform(x)
            x_art = self.art_transform(x).unsqueeze(1)
        sem_feat, art_feat, npr_feat = self.extract_features(x_sem, x_art)
        return self.forward_features(
            sem_feat, art_feat, npr_feat, return_feat=return_feat
        )

    def predict_views(self, sem_images, artifact_views):
        sem_images = sem_images.to(next(self.parameters()).device)
        artifact_views = artifact_views.to(next(self.parameters()).device)
        return self.forward((sem_images, artifact_views)).sigmoid().flatten().tolist()

    def save_weights(self, weights_path):
        if self.fusion_version == "v1":
            payload = {
                "version": "v1",
                "fc.weight": self.fc.weight.detach().cpu(),
                "fc.bias": self.fc.bias.detach().cpu(),
            }
        else:
            payload = {
                "version": self.fusion_version,
                "fusion_head": {
                    key: value.detach().cpu()
                    for key, value in self.fusion_head.state_dict().items()
                },
                "feature_dims": list(self.fusion_head.feature_dims),
                "branch_dropout": self.fusion_head.branch_dropout,
            }
        torch.save(payload, weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location="cpu")
        saved_version = weights.get("version", "v1")
        if saved_version != self.fusion_version:
            raise ValueError(
                f"Checkpoint version {saved_version} does not match "
                f"fusion_version={self.fusion_version}"
            )
        if saved_version == "v1":
            self.fc.weight.data.copy_(weights["fc.weight"])
            self.fc.bias.data.copy_(weights["fc.bias"])
            return
        self.fusion_head.load_state_dict(weights["fusion_head"], strict=True)
