"""Feature-level adaptive fusion head used by CO-SPY v2/v3.

This module intentionally depends only on PyTorch so the fusion head can be
tested independently from the large CLIP/VAE feature extractors.
"""

from typing import Dict, Sequence

import torch
from torch import nn


class AdaptiveFusionHead(nn.Module):
    """Regulate, concatenate, and classify frozen feature streams.

    Each input stream is independently normalized and multiplied by a learned
    scalar regulator. During training, a whole stream can be dropped so the
    classifier cannot rely on only one shortcut.
    """

    def __init__(
        self,
        feature_dims: Sequence[int],
        num_classes: int = 1,
        dropout: float = 0.2,
        branch_dropout: float = 0.15,
    ):
        super().__init__()
        if len(feature_dims) < 2:
            raise ValueError("Adaptive fusion needs at least two feature streams")
        if not 0.0 <= branch_dropout < 1.0:
            raise ValueError("branch_dropout must be in [0, 1)")

        self.feature_dims = tuple(int(dim) for dim in feature_dims)
        self.branch_dropout = float(branch_dropout)
        self.norms = nn.ModuleList(
            [nn.LayerNorm(dim) for dim in self.feature_dims]
        )
        self.regulators = nn.ModuleList(
            [nn.Linear(dim, 1) for dim in self.feature_dims]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(sum(self.feature_dims), num_classes)

        # Start with both streams enabled and comparable importance.
        for regulator in self.regulators:
            nn.init.zeros_(regulator.weight)
            nn.init.zeros_(regulator.bias)

    def _apply_branch_dropout(
        self, feats: Sequence[torch.Tensor]
    ) -> Sequence[torch.Tensor]:
        if not self.training or self.branch_dropout <= 0.0:
            return feats

        batch_size = feats[0].shape[0]
        drop = torch.rand(batch_size, device=feats[0].device) < self.branch_dropout
        if not bool(drop.any()):
            return feats

        stream_ids = torch.randint(
            len(feats), (batch_size,), device=feats[0].device
        )
        kept = []
        for stream_id, feat in enumerate(feats):
            drop_stream = torch.logical_and(drop, stream_ids == stream_id)
            keep_mask = torch.logical_not(drop_stream).to(feat.dtype)
            kept.append(feat * keep_mask.unsqueeze(1))
        return kept

    def forward_streams(
        self, features: Sequence[torch.Tensor], return_info: bool = False
    ):
        if len(features) != len(self.feature_dims):
            raise ValueError(
                f"Expected {len(self.feature_dims)} feature streams, got {len(features)}"
            )

        normalized = []
        regulated = []
        coefficients = []
        for feat, norm, regulator in zip(
            features, self.norms, self.regulators
        ):
            if feat.ndim != 2:
                raise ValueError(
                    f"Expected a 2-D feature tensor, got shape {tuple(feat.shape)}"
                )
            feat = feat.to(dtype=norm.weight.dtype)
            normed = norm(feat)
            coefficient = torch.sigmoid(regulator(normed))
            normalized.append(normed)
            regulated.append(normed * coefficient)
            coefficients.append(coefficient)

        regulated = self._apply_branch_dropout(regulated)
        fused = torch.cat(regulated, dim=1)
        logits = self.classifier(self.dropout(fused))

        if return_info:
            info: Dict[str, torch.Tensor] = {
                "features": fused,
                "coefficients": torch.cat(coefficients, dim=1),
            }
            return logits, info
        return logits

    def forward(self, *features: torch.Tensor):
        return self.forward_streams(features)
