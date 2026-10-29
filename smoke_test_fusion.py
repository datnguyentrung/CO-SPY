"""CPU smoke test for the adaptive fusion head."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


def load_fusion_head():
    path = Path(__file__).parent / "detectors" / "sd_v1_4" / "fusion_head.py"
    spec = importlib.util.spec_from_file_location("fusion_head", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdaptiveFusionHead


def main():
    AdaptiveFusionHead = load_fusion_head()
    head = AdaptiveFusionHead([8, 4], branch_dropout=0.5)
    sem = torch.randn(64, 8)
    art = torch.randn(64, 4)

    head.train()
    train_logits = head(sem, art)
    assert train_logits.shape == (64, 1)
    assert torch.isfinite(train_logits).all()

    regulated = [head.norms[0](sem), head.norms[1](art)]
    kept = head._apply_branch_dropout(regulated)
    assert not torch.any((kept[0].abs().sum(1) == 0) & (kept[1].abs().sum(1) == 0))

    head.eval()
    first = head(sem, art)
    second = head(sem, art)
    assert torch.equal(first, second)

    state = {key: value.clone() for key, value in head.state_dict().items()}
    restored = AdaptiveFusionHead([8, 4], branch_dropout=0.5)
    restored.load_state_dict(state)
    restored.eval()
    assert torch.allclose(first, restored(sem, art))
    print("adaptive fusion smoke test: OK")


if __name__ == "__main__":
    main()
