from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


APP_ROOT = Path(__file__).resolve().parent / "co-spy"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.inference.runtime import Prediction, _require_fusion_v3_state


def _valid_v3_state() -> dict[str, torch.Tensor]:
    return {
        "sem.fc.weight": torch.zeros(1, 1152),
        "sem.fc.bias": torch.zeros(1),
        "art.fc.weight": torch.zeros(1, 512),
        "art.fc.bias": torch.zeros(1),
        "npr.fc.weight": torch.zeros(1, 512),
        "npr.fc.bias": torch.zeros(1),
        "fusion_head.classifier.weight": torch.zeros(1, 2177),
        "fusion_head.classifier.bias": torch.zeros(1),
    }


def test_v3_checkpoint_dimensions_include_semantic_logit() -> None:
    _require_fusion_v3_state(_valid_v3_state())


def test_v3_checkpoint_rejects_the_legacy_two_branch_head() -> None:
    state = _valid_v3_state()
    state["fusion_head.classifier.weight"] = torch.zeros(1, 2176)

    with pytest.raises(ValueError, match="branches provide 2177"):
        _require_fusion_v3_state(state)


def test_v3_checkpoint_requires_npr_weights() -> None:
    state = _valid_v3_state()
    del state["npr.fc.weight"]

    with pytest.raises(ValueError, match="npr.fc.weight"):
        _require_fusion_v3_state(state)


def test_prediction_can_return_real() -> None:
    result = Prediction("Fusion v3", 0.2, 10, -1.3863).as_dict()

    assert result["prediction"] == "REAL"
    assert result["real_probability"] == pytest.approx(0.8)
