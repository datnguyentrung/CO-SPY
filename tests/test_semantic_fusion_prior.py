import importlib
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]


class StubBaseDetector(torch.nn.Module):
    pass


def load_fusion_module():
    detectors_package = types.ModuleType("detectors")
    detectors_package.__path__ = [str(ROOT / "detectors")]
    sd_package = types.ModuleType("detectors.sd_v1_4")
    sd_package.__path__ = [str(ROOT / "detectors" / "sd_v1_4")]

    artifact_module = types.ModuleType("detectors.sd_v1_4.artifact")
    artifact_module.AlignedCrop = object
    artifact_module.ArtifactDetector = object
    base_module = types.ModuleType("detectors.sd_v1_4.base")
    base_module.BaseDetector = StubBaseDetector
    npr_module = types.ModuleType("detectors.sd_v1_4.npr")
    npr_module.NPRDetector = object
    semantic_module = types.ModuleType("detectors.sd_v1_4.semantic")
    semantic_module.SemanticDetector = object

    stubs = {
        "detectors": detectors_package,
        "detectors.sd_v1_4": sd_package,
        "detectors.sd_v1_4.artifact": artifact_module,
        "detectors.sd_v1_4.base": base_module,
        "detectors.sd_v1_4.npr": npr_module,
        "detectors.sd_v1_4.semantic": semantic_module,
        "detectors.sd_v1_4.fusion": sys.modules.get(
            "detectors.sd_v1_4.fusion"
        ),
        "detectors.sd_v1_4.fusion_head": sys.modules.get(
            "detectors.sd_v1_4.fusion_head"
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    sys.modules.pop("detectors.sd_v1_4.fusion", None)
    sys.modules.pop("detectors.sd_v1_4.fusion_head", None)
    try:
        return importlib.import_module("detectors.sd_v1_4.fusion")
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


class SemanticFusionPriorTests(unittest.TestCase):
    def test_adaptive_fusion_adds_trained_semantic_probe(self):
        fusion_module = load_fusion_module()
        detector = object.__new__(fusion_module.CoSpyFusionDetector)
        torch.nn.Module.__init__(detector)

        detector.sem = torch.nn.Module()
        detector.sem.fc = torch.nn.Linear(4, 1)
        with torch.no_grad():
            detector.sem.fc.weight.copy_(
                torch.tensor([[1.0, 2.0, 0.0, 0.0]])
            )
            detector.sem.fc.bias.fill_(0.25)
        detector.sem.fc.requires_grad_(False)
        detector.art = torch.nn.Module()
        detector.npr = None
        detector.fusion_head = fusion_module.AdaptiveFusionHead(
            [4, 3], branch_dropout=0.0
        )
        with torch.no_grad():
            detector.fusion_head.classifier.weight.zero_()
            detector.fusion_head.classifier.bias.zero_()

        sem_feature = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 0.0, 2.0]]
        )
        art_feature = torch.tensor(
            [
                [[0.1, 0.2, 0.3]],
                [[0.4, 0.5, 0.6]],
            ]
        )
        semantic_logit = detector.sem.fc(sem_feature)

        output = detector.forward_features(sem_feature, art_feature)

        # The adaptive head is zeroed above, so the output is the trained
        # semantic logit multiplied by its initial regulator gate of 0.5.
        self.assertTrue(
            torch.allclose(output, 0.5 * semantic_logit, atol=1e-6)
        )

        with torch.no_grad():
            detector.sem.fc.weight[0, 0] = -3.0
        changed = detector.forward_features(sem_feature, art_feature)
        self.assertFalse(torch.allclose(output, changed))

    def test_semantic_prior_obeys_branch_dropout(self):
        fusion_module = load_fusion_module()
        detector = object.__new__(fusion_module.CoSpyFusionDetector)
        torch.nn.Module.__init__(detector)

        detector.sem = torch.nn.Module()
        detector.sem.fc = torch.nn.Linear(4, 1)
        with torch.no_grad():
            detector.sem.fc.weight.fill_(1.0)
            detector.sem.fc.bias.fill_(1.0)
        detector.npr = None
        detector.fusion_head = fusion_module.AdaptiveFusionHead(
            [4, 3], branch_dropout=0.5
        )
        with torch.no_grad():
            detector.fusion_head.classifier.weight.zero_()
            detector.fusion_head.classifier.bias.zero_()
        detector.fusion_head.train()

        sem_feature = torch.ones(2, 4)
        art_feature = torch.ones(2, 1, 3)
        drop_every_sample = torch.zeros(2)
        drop_semantic_stream = torch.zeros(2, dtype=torch.long)
        with mock.patch.object(
            torch, "rand", return_value=drop_every_sample
        ), mock.patch.object(
            torch, "randint", return_value=drop_semantic_stream
        ):
            output = detector.forward_features(sem_feature, art_feature)

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))


if __name__ == "__main__":
    unittest.main()
