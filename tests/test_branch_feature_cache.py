import gc
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "detectors"
    / "sd_v1_4"
    / "feature_cache.py"
)


class FakeFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = torch.from_numpy(
            self.features[index].copy()
        )
        return feature, int(self.labels[index])


datasets_module = types.ModuleType("dataSets")
datasets_module.BranchFeatureDataset = FakeFeatureDataset
datasets_module.FusionFeatureDataset = FakeFeatureDataset
sys.modules["dataSets"] = datasets_module
SPEC = importlib.util.spec_from_file_location("branch_feature_cache", MODULE_PATH)
FEATURE_CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FEATURE_CACHE)
build_branch_feature_cache = FEATURE_CACHE.build_branch_feature_cache


class TinyFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, count=6):
        self.total_list = [f"data/train/sd-v1_4/sample_{i}.png" for i in range(count)]
        self.labels_dict = {
            path: index % 2 for index, path in enumerate(self.total_list)
        }
        self.features = torch.arange(
            count * 4,
            dtype=torch.float32,
        ).reshape(count, 4)

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, index):
        return self.features[index], int(self.labels_dict[self.total_list[index]])


class TinyBranchModel(torch.nn.Module):
    def __init__(self, branch):
        super().__init__()
        self.fc = torch.nn.Linear(4, 1)
        if branch == "semantic":
            self.clip = SimpleNamespace(encode_image=lambda inputs: inputs)
        else:
            self.artifact_encoder = torch.nn.Identity()


class BranchFeatureCacheTests(unittest.TestCase):
    def test_cache_is_written_and_reused(self):
        dataset = TinyFeatureDataset()
        for branch in ("artifact", "semantic"):
            with self.subTest(branch=branch):
                model = TinyBranchModel(branch)
                with tempfile.TemporaryDirectory() as temp:
                    cached = build_branch_feature_cache(
                        model,
                        dataset,
                        temp,
                        "train",
                        branch,
                        "cpu",
                        batch_size=2,
                        num_workers=0,
                    )
                    self.assertEqual(len(cached), len(dataset))
                    features, label = cached[0]
                    self.assertTrue(
                        torch.equal(features, dataset.features[0].half())
                    )
                    self.assertEqual(label, dataset.labels_dict[dataset.total_list[0]])

                    reused = build_branch_feature_cache(
                        model,
                        dataset,
                        temp,
                        "train",
                        branch,
                        "cpu",
                        batch_size=2,
                        num_workers=0,
                    )
                    self.assertEqual(len(reused), len(dataset))
                    del features, label, cached, reused
                    gc.collect()


if __name__ == "__main__":
    unittest.main()
