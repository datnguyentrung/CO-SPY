import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, split, **_kwargs):
        count = 8 if split == "train" else 4
        self.total_list = [
            f"data/train/progan/{split}/image_{index}.png"
            for index in range(count)
        ]
        self.labels_dict = {
            path: index % 2 for index, path in enumerate(self.total_list)
        }
        self.samples = [
            (
                torch.tensor(
                    [
                        float((index + 1) % 5) / 5.0,
                        float((index + 2) % 7) / 7.0,
                        float((index + 3) % 3) / 3.0,
                        float((index + 4) % 4) / 4.0,
                    ]
                ),
                float(index % 2),
            )
            for index in range(count)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FusionDataset(TinyDataset):
    pass


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 1)
        self.train_transform = None
        self.val_transform = None

    def forward(self, inputs, return_feat=False):
        outputs = self.fc(inputs)
        if return_feat:
            return inputs, outputs
        return outputs

    def predict(self, inputs):
        inputs = inputs.to(next(self.parameters()).device)
        return self.forward(inputs).sigmoid().flatten().tolist()


class FakeLoguru:
    def add(self, *_args, **_kwargs):
        return 1

    def remove(self, *_args, **_kwargs):
        return None

    def complete(self):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def exception(self, *_args, **_kwargs):
        return None


def fake_evaluate(y_pred, y_true):
    predictions = np.asarray(y_pred) > 0.5
    labels = np.asarray(y_true)
    accuracy = float((predictions == labels).mean())
    return accuracy, accuracy


class TrainerRecoveryIntegrationTests(unittest.TestCase):
    def test_trainer_resume_matches_uninterrupted_run(self):
        detector_module = types.ModuleType("detectors.progan")
        detector_module.ArtifactDetector = TinyDetector
        detector_module.SemanticDetector = TinyDetector
        detector_module.End2EndDetector = TinyDetector
        detector_module.CoSpyFusionDetector = TinyDetector

        loguru_module = types.ModuleType("loguru")
        loguru_module.logger = FakeLoguru()
        datasets_module = types.ModuleType("dataSets")
        datasets_module.TrainDataset = TinyDataset
        datasets_module.FusionTrainDataset = FusionDataset
        utils_module = types.ModuleType("utils")
        utils_module.evaluate = fake_evaluate

        with mock.patch.dict(
            sys.modules,
            {
                "loguru": loguru_module,
                "dataSets": datasets_module,
                "utils": utils_module,
            },
        ):
            sys.modules.pop("train", None)
            train_module = importlib.import_module("train")
            with mock.patch.object(  # noqa: SIM117 - separate setup context
                train_module.importlib,
                "import_module",
                return_value=detector_module,
            ):
                with tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)

                    def build_trainer(checkpoint_root):
                        torch.manual_seed(1024)
                        return train_module.Trainer(
                            mode="branch",
                            device="cpu",
                            branch="artifact",
                            train_dataset="progan",
                            ckpt=str(checkpoint_root),
                            epochs=2,
                            batch_size=2,
                            seed=1024,
                            checkpoint_interval_minutes=0,
                            session_timeout_minutes=0,
                            pre_timeout_minutes=0,
                            num_workers=0,
                        )

                    reference = build_trainer(root / "reference")
                    reference.train()

                    resumed_root = root / "resume"
                    interrupted = build_trainer(resumed_root)
                    internal_calls = {"count": 0}
                    original_should_stop = (
                        train_module.GracefulStopController.should_stop
                    )

                    def stop_after_two_batches(controller):
                        internal_calls["count"] += 1
                        if internal_calls["count"] >= 4:
                            controller.request_stop("integration test interruption")
                            return True
                        return original_should_stop(controller)

                    with mock.patch.object(
                        train_module.GracefulStopController,
                        "should_stop",
                        stop_after_two_batches,
                    ):
                        interrupted.train()

                    model_dir = resumed_root / "progan" / "artifact"
                    self.assertTrue((model_dir / "checkpoint_recovery.pth").is_file())
                    self.assertFalse((model_dir / "checkpoint_latest.pth").is_file())

                    resumed = build_trainer(resumed_root)
                    resumed.train()

                    self.assertTrue((model_dir / "checkpoint_latest.pth").is_file())
                    self.assertTrue((model_dir / "checkpoint_best.pth").is_file())
                    self.assertFalse(list(model_dir.glob("epoch_*.pth")))
                    for expected, actual in zip(
                        reference.model.state_dict().values(),
                        resumed.model.state_dict().values(),
                    ):
                        self.assertTrue(torch.equal(expected, actual))

    def test_branch_early_stopping_uses_ap(self):
        detector_module = types.ModuleType("detectors.progan")
        detector_module.ArtifactDetector = TinyDetector
        detector_module.SemanticDetector = TinyDetector
        detector_module.End2EndDetector = TinyDetector
        detector_module.CoSpyFusionDetector = TinyDetector

        loguru_module = types.ModuleType("loguru")
        loguru_module.logger = FakeLoguru()
        datasets_module = types.ModuleType("dataSets")
        datasets_module.TrainDataset = TinyDataset
        datasets_module.FusionTrainDataset = FusionDataset
        utils_module = types.ModuleType("utils")
        utils_module.evaluate = fake_evaluate

        with mock.patch.dict(
            sys.modules,
            {
                "loguru": loguru_module,
                "dataSets": datasets_module,
                "utils": utils_module,
            },
        ):
            sys.modules.pop("train", None)
            train_module = importlib.import_module("train")
            with (
                mock.patch.object(
                    train_module.importlib,
                    "import_module",
                    return_value=detector_module,
                ),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                torch.manual_seed(1024)
                trainer = train_module.Trainer(
                    mode="branch",
                    device="cpu",
                    branch="artifact",
                    train_dataset="progan",
                    ckpt=str(root),
                    epochs=4,
                    batch_size=2,
                    seed=1024,
                    checkpoint_interval_minutes=0,
                    session_timeout_minutes=0,
                    pre_timeout_minutes=0,
                    num_workers=0,
                    early_stopping_patience=1,
                )
                with mock.patch.object(
                    train_module,
                    "evaluate",
                    return_value=(0.8, 0.9),
                ):
                    trainer.train()

                state_path = (
                    root
                    / "progan"
                    / "artifact"
                    / "training_state.json"
                )
                state = json.loads(state_path.read_text(encoding="utf-8"))
                self.assertEqual(state["training_status"], "COMPLETED")
                self.assertEqual(state["current_epoch"], 2)
                self.assertTrue(state["early_stopped"])
                self.assertEqual(state["best_metric"]["name"], "ap")


if __name__ == "__main__":
    unittest.main()
