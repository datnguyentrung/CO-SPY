import random
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from checkpoint_manager import (
    BEST_NAME,
    LATEST_NAME,
    RECOVERY_NAME,
    CheckpointCompatibilityError,
    CheckpointManager,
    GracefulStopController,
    detect_storage_layout,
    find_best_weights,
    is_metric_improved,
    load_torch_checkpoint,
)


def build_manager(
    root: Path,
    *,
    resume_mode="auto",
    optimizer_lr=0.1,
    resume_path=None,
    recovery_retention=1,
):
    return CheckpointManager(
        checkpoint_dir=root,
        checkpoint_root=root,
        experiment_id="unit-test",
        architecture_id="tiny-linear:v1",
        model_config={"width": 4},
        dataset_id="sha256:test-dataset",
        training_config={
            "epochs": 4,
            "batch_size": 2,
            "steps_per_epoch": 3,
            "seed": 7,
        },
        optimizer_config={
            "class": "AdamW",
            "param_groups": [{"lr": optimizer_lr, "weight_decay": 0.0}],
        },
        scheduler_config={"class": "CosineAnnealingLR", "T_max": 12},
        total_epochs=4,
        resume_mode=resume_mode,
        resume_path=resume_path,
        checkpoint_interval_seconds=0,
        session_timeout_seconds=0,
        pre_timeout_seconds=0,
        recovery_retention=recovery_retention,
    )


def build_runtime(seed=11):
    torch.manual_seed(seed)
    model = torch.nn.Linear(4, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
    return model, optimizer, scheduler


class CheckpointManagerTests(unittest.TestCase):
    def test_full_state_round_trip_preserves_training_position(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = build_manager(root)
            model, optimizer, scheduler = build_runtime()
            inputs = torch.tensor(
                [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 1.5, -2.0]]
            )
            labels = torch.tensor([[1.0], [0.0]])
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(inputs), labels
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            random.seed(123)
            np.random.seed(456)
            torch.manual_seed(789)
            expected_weight = model.weight.detach().clone()

            checkpoint = manager.save_full(
                kind=LATEST_NAME,
                models={"model": model},
                optimizers={"optimizer": optimizer},
                schedulers={"scheduler": scheduler},
                resume_epoch=2,
                batch_position=3,
                global_step=9,
                best_metric={"name": "accuracy", "value": 0.81, "mode": "max"},
                epochs_without_improvement=1,
                last_completed_epoch=1,
                additional_state={
                    "data_loader": {
                        "epoch": 2,
                        "batch_position": 3,
                        "sampler_indices": [2, 0, 1],
                    }
                },
            )
            saved_payload = load_torch_checkpoint(checkpoint)

            restored_model, restored_optimizer, restored_scheduler = build_runtime(
                seed=999
            )
            restore_manager = build_manager(root)
            result = restore_manager.find_and_restore(
                models={"model": restored_model},
                optimizers={"optimizer": restored_optimizer},
                schedulers={"scheduler": restored_scheduler},
            )

            self.assertTrue(result.resumed)
            self.assertEqual(result.start_epoch, 2)
            self.assertEqual(result.batch_position, 3)
            self.assertTrue(torch.equal(restored_model.weight, expected_weight))
            self.assertEqual(restored_optimizer.state_dict()["param_groups"][0]["lr"], optimizer.param_groups[0]["lr"])
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
            self.assertTrue(
                torch.equal(
                    torch.get_rng_state(),
                    saved_payload["rng_state"]["torch_cpu"],
                )
            )
            self.assertEqual(
                restore_manager.state["global_step"],
                9,
            )

    def test_corrupted_latest_falls_back_to_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = build_manager(root)
            model, optimizer, scheduler = build_runtime()
            manager.save_full(
                kind=RECOVERY_NAME,
                models={"model": model},
                optimizers={"optimizer": optimizer},
                schedulers={"scheduler": scheduler},
                resume_epoch=1,
                batch_position=0,
                global_step=4,
                best_metric=None,
            )
            optimizer.param_groups[0]["lr"] = 0.03
            manager.save_full(
                kind=LATEST_NAME,
                models={"model": model},
                optimizers={"optimizer": optimizer},
                schedulers={"scheduler": scheduler},
                resume_epoch=2,
                batch_position=2,
                global_step=12,
                best_metric={"name": "accuracy", "value": 0.9, "mode": "max"},
                last_completed_epoch=1,
            )
            (root / LATEST_NAME).write_bytes(b"not a torch checkpoint")

            restored_model, restored_optimizer, restored_scheduler = build_runtime()
            result = build_manager(root).find_and_restore(
                models={"model": restored_model},
                optimizers={"optimizer": restored_optimizer},
                schedulers={"scheduler": restored_scheduler},
            )

            self.assertTrue(result.resumed)
            self.assertEqual(result.checkpoint_path.name, RECOVERY_NAME)
            self.assertEqual(result.start_epoch, 1)
            self.assertEqual(result.batch_position, 0)
            self.assertTrue(result.rejected)

    def test_incompatible_full_checkpoint_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = build_manager(root, optimizer_lr=0.1)
            model, optimizer, scheduler = build_runtime()
            checkpoint = manager.save_full(
                kind=LATEST_NAME,
                models={"model": model},
                optimizers={"optimizer": optimizer},
                schedulers={"scheduler": scheduler},
                resume_epoch=1,
                batch_position=0,
                global_step=3,
                best_metric=None,
            )

            restore_model, restore_optimizer, restore_scheduler = build_runtime()
            result = build_manager(root, optimizer_lr=0.2).find_and_restore(
                models={"model": restore_model},
                optimizers={"optimizer": restore_optimizer},
                schedulers={"scheduler": restore_scheduler},
            )
            self.assertFalse(result.resumed)
            self.assertTrue(result.rejected)

            with self.assertRaises(CheckpointCompatibilityError):
                build_manager(
                    root,
                    optimizer_lr=0.2,
                    resume_path=checkpoint,
                ).find_and_restore(
                    models={"model": restore_model},
                    optimizers={"optimizer": restore_optimizer},
                    schedulers={"scheduler": restore_scheduler},
                )

            weights_result = build_manager(
                root,
                optimizer_lr=0.2,
                resume_mode="weights",
            ).find_and_restore(
                models={"model": restore_model},
                optimizers={"optimizer": restore_optimizer},
                schedulers={"scheduler": restore_scheduler},
            )
            self.assertTrue(weights_result.resumed)
            self.assertTrue(weights_result.weights_only)

    def test_best_weights_and_managed_retention(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manager = build_manager(root)
            model = torch.nn.Linear(4, 1)
            manager.save_best(
                model,
                epoch=3,
                global_step=14,
                metric={"name": "accuracy", "value": 0.95, "mode": "max"},
            )
            self.assertEqual(find_best_weights(root).name, BEST_NAME)
            for step in range(3):
                manager.save_full(
                    kind=RECOVERY_NAME,
                    models={"model": model},
                    optimizers={"optimizer": torch.optim.AdamW(model.parameters())},
                    schedulers=None,
                    resume_epoch=step,
                    batch_position=0,
                    global_step=step,
                    best_metric=None,
                )
            checkpoints = sorted(path.name for path in root.glob("checkpoint_*.pth"))
            self.assertEqual(checkpoints, sorted([BEST_NAME, RECOVERY_NAME]))
            self.assertFalse(list(root.glob("*.tmp")))
            report = manager.report_storage(root)
            self.assertIn("CHECKPOINT:", report)
            self.assertIn("TOP 20 LARGEST FILES:", report)

            rotating_manager = build_manager(root, recovery_retention=3)
            for step in range(3):
                rotating_manager.save_full(
                    kind=RECOVERY_NAME,
                    models={"model": model},
                    optimizers={"optimizer": torch.optim.AdamW(model.parameters())},
                    schedulers=None,
                    resume_epoch=step,
                    batch_position=step,
                    global_step=20 + step,
                    best_metric=None,
                )
            self.assertTrue((root / "checkpoint_recovery.1.pth").is_file())
            self.assertTrue((root / "checkpoint_recovery.2.pth").is_file())

    def test_mid_epoch_resume_matches_uninterrupted_optimizer_updates(self):
        batches = [
            (
                torch.tensor([[float(index), 1.0, -1.0, 0.5]]),
                torch.tensor([[float(index % 2)]]),
            )
            for index in range(1, 9)
        ]

        def run_steps(model, optimizer, scheduler, start, stop):
            for index in range(start, stop):
                inputs, labels = batches[index]
                optimizer.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    model(inputs), labels
                )
                loss.backward()
                optimizer.step()
                scheduler.step()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            torch.manual_seed(31)
            reference_model = torch.nn.Linear(4, 1)
            reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=0.1)
            reference_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                reference_optimizer, T_max=8
            )
            run_steps(
                reference_model,
                reference_optimizer,
                reference_scheduler,
                0,
                len(batches),
            )

            torch.manual_seed(31)
            interrupted_model = torch.nn.Linear(4, 1)
            interrupted_optimizer = torch.optim.AdamW(
                interrupted_model.parameters(), lr=0.1
            )
            interrupted_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                interrupted_optimizer, T_max=8
            )
            run_steps(
                interrupted_model,
                interrupted_optimizer,
                interrupted_scheduler,
                0,
                3,
            )
            manager = build_manager(root)
            manager.save_full(
                kind=RECOVERY_NAME,
                models={"model": interrupted_model},
                optimizers={"optimizer": interrupted_optimizer},
                schedulers={"scheduler": interrupted_scheduler},
                resume_epoch=0,
                batch_position=3,
                global_step=3,
                best_metric=None,
            )

            torch.manual_seed(999)
            resumed_model = torch.nn.Linear(4, 1)
            resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.1)
            resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                resumed_optimizer, T_max=8
            )
            result = build_manager(root).find_and_restore(
                models={"model": resumed_model},
                optimizers={"optimizer": resumed_optimizer},
                schedulers={"scheduler": resumed_scheduler},
            )
            self.assertEqual(result.batch_position, 3)
            run_steps(
                resumed_model,
                resumed_optimizer,
                resumed_scheduler,
                result.batch_position,
                len(batches),
            )
            for expected, actual in zip(
                reference_model.state_dict().values(),
                resumed_model.state_dict().values(),
            ):
                self.assertTrue(torch.equal(expected, actual))

    def test_timeout_requests_safe_stop(self):
        controller = GracefulStopController(
            session_timeout_seconds=0.03,
            pre_timeout_seconds=0.02,
        )
        time.sleep(0.02)
        self.assertTrue(controller.should_stop())
        self.assertIn("timeout", controller.reason)

    def test_storage_layout_supports_configured_and_kaggle_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous = root / "previous-output"
            layout = detect_storage_layout(
                root / "working",
                input_roots=[previous],
                environment="kaggle",
            )
            self.assertEqual(layout.environment, "kaggle")
            self.assertIn(previous.resolve(), layout.discovery_roots)
            self.assertIn(Path("/kaggle/input").resolve(), layout.discovery_roots)

    def test_best_metric_requires_strict_improvement(self):
        self.assertFalse(is_metric_improved(0.8, 0.8))
        self.assertFalse(is_metric_improved(0.8, 0.7))
        self.assertTrue(is_metric_improved(0.8, 0.8001))


if __name__ == "__main__":
    unittest.main()
