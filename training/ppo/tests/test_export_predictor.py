"""
Stage 8 exporter tests.

这些测试使用轻量级的 CPU-only 构造，验证：
    * PPO 策略均值 -> EPD predictor logits 的转换是否正确。
    * 导出的 training_options.json 是否填充了核心字段与 export_info。
    * manifest 记录了 sanitize 统计以及最新 metrics 摘要。
"""

from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from training.networks import EPD_predictor
from training.ppo.cold_start import (
    EPDTable,
    build_dirichlet_params,
    dirichlet_alpha_from_mean,
    positions_to_segments,
)
from training.ppo.export_epd_predictor import ExportError, export_policy_mean_to_predictor
from training.ppo.policy import EPDParamPolicy


def _make_base_table() -> EPDTable:
    positions = np.array(
        [
            [0.25, 0.75],
            [0.3, 0.65],
            [0.15, 0.45],
        ],
        dtype=np.float64,
    )
    weights = np.array(
        [
            [0.4, 0.6],
            [0.55, 0.45],
            [0.33, 0.67],
        ],
        dtype=np.float64,
    )
    return EPDTable(
        positions=positions,
        weights=weights,
        num_steps=positions.shape[0] + 1,
        num_points=positions.shape[1],
        schedule_type="discrete",
        schedule_rho=1.0,
        metadata={
            "dataset_name": "ms_coco",
            "sampler_stu": "epd",
            "guidance_type": "cfg",
            "guidance_rate": 7.5,
            "predict_x0": False,
            "lower_order_final": True,
        },
    )


def _save_predictor_snapshot(path: Path, table: EPDTable) -> None:
    predictor = EPD_predictor(
        num_points=table.num_points,
        num_steps=table.num_steps,
        dataset_name=table.metadata.get("dataset_name"),
        sampler_stu=table.metadata.get("sampler_stu", "epd"),
        sampler_tea="dpm",
        guidance_type=table.metadata.get("guidance_type"),
        guidance_rate=table.metadata.get("guidance_rate"),
        schedule_type=table.schedule_type,
        schedule_rho=table.schedule_rho,
        scale_dir=0.0,
        scale_time=0.0,
        predict_x0=table.metadata.get("predict_x0", False),
        lower_order_final=table.metadata.get("lower_order_final", True),
        M=1,
    )
    with torch.no_grad():
        pos_tensor = torch.from_numpy(table.positions).clamp(1e-6, 1 - 1e-6)
        weight_tensor = torch.from_numpy(table.weights).clamp(1e-6, 1.0)
        predictor.r_params.copy_(torch.logit(pos_tensor))
        predictor.weight_s.copy_(torch.log(weight_tensor))
        predictor.scale_dir_params.zero_()
        predictor.scale_time_params.zero_()
    with path.open("wb") as handle:
        pickle.dump({"model": predictor.cpu()}, handle)


class ExportPredictorTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        (self.root / "configs").mkdir()
        (self.root / "logs").mkdir()
        (self.root / "checkpoints").mkdir()

        self.base_table = _make_base_table()
        self.snapshot_path = self.root / "base_snapshot.pkl"
        _save_predictor_snapshot(self.snapshot_path, self.base_table)

        training_options = {
            "loss_kwargs": {"class_name": "training.loss.EPD_loss"},
            "pred_kwargs": {
                "class_name": "training.networks.EPD_predictor",
                "num_steps": self.base_table.num_steps,
                "num_points": self.base_table.num_points,
                "dataset_name": "ms_coco",
                "guidance_type": "cfg",
                "guidance_rate": 7.5,
                "schedule_type": "discrete",
                "schedule_rho": 1.0,
                "sampler_stu": "epd",
                "sampler_tea": "dpm",
                "M": 1,
                "scale_dir": 0.0,
                "scale_time": 0.0,
                "predict_x0": False,
                "lower_order_final": True,
                "alpha": 10.0,
                "max_order": 2,
                "backend": "ldm",
                "backend_config": {},
            },
        }
        with (self.snapshot_path.parent / "training_options.json").open("w", encoding="utf-8") as handle:
            json.dump(training_options, handle)

        config_dict = {
            "run": {
                "output_root": str(self.root.parent),
                "run_name": "unit_test_run",
                "seed": 0,
                "run_dir": str(self.root),
            },
            "data": {"predictor_snapshot": str(self.snapshot_path)},
            "model": {
                "dataset_name": "ms_coco",
                "guidance_type": "cfg",
                "guidance_rate": 7.5,
                "schedule_type": "discrete",
                "schedule_rho": 1.0,
                "backend": "ldm",
                "backend_options": {},
            },
            "reward": {
                "weights_path": "weights/HPS_v2.1_compressed.pt",
                "batch_size": 2,
                "enable_amp": False,
                "cache_dir": None,
            },
            "ppo": {
                "rollout_batch_size": 2,
                "rloo_k": 1,
                "ppo_epochs": 1,
                "minibatch_size": 2,
                "learning_rate": 1e-4,
                "clip_range": 0.2,
                "kl_coef": 0.01,
                "entropy_coef": 0.0,
                "max_grad_norm": 1.0,
                "decode_rgb": False,
                "steps": 1,
                "dirichlet_concentration": 120.0,
            },
            "logging": {"log_interval": 1, "save_interval": 5},
        }

        import yaml

        resolved_path = self.root / "configs" / "resolved_config.yaml"
        with resolved_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(config_dict, handle)

        metrics_path = self.root / "logs" / "metrics.jsonl"
        with metrics_path.open("w", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"step": 10, "mixed_reward_mean": 1.23, "policy_loss": -0.42}) + "\n"
            )

        # 准备策略 checkpoint（使用与目标表不同的均值以验证导出逻辑）
        self.target_positions = np.array(
            [
                [0.2, 0.8],
                [0.4, 0.7],
                [0.3, 0.6],
            ],
            dtype=np.float64,
        )
        self.target_weights = np.array(
            [
                [0.35, 0.65],
                [0.6, 0.4],
                [0.25, 0.75],
            ],
            dtype=np.float64,
        )
        dirichlet = build_dirichlet_params(self.base_table, concentration=150.0)
        policy = EPDParamPolicy(
            num_steps=self.base_table.num_steps,
            num_points=self.base_table.num_points,
            hidden_dim=64,
            num_layers=1,
            dirichlet_init=dirichlet,
        )

        segments = positions_to_segments(self.target_positions)
        alpha_pos = dirichlet_alpha_from_mean(segments, concentration=300.0)
        alpha_weight = dirichlet_alpha_from_mean(self.target_weights, concentration=300.0)
        with torch.no_grad():
            policy.base_log_alpha_pos.copy_(torch.log(torch.from_numpy(alpha_pos).float()))
            policy.base_log_alpha_weight.copy_(torch.log(torch.from_numpy(alpha_weight).float()))

        checkpoint_path = self.root / "checkpoints" / "policy-step000010.pt"
        torch.save(policy.state_dict(), checkpoint_path)
        self.checkpoint_path = checkpoint_path

    def test_export_end_to_end(self):
        result = export_policy_mean_to_predictor(self.root, checkpoint=self.checkpoint_path)

        self.assertTrue(result.snapshot_path.is_file())
        self.assertTrue(result.training_options_path.is_file())
        self.assertIsNotNone(result.manifest_path)
        self.assertTrue(result.manifest_path.is_file())

        with result.snapshot_path.open("rb") as handle:
            snapshot = pickle.load(handle)
        self.assertIn("model", snapshot)
        predictor: EPD_predictor = snapshot["model"]
        with torch.no_grad():
            exported_positions = torch.sigmoid(predictor.r_params).cpu().numpy()
            exported_weights = torch.softmax(predictor.weight_s, dim=-1).cpu().numpy()

        np.testing.assert_allclose(exported_positions, self.target_positions, atol=1e-4)
        np.testing.assert_allclose(exported_weights, self.target_weights, atol=1e-4)

        with result.training_options_path.open("r", encoding="utf-8") as handle:
            options = json.load(handle)
        self.assertIn("pred_kwargs", options)
        pred_kwargs = options["pred_kwargs"]
        self.assertEqual(pred_kwargs["num_steps"], self.base_table.num_steps)
        self.assertEqual(pred_kwargs["num_points"], self.base_table.num_points)
        self.assertEqual(pred_kwargs["dataset_name"], "ms_coco")
        self.assertEqual(pred_kwargs["backend"], "ldm")
        self.assertEqual(pred_kwargs["backend_config"], {})
        self.assertIn("export_info", options)
        self.assertIn("sanitized_rows", options["export_info"])

        with result.manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["export_step"], 10)
        self.assertIn("latest_metrics", manifest)
        self.assertEqual(manifest["latest_metrics"]["mixed_reward_mean"], 1.23)

    def test_missing_checkpoint_raises(self):
        with self.assertRaises(ExportError):
            export_policy_mean_to_predictor(self.root, checkpoint=self.root / "checkpoints" / "missing.pt")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

'''

python -m training.ppo.export_epd_predictor exps/20251030-211927-sd15_rl_base \
    --checkpoint checkpoints/policy-step000002.pt

'''
