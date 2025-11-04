"""
Unit tests for the Stage 6 PPO trainer.

默认情况下使用轻量级 stub 验证训练循环；当设置
`EPD_INTEGRATION_TEST=1` 且具备 GPU 时，可运行真实的
Stable Diffusion + HPS 集成测试。

python -m training.ppo.tests.test_ppo_trainer
EPD_INTEGRATION_TEST=1 python -m training.ppo.tests.test_ppo_trainer
"""

import math
import os
from pathlib import Path
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from training.ppo import rl_runner
from training.ppo.cold_start import build_dirichlet_params, load_predictor_table
from training.ppo.policy import EPDParamPolicy, PolicyOutput
from training.ppo.ppo_trainer import PPOTrainer, PPOTrainerConfig
from training.ppo.reward_hps import RewardHPS, RewardHPSConfig
from training.ppo.rl_runner import RolloutBatch


class _StubReward:
    def __init__(self, device: torch.device) -> None:
        self.config = SimpleNamespace(device=device)

    def score_tensor(self, images, prompts, return_metadata: bool = False):
        scores = torch.linspace(0.1, 0.5, images.shape[0], device=images.device)
        if return_metadata:
            return scores, {
                "num_images": int(images.shape[0]),
                "duration": 0.0,
                "device": str(images.device),
                "raw_scores": {"hps": scores.detach().cpu()},
            }
        return scores


class _StubRunner:
    def __init__(self, policy: EPDParamPolicy, num_steps: int, num_points: int, device: torch.device, rloo_k: int) -> None:
        self.policy = policy
        self.device = device
        self.config = SimpleNamespace(
            num_steps=num_steps,
            num_points=num_points,
            rloo_k=rloo_k,
        )
        self.net = SimpleNamespace()

    def rollout(self, batch_size: int) -> RolloutBatch:
        intervals = self.config.num_steps - 1
        step_indices = torch.arange(intervals, device=self.device)
        step_idx_expanded = step_indices.unsqueeze(0).repeat(batch_size, 1).reshape(-1)
        base_output = self.policy(step_idx_expanded)

        shaped_output = PolicyOutput(
            alpha_pos=base_output.alpha_pos.view(batch_size, intervals, -1),
            alpha_weight=base_output.alpha_weight.view(batch_size, intervals, -1),
            log_alpha_pos=base_output.log_alpha_pos.view(batch_size, intervals, -1),
            log_alpha_weight=base_output.log_alpha_weight.view(batch_size, intervals, -1),
        )

        flat_output = PolicyOutput(
            alpha_pos=shaped_output.alpha_pos.reshape(-1, shaped_output.alpha_pos.shape[-1]),
            alpha_weight=shaped_output.alpha_weight.reshape(-1, shaped_output.alpha_weight.shape[-1]),
            log_alpha_pos=shaped_output.log_alpha_pos.reshape(-1, shaped_output.log_alpha_pos.shape[-1]),
            log_alpha_weight=shaped_output.log_alpha_weight.reshape(-1, shaped_output.log_alpha_weight.shape[-1]),
        )

        sample = self.policy.sample_table(flat_output)
        sample.positions = sample.positions.view(batch_size, intervals, -1)
        sample.weights = sample.weights.view(batch_size, intervals, -1)
        sample.segments = sample.segments.view(batch_size, intervals, -1)
        sample.log_prob = sample.log_prob.view(batch_size, intervals).sum(dim=-1)
        sample.entropy_pos = sample.entropy_pos.view(batch_size, intervals).sum(dim=-1)
        sample.entropy_weight = sample.entropy_weight.view(batch_size, intervals).sum(dim=-1)

        images = torch.zeros(batch_size, 3, 64, 64, device=self.device)
        latents = torch.zeros_like(images)
        prompts = [f"prompt_{i // self.config.rloo_k}" for i in range(batch_size)]
        seeds = list(range(batch_size))

        return RolloutBatch(
            images=images,
            prompts=prompts,
            seeds=seeds,
            policy_output=shaped_output,
            policy_sample=sample,
            log_prob=sample.log_prob,
            entropy_pos=sample.entropy_pos,
            entropy_weight=sample.entropy_weight,
            step_indices=step_indices,
            latents=latents,
            metadata={"status": "ok"},
        )


class PPOTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.device = torch.device("cpu")
        self.policy = EPDParamPolicy(num_steps=4, num_points=2, hidden_dim=16, num_layers=1)

    def test_train_step_returns_metrics(self) -> None:
        runner = _StubRunner(self.policy, num_steps=4, num_points=2, device=self.device, rloo_k=1)
        reward = _StubReward(device=self.device)
        config = PPOTrainerConfig(
            device=self.device,
            rollout_batch_size=4,
            rloo_k=1,
            minibatch_size=2,
            ppo_epochs=2,
            decode_rgb=False,
        )
        trainer = PPOTrainer(self.policy, runner, reward, config)
        metrics = trainer.train_step()

        expected_keys = {
            "loss",
            "policy_loss",
            "kl",
            "ratio",
            "grad_norm",
            "mixed_reward_mean",
            "mixed_reward_std",
            "hps_mean",
            "aesthetic_mean",
            "clip_mean",
            "imagereward_mean",
            "pickscore_mean",
            "step",
        }
        self.assertTrue(expected_keys.issubset(metrics.keys()))

    def test_rloo_advantages_shape(self) -> None:
        runner = _StubRunner(self.policy, num_steps=4, num_points=2, device=self.device, rloo_k=2)
        reward = _StubReward(device=self.device)
        config = PPOTrainerConfig(
            device=self.device,
            rollout_batch_size=6,
            rloo_k=2,
            minibatch_size=3,
            ppo_epochs=1,
            decode_rgb=False,
        )
        trainer = PPOTrainer(self.policy, runner, reward, config)

        rewards = torch.linspace(0.0, 1.0, config.rollout_batch_size)
        advantages = trainer._compute_advantages(rewards)
        self.assertEqual(advantages.shape[0], rewards.shape[0])
        self.assertTrue(torch.isfinite(advantages).all())


SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[3]
    / "exps"
    / "00036-ms_coco-10-36-epd-dpm-1-discrete"
    / "network-snapshot-000005.pkl"
)
HPS_WEIGHTS_PATH = Path(__file__).resolve().parents[3] / "weights" / "HPS_v2.1_compressed.pt"
EPD_INTEGRATION_ENABLED = os.environ.get("EPD_INTEGRATION_TEST") == "1"


class PPOTrainerIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(EPD_INTEGRATION_ENABLED, "Integration test disabled")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    @unittest.skipUnless(SNAPSHOT_PATH.exists(), "Predictor snapshot not available")
    @unittest.skipUnless(HPS_WEIGHTS_PATH.exists(), "HPS weights not available")
    def test_full_rl_step(self) -> None:
        from sample import create_model  # 延迟导入，避免非集成场景加重依赖

        device = torch.device("cuda")
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        table = load_predictor_table(SNAPSHOT_PATH, map_location="cpu")
        init = build_dirichlet_params(table, concentration=200.0)

        policy = EPDParamPolicy(
            num_steps=table.num_steps,
            num_points=table.num_points,
            hidden_dim=128,
            num_layers=2,
            dirichlet_init=init,
        ).to(device)

        net, model_source = create_model(
            dataset_name=table.metadata.get("dataset_name", "ms_coco"),
            guidance_type=table.metadata.get("guidance_type", "cfg"),
            guidance_rate=float(table.metadata.get("guidance_rate", 7.5)),
            device=device,
        )
        net = net.to(device)
        net.eval()

        runner_config = rl_runner.RLRunnerConfig(
            policy=policy,
            net=net,
            num_steps=table.num_steps,
            num_points=table.num_points,
            device=device,
            guidance_type=table.metadata.get("guidance_type", "cfg"),
            guidance_rate=float(table.metadata.get("guidance_rate", 7.5)),
            schedule_type=table.metadata.get("schedule_type", "discrete"),
            schedule_rho=float(table.metadata.get("schedule_rho", 1.0)),
            dataset_name=table.metadata.get("dataset_name", "ms_coco"),
            precision=torch.float32,
            prompt_csv=None,
            rloo_k=2,
            rng_seed=123,
            verbose=False,
            model_source=model_source,
        )
        runner = rl_runner.EPDRolloutRunner(runner_config)

        reward = RewardHPS(
            RewardHPSConfig(
                device=device,
                batch_size=2,
                enable_amp=True,
                weights_path=HPS_WEIGHTS_PATH,
                cache_dir=HPS_WEIGHTS_PATH.parent,
            )
        )

        trainer_config = PPOTrainerConfig(
            device=device,
            rollout_batch_size=4,
            rloo_k=2,
            ppo_epochs=1,
            minibatch_size=2,
            learning_rate=5e-5,
            decode_rgb=True,
            clip_range=0.2,
            kl_coef=0.01,
            entropy_coef=0.0,
            max_grad_norm=1.0,
        )
        trainer = PPOTrainer(policy, runner, reward, trainer_config)

        preview_batch = runner.rollout(trainer_config.rollout_batch_size)
        with torch.no_grad():
            decoded = preview_batch.images
            decoder = getattr(getattr(runner.net, "model", None), "decode_first_stage", None)
            if callable(decoder):
                decoded = decoder(decoded)
            decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0)
        scores, meta = reward.score_tensor(decoded, preview_batch.prompts, return_metadata=True)

        output_dir = Path(__file__).resolve().parent / "integration_outputs"
        output_dir.mkdir(exist_ok=True)
        for idx in range(decoded.shape[0]):
            img = decoded[idx].detach().cpu()
            img_np = (img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_np).save(output_dir / f"ppo_trainer_integration_sample_{idx}.png")

        for idx, (prompt, score) in enumerate(zip(preview_batch.prompts, scores.detach().cpu().tolist())):
            print(f"[PPOIntegration] prompt[{idx}]: {prompt}")
            print(f"[PPOIntegration] score[{idx}]: {score:.4f}")
        print(f"[PPOIntegration] metadata: {meta}")
        print(f"[PPOIntegration] saved images to {output_dir}")

        metrics = trainer.train_step()
        required = [
            "loss",
            "policy_loss",
            "kl",
            "mixed_reward_mean",
            "mixed_reward_std",
            "ratio",
            "grad_norm",
        ]
        for key in required:
            self.assertIn(key, metrics)
            self.assertTrue(math.isfinite(metrics[key]))

        del runner, reward, trainer, net
        torch.cuda.empty_cache()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
