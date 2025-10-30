"""
Unit tests for Stage 4 rollout runner utilities.

Execute manually in a fully configured environment:
    python -m training.ppo.tests.test_rl_runner
    EPD_INTEGRATION_TEST=1 python -m training.ppo.tests.test_rl_runner
"""

import os
from pathlib import Path
import unittest
from unittest import mock

import torch

from training.ppo import rl_runner
from training.ppo.policy import EPDParamPolicy, PolicyOutput, PolicySample
from training.ppo.cold_start import load_predictor_table, build_dirichlet_params


class _StubPolicy(torch.nn.Module):
    """Minimal policy stub returning deterministic outputs."""

    def __init__(self, num_steps, num_points):
        super().__init__()
        self.num_steps = num_steps
        self.num_points = num_points

    def forward(self, step_indices, context=None):
        batch = step_indices.shape[0]
        alpha_pos = torch.ones(batch, self.num_points + 1)
        alpha_weight = torch.ones(batch, self.num_points)
        return PolicyOutput(
            alpha_pos=alpha_pos,
            alpha_weight=alpha_weight,
            log_alpha_pos=alpha_pos.log(),
            log_alpha_weight=alpha_weight.log(),
        )

    def sample_table(self, policy_output):
        batch = policy_output.alpha_pos.shape[0]
        positions = torch.linspace(0.2, 0.8, self.num_points).expand(batch, -1)
        weights = torch.full((batch, self.num_points), 1.0 / self.num_points)
        segments = torch.full((batch, self.num_points + 1), 1.0 / (self.num_points + 1))
        log_prob = torch.zeros(batch)
        entropy = torch.ones(batch)
        return PolicySample(
            positions=positions,
            weights=weights,
            segments=segments,
            log_prob=log_prob,
            entropy_pos=entropy,
            entropy_weight=entropy,
        )


class _StubNet(torch.nn.Module):
    def __init__(self, channels=4, resolution=64):
        super().__init__()
        self.img_channels = channels
        self.img_resolution = resolution
        self.sigma_min = 0.002
        self.sigma_max = 80.0
        self.model = mock.Mock()
        self.model.model = mock.Mock()
        diffusion = mock.Mock()
        diffusion.middle_block = mock.Mock()
        diffusion.middle_block.register_forward_hook = mock.Mock(return_value=mock.Mock())
        self.model.model.diffusion_model = diffusion
        self.model.enc = {
            "8x8_block2": mock.Mock(register_forward_hook=mock.Mock(return_value=mock.Mock())),
            "8x8_block3": mock.Mock(register_forward_hook=mock.Mock(return_value=mock.Mock())),
        }

    def sigma_inv(self, value):
        return torch.log(value)

    def sigma(self, value):
        return value.exp()

    def forward(self, *args, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError


class RLRunnerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.prompt_patch = mock.patch(
            "training.ppo.rl_runner._load_prompts",
            return_value=["prompt_a", "prompt_b"],
        )
        self.prompt_patch.start()
        self.addCleanup(self.prompt_patch.stop)

        self.num_steps = 4
        self.num_points = 2
        self.policy = _StubPolicy(self.num_steps, self.num_points)
        self.net = _StubNet()
        self.config = rl_runner.RLRunnerConfig(
            policy=self.policy,
            net=self.net,
            num_steps=self.num_steps,
            num_points=self.num_points,
            device=torch.device("cpu"),
            prompt_csv=None,
            rloo_k=1,
            rng_seed=123,
        )
        self.runner = rl_runner.EPDRolloutRunner(self.config)

    def test_policy_adapter_shapes(self):
        sample = self.policy.sample_table(
            PolicyOutput(
                alpha_pos=torch.ones(1, self.num_points + 1),
                alpha_weight=torch.ones(1, self.num_points),
                log_alpha_pos=torch.zeros(1, self.num_points + 1),
                log_alpha_weight=torch.zeros(1, self.num_points),
            )
        )
        adapter = rl_runner.PolicyPredictorAdapter(sample, self.config)
        r, scale_dir, scale_time, weight = adapter(batch_size=1, step_idx=0)
        self.assertEqual(r.shape, (1, self.num_points, 1, 1))
        self.assertTrue(torch.all(scale_dir == 1))
        self.assertTrue(torch.all(scale_time == 1))
        self.assertAlmostEqual(weight.sum().item(), 1.0, places=6)

    @mock.patch("training.ppo.rl_runner.epd_sampler")
    def test_rollout_basic_flow(self, mock_sampler):
        mock_sampler.return_value = (torch.zeros(2, 4, 64, 64), None)
        batch = self.runner.rollout(batch_size=2)
        self.assertEqual(batch.images.shape[0], 2)
        self.assertEqual(len(batch.prompts), 2)
        self.assertEqual(len(batch.seeds), 2)
        self.assertTrue(torch.all(torch.isfinite(batch.log_prob)))
        self.assertEqual(batch.metadata["status"], "ok")
        mock_sampler.assert_called_once()

    def test_latent_reproducibility(self):
        seeds = [1, 2]
        latents_a = self.runner._prepare_latents(seeds, (2, 4, 64, 64))
        latents_b = self.runner._prepare_latents(seeds, (2, 4, 64, 64))
        self.assertTrue(torch.allclose(latents_a, latents_b))

    @mock.patch("training.ppo.rl_runner.epd_sampler")
    def test_prompt_rotation(self, mock_sampler):
        mock_sampler.return_value = (torch.zeros(2, 4, 64, 64), None)
        batch = self.runner.rollout(batch_size=2)
        self.assertEqual(batch.prompts[0], "prompt_a")
        self.assertEqual(batch.prompts[1], "prompt_b")
        self.assertEqual(self.runner.prompt_cursor, 0)
        mock_sampler.return_value = (torch.zeros(1, 4, 64, 64), None)
        next_batch = self.runner.rollout(batch_size=1)
        self.assertEqual(next_batch.prompts[0], "prompt_a")

    @mock.patch("training.ppo.rl_runner.epd_sampler", side_effect=RuntimeError("OOM"))
    def test_rollout_error_metadata(self, mock_sampler):
        with self.assertRaises(RuntimeError) as ctx:
            self.runner.rollout(batch_size=1)
        info = ctx.exception.args[0]
        self.assertIsInstance(info, dict)
        self.assertEqual(info["status"], "error")
        self.assertIn("exception", info)
        mock_sampler.assert_called_once()

    def test_policy_adapter_multi_interval(self):
        positions = torch.tensor([[[0.1, 0.4], [0.3, 0.6]]], dtype=torch.float32)
        weights = torch.tensor([[[0.5, 0.5], [0.7, 0.3]]], dtype=torch.float32)
        sample = PolicySample(
            positions=positions,
            weights=weights,
            segments=torch.full((1, 2, 3), 1.0 / 3),
            log_prob=torch.zeros(1),
            entropy_pos=torch.ones(1),
            entropy_weight=torch.ones(1),
        )
        adapter = rl_runner.PolicyPredictorAdapter(sample, self.config)
        r, _, _, w = adapter(batch_size=1, step_idx=1)
        self.assertTrue(torch.allclose(r.squeeze(-1).squeeze(-1), positions[0, 1]))
        self.assertAlmostEqual(w.sum().item(), 1.0, places=6)


SNAPSHOT_PATH = Path(__file__).resolve().parents[3] / "exps" / "00036-ms_coco-10-36-epd-dpm-1-discrete" / "network-snapshot-000005.pkl"


class RealRolloutIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("EPD_INTEGRATION_TEST") == "1", "Integration test disabled")
    @unittest.skipUnless(SNAPSHOT_PATH.exists(), "Snapshot not available")
    def test_real_rollout(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        table = load_predictor_table(SNAPSHOT_PATH, map_location="cpu")
        init = build_dirichlet_params(table, concentration=100.0)
        policy = EPDParamPolicy(
            num_steps=table.num_steps,
            num_points=table.num_points,
            hidden_dim=128,
            num_layers=2,
            context_dim=0,
            dirichlet_init=init,
        ).to(device)
        policy.eval()

        from sample import create_model  # local import to avoid overhead if skipped

        net, model_source = create_model(
            dataset_name=table.metadata.get("dataset_name", "ms_coco"),
            guidance_type=table.metadata.get("guidance_type", "cfg"),
            guidance_rate=table.metadata.get("guidance_rate", 7.5),
            device=device,
        )
        net = net.to(device)
        net.eval()

        config = rl_runner.RLRunnerConfig(
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
            rloo_k=1,
            rng_seed=0,
            verbose=False,
            model_source=model_source,
        )

        runner = rl_runner.EPDRolloutRunner(config)
        batch = runner.rollout(batch_size=1)
        self.assertEqual(batch.images.shape[0], 1)
        self.assertEqual(batch.metadata["status"], "ok")
        self.assertTrue(torch.isfinite(batch.log_prob).all())

        del batch, net, policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

