"""
Unit tests for the HPSv2 reward wrapper.

These tests rely on heavy mocking so they can be executed without the
actual checkpoint present. Run manually:
    python -m training.ppo.tests.test_reward_hps
    EPD_INTEGRATION_TEST=1 python -m training.ppo.tests.test_reward_hps
"""

import os
import unittest
from pathlib import Path
from unittest import mock

import torch

from training.ppo.reward_hps import RewardHPS, RewardHPSConfig
from training.ppo import rl_runner
from training.ppo.policy import EPDParamPolicy
from training.ppo.cold_start import load_predictor_table, build_dirichlet_params


SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[3]
    / "exps"
    / "00036-ms_coco-10-36-epd-dpm-1-discrete"
    / "network-snapshot-000005.pkl"
)
HPS_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3] / "weights" / "HPS_v2.1_compressed.pt"
)


class RewardHPSTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.config = RewardHPSConfig(
            device=self.device,
            batch_size=2,
            enable_amp=False,
            cache_dir="/tmp/hps_cache",
        )

    @mock.patch("training.ppo.reward_hps.hf_hub_download", autospec=True)
    @mock.patch("training.ppo.reward_hps.create_model_and_transforms", autospec=True)
    @mock.patch("training.ppo.reward_hps.get_tokenizer", autospec=True)
    def test_model_loaded_once(self, mock_tokenizer, mock_create_model, mock_hub):
        fake_model = mock.Mock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model

        def forward_fn(images, tokens):
            batch = images.shape[0]
            features = torch.eye(batch)
            return {"image_features": features, "text_features": features}

        fake_model.__call__ = mock.Mock(side_effect=forward_fn)

        preprocess = mock.Mock(side_effect=lambda pil: torch.ones(3, 224, 224))
        mock_create_model.return_value = (fake_model, None, preprocess)

        mock_tokenizer.return_value = mock.Mock(side_effect=lambda prompts: torch.ones(len(prompts), 77, dtype=torch.long))
        mock_hub.return_value = str(Path("/tmp") / "weights.pt")

        reward = RewardHPS(self.config)
        with mock.patch("torch.load", autospec=True) as mock_load:
            mock_load.return_value = {"state_dict": fake_model.state_dict()}
            dummy_images = torch.rand(2, 3, 64, 64)
            dummy_prompts = ["a cat", "a dog"]
            with mock.patch.object(fake_model, "__call__", autospec=True) as mock_call:
                mock_call.side_effect = forward_fn
                reward.score_tensor(dummy_images, dummy_prompts)
                reward.score_tensor(dummy_images, dummy_prompts)

        self.assertEqual(mock_create_model.call_count, 1)
        self.assertEqual(mock_tokenizer.call_count, 1)

    @mock.patch("training.ppo.reward_hps.create_model_and_transforms", autospec=True)
    @mock.patch("training.ppo.reward_hps.get_tokenizer", autospec=True)
    def test_score_tensor_shapes(self, mock_tokenizer, mock_create_model):
        fake_model = mock.Mock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model

        def forward_fn(images, tokens):
            batch = images.shape[0]
            features = torch.eye(batch)
            return {"image_features": features, "text_features": features}

        fake_model.__call__ = mock.Mock(side_effect=forward_fn)  # type: ignore[attr-defined]

        preprocess = mock.Mock(side_effect=lambda pil: torch.ones(3, 224, 224))
        mock_create_model.return_value = (fake_model, None, preprocess)

        mock_tokenizer.return_value = mock.Mock(
            side_effect=lambda prompts: torch.ones(len(prompts), 77, dtype=torch.long)
        )

        with mock.patch("torch.load", autospec=True) as mock_load, mock.patch(
            "training.ppo.reward_hps.hf_hub_download", autospec=True
        ) as mock_hub:
            mock_hub.return_value = str(Path("/tmp") / "weights.pt")
            mock_load.return_value = {"state_dict": {}}

            reward = RewardHPS(self.config)
            images = torch.rand(2, 3, 64, 64)
            prompts = ["prompt1", "prompt2"]
            scores, meta = reward.score_tensor(images, prompts, return_metadata=True)

        self.assertEqual(scores.shape, (2,))
        self.assertIn("duration", meta)
        self.assertEqual(meta["num_images"], 2)

    @mock.patch("training.ppo.reward_hps.create_model_and_transforms", autospec=True)
    @mock.patch("training.ppo.reward_hps.get_tokenizer", autospec=True)
    def test_score_tensor_batches(self, mock_tokenizer, mock_create_model):
        fake_model = mock.Mock()
        fake_model.to.return_value = fake_model
        fake_model.eval.return_value = fake_model

        def forward_single(images, tokens):
            batch = images.shape[0]
            features = torch.eye(batch)
            return {"image_features": features, "text_features": features}

        fake_model.__call__ = mock.Mock(side_effect=forward_single)  # type: ignore[attr-defined]

        preprocess = mock.Mock(side_effect=lambda pil: torch.ones(3, 224, 224))
        mock_create_model.return_value = (fake_model, None, preprocess)
        mock_tokenizer.return_value = mock.Mock(
            side_effect=lambda prompts: torch.ones(len(prompts), 77, dtype=torch.long)
        )

        with mock.patch("torch.load", autospec=True) as mock_load, mock.patch(
            "training.ppo.reward_hps.hf_hub_download", autospec=True
        ) as mock_hub:
            mock_hub.return_value = str(Path("/tmp") / "weights.pt")
            mock_load.return_value = {"state_dict": {}}

            small_config = RewardHPSConfig(
                device=self.device,
                batch_size=1,
                enable_amp=False,
            )
            reward = RewardHPS(small_config)
            images = torch.rand(3, 3, 64, 64)
            prompts = ["p0", "p1", "p2"]
            reward.score_tensor(images, prompts)

        self.assertEqual(fake_model.__call__.call_count, 3)

    def test_prompt_mismatch(self):
        reward = RewardHPS(self.config)
        images = torch.rand(2, 3, 64, 64)
        prompts = ["only one prompt"]
        with self.assertRaises(ValueError):
            reward.score_tensor(images, prompts)


class RewardHPSIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("EPD_INTEGRATION_TEST") == "1",
        "Integration test disabled",
    )
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    @unittest.skipUnless(SNAPSHOT_PATH.exists(), "Predictor snapshot not available")
    @unittest.skipUnless(HPS_WEIGHTS_PATH.exists(), "HPS weights not available")
    def test_reward_with_real_images(self):
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
            context_dim=0,
            dirichlet_init=init,
        ).to(device)
        policy.eval()

        from sample import create_model  # local import to avoid overhead when skipped

        net, model_source = create_model(
            dataset_name=table.metadata.get("dataset_name", "ms_coco"),
            guidance_type=table.metadata.get("guidance_type", "cfg"),
            guidance_rate=float(table.metadata.get("guidance_rate", 7.5)),
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
            rng_seed=123,
            verbose=False,
            model_source=model_source,
        )

        runner = rl_runner.EPDRolloutRunner(config)
        batch = runner.rollout(batch_size=1)
        self.assertEqual(batch.images.shape[0], 1)
        self.assertEqual(len(batch.prompts), 1)

        with torch.no_grad():
            decoded = net.model.decode_first_stage(batch.images)
        decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0).to(
            dtype=torch.float32, device=device
        )

        reward = RewardHPS(
            RewardHPSConfig(
                device=device,
                batch_size=1,
                enable_amp=True,
                weights_path=HPS_WEIGHTS_PATH,
                cache_dir=HPS_WEIGHTS_PATH.parent,
            )
        )
        scores, meta = reward.score_tensor(decoded, batch.prompts, return_metadata=True)

        self.assertEqual(scores.shape, (1,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(meta["num_images"], 1)
        self.assertGreater(meta["duration"], 0.0)
        self.assertEqual(meta["device"], str(device))

        repeat_scores = reward.score_tensor(decoded, batch.prompts)
        self.assertTrue(torch.allclose(repeat_scores, scores, atol=1e-4))

        prompt = batch.prompts[0]
        print(f"[RewardHPSIntegration] prompt: {prompt}")
        print(f"[RewardHPSIntegration] score: {scores.item():.6f}")
        print(f"[RewardHPSIntegration] metadata: {meta}")

        del batch, runner, net, policy, reward
        torch.cuda.empty_cache()

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
