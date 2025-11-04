import os
import types
import unittest
from pathlib import Path
from typing import Optional

import torch

from training.ppo import rl_runner
from training.ppo.cold_start import build_dirichlet_params, load_predictor_table
from training.ppo.policy import EPDParamPolicy
from training.ppo.reward_multi import RewardMetricWeights, RewardMultiMetric, RewardMultiMetricConfig, RewardMultiMetricPaths
from training.ppo.reward_hps import RewardHPSConfig

'''
python -m training.ppo.tests.test_reward_multi
EPD_INTEGRATION_TEST=1 python -m training.ppo.tests.test_reward_multi

'''

SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[3]
    / "exps"
    / "00036-ms_coco-10-36-epd-dpm-1-discrete"
    / "network-snapshot-000005.pkl"
)
HPS_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[3] / "weights" / "HPS_v2.1_compressed.pt"
)

class RewardMultiMetricTest(unittest.TestCase):
    def test_weighted_average_and_metadata(self) -> None:
        config = RewardMultiMetricConfig(
            device=torch.device("cpu"),
            batch_size=2,
            weights=RewardMetricWeights(hps=1.0, pickscore=1.0, imagereward=0.0, clip=0.0, aesthetic=0.0),
            hps=RewardHPSConfig(weights_path=Path("/tmp/dummy.pt")),
        )

        reward = RewardMultiMetric(config)

        def _stub_hps(images, prompts, batch_size=None, return_metadata=False):
            scores = torch.tensor([0.25, 0.20], dtype=torch.float32)
            if return_metadata:
                return scores, {}
            return scores

        reward._hps = types.SimpleNamespace(score_tensor=_stub_hps)  # type: ignore[attr-defined]
        reward._score_pickscore = lambda prompts, images, batch: torch.tensor([20.0, 24.0], dtype=torch.float32)  # type: ignore[attr-defined]
        reward._score_imagereward = lambda prompts, images: torch.zeros(len(prompts), dtype=torch.float32)  # type: ignore[attr-defined]
        reward._score_clip = lambda prompts, images, batch: torch.zeros(len(prompts), dtype=torch.float32)  # type: ignore[attr-defined]
        reward._score_aesthetic = lambda images, batch: torch.zeros(len(images), dtype=torch.float32)  # type: ignore[attr-defined]

        images = torch.zeros(2, 3, 8, 8)
        prompts = ["a", "b"]
        scores, metadata = reward.score_tensor(images, prompts, return_metadata=True)

        expected_hps = torch.tensor([0.25, 0.20], dtype=torch.float32)
        expected_pick = torch.tensor([20.0, 24.0], dtype=torch.float32)
        expected_hps_norm = torch.clamp(expected_hps, 0.0, 1.0)
        expected_pick_norm = torch.clamp(expected_pick / 26.0, 0.0, 1.0)
        expected = (expected_hps_norm + expected_pick_norm) / 2.0

        self.assertTrue(torch.allclose(scores, expected, atol=1e-6))

        raw_scores = metadata.get("raw_scores", {}) if isinstance(metadata, dict) else {}
        self.assertIn("hps", raw_scores)
        self.assertIn("pickscore", raw_scores)
        self.assertTrue(torch.allclose(raw_scores["hps"], expected_hps, atol=1e-6))
        self.assertTrue(torch.allclose(raw_scores["pickscore"], expected_pick, atol=1e-6))

        normalized = metadata.get("normalized_scores", {}) if isinstance(metadata, dict) else {}
        self.assertIn("hps", normalized)
        self.assertIn("pickscore", normalized)
        self.assertTrue(torch.allclose(normalized["hps"], expected_hps_norm, atol=1e-6))
        self.assertTrue(torch.allclose(normalized["pickscore"], expected_pick_norm, atol=1e-6))

        weights = metadata.get("weights", {}) if isinstance(metadata, dict) else {}
        self.assertEqual(weights.get("hps"), 1.0)
        self.assertEqual(weights.get("pickscore"), 1.0)


class RewardMultiMetricIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("EPD_INTEGRATION_TEST") == "1",
        "Integration test disabled",
    )
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device required")
    @unittest.skipUnless(SNAPSHOT_PATH.exists(), "Predictor snapshot not available")
    @unittest.skipUnless(HPS_WEIGHTS_PATH.exists(), "HPS weights not available")
    def test_multi_reward_with_real_images(self) -> None:
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

        from sample import create_model  # local import to avoid overhead unless running

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
            rng_seed=321,
            verbose=False,
            model_source=model_source,
        )

        runner = rl_runner.EPDRolloutRunner(config)
        batch = runner.rollout(batch_size=1)
        self.assertEqual(batch.images.shape[0], 1)
        self.assertEqual(len(batch.prompts), 1)

        with torch.no_grad():
            decoded = net.model.decode_first_stage(batch.images)
        decoded = torch.clamp((decoded + 1.0) / 2.0, 0.0, 1.0).to(dtype=torch.float32, device=device)

        def resolve_path(*candidates: str) -> Optional[str]:
            for candidate in candidates:
                if not candidate:
                    continue
                path = Path(candidate).expanduser()
                if path.exists():
                    return str(path)
            return None

        def hf_snapshot(repo_id: str) -> Optional[str]:
            hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
            base = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
            snapshots = base / "snapshots"
            if snapshots.is_dir():
                ordered = sorted(p for p in snapshots.iterdir() if p.is_dir())
                if ordered:
                    return str(ordered[-1])
            return None

        weights_root = Path(__file__).resolve().parents[3] / "weights"

        env_paths = {
            "PICKSCORE_MODEL_PATH": os.environ.get("PICKSCORE_MODEL_PATH") or hf_snapshot("yuvalkirstain/PickScore_v1"),
            "PICKSCORE_PROCESSOR_PATH": os.environ.get("PICKSCORE_PROCESSOR_PATH") or hf_snapshot("laion/CLIP-ViT-H-14-laion2B-s32B-b79K"),
            "IMAGEREWARD_CHECKPOINT_PATH": os.environ.get("IMAGEREWARD_CHECKPOINT_PATH") or resolve_path(
                weights_root / "ImageReward.pt",
                hf_snapshot("THUDM/ImageReward"),
            ),
            "IMAGEREWARD_MED_CONFIG_PATH": os.environ.get("IMAGEREWARD_MED_CONFIG_PATH") or resolve_path(
                weights_root / "med_config.json",
            ),
            "IMAGEREWARD_CACHE_DIR": os.environ.get("IMAGEREWARD_CACHE_DIR") or resolve_path(weights_root),
            "CLIP_MODEL_CACHE_DIR": os.environ.get("CLIP_MODEL_CACHE_DIR") or resolve_path(weights_root / "clip"),
            "AESTHETIC_PREDICTOR_PATH": os.environ.get("AESTHETIC_PREDICTOR_PATH") or resolve_path(
                weights_root / "sac+logos+ava1-l14-linearMSE.pth",
            ),
            "AESTHETIC_CLIP_CACHE_DIR": os.environ.get("AESTHETIC_CLIP_CACHE_DIR") or resolve_path(weights_root / "clip"),
        }

        required_keys = (
            "PICKSCORE_MODEL_PATH",
            "PICKSCORE_PROCESSOR_PATH",
            "IMAGEREWARD_CHECKPOINT_PATH",
            "IMAGEREWARD_MED_CONFIG_PATH",
            "AESTHETIC_PREDICTOR_PATH",
        )
        missing = [name for name in required_keys if not env_paths.get(name)]
        if missing:
            self.skipTest(
                "Missing local caches for multi-reward integration: " + ", ".join(missing)
            )

        reward = RewardMultiMetric(
            RewardMultiMetricConfig(
                device=device,
                batch_size=1,
                weights=RewardMetricWeights(
                    hps=1.0,
                    pickscore=1.0,
                    imagereward=1.0,
                    clip=1.0,
                    aesthetic=1.0,
                ),
                hps=RewardHPSConfig(
                    device=device,
                    batch_size=1,
                    enable_amp=True,
                    weights_path=HPS_WEIGHTS_PATH,
                    cache_dir=HPS_WEIGHTS_PATH.parent,
                ),
                pickscore_model_name_or_path=env_paths["PICKSCORE_MODEL_PATH"],
                pickscore_processor_name_or_path=env_paths["PICKSCORE_PROCESSOR_PATH"],
                paths=RewardMultiMetricPaths(
                    imagereward_checkpoint=env_paths["IMAGEREWARD_CHECKPOINT_PATH"],
                    imagereward_med_config=env_paths["IMAGEREWARD_MED_CONFIG_PATH"],
                    imagereward_cache_dir=env_paths["IMAGEREWARD_CACHE_DIR"],
                    clip_cache_dir=env_paths["CLIP_MODEL_CACHE_DIR"],
                    aesthetic_clip_path=env_paths["AESTHETIC_CLIP_CACHE_DIR"],
                    aesthetic_predictor_path=env_paths["AESTHETIC_PREDICTOR_PATH"],
                ),
            )
        )

        scores, metadata = reward.score_tensor(decoded, batch.prompts, return_metadata=True)

        self.assertEqual(scores.shape, (1,))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertIn("raw_scores", metadata)
        self.assertIn("normalized_scores", metadata)
        self.assertIn("weights", metadata)

        components = ("hps", "pickscore", "imagereward", "clip", "aesthetic")
        normalized_scores = metadata["normalized_scores"]
        raw_scores = metadata["raw_scores"]
        for key in components:
            self.assertIn(key, raw_scores)
            self.assertIn(key, normalized_scores)
            tensor_raw = raw_scores[key]
            tensor_norm = normalized_scores[key]
            self.assertTrue(torch.isfinite(tensor_raw).all())
            self.assertTrue(torch.isfinite(tensor_norm).all())

        average = sum(normalized_scores[key] for key in components) / float(len(components))
        self.assertTrue(torch.allclose(scores.cpu(), average.cpu(), atol=1e-6))
        for key in components:
            self.assertEqual(metadata["weights"][key], 1.0)

        repeat_scores = reward.score_tensor(decoded, batch.prompts)
        self.assertTrue(torch.allclose(repeat_scores, scores, atol=1e-4))

        prompt = batch.prompts[0]
        for key in components:
            raw_mean = raw_scores[key].mean().item()
            print(f"[RewardMultiIntegration] {key}_raw_mean={raw_mean:.6f}")
        print(f"[RewardMultiIntegration] prompt: {prompt} mixed={scores.item():.6f}")

        del batch, runner, net, policy, reward
        torch.cuda.empty_cache()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
