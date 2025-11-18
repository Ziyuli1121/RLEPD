#!/usr/bin/env python3
"""
Prompts/conditioning self-test for RL runner paths.

This script focuses on Stage 3 requirements:
  * Loads a predictor snapshot and resolved config.
  * Instantiates the correct backend (LDM or SD3) via sample.create_model_backend.
  * Invokes RLRunnerConfig._prepare_conditions to verify that both positive
    and negative prompts are handled as expected.

Usage examples:
    python tests/test_rl_conditioning.py --config training/ppo/cfgs/sd15_base.yaml
    python tests/test_rl_conditioning.py --config training/ppo/cfgs/sd3_smoke.yaml

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample import create_model_backend
from training.ppo import config as cfg
from training.ppo.cold_start import load_predictor_table
from training.ppo.rl_runner import RLRunnerConfig, EPDRolloutRunner


def _load_config(path: Path, overrides: Optional[List[str]] = None) -> cfg.FullConfig:
    raw = cfg.load_raw_config(path, overrides=overrides or [])
    full = cfg.build_config(raw)
    cfg.validate_config(full, check_paths=False)
    return full


def _parse_backend_overrides(raw: str | None) -> Dict[str, object]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--backend-overrides must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--backend-overrides must be a JSON object.")
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RL runner conditioning smoke test.")
    parser.add_argument("--config", type=Path, required=True, help="Resolved YAML config used for PPO stage.")
    parser.add_argument(
        "--predictor",
        type=Path,
        help="Optional override for predictor snapshot (defaults to config.data.predictor_snapshot).",
    )
    parser.add_argument(
        "--backend-overrides",
        type=str,
        default=None,
        help="JSON object to override model.backend/backend_options (useful for SD3 smoke tests).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-prompts", type=int, default=None, help="Number of prompts to pull from the prompt list.")
    return parser


def _create_runner(full_config: cfg.FullConfig, device: torch.device, overrides: Dict[str, object]) -> EPDRolloutRunner:
    backend = overrides.get("backend", full_config.model.backend)
    backend_options = overrides.get("backend_options", full_config.model.backend_options)

    net, model_source = create_model_backend(
        dataset_name=full_config.model.dataset_name,
        guidance_type=full_config.model.guidance_type,
        guidance_rate=full_config.model.guidance_rate,
        backend=backend,
        backend_config=backend_options,
        device=device,
    )

    table = load_predictor_table(
        full_config.data.predictor_snapshot,
        map_location="cpu",
    )

    runner_config = RLRunnerConfig(
        policy=None,
        net=net,
        num_steps=table.num_steps,
        num_points=table.num_points,
        device=device,
        guidance_type=full_config.model.guidance_type,
        guidance_rate=full_config.model.guidance_rate,
        schedule_type=full_config.model.schedule_type,
        schedule_rho=full_config.model.schedule_rho,
        dataset_name=full_config.model.dataset_name,
        precision=torch.float32,
        prompt_csv=full_config.data.prompt_csv,
        rloo_k=full_config.ppo.rloo_k,
        rng_seed=full_config.run.seed,
        model_source=model_source,
        backend=backend,
        backend_config=backend_options,
    )
    runner = EPDRolloutRunner(runner_config)
    return runner


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    full_config = _load_config(args.config)
    backend_overrides = _parse_backend_overrides(args.backend_overrides)
    if args.predictor:
        full_config.data.predictor_snapshot = args.predictor

    device = torch.device(args.device)
    runner = _create_runner(full_config, device, backend_overrides)

    desired = args.num_prompts or runner.config.rloo_k
    if desired % runner.config.rloo_k != 0:
        desired = runner.config.rloo_k * ((desired // runner.config.rloo_k) + 1)
    prompts, _ = runner._sample_prompts_and_seeds(desired)
    condition, unconditional, _ = runner._prepare_conditions(prompts)

    backend = getattr(runner.net, "backend", "ldm")
    print(f"Backend: {backend} | prompts: {prompts}")

    if backend == "sd3":
        assert condition is not None, "SD3 backend must return SD3Conditioning."
        print(
            f"SD3 condition prepared | guidance_scale={condition.guidance_scale} "
            f"| prompt_embeds={tuple(condition.prompt_embeds.shape)}"
        )
    else:
        assert condition is not None, "LDM backend must return cond tensor"
        print(f"LDM condition prepared | cond={tuple(condition.shape)}")
        if unconditional is not None:
            print(f"Unconditional condition available | shape={tuple(unconditional.shape)}")

    print("RL runner conditioning test finished successfully.")


if __name__ == "__main__":
    main()
