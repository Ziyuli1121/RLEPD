from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from . import config as cfg
from .cold_start import build_dirichlet_params, load_predictor_table
from .policy import EPDParamPolicy
from .ppo_trainer import PPOTrainer, PPOTrainerConfig
from .reward_hps import RewardHPS, RewardHPSConfig
from .rl_runner import EPDRolloutRunner, RLRunnerConfig


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch PPO fine-tuning for EPD predictor tables.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/ppo/cfgs/sd15_base.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values (dot-separated keys).",
    )
    parser.add_argument("--run-name", type=str, help="Override run.run_name without editing YAML.")
    parser.add_argument("--max-steps", type=int, help="Override number of PPO steps.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load and validate configuration, print resolved values, then exit.",
    )
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_run_directory(run_config: cfg.RunConfig) -> None:
    base = run_config.output_root / run_config.run_id
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = run_config.output_root / f"{run_config.run_id}-{suffix:02d}"
        suffix += 1
    run_config.run_dir = candidate.resolve()
    (run_config.run_dir / "configs").mkdir(parents=True, exist_ok=True)
    (run_config.run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_config.run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_config.run_dir / "samples").mkdir(parents=True, exist_ok=True)


def enrich_model_dimensions(full_config: cfg.FullConfig, dry_run: bool) -> Dict[str, int]:
    table = load_predictor_table(full_config.data.predictor_snapshot, map_location="cpu")
    if full_config.model.num_steps is not None and full_config.model.num_steps != table.num_steps:
        raise cfg.ConfigError(
            f"num_steps mismatch: config={full_config.model.num_steps} vs table={table.num_steps}"
        )
    if full_config.model.num_points is not None and full_config.model.num_points != table.num_points:
        raise cfg.ConfigError(
            f"num_points mismatch: config={full_config.model.num_points} vs table={table.num_points}"
        )
    full_config.model.num_steps = table.num_steps
    full_config.model.num_points = table.num_points
    return {
        "num_steps": table.num_steps,
        "num_points": table.num_points,
        "schedule_type": table.metadata.get("schedule_type"),
        "schedule_rho": table.metadata.get("schedule_rho"),
    }


def save_config_snapshot(full_config: cfg.FullConfig) -> None:
    config_path = full_config.run.run_dir / "configs" / "resolved_config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        handle.write(cfg.pretty_format_config(full_config))


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser()

    try:
        raw = cfg.load_raw_config(config_path, overrides=args.override)
        if args.run_name:
            raw.setdefault("run", {})["run_name"] = args.run_name
        if args.max_steps is not None:
            raw.setdefault("ppo", {})["steps"] = args.max_steps
        full_config = cfg.build_config(raw)
        meta = enrich_model_dimensions(full_config, dry_run=args.dry_run)
        cfg.validate_config(full_config, check_paths=not args.dry_run)
    except cfg.ConfigError as err:
        print(f"[ConfigError] {err}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("==== Resolved Configuration ====")
        print(cfg.pretty_format_config(full_config))
        print("Derived from predictor snapshot:")
        print(json.dumps(meta, indent=2))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_run_directory(full_config.run)
    save_config_snapshot(full_config)
    seed_everything(full_config.run.seed)

    print(f"[Launch] run_dir={full_config.run.run_dir}")
    print(f"[Launch] Using device: {device}")
    print("[Launch] Loading predictor snapshot...")
    predictor_table = load_predictor_table(full_config.data.predictor_snapshot, map_location="cpu")
    dirichlet_init = build_dirichlet_params(
        predictor_table,
        concentration=full_config.ppo.dirichlet_concentration,
    )
    policy = EPDParamPolicy(
        num_steps=predictor_table.num_steps,
        num_points=predictor_table.num_points,
        hidden_dim=128,
        num_layers=2,
        dirichlet_init=dirichlet_init,
    ).to(device)

    print("[Launch] Loading Stable Diffusion model...")
    from sample import create_model  # Local import to avoid overhead on dry run

    net, model_source = create_model(
        dataset_name=full_config.model.dataset_name,
        guidance_type=full_config.model.guidance_type,
        guidance_rate=full_config.model.guidance_rate,
        device=device,
    )
    net = net.to(device)
    net.eval()

    reward = RewardHPS(
        RewardHPSConfig(
            device=device,
            batch_size=full_config.reward.batch_size,
            enable_amp=full_config.reward.enable_amp,
            weights_path=full_config.reward.weights_path,
            cache_dir=full_config.reward.cache_dir,
        )
    )

    runner_config = RLRunnerConfig(
        policy=policy,
        net=net,
        num_steps=predictor_table.num_steps,
        num_points=predictor_table.num_points,
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
        verbose=False,
        model_source=model_source,
    )
    runner = EPDRolloutRunner(runner_config)

    trainer_config = PPOTrainerConfig(
        device=device,
        rollout_batch_size=full_config.ppo.rollout_batch_size,
        rloo_k=full_config.ppo.rloo_k,
        ppo_epochs=full_config.ppo.ppo_epochs,
        minibatch_size=full_config.ppo.minibatch_size,
        learning_rate=full_config.ppo.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        clip_range=full_config.ppo.clip_range,
        kl_coef=full_config.ppo.kl_coef,
        entropy_coef=full_config.ppo.entropy_coef,
        normalize_advantages=True,
        max_grad_norm=full_config.ppo.max_grad_norm,
        decode_rgb=full_config.ppo.decode_rgb,
        image_value_range=(0.0, 1.0),
    )
    trainer = PPOTrainer(policy, runner, reward, trainer_config)

    metrics_path = full_config.run.run_dir / "logs" / "metrics.jsonl"
    print(f"[Launch] Writing metrics to {metrics_path}")

    start_time = time.time()
    step = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        while step < full_config.ppo.steps:
            step += 1
            metrics = trainer.train_step()
            metrics["step"] = step
            metrics["elapsed_sec"] = time.time() - start_time
            print(json.dumps(metrics), file=metrics_file, flush=True)
            if step % full_config.logging.log_interval == 0:
                summary = ", ".join(
                    f"{k}={metrics[k]:.4f}"
                    for k in ("reward_mean", "kl", "policy_loss", "entropy")
                    if k in metrics
                )
                print(f"[Step {step}] {summary}")

    print(f"[Launch] Finished {full_config.ppo.steps} PPO steps. Results saved to {full_config.run.run_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()


'''

看一下配置是否符合冷启动蒸馏表参数
python -m training.ppo.launch --dry-run

'''