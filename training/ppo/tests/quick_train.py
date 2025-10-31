"""
Quick smoke test for the PPO training pipeline.

This script reuses the Stage 7 launch entrypoint and runs a very small
number of PPO iterations (default: 2 steps) to ensure the entire stack
— predictor loading, Stable Diffusion sampling, HPS scoring, PPO update,
logging — works end to end before Stage 8 development.

Example:
    python -m training.ppo.tests.quick_train \
        --config training/ppo/cfgs/sd15_base.yaml \
        --steps 20
"""

import argparse
from pathlib import Path
from typing import List, Optional

import json
import time

import numpy as np
import torch
from PIL import Image

from training.ppo.config import (
    ConfigError,
    build_config,
    load_raw_config,
    pretty_format_config,
    validate_config,
)
from training.ppo.launch import (
    enrich_model_dimensions,
    ensure_run_directory,
    save_config_snapshot,
    seed_everything,
    save_policy_checkpoint,
)
from training.ppo.cold_start import build_dirichlet_params, load_predictor_table
from training.ppo.policy import EPDParamPolicy
from training.ppo.ppo_trainer import PPOTrainer, PPOTrainerConfig
from training.ppo.reward_hps import RewardHPS, RewardHPSConfig
from training.ppo.rl_runner import EPDRolloutRunner, RLRunnerConfig

from sample import create_model


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick PPO smoke test.")
    parser.add_argument(
        "--config",
        type=str,
        default="training/ppo/cfgs/sd15_base.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1,
        help="Number of PPO iterations to run during the smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the resolved configuration (no training).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional configuration overrides.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    steps = max(2, args.steps)
    overrides = list(args.override)
    overrides.append(f"ppo.steps={steps}")
    overrides.append("logging.log_interval=1")

    try:
        raw = load_raw_config(Path(args.config), overrides)
        full_config = build_config(raw)
        meta = enrich_model_dimensions(full_config, dry_run=args.dry_run)
        validate_config(full_config, check_paths=not args.dry_run)
    except ConfigError as err:
        print(f"[QuickTrain] ConfigError: {err}")
        return

    print("==== Resolved Configuration for Quick Train ====")
    print(pretty_format_config(full_config))
    print("Derived from predictor snapshot:")
    print(meta)

    if args.dry_run:
        return

    ensure_run_directory(full_config.run)
    save_config_snapshot(full_config)
    out_samples = full_config.run.run_dir / "samples"
    out_samples.mkdir(parents=True, exist_ok=True)
    metrics_path = full_config.run.run_dir / "logs" / "metrics.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(full_config.run.seed)

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

    captured_batches: List = []
    original_rollout = runner.rollout

    def wrapped_rollout(batch_size: int):
        batch = original_rollout(batch_size)
        captured_batches.append(batch)
        return batch

    runner.rollout = wrapped_rollout  # type: ignore

    print(f"[QuickTrain] Saving initial rollout images to {out_samples}")
    initial_batch = runner.rollout(full_config.ppo.rollout_batch_size)
    initial_images = trainer._prepare_images(initial_batch).detach().cpu()
    for idx in range(initial_images.shape[0]):
        img = initial_images[idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0.0, 1.0)
        Image.fromarray((img * 255).astype(np.uint8)).save(out_samples / f"step0_image_{idx}.png")

    start_time = time.time()
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        for step in range(1, steps + 1):
            captured_start = len(captured_batches)
            metrics = trainer.train_step()
            metrics_record = {
                "step": step,
                "reward_mean": float(metrics.get("reward_mean", float("nan"))),
                "reward_std": float(metrics.get("reward_std", float("nan"))),
                "kl": float(metrics.get("kl", float("nan"))),
                "policy_loss": float(metrics.get("policy_loss", float("nan"))),
                "entropy": float(metrics.get("entropy", float("nan"))),
                "ratio": float(metrics.get("ratio", float("nan"))),
                "elapsed_sec": float(time.time() - start_time),
            }
            print(json.dumps(metrics_record), file=metrics_file, flush=True)

            print(
                f"[QuickTrain] Step {step}: reward_mean={metrics_record['reward_mean']:.4f} "
                f"kl={metrics_record['kl']:.4f} policy_loss={metrics_record['policy_loss']:.4f}"
            )

            if step % max(1, full_config.logging.save_interval) == 0 or step == steps:
                ckpt_path = save_policy_checkpoint(full_config.run, trainer.policy, step)
                print(f"[QuickTrain] Saved policy checkpoint to {ckpt_path}")

            if len(captured_batches) == captured_start:
                continue
            batch = captured_batches[captured_start]
            images = trainer._prepare_images(batch).detach().cpu()
            with torch.no_grad():
                scored = reward.score_tensor(images.to(device), batch.prompts, return_metadata=False)
                if isinstance(scored, tuple):
                    scores = scored[0]
                else:
                    scores = scored
            scores = scores.detach().cpu().tolist()

            for idx in range(images.shape[0]):
                img = images[idx].permute(1, 2, 0).numpy()
                img = np.clip(img, 0.0, 1.0)
                score = scores[idx] if idx < len(scores) else float("nan")
                prompt = batch.prompts[idx] if idx < len(batch.prompts) else f"prompt_{idx}"
                Image.fromarray((img * 255).astype(np.uint8)).save(out_samples / f"step{step}_image_{idx}.png")
                print(f"[QuickTrain] Saved step {step} image {idx} (score={score:.4f}) prompt=\"{prompt}\"")
            captured_batches[:] = captured_batches[:captured_start]

    print(f"[QuickTrain] Images saved under {out_samples}")


if __name__ == "__main__":  # pragma: no cover
    main()




'''

python -m training.ppo.tests.quick_train --steps 2          # 跑2个step
python -m training.ppo.tests.quick_train --dry-run          # 看配置
python -m training.ppo.tests.quick_train --steps 40 \
       --override ppo.rollout_batch_size=24


'''
