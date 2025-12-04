#!/usr/bin/env python3
"""Generate a faux EPD predictor snapshot without Stage-1 training."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "fake_train.py requires PyTorch. Install torch in the current environment."
    ) from exc

from training.networks import EPD_predictor

# FlowMatch Euler defaults used by SD3 scheduler (shift=3.0)
SD3_FLOWMATCH_SIGMA_MIN = 0.0029940119760479044
SD3_FLOWMATCH_SIGMA_MAX = 1.0


@dataclass
class SnapshotConfig:
    num_steps: int
    num_points: int
    sampler_stu: str
    sampler_tea: str
    M: int
    guidance_type: str
    guidance_rate: float
    schedule_type: str
    schedule_rho: float
    dataset_name: str
    afs: bool
    scale_dir: float
    scale_time: float
    fcn: bool
    max_order: int
    predict_x0: bool
    lower_order_final: bool
    alpha: float
    r_base: float
    r_epsilon: float
    weight_base: float | None
    weight_epsilon: float
    backend: str = "ldm"
    backend_options: dict = field(default_factory=dict)
    sigma_min: Optional[float] = None
    sigma_max: Optional[float] = None
    flowmatch_mu: Optional[float] = None
    flowmatch_shift: Optional[float] = None
    resolution: int = 1024


def _positive_float(text: str) -> float:
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _bool_flag(text: str) -> bool:
    lowered = text.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {text}")


def _parse_betas(text: str) -> Tuple[float, float]:
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("optimizer betas must contain exactly two comma-separated values.")
    try:
        beta1, beta2 = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("optimizer betas must be numeric.") from exc
    return beta1, beta2


def _parse_backend_options(text: str | None) -> dict:
    if text is None:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"backend-options must be valid JSON: {text}") from exc
    if not isinstance(data, dict):
        raise argparse.ArgumentTypeError("backend-options must be a JSON object.")
    return data


def _format_run_dir(raw_outdir: Path, override: str | None) -> str:
    if override:
        return override
    if raw_outdir.is_absolute():
        return str(raw_outdir)
    posix = raw_outdir.as_posix()
    return posix if posix.startswith("./") else f"./{posix}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fake network-snapshot-*.pkl for PPO warm starts."
    )
    parser.add_argument("--outdir", type=Path, default=Path("exps/fake-run"))
    parser.add_argument("--snapshot-step", type=int, default=5, help="Suffix used in filename.")
    parser.add_argument(
        "--snapshot-name",
        type=str,
        default=None,
        help="Optional explicit filename. Overrides --snapshot-step.",
    )
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--num-points", type=int, required=True)
    parser.add_argument("--sampler-stu", type=str, default="epd")
    parser.add_argument("--sampler-tea", type=str, default="dpm")
    parser.add_argument("--M", type=int, default=1)
    parser.add_argument("--guidance-type", type=str, default="cfg")
    parser.add_argument("--guidance-rate", type=float, default=7.5)
    parser.add_argument("--schedule-type", type=str, default="discrete")
    parser.add_argument("--schedule-rho", type=float, default=1.0)
    parser.add_argument("--dataset-name", type=str, default="ms_coco")
    parser.add_argument("--afs", type=_bool_flag, default=False)
    parser.add_argument("--scale-dir", type=float, default=0.0)
    parser.add_argument("--scale-time", type=float, default=0.0)
    parser.add_argument("--fcn", type=_bool_flag, default=False)
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--predict-x0", type=_bool_flag, default=False)
    parser.add_argument("--lower-order-final", type=_bool_flag, default=True)
    parser.add_argument("--alpha", type=float, default=10.0)
    parser.add_argument("--r-base", type=float, default=0.5, help="Target value for r entries.")
    parser.add_argument(
        "--r-epsilon",
        type=_positive_float,
        default=1e-3,
        help="Perturbation applied around r-base to keep Dirichlet well-behaved.",
    )
    parser.add_argument(
        "--weight-base",
        type=float,
        default=None,
        help="Optional base weight value. Defaults to uniform 1/num_points.",
    )
    parser.add_argument(
        "--weight-epsilon",
        type=float,
        default=0.0,
        help="Optional perturbation around the base weight value.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total-kimg", type=int, default=5)
    parser.add_argument("--kimg-per-tick", type=int, default=1)
    parser.add_argument("--snapshot-ticks", type=int, default=5)
    parser.add_argument("--state-dump-ticks", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-gpu", type=int, default=None)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--cudnn-benchmark", type=_bool_flag, default=True)
    parser.add_argument("--optimizer-lr", type=float, default=0.01)
    parser.add_argument("--optimizer-eps", type=float, default=1e-8)
    parser.add_argument("--optimizer-betas", type=str, default="0.9,0.999")
    parser.add_argument("--cos-lr-schedule", type=_bool_flag, default=False)
    parser.add_argument("--prompt-path", type=Path, default=None)
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--write-log", type=_bool_flag, default=True)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write a JSON summary of the generated table.",
    )
    parser.add_argument(
        "--training-options",
        type=Path,
        default=None,
        help="Optional JSON file mirroring train.py options for traceability.",
    )
    parser.add_argument(
        "--skip-training-options",
        action="store_true",
        help="Skip writing training_options.json even though defaults are provided.",
    )
    parser.add_argument("--backend", type=str, default="ldm", help="Backend identifier, e.g., 'ldm' or 'sd3'.")
    parser.add_argument(
        "--backend-options",
        type=str,
        default=None,
        help="JSON object with backend-specific options (e.g., model IDs).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        choices=[512, 1024],
        default=1024,
        help="Target image resolution for SD3 backends.",
    )
    parser.add_argument(
        "--sigma-min",
        type=_positive_float,
        default=None,
        help="Optional explicit sigma_min for the time schedule metadata.",
    )
    parser.add_argument(
        "--sigma-max",
        type=_positive_float,
        default=None,
        help="Optional explicit sigma_max for the time schedule metadata.",
    )
    parser.add_argument(
        "--flowmatch-mu",
        type=float,
        default=None,
        help="Optional mu override when using flowmatch schedules.",
    )
    parser.add_argument(
        "--flowmatch-shift",
        type=float,
        default=None,
        help="Optional shift override when using flowmatch schedules.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    """Basic sanity checks so we fail fast on incompatible settings."""

    if args.num_steps < 2:
        raise ValueError("--num-steps must be at least 2.")
    if args.num_points < 1:
        raise ValueError("--num-points must be at least 1.")

    backend = (args.backend or "ldm").lower()
    if backend == "sd3":
        if args.schedule_type != "flowmatch":
            raise ValueError("SD3 backend requires --schedule-type flowmatch.")
        # Flow-matching runs on sigma in [0,1]; make sure bounds are set.
        if args.sigma_min is None:
            args.sigma_min = SD3_FLOWMATCH_SIGMA_MIN
        if args.sigma_max is None:
            args.sigma_max = SD3_FLOWMATCH_SIGMA_MAX


def _ensure_monotonic(row: np.ndarray, min_gap: float = 1e-4) -> np.ndarray:
    prev = min_gap
    max_allowed = 1.0 - min_gap
    for idx in range(row.shape[0]):
        remaining = row.shape[0] - idx - 1
        lower = prev + min_gap
        upper = max_allowed - remaining * min_gap
        value = float(np.clip(row[idx], lower, upper))
        row[idx] = value
        prev = value
    return row


def _build_positions(config: SnapshotConfig) -> np.ndarray:
    base = np.full((config.num_steps - 1, config.num_points), config.r_base, dtype=np.float64)
    offsets = (np.arange(config.num_points) - (config.num_points - 1) / 2.0) * config.r_epsilon
    positions = base + offsets

    if config.num_points == 1:
        alt = ((np.arange(config.num_steps - 1) % 2) * 2 - 1) * config.r_epsilon
        positions[:, 0] = np.clip(config.r_base + alt, 1e-4, 1 - 1e-4)
        return positions

    positions = np.clip(positions, 1e-4, 1 - 1e-4)
    for row in positions:
        _ensure_monotonic(row, min_gap=1e-4)
    return positions


def _build_weights(config: SnapshotConfig) -> np.ndarray:
    if config.num_points == 1:
        return np.ones((config.num_steps - 1, 1), dtype=np.float64)

    base_value = (
        config.weight_base if config.weight_base is not None else 1.0 / float(config.num_points)
    )
    offsets = (np.arange(config.num_points) - (config.num_points - 1) / 2.0) * config.weight_epsilon
    weights_row = np.maximum(base_value + offsets, 1e-6)
    weights_row /= weights_row.sum()
    weights = np.tile(weights_row, (config.num_steps - 1, 1))
    return weights


def _logit(array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(array, eps, 1 - eps)
    return np.log(clipped) - np.log1p(-clipped)


def _weight_logits(weights: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    clipped = np.clip(weights, eps, None)
    logits = np.log(clipped)
    logits -= logits.mean(axis=-1, keepdims=True)
    return logits


def _instantiate_predictor(config: SnapshotConfig) -> EPD_predictor:
    predictor = EPD_predictor(
        num_points=config.num_points,
        dataset_name=config.dataset_name,
        img_resolution=config.resolution,
        num_steps=config.num_steps,
        sampler_stu=config.sampler_stu,
        sampler_tea=config.sampler_tea,
        M=config.M,
        guidance_type=config.guidance_type,
        guidance_rate=config.guidance_rate,
        schedule_type=config.schedule_type,
        schedule_rho=config.schedule_rho,
        afs=config.afs,
        scale_dir=config.scale_dir,
        scale_time=config.scale_time,
        fcn=config.fcn,
        max_order=config.max_order,
        predict_x0=config.predict_x0,
        lower_order_final=config.lower_order_final,
        alpha=config.alpha,
        backend=config.backend,
        backend_config=config.backend_options,
    )
    return predictor


def _configure_state(
    predictor: EPD_predictor, positions: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r_params = _logit(positions)
    weight_logits = _weight_logits(weights)
    with torch.no_grad():
        predictor.r_params.copy_(torch.from_numpy(r_params).float())
        predictor.weight_s.copy_(torch.from_numpy(weight_logits).float())
        predictor.scale_dir_params.zero_()
        predictor.scale_time_params.zero_()
    return r_params, weight_logits


def _build_snapshot(
    predictor: EPD_predictor, config: SnapshotConfig, summary_path: Path | None, extras: dict
) -> dict:
    predictor = predictor.cpu().eval()
    snapshot: dict = {"model": predictor}

    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "num_steps": config.num_steps,
            "num_points": config.num_points,
            "sampler_stu": config.sampler_stu,
            "sampler_tea": config.sampler_tea,
            "guidance_type": config.guidance_type,
            "guidance_rate": config.guidance_rate,
            "schedule_type": config.schedule_type,
            "schedule_rho": config.schedule_rho,
            "dataset_name": config.dataset_name,
            "img_resolution": config.resolution,
            "scale_dir": config.scale_dir,
            "scale_time": config.scale_time,
            "fcn": config.fcn,
            "afs": config.afs,
            "alpha": config.alpha,
            "backend": config.backend,
            "backend_config": config.backend_options,
            "sigma_min": config.sigma_min,
            "sigma_max": config.sigma_max,
            "flowmatch_mu": config.flowmatch_mu,
            "flowmatch_shift": config.flowmatch_shift,
            **extras,
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
    return snapshot


def main() -> None:
    args = parse_args()
    _validate_args(args)
    optimizer_betas = _parse_betas(args.optimizer_betas)
    backend_options = _parse_backend_options(args.backend_options)
    if not isinstance(backend_options, dict):
        backend_options = {}
    else:
        backend_options = dict(backend_options)
    if args.sigma_min is not None:
        backend_options.setdefault("sigma_min", args.sigma_min)
    if args.sigma_max is not None:
        backend_options.setdefault("sigma_max", args.sigma_max)
    if args.flowmatch_mu is not None:
        backend_options.setdefault("flowmatch_mu", args.flowmatch_mu)
    if args.flowmatch_shift is not None:
        backend_options.setdefault("flowmatch_shift", args.flowmatch_shift)
    backend_options.setdefault("resolution", args.resolution)

    outdir_raw = Path(args.outdir)
    outdir = outdir_raw.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    filename = (
        args.snapshot_name
        if args.snapshot_name
        else f"network-snapshot-{args.snapshot_step:06d}.pkl"
    )
    snapshot_path = outdir / filename

    config = SnapshotConfig(
        num_steps=args.num_steps,
        num_points=args.num_points,
        sampler_stu=args.sampler_stu,
        sampler_tea=args.sampler_tea,
        M=args.M,
        guidance_type=args.guidance_type,
        guidance_rate=args.guidance_rate,
        schedule_type=args.schedule_type,
        schedule_rho=args.schedule_rho,
        dataset_name=args.dataset_name,
        afs=args.afs,
        scale_dir=args.scale_dir,
        scale_time=args.scale_time,
        fcn=args.fcn,
        max_order=args.max_order,
        predict_x0=args.predict_x0,
        lower_order_final=args.lower_order_final,
        alpha=args.alpha,
        r_base=args.r_base,
        r_epsilon=args.r_epsilon,
        weight_base=args.weight_base,
        weight_epsilon=args.weight_epsilon,
        backend=args.backend,
        backend_options=backend_options,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        flowmatch_mu=args.flowmatch_mu,
        flowmatch_shift=args.flowmatch_shift,
        resolution=args.resolution,
    )

    positions = _build_positions(config)
    weights = _build_weights(config)

    predictor = _instantiate_predictor(config)
    r_params, weight_logits = _configure_state(predictor, positions, weights)
    # Persist scheduler/back-end hints on the module for downstream tooling.
    predictor.sigma_min = config.sigma_min
    predictor.sigma_max = config.sigma_max
    predictor.flowmatch_mu = config.flowmatch_mu
    predictor.flowmatch_shift = config.flowmatch_shift

    extras = {
        "r_mean": positions.mean().item(),
        "weight_mean": weights.mean().item(),
        "r_params_norm": float(np.linalg.norm(r_params)),
        "weight_logits_norm": float(np.linalg.norm(weight_logits)),
        "seed": args.seed,
    }

    snapshot = _build_snapshot(predictor, config, args.summary_json, extras)

    with snapshot_path.open("wb") as handle:
        pickle.dump(snapshot, handle)

    training_options_path: Optional[Path] = None
    if not args.skip_training_options:
        target_path = args.training_options or (outdir / "training_options.json")
        training_options_path = Path(target_path).expanduser()
        training_options_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path_str = args.prompt_path.as_posix() if args.prompt_path else None
        run_dir_display = _format_run_dir(outdir_raw, args.run_dir)
        training_options = {
            "loss_kwargs": {
                "class_name": "training.loss.EPD_loss",
            },
            "pred_kwargs": {
                "class_name": "training.networks.EPD_predictor",
                "num_steps": config.num_steps,
                "sampler_stu": config.sampler_stu,
                "sampler_tea": config.sampler_tea,
                "M": config.M,
                "guidance_type": config.guidance_type,
                "guidance_rate": config.guidance_rate,
                "schedule_rho": config.schedule_rho,
                "schedule_type": config.schedule_type,
                "afs": config.afs,
                "dataset_name": config.dataset_name,
                "scale_dir": config.scale_dir,
                "scale_time": config.scale_time,
                "num_points": config.num_points,
                "fcn": config.fcn,
                "alpha": config.alpha,
                "max_order": config.max_order,
                "predict_x0": config.predict_x0,
                "lower_order_final": config.lower_order_final,
                "backend": config.backend,
                "backend_config": config.backend_options,
            },
            "optimizer_kwargs": {
                "class_name": "torch.optim.Adam",
                "lr": args.optimizer_lr,
                "betas": [optimizer_betas[0], optimizer_betas[1]],
                "eps": args.optimizer_eps,
            },
            "cos_lr_schedule": args.cos_lr_schedule,
            "alpha": config.alpha,
            "total_kimg": args.total_kimg,
            "kimg_per_tick": args.kimg_per_tick,
            "snapshot_ticks": args.snapshot_ticks,
            "state_dump_ticks": args.state_dump_ticks,
            "dataset_name": config.dataset_name,
            "batch_size": args.batch_size,
            "batch_gpu": args.batch_gpu,
            "gpus": args.gpus,
            "cudnn_benchmark": args.cudnn_benchmark,
            "guidance_type": config.guidance_type,
            "guidance_rate": config.guidance_rate,
            "prompt_path": prompt_path_str,
            "seed": args.seed,
            "run_dir": run_dir_display,
        }
        with training_options_path.open("w", encoding="utf-8") as handle:
            json.dump(training_options, handle, indent=2, ensure_ascii=False)

    if args.write_log:
        log_path = outdir / "log.txt"
        log_lines = [
            "fake_train: generated snapshot with custom parameters.\n",
            f"snapshot: {snapshot_path.name}\n",
            f"num_steps={config.num_steps}, num_points={config.num_points}, seed={args.seed}\n",
        ]
        with log_path.open("w", encoding="utf-8") as handle:
            handle.writelines(log_lines)

    print(f"[fake_train] Wrote snapshot to {snapshot_path}")
    if training_options_path is not None:
        print(f"[fake_train] Wrote training options to {training_options_path}")
    if args.write_log:
        print(f"[fake_train] Wrote log to {outdir / 'log.txt'}")


if __name__ == "__main__":
    main()


'''


python fake_train.py \
  --outdir exps/fake-sd3-15 \
  --num-steps 15 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}'

python fake_train.py \
  --outdir exps/fake-sd3-9 \
  --num-steps 9 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}'

python fake_train.py \
  --outdir exps/fake-sd35-15 \
  --num-steps 15 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3.5-medium"}'

python fake_train.py \
  --outdir exps/fake-sd35-9 \
  --num-steps 9 \
  --num-points 2 \
  --guidance-rate 4.5 \
  --schedule-type flowmatch \
  --backend sd3 \
  --backend-options '{"model_name_or_path":"stabilityai/stable-diffusion-3.5-medium"}'
'''
