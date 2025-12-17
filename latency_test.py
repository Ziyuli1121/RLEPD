#!/usr/bin/env python3
"""
Benchmark EPD parallel sampler latency for sd1.5 and sd3 predictors.

Usage example:
    python latency_test.py \\
        --predictor exps/latency/sd3-512_k2_nfe16/network-snapshot-000005.pkl \\
        --prompt-file ./prompts.txt \\
        --seeds 0-7 \\
        --outdir ./latency_runs/sd3_k2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import List, Sequence
from contextlib import nullcontext

import torch
from torchvision.utils import make_grid, save_image

from sample import create_model_backend
import solvers
from solvers import epd_parallel_sampler
from training.networks import EPD_predictor


# -----------------------------------------------------------------------------
# Helpers


class StackedRandomGenerator:
    """Batch-aware RNG so each seed owns its own generator."""

    def __init__(self, device: torch.device, seeds: Sequence[int]):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])


def parse_int_list(s: str | Sequence[int]):
    if isinstance(s, list):
        return s
    ranges: list[int] = []
    range_re = None
    import re

    range_re = re.compile(r"^(\d+)-(\d+)$")
    for part in s.split(","):
        m = range_re.match(part)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(part))
    return ranges


def _load_prompts(prompt: str | None, prompt_file: str | None, count: int) -> List[str]:
    if prompt is not None:
        return [prompt] * count
    if prompt_file:
        path = Path(prompt_file)
        lines: List[str] = []
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "text" in row and row["text"].strip():
                        lines.append(row["text"].strip())
        else:
            with path.open("r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        if not lines:
            raise RuntimeError(f"No prompts found in '{prompt_file}'.")
        reps = (count + len(lines) - 1) // len(lines)
        lines = (lines * reps)[:count]
        return lines
    return [""] * count


def _load_predictor(path: Path, device: torch.device) -> EPD_predictor:
    import pickle

    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    predictor = snapshot["model"].to(device).eval()
    return predictor


def _prepare_condition_sd15(net, prompts: List[str], guidance_rate: float):
    c = net.model.get_learned_conditioning(prompts)
    uc = None
    if guidance_rate != 1.0:
        uc = net.model.get_learned_conditioning([""] * len(prompts))
    return c, uc


def _prepare_condition_sd3(net, prompts: List[str], guidance_rate: float, backend_config: dict):
    if guidance_rate == 1.0:
        negative_prompt = None
    else:
        base_negative = backend_config.get("negative_prompt", "")
        if isinstance(base_negative, list):
            if len(base_negative) != len(prompts):
                raise ValueError("Length of negative_prompt list must match batch size.")
            negative_prompt = base_negative
        else:
            negative_prompt = [str(base_negative)] * len(prompts)
    return net.prepare_condition(prompt=prompts, negative_prompt=negative_prompt, guidance_scale=guidance_rate)


def _filter_outliers(values: list[float]) -> tuple[list[float], dict | None]:
    """Use IQR to drop outliers before computing summary stats."""
    if len(values) < 4:
        return values, None
    t = torch.tensor(values)
    q1, q3 = torch.quantile(t, torch.tensor([0.25, 0.75], device=t.device))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    mask = (t >= lower) & (t <= upper)
    filtered = t[mask].tolist()
    meta = {
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "removed": int((~mask).sum().item()),
        "kept": int(mask.sum().item()),
    }
    return filtered, meta


# -----------------------------------------------------------------------------
# Main


def main() -> None:
    parser = argparse.ArgumentParser(description="Latency benchmark using epd_parallel_sampler.")
    parser.add_argument("--predictor", type=Path, required=True, help="EPD predictor .pkl (fake_train output).")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for all seeds.")
    parser.add_argument("--prompt-file", type=str, default=None, help="Text/CSV prompt file.")
    parser.add_argument("--seeds", type=parse_int_list, default="0-3", help="Seed list/range (e.g. 0-7).")
    parser.add_argument("--outdir", type=str, default="./latency_samples", help="Output directory.")
    parser.add_argument("--max-batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--grid", action="store_true", help="Save grid per batch.")
    parser.add_argument("--backend", type=str, default=None, help="Optional backend override (ldm/sd3).")
    parser.add_argument("--backend-config", type=str, default=None, help="JSON string to override backend config.")
    parser.add_argument("--latency-json", type=str, default="latency.json", help="Path to write per-image latency records.")
    parser.add_argument(
        "--sampler",
        type=str,
        default="epd_parallel",
        choices=["epd", "epd_parallel"],
        help="Sampler to benchmark; default epd_parallel.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictor_path = args.predictor
    predictor_module = _load_predictor(predictor_path, device=device)

    backend_cfg = {}
    predictor_backend_cfg = getattr(predictor_module, "backend_config", None)
    if isinstance(predictor_backend_cfg, dict):
        backend_cfg = dict(predictor_backend_cfg)
    if args.backend_config:
        try:
            override_cfg = json.loads(args.backend_config)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--backend-config JSON decode failed: {exc}") from exc
        if not isinstance(override_cfg, dict):
            raise SystemExit("--backend-config must decode to a JSON object.")
        backend_cfg.update(override_cfg)

    resolved_backend = (args.backend or getattr(predictor_module, "backend", None) or "ldm").lower()
    dataset_name = getattr(predictor_module, "dataset_name", None) or "ms_coco"
    guidance_rate = predictor_module.guidance_rate

    net, _ = create_model_backend(
        dataset_name=dataset_name,
        guidance_type=predictor_module.guidance_type,
        guidance_rate=guidance_rate,
        backend=resolved_backend,
        backend_config=backend_cfg,
        device=device,
    )

    num_steps = predictor_module.num_steps
    schedule_type = getattr(predictor_module, "schedule_type", None) or "discrete"
    schedule_rho = getattr(predictor_module, "schedule_rho", None) or 1.0
    sigma_min = getattr(predictor_module, "sigma_min", None) or getattr(net, "sigma_min", 0.002)
    sigma_max = getattr(predictor_module, "sigma_max", None) or getattr(net, "sigma_max", 80.0)

    seeds: List[int] = parse_int_list(args.seeds)
    prompts = _load_prompts(args.prompt, args.prompt_file, len(seeds))
    seeds_tensor = torch.as_tensor(seeds)
    num_batches = ((len(seeds) - 1) // args.max_batch_size) + 1
    all_batches = seeds_tensor.tensor_split(num_batches)

    os.makedirs(args.outdir, exist_ok=True)

    # Warm-up to avoid first-iteration latency spikes (CUDA graph init, kernel JIT).
    try:
        warm_latents = torch.randn(
            [1, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
            dtype=getattr(net, "dtype", None) or getattr(getattr(net, "pipeline", None), "dtype", None) or torch.float32,
        )
        if resolved_backend == "sd3":
            warm_condition = _prepare_condition_sd3(net, ["warmup"], guidance_rate, backend_cfg)
            warm_uc = None
        else:
            warm_condition, warm_uc = _prepare_condition_sd15(net, ["warmup"], guidance_rate)
        with torch.no_grad():
            warm_out, _ = epd_parallel_sampler(
                net=net,
                latents=warm_latents,
                condition=warm_condition,
                unconditional_condition=warm_uc,
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                schedule_type=schedule_type,
                schedule_rho=schedule_rho,
                guidance_type=predictor_module.guidance_type,
                guidance_rate=guidance_rate,
                predictor=predictor_module,
                afs=bool(getattr(predictor_module, "afs", False)),
                return_inters=False,
                train=False,
            )
            if resolved_backend == "sd3":
                warm_out = net.vae_decode(warm_out)
            else:
                warm_out = net.model.decode_first_stage(warm_out)
        if device.type == "cuda":
            torch.cuda.synchronize()
        del warm_latents, warm_out
        if resolved_backend == "sd3":
            del warm_condition
        else:
            del warm_condition, warm_uc
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("[latency] warm-up pass completed.")
    except Exception as exc:  # pragma: no cover - best-effort warm-up
        print(f"[latency] warm-up skipped due to: {exc}")

    total_time = 0.0  # sampling-only time
    total_images = 0
    latency_records: list[dict] = []
    per_image_sample_times: list[float] = []
    per_image_decode_times: list[float] = []
    peak_mem_bytes = None

    print("[latency] configuration:")
    print(f"  - predictor: {predictor_path}")
    print(f"  - backend: {resolved_backend}")
    print(f"  - num_steps: {num_steps}")
    print(f"  - num_points: {getattr(predictor_module, 'num_points', None)}")
    print(f"  - schedule_type: {schedule_type}")
    print(f"  - sigma_min: {sigma_min}")
    print(f"  - sigma_max: {sigma_max}")
    print(f"  - guidance_rate: {guidance_rate}")
    print(f"  - device: {device}")

    for batch_idx, batch_seeds in enumerate(all_batches):
        batch_prompts = prompts[batch_idx * batch_seeds.numel() : batch_idx * batch_seeds.numel() + batch_seeds.numel()]
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [len(batch_seeds), net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
            dtype=getattr(net, "dtype", None) or getattr(getattr(net, "pipeline", None), "dtype", None) or torch.float32,
        )

        if resolved_backend == "sd3":
            condition = _prepare_condition_sd3(net, batch_prompts, guidance_rate, backend_cfg)
            unconditional_condition = None
        else:
            condition, unconditional_condition = _prepare_condition_sd15(net, batch_prompts, guidance_rate)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_sample_start = time.perf_counter()

        if device.type == "cuda" and resolved_backend == "ldm":
            autocast_ctx = torch.cuda.amp.autocast()
        else:
            autocast_ctx = nullcontext()
        with torch.no_grad(), autocast_ctx:
            sampler_fn = epd_parallel_sampler if args.sampler == "epd_parallel" else solvers.epd_sampler
            images, _ = sampler_fn(
                net=net,
                latents=latents,
                condition=condition,
                unconditional_condition=unconditional_condition,
                num_steps=num_steps,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                schedule_type=schedule_type,
                schedule_rho=schedule_rho,
                guidance_type=predictor_module.guidance_type,
                guidance_rate=guidance_rate,
                predictor=predictor_module,
                afs=bool(getattr(predictor_module, "afs", False)),
                return_inters=False,
                train=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_time_sample = time.perf_counter() - t_sample_start

        t_decode_start = time.perf_counter()
        with torch.no_grad():
            if resolved_backend == "sd3":
                images = net.vae_decode(images)
            else:
                images = net.model.decode_first_stage(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_time_decode = time.perf_counter() - t_decode_start
        # Only sampling time is accumulated for latency stats; decode is reported separately.
        batch_time = batch_time_sample

        images = torch.clamp(images / 2 + 0.5, 0, 1)

        if args.grid:
            grid_img = make_grid(images, nrow=int(len(images) ** 0.5) or 1, padding=0)
            save_image(grid_img, os.path.join(args.outdir, f"grid_batch{batch_idx:04d}.png"))

        images_np = (images * 255).round().to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        for seed_val, image_np in zip(batch_seeds.tolist(), images_np):
            image_dir = os.path.join(args.outdir, f"{seed_val - seed_val % 1000:06d}")
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{seed_val:06d}.png")
            from PIL import Image

            Image.fromarray(image_np, "RGB").save(image_path)

        total_time += batch_time  # sampling-only
        total_images += len(batch_seeds)
        per_image = batch_time / len(batch_seeds)
        per_image_sample = batch_time_sample / len(batch_seeds)
        per_image_decode = batch_time_decode / len(batch_seeds)
        print(
            f"[latency] batch {batch_idx}: {len(batch_seeds)} images, "
            f"sample {batch_time_sample:.3f}s ({per_image_sample:.4f}s/img), "
            f"decode {batch_time_decode:.3f}s ({per_image_decode:.4f}s/img)"
        )
        for seed_val, prompt_str in zip(batch_seeds.tolist(), batch_prompts):
            latency_records.append(
                {
                    "seed": int(seed_val),
                    "prompt": prompt_str,
                    "batch_time_sample_sec": batch_time_sample,
                    "batch_time_decode_sec": batch_time_decode,
                    "per_image_time_sample_sec": per_image_sample,
                    "per_image_time_decode_sec": per_image_decode,
                }
            )
            per_image_sample_times.append(per_image_sample)
            per_image_decode_times.append(per_image_decode)

    filtered_sample_times, filter_meta_sample = _filter_outliers(per_image_sample_times)
    raw_avg = total_time / total_images if total_images else None
    raw_std = float(torch.tensor(per_image_sample_times).std().item()) if per_image_sample_times else None
    filtered_avg = float(torch.tensor(filtered_sample_times).mean().item()) if filtered_sample_times else None
    filtered_std = float(torch.tensor(filtered_sample_times).std().item()) if filtered_sample_times else None
    raw_sample_avg = raw_avg
    raw_sample_std = raw_std
    filtered_sample_avg = filtered_avg
    filtered_sample_std = filtered_std
    if torch.cuda.is_available():
        peak_mem_bytes = torch.cuda.max_memory_allocated(device)
        # reset peak for potential future runs
        torch.cuda.reset_peak_memory_stats(device)

    if total_images > 0:
        print(f"[latency] overall (sampling raw): {total_images} images, {total_time:.3f}s total, avg {raw_avg:.4f}s/image")
        if filtered_sample_times and (len(filtered_sample_times) != len(per_image_sample_times)):
            print(
                f"[latency] overall (sampling filtered): kept {len(filtered_sample_times)}/{len(per_image_sample_times)} images, "
                f"avg {filtered_sample_avg:.4f}s/image, std {filtered_sample_std:.4f}s"
            )
    else:
        print("[latency] no images generated; check seeds/prompt inputs.")

    if latency_records:
        latency_path = Path(args.latency_json)
        latency_path.parent.mkdir(parents=True, exist_ok=True)
        with latency_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "predictor": str(predictor_path),
                    "backend": resolved_backend,
                    "num_steps": num_steps,
                    "num_points": getattr(predictor_module, "num_points", None),
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                    "schedule_type": schedule_type,
                    "schedule_rho": schedule_rho,
                    "guidance_rate": guidance_rate,
                    "device": str(device),
                    "total_images": total_images,
                    "total_time_sec": total_time,
                    "raw_avg_time_per_image_sec": raw_avg,
                    "raw_std_time_per_image_sec": raw_std,
                    "raw_sample_avg_time_per_image_sec": raw_sample_avg,
                    "raw_sample_std_time_per_image_sec": raw_sample_std,
                    "avg_time_per_image_sec": filtered_avg,
                    "std_time_per_image_sec": filtered_std,
                    "avg_sample_time_per_image_sec": filtered_sample_avg,
                    "std_sample_time_per_image_sec": filtered_sample_std,
                    "outlier_filter_sample": filter_meta_sample,
                    "peak_memory_bytes": int(peak_mem_bytes) if peak_mem_bytes is not None else None,
                    "records": latency_records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[latency] wrote latency JSON to {latency_path}")


if __name__ == "__main__":
    main()


'''

python latency_test.py \
  --predictor exps/latency/sd3-512_k2_nfe16/network-snapshot-000005.pkl \
  --prompt-file src/prompts/test.txt \
  --seeds 0-7 \
  --max-batch-size 1 \
  --outdir ./latency_runs/sd3-512_k2_nfe16 \
  --latency-json ./latency_runs/sd3-512_k2_nfe16/latency.json


'''
