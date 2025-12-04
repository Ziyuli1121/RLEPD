#!/usr/bin/env python3
"""
Minimal SD3 baseline sampler using the official diffusers FlowMatch Euler scheduler.

This mirrors `sample_baseline.py` 的输出结构：按 seed 保存 PNG，可选 grid 输出。
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import List, Sequence

import click
import torch
from torchvision.utils import make_grid, save_image

from sample import create_model_sd3
from training.loss import get_solver_fn


# -----------------------------------------------------------------------------
# Helpers (trimmed版本)


class StackedRandomGenerator:
    def __init__(self, device: torch.device, seeds: Sequence[int]):
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
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
        # 如果提供的行少于请求数量，则循环使用
        reps = (count + len(lines) - 1) // len(lines)
        lines = (lines * reps)[:count]
        return lines
    # 默认空 prompt
    return [""] * count


# -----------------------------------------------------------------------------
# CLI


@click.command()
@click.option(
    "--sampler",
    type=click.Choice(["sd3", "flowmatch", "edm", "dpm", "dpm2", "ipndm"], case_sensitive=False),
    default="sd3",
    show_default=True,
    help="Sampler backend (flowmatch=official SD3 Euler; edm/dpm/dpm2/ipndm use RLEPD solvers).",
)
@click.option("--num-steps", type=int, default=28, show_default=True, help="Number of inference steps.")
@click.option("--batch", "max_batch_size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--seeds", type=parse_int_list, default="0-3", show_default=True, help="Random seeds list/range.")
@click.option("--prompt", type=str, default=None, help="Single prompt to use for all seeds.")
@click.option(
    "--prompt-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Text file or CSV with 'text' column.",
)
@click.option(
    "--model-id",
    type=str,
    default="stabilityai/stable-diffusion-3-medium-diffusers",
    show_default=True,
    help="Diffusers model repo or local path (e.g., SD3 or SD3.5).",
)
@click.option("--outdir", type=str, default="./samples/sd3", show_default=True, help="Output directory.")
@click.option("--grid", type=bool, default=False, help="Whether to save a grid image per batch.")
@click.option(
    "--subdirs",
    is_flag=True,
    default=True,
    help="Create subdirectory for every 1000 seeds (mirrors sample_baseline).",
)
@click.option("--schedule-type", type=str, default="flowmatch", show_default=True, help="Time schedule type.")
@click.option("--schedule-rho", type=float, default=7.0, show_default=True, help="Schedule rho/exponent.")
@click.option("--guidance-rate", type=float, default=4.5, show_default=True, help="Classifier-free guidance scale.")
@click.option("--afs", is_flag=True, default=False, help="Apply AFS on first step (for dpm/ipndm).")
@click.option("--max-order", type=int, default=3, show_default=True, help="Max order for IPNDM.")
@click.option("--inner-steps", type=int, default=None, help="Inner steps for DPM (default 2).")
@click.option(
    "--resolution",
    type=click.Choice(["512", "1024"], case_sensitive=False),
    default="1024",
    show_default=True,
    help="Square image resolution.",
)
def main(
    sampler: str,
    num_steps: int,
    max_batch_size: int,
    seeds: List[int],
    prompt: str | None,
    prompt_file: str | None,
    outdir: str,
    grid: bool,
    subdirs: bool,
    model_id: str,
    schedule_type: str,
    schedule_rho: float,
    guidance_rate: float,
    afs: bool,
    max_order: int,
    inner_steps: int | None,
    resolution: str,
) -> None:
    sampler_choice = sampler.lower()
    sampler_mode = "flowmatch" if sampler_choice in ["sd3", "flowmatch"] else sampler_choice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_resolution = int(resolution)

    backend_cfg = {"model_id": model_id, "model_name_or_path": model_id, "resolution": target_resolution}
    backend, _ = create_model_sd3(
        guidance_rate=guidance_rate,
        device=device,
        backend_config=backend_cfg,
    )
    if not hasattr(backend, "round_sigma"):
        backend.round_sigma = lambda x: x  # EDM fallback for SD3 backend
    pipe = backend.pipeline
    pipe.set_progress_bar_config(disable=True)

    solver_fn = None if sampler_mode == "flowmatch" else get_solver_fn(sampler_mode)
    if sampler_mode == "edm" and schedule_rho == 1.0:
        schedule_rho = 7.0

    seeds_tensor = torch.as_tensor(seeds)
    num_batches = ((len(seeds) - 1) // max_batch_size) + 1
    all_batches = seeds_tensor.tensor_split(num_batches)

    prompts = _load_prompts(prompt, prompt_file, len(seeds))

    os.makedirs(outdir, exist_ok=True)

    for batch_idx, batch_seeds in enumerate(all_batches):
        batch_prompts = prompts[
            batch_idx * batch_seeds.numel() : batch_idx * batch_seeds.numel() + batch_seeds.numel()
        ]
        if sampler_mode == "flowmatch":
            gen = [torch.Generator(device).manual_seed(int(s.item()) % (1 << 32)) for s in batch_seeds]
            with torch.no_grad(), torch.cuda.amp.autocast():  # autocast no-op on CPU
                result = pipe(
                    prompt=batch_prompts,
                    negative_prompt=[""] * len(batch_prompts),
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_rate,
                    generator=gen,
                    height=target_resolution,
                    width=target_resolution,
                    output_type="pt",
                )
            images = torch.clamp(result.images, 0, 1)
        else:
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn(
                [len(batch_seeds), backend.img_channels, backend.img_resolution, backend.img_resolution],
                device=device,
                dtype=pipe.transformer.dtype,
            )
            negative_prompt = [""] * len(batch_prompts) if guidance_rate > 1.0 else None
            condition = backend.prepare_condition(
                prompt=batch_prompts,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_rate,
            )
            solver_kwargs = dict(
                num_steps=num_steps,
                sigma_min=backend.sigma_min,
                sigma_max=backend.sigma_max,
                schedule_type=schedule_type,
                schedule_rho=schedule_rho,
                afs=afs,
                predictor=None,
                train=False,
            )
            if sampler_mode == "dpm":
                solver_kwargs["inner_steps"] = inner_steps if inner_steps is not None else 2
            if sampler_mode == "ipndm":
                solver_kwargs["max_order"] = max_order

            with torch.no_grad():
                samples, _ = solver_fn(
                    net=backend,
                    latents=latents,
                    condition=condition,
                    **solver_kwargs,
                )
                images = backend.vae_decode(samples)
                images = torch.clamp(images / 2 + 0.5, 0, 1)

        if grid:
            grid_img = make_grid(images, nrow=int(len(images) ** 0.5) or 1, padding=0)
            save_image(grid_img, os.path.join(outdir, f"grid_batch{batch_idx:04d}.png"))

        images_np = (images * 255).round().to(torch.uint8)
        images_np = images_np.permute(0, 2, 3, 1).cpu().numpy()
        for seed_val, image_np in zip(batch_seeds.tolist(), images_np):
            image_dir = (
                os.path.join(outdir, f"{seed_val - seed_val % 1000:06d}") if subdirs else outdir
            )
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f"{seed_val:06d}.png")
            from PIL import Image

            Image.fromarray(image_np, "RGB").save(image_path)

    print("SD3 baseline sampling done.")


if __name__ == "__main__":
    main()
