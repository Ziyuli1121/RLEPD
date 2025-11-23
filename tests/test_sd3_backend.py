#!/usr/bin/env python3
"""
Self-test for the SD3 backend adapter.

This script exercises a single forward step and prompt encoding for the
new diffusers-based backend (integration plan Step 2).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.backends import SD3DiffusersBackend


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SD3 backend smoke test.")
    parser.add_argument(
        "--model",
        default=os.environ.get("SD3_MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers"),
        help="Model name or local path for Stable Diffusion 3 weights.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the pipeline (default: auto-detect).",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Force a dtype; 'auto' selects fp16 on CUDA and fp32 otherwise.",
    )
    parser.add_argument(
        "--default-guidance-scale",
        type=float,
        default=7.5,
        help="Default CFG scale used when preparing conditions without overrides.",
    )
    parser.add_argument(
        "--guidance-scales",
        type=str,
        default="1.0,7.5",
        help="Comma-separated list of guidance scales to exercise.",
    )
    parser.add_argument("--prompt", type=str, default="A photorealistic cat", help="Prompt used for the smoke test.")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="blurry, low quality",
        help="Negative prompt used when CFG is enabled.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the latent test tensor.")
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.8,
        help="Flow-matching timestep used for the single forward step.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable diffusers sequential CPU offload to reduce VRAM usage.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision identifier.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Optional model variant (e.g., fp16).",
    )
    parser.add_argument(
        "--no-safetensors",
        action="store_false",
        dest="use_safetensors",
        help="Disable safetensors checkpoints if the environment lacks support.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token if login via CLI is not configured.",
    )
    parser.set_defaults(use_safetensors=True)
    return parser.parse_args()


def _resolve_dtype(name: str, device: str) -> torch.dtype:
    if name == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _parse_guidance_scales(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("guidance_scales must contain at least one float.")
    return values


@torch.no_grad()
def _run_case(
    backend: SD3DiffusersBackend,
    prompt: str,
    negative_prompt: str,
    guidance_scale: float,
    batch_size: int,
    timestep: float,
) -> None:
    exec_device = backend.pipeline._execution_device
    latent_dtype = backend.pipeline.transformer.dtype

    condition = backend.prepare_condition(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
    )
    assert condition.guidance_scale == guidance_scale

    latents = torch.randn(
        batch_size,
        backend.img_channels,
        backend.img_resolution,
        backend.img_resolution,
        device=exec_device,
        dtype=latent_dtype,
    )

    timestep_tensor = torch.full((batch_size,), float(timestep), device=exec_device, dtype=latent_dtype)
    denoised = backend(latents, timestep_tensor, condition=condition)

    if denoised.shape != latents.shape:
        raise AssertionError(f"Denoised tensor shape {denoised.shape} != latents shape {latents.shape}")

    decoded = backend.vae_decode(latents[:1].clone())
    if decoded.dim() != 4:
        raise AssertionError(f"Decoded tensor must be NCHW, got shape {decoded.shape}")

    print(
        f"[OK] guidance_scale={guidance_scale} "
        f"| latents {tuple(latents.shape)} "
        f"| denoised stats (min={denoised.min().item():.3f}, max={denoised.max().item():.3f})"
    )
    schedule = backend.make_flowmatch_schedule(num_steps=3, device=exec_device)
    print(f"[OK] flowmatch schedule shape={tuple(schedule.shape)} first={schedule[0].item():.4f}")


def main() -> None:
    args = _parse_args()
    dtype = _resolve_dtype(args.torch_dtype, args.device)
    guidance_scales = _parse_guidance_scales(args.guidance_scales)

    backend = SD3DiffusersBackend(
        model_name_or_path=args.model,
        device=args.device,
        torch_dtype=dtype,
        guidance_scale=args.default_guidance_scale,
        enable_model_cpu_offload=args.enable_cpu_offload,
        revision=args.revision,
        variant=args.variant,
        use_safetensors=args.use_safetensors,
        token=args.token,
    )

    for scale in guidance_scales:
        _run_case(
            backend=backend,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_scale=scale,
            batch_size=args.batch_size,
            timestep=args.timestep,
        )

    print("SD3 backend smoke test finished successfully.")


if __name__ == "__main__":
    main()
