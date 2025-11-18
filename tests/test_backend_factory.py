#!/usr/bin/env python3
"""
Smoke test for the backend-aware model factory (Step 3 of the integration guide).

Usage examples:
    python tests/test_backend_factory.py --backend ldm
    python tests/test_backend_factory.py --backend sd3 --backend-config '{"model_name_or_path":"stabilityai/stable-diffusion-3-medium-diffusers"}'
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sample import create_model_backend


def _parse_backend_config(raw: str | None) -> Dict[str, Any]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"--backend-config must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--backend-config must decode to a JSON object.")
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backend factory smoke test.")
    parser.add_argument("--backend", type=str, default="ldm", help="Backend identifier (e.g., ldm or sd3).")
    parser.add_argument(
        "--backend-config",
        type=str,
        default=None,
        help="JSON object with backend-specific overrides (model IDs, tokens, etc.).",
    )
    parser.add_argument("--dataset-name", type=str, default="ms_coco")
    parser.add_argument("--guidance-type", type=str, default="cfg")
    parser.add_argument("--guidance-rate", type=float, default=7.5)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prompt", type=str, default="A scenic landscape", help="Prompt used when exercising SD3.")
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default="low quality",
        help="Negative prompt used when CFG > 1.0 (SD3 only).",
    )
    parser.add_argument("--timestep", type=float, default=0.8, help="Timestep used for the SD3 forward pass.")
    return parser


def _run_sd3_forward(net, prompt: str, negative_prompt: str, guidance_rate: float, timestep: float) -> None:
    prompts = [prompt]
    negative = None
    if guidance_rate != 1.0:
        negative = [negative_prompt]
    condition = net.prepare_condition(
        prompt=prompts,
        negative_prompt=negative,
        guidance_scale=guidance_rate,
    )
    latents = torch.randn(
        1,
        net.img_channels,
        net.img_resolution,
        net.img_resolution,
        device=net.pipeline._execution_device,
        dtype=net.pipeline.transformer.dtype,
    )
    t = torch.full((1,), float(timestep), device=latents.device, dtype=latents.dtype)
    denoised = net(latents, t, condition=condition)
    print(
        f"[SD3] forward ok | shape={tuple(denoised.shape)} "
        f"| stats min={denoised.min().item():.3f} max={denoised.max().item():.3f}"
    )


def _inspect_ldm_backend(net) -> None:
    test_prompt = ["A quick factory smoke test"]
    condition = net.model.get_learned_conditioning(test_prompt).to(net.model.device)
    t = torch.ones(1, device=condition.device) * 1.0
    latents = torch.randn(1, net.img_channels, net.img_resolution, net.img_resolution, device=condition.device)
    out = net(latents, t, condition=condition, unconditional_condition=None)
    print(f"[LDM] forward ok | shape={tuple(out.shape)}")
    decoded = net.model.decode_first_stage(out)
    print(f"[LDM] decode ok | shape={tuple(decoded.shape)}")


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    backend_config = _parse_backend_config(args.backend_config)
    device = torch.device(args.device)

    net, model_source = create_model_backend(
        dataset_name=args.dataset_name,
        guidance_type=args.guidance_type,
        guidance_rate=args.guidance_rate,
        backend=args.backend,
        backend_config=backend_config,
        device=device,
    )
    print(f"Loaded backend '{model_source}' | resolution={net.img_resolution} | channels={net.img_channels}")

    if model_source == "sd3":
        _run_sd3_forward(
            net=net,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            guidance_rate=args.guidance_rate,
            timestep=args.timestep,
        )
    elif model_source == "ldm":
        _inspect_ldm_backend(net)

    print("Backend factory smoke test finished.")


if __name__ == "__main__":
    main()
