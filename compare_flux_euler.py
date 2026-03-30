#!/usr/bin/env python3
"""
Internal debug helper to compare:

- official diffusers FLUX Euler baseline
- RLEPD `ddim_sampler` under `schedule_type=flowmatch`

This script is intentionally diagnostic. It does not change public sampling
interfaces and is only used to make the equivalence discussion concrete.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import click
import torch
from torchvision.utils import save_image

from sample import _prepare_flux_condition, create_model_flux
from solvers import ddim_sampler
from training.ppo.pipeline_utils import load_prompts_file


def _resolve_prompt(prompt: Optional[str], prompt_file: Optional[str]) -> str:
    if prompt is not None:
        return prompt
    if prompt_file:
        prompts = load_prompts_file(prompt_file)
        if not prompts:
            raise ValueError(f"No prompts found in {prompt_file}")
        return prompts[0]
    raise ValueError("Either --prompt or --prompt-file must be provided.")


def _shared_latents(
    backend,
    *,
    seed: int,
) -> torch.Tensor:
    device = backend.pipeline._execution_device
    dtype = backend.pipeline.transformer.dtype
    generator = torch.Generator(device=device).manual_seed(int(seed) % (1 << 32))
    return torch.randn(
        [1, backend.img_channels, backend.img_resolution, backend.img_resolution],
        generator=generator,
        device=device,
        dtype=dtype,
    )


def _official_schedule(backend, *, num_steps: int) -> Dict[str, torch.Tensor]:
    scheduler = backend.pipeline.scheduler
    mu = backend.resolve_flowmatch_mu()
    scheduler.set_timesteps(
        num_inference_steps=num_steps,
        device=backend.pipeline._execution_device,
        mu=mu,
    )
    return {
        "mu": torch.tensor(float(mu) if mu is not None else float("nan")),
        "timesteps": scheduler.timesteps.detach().clone(),
        "sigmas": scheduler.sigmas.detach().clone(),
    }


def _run_official_flux(
    backend,
    *,
    prompt: str,
    guidance_rate: float,
    num_steps: int,
    latents_4d: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    pipe = backend.pipeline
    packed_latents = pipe._pack_latents(
        latents_4d,
        latents_4d.shape[0],
        backend.img_channels,
        latents_4d.shape[2],
        latents_4d.shape[3],
    )
    with torch.no_grad():
        result = pipe(
            prompt=[prompt],
            num_inference_steps=num_steps,
            guidance_scale=guidance_rate,
            latents=packed_latents,
            height=backend.output_resolution,
            width=backend.output_resolution,
            output_type="latent",
        )
    packed_final = result.images.detach()
    latent_map = backend._unpack_latents(packed_final, backend.img_resolution, backend.img_resolution)
    image = torch.clamp(backend.vae_decode(latent_map) / 2 + 0.5, 0, 1)
    return {
        "packed_latents": packed_final,
        "latents": latent_map,
        "image": image,
    }


def _run_project_ddim(
    backend,
    *,
    prompt: str,
    guidance_rate: float,
    num_steps: int,
    latents_4d: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    backend_cfg = dict(getattr(backend, "backend_config", {}) or {})
    condition = _prepare_flux_condition(backend, [prompt], guidance_rate, backend_cfg)
    with torch.no_grad():
        samples, _ = ddim_sampler(
            net=backend,
            latents=latents_4d.clone(),
            condition=condition,
            num_steps=num_steps,
            sigma_min=backend.sigma_min,
            sigma_max=backend.sigma_max,
            schedule_type="flowmatch",
            schedule_rho=1.0,
            afs=False,
            ddim_eta=0.0,
            train=False,
        )
    image = torch.clamp(backend.vae_decode(samples) / 2 + 0.5, 0, 1)
    return {"latents": samples, "image": image}


def _replay_project_ddim_with_schedule(
    backend,
    *,
    prompt: str,
    guidance_rate: float,
    t_steps: torch.Tensor,
    latents_4d: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    backend_cfg = dict(getattr(backend, "backend_config", {}) or {})
    condition = _prepare_flux_condition(backend, [prompt], guidance_rate, backend_cfg)
    x_next = latents_4d.clone()
    with torch.no_grad():
        for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
            denoised = backend(x_next, t_cur, condition=condition)
            d_cur = (x_next - denoised) / t_cur
            x_next = x_next + (t_next - t_cur) * d_cur
    image = torch.clamp(backend.vae_decode(x_next) / 2 + 0.5, 0, 1)
    return {"latents": x_next, "image": image}


def _tensor_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    diff = (a.float() - b.float()).detach()
    return {
        "mae": float(diff.abs().mean().item()),
        "rmse": float(diff.pow(2).mean().sqrt().item()),
        "max_abs": float(diff.abs().max().item()),
    }


@click.command()
@click.option("--model-id", type=str, required=True, help="Local FLUX snapshot or repo id.")
@click.option("--prompt", type=str, default=None, help="Single prompt to compare.")
@click.option(
    "--prompt-file",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Optional text/CSV prompt file; the first prompt is used.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--num-steps", type=int, default=28, show_default=True)
@click.option("--guidance-rate", type=float, default=3.5, show_default=True)
@click.option("--outdir", type=click.Path(file_okay=False), default=None, help="Optional directory to save comparison images/report.")
def main(
    model_id: str,
    prompt: Optional[str],
    prompt_file: Optional[str],
    seed: int,
    num_steps: int,
    guidance_rate: float,
    outdir: Optional[str],
) -> None:
    prompt_text = _resolve_prompt(prompt, prompt_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend_cfg = {
        "model_id": model_id,
        "model_name_or_path": model_id,
        "resolution": 1024,
        "torch_dtype": "bfloat16",
        "enable_model_cpu_offload": False,
    }
    backend, _ = create_model_flux(
        dataset_name="ms_coco",
        guidance_type="cfg",
        guidance_rate=guidance_rate,
        device=device,
        backend_config=backend_cfg,
    )
    latents_4d = _shared_latents(backend, seed=seed)

    official_schedule = _official_schedule(backend, num_steps=num_steps)
    official = _run_official_flux(
        backend,
        prompt=prompt_text,
        guidance_rate=guidance_rate,
        num_steps=num_steps,
        latents_4d=latents_4d,
    )
    project_ddim = _run_project_ddim(
        backend,
        prompt=prompt_text,
        guidance_rate=guidance_rate,
        num_steps=num_steps,
        latents_4d=latents_4d,
    )

    project_sigmas = backend.make_flowmatch_schedule(num_steps, device=latents_4d.device).detach().clone()
    project_sigmas_with_terminal = torch.cat([project_sigmas, torch.zeros_like(project_sigmas[:1])])
    project_ddim_terminal = _replay_project_ddim_with_schedule(
        backend,
        prompt=prompt_text,
        guidance_rate=guidance_rate,
        t_steps=project_sigmas_with_terminal,
        latents_4d=latents_4d,
    )

    report = {
        "model_id": model_id,
        "prompt": prompt_text,
        "seed": int(seed),
        "guidance_rate": float(guidance_rate),
        "num_steps": int(num_steps),
        "schedule": {
            "official_num_timesteps": int(official_schedule["timesteps"].numel()),
            "official_num_sigmas": int(official_schedule["sigmas"].numel()),
            "official_has_terminal_zero": bool(abs(float(official_schedule["sigmas"][-1])) < 1e-12),
            "official_mu": float(official_schedule["mu"].item()),
            "project_num_sigmas": int(project_sigmas.numel()),
            "project_updates": int(max(project_sigmas.numel() - 1, 0)),
            "project_terminal_updates": int(max(project_sigmas_with_terminal.numel() - 1, 0)),
            "sigma_match_without_terminal": _tensor_metrics(official_schedule["sigmas"][:-1], project_sigmas),
        },
        "comparison": {
            "official_vs_project_ddim_latents": _tensor_metrics(official["latents"], project_ddim["latents"]),
            "official_vs_project_ddim_images": _tensor_metrics(official["image"], project_ddim["image"]),
            "official_vs_project_ddim_with_terminal_latents": _tensor_metrics(
                official["latents"], project_ddim_terminal["latents"]
            ),
            "official_vs_project_ddim_with_terminal_images": _tensor_metrics(
                official["image"], project_ddim_terminal["image"]
            ),
        },
        "interpretation": {
            "official_reference_solver": "diffusers FluxPipeline + FlowMatchEulerDiscreteScheduler",
            "project_reference_solver": "ddim(flowmatch)",
            "project_ddim_is_one_step_shorter_than_official": bool(
                int(project_sigmas.numel()) == int(official_schedule["timesteps"].numel())
                and int(max(project_sigmas.numel() - 1, 0)) == int(official_schedule["timesteps"].numel()) - 1
            )
        },
    }

    if outdir:
        outdir_path = Path(outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        save_image(official["image"], outdir_path / "official_flux.png")
        save_image(project_ddim["image"], outdir_path / "project_ddim.png")
        save_image(project_ddim_terminal["image"], outdir_path / "project_ddim_with_terminal.png")
        with open(outdir_path / "report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
