#!/usr/bin/env python3
"""
Compare SD3 default FlowMatch sampling vs. EPD sampler step-by-step:
  * Logs timesteps (scheduler sigmas) and velocity norms for each step.
  * Runs both pipelines for the same seeds/prompts and saves final images.

python tests/test_epd_suit.py \
  --predictor exps/fake-sd3-15/network-snapshot-000005.pkl \
  --prompt "A dog." \
  --num-steps 15 \
  --seed 0 \
  --outdir tmp_epd_vs_sd3

"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torchvision.utils import save_image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusers import StableDiffusion3Pipeline
from models.backends import SD3DiffusersBackend
from sample import create_model_sd3
from solvers import epd_sampler
from solver_utils import get_schedule
from training.networks import EPD_predictor


def _format_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} GB"


def _load_predictor(path: Path, device: torch.device) -> EPD_predictor:
    import pickle

    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    predictor = snapshot["model"].to(device).eval()
    return predictor


def _assert_close(name: str, a: float, b: float, tol: float = 1e-6) -> None:
    if a is None or b is None:
        return
    if abs(float(a) - float(b)) > tol:
        raise RuntimeError(f"{name} mismatch: {a} vs {b}")


def _check_schedule_parity(
    predictor: EPD_predictor,
    backend: SD3DiffusersBackend,
    pipe_scheduler,
    mu_override: float | None,
) -> None:
    """
    Ensure sigma_min/max and shift/mu stay consistent across predictor/backends/baseline.
    """
    sigma_min_pred = getattr(predictor, "sigma_min", None)
    sigma_max_pred = getattr(predictor, "sigma_max", None)
    sigma_min_backend = getattr(backend, "sigma_min", None)
    sigma_max_backend = getattr(backend, "sigma_max", None)
    sigma_min_pipe = getattr(pipe_scheduler, "sigma_min", None)
    sigma_max_pipe = getattr(pipe_scheduler, "sigma_max", None)

    _assert_close("sigma_min predictor vs backend", sigma_min_pred, sigma_min_backend)
    _assert_close("sigma_max predictor vs backend", sigma_max_pred, sigma_max_backend)
    _assert_close("sigma_min predictor vs pipe", sigma_min_pred, sigma_min_pipe)
    _assert_close("sigma_max predictor vs pipe", sigma_max_pred, sigma_max_pipe)

    shift_backend = getattr(backend, "flow_shift", None)
    shift_pipe = getattr(pipe_scheduler.config, "shift", None)
    _assert_close("flow shift backend vs pipe", shift_backend, shift_pipe)

    if mu_override is not None:
        resolved_mu = backend.resolve_flowmatch_mu(override=mu_override)
        _assert_close("flowmatch mu override", mu_override, resolved_mu)


@torch.no_grad()
def _run_sd3_baseline(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    num_steps: int,
    latents: torch.Tensor,
    guidance_scale: float,
    mu: float | None = None,
) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
    pipe.scheduler.set_timesteps(num_steps, device=pipe._execution_device, mu=mu)
    timesteps = pipe.scheduler.timesteps

    do_cfg = guidance_scale > 1.0
    embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        device=pipe._execution_device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=[""] if do_cfg else None,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        clip_skip=getattr(pipe, "clip_skip", None),
        max_sequence_length=256,
        lora_scale=None,
    )
    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = embeds
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

    records: List[Tuple[int, torch.Tensor]] = []
    for i, t in enumerate(timesteps):
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        timestep_in = t.expand(latent_model_input.shape[0])
        noise_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep_in,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=getattr(pipe, "_joint_attention_kwargs", None),
            return_dict=False,
        )[0]
        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        records.append((int(t.item()), noise_pred.norm().detach().cpu()))
        latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = torch.clamp(image / 2 + 0.5, 0, 1)
    return image, records


@torch.no_grad()
def _run_epd(
    backend: SD3DiffusersBackend,
    predictor: EPD_predictor,
    prompt: str,
    latents: torch.Tensor,
    guidance_scale: float,
    num_steps: int,
) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
    device = backend.pipeline._execution_device
    condition = backend.prepare_condition(
        prompt=[prompt],
        negative_prompt=[""] if guidance_scale > 1.0 else None,
        guidance_scale=guidance_scale,
    )

    # Use the real solver to respect predictor tables and multi-point updates.
    schedule_type = predictor.schedule_type or "flowmatch"
    sigma_min = getattr(predictor, "sigma_min", None) or backend.sigma_min
    sigma_max = getattr(predictor, "sigma_max", None) or backend.sigma_max
    t_steps = get_schedule(
        num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        device=device,
        schedule_type=schedule_type,
        schedule_rho=predictor.schedule_rho,
        net=backend,
    )

    inters = epd_sampler(
        net=backend,
        latents=latents.clone(),
        condition=condition,
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        schedule_type=schedule_type,
        schedule_rho=predictor.schedule_rho,
        guidance_type=predictor.guidance_type,
        guidance_rate=guidance_scale,
        predictor=predictor,
        afs=False,
        return_inters=True,
        train=False,
    )

    # inters: [num_steps, B, C, H, W]; compute velocity norms for logging.
    records: List[Tuple[int, torch.Tensor]] = []
    for t_cur, x_cur in zip(t_steps, inters):
        denoised = backend(x_cur, t_cur, condition=condition)
        v = (x_cur - denoised) / t_cur
        timestep_val = int((t_cur * backend.pipeline.scheduler.config.num_train_timesteps).item())
        records.append((timestep_val, v.norm().detach().cpu()))

    decoded = backend.vae_decode(inters[-1])
    decoded = torch.clamp(decoded / 2 + 0.5, 0, 1)
    return decoded, records


def main() -> None:
    parser = argparse.ArgumentParser(description="EPD vs SD3 baseline step trace.")
    parser.add_argument("--prompt", type=str, default="A futuristic skyline at sunset")
    parser.add_argument("--num-steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("tmp_epd_vs_sd3"))
    parser.add_argument("--predictor", type=Path, required=True, help="EPD predictor snapshot (pkl).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Baseline pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # EPD backend
    backend, _ = create_model_sd3(guidance_rate=4.5, backend_config={}, device=device)
    predictor = _load_predictor(args.predictor, device=backend.pipeline._execution_device)

    # Baseline uses CLI-provided step count; EPD uses predictor.num_steps.
    expected_steps = args.num_steps
    guidance_scale = float(getattr(predictor, "guidance_rate", 4.5))
    mu_override = getattr(predictor, "flowmatch_mu", None)
    schedule_type = predictor.schedule_type or "flowmatch"
    sigma_min = getattr(predictor, "sigma_min", None) or backend.sigma_min
    sigma_max = getattr(predictor, "sigma_max", None) or backend.sigma_max

    # Log solver and baseline parameters for reproducibility.
    solver_config = {
        "schedule_type": schedule_type,
        "schedule_rho": predictor.schedule_rho,
        "num_steps_pred": predictor.num_steps,
        "num_points": predictor.num_points,
        "guidance_rate": guidance_scale,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "flowmatch_mu": mu_override,
        "flowmatch_shift": getattr(backend, "flow_shift", None),
    }
    print("[epd solver config]")
    for k, v in solver_config.items():
        print(f"  - {k}: {v}")
    print("[baseline config]")
    print(f"  - num_steps: {expected_steps}")
    print(f"  - guidance_rate: {guidance_scale}")
    print(f"  - flowmatch_mu: {mu_override}")

    _check_schedule_parity(predictor, backend, pipe.scheduler, mu_override)

    gen = torch.Generator(device=pipe._execution_device).manual_seed(args.seed % (1 << 32))
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.config.in_channels,
        height=pipe.transformer.config.sample_size * pipe.vae_scale_factor,
        width=pipe.transformer.config.sample_size * pipe.vae_scale_factor,
        dtype=pipe.transformer.dtype,
        device=pipe._execution_device,
        generator=gen,
        latents=None,
    )

    # Keep EPD backend scheduler in sync with the optional mu override.
    if mu_override is not None:
        backend.default_flowmatch_mu = mu_override
        backend.current_flowmatch_mu = mu_override

    img_sd3, records_sd3 = _run_sd3_baseline(
        pipe,
        args.prompt,
        expected_steps,
        latents=latents.clone(),
        guidance_scale=guidance_scale,
        mu=mu_override,
    )
    img_epd, records_epd = _run_epd(
        backend,
        predictor,
        args.prompt,
        latents=latents.clone(),
        guidance_scale=guidance_scale,
        num_steps=predictor.num_steps,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    save_image(img_sd3, args.outdir / "sd3_baseline.png")
    save_image(img_epd, args.outdir / "epd.png")

    print("=== Baseline (SD3 FlowMatch) ===")
    for idx, (t, norm_v) in enumerate(records_sd3):
        print(f"step {idx:02d} | timestep={t} | ||v||={norm_v.item():.4f}")
    print(f"=== EPD (epd_sampler, K={predictor.num_points}) ===")
    for idx, (t, norm_v) in enumerate(records_epd):
        print(f"step {idx:02d} | timestep={t} | ||v||={norm_v.item():.4f}")
    print(f"Saved images to {args.outdir}")


if __name__ == "__main__":
    main()
