#!/usr/bin/env python3
"""
Generate one SD3 image with four baseline samplers (Euler/EDM/DPM2/IPNDM)
plus one EPD sampler, all from a single prompt.

The output directory will contain:
    - euler.png
    - edm.png
    - dpm.png
    - ipndm.png
    - epd.png
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import click
import torch

from sample import create_model_sd3
from solvers import epd_sampler
from training.loss import get_solver_fn


def flowmatch_euler_no_terminal(backend, latents, condition, t_steps: torch.Tensor) -> torch.Tensor:
    """
    FlowMatch Euler rollout without the final sigma=0 step (mirrors sample_sd3_baseline).
    """
    x = latents
    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        t_cur_scalar = t_cur.to(device=x.device, dtype=x.dtype)
        if t_cur_scalar.ndim != 0:
            t_cur_scalar = t_cur_scalar.reshape(())
        t_next_scalar = t_next.to(device=x.device, dtype=x.dtype)
        if t_next_scalar.ndim != 0:
            t_next_scalar = t_next_scalar.reshape(())

        denoised = backend(x, t_cur_scalar, condition=condition)

        t_cur_b = t_cur_scalar.reshape(1, 1, 1, 1)
        t_next_b = t_next_scalar.reshape(1, 1, 1, 1)
        d_cur = (x - denoised) / t_cur_b
        x = x + (t_next_b - t_cur_b) * d_cur
    return x


def _load_predictor(path: Path, device: torch.device):
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    predictor = snapshot["model"].to(device).eval()
    return predictor


def _prepare_condition(backend, prompt: str, guidance_scale: float):
    negative_prompt = [""] if guidance_scale > 1.0 else None
    return backend.prepare_condition(
        prompt=[prompt],
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
    )


def _decode_image(backend, samples: torch.Tensor) -> torch.Tensor:
    images = backend.vae_decode(samples)
    images = torch.clamp(images / 2 + 0.5, 0, 1)
    return images[0]


def _save_image(image: torch.Tensor, path: Path) -> None:
    image_np = (image * 255).round().to(torch.uint8)
    image_np = image_np.permute(1, 2, 0).cpu().numpy()
    from PIL import Image

    Image.fromarray(image_np, "RGB").save(str(path))


def _make_latents(backend, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device).manual_seed(int(seed) % (1 << 32))
    return torch.randn(
        [1, backend.img_channels, backend.img_resolution, backend.img_resolution],
        device=device,
        dtype=backend.pipeline.transformer.dtype,
        generator=gen,
    )


def _run_flowmatch(
    backend,
    base_latents: torch.Tensor,
    prompt: str,
    guidance_scale: float,
    num_steps: int,
    skip_final: bool = True,
    seed: int | None = None,
) -> torch.Tensor:
    latents = base_latents.clone()
    condition = _prepare_condition(backend, prompt, guidance_scale)
    if skip_final:
        t_steps = backend.make_flowmatch_schedule(num_steps, device=latents.device)
        with torch.no_grad():
            samples = flowmatch_euler_no_terminal(backend, latents, condition, t_steps)
        return _decode_image(backend, samples)
    # Fallback to pipeline if the final sigma=0 step is desired.
    pipe = backend.pipeline
    generator = torch.Generator(latents.device).manual_seed(0 if seed is None else int(seed) % (1 << 32))
    with torch.no_grad():
        result = pipe(
            prompt=[prompt],
            negative_prompt=[""] if guidance_scale > 1.0 else None,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=backend.output_resolution,
            width=backend.output_resolution,
            output_type="pt",
        )
    images = torch.clamp(result.images, 0, 1)
    return images[0]


def _run_solver(
    backend,
    sampler: str,
    base_latents: torch.Tensor,
    prompt: str,
    guidance_scale: float,
    num_steps: int,
    schedule_type: str,
    schedule_rho: float,
    ipndm_max_order: int,
) -> torch.Tensor:
    latents = base_latents.clone()
    condition = _prepare_condition(backend, prompt, guidance_scale)
    solver_fn = get_solver_fn(sampler)
    solver_kwargs = dict(
        num_steps=num_steps,
        sigma_min=backend.sigma_min,
        sigma_max=backend.sigma_max,
        schedule_type=schedule_type,
        schedule_rho=schedule_rho,
        afs=False,
        predictor=None,
        train=False,
    )
    if sampler == "ipndm":
        solver_kwargs["max_order"] = ipndm_max_order
    with torch.no_grad():
        samples, _ = solver_fn(
            net=backend,
            latents=latents,
            condition=condition,
            **solver_kwargs,
        )
    return _decode_image(backend, samples)


def _run_epd(
    backend,
    predictor,
    base_latents: torch.Tensor,
    prompt: str,
) -> torch.Tensor:
    latents = base_latents.clone()
    guidance_rate = predictor.guidance_rate
    condition = _prepare_condition(
        backend,
        prompt,
        guidance_scale=guidance_rate,
    )
    sigma_min = getattr(predictor, "sigma_min", None) or backend.sigma_min
    sigma_max = getattr(predictor, "sigma_max", None) or backend.sigma_max
    schedule_type = getattr(predictor, "schedule_type", None) or "flowmatch"
    schedule_rho = getattr(predictor, "schedule_rho", None)
    with torch.no_grad():
        samples, _ = epd_sampler(
            net=backend,
            latents=latents,
            condition=condition,
            num_steps=predictor.num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            schedule_type=schedule_type,
            schedule_rho=schedule_rho,
            guidance_type=predictor.guidance_type,
            guidance_rate=guidance_rate,
            predictor=predictor,
            afs=bool(getattr(predictor, "afs", False)),
            return_inters=False,
            train=False,
        )
    return _decode_image(backend, samples)


@click.command()
@click.option("--prompt", type=str, required=True, help="Single prompt string.")
@click.option(
    "--predictor",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="EPD predictor .pkl file.",
)
@click.option(
    "--outdir",
    type=str,
    required=True,
    help="Directory to write euler.png/edm.png/dpm.png/ipndm.png/epd.png.",
)
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed shared by all samplers.")
@click.option(
    "--model-id",
    type=str,
    default="stabilityai/stable-diffusion-3-medium-diffusers",
    show_default=True,
    help="Diffusers model repo or local path.",
)
@click.option(
    "--resolution",
    type=click.Choice(["512", "1024"], case_sensitive=False),
    default="1024",
    show_default=True,
    help="Output image resolution.",
)
@click.option(
    "--guidance-rate",
    type=float,
    default=4.5,
    show_default=True,
    help="CFG scale for baseline samplers.",
)
@click.option(
    "--schedule-rho",
    type=float,
    default=7.0,
    show_default=True,
    help="Schedule exponent for baseline solvers.",
)
@click.option(
    "--skip-final-flowmatch-step/--keep-final-flowmatch-step",
    default=True,
    show_default=True,
    help="Drop the terminal sigma=0 FlowMatch Euler step.",
)
@click.option(
    "--ipndm-max-order",
    type=int,
    default=2,
    show_default=True,
    help="Maximum order for the IPNDM baseline.",
)
@click.option("--euler-steps", type=int, default=28, show_default=True, help="Steps for FlowMatch Euler baseline.")
@click.option("--edm-steps", type=int, default=14, show_default=True, help="Steps for EDM baseline.")
@click.option("--dpm-steps", type=int, default=14, show_default=True, help="Steps for DPM2 baseline.")
@click.option("--ipndm-steps", type=int, default=28, show_default=True, help="Steps for IPNDM baseline.")
def main(
    prompt: str,
    predictor: str,
    outdir: str,
    seed: int,
    model_id: str,
    resolution: str,
    guidance_rate: float,
    schedule_rho: float,
    skip_final_flowmatch_step: bool,
    ipndm_max_order: int,
    euler_steps: int,
    edm_steps: int,
    dpm_steps: int,
    ipndm_steps: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor_path = Path(predictor)
    predictor_module = _load_predictor(predictor_path, device=device)

    backend_cfg = {}
    predictor_backend_cfg = getattr(predictor_module, "backend_config", None)
    if isinstance(predictor_backend_cfg, dict):
        backend_cfg = dict(predictor_backend_cfg)
    backend_cfg["model_name_or_path"] = model_id
    backend_cfg["model_id"] = model_id
    backend_cfg.setdefault("flowmatch_mu", getattr(predictor_module, "flowmatch_mu", None))
    backend_cfg.setdefault("sigma_min", getattr(predictor_module, "sigma_min", None))
    backend_cfg.setdefault("sigma_max", getattr(predictor_module, "sigma_max", None))

    target_resolution = int(resolution)
    predictor_resolution = getattr(predictor_module, "img_resolution", None)
    if predictor_resolution is not None and int(predictor_resolution) != target_resolution:
        raise click.ClickException(
            f"Resolution mismatch: predictor expects {predictor_resolution}, got --resolution={target_resolution}."
        )
    backend_cfg["resolution"] = target_resolution

    backend, _ = create_model_sd3(
        guidance_rate=guidance_rate,
        device=device,
        backend_config=backend_cfg,
    )
    if not hasattr(backend, "round_sigma"):
        backend.round_sigma = lambda x: x  # EDM fallback for SD3 backend
    backend.pipeline.set_progress_bar_config(disable=True)

    os.makedirs(outdir, exist_ok=True)
    base_latents = _make_latents(backend, seed=seed, device=device)

    print(f"[sampler] prompt: {prompt}")
    print(f"[sampler] outdir: {outdir}")
    print(f"[sampler] seed: {seed}")
    print("[sampler] running baseline samplers (euler/edm/dpm/ipndm)...")

    euler_image = _run_flowmatch(
        backend,
        base_latents,
        prompt=prompt,
        guidance_scale=guidance_rate,
        num_steps=euler_steps,
        skip_final=skip_final_flowmatch_step,
        seed=seed,
    )
    _save_image(euler_image, Path(outdir) / "euler.png")

    edm_image = _run_solver(
        backend,
        sampler="edm",
        base_latents=base_latents,
        prompt=prompt,
        guidance_scale=guidance_rate,
        num_steps=edm_steps,
        schedule_type="flowmatch",
        schedule_rho=schedule_rho,
        ipndm_max_order=ipndm_max_order,
    )
    _save_image(edm_image, Path(outdir) / "edm.png")

    dpm_image = _run_solver(
        backend,
        sampler="dpm2",
        base_latents=base_latents,
        prompt=prompt,
        guidance_scale=guidance_rate,
        num_steps=dpm_steps,
        schedule_type="flowmatch",
        schedule_rho=schedule_rho,
        ipndm_max_order=ipndm_max_order,
    )
    _save_image(dpm_image, Path(outdir) / "dpm.png")

    ipndm_image = _run_solver(
        backend,
        sampler="ipndm",
        base_latents=base_latents,
        prompt=prompt,
        guidance_scale=guidance_rate,
        num_steps=ipndm_steps,
        schedule_type="flowmatch",
        schedule_rho=schedule_rho,
        ipndm_max_order=ipndm_max_order,
    )
    _save_image(ipndm_image, Path(outdir) / "ipndm.png")

    print("[sampler] running EPD sampler...")
    epd_image = _run_epd(
        backend,
        predictor=predictor_module,
        base_latents=base_latents,
        prompt=prompt,
    )
    _save_image(epd_image, Path(outdir) / "epd.png")

    print("Sampling complete. Images saved to", outdir)


if __name__ == "__main__":
    main()



'''
python sd3_prompt_sample.py \
    --prompt "Five cars on the street." \
    --predictor exps/20251206-131339-sd3_1024/export/network-snapshot-export-step005500.pkl \
    --outdir ./sd3_images/10 \
    --seed 0


1. a photo of a toothbrush below a pizza
2. A robot holding a piece of paper with the text "Hello AI" written on it.
3. (masterpiece, best quality), 1 girl, solo, portrait, up close, long white hair, flowing hair, glowing golden eyes, holding white flowers, flowers in hair, ethereal atmosphere, soft lighting, backlight, cinematic lighting, delicate features, dreamlike, anime style illustration, detailed eyes.
4. misaka mikoto
5. A semi-realistic digital painting of a young woman looking back over her shoulder on a rainy cyberpunk street at night. She has an anime-influenced face but realistic proportions and detailed clothing textures. Neon signs reflecting on wet pavement, volumetric fog, cinematic lighting, thick brushstrokes, highly detailed, rich colors, evocative atmosphere, game concept art style.
6. A cyberpunk-style paramedic overlooks a futuristic city tower.
7. Product photograph of an ornate antique steampunk pocket watch, exposed gears and springs, engraved brass casing, intricate mechanical details, studio lighting.
8. Macro shot of rain droplets on a waxy green leaf, clear refractions inside the droplets, detailed leaf veins, natural morning dew aesthetic.
9. Five cars on the street.
'''