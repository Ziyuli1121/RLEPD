#!/usr/bin/env python3
"""
Generate one SD1.5 image with four baseline samplers (DDIM/DPM2/EDM/IPNDM)
plus one EPD sampler (default predictor: sd15 RL base export).

Outputs inside the specified outdir:
    - ddim.png
    - dpm2.png
    - edm.png
    - ipndm.png
    - epd.png
"""

from __future__ import annotations

import os
import pickle
from contextlib import nullcontext
from pathlib import Path

import click
import torch

from sample import create_model_backend
from solvers import epd_sampler
from training.loss import get_solver_fn


DEFAULT_PREDICTOR = (
    "/work/nvme/betk/zli42/RLEPD/exps/20251114-202156-sd15_rl_base/export/network-snapshot-export-step003500.pkl"
)


def _load_predictor(path: Path, device: torch.device):
    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    predictor = snapshot["model"].to(device).eval()
    return predictor


def _prepare_condition(net, prompt: str, guidance_rate: float):
    prompts = [prompt]
    c = net.model.get_learned_conditioning(prompts)
    uc = None
    if guidance_rate != 1.0:
        uc = net.model.get_learned_conditioning([""])
    return c, uc


def _decode_image(net, samples: torch.Tensor) -> torch.Tensor:
    decoded = net.model.decode_first_stage(samples)
    decoded = torch.clamp(decoded / 2 + 0.5, 0, 1)
    return decoded[0]


def _save_image(image: torch.Tensor, path: Path) -> None:
    image_np = (image * 255).round().to(torch.uint8)
    image_np = image_np.permute(1, 2, 0).cpu().numpy()
    from PIL import Image

    Image.fromarray(image_np, "RGB").save(str(path))


def _make_latents(net, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device).manual_seed(int(seed) % (1 << 32))
    return torch.randn(
        [1, net.img_channels, net.img_resolution, net.img_resolution],
        device=device,
        generator=gen,
    )


def _run_baseline(
    net,
    sampler: str,
    base_latents: torch.Tensor,
    prompt: str,
    num_steps: int,
    schedule_type: str,
    schedule_rho: float,
    ipndm_max_order: int,
    ddim_eta: float,
) -> torch.Tensor:
    latents = base_latents.clone()
    guidance_rate = net.guidance_rate
    condition, unconditional_condition = _prepare_condition(net, prompt, guidance_rate)
    solver_fn = get_solver_fn(sampler)
    sigma_min = getattr(net, "sigma_min", 0.002)
    sigma_max = getattr(net, "sigma_max", 80.0)
    solver_kwargs = dict(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        schedule_type=schedule_type,
        schedule_rho=schedule_rho,
        afs=False,
        predictor=None,
        train=False,
    )
    if sampler == "ipndm":
        solver_kwargs["max_order"] = ipndm_max_order
    if sampler == "edm" and schedule_rho == 1.0:
        solver_kwargs["schedule_rho"] = 7.0
    if sampler == "edm":
        solver_kwargs.update(S_churn=0.0, S_min=0.0, S_max=float("inf"), S_noise=1.0, guidance_rate=guidance_rate)
    if sampler == "ddim":
        solver_kwargs["ddim_eta"] = ddim_eta

    ctx = torch.cuda.amp.autocast() if latents.device.type == "cuda" else nullcontext()
    with torch.no_grad(), ctx:
        with net.model.ema_scope() if hasattr(net.model, "ema_scope") else nullcontext():
            samples, _ = solver_fn(
                net=net,
                latents=latents,
                condition=condition,
                unconditional_condition=unconditional_condition,
                **solver_kwargs,
            )
    return _decode_image(net, samples)


def _run_epd(
    net,
    predictor,
    base_latents: torch.Tensor,
    prompt: str,
) -> torch.Tensor:
    latents = base_latents.clone()
    guidance_rate = predictor.guidance_rate
    condition, unconditional_condition = _prepare_condition(net, prompt, guidance_rate)
    sigma_min = getattr(predictor, "sigma_min", None) or getattr(net, "sigma_min", 0.002)
    sigma_max = getattr(predictor, "sigma_max", None) or getattr(net, "sigma_max", 80.0)
    schedule_type = getattr(predictor, "schedule_type", None) or "discrete"
    schedule_rho = getattr(predictor, "schedule_rho", None) or 1.0
    with torch.no_grad():
        with net.model.ema_scope() if hasattr(net.model, "ema_scope") else nullcontext():
            samples, _ = epd_sampler(
                net=net,
                latents=latents,
                class_labels=None,
                condition=condition,
                unconditional_condition=unconditional_condition if guidance_rate != 1.0 else None,
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
    return _decode_image(net, samples)


@click.command()
@click.option("--prompt", type=str, required=True, help="Single prompt string.")
@click.option(
    "--predictor",
    type=click.Path(exists=True, dir_okay=False),
    default=DEFAULT_PREDICTOR,
    show_default=True,
    help="EPD predictor .pkl file.",
)
@click.option(
    "--outdir",
    type=str,
    required=True,
    help="Directory to write ddim.png/dpm2.png/edm.png/ipndm.png/epd.png.",
)
@click.option("--seed", type=int, default=0, show_default=True, help="Random seed shared by all samplers.")
@click.option(
    "--schedule-type",
    type=str,
    default="discrete",
    show_default=True,
    help="Schedule type for baseline samplers.",
)
@click.option(
    "--schedule-rho",
    type=float,
    default=1.0,
    show_default=True,
    help="Schedule exponent for baseline samplers.",
)
@click.option(
    "--ipndm-max-order",
    type=int,
    default=4,
    show_default=True,
    help="Maximum order for the IPNDM baseline.",
)
@click.option("--ddim-steps", type=int, default=16, show_default=True, help="Steps for DDIM baseline.")
@click.option("--dpm-steps", type=int, default=11, show_default=True, help="Steps for DPM2 baseline.")
@click.option("--edm-steps", type=int, default=11, show_default=True, help="Steps for EDM baseline.")
@click.option("--ipndm-steps", type=int, default=16, show_default=True, help="Steps for IPNDM baseline.")
@click.option("--ddim-eta", type=float, default=0.0, show_default=True, help="DDIM eta parameter.")
def main(
    prompt: str,
    predictor: str,
    outdir: str,
    seed: int,
    schedule_type: str,
    schedule_rho: float,
    ipndm_max_order: int,
    ddim_steps: int,
    dpm_steps: int,
    edm_steps: int,
    ipndm_steps: int,
    ddim_eta: float,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor_path = Path(predictor)
    predictor_module = _load_predictor(predictor_path, device=device)

    backend_cfg = {}
    predictor_backend_cfg = getattr(predictor_module, "backend_config", None)
    if isinstance(predictor_backend_cfg, dict):
        backend_cfg = dict(predictor_backend_cfg)

    dataset_name = getattr(predictor_module, "dataset_name", None) or "ms_coco"
    guidance_rate = predictor_module.guidance_rate

    net, _ = create_model_backend(
        dataset_name=dataset_name,
        guidance_type=predictor_module.guidance_type,
        guidance_rate=guidance_rate,
        backend=getattr(predictor_module, "backend", "ldm"),
        backend_config=backend_cfg,
        device=device,
    )

    os.makedirs(outdir, exist_ok=True)
    base_latents = _make_latents(net, seed=seed, device=device)

    print(f"[sampler] prompt: {prompt}")
    print(f"[sampler] outdir: {outdir}")
    print(f"[sampler] seed: {seed}")
    print("[sampler] running baseline samplers (ddim/dpm2/edm/ipndm)...")

    ddim_image = _run_baseline(
        net,
        sampler="ddim",
        base_latents=base_latents,
        prompt=prompt,
        num_steps=ddim_steps,
        schedule_type=schedule_type,
        schedule_rho=schedule_rho,
        ipndm_max_order=ipndm_max_order,
        ddim_eta=ddim_eta,
    )
    _save_image(ddim_image, Path(outdir) / "ddim.png")

    dpm_image = _run_baseline(
        net,
        sampler="dpm2",
        base_latents=base_latents,
        prompt=prompt,
        num_steps=dpm_steps,
        schedule_type=schedule_type,
        schedule_rho=schedule_rho,
        ipndm_max_order=ipndm_max_order,
        ddim_eta=ddim_eta,
    )
    _save_image(dpm_image, Path(outdir) / "dpm2.png")

    edm_image = _run_baseline(
        net,
        sampler="edm",
        base_latents=base_latents,
        prompt=prompt,
        num_steps=edm_steps,
        schedule_type=schedule_type,
        schedule_rho=schedule_rho,
        ipndm_max_order=ipndm_max_order,
        ddim_eta=ddim_eta,
    )
    _save_image(edm_image, Path(outdir) / "edm.png")

    ipndm_image = _run_baseline(
        net,
        sampler="ipndm",
        base_latents=base_latents,
        prompt=prompt,
        num_steps=ipndm_steps,
        schedule_type=schedule_type,
        schedule_rho=schedule_rho,
        ipndm_max_order=ipndm_max_order,
        ddim_eta=ddim_eta,
    )
    _save_image(ipndm_image, Path(outdir) / "ipndm.png")

    print("[sampler] running EPD sampler...")
    epd_image = _run_epd(
        net,
        predictor=predictor_module,
        base_latents=base_latents,
        prompt=prompt,
    )
    _save_image(epd_image, Path(outdir) / "epd.png")

    print("Sampling complete. Images saved to", outdir)


if __name__ == "__main__":
    main()


'''
python sd15_prompt_sample.py \
  --prompt "Modern minimalist architecture, white concrete house, straight lines, blue sky, geometric shapes." \
  --outdir ./sd15_images/3 \
  --seed 42


1. A watercolor fox in a forest.
2. One bright yellow sunflower in a field on a sunny day.
3. 


'''