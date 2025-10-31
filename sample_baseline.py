import os
import re
import csv
import pickle
import click
import tqdm
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch import autocast
from torch_utils import distributed as dist
from torchvision.utils import make_grid, save_image
from torch_utils.download_util import check_file_by_key

from training.loss import get_solver_fn
from contextlib import nullcontext
from pathlib import Path


# -----------------------------------------------------------------------------
# Helper utilities copied from sample.py


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape,
            dtype=input.dtype,
            layout=input.layout,
            device=input.device,
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators]
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


def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None):
    model_path, classifier_path = check_file_by_key(dataset_name)
    dist.print0(f'Loading the pre-trained diffusion model from "{model_path}"...')

    if dataset_name in ["cifar10", "ffhq", "afhqv2", "imagenet64"]:
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)["ema"].to(device)
        net.sigma_min = 0.002
        net.sigma_max = 80
        model_source = "edm"
    elif dataset_name in ["lsun_bedroom"]:
        from models.cm.cm_model_loader import load_cm_model
        from models.networks_edm import CMPrecond

        net = load_cm_model(model_path)
        net = CMPrecond(net).to(device)
        model_source = "cm"
    else:
        if guidance_type == "cg":
            from models.guided_diffusion.cg_model_loader import load_cg_model
            from models.networks_edm import CGPrecond

            assert classifier_path is not None
            net, classifier = load_cg_model(model_path, classifier_path)
            net = CGPrecond(
                net, classifier, guidance_rate=guidance_rate, guidance_type=guidance_type, label_dim=0
            ).to(device)
            model_source = "adm"
        elif guidance_type in ["uncond", "cfg"]:
            from omegaconf import OmegaConf
            from models.networks_edm import CFGPrecond

            if dataset_name in ["lsun_bedroom_ldm"]:
                config = OmegaConf.load(
                    "./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml"
                )
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(
                    net,
                    img_resolution=64,
                    img_channels=3,
                    guidance_rate=1.0,
                    guidance_type="uncond",
                    label_dim=0,
                ).to(device)
            elif dataset_name in ["ms_coco"]:
                assert guidance_type == "cfg"
                config = OmegaConf.load("./models/ldm/configs/stable-diffusion/v1-inference.yaml")
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(
                    net,
                    img_resolution=64,
                    img_channels=4,
                    guidance_rate=guidance_rate,
                    guidance_type="classifier-free",
                    label_dim=True,
                ).to(device)
            model_source = "ldm"
        else:
            raise ValueError("Got wrong settings: check dataset_name and guidance_type!")
    if net is None:
        raise ValueError("Got wrong settings: check dataset_name and guidance_type!")
    net.eval()

    return net, model_source


def load_ldm_model(config, ckpt, verbose=False):
    from models.ldm.util import instantiate_from_config

    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        dist.print0(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


# -----------------------------------------------------------------------------
# Baseline sampling entrypoint


@click.command()
@click.option(
    "--sampler",
    type=click.Choice(["dpm", "heun", "ipndm", "edm", "ddim"], case_sensitive=False),
    required=True,
    help="Name of the teacher sampler.",
)
@click.option(
    "--dataset-name",
    type=str,
    required=True,
    help="Dataset key (e.g., ms_coco, cifar10) used to load the base diffusion model.",
)
@click.option("--model-path", type=str, help="Optional override for diffusion model weights.")
@click.option(
    "--batch",
    "max_batch_size",
    help="Maximum batch size per iteration.",
    metavar="INT",
    type=click.IntRange(min=1),
    default=64,
    show_default=True,
)
@click.option(
    "--seeds",
    help="Random seeds (e.g. 1,2,5-10)",
    metavar="LIST",
    type=parse_int_list,
    default="0-63",
    show_default=True,
)
@click.option("--prompt", help="Single prompt for sampling.", metavar="STR", type=str)
@click.option(
    "--prompt-file",
    help="Path to text/CSV file with prompts (one per line or CSV with 'text' column).",
    metavar="PATH",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--use-fp16", help="Whether to use mixed precision", metavar="BOOL", type=bool, default=False)
@click.option("--outdir", help="Where to save the output images", metavar="DIR", type=str)
@click.option("--grid", help="Whether to make grid", type=bool, default=False)
@click.option(
    "--subdirs",
    help="Create subdirectory for every 1000 seeds",
    type=bool,
    default=True,
    is_flag=True,
)
@click.option("--guidance-type", type=str, default="cfg", show_default=True)
@click.option("--guidance-rate", type=float, default=7.5, show_default=True)
@click.option("--num-steps", type=int, default=20, show_default=True)
@click.option("--sigma-min", type=float, default=None)
@click.option("--sigma-max", type=float, default=None)
@click.option("--schedule-type", type=str, default="discrete", show_default=True)
@click.option("--schedule-rho", type=float, default=1.0, show_default=True)
@click.option("--afs", type=bool, default=False, is_flag=True, help="Apply AFS trick on first step.")
@click.option(
    "--inner-steps",
    type=int,
    default=None,
    help="Number of inner steps for multi-stage solvers (dpm/heun).",
)
@click.option("--solver-r", type=float, default=0.5, help="DPM relaxation factor.", show_default=True)
@click.option(
    "--max-order",
    type=int,
    default=4,
    show_default=True,
    help="Maximum order for IPNDM solver.",
)
@click.option("--ddim-eta", type=float, default=0.0, show_default=True, help="DDIM eta parameter.")
@click.option("--ddim-steps", type=int, default=None, help="Override number of DDIM steps.")
@click.option("--edm-s-churn", type=float, default=0.0, show_default=True, help="EDM sampler S_churn parameter.")
@click.option("--edm-s-min", type=float, default=0.0, show_default=True, help="EDM sampler S_min parameter.")
@click.option("--edm-s-max", type=float, default=float('inf'), show_default=True, help="EDM sampler S_max parameter.")
@click.option("--edm-s-noise", type=float, default=1.0, show_default=True, help="EDM sampler S_noise parameter.")
def main(
    sampler,
    dataset_name,
    model_path,
    max_batch_size,
    seeds,
    grid,
    outdir,
    subdirs,
    prompt,
    prompt_file,
    use_fp16,
    guidance_type,
    guidance_rate,
    num_steps,
    sigma_min,
    sigma_max,
    schedule_type,
    schedule_rho,
    afs,
    inner_steps,
    solver_r,
    max_order,
    ddim_eta,
    ddim_steps,
    edm_s_churn,
    edm_s_min,
    edm_s_max,
    edm_s_noise,
    **_
):
    sampler = sampler.lower()

    dist.init()
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    max_batch_size = int(max_batch_size)
    seeds = list(seeds)

    num_batches = ((len(seeds) - 1) // (max_batch_size * world_size) + 1) * world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[rank :: world_size]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load diffusion model
    net, model_source = create_model(
        dataset_name=dataset_name,
        guidance_type=guidance_type,
        guidance_rate=guidance_rate,
        device=device,
    )
    if model_path:
        raise NotImplementedError("--model-path override is not yet supported without predictor.")

    sigma_min = sigma_min if sigma_min is not None else getattr(net, "sigma_min", 0.002)
    sigma_max = sigma_max if sigma_max is not None else getattr(net, "sigma_max", 80.0)

    solver_kwargs = {
        "solver": sampler,
        "num_steps": num_steps,
        "sigma_min": sigma_min,
        "sigma_max": sigma_max,
        "schedule_type": schedule_type,
        "schedule_rho": schedule_rho,
        "afs": afs,
        "denoise_to_zero": False,
        "dataset_name": dataset_name,
        "guidance_type": guidance_type,
        "guidance_rate": guidance_rate,
        "model_source": model_source,
    }

    if sampler in ["dpm", "heun"]:
        default_inner = 2 if sampler == "dpm" else 3
        solver_kwargs["inner_steps"] = inner_steps if inner_steps is not None else default_inner
        if sampler == "dpm":
            solver_kwargs["r"] = solver_r
    if sampler == "ipndm":
        solver_kwargs["max_order"] = max_order
    if sampler == "edm":
        solver_kwargs["schedule_rho"] = 7.0 if schedule_rho == 1.0 else schedule_rho
        solver_kwargs.update(
            S_churn=edm_s_churn,
            S_min=edm_s_min,
            S_max=edm_s_max,
            S_noise=edm_s_noise,
        )
    if sampler == "ddim":
        if ddim_steps is not None:
            solver_kwargs["num_steps"] = int(ddim_steps)
        solver_kwargs["ddim_eta"] = ddim_eta

    # Load prompts
    prompts_list = None
    if prompt_file and prompt is None:
        path = Path(prompt_file)
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                prompts_list = [row.get("text", "").strip() for row in reader if row.get("text")]
        else:
            with path.open("r", encoding="utf-8") as handle:
                prompts_list = [line.strip() for line in handle if line.strip()]
        if not prompts_list:
            raise RuntimeError(f"No prompts found in '{prompt_file}'.")
    elif dataset_name in ["ms_coco"] and prompt is None:
        prompt_path, _ = check_file_by_key("prompts")
        prompts_list = []
        with open(prompt_path, "r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text = row.get("text", "").strip()
                if text:
                    prompts_list.append(text)
        if not prompts_list:
            raise RuntimeError(f"No prompts found in '{prompt_path}'.")

    # Determine output directory
    nfe = num_steps
    if dataset_name in ["ms_coco"] and guidance_type == "cfg":
        nfe = 2 * nfe
    sampler_tag = f"{sampler}"
    if outdir is None:
        if grid:
            outdir = os.path.join(f"./samples/grids/{dataset_name}", f"{sampler_tag}_nfe{nfe}")
        else:
            outdir = os.path.join(f"./samples/{dataset_name}", f"{sampler_tag}_nfe{nfe}")
    dist.print0(f"Generating {len(seeds)} images to \"{outdir}\"...")

    sampler_fn = get_solver_fn(sampler)

    for batch_id, batch_seeds in enumerate(
        tqdm.tqdm(rank_batches, unit="batch", disable=(rank != 0))
    ):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
        )

        class_labels = c = uc = None
        if net.label_dim:
            if model_source == "ldm" and dataset_name == "ms_coco":
                if prompt is None:
                    if prompts_list is None:
                        raise RuntimeError("Prompt list is empty; provide --prompt or --prompt-file.")
                    start = int(batch_seeds[0])
                    end = int(batch_seeds[-1])
                    if end >= len(prompts_list):
                        raise RuntimeError(
                            f"Seed index {end} exceeds available prompts ({len(prompts_list)})."
                        )
                    prompts = prompts_list[start : end + 1]
                else:
                    prompts = [prompt for _ in range(batch_size)]

                if guidance_rate != 1.0:
                    uc = net.model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = net.model.get_learned_conditioning(prompts)
            else:
                class_labels = rnd.randint(net.label_dim, size=(batch_size,), device=device)

        call_kwargs = dict(solver_kwargs)
        call_kwargs.update(
            condition=c,
            unconditional_condition=uc,
            class_labels=class_labels,
        )

        if sampler == "ipndm":
            call_kwargs.update(predictor=None, train=False)

        with torch.no_grad():
            if model_source == "ldm":
                ctx = autocast("cuda") if use_fp16 else nullcontext()
                with ctx:
                    with net.model.ema_scope():
                        if batch_id == 0:
                            call_kwargs["verbose"] = True
                        images, _ = sampler_fn(net, latents, **call_kwargs)
                        images = net.model.decode_first_stage(images)
            else:
                if batch_id == 0:
                    call_kwargs["verbose"] = True
                images, _ = sampler_fn(net, latents, **call_kwargs)

        if grid:
            images = torch.clamp(images / 2 + 0.5, 0, 1)
            os.makedirs(outdir, exist_ok=True)
            nrows = int(images.shape[0] ** 0.5)
            image_grid = make_grid(images, nrows, padding=0)
            save_image(image_grid, os.path.join(outdir, "grid.png"))
        else:
            images_np = (
                (images * 127.5 + 128)
                .clip(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f"{seed - seed % 1000:06d}") if subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f"{seed:06d}.png")
                PIL.Image.fromarray(image_np, "RGB").save(image_path)

    torch.distributed.barrier()
    dist.print0("Done.")


if __name__ == "__main__":
    main()
