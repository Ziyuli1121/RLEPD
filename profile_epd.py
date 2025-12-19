#!/usr/bin/env python3
"""
Profile EPD samplers (parallel vs non-parallel) on a given predictor.
Outputs a top-k CUDA time table to stdout. No image writing.

Usage examples:
  python profile_epd.py --predictor exps/latency/sd15_k2_nfe20/network-snapshot-000005.pkl --sampler epd_parallel
  python profile_epd.py --predictor exps/latency/sd3-512_k2_nfe20/network-snapshot-000005.pkl --sampler epd
"""

from __future__ import annotations

import argparse
import pickle
import torch

import solvers
from sample import create_model_backend


def load_predictor(path: str, device: torch.device):
    with open(path, "rb") as f:
        snap = pickle.load(f)
    pred = snap["model"].to(device).eval()
    return pred


def prepare_condition(net, predictor, batch_size: int):
    backend_name = getattr(net, "backend", None)
    if backend_name == "sd3":
        cond = net.prepare_condition(prompt=[""] * batch_size, negative_prompt=[""] * batch_size, guidance_scale=predictor.guidance_rate)
        uc = None
    else:
        cond = net.model.get_learned_conditioning([""] * batch_size)
        uc = net.model.get_learned_conditioning([""] * batch_size) if predictor.guidance_rate != 1.0 else None
    return cond, uc


def run_once(sampler_name: str, sampler_fn, predictor, net, latents, cond, uc):
    return sampler_fn(
        net=net,
        latents=latents,
        condition=cond,
        unconditional_condition=uc,
        num_steps=predictor.num_steps,
        sigma_min=getattr(predictor, "sigma_min", None) or getattr(net, "sigma_min", 0.002),
        sigma_max=getattr(predictor, "sigma_max", None) or getattr(net, "sigma_max", 80.0),
        schedule_type=getattr(predictor, "schedule_type", None) or "discrete",
        schedule_rho=getattr(predictor, "schedule_rho", None) or 1.0,
        guidance_type=predictor.guidance_type,
        guidance_rate=predictor.guidance_rate,
        predictor=predictor,
        afs=bool(getattr(predictor, "afs", False)),
        return_inters=False,
        train=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile EPD samplers with torch.profiler.")
    parser.add_argument("--predictor", required=True, help="Path to predictor .pkl")
    parser.add_argument("--sampler", choices=["epd", "epd_parallel"], default="epd_parallel", help="Sampler to profile.")
    parser.add_argument("--seeds", type=str, default="0", help="Seeds (ignored; batch size derived)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for latents.")
    parser.add_argument("--topk", type=int, default=30, help="Rows to show in profiler table.")
    parser.add_argument(
        "--compile-runner",
        action="store_true",
        help="Wrap the sampler call in torch.compile for a single graph (experimental).",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        help="torch.compile mode to use when --compile-runner is set.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = load_predictor(args.predictor, device=device)

    net, _ = create_model_backend(
        dataset_name=getattr(predictor, "dataset_name", None) or "ms_coco",
        guidance_type=predictor.guidance_type,
        guidance_rate=predictor.guidance_rate,
        backend=getattr(predictor, "backend", None) or "ldm",
        backend_config=getattr(predictor, "backend_config", None) or {},
        device=device,
    )

    latents = torch.randn(
        [args.batch_size, net.img_channels, net.img_resolution, net.img_resolution],
        device=device,
        dtype=getattr(net, "dtype", None) or getattr(getattr(net, "pipeline", None), "dtype", None) or torch.float32,
    )
    cond, uc = prepare_condition(net, predictor, args.batch_size)

    sampler_fn = solvers.epd_parallel_sampler if args.sampler == "epd_parallel" else solvers.epd_sampler
    runner = lambda latents, cond, uc: sampler_fn(
        net=net,
        latents=latents,
        condition=cond,
        unconditional_condition=uc,
        num_steps=predictor.num_steps,
        sigma_min=getattr(predictor, "sigma_min", None) or getattr(net, "sigma_min", 0.002),
        sigma_max=getattr(predictor, "sigma_max", None) or getattr(net, "sigma_max", 80.0),
        schedule_type=getattr(predictor, "schedule_type", None) or "discrete",
        schedule_rho=getattr(predictor, "schedule_rho", None) or 1.0,
        guidance_type=predictor.guidance_type,
        guidance_rate=predictor.guidance_rate,
        predictor=predictor,
        afs=bool(getattr(predictor, "afs", False)),
        return_inters=False,
        train=False,
    )

    if args.compile_runner:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build.")
        runner = torch.compile(runner, mode=args.compile_mode, fullgraph=False)

    # Warm-up
    with torch.no_grad():
        imgs, _ = runner(latents, cond, uc)
        if getattr(net, "backend", None) == "sd3":
            imgs = net.vae_decode(imgs)
        else:
            imgs = net.model.decode_first_stage(imgs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with torch.no_grad():
            imgs, _ = runner(latents, cond, uc)
            if getattr(net, "backend", None) == "sd3":
                imgs = net.vae_decode(imgs)
            else:
                imgs = net.model.decode_first_stage(imgs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    print(f"[profile] sampler={args.sampler}, predictor={args.predictor}, backend={getattr(net,'backend',None)}, batch={args.batch_size}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=args.topk))


if __name__ == "__main__":
    main()



'''

python profile_epd.py --predictor exps/latency/sd15_k2_nfe20/network-snapshot-000005.pkl --sampler epd_parallel --batch-size 1

python profile_epd.py --predictor exps/latency/sd3-512_k1_nfe20/network-snapshot-000005.pkl --sampler epd_parallel --batch-size 1
python profile_epd.py --predictor exps/latency/sd3-512_k2_nfe20/network-snapshot-000005.pkl --sampler epd_parallel --batch-size 1
# python profile_epd.py --predictor exps/latency/sd3-512_k1_nfe20/network-snapshot-000005.pkl --sampler epd --batch-size 1
# python profile_epd.py --predictor exps/latency/sd3-512_k2_nfe20/network-snapshot-000005.pkl --sampler epd --batch-size 1


OMP_NUM_THREADS=512 MKL_NUM_THREADS=512 NUMEXPR_NUM_THREADS=512 \
  python profile_epd.py --predictor exps/latency/sd3-512_k1_nfe20/network-snapshot-000005.pkl --sampler epd_parallel --batch-size 1

OMP_NUM_THREADS=512 MKL_NUM_THREADS=512 NUMEXPR_NUM_THREADS=512 \
  python profile_epd.py --predictor exps/latency/sd3-512_k2_nfe20/network-snapshot-000005.pkl --sampler epd_parallel --batch-size 1


python -m trace --count profile_epd.py --predictor exps/latency/sd3-512_k1_nfe20/network-snapshot-000005.pkl --sampler epd_parallel --batch-size 1

python -m trace --count profile_epd.py --predictor exps/latency/sd3-512_k2_nfe20/network-snapshot-000005.pkl --sampler epd_parallel --batch-size 1


OMP_NUM_THREADS=1 MALLOC_CHECK_=3 \
python -m trace --count profile_epd.py --predictor exps/latency/sd3-512_k1_nfe20/network-snapshot-000005.pkl --sampler epd_parallel --batch-size 1

OMP_NUM_THREADS=1 MALLOC_CHECK_=3 \
python -m trace --count profile_epd.py --predictor exps/latency/sd3-512_k2_nfe20/network-snapshot-000005.pkl --sampler epd_parallel --batch-size 1


python profile_epd.py \
  --predictor exps/latency/sd3-512_k1_nfe20/network-snapshot-000005.pkl \
  --sampler epd_parallel \
  --batch-size 1 \
  --compile-runner \
  --compile-mode reduce-overhead

1.353s, 238.594ms

python profile_epd.py \
  --predictor exps/latency/sd3-512_k2_nfe20/network-snapshot-000005.pkl \
  --sampler epd_parallel \
  --batch-size 1 \
  --compile-runner \
  --compile-mode reduce-overhead

2.046s, 330.019ms
'''
