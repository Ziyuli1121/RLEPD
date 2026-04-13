#!/usr/bin/env python3
"""
Rigorous FLUX solver latency benchmark.

This script compares:
  - official FLUX Euler baseline
  - generic project solvers: edm / dpm2 / ipndm
  - exported EPD solver replay from a checkpoint/predictor

Timing scopes:
  - sampling-only: prompt conditioning / latent prep / solver denoising, no decode
  - end-to-end: sampling-only + decode/postprocess, still excluding model load and PNG writes
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from sample import _prepare_flux_condition, create_model_flux
from training.loss import get_solver_fn
from training.networks import EPD_predictor
from training.ppo.export_epd_predictor import export_policy_mean_to_predictor
from training.ppo.pipeline_utils import (
    load_prompts_file,
    resolve_flux_runtime_metadata,
    resolve_predictor_path,
)


class StackedRandomGenerator:
    """Batch-aware RNG so each seed owns its own generator."""

    def __init__(self, device: torch.device, seeds: Sequence[int]):
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])


@dataclass
class SolverSpec:
    name: str
    kind: str  # official_euler | generic | epd
    configured_steps: int
    nominal_nfe: int
    estimated_model_forwards: int
    guidance_rate: float
    schedule_type: str
    schedule_rho: float
    sigma_min: float
    sigma_max: float
    sampler_fn: Optional[Any] = None
    predictor: Optional[EPD_predictor] = None
    predictor_path: Optional[str] = None
    afs: bool = False
    extra_solver_kwargs: Optional[Dict[str, Any]] = None


def parse_int_list(value: str | Sequence[int]) -> List[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    values: List[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def parse_solver_list(value: str | Sequence[str]) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip().lower() for item in value if str(item).strip()]
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def _load_prompts(prompt_file: str, count: int) -> List[str]:
    prompts = load_prompts_file(prompt_file)
    if not prompts:
        raise ValueError(f"No prompts found in {prompt_file}")
    reps = (count + len(prompts) - 1) // len(prompts)
    return (prompts * reps)[:count]


def _load_predictor(path: Path, device: torch.device) -> EPD_predictor:
    import pickle

    with path.open("rb") as handle:
        snapshot = pickle.load(handle)
    predictor = snapshot["model"].to(device).eval()
    return predictor


def _decode_flux_packed_latents_to_pt(pipe, packed_latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
    latents = pipe._unpack_latents(packed_latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pt")


def _save_first_image(image_tensor: torch.Tensor, output_path: Path) -> None:
    from PIL import Image

    image = torch.clamp(image_tensor.detach().cpu(), 0, 1)
    image_np = (image * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_np).save(output_path)


def _safe_import_version(module_name: str) -> Optional[str]:
    try:
        module = __import__(module_name)
    except Exception:
        return None
    return getattr(module, "__version__", None)


def _stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {key: float("nan") for key in ("mean", "std", "p50", "p90", "p95")}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def _format_seconds(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _build_baseline_spec(
    sampler_name: str,
    *,
    steps: int,
    guidance_rate: float,
    sigma_min: float,
    sigma_max: float,
) -> SolverSpec:
    sampler = sampler_name.lower()
    if sampler == "euler":
        return SolverSpec(
            name="euler",
            kind="official_euler",
            configured_steps=int(steps),
            nominal_nfe=int(steps),
            estimated_model_forwards=int(steps),
            guidance_rate=float(guidance_rate),
            schedule_type="flowmatch",
            schedule_rho=7.0,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
        )
    if sampler == "ipndm":
        return SolverSpec(
            name="ipndm",
            kind="generic",
            configured_steps=int(steps),
            nominal_nfe=max(int(steps) - 1, 0),
            estimated_model_forwards=max(int(steps) - 1, 0),
            guidance_rate=float(guidance_rate),
            schedule_type="flowmatch",
            schedule_rho=7.0,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            sampler_fn=get_solver_fn("ipndm"),
            extra_solver_kwargs={"max_order": 4},
        )
    if sampler == "dpm2":
        intervals = max(int(steps) - 1, 0)
        return SolverSpec(
            name="dpm2",
            kind="generic",
            configured_steps=int(steps),
            nominal_nfe=2 * intervals,
            estimated_model_forwards=2 * intervals,
            guidance_rate=float(guidance_rate),
            schedule_type="flowmatch",
            schedule_rho=7.0,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            sampler_fn=get_solver_fn("dpm2"),
            extra_solver_kwargs={},
        )
    if sampler == "edm":
        steps_int = int(steps)
        return SolverSpec(
            name="edm",
            kind="generic",
            configured_steps=steps_int,
            nominal_nfe=max((2 * steps_int) - 1, 0),
            estimated_model_forwards=max((2 * steps_int) - 1, 0),
            guidance_rate=float(guidance_rate),
            schedule_type="flowmatch",
            schedule_rho=7.0,
            sigma_min=float(sigma_min),
            sigma_max=float(sigma_max),
            sampler_fn=get_solver_fn("edm"),
            extra_solver_kwargs={},
        )
    raise ValueError(f"Unsupported baseline solver: {sampler_name}")


def _build_epd_spec(
    predictor: EPD_predictor,
    predictor_path: str,
    *,
    sigma_min: float,
    sigma_max: float,
) -> SolverSpec:
    steps = int(getattr(predictor, "num_steps"))
    num_points = int(getattr(predictor, "num_points"))
    intervals = max(steps - 1, 0)
    afs = bool(getattr(predictor, "afs", False))
    sampler_name = (getattr(predictor, "sampler_stu", "epd") or "epd").lower()
    if sampler_name == "epd":
        sampler_name = "epd_parallel"
    return SolverSpec(
        name="epd",
        kind="epd",
        configured_steps=steps,
        nominal_nfe=num_points * intervals,
        estimated_model_forwards=max((1 + num_points) * intervals - (1 if afs and intervals > 0 else 0), 0),
        guidance_rate=float(getattr(predictor, "guidance_rate")),
        schedule_type=str(getattr(predictor, "schedule_type") or "flowmatch"),
        schedule_rho=float(getattr(predictor, "schedule_rho", 7.0)),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
        sampler_fn=get_solver_fn(sampler_name),
        predictor=predictor,
        predictor_path=predictor_path,
        afs=afs,
        extra_solver_kwargs={},
    )


def _run_solver_batch(
    spec: SolverSpec,
    *,
    backend,
    backend_cfg: Dict[str, Any],
    batch_prompts: List[str],
    batch_seeds: List[int],
    target_resolution: int,
) -> tuple[torch.Tensor, float, float]:
    device = backend.pipeline._execution_device
    pipe = backend.pipeline

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    if spec.kind == "official_euler":
        generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in batch_seeds]
        result = pipe(
            prompt=batch_prompts,
            num_inference_steps=spec.configured_steps,
            guidance_scale=spec.guidance_rate,
            generator=generators,
            height=target_resolution,
            width=target_resolution,
            output_type="latent",
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_after_sampling = time.perf_counter()
        images = _decode_flux_packed_latents_to_pt(pipe, result.images, target_resolution, target_resolution)
    else:
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [len(batch_seeds), backend.img_channels, backend.img_resolution, backend.img_resolution],
            device=device,
            dtype=pipe.transformer.dtype,
        )
        condition = _prepare_flux_condition(
            backend,
            batch_prompts,
            spec.guidance_rate,
            backend_cfg,
        )
        solver_kwargs = dict(
            num_steps=spec.configured_steps,
            sigma_min=spec.sigma_min,
            sigma_max=spec.sigma_max,
            schedule_type=spec.schedule_type,
            schedule_rho=spec.schedule_rho,
            afs=spec.afs,
            predictor=spec.predictor if spec.kind == "epd" else None,
            train=False,
        )
        if spec.extra_solver_kwargs:
            solver_kwargs.update(spec.extra_solver_kwargs)
        samples, _ = spec.sampler_fn(
            net=backend,
            latents=latents,
            condition=condition,
            **solver_kwargs,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_after_sampling = time.perf_counter()
        images = backend.vae_decode(samples)
        images = torch.clamp(images / 2 + 0.5, 0, 1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_end = time.perf_counter()
    return images, t_after_sampling - t_start, t_end - t_start


def _print_markdown_table(rows: List[Dict[str, Any]]) -> None:
    headers = [
        "solver",
        "steps",
        "nominal_nfe",
        "est_fwds",
        "sample_mean",
        "sample_p95",
        "e2e_mean",
        "e2e_p95",
        "img/s",
        "peak_gb",
    ]
    print("")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        peak_gb = row["peak_memory_bytes"] / (1024**3) if row["peak_memory_bytes"] is not None else float("nan")
        values = [
            row["solver"],
            str(row["configured_steps"]),
            str(row["nominal_nfe"]),
            str(row["estimated_model_forwards"]),
            _format_seconds(row["sampling_mean"]),
            _format_seconds(row["sampling_p95"]),
            _format_seconds(row["e2e_mean"]),
            _format_seconds(row["e2e_p95"]),
            f"{row['throughput_img_per_sec']:.3f}",
            "nan" if np.isnan(peak_gb) else f"{peak_gb:.2f}",
        ]
        print("| " + " | ".join(values) + " |")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FLUX baseline solvers and EPD solver latency.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="Local FLUX snapshot path or HF repo id.",
    )
    parser.add_argument("--prompt-file", type=str, default="src/prompts/test.txt", help="Prompt file for benchmarking.")
    parser.add_argument("--seeds", type=parse_int_list, default="0-99", help="Seed range/list.")
    parser.add_argument("--batch-size", type=int, default=1, help="Benchmark batch size.")
    parser.add_argument(
        "--baseline-solvers",
        type=parse_solver_list,
        default="euler,edm,dpm2,ipndm",
        help="Comma-separated baseline solver list.",
    )
    parser.add_argument("--euler-steps", type=int, default=24, help="Official Euler steps.")
    parser.add_argument("--edm-steps", type=int, default=12, help="EDM steps.")
    parser.add_argument("--dpm2-steps", type=int, default=13, help="DPM2 steps.")
    parser.add_argument("--ipndm-steps", type=int, default=25, help="IPNDM steps.")
    parser.add_argument("--epd-predictor", type=str, default=None, help="Exported EPD predictor .pkl or run dir.")
    parser.add_argument("--epd-run-dir", type=str, default=None, help="PPO run dir to export predictor from.")
    parser.add_argument("--epd-checkpoint", type=str, default=None, help="Checkpoint path relative to run dir or absolute.")
    parser.add_argument("--outdir", type=str, default="results/flux_solver_latency", help="Benchmark output directory.")
    parser.add_argument("--warmup-prompts", type=int, default=1, help="Number of warmup images per solver.")
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save all generated images per solver.",
    )
    parser.add_argument(
        "--save-first-image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the first image per solver as a sanity check.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string.")
    args = parser.parse_args()

    if args.epd_predictor and args.epd_run_dir:
        raise SystemExit("Specify either --epd-predictor or --epd-run-dir, not both.")
    if args.epd_checkpoint and not args.epd_run_dir:
        raise SystemExit("--epd-checkpoint requires --epd-run-dir.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1.")
    if args.warmup_prompts < 0:
        raise SystemExit("--warmup-prompts must be >= 0.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    sanity_dir = outdir / "sanity_images"
    all_image_dir = outdir / "images"

    seeds = [int(seed) for seed in parse_int_list(args.seeds)]
    baseline_solvers = parse_solver_list(args.baseline_solvers)
    prompts = _load_prompts(args.prompt_file, len(seeds))
    total_images = len(seeds)
    batches = [list(range(i, min(i + args.batch_size, total_images))) for i in range(0, total_images, args.batch_size)]
    warmup_count = min(args.warmup_prompts, total_images)
    warmup_batches = [list(range(i, min(i + args.batch_size, warmup_count))) for i in range(0, warmup_count, args.batch_size)]

    resolved_flux = resolve_flux_runtime_metadata(
        backend_options={"model_name_or_path": args.model_id, "resolution": 1024},
        resolution=1024,
    )
    backend_cfg = dict(resolved_flux["backend_options"])
    backend, _ = create_model_flux(
        guidance_rate=3.5,
        guidance_type="cfg",
        device=device,
        backend_config=backend_cfg,
    )
    backend.pipeline.set_progress_bar_config(disable=True)

    predictor_path: Optional[Path] = None
    predictor: Optional[EPD_predictor] = None
    epd_resolved_flux: Optional[Dict[str, Any]] = None
    if args.epd_predictor or args.epd_run_dir:
        if args.epd_predictor:
            predictor_path = resolve_predictor_path(args.epd_predictor)
        else:
            export_dir = outdir / "epd_export"
            export_result = export_policy_mean_to_predictor(
                Path(args.epd_run_dir),
                checkpoint=Path(args.epd_checkpoint) if args.epd_checkpoint else None,
                output_dir=export_dir,
                device="cpu",
                include_manifest=True,
            )
            predictor_path = export_result.snapshot_path

        predictor = _load_predictor(Path(predictor_path), device=device)
        epd_resolved_flux = resolve_flux_runtime_metadata(
            backend_options=dict(getattr(predictor, "backend_config", {}) or backend_cfg),
            resolution=getattr(predictor, "img_resolution", None) or 1024,
            sigma_min=getattr(predictor, "sigma_min", None),
            sigma_max=getattr(predictor, "sigma_max", None),
            flowmatch_mu=getattr(predictor, "flowmatch_mu", None),
            flowmatch_shift=getattr(predictor, "flowmatch_shift", None),
        )

    specs: List[SolverSpec] = []
    for solver_name in baseline_solvers:
        if solver_name == "euler":
            steps = args.euler_steps
        elif solver_name == "edm":
            steps = args.edm_steps
        elif solver_name == "dpm2":
            steps = args.dpm2_steps
        elif solver_name == "ipndm":
            steps = args.ipndm_steps
        else:
            raise SystemExit(f"Unsupported baseline solver in --baseline-solvers: {solver_name}")
        specs.append(
            _build_baseline_spec(
                solver_name,
                steps=steps,
                guidance_rate=3.5,
                sigma_min=float(resolved_flux["sigma_min"]),
                sigma_max=float(resolved_flux["sigma_max"]),
            )
        )
    if predictor is not None and predictor_path is not None and epd_resolved_flux is not None:
        specs.append(
            _build_epd_spec(
                predictor,
                str(predictor_path),
                sigma_min=float(epd_resolved_flux["sigma_min"]),
                sigma_max=float(epd_resolved_flux["sigma_max"]),
            )
        )
    if not specs:
        raise SystemExit("Nothing to benchmark: provide --baseline-solvers and/or --epd-predictor/--epd-run-dir.")

    summary_rows: List[Dict[str, Any]] = []
    latency_records: List[Dict[str, Any]] = []

    print("[latency] configuration:")
    print(f"  - model_id: {backend_cfg['model_name_or_path']}")
    print(f"  - prompt_file: {args.prompt_file}")
    print(f"  - num_images: {total_images}")
    print(f"  - batch_size: {args.batch_size}")
    print(f"  - warmup_prompts: {warmup_count}")
    print(f"  - device: {device}")
    print(f"  - outdir: {outdir}")

    for spec in specs:
        print(
            f"[latency] benchmarking solver={spec.name} "
            f"steps={spec.configured_steps} nominal_nfe={spec.nominal_nfe} "
            f"est_fwds={spec.estimated_model_forwards}"
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Untimed warmup.
        for warm_batch in warmup_batches:
            warm_prompts = [prompts[idx] for idx in warm_batch]
            warm_seeds = [seeds[idx] for idx in warm_batch]
            with torch.no_grad():
                images, _, _ = _run_solver_batch(
                    spec,
                    backend=backend,
                    backend_cfg=backend_cfg,
                    batch_prompts=warm_prompts,
                    batch_seeds=warm_seeds,
                    target_resolution=1024,
                )
                del images

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        sampling_latencies: List[float] = []
        e2e_latencies: List[float] = []
        saved_first = False

        for batch_indices in batches:
            batch_prompts = [prompts[idx] for idx in batch_indices]
            batch_seeds = [seeds[idx] for idx in batch_indices]
            with torch.no_grad():
                images, sampling_time, e2e_time = _run_solver_batch(
                    spec,
                    backend=backend,
                    backend_cfg=backend_cfg,
                    batch_prompts=batch_prompts,
                    batch_seeds=batch_seeds,
                    target_resolution=1024,
                )

            per_image_sampling = sampling_time / len(batch_seeds)
            per_image_e2e = e2e_time / len(batch_seeds)
            sampling_latencies.extend([per_image_sampling] * len(batch_seeds))
            e2e_latencies.extend([per_image_e2e] * len(batch_seeds))

            if args.save_images:
                solver_image_dir = all_image_dir / spec.name
                for seed_value, image_tensor in zip(batch_seeds, images):
                    _save_first_image(image_tensor, solver_image_dir / f"{seed_value:06d}.png")

            if args.save_first_image and not saved_first and images.shape[0] > 0:
                _save_first_image(images[0], sanity_dir / f"{spec.name}.png")
                saved_first = True

            for seed_value, prompt_str in zip(batch_seeds, batch_prompts):
                latency_records.append(
                    {
                        "solver": spec.name,
                        "seed": int(seed_value),
                        "prompt": prompt_str,
                        "configured_steps": spec.configured_steps,
                        "nominal_nfe": spec.nominal_nfe,
                        "estimated_model_forwards": spec.estimated_model_forwards,
                        "sampling_sec": per_image_sampling,
                        "e2e_sec": per_image_e2e,
                    }
                )

        peak_memory_bytes = None
        if device.type == "cuda":
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))

        sampling_stats = _stats(sampling_latencies)
        e2e_stats = _stats(e2e_latencies)
        throughput = float(len(seeds) / sum(e2e_latencies)) if e2e_latencies and sum(e2e_latencies) > 0 else float("nan")
        summary_rows.append(
            {
                "solver": spec.name,
                "configured_steps": spec.configured_steps,
                "nominal_nfe": spec.nominal_nfe,
                "estimated_model_forwards": spec.estimated_model_forwards,
                "num_images": len(seeds),
                "sampling_mean": sampling_stats["mean"],
                "sampling_std": sampling_stats["std"],
                "sampling_p50": sampling_stats["p50"],
                "sampling_p90": sampling_stats["p90"],
                "sampling_p95": sampling_stats["p95"],
                "e2e_mean": e2e_stats["mean"],
                "e2e_std": e2e_stats["std"],
                "e2e_p50": e2e_stats["p50"],
                "e2e_p90": e2e_stats["p90"],
                "e2e_p95": e2e_stats["p95"],
                "throughput_img_per_sec": throughput,
                "peak_memory_bytes": peak_memory_bytes,
                "prompt_file": args.prompt_file,
                "seeds": f"{seeds[0]}-{seeds[-1]}" if seeds else "",
                "batch_size": args.batch_size,
                "schedule_type": spec.schedule_type,
                "schedule_rho": spec.schedule_rho,
                "guidance_rate": spec.guidance_rate,
                "sigma_min": spec.sigma_min,
                "sigma_max": spec.sigma_max,
                "predictor_path": spec.predictor_path,
            }
        )

    summary_payload = {
        "meta": {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "diffusers_version": _safe_import_version("diffusers"),
            "transformers_version": _safe_import_version("transformers"),
            "huggingface_hub_version": _safe_import_version("huggingface_hub"),
            "dtype": str(backend.pipeline.transformer.dtype).replace("torch.", ""),
            "model_id": backend_cfg["model_name_or_path"],
        },
        "benchmark": {
            "prompt_file": args.prompt_file,
            "num_images": total_images,
            "batch_size": args.batch_size,
            "warmup_prompts": warmup_count,
            "save_images": bool(args.save_images),
            "save_first_image": bool(args.save_first_image),
            "timing_scope_sampling": "conditioning + latent generation + solver denoising (no decode, no image writes)",
            "timing_scope_e2e": "conditioning + latent generation + solver denoising + decode/postprocess (no image writes)",
            "epd_predictor_path": str(predictor_path) if predictor_path is not None else None,
        },
        "solvers": summary_rows,
    }

    summary_path = outdir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    csv_path = outdir / "latency_records.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "solver",
            "seed",
            "prompt",
            "configured_steps",
            "nominal_nfe",
            "estimated_model_forwards",
            "sampling_sec",
            "e2e_sec",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(latency_records)

    _print_markdown_table(summary_rows)
    print(f"[latency] wrote summary JSON to {summary_path}")
    print(f"[latency] wrote per-image CSV to {csv_path}")


if __name__ == "__main__":
    main()


'''

python benchmark_flux_solver_latency.py \
  --model-id /work/nvme/betk/zli42/RLEPD/weights/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --batch-size 1 \
  --baseline-solvers euler,edm,dpm2,ipndm \
  --euler-steps 16 \
  --edm-steps 8 \
  --dpm2-steps 9 \
  --ipndm-steps 17 \
  --outdir results/latency_baseline_16

python benchmark_flux_solver_latency.py \
  --model-id /work/nvme/betk/zli42/RLEPD/weights/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --batch-size 1 \
  --baseline-solvers euler,edm,dpm2,ipndm \
  --euler-steps 20 \
  --edm-steps 10 \
  --dpm2-steps 11 \
  --ipndm-steps 21 \
  --outdir results/latency_baseline_20

python benchmark_flux_solver_latency.py \
  --model-id /work/nvme/betk/zli42/RLEPD/weights/hf_cache/hub/models--black-forest-labs--FLUX.1-dev/snapshots/3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 \
  --prompt-file src/prompts/test.txt \
  --seeds 0-99 \
  --batch-size 1 \
  --baseline-solvers euler,edm,dpm2,ipndm \
  --euler-steps 24 \
  --edm-steps 12 \
  --dpm2-steps 13 \
  --ipndm-steps 25 \
  --epd-predictor exps/20260402-094545-flux_dev/export/network-snapshot-export-step002800.pkl \
  --outdir results/latency_baseline_epd_24
  
'''
