#!/usr/bin/env python3
"""
Visualise Dirichlet parameters derived from an EPD predictor checkpoint.

The script loads a distilled predictor snapshot (.npz/.pkl) via
`training.ppo.cold_start.load_predictor_table`, converts its positions/weights
into Dirichlet concentrations with `build_dirichlet_params`, and renders the
concentration tensors as heatmaps. Optionally, it also samples from a specific
step's Dirichlet factors and plots 1-D histograms for `r` (positions) and `w`
(weights) to provide intuition about the distribution shape.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

# Ensure the repository root (where `training` lives) is on sys.path.
import sys

import math

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:  # pragma: no cover - type helper
    _MATPLOTLIB_IMPORT_ERROR = None

from training.ppo.cold_start import build_dirichlet_params, load_predictor_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise Dirichlet parameters for RLEPD.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to predictor checkpoint (.npz or .pkl).",
    )
    parser.add_argument(
        "--concentration",
        type=float,
        default=200.0,
        help="Shared Dirichlet concentration (matches PPO config; larger = sharper).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the heatmap figure. If omitted, only shows interactively.",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=None,
        help="Optional coarse step index to sample and inspect (0 <= idx < num_steps-1).",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=None,
        help="Optional path to save the sample histogram figure.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2048,
        help="Number of samples to draw per Dirichlet when --sample-step is set.",
    )
    parser.add_argument(
        "--max-dims",
        type=int,
        default=6,
        help="Maximum number of dimensions to plot in the sample histograms (avoids clutter).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when drawing samples.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI when saving to disk.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive plt.show() (useful for headless environments).",
    )
    parser.add_argument(
        "--surface-step",
        type=int,
        default=None,
        help="Optional step index for 3D surface visualisation. Requires K=2 (weights) or segments=3.",
    )
    parser.add_argument(
        "--surface-target",
        choices=("weight", "position"),
        default="weight",
        help="Choose whether the 3D surface is drawn from weight Dirichlet (K=2) or position segments (requires num_points=2).",
    )
    parser.add_argument(
        "--surface-output",
        type=Path,
        default=None,
        help="Optional path to save the 3D surface plot.",
    )
    parser.add_argument(
        "--surface-resolution",
        type=int,
        default=200,
        help="Grid resolution for the 3D surface plot (larger = smoother but slower).",
    )
    return parser.parse_args()


def _format_row(name: str, values: np.ndarray) -> str:
    parts = [f"{name}{idx}:{float(val):.5f}" for idx, val in enumerate(values)]
    return " ".join(parts)


def print_parameter_table(table) -> None:
    """Print the raw predictor table (positions & weights) step by step."""

    num_steps = table.num_steps - 1
    num_points = table.num_points
    print(
        f"[Predictor Table] steps={num_steps}, points={num_points}, "
        f"schedule={table.schedule_type}, rho={table.schedule_rho}"
    )
    for step in range(num_steps):
        r_line = _format_row("r", table.positions[step])
        w_line = _format_row("w", table.weights[step])
        print(f"step {step:02d} | {r_line} | {w_line}")


def plot_heatmaps(alpha_pos: np.ndarray, alpha_weight: np.ndarray) -> plt.Figure:
    """Render heatmaps for alpha tensors (steps x dims)."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    pos_img = axes[0].imshow(alpha_pos, aspect="auto", origin="lower")
    axes[0].set_title("Dirichlet α (segments → r)")
    axes[0].set_xlabel("Segment index")
    axes[0].set_ylabel("Step index")
    fig.colorbar(pos_img, ax=axes[0], label="Concentration")

    weight_img = axes[1].imshow(alpha_weight, aspect="auto", origin="lower")
    axes[1].set_title("Dirichlet α (weights w)")
    axes[1].set_xlabel("Weight index")
    axes[1].set_ylabel("Step index")
    fig.colorbar(weight_img, ax=axes[1], label="Concentration")
    return fig


def _plot_histograms(
    ax: plt.Axes,
    samples: np.ndarray,
    labels: Iterable[str],
    max_dims: int,
    title: str,
) -> None:
    bins = min(60, max(20, samples.shape[0] // 40))
    for dim, label in enumerate(labels):
        if dim >= max_dims:
            break
        ax.hist(
            samples[:, dim],
            bins=bins,
            density=True,
            alpha=0.6,
            label=label,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_title(title)
    ax.set_ylabel("Density")
    ax.set_xlabel("Value")
    ax.legend(loc="upper right", fontsize="small")


def plot_step_samples(
    alpha_pos: np.ndarray,
    alpha_weight: np.ndarray,
    step_idx: int,
    num_samples: int,
    max_dims: int,
    seed: int,
) -> plt.Figure:
    """Draw samples from one step's Dirichlet factors and plot histograms for r / w."""

    rng = np.random.default_rng(seed)

    segments = rng.dirichlet(alpha_pos[step_idx], size=num_samples)
    positions = np.cumsum(segments[..., :-1], axis=-1)

    weights = rng.dirichlet(alpha_weight[step_idx], size=num_samples)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    position_labels = [f"r[{i}]" for i in range(positions.shape[1])]
    weight_labels = [f"w[{i}]" for i in range(weights.shape[1])]

    _plot_histograms(
        axes[0],
        positions,
        position_labels,
        max_dims=max_dims,
        title=f"Step {step_idx}: r samples (derived from segments)",
    )
    _plot_histograms(
        axes[1],
        weights,
        weight_labels,
        max_dims=max_dims,
        title=f"Step {step_idx}: w samples",
    )
    return fig


def _beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Compute Beta PDF robustly."""

    x = np.clip(x, 1e-9, 1 - 1e-9)
    log_norm = math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    log_pdf = (alpha - 1.0) * np.log(x) + (beta - 1.0) * np.log(1.0 - x)
    return np.exp(log_norm + log_pdf)


def plot_beta_surface(alpha: np.ndarray, step_idx: int, resolution: int) -> plt.Figure:
    """Render a 3D surface for a Beta distribution (Dirichlet with K=2)."""

    a0, a1 = alpha.astype(np.float64)
    x = np.linspace(1e-3, 1 - 1e-3, resolution)
    y = np.linspace(0.0, 1.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = _beta_pdf(X, a0, a1)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title(f"Step {step_idx}: Dirichlet (weights) as Beta surface")
    ax.set_xlabel("w[0]")
    ax.set_ylabel("aux-axis")
    ax.set_zlabel("Density")
    return fig


def _dirichlet_pdf(points: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compute Dirichlet PDF for 3 components."""

    points = np.clip(points, 1e-9, 1.0)
    alpha = alpha.astype(np.float64)
    log_norm = math.lgamma(alpha.sum()) - np.sum([math.lgamma(a) for a in alpha])
    log_pdf = ((alpha - 1.0) * np.log(points)).sum(axis=-1)
    return np.exp(log_norm + log_pdf)


def plot_simplex_surface(alpha: np.ndarray, step_idx: int, resolution: int) -> plt.Figure:
    """Plot a Dirichlet PDF over the 2-simplex (3 components)."""

    grid = np.linspace(0.0, 1.0, resolution)
    U, V = np.meshgrid(grid, grid)
    mask = U + V <= 1.0
    x = U[mask]
    y = V[mask]
    z = 1.0 - x - y
    coords = np.stack([x, y, z], axis=-1)
    density = _dirichlet_pdf(coords, alpha)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(x, y, density, cmap="plasma", linewidth=0.1, antialiased=True)
    ax.set_title(f"Step {step_idx}: Dirichlet simplex surface (segments)")
    ax.set_xlabel("Segment 0")
    ax.set_ylabel("Segment 1")
    ax.set_zlabel("Density")
    return fig


def main() -> None:
    args = parse_args()

    if plt is None:
        raise ModuleNotFoundError(
            "matplotlib is required to visualise Dirichlet parameters. Install it via "
            "`pip install matplotlib` within your environment."
        ) from _MATPLOTLIB_IMPORT_ERROR

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    table = load_predictor_table(checkpoint)
    print_parameter_table(table)
    dirichlet_init = build_dirichlet_params(table, concentration=args.concentration)

    heatmap_fig = plot_heatmaps(dirichlet_init.alpha_pos, dirichlet_init.alpha_weight)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        heatmap_fig.savefig(args.output, dpi=args.dpi)

    sample_fig: Optional[plt.Figure] = None
    if args.sample_step is not None:
        step = int(args.sample_step)
        max_idx = dirichlet_init.alpha_pos.shape[0] - 1
        if not 0 <= step <= max_idx:
            raise ValueError(f"--sample-step must be in [0, {max_idx}], got {step}.")
        sample_fig = plot_step_samples(
            dirichlet_init.alpha_pos,
            dirichlet_init.alpha_weight,
            step_idx=step,
            num_samples=args.num_samples,
            max_dims=args.max_dims,
            seed=args.seed,
        )
        if args.sample_output is not None:
            args.sample_output.parent.mkdir(parents=True, exist_ok=True)
            sample_fig.savefig(args.sample_output, dpi=args.dpi)

    surface_fig: Optional[plt.Figure] = None
    if args.surface_step is not None:
        step = int(args.surface_step)
        max_idx = dirichlet_init.alpha_pos.shape[0] - 1
        if not 0 <= step <= max_idx:
            raise ValueError(f"--surface-step must be in [0, {max_idx}], got {step}.")

        if args.surface_target == "weight":
            alpha_vec = dirichlet_init.alpha_weight[step]
            if alpha_vec.shape[-1] != 2:
                raise ValueError(
                    "Weight surface requires num_points == 2. "
                    f"Got {alpha_vec.shape[-1]} points. Use --surface-target position for segments."
                )
            surface_fig = plot_beta_surface(alpha_vec, step, args.surface_resolution)
        else:
            alpha_vec = dirichlet_init.alpha_pos[step]
            if alpha_vec.shape[-1] != 3:
                raise ValueError(
                    "Position surface requires num_points == 2 (segments=3). "
                    f"Got {alpha_vec.shape[-1]} segments."
                )
            surface_fig = plot_simplex_surface(alpha_vec, step, args.surface_resolution)

        if args.surface_output is not None and surface_fig is not None:
            args.surface_output.parent.mkdir(parents=True, exist_ok=True)
            surface_fig.savefig(args.surface_output, dpi=args.dpi)

    if args.no_show:
        plt.close(heatmap_fig)
        if sample_fig is not None:
            plt.close(sample_fig)
        if surface_fig is not None:
            plt.close(surface_fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()


'''

python visualize_dirichlet.py \
    --checkpoint /work/nvme/betk/zli42/RLEPD/exps/20251206-131339-sd3_1024/export/network-snapshot-export-step005000.pkl \
    --output dirichlet_heatmap.png \
    --surface-step 5 \
    --surface-target position \
    --surface-output beta_surface_step5.png \
    --concentration 10

'''
