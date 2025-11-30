#!/usr/bin/env python3
"""Plot experiment metrics against training step from a JSONL log."""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot metrics over training steps from a metrics.jsonl file. "
            "Metrics to visualize are provided as command-line arguments."
        )
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=Path("/work/nvme/betk/zli42/RLEPD/exps/20251030-235041-sd15_rl_base/logs/metrics.jsonl"),
        help="Path to the metrics JSONL log file.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="Metric names to plot (space-separated).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the plot instead of displaying it.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Optional window size for moving-average smoothing. "
            "Set to 0 to disable smoothing."
        ),
    )
    return parser.parse_args()


def load_metrics(metrics_path: Path) -> List[Dict[str, float]]:
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    records: List[Dict[str, float]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                print(f"[WARN] Failed to parse JSON on line {idx}: {err}", file=sys.stderr)
                continue
            if "step" not in record:
                print(f"[WARN] Missing 'step' field on line {idx}, skipping.", file=sys.stderr)
                continue
            records.append(record)
    if not records:
        raise ValueError(f"No valid metric entries found in {metrics_path}")
    return records


def extract_series(
    records: Iterable[Dict[str, float]], metrics: Iterable[str]
) -> Tuple[List[float], Dict[str, List[float]]]:
    steps: List[float] = []
    values: Dict[str, List[float]] = {metric: [] for metric in metrics}
    warned: Dict[str, bool] = {metric: False for metric in metrics}

    for record in records:
        steps.append(record["step"])
        for metric in metrics:
            if metric in record:
                values[metric].append(record[metric])
            else:
                values[metric].append(math.nan)
                if not warned[metric]:
                    print(
                        f"[WARN] Metric '{metric}' missing for some entries; using NaN.",
                        file=sys.stderr,
                    )
                    warned[metric] = True
    return steps, values


def moving_average(values: List[float], window: int) -> List[float]:
    """Return a moving-average series; keep NaN when window is not fully populated."""
    if window <= 1:
        return list(values)

    smoothed: List[float] = []
    window_values: List[float] = []
    valid_count = 0
    running_sum = 0.0

    for value in values:
        window_values.append(value)
        if not math.isnan(value):
            running_sum += value
            valid_count += 1

        if len(window_values) > window:
            removed = window_values.pop(0)
            if not math.isnan(removed):
                running_sum -= removed
                valid_count -= 1

        if len(window_values) < window or valid_count < window:
            smoothed.append(math.nan)
        else:
            smoothed.append(running_sum / valid_count)

    return smoothed


def plot_metrics(
    steps: List[float],
    series: Dict[str, List[float]],
    title: Optional[str] = None,
    smooth_window: int = 0,
) -> None:
    plt.figure(figsize=(10, 6))
    for metric, values in series.items():
        plot_values = values
        label = metric
        if smooth_window > 1:
            plot_values = moving_average(values, smooth_window)
            label = f"{metric} (smoothed)"
        plt.plot(steps, plot_values, label=label)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(title or "Training Metrics Over Steps")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()


def main() -> None:
    args = parse_args()
    if args.smooth_window < 0:
        raise ValueError("--smooth-window must be non-negative")
    records = load_metrics(args.metrics_file)
    steps, metric_series = extract_series(records, args.metrics)

    if args.smooth_window > 1:
        print(f"[INFO] Applying moving-average smoothing with window={args.smooth_window}")
    elif args.smooth_window == 1:
        print("[INFO] Smoothing window of 1 has no effect; plotting raw values.")

    plot_metrics(steps, metric_series, args.title, smooth_window=args.smooth_window)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()


'''

python exp_visuals/scripts/plot_metrics.py \
    --metrics-file /work/nvme/betk/zli42/RLEPD/exps/20251123-215008-sd3_smoke/logs/metrics.jsonl \
    --metrics hps_mean \
    --smooth-window 500 \
    --output exp_visuals/sd3/hps_mean.png

python exp_visuals/scripts/plot_metrics.py \
    --metrics-file /work/nvme/betk/zli42/RLEPD/exps/20251123-215008-sd3_smoke/logs/metrics.jsonl \
    --metrics kl \
    --smooth-window 0 \
    --output exp_visuals/sd3/kl.png

'''
