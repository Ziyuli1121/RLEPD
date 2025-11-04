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


def plot_metrics(
    steps: List[float], series: Dict[str, List[float]], title: Optional[str] = None
) -> None:
    plt.figure(figsize=(10, 6))
    for metric, values in series.items():
        plt.plot(steps, values, label=metric)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(title or "Training Metrics Over Steps")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()


def main() -> None:
    args = parse_args()
    records = load_metrics(args.metrics_file)
    steps, metric_series = extract_series(records, args.metrics)

    plot_metrics(steps, metric_series, args.title)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output)
        print(f"Saved plot to {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()


'''

python3 exp_visuals/scripts/plot_metrics.py --metrics mixed_reward_mean --output exp_visuals/output/reward.png

'''
