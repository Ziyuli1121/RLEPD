"""
Utility CLI to compute ImageReward scores for a directory of generated images.

Usage example:
    python -m training.ppo.scripts.score_imagereward \\
        --images exps/<run-id>/samples \\
        --prompts prompts.csv \\
        --weights weights/ImageReward.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import warnings
import os
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import torch

from training.ppo.pipeline_utils import (
    WEIGHTS_ROOT,
    collect_image_files,
    load_prompts_file,
    resolve_weight_path,
    summarize_scores,
)
from training.ppo.reward_models.imagereward.utils import load as load_imagereward

warnings.filterwarnings("ignore", category=FutureWarning, module=r"timm.*")


def _build_reward(args: argparse.Namespace):
    cache_dir = args.cache_dir.expanduser() if args.cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    download_root = str(cache_dir) if cache_dir is not None else None

    weights_path = resolve_weight_path("imagereward", args.weights) or args.weights.expanduser()
    weight_ref = str(weights_path)
    if not weights_path.exists():
        weight_ref = os.environ.get("IMAGEREWARD_WEIGHTS_NAME", "ImageReward-v1.0")

    med_config = os.environ.get("IMAGEREWARD_MED_CONFIG")
    if med_config is None:
        local_med = WEIGHTS_ROOT / "med_config.json"
        if local_med.is_file():
            med_config = str(local_med)
    reward = load_imagereward(
        name=weight_ref,
        device=args.device,
        download_root=download_root,
        med_config=med_config,
    )
    return reward, weight_ref, download_root, med_config


def _score(
    reward,
    image_files: Sequence[Path],
    prompts: Sequence[str],
    batch_size: int,
    device_hint: str,
) -> tuple[torch.Tensor, dict]:
    if len(image_files) != len(prompts):
        raise RuntimeError(f"图像文件数量 ({len(image_files)}) 与 prompt 数量 ({len(prompts)}) 不一致。")

    values: List[float] = []
    start = time.time()

    for offset in range(0, len(image_files), batch_size):
        chunk_files = image_files[offset : offset + batch_size]
        chunk_prompts = prompts[offset : offset + batch_size]
        for path, prompt in zip(chunk_files, chunk_prompts):
            value = reward.score(prompt, str(path))
            values.append(float(value))

    duration = time.time() - start
    tensor = torch.tensor(values, dtype=torch.float32)
    metadata = {
        "duration": duration,
        "num_images": len(values),
        "batch_size": batch_size,
        "device": str(getattr(reward, "device", device_hint)),
    }
    return tensor, metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute ImageReward scores for generated images.")
    parser.add_argument("--images", type=Path, required=True, help="包含生成图像的目录。")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="匹配图像文件的 glob pattern（默认: *.png）。",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        required=True,
        help="Prompt 列表文件（CSV 需包含 text 列，或纯文本每行一个 prompt）。",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/ImageReward.pt"),
        help="ImageReward 模型 checkpoint 路径（默认自动下载官方权重）。",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("weights"),
        help="ImageReward 模型缓存目录。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="评估时的 batch size。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行 ImageReward 评估的设备。",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="保留参数以兼容 score_hps 接口；ImageReward 不支持 AMP 切换。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="可选：将结果写入 JSON 文件。",
    )

    args = parser.parse_args(argv)
    images_dir = args.images.resolve()
    prompts_path = args.prompts.resolve()

    image_files = collect_image_files(images_dir, args.pattern)
    prompts = load_prompts_file(prompts_path)

    reward, weight_ref, download_root, med_config = _build_reward(args)
    scores, meta = _score(reward, image_files, prompts, args.batch_size, args.device)
    stats = summarize_scores(scores)

    meta["weights"] = weight_ref
    if download_root is not None:
        meta["cache_dir"] = download_root
    if med_config is not None:
        meta["med_config"] = med_config

    result = {
        "images_dir": str(images_dir),
        "pattern": args.pattern,
        "prompts_file": str(prompts_path),
        "stats": stats,
        "metadata": meta,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
