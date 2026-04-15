#!/usr/bin/env python3
"""
Compute legacy Inception-FID for a generated image directory against a precomputed
reference statistics .npz file.

This protocol is intentionally separate from the clean-fid based COCO-10k path.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import dnnlib
import numpy as np

from training.ppo.pipeline_utils import collect_image_files


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REF_NPZ = REPO_ROOT / "src" / "ms_coco-512x512.npz"
DEFAULT_DETECTOR_PKL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/"
    "metrics/inception-2015-12-05.pkl"
)


class _ImageFolderDataset:
    def __init__(self, image_files: Sequence[Path]) -> None:
        self.image_files = list(image_files)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> torch.Tensor:
        from PIL import Image
        import torch

        path = self.image_files[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            array = np.asarray(image, dtype=np.uint8)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute legacy Inception-FID against a precomputed reference .npz."
    )
    parser.add_argument("--images", type=Path, required=True, help="包含 fake 图像的目录。")
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.png",
        help="匹配 fake 图像的 glob pattern（默认: **/*.png）。",
    )
    parser.add_argument(
        "--ref-npz",
        type=Path,
        default=DEFAULT_REF_NPZ,
        help="Reference Inception statistics .npz（默认: src/ms_coco-512x512.npz）。",
    )
    parser.add_argument(
        "--num-expected",
        type=int,
        default=10000,
        help="要求 fake 图数量严格等于该值（默认: 10000）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="特征提取 batch size（默认: 64）。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker 数（默认: 2）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行 Inception 特征提取的设备。",
    )
    parser.add_argument(
        "--detector-pkl",
        type=str,
        default=DEFAULT_DETECTOR_PKL,
        help="NVIDIA Inception detector pkl 的本地路径或 URL。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="可选：将结果写入 JSON 文件。",
    )
    return parser.parse_args(argv)


def _load_ref_stats(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ref = np.load(path, allow_pickle=False)
    if "mu" not in ref.files or "sigma" not in ref.files:
        raise RuntimeError(f"Reference npz must contain 'mu' and 'sigma': {path}")
    mu = np.asarray(ref["mu"])
    sigma = np.asarray(ref["sigma"])
    if mu.shape != (2048,):
        raise RuntimeError(f"Reference mu must have shape (2048,), got {mu.shape} from {path}")
    if sigma.shape != (2048, 2048):
        raise RuntimeError(f"Reference sigma must have shape (2048, 2048), got {sigma.shape} from {path}")
    return mu.astype(np.float64, copy=False), sigma.astype(np.float64, copy=False)


def _load_detector(detector_pkl: str, device: torch.device):
    import torch

    try:
        with dnnlib.util.open_url(detector_pkl, verbose=True) as handle:
            detector = pickle.load(handle).to(device)
    except Exception as exc:  # pragma: no cover - environment/network dependent
        raise RuntimeError(
            "Failed to load the Inception detector. Provide a reachable --detector-pkl "
            "URL or a local cached pkl path."
        ) from exc
    detector.eval()
    return detector


def _calculate_inception_stats(
    *,
    image_files: Sequence[Path],
    detector,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    import torch

    if len(image_files) < 2:
        raise RuntimeError(f"Need at least 2 images to compute FID, found {len(image_files)}.")

    dataset = _ImageFolderDataset(image_files)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    feature_dim = 2048
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    count = 0

    with torch.no_grad():
        for images in loader:
            if images.numel() == 0:
                continue
            images = images.to(device)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            features = detector(images, return_features=True).to(torch.float64)
            mu += features.sum(0)
            sigma += features.T @ features
            count += int(features.shape[0])

    mu /= count
    sigma -= mu.ger(mu) * count
    sigma /= count - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def _calculate_fid(mu: np.ndarray, sigma: np.ndarray, mu_ref: np.ndarray, sigma_ref: np.ndarray) -> float:
    import scipy.linalg

    mean_term = np.square(mu - mu_ref).sum()
    covmean, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = mean_term + np.trace(sigma + sigma_ref - covmean * 2.0)
    return float(np.real(fid))


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    import torch

    device = torch.device(args.device)
    images_dir = args.images.expanduser().resolve()
    ref_npz = args.ref_npz.expanduser().resolve()

    image_files = collect_image_files(images_dir, args.pattern)
    if len(image_files) != args.num_expected:
        raise RuntimeError(
            f"Fake image count ({len(image_files)}) must equal --num-expected ({args.num_expected})."
        )

    mu_ref, sigma_ref = _load_ref_stats(ref_npz)

    start = time.time()
    detector = _load_detector(args.detector_pkl, device=device)
    mu_fake, sigma_fake = _calculate_inception_stats(
        image_files=image_files,
        detector=detector,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    fid = _calculate_fid(mu_fake, sigma_fake, mu_ref, sigma_ref)
    duration = time.time() - start

    result = {
        "images_dir": str(images_dir),
        "pattern": args.pattern,
        "ref_npz": str(ref_npz),
        "stats": {
            "fid": fid,
            "num_fake": len(image_files),
        },
        "metadata": {
            "duration": duration,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": str(device),
            "detector_pkl": args.detector_pkl,
            "protocol": "legacy_fid_ms_coco_512x512_npz",
        },
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
