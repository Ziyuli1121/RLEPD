#!/usr/bin/env python3
"""
Compute FID for a generated image directory against a fixed COCO real subset.

This script is the formal RLEPD entrypoint for:

  FID-10k (fixed COCO subset, clean preprocessing, native 1024 generation)

It expects:
  - fake images in a generated samples directory
  - a CSV manifest with at least a `file_name` column
  - a real-image directory corresponding to that fixed manifest subset

The script uses clean-fid when available, but does not install dependencies.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
import shutil
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from training.ppo.pipeline_utils import collect_image_files


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "src" / "prompts" / "coco10k.csv"
DEFAULT_REAL_IMAGES = REPO_ROOT / "src" / "coco10k_real_val2014"
DEFAULT_CACHE_DIR = REPO_ROOT / "results" / "fid_cache"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute FID-10k for generated images against a fixed COCO real subset."
    )
    parser.add_argument("--images", type=Path, required=True, help="包含 fake 图像的目录。")
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.png",
        help="匹配 fake 图像的 glob pattern（默认: **/*.png）。",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Formal protocol manifest CSV（默认: src/prompts/coco10k.csv）。",
    )
    parser.add_argument(
        "--real-images",
        type=Path,
        default=DEFAULT_REAL_IMAGES,
        help="固定 real subset 目录（默认: src/coco10k_real_val2014）。",
    )
    parser.add_argument(
        "--real-stats",
        type=Path,
        help="可选：复用已缓存的 clean-fid real stats 文件。",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="用于 fake staging 和 cached real stats 的目录（默认: results/fid_cache）。",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="clean",
        help="clean-fid mode（默认: clean）。",
    )
    parser.add_argument(
        "--eval-res",
        type=int,
        default=256,
        help="Formal evaluation resolution label（默认: 256）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="FID 特征提取 batch size（默认: 32）。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="clean-fid dataloader worker 数（默认: 2）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行 clean-fid 的设备。",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="可选：将结果写入 JSON 文件。",
    )
    return parser.parse_args(argv)


def _resolve_distribution_version() -> str:
    for dist_name in ("clean-fid", "cleanfid"):
        try:
            return str(importlib_metadata.version(dist_name))
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unknown"


def _load_cleanfid():
    try:
        module = importlib.import_module("cleanfid")
        fid_module = importlib.import_module("cleanfid.fid")
    except ImportError as exc:
        raise SystemExit(
            "cleanfid is not installed in the active environment. "
            "Install it explicitly with: pip install clean-fid"
        ) from exc
    return module, fid_module


def _call_with_supported_kwargs(fn, *args, **kwargs):
    signature = inspect.signature(fn)
    accepted = {name for name in signature.parameters}
    filtered = {key: value for key, value in kwargs.items() if key in accepted and value is not None}
    return fn(*args, **filtered)


def _load_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [field for field in (reader.fieldnames or []) if field]
        if "file_name" not in fieldnames:
            raise RuntimeError(f"Manifest does not contain a 'file_name' column: {manifest_path}")
        rows: List[Dict[str, str]] = []
        for raw_row in reader:
            row = {key: str(value).strip() for key, value in raw_row.items() if key}
            if row.get("file_name"):
                rows.append(row)
    if not rows:
        raise RuntimeError(f"No valid manifest rows found in {manifest_path}")
    file_names = [row["file_name"] for row in rows]
    if len(set(file_names)) != len(file_names):
        raise RuntimeError(f"Manifest contains duplicate file_name entries: {manifest_path}")
    return rows


def _list_image_files_in_dir(directory: Path) -> List[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _validate_protocol(
    *,
    image_files: Sequence[Path],
    manifest_path: Path,
    manifest_rows: Sequence[Mapping[str, str]],
    real_images_dir: Path,
) -> Tuple[int, int]:
    num_manifest = len(manifest_rows)
    num_fake = len(image_files)
    if num_fake != num_manifest:
        raise RuntimeError(
            f"Fake image count ({num_fake}) does not match manifest row count ({num_manifest}) for {manifest_path}."
        )

    manifest_filenames = [row["file_name"] for row in manifest_rows]
    missing_real = [name for name in manifest_filenames if not (real_images_dir / name).is_file()]
    if missing_real:
        raise RuntimeError(
            f"Real subset is missing {len(missing_real)} manifest files under {real_images_dir}. "
            f"First missing entries: {missing_real[:10]}"
        )

    real_image_files = _list_image_files_in_dir(real_images_dir)
    num_real = len(real_image_files)
    if num_real != num_manifest:
        raise RuntimeError(
            f"Real image count ({num_real}) does not match manifest row count ({num_manifest}) for {real_images_dir}."
        )

    return num_fake, num_real


def _sanitize_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def _protocol_custom_name(manifest_path: Path, real_images_dir: Path, mode: str, eval_res: int) -> str:
    stable = f"{manifest_path.resolve()}::{real_images_dir.resolve()}::{mode}::res{eval_res}"
    digest = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:10]
    prefix = _sanitize_name(f"{manifest_path.stem}__{real_images_dir.name}__{mode}__res{eval_res}")
    return f"{prefix}__{digest}"


def _fake_stage_key(images_dir: Path, pattern: str, image_files: Sequence[Path]) -> str:
    if not image_files:
        raise RuntimeError("No fake image files available for staging.")
    stable = "::".join(
        [
            str(images_dir.resolve()),
            pattern,
            str(len(image_files)),
            image_files[0].name,
            image_files[-1].name,
        ]
    )
    return hashlib.sha1(stable.encode("utf-8")).hexdigest()[:12]


def _materialize_link(src: Path, dst: Path) -> str:
    try:
        dst.symlink_to(src.resolve())
        return "symlink"
    except OSError:
        pass
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _stage_fake_images(cache_dir: Path, images_dir: Path, pattern: str, image_files: Sequence[Path]) -> Tuple[Path, str]:
    stage_key = _fake_stage_key(images_dir, pattern, image_files)
    stage_root = cache_dir / "fake_stage" / stage_key
    stage_root.mkdir(parents=True, exist_ok=True)
    stage_meta_path = stage_root / "stage_meta.json"
    staged_pngs = sorted(stage_root.glob("*.png"))
    if stage_meta_path.is_file() and len(staged_pngs) == len(image_files):
        try:
            with stage_meta_path.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            return stage_root, str(meta.get("staging_mode", "reused"))
        except json.JSONDecodeError:
            pass

    for path in stage_root.iterdir():
        if path.name == "stage_meta.json":
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    staging_mode = "symlink"
    for idx, src in enumerate(image_files):
        dst = stage_root / f"{idx:06d}{src.suffix.lower()}"
        used_mode = _materialize_link(src, dst)
        if idx == 0:
            staging_mode = used_mode

    with stage_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "images_dir": str(images_dir.resolve()),
                "pattern": pattern,
                "count": len(image_files),
                "staging_mode": staging_mode,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return stage_root, staging_mode


def _cleanfid_stats_dirs(cleanfid_module) -> List[Path]:
    package_root = Path(cleanfid_module.__file__).resolve().parent
    candidates = [
        package_root / "stats",
        Path.home() / ".cache" / "cleanfid" / "stats",
    ]
    deduped: List[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if str(resolved) not in seen:
            seen.add(str(resolved))
            deduped.append(resolved)
    return deduped


def _find_internal_stats_file(cleanfid_module, custom_name: str, mode: str) -> Path | None:
    candidates = []
    for stats_dir in _cleanfid_stats_dirs(cleanfid_module):
        if not stats_dir.exists():
            continue
        for path in stats_dir.glob(f"*{custom_name}*.npz"):
            if custom_name in path.name and mode in path.name:
                candidates.append(path.resolve())
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime)
    return candidates[-1]


def _copy_stats_into_cleanfid_cache(cleanfid_module, source_stats: Path, internal_name: str) -> None:
    installed = False
    for stats_dir in _cleanfid_stats_dirs(cleanfid_module):
        try:
            stats_dir.mkdir(parents=True, exist_ok=True)
            target = stats_dir / internal_name
            if not target.exists():
                shutil.copy2(source_stats, target)
            installed = True
        except OSError:
            continue
    if not installed:
        raise RuntimeError(
            f"Could not install clean-fid custom stats into any known stats directory. Source: {source_stats}"
        )


def _prepare_real_stats(
    *,
    cleanfid_pkg,
    fid_module,
    manifest_path: Path,
    real_images_dir: Path,
    cache_dir: Path,
    mode: str,
    eval_res: int,
    batch_size: int,
    num_workers: int,
    device: str,
    explicit_real_stats: Path | None,
) -> Tuple[Path, str, bool]:
    real_stats_dir = cache_dir / "real_stats"
    real_stats_dir.mkdir(parents=True, exist_ok=True)

    if explicit_real_stats is not None:
        stats_path = explicit_real_stats.expanduser().resolve()
        if not stats_path.is_file():
            raise RuntimeError(f"Requested real stats file does not exist: {stats_path}")
        sidecar_path = stats_path.with_suffix(".json")
        sidecar = {}
        if sidecar_path.is_file():
            with sidecar_path.open("r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
        custom_name = str(sidecar.get("custom_name") or stats_path.stem)
        internal_name = str(sidecar.get("internal_stats_filename") or stats_path.name)
        _copy_stats_into_cleanfid_cache(cleanfid_pkg, stats_path, internal_name)
        return stats_path, custom_name, False

    custom_name = _protocol_custom_name(manifest_path, real_images_dir, mode, eval_res)
    cache_npz = real_stats_dir / f"{custom_name}.npz"
    cache_sidecar = real_stats_dir / f"{custom_name}.json"
    if cache_npz.is_file():
        sidecar = {}
        if cache_sidecar.is_file():
            with cache_sidecar.open("r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
        internal_name = str(sidecar.get("internal_stats_filename") or cache_npz.name)
        _copy_stats_into_cleanfid_cache(cleanfid_pkg, cache_npz, internal_name)
        return cache_npz, custom_name, False

    _call_with_supported_kwargs(
        fid_module.make_custom_stats,
        custom_name,
        str(real_images_dir),
        mode=mode,
        dataset_res=eval_res,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )

    internal_stats_path = _find_internal_stats_file(cleanfid_pkg, custom_name, mode)
    if internal_stats_path is None:
        raise RuntimeError(
            f"clean-fid did not create a discoverable stats file for custom_name='{custom_name}'."
        )

    shutil.copy2(internal_stats_path, cache_npz)
    with cache_sidecar.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "custom_name": custom_name,
                "internal_stats_filename": internal_stats_path.name,
                "manifest": str(manifest_path.resolve()),
                "real_images": str(real_images_dir.resolve()),
                "mode": mode,
                "eval_res": eval_res,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return cache_npz, custom_name, True


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    images_dir = args.images.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    real_images_dir = args.real_images.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_files = collect_image_files(images_dir, args.pattern)
    manifest_rows = _load_manifest_rows(manifest_path)
    num_fake, num_real = _validate_protocol(
        image_files=image_files,
        manifest_path=manifest_path,
        manifest_rows=manifest_rows,
        real_images_dir=real_images_dir,
    )

    cleanfid_pkg, fid_module = _load_cleanfid()
    cleanfid_version = _resolve_distribution_version()

    stage_start = time.time()
    fake_stage_dir, staging_mode = _stage_fake_images(cache_dir, images_dir, args.pattern, image_files)
    staging_duration = time.time() - stage_start

    stats_start = time.time()
    real_stats_path, custom_name, real_stats_created = _prepare_real_stats(
        cleanfid_pkg=cleanfid_pkg,
        fid_module=fid_module,
        manifest_path=manifest_path,
        real_images_dir=real_images_dir,
        cache_dir=cache_dir,
        mode=args.mode,
        eval_res=args.eval_res,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        explicit_real_stats=args.real_stats,
    )
    real_stats_prepare_duration = time.time() - stats_start

    fid_start = time.time()
    fid_score = _call_with_supported_kwargs(
        fid_module.compute_fid,
        str(fake_stage_dir),
        mode=args.mode,
        dataset_name=custom_name,
        dataset_split="custom",
        dataset_res=args.eval_res,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    fid_duration = time.time() - fid_start

    result = {
        "images_dir": str(images_dir),
        "pattern": args.pattern,
        "manifest": str(manifest_path),
        "real_images": str(real_images_dir),
        "real_stats": str(real_stats_path),
        "stats": {
            "fid": float(fid_score),
            "num_fake": num_fake,
            "num_real": num_real,
        },
        "metadata": {
            "duration": staging_duration + real_stats_prepare_duration + fid_duration,
            "staging_duration": staging_duration,
            "real_stats_prepare_duration": real_stats_prepare_duration,
            "fid_duration": fid_duration,
            "mode": args.mode,
            "eval_res": args.eval_res,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "cache_dir": str(cache_dir),
            "real_stats_path": str(real_stats_path),
            "real_stats_created": real_stats_created,
            "fake_staging_mode": staging_mode,
            "fake_stage_dir": str(fake_stage_dir),
            "custom_stats_name": custom_name,
            "protocol": "FID-10k (fixed COCO subset, clean preprocessing, native 1024 generation)",
            "cleanfid_version": cleanfid_version,
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
