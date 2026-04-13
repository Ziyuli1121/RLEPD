#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed COCO real-image subset from a CSV manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV manifest containing a file_name column (for example src/prompts/coco10k.csv).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the full real COCO images (for example src/val2014).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for the fixed real-image subset.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to materialize files in the subset directory.",
    )
    return parser.parse_args()


def _load_filenames(manifest_path: Path) -> List[str]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "file_name" not in (reader.fieldnames or []):
            raise RuntimeError(f"Manifest does not contain a 'file_name' column: {manifest_path}")
        filenames = [str(row["file_name"]).strip() for row in reader if str(row.get("file_name", "")).strip()]
    if not filenames:
        raise RuntimeError(f"No file_name entries found in manifest: {manifest_path}")
    return filenames


def _materialize_file(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists():
        return
    if link_mode == "hardlink":
        os.link(src, dst)
    elif link_mode == "symlink":
        dst.symlink_to(src.resolve())
    elif link_mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported link mode: {link_mode}")


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()

    filenames = _load_filenames(manifest_path)
    missing = []

    outdir.mkdir(parents=True, exist_ok=True)
    for file_name in filenames:
        src = source_dir / file_name
        dst = outdir / file_name
        if not src.is_file():
            missing.append(file_name)
            continue
        _materialize_file(src, dst, args.link_mode)

    summary = {
        "manifest": str(manifest_path),
        "source_dir": str(source_dir),
        "outdir": str(outdir),
        "link_mode": args.link_mode,
        "requested_count": len(filenames),
        "materialized_count": len(filenames) - len(missing),
        "missing_count": len(missing),
        "missing_examples": missing[:20],
    }
    summary_path = outdir / "subset_manifest_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if missing:
        raise SystemExit(f"Missing {len(missing)} files while building subset; see {summary_path}")


if __name__ == "__main__":
    main()
