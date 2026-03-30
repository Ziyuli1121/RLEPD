from __future__ import annotations

import csv
import importlib.util
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPS_ROOT = REPO_ROOT / "exps"
WEIGHTS_ROOT = REPO_ROOT / "weights"
DEFAULT_PROMPTS_TXT = REPO_ROOT / "src" / "prompts" / "test.txt"
DEFAULT_PROMPTS_CSV = REPO_ROOT / "src" / "prompts" / "MS-COCO_val2014_30k_captions.csv"


_WEIGHT_ALIASES = {
    "hps": [WEIGHTS_ROOT / "HPS_v2.1_compressed.pt"],
    "imagereward": [WEIGHTS_ROOT / "ImageReward.pt", WEIGHTS_ROOT / "ImageReward-v1.0.pt"],
    "pickscore": [WEIGHTS_ROOT / "PickScore_v1"],
    "clip": [WEIGHTS_ROOT / "clip"],
    "aesthetic": [WEIGHTS_ROOT / "sac+logos+ava1-l14-linearMSE.pth"],
    "mps": [WEIGHTS_ROOT / "MPS_overall_checkpoint.pth"],
}

_PROMPT_COLUMNS = ("text", "prompt", "caption")


def bootstrap_local_diffusers() -> None:
    # The vendored diffusers tree in this repo tracks a Python >=3.10 codebase.
    # When running under Python 3.9, prefer an installed site-packages diffusers
    # if available instead of prepending the local vendored source tree.
    if sys.version_info < (3, 10):
        installed = importlib.util.find_spec("diffusers")
        if installed is not None:
            return
    local_src = REPO_ROOT / "diffusers" / "src"
    if local_src.is_dir():
        local_src_str = str(local_src)
        if local_src_str not in sys.path:
            sys.path.insert(0, local_src_str)


def bootstrap_local_hps() -> None:
    local_root = REPO_ROOT / "HPSv2"
    if local_root.is_dir():
        local_root_str = str(local_root)
        if local_root_str not in sys.path:
            sys.path.insert(0, local_root_str)


def bootstrap_local_taming() -> None:
    local_root = REPO_ROOT / "src" / "taming-transformers"
    if local_root.is_dir():
        local_root_str = str(local_root)
        if local_root_str not in sys.path:
            sys.path.insert(0, local_root_str)


def first_existing_path(candidates: Sequence[Path | str | None]) -> Optional[Path]:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path.resolve()
    return None


def weight_candidates(kind: str, preferred: Path | str | None = None) -> List[Path]:
    candidates: List[Path] = []
    if preferred is not None:
        candidates.append(Path(preferred).expanduser())
    for alias in _WEIGHT_ALIASES.get(kind, []):
        if alias not in candidates:
            candidates.append(alias)
    return candidates


def resolve_weight_path(kind: str, preferred: Path | str | None = None) -> Optional[Path]:
    return first_existing_path(weight_candidates(kind, preferred))


def load_prompts_file(path: Path | str) -> List[str]:
    prompt_path = Path(path).expanduser()
    suffix = prompt_path.suffix.lower()
    prompts: List[str] = []
    if suffix == ".csv":
        with prompt_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [field for field in (reader.fieldnames or []) if field]
            preferred_columns = [column for column in _PROMPT_COLUMNS if column in fieldnames]
            for row in reader:
                text = ""
                if preferred_columns:
                    for column in preferred_columns:
                        value = row.get(column, "")
                        if value and value.strip():
                            text = value.strip()
                            break
                else:
                    for column in fieldnames:
                        value = row.get(column, "")
                        if value and value.strip():
                            text = value.strip()
                            break
                if text:
                    prompts.append(text)
    else:
        with prompt_path.open("r", encoding="utf-8") as handle:
            prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise RuntimeError(f"No prompts found in '{prompt_path}'.")
    return prompts


def default_prompt_path() -> Path:
    prompt_path = first_existing_path([DEFAULT_PROMPTS_TXT, DEFAULT_PROMPTS_CSV])
    if prompt_path is None:
        raise FileNotFoundError(
            "No default prompt file found under src/prompts. Expected test.txt or MS-COCO prompt CSV."
        )
    return prompt_path


def default_prompt_csv_path() -> Path:
    prompt_path = first_existing_path([DEFAULT_PROMPTS_CSV])
    if prompt_path is None:
        raise FileNotFoundError(f"Default prompt CSV not found: {DEFAULT_PROMPTS_CSV}")
    return prompt_path


def repeat_to_length(items: Sequence[str], count: int) -> List[str]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return []
    if not items:
        raise ValueError("items must not be empty")
    repeats = (count + len(items) - 1) // len(items)
    return (list(items) * repeats)[:count]


def prompt_slice(prompts: Sequence[str], start: int, size: int) -> List[str]:
    if size < 0:
        raise ValueError("size must be non-negative")
    if size == 0:
        return []
    if not prompts:
        raise ValueError("prompts must not be empty")
    total = len(prompts)
    if start >= 0 and start + size <= total:
        return list(prompts[start : start + size])
    return [prompts[(start + idx) % total] for idx in range(size)]


def collect_image_files(directory: Path | str, pattern: str) -> List[Path]:
    image_dir = Path(directory).expanduser()
    files = sorted(image_dir.glob(pattern))
    if not files:
        raise RuntimeError(f"在 {image_dir} 下未找到匹配 {pattern} 的图像文件。")
    return files


def _extract_step(path: Path) -> int:
    match = re.search(r"(\d+)(?=\.pkl$)", path.name)
    if match:
        return int(match.group(1))
    return -1


def _latest_predictor_in_dir(directory: Path) -> Optional[Path]:
    search_groups = (
        directory.glob("export/network-snapshot-export-*.pkl"),
        directory.glob("network-snapshot-export-*.pkl"),
        directory.glob("network-snapshot-*.pkl"),
    )
    for group in search_groups:
        candidates = sorted((path for path in group if path.is_file()), key=lambda item: (_extract_step(item), item.name))
        if candidates:
            return candidates[-1].resolve()
    return None


def resolve_predictor_path(identifier: str | Path, exps_root: Path | str | None = None) -> Path:
    raw = str(identifier).strip()
    if not raw:
        raise FileNotFoundError("predictor path is empty")

    root = Path(exps_root).expanduser().resolve() if exps_root is not None else EXPS_ROOT.resolve()
    direct = Path(raw).expanduser()
    search_targets: List[Path] = [direct]
    if not direct.is_absolute():
        search_targets.append((REPO_ROOT / direct).resolve())
        if direct.parts[:1] != ("exps",):
            search_targets.append((root / direct).resolve())

    seen = set()
    for target in search_targets:
        key = str(target)
        if key in seen:
            continue
        seen.add(key)
        if target.is_file():
            return target.resolve()
        if target.is_dir():
            resolved = _latest_predictor_in_dir(target)
            if resolved is not None:
                return resolved

    if raw.isdigit() and root.is_dir():
        prefix = raw.zfill(5)
        candidates = sorted(
            path for path in root.iterdir() if path.is_dir() and path.name.split("-", 1)[0] == prefix
        )
        for directory in reversed(candidates):
            resolved = _latest_predictor_in_dir(directory)
            if resolved is not None:
                return resolved

    raise FileNotFoundError(f"Could not resolve predictor from '{identifier}'.")


def summarize_scores(scores) -> dict:
    return {
        "count": int(scores.shape[0]),
        "mean": float(scores.mean().item()),
        "std": float(scores.std(unbiased=False).item()) if scores.numel() > 1 else 0.0,
        "min": float(scores.min().item()),
        "max": float(scores.max().item()),
    }
