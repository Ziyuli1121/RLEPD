from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigError(RuntimeError):
    """Raised when configuration values are invalid."""


def _parse_override_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        if lowered.startswith("0x"):
            return int(value, 16)
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _apply_override(raw: Dict[str, Any], key: str, value: str) -> None:
    parts = key.split(".")
    node = raw
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = _parse_override_value(value)


def load_raw_config(path: Path, overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML structure must be a mapping.")
    merged = copy.deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(f"Override must be in key=value form: {item}")
        key, value = item.split("=", 1)
        if not key:
            raise ConfigError(f"Override key is empty: {item}")
        _apply_override(merged, key, value)
    return merged


@dataclass
class RunConfig:
    output_root: Path
    run_name: str
    seed: int
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S"))
    run_id: Optional[str] = None
    run_dir: Optional[Path] = None

    def assign_run_dir(self) -> None:
        if self.run_id is None:
            self.run_id = f"{self.timestamp}-{self.run_name}"
        self.run_dir = (self.output_root / self.run_id).resolve()


@dataclass
class DataConfig:
    predictor_snapshot: Path
    prompt_csv: Optional[Path] = None


@dataclass
class ModelConfig:
    dataset_name: str
    guidance_type: str
    guidance_rate: float
    schedule_type: str
    schedule_rho: float
    num_steps: Optional[int] = None
    num_points: Optional[int] = None


@dataclass
class RewardConfig:
    weights_path: Path
    batch_size: int
    enable_amp: bool
    cache_dir: Optional[Path] = None


@dataclass
class PPOConfig:
    rollout_batch_size: int
    rloo_k: int
    ppo_epochs: int
    minibatch_size: int
    learning_rate: float
    clip_range: float
    kl_coef: float
    entropy_coef: float
    max_grad_norm: Optional[float]
    decode_rgb: bool
    steps: int
    dirichlet_concentration: float


@dataclass
class LoggingConfig:
    log_interval: int
    save_interval: int


@dataclass
class FullConfig:
    run: RunConfig
    data: DataConfig
    model: ModelConfig
    reward: RewardConfig
    ppo: PPOConfig
    logging: LoggingConfig
    raw: Dict[str, Any] = field(repr=False)

    def to_dict(self) -> Dict[str, Any]:
        run_dict = asdict(self.run)
        run_dict["output_root"] = str(run_dict["output_root"])
        if run_dict.get("run_dir") is not None:
            run_dict["run_dir"] = str(run_dict["run_dir"])
        return {
            "run": run_dict,
            "data": {
                "predictor_snapshot": str(self.data.predictor_snapshot),
                "prompt_csv": str(self.data.prompt_csv) if self.data.prompt_csv else None,
            },
            "model": asdict(self.model),
            "reward": {
                "weights_path": str(self.reward.weights_path),
                "batch_size": self.reward.batch_size,
                "enable_amp": self.reward.enable_amp,
                "cache_dir": str(self.reward.cache_dir) if self.reward.cache_dir else None,
            },
            "ppo": asdict(self.ppo),
            "logging": asdict(self.logging),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def build_config(raw: Dict[str, Any]) -> FullConfig:
    def require(section: str) -> Dict[str, Any]:
        if section not in raw or not isinstance(raw[section], dict):
            raise ConfigError(f"Missing configuration section: {section}")
        return raw[section]

    run_raw = require("run")
    data_raw = require("data")
    model_raw = require("model")
    reward_raw = require("reward")
    ppo_raw = require("ppo")
    logging_raw = require("logging")

    run = RunConfig(
        output_root=Path(run_raw.get("output_root", "exps")).expanduser(),
        run_name=str(run_raw.get("run_name", "rl_run")),
        seed=int(run_raw.get("seed", 0)),
    )
    data = DataConfig(
        predictor_snapshot=Path(data_raw["predictor_snapshot"]).expanduser(),
        prompt_csv=Path(data_raw["prompt_csv"]).expanduser() if data_raw.get("prompt_csv") else None,
    )
    model = ModelConfig(
        dataset_name=str(model_raw.get("dataset_name", "ms_coco")),
        guidance_type=str(model_raw.get("guidance_type", "cfg")),
        guidance_rate=float(model_raw.get("guidance_rate", 7.5)),
        schedule_type=str(model_raw.get("schedule_type", "discrete")),
        schedule_rho=float(model_raw.get("schedule_rho", 1.0)),
        num_steps=model_raw.get("num_steps"),
        num_points=model_raw.get("num_points"),
    )
    reward = RewardConfig(
        weights_path=Path(reward_raw["weights_path"]).expanduser(),
        batch_size=int(reward_raw.get("batch_size", 2)),
        enable_amp=bool(reward_raw.get("enable_amp", True)),
        cache_dir=Path(reward_raw["cache_dir"]).expanduser() if reward_raw.get("cache_dir") else None,
    )
    ppo = PPOConfig(
        rollout_batch_size=int(ppo_raw.get("rollout_batch_size", 4)),
        rloo_k=int(ppo_raw.get("rloo_k", 2)),
        ppo_epochs=int(ppo_raw.get("ppo_epochs", 1)),
        minibatch_size=int(ppo_raw.get("minibatch_size", 2)),
        learning_rate=float(ppo_raw.get("learning_rate", 5e-5)),
        clip_range=float(ppo_raw.get("clip_range", 0.2)),
        kl_coef=float(ppo_raw.get("kl_coef", 0.01)),
        entropy_coef=float(ppo_raw.get("entropy_coef", 0.0)),
        max_grad_norm=(
            float(ppo_raw["max_grad_norm"]) if ppo_raw.get("max_grad_norm") is not None else None
        ),
        decode_rgb=bool(ppo_raw.get("decode_rgb", True)),
        steps=int(ppo_raw.get("steps", 10)),
        dirichlet_concentration=float(ppo_raw.get("dirichlet_concentration", 200.0)),
    )
    logging = LoggingConfig(
        log_interval=int(logging_raw.get("log_interval", 1)),
        save_interval=int(logging_raw.get("save_interval", 5)),
    )
    run.assign_run_dir()
    return FullConfig(run=run, data=data, model=model, reward=reward, ppo=ppo, logging=logging, raw=raw)


def validate_config(config: FullConfig, check_paths: bool = True) -> None:
    if config.ppo.rollout_batch_size <= 0:
        raise ConfigError("rollout_batch_size must be positive.")
    if config.ppo.rollout_batch_size % config.ppo.rloo_k != 0:
        raise ConfigError("rollout_batch_size must be a multiple of rloo_k.")
    if config.ppo.minibatch_size <= 0 or config.ppo.minibatch_size > config.ppo.rollout_batch_size:
        raise ConfigError("minibatch_size must be in (0, rollout_batch_size].")
    if config.reward.batch_size <= 0 or config.reward.batch_size > config.ppo.rollout_batch_size:
        raise ConfigError("reward.batch_size must be in (0, rollout_batch_size].")
    if config.ppo.steps <= 0:
        raise ConfigError("ppo.steps must be positive.")
    if check_paths:
        if not config.data.predictor_snapshot.is_file():
            raise ConfigError(f"Predictor snapshot not found: {config.data.predictor_snapshot}")
        if not config.reward.weights_path.is_file():
            raise ConfigError(f"HPS weights not found: {config.reward.weights_path}")
        if config.data.prompt_csv and not config.data.prompt_csv.is_file():
            raise ConfigError(f"Prompt CSV not found: {config.data.prompt_csv}")


def pretty_format_config(config: FullConfig) -> str:
    return yaml.safe_dump(config.to_dict(), sort_keys=False, allow_unicode=True)
