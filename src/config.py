from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class EnvConfig:
    board_size: float = 100.0
    episode_steps: int = 500
    max_targets: int = 30
    k_neighbors: int = 3
    ship_fractions: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    max_planets: int = 48
    max_ships: float = 400.0
    max_production: float = 5.0


@dataclass(slots=True)
class ModelConfig:
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    pos_hidden: int = 64


@dataclass(slots=True)
class PPOConfig:
    rollout_steps: int = 64
    num_envs: int = 2
    total_updates: int = 2000
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5


@dataclass(slots=True)
class RewardConfig:
    sparse: bool = True
    dense_ship_coef: float = 0.001
    dense_prod_coef: float = 0.005


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    run_name: str = "transformer_ppo"
    device: str = "auto"
    save_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    checkpoint_every: int = 50
    log_every: int = 1
    opponent: str = "competitive"
    self_play_update_interval: int = 50
    self_play_deterministic: bool = False
    alternate_player_sides: bool = True
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return train_config_from_dict(data)


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    cfg = TrainConfig()
    _update_dataclass(cfg, data, skip={"env", "model", "ppo", "reward"})
    _update_dataclass(cfg.env, data.get("env", {}))
    _update_dataclass(cfg.model, data.get("model", {}))
    _update_dataclass(cfg.ppo, data.get("ppo", {}))
    _update_dataclass(cfg.reward, data.get("reward", {}))
    return cfg


def _update_dataclass(instance: Any, values: dict[str, Any], skip: set[str] | None = None) -> None:
    if not isinstance(values, dict):
        return
    skip = skip or set()
    for key, value in values.items():
        if key in skip or not hasattr(instance, key):
            continue
        default = getattr(instance, key)
        setattr(instance, key, _coerce_value(value, default))


def _coerce_value(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        if isinstance(value, list):
            return value
        return default
    return value
