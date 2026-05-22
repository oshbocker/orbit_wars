from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class V2EnvConfig:
    max_planets: int = 40
    board_size: float = 100.0
    episode_steps: int = 500
    allocation_threshold: float = 0.05
    min_ships_to_send: int = 1


@dataclass(slots=True)
class V2ModelConfig:
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 256
    planet_feat_dim: int = 22
    global_feat_dim: int = 8


@dataclass(slots=True)
class V2PPOConfig:
    rollout_steps: int = 64
    num_envs: int = 2
    total_updates: int = 3000
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
class V2RewardConfig:
    reward_mode: str = "dense_relative"
    dense_ship_coef: float = 0.002
    dense_prod_coef: float = 0.005
    early_prod_bonus: float = 9.0
    early_prod_bonus_steps: int = 50


@dataclass(slots=True)
class V2EvalConfig:
    eval_every: int = 100
    eval_games: int = 10
    eval_opponents: list[str] = field(default_factory=lambda: ["apex", "random"])


@dataclass(slots=True)
class V2ImitationConfig:
    enabled: bool = False
    bc_expert: str = "apex"
    bc_games: int = 200
    bc_demo_opponent: str = "random"
    bc_epochs: int = 50
    bc_lr: float = 1e-3
    bc_batch_size: int = 256


@dataclass(slots=True)
class V2Config:
    seed: int = 42
    run_name: str = "v2_default"
    device: str = "auto"
    save_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    checkpoint_every: int = 50
    log_every: int = 1
    opponent: str = "apex"
    alternate_player_sides: bool = True
    self_play_update_interval: int = 50
    self_play_deterministic: bool = False
    four_player_prob: float = 0.0
    rule_based_prob_start: float = 1.0
    rule_based_prob_end: float = 0.2
    rule_based_decay_updates: int = 2000
    env: V2EnvConfig = field(default_factory=V2EnvConfig)
    model: V2ModelConfig = field(default_factory=V2ModelConfig)
    ppo: V2PPOConfig = field(default_factory=V2PPOConfig)
    reward: V2RewardConfig = field(default_factory=V2RewardConfig)
    eval: V2EvalConfig = field(default_factory=V2EvalConfig)
    imitation: V2ImitationConfig = field(default_factory=V2ImitationConfig)


def load_v2_config(path: str | Path) -> V2Config:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return v2_config_from_dict(data)


def v2_config_from_dict(data: dict[str, Any]) -> V2Config:
    cfg = V2Config()
    sub = {"env", "model", "ppo", "reward", "eval", "imitation"}
    _update_dataclass(cfg, data, skip=sub)
    _update_dataclass(cfg.env, data.get("env", {}))
    _update_dataclass(cfg.model, data.get("model", {}))
    _update_dataclass(cfg.ppo, data.get("ppo", {}))
    _update_dataclass(cfg.reward, data.get("reward", {}))
    _update_dataclass(cfg.eval, data.get("eval", {}))
    _update_dataclass(cfg.imitation, data.get("imitation", {}))
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
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        return value if isinstance(value, list) else default
    return value
