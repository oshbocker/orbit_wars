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
    reward_mode: str = "sparse"  # "sparse", "dense_absolute", "dense_relative"
    dense_ship_coef: float = 0.002
    dense_prod_coef: float = 0.005


@dataclass(slots=True)
class EvalConfig:
    eval_every: int = 100
    eval_games: int = 10
    eval_opponents: list[str] = field(default_factory=lambda: ["apex", "random"])


@dataclass(slots=True)
class ImitationConfig:
    enabled: bool = False
    bc_expert: str = "apex"
    bc_games: int = 50
    bc_demo_opponent: str = "random"
    bc_epochs: int = 20
    bc_lr: float = 1e-3
    bc_batch_size: int = 256
    coef_start: float = 0.5
    coef_decay_updates: int = 500
    distilled_opponent: bool = True


@dataclass(slots=True)
class TrainConfig:
    seed: int = 42
    run_name: str = "transformer_ppo"
    device: str = "auto"
    save_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    checkpoint_every: int = 50
    log_every: int = 1
    opponent: str = "apex"
    self_play_update_interval: int = 50
    self_play_deterministic: bool = False
    alternate_player_sides: bool = True
    four_player_prob: float = 0.0
    rule_based_prob_start: float = 1.0
    rule_based_prob_end: float = 0.2
    rule_based_decay_updates: int = 2000
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    imitation: ImitationConfig = field(default_factory=ImitationConfig)


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return train_config_from_dict(data)


def train_config_from_dict(data: dict[str, Any]) -> TrainConfig:
    cfg = TrainConfig()
    _update_dataclass(cfg, data, skip={"env", "model", "ppo", "reward", "eval", "imitation"})
    _update_dataclass(cfg.env, data.get("env", {}))
    _update_dataclass(cfg.model, data.get("model", {}))
    _update_dataclass(cfg.ppo, data.get("ppo", {}))

    # Backward compat: map old sparse: true/false to reward_mode
    reward_data = dict(data.get("reward", {}))
    if "sparse" in reward_data and "reward_mode" not in reward_data:
        sparse_val = reward_data.pop("sparse")
        if isinstance(sparse_val, str):
            sparse_val = sparse_val.strip().lower() in {"1", "true", "yes", "on"}
        reward_data["reward_mode"] = "sparse" if sparse_val else "dense_absolute"
    reward_data.pop("sparse", None)  # remove stale key either way
    _update_dataclass(cfg.reward, reward_data)

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
