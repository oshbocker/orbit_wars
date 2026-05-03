"""
Core PPO training logic for Orbit Wars.

Separates "what to train" from "how to invoke it" (see scripts/train.py).
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


# ── config helpers ─────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return it as a nested dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def merge_config(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base (override wins on conflicts)."""
    result = base.copy()
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result


def apply_dotted_overrides(config: dict, overrides: list[str]) -> dict:
    """
    Apply dot-notation CLI overrides like "training.learning_rate=1e-4".
    """
    config = config.copy()
    for item in overrides:
        key_path, _, value_str = item.partition("=")
        keys = key_path.strip().split(".")
        # Parse value
        try:
            value: Any = yaml.safe_load(value_str)
        except Exception:
            value = value_str

        node = config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return config


# ── opponent resolution ─────────────────────────────────────────────────────

def resolve_opponent(opponent_str: str, checkpoint_dir: str | Path | None = None):
    """
    Convert the string from config['env']['opponent'] to a callable.

    'random'   → built-in random agent
    'baseline' → deterministic rule-based agent
    'self'     → latest checkpoint in checkpoint_dir (falls back to baseline)
    """
    if opponent_str == "random":
        from envs.orbit_wars_env import _random_opponent
        return _random_opponent

    if opponent_str == "baseline":
        from agents.baseline import agent as baseline_agent
        return baseline_agent

    if opponent_str == "self":
        if checkpoint_dir is not None:
            latest = _find_latest_checkpoint(Path(checkpoint_dir))
            if latest is not None:
                print(f"Self-play: loading opponent from {latest}")
                from agents.rl_agent import RLAgent
                return RLAgent(latest, device="cpu")
        # Fall back to baseline if no checkpoint yet
        print("Self-play: no checkpoint found, falling back to baseline opponent.")
        from agents.baseline import agent as baseline_agent
        return baseline_agent

    raise ValueError(f"Unknown opponent: {opponent_str!r}. Use 'random', 'baseline', or 'self'.")


def _find_latest_checkpoint(checkpoint_dir: Path):
    zips = sorted(checkpoint_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
    return zips[-1] if zips else None


# ── training ────────────────────────────────────────────────────────────────

def train(config: dict, resume_from: str | Path | None = None) -> Any:
    """
    Train a PPO agent with the given config dict.

    Parameters
    ----------
    config : nested dict matching configs/*.yaml structure
    resume_from : path to a .zip checkpoint to continue training from

    Returns
    -------
    Trained SB3 PPO model.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import (
        EvalCallback, CheckpointCallback, CallbackList
    )
    from envs.orbit_wars_env import OrbitWarsEnv

    env_cfg = config.get("env", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("eval", {})
    out_cfg = config.get("output", {})

    # ── output paths ───────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{out_cfg.get('run_name', 'ppo')}_{timestamp}"
    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", "outputs/checkpoints")) / run_name
    log_dir = Path(out_cfg.get("log_dir", "outputs/logs")) / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run:         {run_name}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs:        {log_dir}")

    # ── opponent ───────────────────────────────────────────────────────────
    opponent = resolve_opponent(
        env_cfg.get("opponent", "random"),
        checkpoint_dir=checkpoint_dir,
    )

    # ── environments ───────────────────────────────────────────────────────
    n_envs = env_cfg.get("n_envs", 4)
    reward_shaping = env_cfg.get("reward_shaping", True)

    def _make_env():
        e = OrbitWarsEnv(opponent=opponent, reward_shaping=reward_shaping)
        return Monitor(e, str(log_dir))

    vec_env = make_vec_env(_make_env, n_envs=n_envs)

    # ── model ──────────────────────────────────────────────────────────────
    device = train_cfg.get("device", "auto")
    net_arch = train_cfg.get("net_arch", [256, 256])
    policy_kwargs = dict(net_arch=net_arch)

    model_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=train_cfg.get("learning_rate", 3e-4),
        n_steps=train_cfg.get("n_steps", 512),
        batch_size=train_cfg.get("batch_size", 256),
        n_epochs=train_cfg.get("n_epochs", 10),
        gamma=train_cfg.get("gamma", 0.995),
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
        clip_range=train_cfg.get("clip_range", 0.2),
        ent_coef=train_cfg.get("ent_coef", 0.01),
        vf_coef=train_cfg.get("vf_coef", 0.5),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=str(log_dir),
    )

    if resume_from is not None:
        print(f"Resuming from: {resume_from}")
        model = PPO.load(str(resume_from), env=vec_env, **{
            k: v for k, v in model_kwargs.items()
            if k not in ("policy", "env")
        })
    else:
        model = PPO(**model_kwargs)

    # ── callbacks ──────────────────────────────────────────────────────────
    eval_opponent_str = eval_cfg.get("opponent", "baseline")
    eval_opponent = resolve_opponent(eval_opponent_str)
    eval_env = Monitor(OrbitWarsEnv(opponent=eval_opponent, reward_shaping=False))

    eval_freq = max(eval_cfg.get("freq", 10_000) // n_envs, 1)
    callbacks = CallbackList([
        EvalCallback(
            eval_env,
            best_model_save_path=str(checkpoint_dir),
            log_path=str(log_dir),
            eval_freq=eval_freq,
            n_eval_episodes=eval_cfg.get("n_episodes", 20),
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=max(50_000 // n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="ckpt",
        ),
    ])

    # ── train ──────────────────────────────────────────────────────────────
    total_timesteps = train_cfg.get("total_timesteps", 500_000)
    print(f"\nTraining for {total_timesteps:,} timesteps on {n_envs} envs...\n")
    t0 = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    elapsed = time.time() - t0
    final_path = checkpoint_dir / "final_model.zip"
    model.save(str(final_path))
    print(f"\nDone in {elapsed/60:.1f} min.  Final model: {final_path}")

    return model, checkpoint_dir
