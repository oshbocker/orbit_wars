"""Run the 5 paper-derived recommendation experiments for v2_bc.

Design
------
All experiments improve the *BC-warm-started* v2_bc agent, so unlike the
earlier no-BC hyperparameter sweep this round:

  1. PRIMES once: collects (and pickle-caches) apex demonstrations and runs
     BC pretraining, saving a shared ``ckpt_000000.pt``.
  2. Each experiment (baseline + R1..R5) RESUMES from that identical BC
     checkpoint and runs ``--updates`` PPO updates, varying exactly one knob.

This guarantees every experiment starts from the *same* warm-started weights
(clean science) and pays the expensive demo-collection / BC-pretrain cost only
once (cheap).

Usage
-----
    uv run python scripts/run_recommendation_experiments.py --updates 200 --max-parallel 3
    uv run python scripts/run_recommendation_experiments.py --bc-games 150 --updates 200
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
CKPT_DIR = EXPERIMENTS_DIR / "checkpoints"
LOG_DIR = EXPERIMENTS_DIR / "logs"
CFG_DIR = EXPERIMENTS_DIR / "configs"
DEMO_CACHE = EXPERIMENTS_DIR / "demos_apex.pkl"
PRIME_NAME = "exp2_prime"
PRIME_CKPT = CKPT_DIR / PRIME_NAME / "ckpt_000000.pt"


@dataclass
class Experiment:
    name: str
    description: str
    overrides: dict


# Five recommendations + baseline. Each varies exactly one knob vs baseline.
EXPERIMENTS = [
    Experiment("baseline", "v2_bc settings (BC warm start, dense_relative, apex)", {}),
    Experiment("r1_ent_anneal",
               "R1: anneal entropy 0.01 -> 0.0 over the run (PPG/MAPPO)",
               {"ppo": {"ent_coef": 0.01, "ent_coef_end": 0.0}}),
    Experiment("r2_high_gamma",
               "R2: discount gamma 0.99 -> 0.997 (long-horizon credit; DreamerV3)",
               {"ppo": {"gamma": 0.997}}),
    Experiment("r3_strong_value",
               "R3: value-loss weight 0.5 -> 1.0 (PPG-lite / MAPPO value emphasis)",
               {"ppo": {"vf_coef": 1.0}}),
    Experiment("r4_fast_selfplay",
               "R4: rule_based decay 1000 -> 150 (self-play sooner; XLand/DeepNash)",
               {"rule_based_decay_updates": 150}),
    Experiment("r5_no_prod_bonus",
               "R5: early_prod_bonus 9.0 -> 0.0 (reward-shaping bias; Go-Explore)",
               {"reward": {"early_prod_bonus": 0.0}}),
]


def base_config(updates: int, bc_games: int, bc_epochs: int) -> dict:
    """v2_bc-style base config (BC enabled, dense_relative, apex)."""
    return {
        "run_name": "PLACEHOLDER",
        "seed": 42,
        "opponent": "apex",
        "alternate_player_sides": True,
        "self_play_update_interval": 50,
        "four_player_prob": 0.0,
        "rule_based_prob_start": 0.5,
        "rule_based_prob_end": 0.1,
        "rule_based_decay_updates": 1000,
        "save_dir": str(CKPT_DIR),
        "log_dir": str(LOG_DIR),
        "checkpoint_every": 100,
        "log_every": 1,
        "env": {"max_planets": 40, "allocation_threshold": 0.05, "min_ships_to_send": 1},
        "model": {"embed_dim": 128, "n_heads": 4, "n_layers": 3, "ff_dim": 256,
                  "planet_feat_dim": 22, "global_feat_dim": 8},
        "ppo": {
            "rollout_steps": 32, "num_envs": 1, "total_updates": updates,
            "epochs": 4, "minibatch_size": 256, "gamma": 0.99, "gae_lambda": 0.95,
            "clip_coef": 0.2, "ent_coef": 0.01, "ent_coef_end": -1.0,
            "vf_coef": 0.5, "lr": 0.0001, "max_grad_norm": 0.5, "num_workers": 0,
        },
        "reward": {"reward_mode": "dense_relative", "dense_ship_coef": 0.002,
                   "dense_prod_coef": 0.005, "early_prod_bonus": 9.0,
                   "early_prod_bonus_steps": 50},
        "eval": {"eval_every": 100, "eval_games": 10, "eval_opponents": ["apex", "random"]},
        "imitation": {
            "enabled": True, "bc_expert": "apex", "bc_games": bc_games,
            "bc_demo_opponent": "random", "bc_epochs": bc_epochs, "bc_lr": 0.001,
            "bc_batch_size": 256, "coef_start": 0.5, "coef_decay_updates": 1000,
            "distilled_opponent": True, "bc_skip_steps": 0,
            "bc_cache_path": str(DEMO_CACHE),
        },
    }


def _apply(base: dict, overrides: dict) -> dict:
    for section, values in overrides.items():
        if section in base and isinstance(base[section], dict):
            base[section] = {**base[section], **values}
        else:
            base[section] = values
    return base


def write_config(name: str, cfg: dict) -> Path:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    path = CFG_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, sort_keys=True)
    return path


def run_prime(updates: int, bc_games: int, bc_epochs: int) -> None:
    """Collect+cache demos and BC-pretrain once, producing the shared ckpt."""
    import copy
    cfg = base_config(updates=0, bc_games=bc_games, bc_epochs=bc_epochs)
    cfg = copy.deepcopy(cfg)
    cfg["run_name"] = PRIME_NAME
    path = write_config(PRIME_NAME, cfg)
    (LOG_DIR / PRIME_NAME).mkdir(parents=True, exist_ok=True)
    print(f"=== PRIME: collecting demos + BC pretrain (bc_games={bc_games}, "
          f"bc_epochs={bc_epochs}) ===")
    log_file = LOG_DIR / PRIME_NAME / "stdout.log"
    with open(log_file, "w") as lf:
        t0 = time.time()
        rc = subprocess.call(
            [sys.executable, "-m", "v2.train", "--config", str(path)],
            cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT,
        )
    print(f"  prime finished rc={rc} in {time.time()-t0:.0f}s")
    if rc != 0 or not PRIME_CKPT.exists():
        raise RuntimeError(f"Prime failed (rc={rc}); ckpt exists={PRIME_CKPT.exists()}")
    print(f"  shared BC checkpoint: {PRIME_CKPT}")


def launch(exp: Experiment, updates: int, bc_games: int, bc_epochs: int) -> subprocess.Popen:
    import copy
    cfg = _apply(copy.deepcopy(base_config(updates, bc_games, bc_epochs)), exp.overrides)
    run_name = f"exp2_{exp.name}"
    cfg["run_name"] = run_name
    path = write_config(run_name, cfg)
    (LOG_DIR / run_name).mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / run_name / "stdout.log"
    lf = open(log_file, "w")
    return subprocess.Popen(
        [sys.executable, "-m", "v2.train", "--config", str(path),
         "--resume", str(PRIME_CKPT)],
        cwd=str(REPO_ROOT), stdout=lf, stderr=subprocess.STDOUT,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--bc-games", type=int, default=150)
    ap.add_argument("--bc-epochs", type=int, default=50)
    ap.add_argument("--max-parallel", type=int, default=3)
    ap.add_argument("--experiments", nargs="*", default=None)
    ap.add_argument("--skip-prime", action="store_true")
    args = ap.parse_args()

    exps = EXPERIMENTS
    if args.experiments:
        exps = [e for e in EXPERIMENTS if e.name in args.experiments]

    if not args.skip_prime:
        run_prime(args.updates, args.bc_games, args.bc_epochs)
    elif not PRIME_CKPT.exists():
        raise RuntimeError(f"--skip-prime but no prime ckpt at {PRIME_CKPT}")

    print(f"\nRunning {len(exps)} experiments, {args.updates} updates, "
          f"max {args.max_parallel} parallel")
    remaining = list(exps)
    while remaining:
        batch = remaining[:args.max_parallel]
        remaining = remaining[args.max_parallel:]
        print(f"\n=== Batch: {[e.name for e in batch]} ===")
        procs = []
        for exp in batch:
            print(f"  Starting: {exp.name} — {exp.description}")
            procs.append((exp, launch(exp, args.updates, args.bc_games, args.bc_epochs)))
        for exp, proc in procs:
            proc.wait()
            status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            print(f"  Finished: {exp.name} — {status}")

    print("\nAll experiments complete.")
    print("Plot with: uv run python scripts/plot_recommendation_experiments.py")


if __name__ == "__main__":
    main()
