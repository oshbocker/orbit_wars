"""Run hyperparameter experiments for V2 OrbitNet.

Each experiment runs 200 PPO updates with different settings.
Results are logged to experiments/<name>/train.log and parsed for comparison.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


@dataclass
class Experiment:
    name: str
    description: str
    overrides: dict  # nested dict of config overrides


# ── Define experiments ───────────────────────────────────────────────────────

EXPERIMENTS = [
    Experiment(
        name="baseline",
        description="Default V2 config (embed=128, layers=3, lr=3e-4, ent=0.03)",
        overrides={},
    ),
    Experiment(
        name="large_net",
        description="Larger network (embed=256, layers=4, ff=512, ~2M params)",
        overrides={
            "model": {"embed_dim": 256, "n_layers": 4, "ff_dim": 512},
        },
    ),
    Experiment(
        name="small_net",
        description="Smaller network (embed=64, layers=2, ff=128, ~135K params)",
        overrides={
            "model": {"embed_dim": 64, "n_layers": 2, "ff_dim": 128},
        },
    ),
    Experiment(
        name="high_lr",
        description="Higher learning rate (lr=1e-3)",
        overrides={
            "ppo": {"lr": 0.001},
        },
    ),
    Experiment(
        name="low_lr",
        description="Lower learning rate (lr=5e-5)",
        overrides={
            "ppo": {"lr": 5e-5},
        },
    ),
    Experiment(
        name="high_entropy",
        description="Higher entropy coefficient (ent=0.1) for more exploration",
        overrides={
            "ppo": {"ent_coef": 0.1},
        },
    ),
    Experiment(
        name="low_entropy",
        description="Lower entropy coefficient (ent=0.005) for less exploration",
        overrides={
            "ppo": {"ent_coef": 0.005},
        },
    ),
    Experiment(
        name="more_epochs",
        description="More PPO epochs per update (epochs=8 instead of 4)",
        overrides={
            "ppo": {"epochs": 8},
        },
    ),
    Experiment(
        name="long_rollout",
        description="Longer rollouts (128 steps instead of 32)",
        overrides={
            "ppo": {"rollout_steps": 128},
        },
    ),
    Experiment(
        name="large_batch",
        description="Larger effective batch (num_envs=2, rollout=64, minibatch=512)",
        overrides={
            "ppo": {"num_envs": 2, "rollout_steps": 64, "minibatch_size": 512},
        },
    ),
]


def build_config(exp: Experiment, updates: int = 200) -> dict:
    """Build a full config dict for an experiment."""
    base = {
        "run_name": f"hparam_{exp.name}",
        "seed": 42,
        "opponent": "apex",
        "alternate_player_sides": True,
        "self_play_update_interval": 50,
        "four_player_prob": 0.0,
        "rule_based_prob_start": 0.5,
        "rule_based_prob_end": 0.1,
        "rule_based_decay_updates": 1000,
        "save_dir": str(EXPERIMENTS_DIR / "checkpoints"),
        "log_dir": str(EXPERIMENTS_DIR / "logs"),
        "checkpoint_every": 200,  # only save at end
        "log_every": 1,
        "env": {
            "max_planets": 40,
            "allocation_threshold": 0.05,
            "min_ships_to_send": 1,
        },
        "model": {
            "embed_dim": 128,
            "n_heads": 4,
            "n_layers": 3,
            "ff_dim": 256,
            "planet_feat_dim": 22,
            "global_feat_dim": 8,
        },
        "ppo": {
            "rollout_steps": 32,
            "num_envs": 1,
            "total_updates": updates,
            "epochs": 4,
            "minibatch_size": 256,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_coef": 0.2,
            "ent_coef": 0.03,
            "vf_coef": 0.5,
            "lr": 0.0003,
            "max_grad_norm": 0.5,
            "num_workers": 0,  # sequential for fair comparison
        },
        "reward": {
            "reward_mode": "dense_relative",
            "dense_ship_coef": 0.002,
            "dense_prod_coef": 0.005,
            "early_prod_bonus": 9.0,
            "early_prod_bonus_steps": 50,
        },
        "eval": {
            "eval_every": 100,
            "eval_games": 10,
            "eval_opponents": ["apex", "random"],
        },
        "imitation": {
            "enabled": False,
        },
    }

    # Apply overrides
    for section, values in exp.overrides.items():
        if section in base and isinstance(base[section], dict):
            base[section].update(values)
        else:
            base[section] = values

    return base


def run_experiment(exp: Experiment, updates: int, dry_run: bool = False) -> subprocess.Popen | None:
    """Launch a single experiment as a subprocess."""
    cfg = build_config(exp, updates)

    # Write config
    cfg_dir = EXPERIMENTS_DIR / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{exp.name}.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    # Ensure log dir exists
    log_dir = Path(cfg["log_dir"]) / cfg["run_name"]
    log_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"  [dry-run] Would run: python -m v2.train --config {cfg_path}")
        return None

    # Launch
    log_file = log_dir / "stdout.log"
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "v2.train", "--config", str(cfg_path)],
            cwd=str(REPO_ROOT),
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    return proc


def parse_train_log(log_path: Path) -> list[dict]:
    """Parse a train.log file into a list of update metrics."""
    results = []
    pattern = re.compile(
        r"update=\s*(\d+)\s+"
        r"reward=([\+\-\d.]+)\s+"
        r"eps=(\d+)\s+"
        r"samples=(\d+)\s+"
        r"loss=([\+\-\d.]+)\s+"
        r"ploss=([\+\-\d.]+)\s+"
        r"vloss=([\+\-\d.]+)\s+"
        r"ent=([\+\-\d.]+)\s+"
        r"dt=([\d.]+)s"
    )
    if not log_path.exists():
        return results

    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            results.append({
                "update": int(m.group(1)),
                "reward": float(m.group(2)),
                "episodes": int(m.group(3)),
                "samples": int(m.group(4)),
                "loss": float(m.group(5)),
                "policy_loss": float(m.group(6)),
                "value_loss": float(m.group(7)),
                "entropy": float(m.group(8)),
                "dt": float(m.group(9)),
            })
    return results


def parse_eval_results(log_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Parse eval results from train.log."""
    results: dict[str, list[tuple[int, float]]] = {}
    current_update = 0
    update_pattern = re.compile(r"update=\s*(\d+)")
    eval_pattern = re.compile(r"vs (\w+): W=([\d.]+)%")

    for line in log_path.read_text().splitlines():
        um = update_pattern.search(line)
        if um:
            current_update = int(um.group(1))
        em = eval_pattern.search(line)
        if em:
            opp = em.group(1)
            wr = float(em.group(2)) / 100.0
            results.setdefault(opp, []).append((current_update, wr))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="Run only these experiments (by name)")
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.experiments:
        exps = [e for e in EXPERIMENTS if e.name in args.experiments]

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(exps)} experiments, {args.updates} updates each, "
          f"max {args.max_parallel} parallel")
    print()

    # Run in batches
    remaining = list(exps)
    all_results: dict[str, list[dict]] = {}

    while remaining:
        batch = remaining[:args.max_parallel]
        remaining = remaining[args.max_parallel:]

        print(f"=== Batch: {[e.name for e in batch]} ===")
        procs: list[tuple[Experiment, subprocess.Popen | None]] = []
        for exp in batch:
            print(f"  Starting: {exp.name} — {exp.description}")
            proc = run_experiment(exp, args.updates, args.dry_run)
            procs.append((exp, proc))

        if args.dry_run:
            continue

        # Wait for batch to complete
        for exp, proc in procs:
            if proc is not None:
                proc.wait()
                status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
                print(f"  Finished: {exp.name} — {status}")

                # Parse results
                log_path = EXPERIMENTS_DIR / "logs" / f"hparam_{exp.name}" / "train.log"
                results = parse_train_log(log_path)
                all_results[exp.name] = results

    if args.dry_run:
        return

    # Save combined results
    results_path = EXPERIMENTS_DIR / "results.json"
    serializable = {}
    for name, updates in all_results.items():
        serializable[name] = updates
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n=== Summary ===")
    print(f"{'Experiment':<16} {'Updates':>8} {'Avg Reward':>12} {'Final Loss':>12} "
          f"{'Avg Entropy':>12} {'Avg dt':>8}")
    print("-" * 80)
    for exp in exps:
        data = all_results.get(exp.name, [])
        if not data:
            print(f"{exp.name:<16} {'NO DATA':>8}")
            continue
        # Use last 50 updates for averages
        recent = data[-50:] if len(data) >= 50 else data
        avg_reward = sum(d["reward"] for d in recent) / len(recent)
        final_loss = recent[-1]["loss"]
        avg_entropy = sum(d["entropy"] for d in recent) / len(recent)
        avg_dt = sum(d["dt"] for d in recent) / len(recent)
        print(f"{exp.name:<16} {len(data):>8} {avg_reward:>+12.3f} {final_loss:>12.4f} "
              f"{avg_entropy:>12.3f} {avg_dt:>7.1f}s")


if __name__ == "__main__":
    main()
