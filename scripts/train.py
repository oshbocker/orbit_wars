#!/usr/bin/env python3
"""
Train an RL agent for Orbit Wars.

Examples
--------
# Default run (500k steps, PPO vs baseline)
python scripts/train.py

# Override config file
python scripts/train.py --config configs/ppo_selfplay.yaml

# Override individual values with dot notation
python scripts/train.py --set training.total_timesteps=1000000 env.n_envs=8

# Resume from a previous checkpoint
python scripts/train.py --resume outputs/checkpoints/ppo_default_20260501_120000/best_model.zip
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path regardless of where this is invoked from
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from training.train import load_config, apply_dotted_overrides, train

DEFAULT_CONFIG = _root / "configs" / "ppo_default.yaml"


def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for Orbit Wars.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default=str(DEFAULT_CONFIG),
        help="Path to YAML config file (default: configs/ppo_default.yaml)",
    )
    parser.add_argument(
        "--set", "-s",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dot-notation overrides, e.g. training.total_timesteps=1000000",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a .zip checkpoint to resume training from",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.set:
        config = apply_dotted_overrides(config, args.set)

    print(f"Config: {args.config}")
    if args.set:
        print(f"Overrides: {args.set}")
    print()

    train(config, resume_from=args.resume)


if __name__ == "__main__":
    main()
