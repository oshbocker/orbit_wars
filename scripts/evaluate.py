#!/usr/bin/env python3
"""
Evaluate Orbit Wars agents head-to-head.

Examples
--------
# Evaluate a trained model vs aggressive and random (auto-included)
python scripts/evaluate.py --model outputs/checkpoints/ppo_default_20260501/best_model.zip

# Compare two models head-to-head
python scripts/evaluate.py \\
    --model outputs/checkpoints/run_a/best_model.zip \\
    --vs     outputs/checkpoints/run_b/best_model.zip \\
    --games 50

# Evaluate the aggressive agent vs random
python scripts/evaluate.py --aggressive --games 30

# Full matrix: two RL models + aggressive + random
python scripts/evaluate.py \\
    --model outputs/checkpoints/run_a/best_model.zip:rl_v1 \\
    --model outputs/checkpoints/run_b/best_model.zip:rl_v2 \\
    --aggressive --random \\
    --games 20
"""

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from evaluation.evaluate import run_games, print_results, head_to_head


def _load_agent(spec: str):
    """Parse 'path.zip' or 'path.zip:label' and return (label, callable)."""
    if ":" in spec:
        path, label = spec.rsplit(":", 1)
    else:
        path = spec
        label = Path(spec).parent.name  # use run directory name as label

    from agents.rl_agent import RLAgent
    return label, RLAgent(path, device="cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Orbit Wars agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", "-m",
        nargs="*",
        default=[],
        metavar="PATH[:LABEL]",
        help="Trained model(s) to evaluate (path to .zip, optional :label suffix)",
    )
    parser.add_argument(
        "--vs",
        default=None,
        metavar="PATH[:LABEL]",
        help="If set, evaluate --model only against this opponent (not a full matrix)",
    )
    parser.add_argument(
        "--aggressive",
        action="store_true",
        help="Include the aggressive (production-rush) agent",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Include the random agent",
    )
    parser.add_argument(
        "--strategic",
        action="store_true",
        help="Include the strategic (tree-search) agent",
    )
    parser.add_argument(
        "--games", "-n",
        type=int,
        default=20,
        help="Number of games per matchup (default: 20)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-game results",
    )
    args = parser.parse_args()

    # Build agent dict
    agents: dict = {}

    if args.random:
        from envs.orbit_wars_env import _random_opponent
        agents["random"] = _random_opponent

    if args.aggressive:
        from agents.aggressive import agent as aggressive_agent
        agents["aggressive"] = aggressive_agent

    if args.strategic:
        from agents.strategic import agent as strategic_agent
        agents["strategic"] = strategic_agent

    for spec in (args.model or []):
        label, ag = _load_agent(spec)
        agents[label] = ag

    if not agents:
        # Default: aggressive vs random
        from agents.aggressive import agent as aggressive_agent
        from envs.orbit_wars_env import _random_opponent
        agents = {"aggressive": aggressive_agent, "random": _random_opponent}
        print("No agents specified — running default: aggressive vs random\n")
    elif args.model and not args.random and not args.aggressive and not args.vs:
        # RL model(s) given without explicit opponents — auto-add aggressive + random
        from agents.aggressive import agent as aggressive_agent
        from envs.orbit_wars_env import _random_opponent
        agents["aggressive"] = aggressive_agent
        agents["random"] = _random_opponent
        print("No opponents specified — auto-adding aggressive and random\n")

    if args.vs:
        # Single opponent mode: evaluate every agent in --model against --vs
        vs_label, vs_agent = _load_agent(args.vs)
        for label, ag in agents.items():
            r = run_games(ag, vs_agent, n_games=args.games, verbose=args.verbose)
            print_results(label, vs_label, r)
    else:
        # Full head-to-head matrix
        head_to_head(agents, n_games=args.games, verbose=args.verbose)


if __name__ == "__main__":
    main()
