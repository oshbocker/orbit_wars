"""Run a trained transformer checkpoint vs an opponent and export game_replay.html.

Usage:
    # Mixed checkpoint vs hybrid (default)
    uv run python scripts/replay.py --checkpoint outputs/checkpoints/transformer_mixed/ckpt_last.pt

    # Dagger checkpoint vs apex
    uv run python scripts/replay.py \
        --checkpoint outputs/checkpoints/transformer_dagger/ckpt_last.pt \
        --config configs/transformer_dagger.yaml \
        --opponent apex

    # Specify output file
    uv run python scripts/replay.py \
        --checkpoint outputs/checkpoints/transformer_mixed/ckpt_last.pt \
        --output my_game.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from kaggle_environments import make

from src.config import load_train_config
from src.logging import make_eval_agent
from src.policy import TransformerPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a game and export HTML replay")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to ckpt_last.pt (or any .pt checkpoint)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/transformer_mixed.yaml",
        help="Config YAML (must match the checkpoint's model architecture)",
    )
    parser.add_argument(
        "--opponent", type=str, default="hybrid",
        choices=["hybrid", "apex", "random"],
        help="Opponent to play against (default: hybrid)",
    )
    parser.add_argument(
        "--output", type=str, default="game_replay.html",
        help="Output HTML file path (default: game_replay.html)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for the game (default: random)",
    )
    parser.add_argument(
        "--side", type=int, default=0, choices=[0, 1],
        help="Which player slot the RL agent takes (0 or 1, default: 0)",
    )
    return parser.parse_args()


def load_opponent(name: str):
    if name == "hybrid":
        from agents.hybrid import agent
        return agent
    if name == "apex":
        from agents.apex import agent
        return agent
    if name == "random":
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent
        return random_agent
    raise ValueError(f"Unknown opponent: {name}")


def main() -> None:
    args = parse_args()

    # Load config and checkpoint
    cfg = load_train_config(args.config)
    device = torch.device("cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    policy = TransformerPolicy(cfg.model, cfg.env).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    update = ckpt.get("update", "?")
    print(f"Loaded checkpoint: {ckpt_path} (update {update})")

    # Build agents
    rl_agent = make_eval_agent(policy, cfg, device)
    opp_agent = load_opponent(args.opponent)

    if args.side == 0:
        players = [rl_agent, opp_agent]
        rl_label, opp_label = "Player 0 (RL)", f"Player 1 ({args.opponent})"
    else:
        players = [opp_agent, rl_agent]
        rl_label, opp_label = f"Player 0 ({args.opponent})", "Player 1 (RL)"

    # Run game
    configuration = {}
    if args.seed is not None:
        configuration["seed"] = args.seed

    print(f"Running: {rl_label} vs {opp_label}...")
    env = make("orbit_wars", configuration=configuration, debug=False)
    env.run(players)

    # Determine result
    rl_reward = env.steps[-1][args.side].reward
    steps = len(env.steps)
    if rl_reward is None or rl_reward == 0:
        result = "TIE"
    elif rl_reward > 0:
        result = "WIN"
    else:
        result = "LOSS"
    print(f"Result: {result} ({steps} steps)")

    # Export replay
    html = env.render(mode="html", width=800, height=600)
    output_path = Path(args.output)
    output_path.write_text(html)
    print(f"Replay saved: {output_path}")


if __name__ == "__main__":
    main()
