"""Run a trained checkpoint vs an opponent and export game_replay.html.

Supports V1 (TransformerPolicy), V2 (OrbitNet), V3 (OrbitNet + pair/comet
features) and V4 (OrbitNet + v3 features + PopArt/PPG/shot-validator/rich
representation) checkpoints. V3/V4 use the same OrbitNet code path as V2 — they
just need their own config so feature dims match the weights (v3: 24-dim planets;
v4: 28-dim planets + rich globals).

Usage:
    # ExIt checkpoint vs apex (defaults to configs/v2_exit.yaml — base-v2 arch)
    uv run python scripts/replay.py \
        --checkpoint outputs/checkpoints/v2_exit_a100/ckpt_last.pt --exit

    # V2 checkpoint vs apex (default)
    uv run python scripts/replay.py \
        --checkpoint outputs/checkpoints/v2_default/ckpt_last.pt

    # V4 checkpoint vs apex (defaults to configs/v4_ceiling.yaml)
    uv run python scripts/replay.py \
        --checkpoint outputs/checkpoints/v4_ceiling/ckpt_last.pt --v4

    # V3 checkpoint vs apex (defaults to configs/v3_features.yaml)
    uv run python scripts/replay.py \
        --checkpoint outputs/checkpoints/v3_a100/ckpt_last.pt --v3

    # V1 checkpoint vs apex
    uv run python scripts/replay.py \
        --checkpoint outputs/checkpoints/transformer_mixed/ckpt_last.pt \
        --v1 --config configs/transformer_mixed.yaml

    # V2 checkpoint vs hybrid with custom output
    uv run python scripts/replay.py \
        --checkpoint outputs/checkpoints/v2_default/ckpt_last.pt \
        --opponent hybrid --output my_game.html

    # Set seed for reproducible games, play as player 1
    uv run python scripts/replay.py \
        --checkpoint outputs/checkpoints/v2_default/ckpt_last.pt \
        --seed 42 --side 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from kaggle_environments import make


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a game and export HTML replay")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to ckpt_last.pt (or any .pt checkpoint)",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Config YAML (required for --v1, default for v2: configs/v2_default.yaml)",
    )
    parser.add_argument(
        "--v1", action="store_true",
        help="Use V1 TransformerPolicy instead of V2 OrbitNet",
    )
    parser.add_argument(
        "--v3", action="store_true",
        help="V3 OrbitNet checkpoint (defaults config to configs/v3_features.yaml)",
    )
    parser.add_argument(
        "--v4", action="store_true",
        help="V4 OrbitNet checkpoint (defaults config to configs/v4_ceiling.yaml)",
    )
    parser.add_argument(
        "--exit", action="store_true",
        help="ExIt checkpoint (base-v2 OrbitNet; defaults config to configs/v2_exit.yaml)",
    )
    parser.add_argument(
        "--opponent", type=str, default="apex",
        choices=["hybrid", "apex", "random"],
        help="Opponent to play against (default: apex)",
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


def load_v1_agent(ckpt_path: Path, config_path: str, device: torch.device):
    from src.config import load_train_config
    from src.logging import make_eval_agent
    from src.policy import TransformerPolicy

    cfg = load_train_config(config_path)
    policy = TransformerPolicy(cfg.model, cfg.env).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    update = ckpt.get("update", "?")
    print(f"Loaded V1 checkpoint: {ckpt_path} (update {update})")
    return make_eval_agent(policy, cfg, device)


def load_v2_agent(ckpt_path: Path, config_path: str, device: torch.device,
                  label: str = "V2"):
    from v2.config import load_v2_config
    from v2.model import OrbitNet
    from v2.train import make_v2_eval_agent

    cfg = load_v2_config(config_path)
    model = OrbitNet(cfg.model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    update = ckpt.get("update", "?")
    print(f"Loaded {label} checkpoint: {ckpt_path} (update {update})")
    return make_v2_eval_agent(model, cfg, device)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return

    if args.v1:
        config_path = args.config or "configs/transformer_mixed.yaml"
        rl_agent = load_v1_agent(ckpt_path, config_path, device)
        agent_label = "V1 RL"
    elif args.v4:
        config_path = args.config or "configs/v4_ceiling.yaml"
        rl_agent = load_v2_agent(ckpt_path, config_path, device, label="V4")
        agent_label = "V4 RL"
    elif args.v3:
        config_path = args.config or "configs/v3_features.yaml"
        rl_agent = load_v2_agent(ckpt_path, config_path, device, label="V3")
        agent_label = "V3 RL"
    elif args.exit:
        config_path = args.config or "configs/v2_exit.yaml"
        rl_agent = load_v2_agent(ckpt_path, config_path, device, label="ExIt")
        agent_label = "ExIt RL"
    else:
        config_path = args.config or "configs/v2_default.yaml"
        rl_agent = load_v2_agent(ckpt_path, config_path, device)
        agent_label = "V2 RL"

    opp_agent = load_opponent(args.opponent)

    if args.side == 0:
        players = [rl_agent, opp_agent]
        rl_label = f"Player 0 ({agent_label})"
        opp_label = f"Player 1 ({args.opponent})"
    else:
        players = [opp_agent, rl_agent]
        rl_label = f"Player 0 ({args.opponent})"
        opp_label = f"Player 1 ({agent_label})"

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
