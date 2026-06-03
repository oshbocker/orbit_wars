"""High-n win-rate-vs-apex eval for ExIt checkpoints (low-noise A/B verdict).

The per-iteration evals logged during training use only n=15 games, which is far
too noisy to compare arms (±~13%). This evaluates specific checkpoints at high n.

    # one run, a few checkpoints
    uv run python scripts/eval_ab_winrate.py --run v2_exit_embed128 \
        --config configs/v2_exit_embed128.yaml --iters 15,25,30 --games 50

    # head-to-head: best checkpoint of each arm
    uv run python scripts/eval_ab_winrate.py --run v2_exit_embed256 \
        --config configs/v2_exit_embed256.yaml --iters last --games 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _ckpt_name(it: str) -> str:
    return "ckpt_last.pt" if it == "last" else f"ckpt_{int(it):06d}.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run_name under outputs/checkpoints/")
    ap.add_argument("--config", required=True, help="matching config YAML (feature dims)")
    ap.add_argument("--iters", default="last", help="comma list of iters, e.g. 15,25,30 or 'last'")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--opponent", default="apex", choices=["apex", "random"])
    args = ap.parse_args()

    from evaluation.evaluate import run_games
    from v2.config import load_v2_config
    from v2.model import OrbitNet
    from v2.train import make_v2_eval_agent

    if args.opponent == "apex":
        from agents.apex import agent as opp
    else:
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent as opp

    cfg = load_v2_config(args.config)
    device = torch.device("cpu")
    ckpt_dir = ROOT / "outputs" / "checkpoints" / args.run

    print(f"run={args.run}  config={args.config}  vs {args.opponent}  n={args.games}")
    print(f"{'iter':>6} | {'win':>5} {'loss':>5} {'tie':>5}")
    print("-" * 30)
    for it in args.iters.split(","):
        path = ckpt_dir / _ckpt_name(it.strip())
        if not path.exists():
            print(f"{it:>6} | (missing {path.name})")
            continue
        model = OrbitNet(cfg.model).to(device)
        ck = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        model.eval()
        agent = make_v2_eval_agent(model, cfg, device)
        r = run_games(agent, opp, n_games=args.games)
        print(f"{it:>6} | {r['win_rate']:>4.0%} {r['loss_rate']:>5.0%} {r['tie_rate']:>5.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
