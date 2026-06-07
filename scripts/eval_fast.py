"""Fast high-n win-rate eval for ExIt / v2 OrbitNet checkpoints.

Plays games on the engine-faithful standalone FastOrbitWars (no Kaggle harness),
in parallel across processes, with side alternation. Because the policy (eval =
deterministic) and apex are both deterministic, each (seed, side) is a fully
deterministic game — so the only variance is the map seed. All checkpoints are
scored on the SAME seed/side set (paired) for low-variance ranking.

    # every checkpoint of the exit run, 60 games each
    uv run python scripts/eval_fast.py --run v2_exit_a100 \
        --config configs/v2_exit.yaml --iters all --games 60 --workers 6

    # specific iters
    uv run python scripts/eval_fast.py --run v2_exit_a100 \
        --config configs/v2_exit.yaml --iters 10,15,25,last --games 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_E: dict = {}


def _eval_init(cfg_dict: dict, state_dict: dict, opponent: str) -> None:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from v2.config import v2_config_from_dict
    from v2.model import OrbitNet
    from v2.train import make_v2_eval_agent

    cfg = v2_config_from_dict(cfg_dict)
    model = OrbitNet(cfg.model)
    model.load_state_dict(state_dict)
    model.eval()
    _E["rl"] = make_v2_eval_agent(model, cfg, torch.device("cpu"))
    if opponent == "apex":
        from agents.apex import agent as opp
    else:
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent as opp
    _E["opp"] = opp


def _eval_game(args: tuple[int, int]) -> str:
    """Play one game; return 'win'/'loss'/'tie' from the RL agent's perspective."""
    from v2.fast_env import FastOrbitWars

    seed, side = args
    sim = FastOrbitWars(num_agents=2, seed=seed)
    rl, opp = _E["rl"], _E["opp"]
    while not sim.done:
        rl_moves = rl(sim.observation(side)) or []
        opp_moves = opp(sim.observation(1 - side)) or []
        acts: list = [None, None]
        acts[side] = list(rl_moves)
        acts[1 - side] = list(opp_moves)
        sim.step(acts)
    rr = sim.rewards[side]
    orr = sim.rewards[1 - side]
    if rr > 0 and orr > 0:
        return "tie"
    return "win" if rr > 0 else "loss"


def _eval_checkpoint(
    path: Path, cfg_dict: dict, n_games: int, workers: int, opponent: str, base_seed: int
) -> tuple[int, int, int]:
    sd = torch.load(path, map_location="cpu", weights_only=True)["model"]
    jobs = [(base_seed + i, i % 2) for i in range(n_games)]  # alternate sides
    if workers > 1 and n_games > 1:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(
            max_workers=workers, initializer=_eval_init, initargs=(cfg_dict, sd, opponent)
        ) as ex:
            res = list(ex.map(_eval_game, jobs))
    else:
        _eval_init(cfg_dict, sd, opponent)
        res = [_eval_game(j) for j in jobs]
    return res.count("win"), res.count("loss"), res.count("tie")


def _resolve_iters(run_dir: Path, spec: str) -> list[str]:
    if spec == "all":
        nums = sorted(int(p.stem.split("_")[1]) for p in run_dir.glob("ckpt_[0-9]*.pt"))
        return [f"{n:06d}" for n in nums] + (
            ["last"] if (run_dir / "ckpt_last.pt").exists() else []
        )
    return [s.strip() for s in spec.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--iters", default="all", help="comma list (e.g. 10,15,last) or 'all'")
    ap.add_argument("--games", type=int, default=60)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--opponent", default="apex", choices=["apex", "random"])
    ap.add_argument("--seed", type=int, default=20000, help="base seed (shared across checkpoints)")
    args = ap.parse_args()

    from v2.config import load_v2_config, v2_config_to_dict

    cfg_dict = v2_config_to_dict(load_v2_config(args.config))
    run_dir = ROOT / "outputs" / "checkpoints" / args.run

    def _name(it: str) -> str:
        return "ckpt_last.pt" if it == "last" else f"ckpt_{int(it):06d}.pt"

    iters = _resolve_iters(run_dir, args.iters)
    print(
        f"run={args.run}  vs {args.opponent}  n={args.games}  workers={args.workers}  "
        f"(fast_env, side-alternated, paired seeds)"
    )
    print(f"{'iter':>6} | {'win%':>5} {'loss%':>6} {'tie%':>5}")
    print("-" * 32)
    best = (-1.0, None)
    rows = []
    for it in iters:
        path = run_dir / _name(it)
        if not path.exists():
            print(f"{it:>6} | (missing {path.name})")
            continue
        w, l, t = _eval_checkpoint(
            path, cfg_dict, args.games, args.workers, args.opponent, args.seed
        )
        wr = w / max(w + l + t, 1)
        rows.append((it, wr))
        print(f"{it:>6} | {wr:>4.0%} {l / max(args.games, 1):>6.0%} {t / max(args.games, 1):>5.0%}")
        if wr > best[0]:
            best = (wr, it)
    if best[1] is not None:
        print(
            f"\nBEST: iter {best[1]}  win-rate {best[0]:.0%} vs {args.opponent}  (n={args.games})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
