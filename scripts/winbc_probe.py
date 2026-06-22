"""Regime-check probe: does the win-weighted BC clone MOVE, and does it beat weak bots?

Loads agents/external/bc_teacher.py with the winbc checkpoint (via OW_BC_TEACHER_CKPT) and
plays N 2P games vs an opponent, reporting the clone's avg launches/turn (the real-play
no-op-collapse check) and win rate. The arena (scripts/arena.py) gives the paired win%; this
adds the movement signal arena doesn't.

    OW_BC_TEACHER_CKPT=outputs/checkpoints/winbc/ckpt.pt \
        uv run python scripts/winbc_probe.py --opponent random --games 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fleet_owner(fl):
    """Fleet owner, tolerating Struct / dict / list rows (fleets = [id, owner, ...])."""
    if hasattr(fl, "owner"):
        return fl.owner
    if isinstance(fl, dict):
        return fl.get("owner", -1)
    try:
        return fl[1]
    except (TypeError, IndexError, KeyError):
        return -1


def _enemy_fleets(obs, seat: int) -> int:
    """Count in-flight fleets owned by a real opponent (not us, not neutral). >0 means the
    opponent has committed force this turn = the 'contested' / under-pressure regime."""
    fleets = obs.get("fleets") if isinstance(obs, dict) else getattr(obs, "fleets", None)
    if not fleets:
        return 0
    return sum(1 for fl in fleets if _fleet_owner(fl) not in (seat, -1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--opponent", default="random")
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--seed", type=int, default=90000)
    args = ap.parse_args()

    from kaggle_environments import make

    from agents import load_named_agent

    clone = load_named_agent("bc_teacher")
    wins = ties = 0
    # [launches, turns] split by whether the opponent has committed force this turn.
    # The no-op collapse is STATE-DEPENDENT (2.21 l/t vs random -> ~0.4 under pressure),
    # so a global launches/turn hides it; "contested" = the states that decide games.
    buckets = {"contested": [0, 0], "quiet": [0, 0]}
    for g in range(args.games):
        env = make("orbit_wars", configuration={"randomSeed": args.seed + g})
        seat = g % 2  # alternate sides
        agents = [None, None]
        agents[seat] = clone
        agents[1 - seat] = load_named_agent(args.opponent)
        env.run(agents)
        steps = env.steps
        for t in range(1, len(steps)):
            cell = steps[t][seat]
            if not isinstance(cell, dict):
                continue
            obs = cell.get("observation") or {}
            key = "contested" if _enemy_fleets(obs, seat) > 0 else "quiet"
            act = cell.get("action")
            buckets[key][0] += len(act) if act else 0
            buckets[key][1] += 1
        rew = steps[-1][seat].get("reward", 0)
        orew = steps[-1][1 - seat].get("reward", 0)
        if rew is not None and orew is not None:
            if rew > orew:
                wins += 1
            elif rew == orew:
                ties += 1
        print(f"  game {g} seat{seat}: result {rew} vs {orew}", flush=True)

    def lpt(b):
        return b[0] / b[1] if b[1] else 0.0
    n_launch = buckets["contested"][0] + buckets["quiet"][0]
    n_turns = buckets["contested"][1] + buckets["quiet"][1]
    overall = n_launch / n_turns if n_turns else 0.0
    print(f"\nvs {args.opponent}  n={args.games}: win {wins}/{args.games} ties {ties}")
    print(f"  launches/turn overall={overall:.2f}  (collapse if ~0.0)")
    print(f"  launches/turn contested={lpt(buckets['contested']):.2f} "
          f"(n={buckets['contested'][1]} turns)  "
          f"quiet={lpt(buckets['quiet']):.2f} (n={buckets['quiet'][1]} turns)")
    print("  -> state-dependent passivity if contested << quiet")


if __name__ == "__main__":
    main()
