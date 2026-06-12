"""Smoke driver for Orbit Wars: run ONE real Kaggle-env game between named agents.

This is the fastest way to verify the repo runs end-to-end: it resolves agents
through the same central resolver the arena uses (``agents.load_named_agent``),
plays a full game on the real ``kaggle_environments`` engine, and prints the
outcome. Optionally dumps the animated HTML replay (open in a browser, or
screenshot headless with firefox).

Usage (from the repo root):
    uv run python .claude/skills/run-orbit-wars/driver.py                       # random vs random (fast)
    uv run python .claude/skills/run-orbit-wars/driver.py --agents v5,producer --seed 42
    uv run python .claude/skills/run-orbit-wars/driver.py --agents v5,producer,ow_proto,enders_1000  # 4P FFA
    uv run python .claude/skills/run-orbit-wars/driver.py --agents v5,producer --html /tmp/game.html

Agent names: v5 | producer | ow_proto | enders_1000 | tamrazov_1224 |
distance_1100 | reinforce_958 | shot_validator_hybrid | random
(anything ``agents.load_named_agent`` accepts; ``exit:<ckpt>:<config>`` and
``v5:key=val`` specs are arena-only — use scripts/arena.py for those).

Exit code 0 iff every seat finished with status DONE.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Repo root = three levels above this file (.claude/skills/run-orbit-wars/).
# Needed because python sets sys.path[0] to the SCRIPT dir, not the cwd, so
# ``import agents`` would fail no matter where the driver is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", default="random,random",
                   help="comma-separated agent names (2 or 4)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--html", default=None, help="write the animated HTML replay here")
    args = p.parse_args()

    names = [n.strip() for n in args.agents.split(",")]
    if len(names) not in (2, 4):
        p.error("--agents needs exactly 2 or 4 names")

    # Heavy imports after argparse so --help stays fast.
    import agents
    from kaggle_environments import make

    callables = [agents.load_named_agent(n) for n in names]
    env = make("orbit_wars", configuration={"randomSeed": args.seed})
    t0 = time.time()
    steps = env.run(callables)
    last = steps[-1]
    statuses = [s["status"] for s in last]
    rewards = [s["reward"] for s in last]
    print(f"seed={args.seed} steps={len(steps)} wall={time.time() - t0:.1f}s")
    for n, st, r in zip(names, statuses, rewards):
        print(f"  {n:<24} status={st:<6} reward={r}")

    if args.html:
        with open(args.html, "w") as f:
            f.write(env.render(mode="html"))
        print(f"replay -> {args.html}")

    return 0 if all(s == "DONE" for s in statuses) else 1


if __name__ == "__main__":
    sys.exit(main())
