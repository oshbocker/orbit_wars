"""Round-robin arena over a pool of agents on the real Kaggle env.

This is THE gate metric for the leaderboard climb (rl_research/LEADERBOARD_CLIMB_PLAN.md):
side-alternated, paired seeds, one process per game (vendored public agents keep
module-level state, so every game gets a freshly-loaded agent in a worker process).
Results append to a CSV; reruns skip already-played (pair, seed, side) games, so you
can bump --games incrementally.

    # first matrix: public pool + apex, 10 paired games per pair
    uv run python scripts/arena.py \
        --agents producer,tamrazov_1224,distance_1100,enders_1000,apex \
        --games 10 --workers 8

    # include our ExIt champion
    uv run python scripts/arena.py \
        --agents producer,apex,exit:outputs/checkpoints/v2_exit_a100/ckpt_000020.pt:configs/v2_exit.yaml \
        --games 10

Agent specs: vendored names from agents.external (producer, tamrazov_1224,
distance_1100, shot_validator_hybrid, enders_1000, ow_proto, reinforce_958),
"apex", "hybrid", or "exit:<ckpt.pt>:<config.yaml>" (label = ckpt run/iter).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_CACHE: dict = {}


def spec_label(spec: str) -> str:
    if spec.startswith("exit:"):
        ckpt = Path(spec.split(":")[1])
        return f"exit_{ckpt.parent.name}_{ckpt.stem.removeprefix('ckpt_')}"
    return spec


def _build_agent(spec: str):
    """Fresh agent callable for one game (worker process)."""
    if spec == "apex":
        from agents.apex import agent

        return agent
    if spec == "hybrid":
        from agents.hybrid import agent

        return agent
    if spec.startswith("exit:"):
        _, ckpt_path, cfg_path = spec.split(":")
        if spec not in _CACHE:
            import torch

            from v2.config import load_v2_config
            from v2.model import OrbitNet
            from v2.train import make_v2_eval_agent

            cfg = load_v2_config(cfg_path)
            model = OrbitNet(cfg.model)
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
            model.load_state_dict(sd)
            model.eval()
            _CACHE[spec] = make_v2_eval_agent(model, cfg, torch.device("cpu"))
        return _CACHE[spec]
    from agents.external import load_agent

    return load_agent(spec)


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _play(job: tuple[str, str, int, int]) -> dict:
    """One game: (spec_a, spec_b, seed, a_side). Returns a result row."""
    spec_a, spec_b, seed, a_side = job
    from kaggle_environments import make

    t0 = time.time()
    agents = [None, None]
    agents[a_side] = _build_agent(spec_a)
    agents[1 - a_side] = _build_agent(spec_b)
    env = make("orbit_wars", configuration={"randomSeed": seed})
    env.run(agents)
    last = env.steps[-1]
    ra, rb = last[a_side].reward, last[1 - a_side].reward
    statuses = [s.status for s in last]
    ra = -1 if ra is None else ra
    rb = -1 if rb is None else rb
    outcome = "tie" if ra == rb else ("a" if ra > rb else "b")
    return {
        "agent_a": spec_label(spec_a),
        "agent_b": spec_label(spec_b),
        "seed": seed,
        "a_side": a_side,
        "outcome": outcome,
        "reward_a": ra,
        "reward_b": rb,
        "status_a": statuses[a_side],
        "status_b": statuses[1 - a_side],
        "seconds": round(time.time() - t0, 1),
    }


FIELDS = [
    "agent_a",
    "agent_b",
    "seed",
    "a_side",
    "outcome",
    "reward_a",
    "reward_b",
    "status_a",
    "status_b",
    "seconds",
]


def load_done(out_csv: Path) -> set[tuple]:
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                done.add((row["agent_a"], row["agent_b"], int(row["seed"]), int(row["a_side"])))
    return done


def print_matrix(out_csv: Path, labels: list[str]) -> None:
    rows = list(csv.DictReader(open(out_csv)))
    # wins[a][b] = (points, games) for a vs b; tie = 0.5
    pts: dict = {a: {b: [0.0, 0] for b in labels} for a in labels}
    for r in rows:
        a, b = r["agent_a"], r["agent_b"]
        if a not in pts or b not in pts[a]:
            continue
        p = 1.0 if r["outcome"] == "a" else 0.0 if r["outcome"] == "b" else 0.5
        pts[a][b][0] += p
        pts[a][b][1] += 1
        pts[b][a][0] += 1 - p
        pts[b][a][1] += 1
    width = max(len(x) for x in labels) + 2
    mean_wr = {}
    for a in labels:
        tp = sum(pts[a][b][0] for b in labels if b != a)
        tg = sum(pts[a][b][1] for b in labels if b != a)
        mean_wr[a] = tp / tg if tg else 0.0
    order = sorted(labels, key=lambda a: -mean_wr[a])
    print("\nwin rate of ROW vs COL (ties = 0.5):")
    print(" " * width + "".join(f"{b[:10]:>12}" for b in order) + f"{'MEAN':>8}")
    for a in order:
        cells = []
        for b in order:
            p, g = pts[a][b]
            cells.append(f"{'—':>12}" if a == b else (f"{p / g:>11.0%} " if g else f"{'·':>12}"))
        print(f"{a:<{width}}" + "".join(cells) + f"{mean_wr[a]:>7.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", required=True, help="comma-separated agent specs")
    ap.add_argument("--games", type=int, default=10, help="games per pair (side-alternated)")
    ap.add_argument("--seed", type=int, default=20000, help="base map seed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default="outputs/arena/arena.csv")
    args = ap.parse_args()

    specs = [s.strip() for s in args.agents.split(",") if s.strip()]
    labels = {s: spec_label(s) for s in specs}
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out_csv)
    jobs = []
    for spec_a, spec_b in combinations(specs, 2):
        for i in range(args.games):
            seed, a_side = args.seed + i, i % 2
            if (labels[spec_a], labels[spec_b], seed, a_side) not in done:
                jobs.append((spec_a, spec_b, seed, a_side))
    print(f"{len(jobs)} games to play ({len(done)} already in {out_csv})")

    new_file = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        if jobs:
            from multiprocessing import Pool

            t0 = time.time()
            with Pool(args.workers, initializer=_worker_init, maxtasksperchild=4) as pool:
                for k, row in enumerate(pool.imap_unordered(_play, jobs), 1):
                    writer.writerow(row)
                    f.flush()
                    err = " ⚠" if "ERROR" in (row["status_a"], row["status_b"]) else ""
                    print(
                        f"[{k}/{len(jobs)}] {row['agent_a']} vs {row['agent_b']} "
                        f"seed={row['seed']} side={row['a_side']} -> {row['outcome']} "
                        f"({row['seconds']}s){err}",
                        flush=True,
                    )
            print(f"done in {(time.time() - t0) / 60:.1f} min")

    print_matrix(out_csv, [labels[s] for s in specs])


if __name__ == "__main__":
    main()
