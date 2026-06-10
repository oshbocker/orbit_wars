"""Round-robin arena over a pool of agents on the real Kaggle env.

This is THE gate metric for the leaderboard climb (rl_research/LEADERBOARD_CLIMB_PLAN.md):
side-alternated, paired seeds, one process per game (vendored public agents keep
module-level state, so every game gets a freshly-loaded agent in a worker process).
Results append to a CSV; reruns skip already-played (pair, seed, side) games, so you
can bump --games incrementally.

    # first matrix: public pool + v5, 10 paired games per pair
    uv run python scripts/arena.py \
        --agents producer,tamrazov_1224,distance_1100,enders_1000,v5 \
        --games 10 --workers 8

    # include our ExIt champion
    uv run python scripts/arena.py \
        --agents producer,v5,exit:outputs/checkpoints/v2_exit_a100/ckpt_000020.pt:configs/v2_exit.yaml \
        --games 10

    # 4-player FFA mode (LB mixes 2P and 4P): every 4-agent combo from the pool,
    # seats rotated per game, players ranked by (engine reward, final board score).
    # --games here = games per COMBO; each pair co-occurs in C(n-2,2) combos.
    uv run python scripts/arena.py --players 4 \
        --agents producer,tamrazov_1224,distance_1100,enders_1000,v5 \
        --games 4 --workers 6

Agent specs: vendored names from agents.external (producer, tamrazov_1224,
distance_1100, shot_validator_hybrid, enders_1000, ow_proto, reinforce_958),
"v5" (+ "v5:key=val+key=val" overrides), "random", or "exit:<ckpt.pt>:<config.yaml>".
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
    if spec.startswith("v5:"):
        # "v5:roi_threshold=1.4+horizon=20" -> "v5_roi_threshold1.4_horizon20"
        kvs = spec.split(":", 1)[1].split("+")
        return "v5_" + "_".join(kv.replace("=", "") for kv in kvs)
    return spec


def _build_agent(spec: str):
    """Fresh agent callable for one game (worker process)."""
    if spec.startswith("v5:"):
        # Our producer fork with config overrides: "v5:roi_threshold=1.4+horizon=20"
        # patches both the 2P default and the 4P preset (A/B sweeps without code
        # edits). Plain names (v5, producer, ...) resolve via the fallthrough.
        import dataclasses
        import importlib.util
        import itertools

        global _V5_COUNTER
        if "_V5_COUNTER" not in globals():
            _V5_COUNTER = itertools.count()
        main_py = ROOT / "agents" / "v5" / "main.py"
        modname = f"_v5ov_{next(_V5_COUNTER)}"
        mspec = importlib.util.spec_from_file_location(modname, main_py)
        assert mspec is not None and mspec.loader is not None
        mod = importlib.util.module_from_spec(mspec)
        sys.modules[modname] = mod
        mspec.loader.exec_module(mod)
        ftypes = {f.name: f.type for f in dataclasses.fields(mod.ProducerLiteConfig)}
        overrides = {}
        for kv in spec.split(":", 1)[1].split("+"):
            k, v = kv.split("=")
            overrides[k] = int(v) if ftypes[k] in ("int", int) else float(v)
        cfg2 = dataclasses.replace(mod.ProducerLiteConfig(), **overrides)
        cfg4 = dataclasses.replace(mod.CONFIG_4P, **overrides)
        mod._config_for = lambda pc: cfg4 if int(pc) >= 4 else cfg2
        fn = mod.agent

        def v5_agent(obs, config=None):
            return fn(obs)

        return v5_agent
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
    from agents import load_named_agent

    return load_named_agent(spec)


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _field(entry, name: str, idx: int):
    """Planet/fleet field access: Struct attribute first, then dict, then list."""
    if hasattr(entry, name):
        return getattr(entry, name)
    if isinstance(entry, dict):
        return entry[name]
    return entry[idx]


def _final_scores(env, n_players: int) -> list[float]:
    """Per-player score (ships on owned planets + ships in owned fleets) at game end."""
    obs = env.steps[-1][0].observation
    scores = [0.0] * n_players
    for p in _field(obs, "planets", 0) or []:
        owner = int(_field(p, "owner", 1))
        if 0 <= owner < n_players:
            scores[owner] += float(_field(p, "ships", 5))
    for f in _field(obs, "fleets", 0) or []:
        owner = int(_field(f, "owner", 1))
        if 0 <= owner < n_players:
            scores[owner] += float(_field(f, "ships", 6))
    return scores


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
        "steps": len(env.steps) - 1,
        "seconds": round(time.time() - t0, 1),
    }


def _play_4p(job: tuple[tuple[str, str, str, str], int, int]) -> dict:
    """One 4P FFA game: (specs in seat order, seed, rot). Returns a result row.

    Rank is by (engine reward, final board score) descending — reward separates the
    winner (+1) from losers (−1); board score orders the losers (engine reward alone
    would tie all non-winners, starving the pairwise matrix of signal).
    """
    specs, seed, rot = job
    from kaggle_environments import make

    t0 = time.time()
    env = make("orbit_wars", configuration={"randomSeed": seed})
    env.run([_build_agent(s) for s in specs])
    last = env.steps[-1]
    rewards = [(-1.0 if s.reward is None else float(s.reward)) for s in last]
    scores = _final_scores(env, 4)
    keys = list(zip(rewards, scores, strict=True))
    ranks = [1 + sum(1 for k in keys if k > keys[i]) for i in range(4)]  # ties share rank
    row: dict = {
        "seed": seed,
        "rot": rot,
        "steps": len(env.steps) - 1,
        "seconds": round(time.time() - t0, 1),
    }
    for i in range(4):
        row[f"agent_{i}"] = spec_label(specs[i])
        row[f"reward_{i}"] = rewards[i]
        row[f"score_{i}"] = round(scores[i], 1)
        row[f"rank_{i}"] = ranks[i]
        row[f"status_{i}"] = last[i].status
    return row


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
    "steps",
    "seconds",
]

FIELDS_4P = (
    [f"agent_{i}" for i in range(4)]
    + ["seed", "rot"]
    + [f"{k}_{i}" for i in range(4) for k in ("reward", "score", "rank", "status")]
    + ["steps", "seconds"]
)


def load_done(out_csv: Path, players: int) -> set[tuple]:
    done = set()
    if out_csv.exists():
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                if players == 4:
                    done.add((*(row[f"agent_{i}"] for i in range(4)), int(row["seed"])))
                else:
                    done.add((row["agent_a"], row["agent_b"], int(row["seed"]), int(row["a_side"])))
    return done


def pairwise_results(out_csv: Path, players: int) -> list[tuple[str, str, float]]:
    """Flatten the CSV into (label_a, label_b, points_for_a) tuples (tie = 0.5).

    4P games contribute one tuple per co-occurring pair (6 per game), decided by rank.
    """
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if players == 4:
            for i in range(4):
                for j in range(i + 1, 4):
                    ri, rj = int(r[f"rank_{i}"]), int(r[f"rank_{j}"])
                    p = 1.0 if ri < rj else 0.0 if ri > rj else 0.5
                    out.append((r[f"agent_{i}"], r[f"agent_{j}"], p))
        else:
            p = 1.0 if r["outcome"] == "a" else 0.0 if r["outcome"] == "b" else 0.5
            out.append((r["agent_a"], r["agent_b"], p))
    return out


def print_mean_ranks(out_csv: Path, labels: list[str]) -> None:
    """4P only: mean finishing rank (1=win .. 4=last) and win rate per agent."""
    with open(out_csv) as f:
        rows = list(csv.DictReader(f))
    acc = {a: [0.0, 0, 0] for a in labels}  # rank_sum, games, wins
    for r in rows:
        for i in range(4):
            a = r[f"agent_{i}"]
            if a in acc:
                acc[a][0] += int(r[f"rank_{i}"])
                acc[a][1] += 1
                acc[a][2] += float(r[f"reward_{i}"]) > 0
    print("\n4P mean finishing rank (1 = winner, 4 = last) and outright-win rate:")
    for a in sorted(labels, key=lambda x: acc[x][0] / max(acc[x][1], 1)):
        rs, g, w = acc[a]
        if g:
            print(f"  {a:<28} rank {rs / g:.2f}   win {w / g:>4.0%}   ({g} games)")


def print_matrix(results: list[tuple[str, str, float]], labels: list[str]) -> None:
    # wins[a][b] = (points, games) for a vs b; tie = 0.5
    pts: dict = {a: {b: [0.0, 0] for b in labels} for a in labels}
    for a, b, p in results:
        if a not in pts or b not in pts[a]:
            continue
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
    ap.add_argument(
        "--games",
        type=int,
        default=10,
        help="2P: games per pair (side-alternated). 4P: games per 4-agent combo (seat-rotated)",
    )
    ap.add_argument("--players", type=int, default=2, choices=(2, 4))
    ap.add_argument("--seed", type=int, default=20000, help="base map seed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out", default=None, help="default outputs/arena/arena[_4p].csv")
    args = ap.parse_args()

    specs = [s.strip() for s in args.agents.split(",") if s.strip()]
    labels = {s: spec_label(s) for s in specs}
    is_4p = args.players == 4
    out_csv = Path(
        args.out or ("outputs/arena/arena_4p.csv" if is_4p else "outputs/arena/arena.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out_csv, args.players)
    jobs: list = []
    if is_4p:
        if len(specs) < 4:
            ap.error("--players 4 needs at least 4 agent specs")
        for combo in combinations(specs, 4):
            for i in range(args.games):
                seed, rot = args.seed + i, i % 4
                seats = tuple(combo[rot:] + combo[:rot])
                if (*(labels[s] for s in seats), seed) not in done:
                    jobs.append((seats, seed, rot))
    else:
        for spec_a, spec_b in combinations(specs, 2):
            for i in range(args.games):
                seed, a_side = args.seed + i, i % 2
                if (labels[spec_a], labels[spec_b], seed, a_side) not in done:
                    jobs.append((spec_a, spec_b, seed, a_side))
    print(f"{len(jobs)} games to play ({len(done)} already in {out_csv})")

    fields = FIELDS_4P if is_4p else FIELDS
    play = _play_4p if is_4p else _play
    new_file = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        if jobs:
            from multiprocessing import Pool

            t0 = time.time()
            with Pool(args.workers, initializer=_worker_init, maxtasksperchild=4) as pool:
                for k, row in enumerate(pool.imap_unordered(play, jobs), 1):
                    writer.writerow(row)
                    f.flush()
                    if is_4p:
                        statuses = [row[f"status_{i}"] for i in range(4)]
                        err = " ⚠" if "ERROR" in statuses else ""
                        order = sorted(range(4), key=lambda i: row[f"rank_{i}"])
                        finish = " > ".join(row[f"agent_{i}"] for i in order)
                        print(
                            f"[{k}/{len(jobs)}] seed={row['seed']} rot={row['rot']} "
                            f"{finish} ({row['seconds']}s){err}",
                            flush=True,
                        )
                    else:
                        err = " ⚠" if "ERROR" in (row["status_a"], row["status_b"]) else ""
                        print(
                            f"[{k}/{len(jobs)}] {row['agent_a']} vs {row['agent_b']} "
                            f"seed={row['seed']} side={row['a_side']} -> {row['outcome']} "
                            f"({row['seconds']}s){err}",
                            flush=True,
                        )
            print(f"done in {(time.time() - t0) / 60:.1f} min")

    print_matrix(pairwise_results(out_csv, args.players), [labels[s] for s in specs])
    if is_4p:
        print_mean_ranks(out_csv, [labels[s] for s in specs])


if __name__ == "__main__":
    main()
