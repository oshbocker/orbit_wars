"""Harvest global-state win/loss labels for the v5 value re-ranker (Axis C).

Plays 2P (or ``--players 4``) games on the REAL Kaggle env among a pool of strong
agents, then records, for EVERY step and EVERY seat, the 16-dim global feature
vector (encoded by the exact same ``encode_global_from_raw`` the agent runs at
inference) labelled with whether that seat won the episode. The value model learns
``state -> P(win for the acting player)`` — used ONLY to break flow-diff ties.

One .npz per (matchup, seed) lands in --out-dir, so reruns resume for free and
scripts/train_value_model.py just globs the directory. Mirror matchups are
included: the model must be calibrated on v5's own state distribution.

    uv run python scripts/harvest_values.py \
        --agents v5,producer,producer_v2 --games 40 --workers 8

NOTE on data tier: this harvests our-tier-and-above self-play (v5/producer/
producer_v2). That is the distribution v5's own near-ties are drawn from, so it is
the relevant signal for a tie-breaker. For an above-our-tier model trained on real
ladder replays, swap the data source (Kaggle EpisodeService GetEpisodeReplay) and
feed the parsed states through the same encoder — the train script is agnostic.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents" / "v5"))

from orbit_lite_v5.value_reranker import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    FEATURE_DIM,
    encode_global_from_raw,
)


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _play_harvest(job: tuple[tuple[str, ...], int, int, Path]) -> dict:
    """One game: encode every (step, seat) global state, label by episode winner."""
    names, n_players, seed, out_path = job
    from kaggle_environments import make

    from agents import load_named_agent

    t0 = time.time()
    agents = [load_named_agent(nm) for nm in names]
    env = make("orbit_wars", configuration={"randomSeed": seed})
    env.run(agents)
    last = env.steps[-1]
    statuses = [s["status"] for s in last]
    rewards = [s["reward"] for s in last]
    # winner label per seat: +1 reward (win or tie-survivor) -> 1, else 0
    labels_by_seat = [1 if (r is not None and r == 1) else 0 for r in rewards]

    # The full board lives only in seat 0's observation (other seats get a sparse
    # obs); owners are ABSOLUTE, so the same board encodes a distinct vector per
    # seat via the player_id argument. Mirrors scripts/harvest_shots.py.
    feats, labels, steps, players = [], [], [], []
    for t, state in enumerate(env.steps):
        obs0 = state[0]["observation"]
        for p in range(n_players):
            try:
                fv = encode_global_from_raw(obs0, p)
            except Exception:
                continue
            feats.append(fv)
            labels.append(labels_by_seat[p])
            steps.append(t)
            players.append(p)

    n = len(labels)
    np.savez_compressed(
        out_path,
        features=np.stack(feats) if n else np.zeros((0, FEATURE_DIM), dtype=np.float32),
        label=np.array(labels, dtype=np.int8),
        step=np.array(steps, dtype=np.int16),
        player=np.array(players, dtype=np.int8),
        agent_names=np.array(list(names)),
        rewards=np.array([float(r) if r is not None else 0.0 for r in rewards], dtype=np.float32),
        seed=np.int64(seed),
        game_steps=np.int64(len(env.steps) - 1),
    )
    return {
        "matchup": " vs ".join(names),
        "seed": seed,
        "n_states": n,
        "pos_rate": float(np.mean(labels)) if n else 0.0,
        "statuses": statuses,
        "rewards": rewards,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", default="v5,producer,producer_v2")
    ap.add_argument("--players", type=int, default=2, choices=(2, 4))
    ap.add_argument("--games", type=int, default=40, help="games per matchup (incl. mirrors)")
    ap.add_argument("--seed", type=int, default=70000, help="base map seed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out-dir", default="outputs/value/raw")
    args = ap.parse_args()

    names = [s.strip() for s in args.agents.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    skipped = 0
    # matchups: all multisets of size n_players from the pool (mirrors included).
    for combo in combinations_with_replacement(names, args.players):
        for i in range(args.games):
            seed = args.seed + i
            tag = "__".join(combo)
            out_path = out_dir / f"{tag}__p{args.players}__{seed}.npz"
            if out_path.exists():
                skipped += 1
                continue
            jobs.append((combo, args.players, seed, out_path))
    print(f"{len(jobs)} games to play ({skipped} already in {out_dir})", flush=True)
    if not jobs:
        return

    from multiprocessing import Pool

    t0 = time.time()
    total = 0
    with Pool(args.workers, initializer=_worker_init, maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(_play_harvest, jobs), 1):
            total += r["n_states"]
            err = " ⚠" if "ERROR" in r["statuses"] else ""
            print(
                f"[{k}/{len(jobs)}] {r['matchup']} seed={r['seed']}: "
                f"{r['n_states']} states (win {r['pos_rate']:.0%}) "
                f"({r['seconds']}s){err}",
                flush=True,
            )
    print(f"done in {(time.time() - t0) / 60:.1f} min, {total} labeled states total")


if __name__ == "__main__":
    main()
