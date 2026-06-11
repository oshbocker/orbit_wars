"""Harvest dense per-shot labels for the v5 shot validator (Phase 2.1).

Plays 2P games on the REAL Kaggle env among a pool of strong agents, records
every shot each side emits (24-dim konbu-style features encoded on the exact
obs the agent saw), then labels each shot post-game with

    label = did the shooter own the ray-cast target planet at any step in
            [arrival, arrival+10]?

One .npz per (matchup, seed) lands in --out-dir, so reruns resume for free and
scripts/train_shot_validator.py just globs the directory. Mirror matchups are
included: the validator must be calibrated on v5's own shot distribution.

    uv run python scripts/harvest_shots.py \
        --agents v5,producer,ow_proto,enders_1000 --games 30 --workers 8
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents" / "v5"))

from orbit_lite_v5.shot_validator import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    FEATURE_DIM,
    encode_shot,
    find_target_ray,
    shot_eta,
)

LABEL_WINDOW = 10  # steps after arrival in which owning the target counts as success


def _field(entry, name: str, idx: int):
    """Planet field access: Struct attribute first, then dict, then list."""
    if hasattr(entry, name):
        return getattr(entry, name)
    if isinstance(entry, dict):
        return entry[name]
    return entry[idx]


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _recording_agent(inner, shots: list):
    """Wrap an agent: record (step, player, src, tgt, ships, own_self, eta, feat)
    for every emitted move whose target the ray cast can identify."""

    def wrapped(obs, config=None):
        moves = inner(obs, config)
        if not moves:
            return moves
        try:
            planets = obs["planets"]
            step = int(obs.get("step", 0))
            player = int(obs.get("player", 0))
            pos = {int(p[0]): (float(p[2]), float(p[3])) for p in planets}
            rows = {int(p[0]): p for p in planets}
            owners = {int(p[0]): int(p[1]) for p in planets}
            for mv in moves:
                try:
                    src_id = int(mv[0])
                    ang = float(mv[1])
                    ships = int(mv[2])
                except (TypeError, ValueError, IndexError):
                    continue
                if src_id not in pos:
                    continue
                tgt_id = find_target_ray(pos[src_id], ang, planets)
                if tgt_id < 0 or tgt_id == src_id:
                    continue
                feat = encode_shot(obs, src_id, tgt_id, ships)
                if feat is None:
                    continue
                eta = shot_eta(rows[src_id], rows[tgt_id], ships)
                own_self = 1 if owners.get(tgt_id, -2) == player else 0
                shots.append((step, player, src_id, tgt_id, ships, own_self, eta, feat))
        except Exception:
            pass  # recording must never break the game
        return moves

    return wrapped


def _play_harvest(job: tuple[str, str, int, Path]) -> dict:
    """One game: record both sides' shots, label from the step history."""
    name_a, name_b, seed, out_path = job
    from kaggle_environments import make

    from agents import load_named_agent

    t0 = time.time()
    shots: list = []
    agents: list = [
        _recording_agent(load_named_agent(name_a), shots),
        _recording_agent(load_named_agent(name_b), shots),
    ]
    env = make("orbit_wars", configuration={"randomSeed": seed})
    env.run(agents)
    statuses = [s["status"] for s in env.steps[-1]]
    last_step = len(env.steps) - 1

    # owner_by_step[t][pid] from player 0's (full-board) observation at step t
    owner_by_step: list[dict[int, int]] = []
    for state in env.steps:
        obs0 = state[0]["observation"]
        owner_by_step.append(
            {
                int(_field(p, "id", 0)): int(_field(p, "owner", 1))
                for p in (_field(obs0, "planets", 0) or [])
            }
        )

    names = (name_a, name_b)
    feats, labels, meta = [], [], []
    dropped = 0
    for step, player, src, tgt, ships, own_self, eta, feat in shots:
        arr = step + max(1, int(math.ceil(eta)))
        if arr > last_step:
            dropped += 1  # outcome unobservable: game ended before arrival
            continue
        hi = min(arr + LABEL_WINDOW, last_step)
        label = int(any(owner_by_step[t].get(tgt, -2) == player for t in range(arr, hi + 1)))
        feats.append(feat)
        labels.append(label)
        meta.append((step, player, src, tgt, ships, own_self, eta))

    n = len(labels)
    np.savez_compressed(
        out_path,
        features=np.stack(feats) if n else np.zeros((0, FEATURE_DIM), dtype=np.float32),
        label=np.array(labels, dtype=np.int8),
        step=np.array([m[0] for m in meta], dtype=np.int16),
        player=np.array([m[1] for m in meta], dtype=np.int8),
        src=np.array([m[2] for m in meta], dtype=np.int16),
        tgt=np.array([m[3] for m in meta], dtype=np.int16),
        ships=np.array([m[4] for m in meta], dtype=np.int32),
        own_self=np.array([m[5] for m in meta], dtype=np.int8),
        eta=np.array([m[6] for m in meta], dtype=np.float32),
        shooter=np.array([names[m[1]] for m in meta]),
        seed=np.int64(seed),
        game_steps=np.int64(last_step),
    )
    return {
        "matchup": f"{name_a} vs {name_b}",
        "seed": seed,
        "n_shots": n,
        "n_dropped": dropped,
        "pos_rate": float(np.mean(labels)) if n else 0.0,
        "statuses": statuses,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", default="v5,producer,ow_proto,enders_1000")
    ap.add_argument("--games", type=int, default=30, help="games per matchup (incl. mirrors)")
    ap.add_argument("--seed", type=int, default=50000, help="base map seed")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--out-dir", default="outputs/validator/raw")
    args = ap.parse_args()

    names = [s.strip() for s in args.agents.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    skipped = 0
    for name_a, name_b in combinations_with_replacement(names, 2):
        for i in range(args.games):
            seed = args.seed + i
            out_path = out_dir / f"{name_a}__{name_b}__{seed}.npz"
            if out_path.exists():
                skipped += 1
                continue
            jobs.append((name_a, name_b, seed, out_path))
    print(f"{len(jobs)} games to play ({skipped} already in {out_dir})", flush=True)
    if not jobs:
        return

    from multiprocessing import Pool

    t0 = time.time()
    total_shots = 0
    with Pool(args.workers, initializer=_worker_init, maxtasksperchild=4) as pool:
        for k, r in enumerate(pool.imap_unordered(_play_harvest, jobs), 1):
            total_shots += r["n_shots"]
            err = " ⚠" if "ERROR" in r["statuses"] else ""
            print(
                f"[{k}/{len(jobs)}] {r['matchup']} seed={r['seed']}: "
                f"{r['n_shots']} shots (pos {r['pos_rate']:.0%}, {r['n_dropped']} dropped) "
                f"({r['seconds']}s){err}",
                flush=True,
            )
    print(f"done in {(time.time() - t0) / 60:.1f} min, {total_shots} labeled shots total")


if __name__ == "__main__":
    main()
