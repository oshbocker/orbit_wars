"""Smoke test for the v4-complete parallel rollout collector.

Exercises the risky new code in v2/parallel.py:
  * full v4 batch serialization (pair_features + offset-merged shot labels),
  * PopArt value_norm broadcast (read-only in workers),
  * distributed PFSP (pool snapshot broadcast + per-worker win/game delta merge).

Asserts batch shapes/fields, shot-index bounds, finite GAE, and that a freshly
snapshotted opponent propagates to workers. Run:
    uv run python scripts/test_parallel_v4_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.opponents import build_opponent
from v2.config import load_v2_config
from v2.model import OrbitNet
from v2.parallel import ParallelRolloutCollector
from v2.ppo import ValueNorm
from v2.train import V2PFSPScheduler


def main() -> int:
    cfg = load_v2_config("configs/v4_ceiling.yaml")
    cfg.device = "cpu"
    cfg.ppo.num_workers = 2
    cfg.ppo.rollout_steps = 24
    cfg.ppo.num_envs = 1
    cfg.four_player_prob = 0.0          # keep 2p for a fast smoke
    cfg.pfsp_snapshot_every = 1         # snapshot immediately so broadcast is exercised
    cfg.pfsp_pool_size = 3
    cfg.imitation.enabled = False
    cfg.eval.eval_every = 0

    assert cfg.ppo.popart and not cfg.ppo.value_symlog, "v4 should use PopArt"
    assert cfg.env.use_pair_features, "v4 should use pair features"
    assert cfg.model.shot_success_head and cfg.ppo.shot_aux_coef > 0.0, "v4 should train shot head"

    device = torch.device("cpu")
    model = OrbitNet(cfg.model).to(device)
    value_norm = ValueNorm(cfg.ppo.popart_beta)
    # Seed value_norm with non-trivial stats so denormalization is actually exercised.
    value_norm.update(torch.randn(256) * 3.0 + 1.0)
    scheduler = V2PFSPScheduler(cfg, build_opponent(cfg.opponent), device)

    P = cfg.env.max_planets
    fails: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            fails.append(msg)

    print(f"spawning {cfg.ppo.num_workers} workers (rollout_steps={cfg.ppo.rollout_steps})...")
    collector = ParallelRolloutCollector(cfg, cfg.ppo.num_workers)
    try:
        collector.sync(model, value_norm, scheduler)

        # ── Round 1: collect + validate the v4 batch ─────────────────────────
        batch, stats, deltas = collector.collect(update=1)
        N = batch.planet_features.shape[0]
        print(f"round1: N={N} rows  eps={stats['episodes_finished']:.0f}  "
              f"pfsp_deltas={ {k: [round(x,1) for x in v] for k,v in deltas.items()} }")

        check(N == cfg.ppo.num_workers * cfg.ppo.rollout_steps,
              f"row count {N} != workers*rollout_steps {cfg.ppo.num_workers*cfg.ppo.rollout_steps}")
        check(batch.pair_features is not None, "pair_features missing from parallel batch")
        if batch.pair_features is not None:
            check(tuple(batch.pair_features.shape) == (N, P, P, cfg.env.pair_feat_dim),
                  f"pair_features shape {tuple(batch.pair_features.shape)} wrong")
        check(torch.isfinite(batch.returns).all().item(), "non-finite returns")
        check(torch.isfinite(batch.advantages).all().item(), "non-finite advantages")
        check(batch.planet_features.shape == (N, P, cfg.model.planet_feat_dim),
              "planet_features shape wrong")

        # Shot labels present, offset-merged, and in-range.
        check(batch.shot_idx is not None, "shot_idx missing (shot head enabled)")
        if batch.shot_idx is not None:
            mn, mx = int(batch.shot_idx.min()), int(batch.shot_idx.max())
            check(0 <= mn and mx < N, f"shot_idx out of range [{mn},{mx}] vs N={N}")
            check(batch.shot_idx.shape == batch.shot_src.shape == batch.shot_tgt.shape
                  == batch.shot_label.shape, "shot_* length mismatch")
            check(set(batch.shot_label.unique().tolist()) <= {0.0, 1.0},
                  "shot_label not binary")
            print(f"  shot labels: {batch.shot_idx.shape[0]} (max_idx={mx}, "
                  f"pos_rate={batch.shot_label.mean():.2f})")

        # PFSP delta games should roughly match episodes finished this round.
        games = sum(d[1] for d in deltas.values())
        check(abs(games - stats["episodes_finished"]) < 1e-6,
              f"pfsp games {games} != episodes_finished {stats['episodes_finished']}")

        # ── Snapshot + broadcast, then round 2 ───────────────────────────────
        scheduler.apply_deltas(deltas)
        scheduler.set_update(1)
        scheduler.maybe_snapshot(model)
        check(len(scheduler.pool) == 2, f"pool should have grown to 2, got {len(scheduler.pool)}")
        print(f"round2: pool={[e['name'] for e in scheduler.pool]}")

        collector.sync(model, value_norm, scheduler)  # broadcasts the new snapshot weights
        batch2, stats2, deltas2 = collector.collect(update=2)
        check(batch2.planet_features.shape[0] == N, "round2 row count changed")
        check(batch2.pair_features is not None, "round2 pair_features missing")
        print(f"round2: N={batch2.planet_features.shape[0]}  "
              f"pfsp_deltas={ {k: [round(x,1) for x in v] for k,v in deltas2.items()} }")
        # Opponent names seen must be a subset of the (broadcast) pool names.
        check(set(deltas2) <= {e["name"] for e in scheduler.pool},
              f"worker reported unknown opponent: {set(deltas2)}")
    finally:
        collector.shutdown()

    print(f"\nfailures: {len(fails)}")
    for f in fails:
        print("  FAIL:", f)
    print("OK" if not fails else "SMOKE TEST FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
