"""Phase 2.2c mechanism probe: does the arrival-resolving horizon create q-signal?

Re-runs the 2026-06-11 floss/horizon diagnostic that found the root cause of the
iter-25 plateau (search horizon-blind: 16.8% of candidates resolve in depth-12,
q-spread 0.0 at p90, pi' = prior on ~5/6 decisions), with exit.arrival_horizon
OFF vs ON. Must show, before any Colab run is worth launching:

  (a) % of candidates whose fleet arrives within the decision depth ~= 100% (ON)
  (b) per-decision candidate q-spread > 0 for most decisions (ON)
  (c) mean fraction-target entropy of the Gumbel distillation target clearly
      below ln4 ~= 1.386 (ON)

Setup mirrors collection: champion ckpt (v2_exit_producer256_a100/ckpt_000025),
one fast_env collection game vs ow_proto, ~30 evenly-spaced StepRecords.

Run: PYTHONPATH=. uv run python scripts/diag_arrival_horizon.py
"""

from __future__ import annotations

import argparse
import dataclasses
import math

import numpy as np
import torch

from src.simulator import evaluate_state
from v2.config import load_v2_config
from v2.exit_train import play_single_game
from v2.model import OrbitNet
from v2.search import (
    _decision_depth,
    _enumerate_candidates,
    _simulate_descriptor,
    search_improve_planet,
)

LN4 = math.log(4.0)


def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-12]
    return float(-(p * np.log(p)).sum())


def probe(records, env_cfg, exit_cfg, label: str) -> None:
    resolved, spreads, fents = [], [], []
    decisions = 0
    depths = []
    for rec in records:
        feats = rec.features
        P = env_cfg.max_planets
        for i in range(P):
            if not feats.own_mask[i]:
                continue
            src, descs = _enumerate_candidates(rec.game_state, feats, i, env_cfg, exit_cfg)
            fr = [d for d in descs if d[0] == "frac"]
            if not fr:
                continue
            decisions += 1
            depth = _decision_depth(descs, exit_cfg)
            depths.append(depth)
            resolved.extend(1.0 if d[4] <= depth else 0.0 for d in fr)
            # Eager q over ALL candidates at the decision depth (legacy-style),
            # passive sim (no opponent flags set) — the signal Gumbel ranks on.
            src_id = src.id if src is not None else -1
            qs = [
                evaluate_state(
                    _simulate_descriptor(d, rec.sim_state, src_id, depth, None, None, 1),
                    rec.player,
                )
                for d in descs
            ]
            spreads.append(float(max(qs) - min(qs)))
            # Gumbel distillation target -> fraction entropy at the top non-hold target.
            tp, fp = search_improve_planet(
                state=rec.game_state,
                features=feats,
                sim_state=rec.sim_state,
                player=rec.player,
                source_slot=i,
                env_cfg=env_cfg,
                exit_cfg=exit_cfg,
                prior_target=rec.prior_target[i],
                prior_frac=rec.prior_frac[i],
                rng_seed=int(rec.step) * P + i,
            )
            cand_js = sorted({d[1] for d in fr})
            jstar = max(cand_js, key=lambda j: float(tp[j + 1]))
            fents.append(_entropy(fp[jstar]))

    resolved_pct = 100.0 * float(np.mean(resolved)) if resolved else 0.0
    sp = np.array(spreads)
    print(f"\n== {label} ==")
    print(
        f"decisions={decisions}  depth p50/p90/max = "
        f"{np.percentile(depths, 50):.0f}/{np.percentile(depths, 90):.0f}/{max(depths)}"
    )
    print(f"(a) candidates resolving within depth: {resolved_pct:.1f}%")
    print(
        f"(b) per-decision q-spread: p50={np.percentile(sp, 50):.3f} "
        f"p90={np.percentile(sp, 90):.3f}; spread>1e-6 in "
        f"{100.0 * float((sp > 1e-6).mean()):.1f}% of decisions"
    )
    print(
        f"(c) fraction-target entropy (top target): mean={np.mean(fents):.3f} "
        f"(ln4={LN4:.3f}); p10={np.percentile(fents, 10):.3f} "
        f"p50={np.percentile(fents, 50):.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_exit_producer256.yaml")
    ap.add_argument(
        "--checkpoint", default="outputs/checkpoints/v2_exit_producer256_a100/ckpt_000025.pt"
    )
    ap.add_argument("--opponent", default="ow_proto")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--records", type=int, default=30)
    args = ap.parse_args()

    cfg = load_v2_config(args.config)
    cfg.exit.opponent = args.opponent
    device = torch.device("cpu")
    model = OrbitNet(cfg.model)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"collecting 1 game vs {args.opponent} (fast_env, seed={args.seed}) ...")
    records, outcome = play_single_game(model, cfg, device, args.seed)
    print(f"game done: {len(records)} records, outcome={outcome:+.0f}")
    idx = np.linspace(0, len(records) - 1, min(args.records, len(records))).astype(int)
    sample = [records[i] for i in sorted(set(idx.tolist()))]
    print(f"probing {len(sample)} records")

    probe(sample, cfg.env, cfg.exit, "arrival_horizon OFF (fixed depth-12 baseline)")
    on_cfg = dataclasses.replace(cfg.exit, arrival_horizon=True)
    probe(
        sample,
        cfg.env,
        on_cfg,
        f"arrival_horizon ON (margin={on_cfg.arrival_settle_margin}, "
        f"cap={on_cfg.arrival_horizon_cap})",
    )


if __name__ == "__main__":
    main()
