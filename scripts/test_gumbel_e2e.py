"""End-to-end smoke test: collect -> search -> distill with the new flags ON,
both sequential and parallel workers. Asserts dists sum to 1 and losses finite.

Covers: Build 1 (gumbel_search) and Build 1+2 (gumbel_search + net_opponent).

Run: uv run python scripts/test_gumbel_e2e.py
"""

from __future__ import annotations

import numpy as np
import torch

from v2.config import load_v2_config
from v2.exit_train import collect_games, search_improve, train_epoch
from v2.model import OrbitNet


def _tiny_cfg(**exit_overrides):
    cfg = load_v2_config("configs/v2_exit.yaml")
    cfg.imitation.enabled = False
    cfg.exit.iterations = 1
    cfg.exit.games_per_iter = 2
    cfg.exit.search_depth = 4
    cfg.exit.search_candidates = 4
    cfg.exit.collect_fast_env = True
    cfg.exit.collect_workers = 0
    cfg.exit.search_workers = 0
    cfg.exit.gumbel_sims = 8
    cfg.exit.gumbel_top_m = 4
    for k, v in exit_overrides.items():
        setattr(cfg.exit, k, v)
    return cfg


def _check_samples(samples, label):
    assert samples, f"{label}: no samples"
    for s in samples:
        # target rows for owned planets sum to 1
        for i in range(s.target_probs.shape[0]):
            if s.own_mask[i]:
                ssum = s.target_probs[i].sum()
                assert abs(ssum - 1.0) < 1e-4, f"{label}: target row {i} sum {ssum}"
        assert np.isfinite(s.target_probs).all(), f"{label}: non-finite target"
        assert np.isfinite(s.frac_probs).all(), f"{label}: non-finite frac"
    print(f"  [OK] {label}: {len(samples)} samples, dists normalized + finite.")


def _run(cfg, model, device, label):
    records, _outcomes, _wr = collect_games(model, cfg, device, cfg.exit.games_per_iter, seed=4242)
    assert records, f"{label}: no records collected"
    # sequential search
    cfg.exit.search_workers = 0
    seq = search_improve(records, cfg, model)
    _check_samples(seq, f"{label} / search seq")
    # parallel search
    cfg.exit.search_workers = 2
    par = search_improve(records, cfg, model)
    _check_samples(par, f"{label} / search parallel")
    cfg.exit.search_workers = 0
    # distill: losses finite
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    m = train_epoch(model, opt, seq, cfg, device)
    model.eval()
    assert all(np.isfinite(v) for v in m.values()), f"{label}: non-finite loss {m}"
    print(f"  [OK] {label} / distill: loss={m['loss']:.4f} (finite).")


def main() -> None:
    torch.set_num_threads(1)
    device = torch.device("cpu")

    print("== Build 1: gumbel_search ==")
    cfg = _tiny_cfg(gumbel_search=True, net_opponent=False)
    model = OrbitNet(cfg.model).to(device).eval()
    _run(cfg, model, device, "gumbel")

    print("== Build 1+2: gumbel_search + net_opponent ==")
    cfg = _tiny_cfg(gumbel_search=True, net_opponent=True, net_opponent_every=2)
    model = OrbitNet(cfg.model).to(device).eval()
    _run(cfg, model, device, "gumbel+netopp")

    print("\nALL E2E SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
