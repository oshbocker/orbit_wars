"""Grid the 2P ENSEMBLE detector gate over (alpha, thr, min_obs).

Replays the runtime gate from per-opponent /tmp/calib2p_<opp>.npy. The ensemble
gate is ON when EITHER model's EMA clears (thr, min_obs). Reports, per opponent:
  - ensemble ON-fraction (any model)  vs
  - single base-model ON-fraction (model 0 only, == v5.4)
so the producer_v2 LIFT and the non-producer OFF-bias are both visible.
"""
from __future__ import annotations

import numpy as np

CLONES = ["producer", "producer_v2"]
OTHERS = ["tamrazov_1224", "ow_proto", "distance_1100", "enders_1000"]


def load(opp):
    return list(np.load(f"/tmp/calib2p_{opp}.npy", allow_pickle=True))


def on_fraction(games, alpha, thr, min_obs, models):
    """ON-fraction of measured turns; gate ON if ANY model in `models` clears."""
    on = tot = 0
    for game in games:
        fid = {0: 0.0, 1: 0.0}
        obs = {0: 0, 1: 0}
        for (_step, mp) in game:
            for m, p in mp.items():
                fid[m] = alpha * fid[m] + (1.0 - alpha) * p
                obs[m] += 1
            tot += 1
            if any(obs[m] >= min_obs and fid[m] >= thr for m in models):
                on += 1
    return on / max(tot, 1)


def main():
    data = {o: load(o) for o in CLONES + OTHERS}
    a, thr, mo = 0.9, 0.55, 8
    print(f"=== Shipped gate alpha={a} thr={thr} min_obs={mo}: ensemble vs single ===")
    print(f"{'opponent':16s}  single(m0)  ensemble(m0|m1)   lift")
    for o in CLONES + OTHERS:
        s = on_fraction(data[o], a, thr, mo, (0,))
        e = on_fraction(data[o], a, thr, mo, (0, 1))
        print(f"{o:16s}    {s:6.2f}        {e:6.2f}       {e - s:+.2f}")
    cmin = min(on_fraction(data[o], a, thr, mo, (0, 1)) for o in CLONES)
    omax = max(on_fraction(data[o], a, thr, mo, (0, 1)) for o in OTHERS)
    print(f"  ensemble clone_min={cmin:.2f}  other_max={omax:.2f}  sep={cmin - omax:.2f}")

    print("\n=== GRID (ensemble ON-fraction; flag * = clones>=.45 & others<=.25) ===")
    hdr = "  ".join(f"{o[:8]:>8}" for o in CLONES + OTHERS)
    print(f"alpha thr  min | {hdr} | clo_min oth_max  sep")
    rows = []
    for alpha in (0.8, 0.9, 0.95):
        for t in (0.45, 0.5, 0.55, 0.6, 0.65):
            for m in (5, 8, 12):
                fr = {o: on_fraction(data[o], alpha, t, m, (0, 1)) for o in data}
                cm = min(fr[o] for o in CLONES)
                om = max(fr[o] for o in OTHERS)
                rows.append((cm - om, alpha, t, m, fr, cm, om))
                r = "  ".join(f"{fr[o]:8.2f}" for o in CLONES + OTHERS)
                flag = " *" if (cm >= 0.45 and om <= 0.25) else ""
                print(f"{alpha:4.2f} {t:.2f} {m:3d} | {r} |  {cm:5.2f}  {om:5.2f} {cm - om:5.2f}{flag}")
    best = max(rows, key=lambda r: (r[0], r[5]))
    print(f"\nBEST sep: alpha={best[1]} thr={best[2]} min_obs={best[3]} "
          f"sep={best[0]:.2f} clone_min={best[5]:.2f} other_max={best[6]:.2f}")


if __name__ == "__main__":
    main()
