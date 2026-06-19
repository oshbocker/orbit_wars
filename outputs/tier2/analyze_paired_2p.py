"""Summarize outputs/arena/paired_2p.csv (works on partial/streaming data)."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

CSV = Path("/home/aeschbacher/git/orbit_wars/outputs/arena/paired_2p.csv")


def main():
    agg = defaultdict(lambda: [0, 0, 0, 0.0, 0.0])  # n, win, tie, rew, margin
    opps = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            k = (r["opp"], r["variant"])
            if r["opp"] not in opps:
                opps.append(r["opp"])
            a = agg[k]
            a[0] += 1
            a[1] += int(r["win"]); a[2] += int(r["tie"])
            a[3] += float(r["reward"]); a[4] += float(r["margin"])
    print(f"{'opp':16s} {'var':7s}  n    win%   tie%  meanRew  meanMargin")
    for opp in opps:
        for vl in ("single", "ens"):
            a = agg.get((opp, vl))
            if not a:
                continue
            n = a[0]
            print(f"{opp:16s} {vl:7s} {n:3d}  {100*a[1]/n:5.1f}  {100*a[2]/n:5.1f}  "
                  f"{a[3]/n:+7.2f}  {a[4]/n:+9.1f}")
    print("\n=== ENS - SINGLE (paired) ===")
    for opp in opps:
        s = agg.get((opp, "single")); e = agg.get((opp, "ens"))
        if not s or not e:
            continue
        print(f"{opp:16s}  n={min(s[0],e[0]):3d}  win {100*(e[1]/e[0]-s[1]/s[0]):+5.1f}  "
              f"rew {e[3]/e[0]-s[3]/s[0]:+.3f}  margin {e[4]/e[0]-s[4]/s[0]:+8.1f}")


if __name__ == "__main__":
    main()
