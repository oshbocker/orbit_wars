"""Aggregate a dose-surface arena CSV into per-dose win-rate / margin / steps
vs each opponent (matches arena.py's tie=0.5 convention).

Usage: uv run python scripts/dose_surface_agg.py outputs/arena/dose_surface_0620.csv
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

CSV = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/arena/dose_surface_0620.csv")

# label -> {opp -> [points(tie .5), n, net_reward_sum, steps_sum]}
acc: dict = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0, 0.0]))


def add(focus, opp, points, net_reward, steps):
    a = acc[focus][opp]
    a[0] += points
    a[1] += 1
    a[2] += net_reward
    a[3] += steps


with open(CSV) as f:
    rows = list(csv.DictReader(f))

for r in rows:
    a, b = r["agent_a"], r["agent_b"]
    steps = float(r.get("steps") or 0)
    ra, rb = float(r["reward_a"]), float(r["reward_b"])
    p_a = 1.0 if r["outcome"] == "a" else 0.0 if r["outcome"] == "b" else 0.5
    add(a, b, p_a, ra, steps)
    add(b, a, 1.0 - p_a, rb, steps)

labels = sorted(acc.keys())
opps = sorted({o for d in acc.values() for o in d})

print(f"\n{CSV} — {len(rows)} games\n")
hdr = f"{'FOCUS agent':<52}{'OPPONENT':<52}{'win%':>7}{'n':>6}{'margin':>9}{'steps':>8}"
print(hdr)
print("-" * len(hdr))
for focus in labels:
    for opp in opps:
        if opp == focus or acc[focus][opp][1] == 0:
            continue
        pts, n, netr, st = acc[focus][opp]
        print(
            f"{focus:<52}{opp:<52}{pts / n:>6.1%}{n:>6}"
            f"{netr / n:>+9.3f}{st / n:>8.0f}"
        )
    print()
