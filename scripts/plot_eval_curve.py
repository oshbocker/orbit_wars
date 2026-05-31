"""Plot the eval win-rate curve for a training run.

Two data sources (use whichever is available):
  1. --log PATH       parse a train.log (pairs each 'vs apex/random' eval line
                      with the most recent 'update=' line). Use this once a
                      COMPLETE train.log is synced from Drive.
  2. --points         plot from hardcoded console-observed points (the reliable
                      source for v2_ppo_a100, whose Drive logs are incomplete
                      due to the resume-across-sessions pattern).

Usage:
    uv run python scripts/plot_eval_curve.py --points --out outputs/logs/v2_ppo_a100_eval.png
    uv run python scripts/plot_eval_curve.py --log outputs/logs/v2_ppo_a100/train.log --out x.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Console-observed eval points for v2_ppo_a100 (update -> {apex, random} win rate).
# Recorded from the live Colab console; the Drive logs only captured the BC eval.
V2_PPO_A100 = {
    0:    {"apex": 0.05, "random": 1.00},   # BC clone
    750:  {"apex": 0.20, "random": 1.00},
    1000: {"apex": 0.25, "random": 1.00},
    1250: {"apex": 0.35, "random": 0.95},
}


def parse_eval_csv(path: Path) -> dict[int, dict[str, float]]:
    """Parse eval_history.csv (the resume-safe, Drive-synced eval log)."""
    import csv
    out: dict[int, dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            u = int(row["update"])
            out.setdefault(u, {})[row["opponent"]] = float(row["win_rate"])
    return out


def parse_log(path: Path) -> dict[int, dict[str, float]]:
    """Pair each eval line with the most recent 'update=' value."""
    out: dict[int, dict[str, float]] = {}
    cur = 0
    upd_re = re.compile(r"update=\s*(\d+)")
    eval_re = re.compile(r"vs (\w+): W=(\d+)%")
    for line in path.read_text().splitlines():
        m = upd_re.search(line)
        if m:
            cur = int(m.group(1))
        e = eval_re.search(line)
        if e:
            opp, wr = e.group(1), int(e.group(2)) / 100.0
            out.setdefault(cur, {})[opp] = wr
    return out


def plot(data: dict[int, dict[str, float]], out: Path, title: str) -> None:
    updates = sorted(data)
    apex = [data[u].get("apex") for u in updates]
    rand = [data[u].get("random") for u in updates]

    fig, ax = plt.subplots(figsize=(10, 6))
    # apex line (the one that matters)
    ax_u = [u for u, v in zip(updates, apex) if v is not None]
    ax_v = [v for v in apex if v is not None]
    ax.plot(ax_u, ax_v, "o-", color="crimson", lw=2.5, ms=9, label="vs apex")
    for u, v in zip(ax_u, ax_v):
        ax.annotate(f"{v:.0%}", (u, v), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=10, color="crimson")
    # random line
    rx_u = [u for u, v in zip(updates, rand) if v is not None]
    rx_v = [v for v in rand if v is not None]
    ax.plot(rx_u, rx_v, "s--", color="steelblue", lw=1.8, ms=7, label="vs random")

    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(updates[-1], 0.5, " 50% (parity w/ apex)", va="bottom", ha="right",
            fontsize=8, color="gray")
    ax.set_xlabel("PPO update")
    ax.set_ylabel("win rate (n=20/eval)")
    ax.set_title(title)
    ax.set_ylim(-0.03, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                    help="eval_history.csv (preferred; resume-safe)")
    ap.add_argument("--log", type=str, default=None, help="train.log to parse")
    ap.add_argument("--points", action="store_true",
                    help="use hardcoded v2_ppo_a100 console points")
    ap.add_argument("--out", type=str, default="outputs/logs/eval_curve.png")
    ap.add_argument("--title", type=str, default="v2_ppo_a100 — eval win rate vs apex")
    args = ap.parse_args()

    if args.csv:
        data = parse_eval_csv(Path(args.csv))
    elif args.log:
        data = parse_log(Path(args.log))
        # If the log only has BC (update 0), supplement with console points.
        if set(data) <= {0}:
            print("Log only contains BC eval; merging console-observed points.")
            data = {**V2_PPO_A100, **data}
    else:
        data = V2_PPO_A100

    if not data:
        print("No eval data found.")
        return
    print("Eval points:", {u: data[u] for u in sorted(data)})
    plot(data, Path(args.out), args.title)


if __name__ == "__main__":
    main()
