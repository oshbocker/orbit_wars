"""Compare the embed_dim=128 vs 256 ExIt arms on win-rate vs apex.

Reads each run's TensorBoard event file (outputs/logs/<run>/) for the
`eval/apex_win_rate` (and `eval/random_win_rate`) scalars, prints an aligned
table, and saves a comparison PNG. Works mid-run (reads whatever is logged so
far). Run: uv run python scripts/plot_embed_ab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RUNS = {"embed128": "v2_exit_embed128", "embed256": "v2_exit_embed256"}
TAGS = ["eval/apex_win_rate", "eval/random_win_rate"]


def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    acc.Reload()
    avail = set(acc.Tags().get("scalars", []))
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in TAGS:
        if tag in avail:
            out[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
    return out


def main() -> int:
    log_root = ROOT / "outputs" / "logs"
    data: dict[str, dict] = {}
    for arm, run in RUNS.items():
        d = log_root / run
        if not d.exists():
            print(f"[{arm}] no logs yet at {d}")
            continue
        data[arm] = load_scalars(d)

    if not data:
        print("No runs found yet. Start with: uv run python scripts/run_embed_ab.py")
        return 0

    # Aligned win-rate-vs-apex table.
    print("\n=== eval/apex_win_rate (higher = better) ===")
    steps = sorted({s for arm in data for s, _ in data[arm].get("eval/apex_win_rate", [])})
    print(f"{'iter':>6} | {'embed128':>9} | {'embed256':>9} | {'Δ(256-128)':>10}")
    print("-" * 44)
    best = {arm: 0.0 for arm in RUNS}
    for s in steps:
        vals = {}
        for arm in RUNS:
            m = dict(data.get(arm, {}).get("eval/apex_win_rate", []))
            if s in m:
                vals[arm] = m[s]
                best[arm] = max(best[arm], m[s])
        v128 = vals.get("embed128")
        v256 = vals.get("embed256")
        c128 = f"{v128:.0%}" if v128 is not None else "  -"
        c256 = f"{v256:.0%}" if v256 is not None else "  -"
        delta = f"{(v256 - v128):+.0%}" if (v128 is not None and v256 is not None) else "  -"
        print(f"{s:>6} | {c128:>9} | {c256:>9} | {delta:>10}")

    print("\n=== best apex win-rate ===")
    for arm in RUNS:
        print(f"  {arm}: {best[arm]:.0%}")
    if best["embed128"] and best["embed256"]:
        gap = best["embed256"] - best["embed128"]
        verdict = (
            "256 wins — capacity helps with search targets"
            if gap > 0.02
            else "128 wins"
            if gap < -0.02
            else "≈ tie (no clear capacity benefit)"
        )
        print(f"  Δ best (256-128): {gap:+.0%}  → {verdict}")

    # Plot (best-effort; headless OK).
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        for arm in RUNS:
            pts = data.get(arm, {}).get("eval/apex_win_rate", [])
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, [y * 100 for y in ys], marker="o", label=arm)
        ax.set_xlabel("ExIt iteration")
        ax.set_ylabel("win-rate vs apex (%)")
        ax.set_title("ExIt capacity A/B: embed_dim 128 vs 256")
        ax.grid(True, alpha=0.3)
        ax.legend()
        out = ROOT / "experiments" / "embed_ab_winrate.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        print(f"\nplot saved -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
