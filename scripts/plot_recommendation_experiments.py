"""Plot the recommendation experiments (exp2_* runs) and build a summary table."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
LOG_DIR = EXPERIMENTS_DIR / "logs"
PLOTS_DIR = EXPERIMENTS_DIR / "plots2"

PREFIX = "exp2_"
SKIP = {"exp2_prime"}

# Stable display order / colors
ORDER = ["baseline", "r1_ent_anneal", "r2_high_gamma", "r3_strong_value",
         "r4_fast_selfplay", "r5_no_prod_bonus"]

TRAIN_RE = re.compile(
    r"update=\s*(\d+)\s+reward=([\+\-\d.]+)\s+eps=(\d+)\s+samples=(\d+)\s+"
    r"loss=([\+\-\d.]+)\s+ploss=([\+\-\d.]+)\s+vloss=([\+\-\d.]+)\s+"
    r"ent=([\+\-\d.]+)\s+dt=([\d.]+)s")
EVAL_RE = re.compile(r"vs (\w+): W=([\d.]+)%")
UPD_RE = re.compile(r"update=\s*(\d+)")


def parse_train(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    for line in p.read_text().splitlines():
        m = TRAIN_RE.search(line)
        if m:
            out.append({"update": int(m.group(1)), "reward": float(m.group(2)),
                        "episodes": int(m.group(3)), "loss": float(m.group(5)),
                        "policy_loss": float(m.group(6)), "value_loss": float(m.group(7)),
                        "entropy": float(m.group(8)), "dt": float(m.group(9))})
    return out


def parse_eval(p: Path) -> dict[str, list[tuple[int, float]]]:
    res: dict[str, list[tuple[int, float]]] = {}
    cur = 0
    if not p.exists():
        return res
    for line in p.read_text().splitlines():
        um = UPD_RE.search(line)
        if um:
            cur = int(um.group(1))
        em = EVAL_RE.search(line)
        if em:
            res.setdefault(em.group(1), []).append((cur, float(em.group(2)) / 100.0))
    return res


def smooth(v, w=10):
    if len(v) < w:
        return v
    return [sum(v[max(0, i - w + 1):i + 1]) / (i - max(0, i - w + 1) + 1) for i in range(len(v))]


def load_all() -> dict[str, list[dict]]:
    res = {}
    for d in sorted(LOG_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith(PREFIX) or d.name in SKIP:
            continue
        data = parse_train(d / "train.log")
        if data:
            res[d.name.replace(PREFIX, "")] = data
    return res


def ordered(keys):
    return sorted(keys, key=lambda k: ORDER.index(k) if k in ORDER else 99)


def plot_metric(all_data, metric, title, ylabel, out, w=10):
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in ordered(all_data):
        data = all_data[name]
        ax.plot([d["update"] for d in data], smooth([d[metric] for d in data], w),
                label=name, linewidth=1.8, alpha=0.85)
    ax.set_xlabel("PPO update"); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  saved {out}")


def plot_eval(all_data, out):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name in ordered(all_data):
        ev = parse_eval(LOG_DIR / f"{PREFIX}{name}" / "train.log")
        for k, opp in enumerate(["apex", "random"]):
            if opp in ev:
                axes[k].plot([u for u, _ in ev[opp]], [w for _, w in ev[opp]],
                             "o-", label=name, markersize=6, alpha=0.85)
    for k, opp in enumerate(["apex", "random"]):
        axes[k].set_xlabel("PPO update"); axes[k].set_ylabel("Win rate")
        axes[k].set_title(f"Win rate vs {opp}"); axes[k].legend(fontsize=8)
        axes[k].grid(True, alpha=0.3); axes[k].set_ylim(-0.05, 1.05)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  saved {out}")


def summary_table(all_data) -> str:
    lines = ["| Experiment | Updates | Avg Reward (last 50) | Final Loss | Final Value Loss | "
             "Avg Entropy | vs Apex | vs Random | Avg dt |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in ordered(all_data):
        data = all_data[name]
        recent = data[-50:] if len(data) >= 50 else data
        avg_r = sum(d["reward"] for d in recent) / len(recent)
        avg_e = sum(d["entropy"] for d in recent) / len(recent)
        avg_dt = sum(d["dt"] for d in data) / len(data)
        ev = parse_eval(LOG_DIR / f"{PREFIX}{name}" / "train.log")
        apex = f"{ev['apex'][-1][1]:.0%}" if ev.get("apex") else "—"
        rand = f"{ev['random'][-1][1]:.0%}" if ev.get("random") else "—"
        lines.append(f"| {name} | {len(data)} | {avg_r:+.3f} | {recent[-1]['loss']:.4f} | "
                     f"{recent[-1]['value_loss']:.4f} | {avg_e:.2f} | {apex} | {rand} | {avg_dt:.1f}s |")
    return "\n".join(lines)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    all_data = load_all()
    if not all_data:
        print("No exp2_* data found.")
        return
    print(f"Found: {ordered(all_data)}")
    plot_metric(all_data, "reward", "Episode Reward", "reward (smoothed)", PLOTS_DIR / "reward.png", 15)
    plot_metric(all_data, "loss", "Total Loss", "loss", PLOTS_DIR / "loss.png")
    plot_metric(all_data, "value_loss", "Value Loss", "value loss", PLOTS_DIR / "value_loss.png")
    plot_metric(all_data, "policy_loss", "Policy Loss", "policy loss", PLOTS_DIR / "policy_loss.png")
    plot_metric(all_data, "entropy", "Policy Entropy", "entropy", PLOTS_DIR / "entropy.png")
    plot_eval(all_data, PLOTS_DIR / "eval_winrates.png")
    table = summary_table(all_data)
    print("\n" + table)
    (EXPERIMENTS_DIR / "summary_table2.md").write_text(table)
    print(f"\nSummary -> {EXPERIMENTS_DIR / 'summary_table2.md'}")


if __name__ == "__main__":
    main()
