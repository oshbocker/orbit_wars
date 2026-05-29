"""Plot hyperparameter experiment results."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def parse_train_log(log_path: Path) -> list[dict]:
    """Parse a train.log file into a list of update metrics."""
    results = []
    pattern = re.compile(
        r"update=\s*(\d+)\s+"
        r"reward=([\+\-\d.]+)\s+"
        r"eps=(\d+)\s+"
        r"samples=(\d+)\s+"
        r"loss=([\+\-\d.]+)\s+"
        r"ploss=([\+\-\d.]+)\s+"
        r"vloss=([\+\-\d.]+)\s+"
        r"ent=([\+\-\d.]+)\s+"
        r"dt=([\d.]+)s"
    )
    if not log_path.exists():
        return results
    for line in log_path.read_text().splitlines():
        m = pattern.search(line)
        if m:
            results.append({
                "update": int(m.group(1)),
                "reward": float(m.group(2)),
                "episodes": int(m.group(3)),
                "samples": int(m.group(4)),
                "loss": float(m.group(5)),
                "policy_loss": float(m.group(6)),
                "value_loss": float(m.group(7)),
                "entropy": float(m.group(8)),
                "dt": float(m.group(9)),
            })
    return results


def parse_eval_from_log(log_path: Path) -> dict[str, list[tuple[int, float]]]:
    """Parse eval win rates from train.log."""
    results: dict[str, list[tuple[int, float]]] = {}
    current_update = 0

    for line in log_path.read_text().splitlines():
        # Track current update
        um = re.search(r"update=\s*(\d+)", line)
        if um:
            current_update = int(um.group(1))
        # Parse eval results
        em = re.search(r"vs (\w+): W=(\d+)%", line)
        if em:
            opp = em.group(1)
            wr = float(em.group(2)) / 100.0
            results.setdefault(opp, []).append((current_update, wr))
    return results


def smooth(values: list[float], window: int = 10) -> list[float]:
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i + 1]) / (i - start + 1))
    return result


def load_all_experiments() -> dict[str, list[dict]]:
    """Find and parse all experiment logs."""
    logs_dir = EXPERIMENTS_DIR / "logs"
    results = {}
    if not logs_dir.exists():
        return results
    for log_dir in sorted(logs_dir.iterdir()):
        if not log_dir.is_dir() or not log_dir.name.startswith("hparam_"):
            continue
        name = log_dir.name.replace("hparam_", "")
        log_path = log_dir / "train.log"
        data = parse_train_log(log_path)
        if data:
            results[name] = data
    return results


def plot_metric(
    all_data: dict[str, list[dict]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
    smooth_window: int = 10,
    figsize: tuple = (12, 6),
):
    """Plot a single metric across all experiments."""
    fig, ax = plt.subplots(figsize=figsize)

    for name, data in sorted(all_data.items()):
        updates = [d["update"] for d in data]
        values = [d[metric] for d in data]
        smoothed = smooth(values, smooth_window)
        ax.plot(updates, smoothed, label=name, alpha=0.8, linewidth=1.5)

    ax.set_xlabel("Update")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_eval_winrates(all_data: dict[str, list[dict]], output_path: Path):
    """Plot eval win rates from logs."""
    logs_dir = EXPERIMENTS_DIR / "logs"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name in sorted(all_data.keys()):
        log_path = logs_dir / f"hparam_{name}" / "train.log"
        evals = parse_eval_from_log(log_path)

        for opp_idx, opp in enumerate(["apex", "random"]):
            if opp in evals:
                updates = [u for u, _ in evals[opp]]
                wrs = [w for _, w in evals[opp]]
                axes[opp_idx].plot(updates, wrs, "o-", label=name, markersize=4, alpha=0.8)

    for idx, opp in enumerate(["apex", "random"]):
        axes[idx].set_xlabel("Update")
        axes[idx].set_ylabel("Win Rate")
        axes[idx].set_title(f"Win Rate vs {opp.capitalize()}")
        axes[idx].legend(loc="best", fontsize=7)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_update_time(all_data: dict[str, list[dict]], output_path: Path):
    """Plot update time comparison as bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = []
    avg_times = []
    for name in sorted(all_data.keys()):
        data = all_data[name]
        times = [d["dt"] for d in data]
        names.append(name)
        avg_times.append(sum(times) / len(times))

    bars = ax.barh(names, avg_times, color="steelblue", alpha=0.8)
    ax.set_xlabel("Average Update Time (seconds)")
    ax.set_title("Training Speed Comparison")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for bar, val in zip(bars, avg_times):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}s", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def generate_summary_table(all_data: dict[str, list[dict]]) -> str:
    """Generate a markdown summary table."""
    logs_dir = EXPERIMENTS_DIR / "logs"
    rows = []

    for name in sorted(all_data.keys()):
        data = all_data[name]
        if not data:
            continue

        recent = data[-50:] if len(data) >= 50 else data
        all_rewards = [d["reward"] for d in data if d["episodes"] > 0]

        avg_reward = sum(d["reward"] for d in recent) / len(recent)
        final_loss = recent[-1]["loss"]
        avg_entropy = sum(d["entropy"] for d in recent) / len(recent)
        avg_dt = sum(d["dt"] for d in data) / len(data)
        total_time = sum(d["dt"] for d in data)

        # Parse eval
        log_path = logs_dir / f"hparam_{name}" / "train.log"
        evals = parse_eval_from_log(log_path)
        apex_wr = evals.get("apex", [(0, 0.0)])[-1][1] if "apex" in evals else None
        random_wr = evals.get("random", [(0, 0.0)])[-1][1] if "random" in evals else None

        rows.append({
            "name": name,
            "avg_reward": avg_reward,
            "final_loss": final_loss,
            "avg_entropy": avg_entropy,
            "avg_dt": avg_dt,
            "total_time": total_time,
            "apex_wr": apex_wr,
            "random_wr": random_wr,
            "n_updates": len(data),
        })

    # Build table
    lines = []
    lines.append("| Experiment | Updates | Avg Reward (last 50) | Final Loss | Avg Entropy | "
                 "vs Apex | vs Random | Avg dt | Total Time |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for r in rows:
        apex = f"{r['apex_wr']:.0%}" if r["apex_wr"] is not None else "—"
        rand = f"{r['random_wr']:.0%}" if r["random_wr"] is not None else "—"
        lines.append(
            f"| {r['name']} | {r['n_updates']} | {r['avg_reward']:+.3f} | "
            f"{r['final_loss']:.4f} | {r['avg_entropy']:.2f} | "
            f"{apex} | {rand} | {r['avg_dt']:.1f}s | {r['total_time']:.0f}s |"
        )

    return "\n".join(lines)


def main():
    plots_dir = EXPERIMENTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment data...")
    all_data = load_all_experiments()

    if not all_data:
        print("No experiment data found!")
        return

    print(f"Found {len(all_data)} experiments: {list(all_data.keys())}")
    print("\nGenerating plots...")

    # Reward over time
    plot_metric(all_data, "reward", "Episode Reward Over Training",
                "Reward (smoothed)", plots_dir / "reward.png", smooth_window=15)

    # Loss over time
    plot_metric(all_data, "loss", "Total Loss Over Training",
                "Loss (smoothed)", plots_dir / "loss.png", smooth_window=10)

    # Policy loss
    plot_metric(all_data, "policy_loss", "Policy Loss Over Training",
                "Policy Loss (smoothed)", plots_dir / "policy_loss.png", smooth_window=10)

    # Value loss
    plot_metric(all_data, "value_loss", "Value Loss Over Training",
                "Value Loss (smoothed)", plots_dir / "value_loss.png", smooth_window=10)

    # Entropy
    plot_metric(all_data, "entropy", "Entropy Over Training",
                "Entropy (smoothed)", plots_dir / "entropy.png", smooth_window=10)

    # Eval win rates
    plot_eval_winrates(all_data, plots_dir / "eval_winrates.png")

    # Update time comparison
    plot_update_time(all_data, plots_dir / "update_time.png")

    # Summary table
    print("\n" + generate_summary_table(all_data))

    # Save summary
    summary_path = EXPERIMENTS_DIR / "summary_table.md"
    summary_path.write_text(generate_summary_table(all_data))
    print(f"\nSummary table saved to {summary_path}")


if __name__ == "__main__":
    main()
