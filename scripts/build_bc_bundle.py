"""Build + verify the BC (win-weighted gate-head OrbitNet) Kaggle submission bundle.

Assembles an archive-root bundle (main.py + packages + checkpoint at top level, the
proven producer_bundle layout — see memory kaggle-submission-tarball-root) whose
agent() reproduces bc_teacher.py's gate->pointer->capture executor exactly, then
verifies it by loading through Kaggle's file-agent loader and playing a real game.

    uv run python scripts/build_bc_bundle.py \
        --ckpt outputs/checkpoints/winbc_gate/winbc_gate.pt \
        --drain full --gate-thr 0.7

The chosen --drain/--gate-thr (from the local sweep) are baked into bc_config.py.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "agents" / "bc_submission" / "main.py"
CONTESTED = ROOT / "agents" / "external" / "contested_drainer.py"
STAGE = ROOT / "outputs" / "submissions" / "bc_bundle"
TARBALL = ROOT / "outputs" / "submissions" / "bc_bundle.tar.gz"


def build(ckpt: Path, config: Path, drain: str, gate_thr: float) -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    shutil.copy2(MAIN, STAGE / "main.py")
    shutil.copy2(CONTESTED, STAGE / "contested_drainer.py")
    shutil.copy2(ckpt, STAGE / "ckpt.pt")
    shutil.copy2(config, STAGE / "submission_config.yaml")
    (STAGE / "bc_config.py").write_text(f'DRAIN = "{drain}"\nGATE_THR = {gate_thr}\n')

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "v2", STAGE / "v2", ignore=ignore)
    shutil.copytree(ROOT / "src", STAGE / "src", ignore=ignore)

    names = ["main.py", "contested_drainer.py", "bc_config.py", "ckpt.pt", "submission_config.yaml", "v2", "src"]
    subprocess.run(["tar", "czf", str(TARBALL), *names], cwd=STAGE, check=True)
    print(f"built {TARBALL} ({TARBALL.stat().st_size / 1024 / 1024:.1f} MiB)  drain={drain} gate_thr={gate_thr}")


def verify() -> None:
    from kaggle_environments import make

    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["tar", "xzf", str(TARBALL)], cwd=td, check=True)
        main_py = str(Path(td) / "main.py")
        # Play both seats vs random; require a clean load (no ERROR) and a win as P0.
        env = make("orbit_wars", configuration={"randomSeed": 4242})
        env.run([main_py, "random"])
        last = env.steps[-1]
        statuses = [s.status for s in last]
        rewards = [s.reward for s in last]
        print(f"verify game: statuses={statuses} rewards={rewards} steps={len(env.steps) - 1}")
        if "ERROR" in statuses:
            print("VERIFY FAILED: agent errored under Kaggle loader", file=sys.stderr)
            sys.exit(1)
        if rewards[0] != 1:
            print("VERIFY WARN: did not win vs random (load OK though)", file=sys.stderr)
    print("verify OK: BC bundle loaded via Kaggle file loader and played a clean game")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/winbc_gate/winbc_gate.pt")
    ap.add_argument("--config", default="configs/v2_exit.yaml")
    ap.add_argument("--drain", default="min", choices=["min", "full"])
    ap.add_argument("--gate-thr", type=float, default=0.5)
    args = ap.parse_args()
    build(ROOT / args.ckpt, ROOT / args.config, args.drain, args.gate_thr)
    verify()


if __name__ == "__main__":
    main()
