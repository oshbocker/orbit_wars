"""Build + verify the v5 submission bundle (producer-fork packaging).

Replicates the proven producer_bundle.tar.gz layout: main.py + orbit_lite_v5/ at
the ARCHIVE ROOT (Kaggle extracts to /kaggle_simulations/agent/ and loads the last
callable from main.py; see memory kaggle-submission-tarball-root). After building,
extracts the tarball to a temp dir and runs a real game through Kaggle's
file-agent loader (the append->exec->pop sequence) vs the built-in random agent.

    uv run python scripts/build_v5_bundle.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "agents" / "v5"
STAGE = ROOT / "outputs" / "submissions" / "v5_bundle"
TARBALL = ROOT / "outputs" / "submissions" / "v5_bundle.tar.gz"


def build() -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    shutil.copy2(SRC / "main.py", STAGE / "main.py")
    shutil.copytree(
        SRC / "orbit_lite_v5",
        STAGE / "orbit_lite_v5",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    # archive root = bundle contents (main.py at top level), like producer_bundle
    subprocess.run(
        ["tar", "czf", str(TARBALL), "main.py", "orbit_lite_v5"],
        cwd=STAGE,
        check=True,
    )
    print(f"built {TARBALL} ({TARBALL.stat().st_size / 1024:.0f} KiB)")
    subprocess.run(["tar", "tzf", str(TARBALL)], check=True)


def verify() -> None:
    from kaggle_environments import make

    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["tar", "xzf", str(TARBALL)], cwd=td, check=True)
        main_py = str(Path(td) / "main.py")
        env = make("orbit_wars", configuration={"randomSeed": 4242})
        env.run([main_py, "random"])
        last = env.steps[-1]
        statuses = [s.status for s in last]
        rewards = [s.reward for s in last]
        print(f"verify game: statuses={statuses} rewards={rewards} steps={len(env.steps) - 1}")
        if "ERROR" in statuses or rewards[0] != 1:
            print("VERIFY FAILED", file=sys.stderr)
            sys.exit(1)
    print("verify OK: bundle agent loaded via Kaggle file loader and won vs random")


if __name__ == "__main__":
    build()
    verify()
