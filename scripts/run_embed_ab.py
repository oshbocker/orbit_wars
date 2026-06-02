"""Run the embed_dim=128 vs 256 ExIt A/B (sequentially, sharing one demo cache).

Arm A (embed128) runs first and collects+caches the apex demos; arm B (embed256)
reuses that cache, so the demonstration set is identical and collected only once.
Each arm is launched as `python -m v2.exit_train --config <arm>` in its own
process, so a crash in one arm doesn't take down the other.

    uv run python scripts/run_embed_ab.py            # both arms, A then B
    uv run python scripts/run_embed_ab.py --only 256 # just arm B (cache must exist)

After (or during) the run, compare with:
    uv run python scripts/plot_embed_ab.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARMS = {
    "128": "configs/v2_exit_embed128.yaml",
    "256": "configs/v2_exit_embed256.yaml",
}


def run_arm(tag: str) -> int:
    cfg = ARMS[tag]
    print(f"\n{'='*70}\n=== ExIt A/B arm embed{tag}  ({cfg}) ===\n{'='*70}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "v2.exit_train", "--config", cfg], cwd=ROOT)
    print(f"=== arm embed{tag} exited code={proc.returncode} in {time.time()-t0:.0f}s ===", flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["128", "256"], default=None,
                    help="run a single arm (default: both, 128 then 256)")
    args = ap.parse_args()

    order = [args.only] if args.only else ["128", "256"]
    for tag in order:
        rc = run_arm(tag)
        if rc != 0:
            print(f"arm embed{tag} failed (code {rc}); stopping.", flush=True)
            return rc
    print("\nA/B complete. Compare with: uv run python scripts/plot_embed_ab.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
