"""Run the embed_dim=128 vs 256 ExIt A/B (sequentially, sharing one demo cache).

Arm A (embed128) runs first and collects+caches the apex demos; arm B (embed256)
reuses that cache, so the demonstration set is identical and collected only once.
Each arm is launched as `python -m v2.exit_train --config <arm>` in its own
process, so a crash in one arm doesn't take down the other.

Local:
    uv run python scripts/run_embed_ab.py              # both arms, A then B
    uv run python scripts/run_embed_ab.py --only 256   # just arm B (cache must exist)

Colab (persist to Drive, use all vCPUs):
    !python scripts/run_embed_ab.py \
        --save-dir /content/drive/MyDrive/orbit_wars_outputs/checkpoints \
        --log-dir  /content/drive/MyDrive/orbit_wars_outputs/logs \
        --demo-cache /content/drive/MyDrive/orbit_wars_outputs/demos_apex_exit_ab.pkl \
        --collect-workers 10 --search-workers 10

Compare with: uv run python scripts/plot_embed_ab.py
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


def _resolve_config(tag: str, args: argparse.Namespace) -> str:
    """Return a config path, writing a Drive/worker-overridden temp copy if any
    override flag is set (so the same committed configs work on Colab unchanged)."""
    base = ARMS[tag]
    overrides = (args.save_dir or args.log_dir or args.demo_cache
                 or args.collect_workers is not None or args.search_workers is not None)
    if not overrides:
        return base

    import yaml

    from v2.config import load_v2_config, v2_config_to_dict
    cfg = load_v2_config(str(ROOT / base))
    if args.save_dir:
        cfg.save_dir = args.save_dir
    if args.log_dir:
        cfg.log_dir = args.log_dir
    if args.demo_cache:
        cfg.imitation.bc_cache_path = args.demo_cache   # shared across arms => fair + collect-once
    if args.collect_workers is not None:
        cfg.exit.collect_workers = args.collect_workers
    if args.search_workers is not None:
        cfg.exit.search_workers = args.search_workers
    out = Path(f"/tmp/embed_ab_{tag}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(v2_config_to_dict(cfg), f, sort_keys=True)
    print(f"  [{tag}] using overridden config -> {out} "
          f"(save_dir={cfg.save_dir}, cache={cfg.imitation.bc_cache_path}, "
          f"collect_workers={cfg.exit.collect_workers})", flush=True)
    return str(out)


def run_arm(tag: str, args: argparse.Namespace) -> int:
    cfg = _resolve_config(tag, args)
    print(f"\n{'='*70}\n=== ExIt A/B arm embed{tag}  ({cfg}) ===\n{'='*70}", flush=True)
    t0 = time.time()
    proc = subprocess.run([sys.executable, "-m", "v2.exit_train", "--config", cfg], cwd=ROOT)
    print(f"=== arm embed{tag} exited code={proc.returncode} in {time.time()-t0:.0f}s ===", flush=True)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["128", "256"], default=None,
                    help="run a single arm (default: both, 128 then 256)")
    ap.add_argument("--save-dir", default=None, help="override checkpoint dir (e.g. Drive)")
    ap.add_argument("--log-dir", default=None, help="override log dir (e.g. Drive)")
    ap.add_argument("--demo-cache", default=None,
                    help="override BC demo cache path (shared across arms)")
    ap.add_argument("--collect-workers", type=int, default=None,
                    help="override exit.collect_workers (set ~vCPU count on Colab)")
    ap.add_argument("--search-workers", type=int, default=None,
                    help="override exit.search_workers (set ~vCPU count on Colab)")
    args = ap.parse_args()

    order = [args.only] if args.only else ["128", "256"]
    for tag in order:
        rc = run_arm(tag, args)
        if rc != 0:
            print(f"arm embed{tag} failed (code {rc}); stopping.", flush=True)
            return rc
    print("\nA/B complete. Compare with: uv run python scripts/plot_embed_ab.py", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
