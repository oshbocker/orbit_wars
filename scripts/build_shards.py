#!/usr/bin/env python
"""Build a SHARDED, win-weighted BC dataset from cached top-ladder replays — the storage
redesign that lets us train at the Kaggle #30 / vkhydras scale (~28M states) without holding
the whole corpus in RAM (the single object-npz approach OOMs above ~1-2M examples).

Each example is one (decision-obs, owned source) -> target/hold label, win-weighted per the
replay's terminal ``rewards`` (winning seat = 1.0, losing seat = ``--w-loss``). Features are
stored at reduced precision (float16 / bool / int16) in COMPRESSED shards so ~47k games fit on
Drive; a manifest lists the shards. ``winbc_train.py --shards`` streams them shard-by-shard.

Fans out across CPUs (Colab has many): each worker labels a slice of replay files and writes
its OWN shards (no giant-array IPC back to the parent), then the parent writes the manifest.

    # Local smoke:
    uv run python scripts/build_shards.py --cache-root outputs/replay_pulse/cache \
        --out /tmp/shards_smoke --shard-size 2000 --max-games 20 --workers 2

    # Colab (Drive-backed, full corpus):
    python scripts/build_shards.py \
        --cache-root /content/drive/MyDrive/orbit_wars/replays \
        --out /content/drive/MyDrive/orbit_wars/shards \
        --shard-size 200000 --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Field -> on-disk dtype. Reduced precision keeps Drive footprint ~4x smaller than float32;
# the trainer upcasts to float32 on load. ti up to max_planets (40) fits int16 easily.
_DTYPES = {
    "pf": np.float16, "gf": np.float16, "pm": np.bool_,
    "om": np.bool_, "ti": np.int16, "sup": np.bool_, "w": np.float16,
}


def _flush(buf: list[dict], out: Path, prefix: str, k: int) -> tuple[str, int]:
    """Write one compressed shard from buffered examples; return (path, count)."""
    n = len(buf)
    arrs = {key: np.stack([e[key] for e in buf]).astype(dt) for key, dt in _DTYPES.items()
            if key != "w"}
    arrs["w"] = np.array([e["w"] for e in buf], dtype=np.float16)
    path = out / f"{prefix}_{k:05d}.npz"
    # pyright's partial numpy stub can't prove **arrs keys don't hit savez's allow_pickle kw.
    np.savez_compressed(path, **arrs)  # pyright: ignore[reportArgumentType]
    return str(path.name), n


def _build_worker(args_t) -> list[tuple[str, int]]:
    """Label a slice of replay files, win-weighted, flushing shards as the buffer fills."""
    files, cfg_path, out_str, shard_size, w_loss, wid = args_t
    # Imports inside the worker (fresh process); keep torch out of it — pure numpy labeling.
    from scripts.teacher_bc import label_seat
    from v2.config import load_v2_config

    cfg = load_v2_config(cfg_path)
    P = cfg.env.max_planets
    out = Path(out_str)
    prefix = f"shard_w{wid}"
    buf: list[dict] = []
    written: list[tuple[str, int]] = []
    k = 0
    for fp in files:
        try:
            with open(fp) as fh:
                rep = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        steps = rep.get("steps")
        rewards = rep.get("rewards")
        if not steps or rewards is None:
            continue
        for seat in range(len(steps[0])):
            r = rewards[seat] if seat < len(rewards) else 0
            w = 1.0 if (r is not None and r > 0) else w_loss
            seat_ex = label_seat(steps, seat, cfg, P)
            for e in seat_ex:
                e["w"] = np.float32(w)
            buf.extend(seat_ex)
            while len(buf) >= shard_size:
                written.append(_flush(buf[:shard_size], out, prefix, k))
                del buf[:shard_size]
                k += 1
    if buf:
        written.append(_flush(buf, out, prefix, k))
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_exit.yaml")
    ap.add_argument("--cache-root", default="outputs/replay_pulse/cache",
                    help="dir of date subdirs of cached replay .json (harvest_all output)")
    ap.add_argument("--out", required=True, help="shard output dir (point at Drive on Colab)")
    ap.add_argument("--shard-size", type=int, default=200000, help="examples per shard")
    ap.add_argument("--w-loss", type=float, default=0.3, help="losing-seat example weight")
    ap.add_argument("--max-games", type=int, default=0, help="0 = all cached games")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.cache_root) / "*" / "*.json")))
    if not files:
        raise SystemExit(f"no cached replays under {args.cache_root} — run harvest_all.py first")
    if args.max_games > 0:
        files = files[: args.max_games]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} replay files -> shards in {out} "
          f"(shard_size={args.shard_size}, workers={args.workers})")

    # Split files across workers; each writes its own shards (no big-array IPC).
    nw = max(1, args.workers)
    chunks = [files[i::nw] for i in range(nw)]
    tasks = [(c, args.config, str(out), args.shard_size, args.w_loss, wid)
             for wid, c in enumerate(chunks) if c]

    shards: list[dict] = []
    total = 0

    def _record(pairs):
        nonlocal total
        for s, n in pairs:
            shards.append({"file": s, "count": n})
            total += n

    if nw == 1:
        _record(_build_worker(tasks[0]))
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(_build_worker, t) for t in tasks]
            for fut in as_completed(futs):
                _record(fut.result())
                print(f"  ...{total} examples in {len(shards)} shards", flush=True)

    manifest = {
        "config": args.config, "n_examples": total, "n_shards": len(shards),
        "shard_size": args.shard_size, "w_loss": args.w_loss,
        "dtypes": {k: np.dtype(v).name for k, v in _DTYPES.items()},
        "shards": sorted(shards, key=lambda d: d["file"]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDONE: {total} examples across {len(shards)} shards -> {out}/manifest.json")


if __name__ == "__main__":
    main()
