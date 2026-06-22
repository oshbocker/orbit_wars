"""Win-weighted full-policy BC of the TOP LADDER — the regime-check the Kaggle #30
discussion (orbit-wars/discussion/707869) says we never actually ran.

Our prior BC attempts all failed in a specific regime: they cloned PRODUCER/v5 (a ~1200
teacher → capped at ~1200), as a DELTA/reranker over producer's exact flow-diff, at tiny
scale. The #30 author reached top-10-20 with a *full standalone policy*, cloned from
*top-ladder daily-episode replays* (not a demonstrator agent), **win/outcome-weighted** so it
learns "good play" not the average passive player, with the no-op label imbalance rebalanced.

This script builds that dataset from the CACHED top-ladder replays in
``outputs/replay_pulse/cache/<date>`` (already curated to ~top-10% of games) and trains a
standalone v2 OrbitNet pointer policy:
  - ALL seats of ALL cached top games (not one team) — win-weighted per the replay ``rewards``
    (winning seat = weight 1.0, losing seat = ``--w_loss``, default 0.3).
  - no-op rebalance: launch rows upweighted ``--launch_weight`` (holds dominate ~5-10:1).
  - target pointer + hold only; sizing is analytic at execution (agents/external/bc_teacher.py
    — a STANDALONE executor, NOT routed through producer/v5).

This is a REGIME CHECK, not the final model: small OrbitNet (~552K), local CPU, a few
top-ladder games. PASS = the clone MOVES (non-trivial launch rate) and CRUSHES the weak
700-1000 baselines in the arena. If it does, the Colab scale-up (25M params, ~28M states,
win-weighted, then lagged self-play RL) is justified; if it collapses to no-op or loses to
weak bots, the regime is still wrong and we diagnose before spending GPU.

    uv run python scripts/winbc_train.py --date 2026-06-12 --max_games 200 \
        --epochs 30 --out outputs/checkpoints/winbc/ckpt.pt

Then gate (sets the executor checkpoint):
    OW_BC_TEACHER_CKPT=outputs/checkpoints/winbc/ckpt.pt \
        uv run python scripts/arena.py --agents bc_teacher,random,enders_1000 --games 30 --workers 6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.teacher_bc import label_seat  # noqa: E402
from v2.config import load_v2_config  # noqa: E402
from v2.model import OrbitNet  # noqa: E402

CACHE = ROOT / "outputs" / "replay_pulse" / "cache"


def build_dataset(cfg, date: str, max_games: int, w_loss: float) -> list[dict]:
    """All seats of all cached top games for ``date``, win-weighted by replay ``rewards``."""
    P = cfg.env.max_planets
    files = sorted(glob.glob(str(CACHE / date / "*.json")))
    if not files:
        raise SystemExit(f"no cached replays at {CACHE / date} — run replay_pulse.py first")
    if max_games > 0:
        files = files[:max_games]
    ex: list[dict] = []
    n_games = n_seats = n_win = n_loss = 0
    for fp in files:
        try:
            rep = json.load(open(fp))
        except Exception:  # noqa: BLE001
            continue
        steps = rep.get("steps")
        rewards = rep.get("rewards")
        if not steps or rewards is None:
            continue
        n_games += 1
        for seat in range(len(steps[0])):
            r = rewards[seat] if seat < len(rewards) else 0
            won = (r is not None) and (r > 0)
            w = 1.0 if won else w_loss
            seat_ex = label_seat(steps, seat, cfg, P)
            if not seat_ex:
                continue
            for e in seat_ex:
                e["w"] = np.float32(w)
            ex.extend(seat_ex)
            n_seats += 1
            if won:
                n_win += len(seat_ex)
            else:
                n_loss += len(seat_ex)
        if n_games % 25 == 0:
            print(f"  {n_games} games -> {len(ex)} examples", flush=True)
    print(f"scanned {n_games} top games, {n_seats} seats -> {len(ex)} examples "
          f"(win-seat {n_win} / lose-seat {n_loss}); w_loss={w_loss}")
    return ex


def _stack(ex, key, device):
    return torch.from_numpy(np.stack([e[key] for e in ex])).to(device)


def make_batch_w(ex, idx, device):
    sub = [ex[i] for i in idx]
    return {
        "planet_features": _stack(sub, "pf", device),
        "global_features": _stack(sub, "gf", device),
        "planet_mask": _stack(sub, "pm", device),
        "own_mask": _stack(sub, "om", device),
        "target_indices": _stack(sub, "ti", device),
        "supervise_mask": _stack(sub, "sup", device),
        "weights": torch.from_numpy(np.array([e["w"] for e in sub], dtype=np.float32)).to(device),
    }


def prestack(ex, device):
    """Stack the whole split into contiguous tensors ONCE (per-batch np.stack of object
    arrays was ~26 min/epoch; tensor-slicing is ~100x faster on CPU)."""
    return {
        "planet_features": torch.from_numpy(np.stack([e["pf"] for e in ex])).to(device),
        "global_features": torch.from_numpy(np.stack([e["gf"] for e in ex])).to(device),
        "planet_mask": torch.from_numpy(np.stack([e["pm"] for e in ex])).to(device),
        "own_mask": torch.from_numpy(np.stack([e["om"] for e in ex])).to(device),
        "target_indices": torch.from_numpy(np.stack([e["ti"] for e in ex])).to(device),
        "supervise_mask": torch.from_numpy(np.stack([e["sup"] for e in ex])).to(device),
        "weights": torch.from_numpy(np.array([e["w"] for e in ex], dtype=np.float32)).to(device),
    }


@torch.no_grad()
def tensor_acc(model, T, bs=4096):
    model.eval()
    n_all = c_all = n_l = c_l = n_h = c_h = 0
    N = T["own_mask"].shape[0]
    for s in range(0, N, bs):
        sl = slice(s, min(s + bs, N))
        out = model(T["planet_features"][sl], T["global_features"][sl],
                    T["planet_mask"][sl], T["own_mask"][sl])
        pred = out.logits.argmax(-1)
        sup = T["own_mask"][sl] & T["supervise_mask"][sl]
        tgt = T["target_indices"][sl]
        correct = (pred == tgt) & sup
        launch = sup & (tgt > 0)
        hold = sup & (tgt == 0)
        n_all += int(sup.sum()); c_all += int(correct.sum())
        n_l += int(launch.sum()); c_l += int((correct & launch).sum())
        n_h += int(hold.sum()); c_h += int((correct & hold).sum())

    def f(c, n):
        return c / n if n else float("nan")
    return f(c_all, n_all), f(c_l, n_l), f(c_h, n_h), n_l, n_h


def winbc_gate_loss(model, batch, launch_weight: float):
    """Decoupled GATE + POINTER loss (Kaggle #30 / vkhydras regime fix).

    Instead of 'hold' competing as a class inside one target softmax (winbc_loss), the act
    decision is its own per-source binary gate:
      - GATE: launch/no-launch BCE over every supervised owned source, with pos_weight=
        launch_weight (launch is the minority positive) AND the per-example outcome weight.
        The no-op rebalance lives HERE now, decoupled from target selection.
      - POINTER: target CE over the P target columns, on LAUNCHED rows ONLY — the pointer is
        never trained to emit 'hold', so its target signal stays clean.
    """
    out = model(batch["planet_features"], batch["global_features"],
                batch["planet_mask"], batch["own_mask"])
    sup = batch["own_mask"] & batch["supervise_mask"]   # [B, P]
    tgt = batch["target_indices"]                        # [B, P]; 0 = hold, k>0 = target k-1
    wexp = batch["weights"][:, None].expand_as(sup)      # [B, P]
    is_launch = tgt > 0

    # GATE: binary launch/no-launch BCE over supervised owned rows.
    glogit = out.gate_logits[sup]
    glabel = is_launch[sup].float()
    gw = wexp[sup]
    pw = torch.as_tensor(float(launch_weight), device=glogit.device)
    if glogit.shape[0] > 0:
        gate_bce = F.binary_cross_entropy_with_logits(
            glogit, glabel, weight=gw, pos_weight=pw, reduction="sum"
        ) / gw.sum().clamp(min=1e-6)
    else:
        gate_bce = torch.zeros((), device=out.gate_logits.device, requires_grad=True)

    # POINTER: target CE over the P target columns, launched rows only, outcome-weighted.
    plaunch = sup & is_launch
    ptr = out.logits[:, :, 1:]                           # [B, P, P] (drop hold column)
    pl = ptr[plaunch]
    pt = tgt[plaunch] - 1
    pwr = wexp[plaunch]
    if pl.shape[0] > 0:
        pce = F.cross_entropy(pl.clamp(min=-1e4), pt, reduction="none")
        pointer_ce = (pce * pwr).sum() / pwr.sum().clamp(min=1e-6)
    else:
        pointer_ce = torch.zeros((), device=ptr.device)
    return gate_bce + pointer_ce


@torch.no_grad()
def gate_pointer_metrics(model, T, bs=4096, k=5, thr=0.5):
    """Gate precision/recall/F1 + pointer Recall@k / MRR — the metrics vkhydras + Wilson #1011
    say to trust (top-1 target acc is misleading; many targets are reasonable)."""
    model.eval()
    tp = fp = fn = 0
    rk = nl = 0
    mrr = 0.0
    N = T["own_mask"].shape[0]
    for s in range(0, N, bs):
        sl = slice(s, min(s + bs, N))
        out = model(T["planet_features"][sl], T["global_features"][sl],
                    T["planet_mask"][sl], T["own_mask"][sl])
        sup = T["own_mask"][sl] & T["supervise_mask"][sl]
        tgt = T["target_indices"][sl]
        is_launch = (tgt > 0) & sup
        pred = (torch.sigmoid(out.gate_logits) > thr) & sup
        tp += int((pred & is_launch).sum())
        fp += int((pred & sup & ~is_launch).sum())
        fn += int((~pred & is_launch).sum())
        if is_launch.any():
            ptr = out.logits[:, :, 1:][is_launch]        # [Nl, P]
            true = (tgt[is_launch] - 1)[:, None]
            true_logit = ptr.gather(1, true)
            rank = (ptr > true_logit).sum(dim=1) + 1      # 1-based rank of the true target
            rk += int((rank <= k).sum())
            mrr += float((1.0 / rank.float()).sum())
            nl += int(is_launch.sum())

    def safe(n, d):
        return n / d if d else float("nan")
    prec, rec = safe(tp, tp + fp), safe(tp, tp + fn)
    f1 = safe(2 * prec * rec, prec + rec) if (prec == prec and rec == rec) else float("nan")
    return prec, rec, f1, safe(rk, nl), safe(mrr, nl), nl


def load_shard(path, device):
    """Load one build_shards .npz into model-ready tensors (upcast float16 -> float32)."""
    d = np.load(path)
    return {
        "planet_features": torch.from_numpy(d["pf"].astype(np.float32)).to(device),
        "global_features": torch.from_numpy(d["gf"].astype(np.float32)).to(device),
        "planet_mask": torch.from_numpy(d["pm"]).to(device),
        "own_mask": torch.from_numpy(d["om"]).to(device),
        "target_indices": torch.from_numpy(d["ti"].astype(np.int64)).to(device),
        "supervise_mask": torch.from_numpy(d["sup"]).to(device),
        "weights": torch.from_numpy(d["w"].astype(np.float32)).to(device),
    }


def concat_shards(paths, device, cap):
    """Concatenate shards into one tensor dict, up to ``cap`` examples (for the val set)."""
    acc: dict[str, list] = {}
    n = 0
    for p in paths:
        T = load_shard(p, device)
        for k, v in T.items():
            acc.setdefault(k, []).append(v)
        n += T["own_mask"].shape[0]
        if n >= cap:
            break
    return {k: torch.cat(v)[:cap] for k, v in acc.items()}


def load_shard_cpu(path):
    """Decode one build_shards .npz to CPU tensors at NATIVE on-disk dtype (no upcast, no
    device move). The per-batch GPU move upcasts (fp16->fp32, int16->int64); keeping the
    RAM/stream copy at reduced precision halves both the cache footprint and H2D bandwidth.
    Pure I/O + numpy (no torch GPU/rng), so a worker thread can run it concurrently with
    the GPU train step (np.load's zlib decompress releases the GIL)."""
    with np.load(path) as d:
        return {
            "planet_features": torch.from_numpy(d["pf"]),   # fp16
            "global_features": torch.from_numpy(d["gf"]),   # fp16
            "planet_mask": torch.from_numpy(d["pm"]),       # bool
            "own_mask": torch.from_numpy(d["om"]),          # bool
            "target_indices": torch.from_numpy(d["ti"]),    # int16
            "supervise_mask": torch.from_numpy(d["sup"]),   # bool
            "weights": torch.from_numpy(d["w"]),            # fp16
        }


def _batch_to_device(T, idx, device):
    """Index a native-dtype CPU shard and move just the batch to GPU, upcasting there.
    Widening fp16->fp32 / int16->int64 is exact, so this is bit-identical to the old
    CPU-side ``load_shard`` upcast — only the work moved (per-batch, half the bytes)."""
    return {
        "planet_features": T["planet_features"][idx].to(device, non_blocking=True).float(),
        "global_features": T["global_features"][idx].to(device, non_blocking=True).float(),
        "planet_mask": T["planet_mask"][idx].to(device, non_blocking=True),
        "own_mask": T["own_mask"][idx].to(device, non_blocking=True),
        "target_indices": T["target_indices"][idx].to(device, non_blocking=True).long(),
        "supervise_mask": T["supervise_mask"][idx].to(device, non_blocking=True),
        "weights": T["weights"][idx].to(device, non_blocking=True).float(),
    }


def _avail_ram_bytes() -> int:
    """Best-effort available RAM (Linux /proc/meminfo, then sysconf). 0 if unknown -> the
    auto fit-check fails closed to streaming."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError):
        return 0


class _RamCache:
    """Decode every train shard to CPU ONCE, then re-serve from RAM each epoch. Eliminates
    the per-epoch re-decompress (the I/O bottleneck) when the corpus fits in RAM."""

    def __init__(self, train_files, first=None):
        self.shards = [
            first if (i == 0 and first is not None) else load_shard_cpu(p)
            for i, p in enumerate(train_files)
        ]

    def epoch(self, order):
        for si in order:
            yield self.shards[si]


class _PrefetchStream:
    """Stream shards from disk, decoding the next shard(s) on a background thread while the
    GPU trains on the current one. OOM-safe (only ``depth`` shards in flight) — the path for
    the >RAM (~28M-state) regime. Overlaps decompress with compute; no full-corpus RAM load."""

    def __init__(self, train_files, depth=2):
        self.files = train_files
        self.depth = max(1, depth)

    def epoch(self, order):
        import queue
        import threading

        q: queue.Queue = queue.Queue(maxsize=self.depth)

        def work():
            try:
                for si in order:
                    q.put(load_shard_cpu(self.files[si]))
            finally:
                q.put(None)  # sentinel

        t = threading.Thread(target=work, daemon=True)
        t.start()
        while True:
            T = q.get()
            if T is None:
                break
            yield T
        t.join()


def _make_shard_provider(train_files, train_count, args):
    """Pick the data path: decode-once RAM cache (fast, when it fits) vs prefetched streaming
    (OOM-safe). ``auto`` estimates the decoded (native-dtype) cache size from the first shard
    and compares against available RAM; ``ram``/``stream`` force the choice."""
    mode = args.shard_cache
    if mode == "stream" or not train_files:
        print("shard-cache: STREAM (prefetched, OOM-safe)", flush=True)
        return _PrefetchStream(train_files, args.prefetch)
    first = load_shard_cpu(train_files[0])
    bpe = sum(v.element_size() * v.nelement() for v in first.values()) / max(
        first["own_mask"].shape[0], 1)
    est = bpe * train_count
    avail = _avail_ram_bytes()
    fits = avail > 0 and est < args.ram_frac * avail
    if mode == "ram" or (mode == "auto" and fits):
        why = "forced" if mode == "ram" else f"fits {args.ram_frac:.0%} budget"
        print(f"shard-cache: RAM ({why}; est {est/1e9:.1f}GB decoded, "
              f"{avail/1e9:.1f}GB avail)", flush=True)
        return _RamCache(train_files, first=first)
    print(f"shard-cache: STREAM (est {est/1e9:.1f}GB decoded >= {args.ram_frac:.0%} of "
          f"{avail/1e9:.1f}GB avail; prefetched)", flush=True)
    return _PrefetchStream(train_files, args.prefetch)


def run_sharded(cfg, args, device):
    """Stream a build_shards corpus shard-by-shard — the path that scales to ~28M states
    without the in-RAM object-npz OOM. Writes per-epoch learning curves to --curves."""
    import csv
    import json as _json

    sd = Path(args.shards)
    manifest = _json.loads((sd / "manifest.json").read_text())
    files = [sd / s["file"] for s in manifest["shards"]]
    counts = [int(s.get("count", 0)) for s in manifest["shards"]]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(files))
    n_val = max(1, int(len(files) * args.val_frac))
    val_files = [files[i] for i in perm[:n_val]]
    train_files = [files[i] for i in perm[n_val:]]
    train_count = sum(counts[i] for i in perm[n_val:])
    print(f"shards: {len(files)} ({manifest['n_examples']} ex) -> "
          f"train={len(train_files)} ({train_count} ex) val={len(val_files)} (cap {args.val_cap})")
    Tva = concat_shards(val_files, device, args.val_cap)

    model = OrbitNet(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"OrbitNet params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"gate_head={args.gate_head}  device={device}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    last = out.with_suffix(".last.pt")

    # Resume from .last.pt (model + optimizer + epoch + best) so a Colab disconnect mid-train
    # picks up where it left off. Safe to always pass --resume: absent ckpt -> fresh from ep 0.
    start_ep, best, resumed = 0, -1.0, False
    if args.resume and last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        start_ep = int(ck.get("epoch", -1)) + 1
        best = float(ck.get("best", -1.0))
        resumed = True
        print(f"RESUMED from {last}: epoch {start_ep}, best={best:.3f}")

    cf = cw = None
    if args.curves:
        Path(args.curves).parent.mkdir(parents=True, exist_ok=True)
        # Append (no header) when resuming an existing curves file; else fresh with header.
        append = resumed and Path(args.curves).exists()
        # Handle must outlive the epoch loop (written once per epoch), so it can't be a
        # `with` block here; closed explicitly after training.
        cf = open(args.curves, "a" if append else "w", newline="")  # noqa: SIM115
        cw = csv.writer(cf)
        if not append:
            cw.writerow(["epoch", "loss", "gate_prec", "gate_rec", "gate_f1",
                         "recall@5", "mrr", "launch_acc", "sel"])
    if start_ep >= args.epochs:
        print(f"already at/past epoch {args.epochs}; nothing to do")
        if cf:
            cf.close()
        return

    # Decode-once RAM cache (fast path) or prefetched streaming (OOM-safe >RAM path). Either
    # way the per-epoch shard order + within-shard shuffle still come from ``rng`` below in the
    # SAME draw sequence as before (the provider only DECODES — no rng) -> identical training.
    provider = _make_shard_provider(train_files, train_count, args)
    for ep in range(start_ep, args.epochs):
        model.train()
        tot = nb = 0.0
        epoch_order = rng.permutation(len(train_files))
        for T in provider.epoch(epoch_order):
            n = T["own_mask"].shape[0]
            order = torch.from_numpy(rng.permutation(n).astype(np.int64))
            for s in range(0, n, args.batch):
                idx = order[s:s + args.batch]
                b = _batch_to_device(T, idx, device)
                loss = (winbc_gate_loss if args.gate_head else winbc_loss)(
                    model, b, args.launch_weight)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                opt.step()
                tot += float(loss.detach())
                nb += 1
        avg = tot / max(nb, 1)
        if args.gate_head:
            prec, rec, f1, rkk, mrr, nl = gate_pointer_metrics(model, Tva)
            sel = (f1 if f1 == f1 else 0.0) + (mrr if mrr == mrr else 0.0)  # noqa: PLR0124
            lacc = float("nan")
            print(f"epoch {ep:3d}  loss={avg:.4f}  gate P/R/F1={prec:.3f}/{rec:.3f}/{f1:.3f}"
                  f"  recall@5={rkk:.3f}  mrr={mrr:.3f}  sel={sel:.3f}", flush=True)
        else:
            acc, lacc, hacc, nl, nh = tensor_acc(model, Tva)
            prec = rec = f1 = rkk = mrr = float("nan")
            sel = lacc
            print(f"epoch {ep:3d}  loss={avg:.4f}  val acc={acc:.3f}  launch_acc={lacc:.3f}",
                  flush=True)
        if cw:
            cw.writerow([ep, f"{avg:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}",
                         f"{rkk:.4f}", f"{mrr:.4f}", f"{lacc:.4f}", f"{sel:.4f}"])
        # 'best' ckpt stays lean (what bc_teacher.py loads); '.last.pt' also carries the
        # optimizer/epoch/best so --resume can continue mid-training after a disconnect.
        ck = {"model": model.state_dict(), "config": args.config, "gate_head": args.gate_head}
        if sel == sel and sel > best:  # noqa: PLR0124
            best = sel
            torch.save(ck, out)
        torch.save({**ck, "optimizer": opt.state_dict(), "epoch": ep, "best": best}, last)
    if cf:
        cf.close()
    print(f"saved best (sel={best:.3f}) -> {out}")


def winbc_loss(model, batch, launch_weight: float):
    """Class-weighted CE on (source -> target/hold), with BOTH the no-op rebalance
    (launch rows × launch_weight) AND per-example OUTCOME weight (win-seat 1.0,
    lose-seat w_loss). No reachability mask, no fraction term (sizing analytic at exec)."""
    out = model(batch["planet_features"], batch["global_features"],
                batch["planet_mask"], batch["own_mask"])
    logits = out.logits  # [B, P, P+1]
    sup = batch["own_mask"] & batch["supervise_mask"]  # [B, P]
    wexp = batch["weights"][:, None].expand_as(sup)  # [B, P]
    flat = logits[sup]
    tgt = batch["target_indices"][sup]
    wrow = wexp[sup]
    if flat.shape[0] > 0:
        true_logit = flat.gather(1, tgt[:, None]).squeeze(1)
        keep = true_logit > -1e3
        flat, tgt, wrow = flat[keep], tgt[keep], wrow[keep]
    if flat.shape[0] == 0:
        return torch.zeros((), device=logits.device, requires_grad=True)
    flat = flat.clamp(min=-1e4)
    ce = F.cross_entropy(flat, tgt, reduction="none")
    w = torch.where(tgt > 0, launch_weight, 1.0) * wrow
    return (ce * w).sum() / w.sum().clamp(min=1e-6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_exit.yaml")
    ap.add_argument("--date", default="2026-06-12")
    ap.add_argument("--max_games", type=int, default=200, help="0 = all cached games")
    ap.add_argument("--w_loss", type=float, default=0.3, help="weight on losing-seat examples")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--launch_weight", type=float, default=8.0)
    ap.add_argument("--gate_head", action="store_true",
                    help="train a SEPARATE launch/no-launch gate in front of the pointer "
                         "(vkhydras regime fix) instead of hold-as-a-class")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--cache", default="outputs/winbc/dataset.npz")
    ap.add_argument("--out", default="outputs/checkpoints/winbc/ckpt.pt")
    ap.add_argument("--rebuild", action="store_true")
    # Scaled streaming path (build_shards corpus) — the Colab/Lambda regime.
    ap.add_argument("--shards", default="", help="shard dir from build_shards.py; "
                    "enables streaming training (no full-corpus RAM load)")
    ap.add_argument("--curves", default="", help="per-epoch metrics CSV (learning curves)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from <out>.last.pt if present (model+optimizer+epoch); "
                         "safe to always pass — no ckpt means a fresh run from epoch 0")
    ap.add_argument("--val-frac", type=float, default=0.05, help="fraction of shards held out")
    ap.add_argument("--val-cap", type=int, default=200000, help="max val examples")
    # Data-loading throughput (sharded path): decode-once RAM cache vs prefetched streaming.
    ap.add_argument("--shard-cache", choices=["auto", "ram", "stream"], default="auto",
                    help="auto = RAM-cache decoded shards if they fit (eliminates the "
                         "per-epoch re-decompress), else prefetched streaming; "
                         "ram/stream force the choice")
    ap.add_argument("--ram-frac", type=float, default=0.5,
                    help="max fraction of available RAM the decoded shard cache may use "
                         "(auto fit-check)")
    ap.add_argument("--prefetch", type=int, default=2,
                    help="streaming prefetch depth (shards decoded ahead on a worker thread)")
    # Model-scale overrides (match capacity to data; default = config values).
    ap.add_argument("--embed-dim", type=int, default=0, help="0 = use config")
    ap.add_argument("--n-layers", type=int, default=0)
    ap.add_argument("--ff-dim", type=int, default=0)
    ap.add_argument("--n-heads", type=int, default=0)
    args = ap.parse_args()

    cfg = load_v2_config(args.config)
    if args.gate_head:
        cfg.model.launch_gate_head = True
    if args.embed_dim:
        cfg.model.embed_dim = args.embed_dim
    if args.n_layers:
        cfg.model.n_layers = args.n_layers
    if args.ff_dim:
        cfg.model.ff_dim = args.ff_dim
    if args.n_heads:
        cfg.model.n_heads = args.n_heads
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(8, os.cpu_count() or 4))

    if args.shards:
        run_sharded(cfg, args, device)
        return

    cache = Path(args.cache)
    if cache.exists() and not args.rebuild:
        print(f"loading cached dataset {cache}")
        ex = list(np.load(cache, allow_pickle=True)["ex"])
    else:
        ex = build_dataset(cfg, args.date, args.max_games, args.w_loss)
        if not ex:
            raise SystemExit("no examples built")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ex=np.array(ex, dtype=object))
        print(f"saved {len(ex)} examples -> {cache}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ex))
    n_val = max(1, len(ex) // 10)
    val = [ex[i] for i in perm[:n_val]]
    train = [ex[i] for i in perm[n_val:]]
    print(f"train={len(train)} val={len(val)}")

    Ttr = prestack(train, device)
    Tva = prestack(val, device)
    n_tr = Ttr["own_mask"].shape[0]

    model = OrbitNet(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"OrbitNet params: {sum(p.numel() for p in model.parameters())}  threads={torch.get_num_threads()}")

    best = -1.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        order = torch.from_numpy(rng.permutation(n_tr).astype(np.int64))
        tot = nb = 0.0
        for s in range(0, n_tr, args.batch):
            idx = order[s:s + args.batch]
            b = {k: v[idx] for k, v in Ttr.items()}
            loss = (winbc_gate_loss if args.gate_head else winbc_loss)(model, b, args.launch_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        if args.gate_head:
            # Select on gate-F1 + pointer-MRR, NOT top-1 launch_acc (misleading per #1011).
            prec, rec, f1, rkk, mrr, nl = gate_pointer_metrics(model, Tva)
            sel = (f1 if f1 == f1 else 0.0) + (mrr if mrr == mrr else 0.0)  # noqa: PLR0124
            print(f"epoch {ep:3d}  loss={tot/nb:.4f}  gate P/R/F1={prec:.3f}/{rec:.3f}/{f1:.3f}"
                  f"  recall@5={rkk:.3f}  mrr={mrr:.3f}  (nl={nl})  sel={sel:.3f}", flush=True)
        else:
            acc, lacc, hacc, nl, nh = tensor_acc(model, Tva)
            sel = lacc
            print(f"epoch {ep:3d}  loss={tot/nb:.4f}  val acc={acc:.3f}  "
                  f"launch_acc={lacc:.3f}(n={nl})  hold_acc={hacc:.3f}(n={nh})", flush=True)
        if sel == sel and sel > best:  # noqa: PLR0124 (NaN guard)
            best = sel
            torch.save({"model": model.state_dict(), "config": args.config,
                        "gate_head": args.gate_head}, out)
    print(f"saved best (sel={best:.3f}) -> {out}")


if __name__ == "__main__":
    main()
