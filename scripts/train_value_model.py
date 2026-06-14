"""Train the v5 global value re-ranker MLP on harvested win/loss labels (Axis C).

Loads every .npz from scripts/harvest_values.py, splits train/val by GAME (no
within-game leakage), trains the 16 -> 32 -> 16 -> 1 MLP on CPU, and saves the
weights in the npz layout consumed by
agents/v5/orbit_lite_v5/value_reranker.ValueModel.

    uv run python scripts/train_value_model.py \
        --data outputs/value/raw --out agents/v5/orbit_lite_v5/value_model_weights.npz

The model is a TIE-BREAKER: AUC well above 0.5 (states are separable by win) is
expected and necessary but NOT sufficient — the only verdict that matters is the
arena mirror gate (v5:value_rerank_eps=X vs v5, n>=120). See LEADERBOARD_CLIMB_PLAN.
"""

from __future__ import annotations

import argparse
import glob
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FEATURE_DIM = 16
HIDDEN = 32


def load_dataset(data_dir: str, val_frac_mod: int = 5):
    """All states; val = games whose name-hash % mod == 0 (split by game)."""
    tr_x, tr_y, va_x, va_y = [], [], [], []
    files = sorted(glob.glob(f"{data_dir}/*.npz"))
    if not files:
        raise SystemExit(f"no .npz files in {data_dir}")
    for f in files:
        d = np.load(f, allow_pickle=True)
        x = d["features"].astype(np.float32)
        y = d["label"].astype(np.float32)
        if len(y) == 0:
            continue
        if zlib.crc32(Path(f).name.encode()) % val_frac_mod == 0:
            va_x.append(x)
            va_y.append(y)
        else:
            tr_x.append(x)
            tr_y.append(y)
    return (
        np.concatenate(tr_x),
        np.concatenate(tr_y),
        np.concatenate(va_x),
        np.concatenate(va_y),
        len(files),
    )


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC (tie-corrected via average ranks)."""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    sp = p[order]
    i = 0
    while i < len(sp):
        j = i
        while j + 1 < len(sp) and sp[j + 1] == sp[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def build_mlp() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(FEATURE_DIM, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, FEATURE_DIM),
        nn.ReLU(),
        nn.Linear(FEATURE_DIM, 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="outputs/value/raw")
    ap.add_argument("--out", default="agents/v5/orbit_lite_v5/value_model_weights.npz")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tr_x, tr_y, va_x, va_y, n_games = load_dataset(args.data)
    print(
        f"{n_games} games -> train {len(tr_y)} states (win {tr_y.mean():.3f}), "
        f"val {len(va_y)} states (win {va_y.mean():.3f})"
    )

    model = build_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.BCEWithLogitsLoss()
    tx = torch.from_numpy(tr_x)
    ty = torch.from_numpy(tr_y)
    vx = torch.from_numpy(va_x)

    best_auc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ty))
        tot = 0.0
        for i in range(0, len(ty), args.batch):
            idx = perm[i : i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(tx[idx]).squeeze(-1), ty[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(vx).squeeze(-1)).numpy()
        auc = auc_score(va_y, vp)
        acc = float(((vp > 0.5) == (va_y > 0.5)).mean())
        marker = ""
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = " *"
        print(f"epoch {epoch:>3}  loss {tot / len(ty):.4f}  val AUC {auc:.4f}  acc {acc:.3f}{marker}")

    assert best_state is not None
    model.load_state_dict(best_state)
    print(f"\nbest val AUC {best_auc:.4f}")

    # win-prob calibration by predicted-decile (sanity that it's a real probability)
    model.eval()
    with torch.no_grad():
        vp = torch.sigmoid(model(vx).squeeze(-1)).numpy()
    print("\ncalibration (val): pred-decile -> empirical win rate")
    order = np.argsort(vp)
    for b in range(10):
        lo = b * len(vp) // 10
        hi = (b + 1) * len(vp) // 10
        idx = order[lo:hi]
        if len(idx):
            print(f"  decile {b}: pred {vp[idx].mean():.3f}  actual {va_y[idx].mean():.3f}  n={len(idx)}")

    sd = model.state_dict()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        w0=sd["0.weight"].numpy().astype(np.float32),
        b0=sd["0.bias"].numpy().astype(np.float32),
        w2=sd["2.weight"].numpy().astype(np.float32),
        b2=sd["2.bias"].numpy().astype(np.float32),
        w4=sd["4.weight"].numpy().astype(np.float32),
        b4=sd["4.bias"].numpy().astype(np.float32),
        in_dim=np.int64(FEATURE_DIM),
        hidden=np.int64(HIDDEN),
    )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
