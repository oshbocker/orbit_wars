"""Train the reject-only shot validator MLP on harvested labels (Phase 2.1).

Loads every .npz from scripts/harvest_shots.py, keeps ATTACK shots only (own-
planet reinforcements are exempt at inference), splits train/val by GAME (no
within-game leakage), trains the 24 -> 64 -> 32 -> 1 MLP on CPU, and saves the
weights in the konbu npz layout consumed by
agents/v5/orbit_lite_v5/shot_validator.NumpyValidator.

    uv run python scripts/train_shot_validator.py \
        --data outputs/validator/raw --out outputs/validator/validator_weights.npz
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

FEATURE_DIM = 24


def load_dataset(data_dir: str, val_frac_mod: int = 5):
    """Attack shots from all games; val = games whose name-hash % mod == 0."""
    tr_x, tr_y, va_x, va_y, va_shooter = [], [], [], [], []
    files = sorted(glob.glob(f"{data_dir}/*.npz"))
    if not files:
        raise SystemExit(f"no .npz files in {data_dir}")
    for f in files:
        d = np.load(f, allow_pickle=True)
        atk = d["own_self"] == 0
        if not atk.any():
            continue
        x = d["features"][atk].astype(np.float32)
        y = d["label"][atk].astype(np.float32)
        if zlib.crc32(Path(f).name.encode()) % val_frac_mod == 0:
            va_x.append(x)
            va_y.append(y)
            va_shooter.append(d["shooter"][atk])
        else:
            tr_x.append(x)
            tr_y.append(y)
    return (
        np.concatenate(tr_x),
        np.concatenate(tr_y),
        np.concatenate(va_x),
        np.concatenate(va_y),
        np.concatenate(va_shooter),
        len(files),
    )


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney AUC (tie-corrected via average ranks)."""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
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
        nn.Linear(FEATURE_DIM, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="outputs/validator/raw")
    ap.add_argument("--out", default="outputs/validator/validator_weights.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tr_x, tr_y, va_x, va_y, va_shooter, n_games = load_dataset(args.data)
    print(
        f"{n_games} games -> train {len(tr_y)} shots (pos {tr_y.mean():.3f}), "
        f"val {len(va_y)} shots (pos {va_y.mean():.3f})"
    )

    model = build_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    model.eval()
    with torch.no_grad():
        vp = torch.sigmoid(model(vx).squeeze(-1)).numpy()
    print(f"\nbest val AUC {best_auc:.4f}")

    print("\nthreshold sweep on val (veto = prob < t):")
    print(f"{'t':>5} {'veto%':>7} {'veto-prec':>10} {'bad-recall':>11} {'kept-pos%':>10}")
    for t in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
        veto = vp < t
        kept = ~veto
        prec = float((va_y[veto] == 0).mean()) if veto.any() else float("nan")
        recall = float(veto[va_y == 0].mean())
        kept_pos = float(va_y[kept].mean()) if kept.any() else float("nan")
        print(f"{t:>5.2f} {veto.mean():>6.1%} {prec:>9.1%} {recall:>10.1%} {kept_pos:>9.1%}")

    print("\nveto rate at t=0.4 by shooter (val):")
    for name in np.unique(va_shooter):
        m = va_shooter == name
        print(
            f"  {name:<14} shots {int(m.sum()):>6}  pos {va_y[m].mean():.3f}  "
            f"veto {(vp[m] < 0.4).mean():.1%}"
        )

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
        hidden=np.int64(64),
    )
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
