"""Slices 1-2: build the macro-action BC dataset (exact relabeling) + train the selector.

Step 0 of the pointer-BC path: BEHAVIOR-CLONE PRODUCER in the macro representation.
Each training example = (decision-obs features, per-owned-source target label) where the
target is the EXACT destination planet (fleet-tracked, not angle-matched) and there is NO
ship-fraction label at all — sizing is done analytically at execution time (slice 3). The
net only learns WHICH source attacks WHICH target (entity selection), exactly the AlphaStar
pointer-head decomposition.

Decisive check: old fraction-regressing BC of producer hit 3% vs producer. If this selector
predicts producer's own targets with high accuracy, a clone that EXECUTES exactly should
reproduce producer (~50% mirror). Run slices 3-4 to confirm in the arena.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch.nn.functional as F  # noqa: E402

from scripts.macro_relabel import resolve_launches_for_step  # noqa: E402
from src.game_types import parse_observation  # noqa: E402
from v2.config import load_v2_config  # noqa: E402
from v2.features import encode_features  # noqa: E402
from v2.model import OrbitNet  # noqa: E402


def macro_bc_loss(model, batch, launch_weight: float):
    """Class-weighted CE on (source -> target/hold) selection. NO reachability mask
    (would -inf the true target) and NO fraction term (sizing is analytic at exec).
    Holds dominate ~5:1, so launch rows are upweighted to avoid hold-collapse."""
    out = model(batch["planet_features"], batch["global_features"],
                batch["planet_mask"], batch["own_mask"])
    logits = out.logits  # [B, P, P+1]
    sup = batch["own_mask"] & batch["supervise_mask"]
    flat = logits[sup]
    tgt = batch["target_indices"][sup]
    # Defensive guard: drop rows whose TRUE target is masked to -inf by the model
    # (non-existent/comet target, self-target). One such row contributes CE ~1e4 and
    # swamps the gradient. The dataset builder already drops these, but guard anyway.
    if flat.shape[0] > 0:
        true_logit = flat.gather(1, tgt[:, None]).squeeze(1)
        keep = true_logit > -1e3
        flat, tgt = flat[keep], tgt[keep]
    if flat.shape[0] == 0:
        return torch.zeros((), device=logits.device, requires_grad=True)
    flat = flat.clamp(min=-1e4)
    ce = F.cross_entropy(flat, tgt, reduction="none")
    w = torch.where(tgt > 0, launch_weight, 1.0)
    return (ce * w).sum() / w.sum()


def _obs(e):
    return e["observation"] if not hasattr(e, "observation") else e.observation


def _comet_args(obs):
    cids = obs.get("comet_planet_ids") if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", None)
    cdata = obs.get("comets") if isinstance(obs, dict) else getattr(obs, "comets", None)
    cids = [int(x) for x in cids] if cids is not None else None
    return cids, cdata


def build_dataset(cfg, n_games: int, seed: int, expert: str):
    from kaggle_environments import make

    from agents import load_named_agent

    P = cfg.env.max_planets
    ex: list[dict] = []
    for g in range(n_games):
        env = make("orbit_wars", configuration={"randomSeed": seed + g})
        env.run([load_named_agent(expert), load_named_agent(expert)])
        steps = env.steps
        for t in range(1, len(steps)):
            for player in (0, 1):
                mls = resolve_launches_for_step(steps, t, player)
                if not _action_nonempty(steps[t][player]) and not mls:
                    pass  # holds are still supervised below
                dec = _obs(steps[t - 1][player])
                state = parse_observation(dec)
                cids, cdata = _comet_args(dec)
                feats = encode_features(state, cfg.env, comet_ids=cids, comets_data=cdata)
                ti = np.zeros(P, dtype=np.int64)
                sup = feats.own_mask.copy()
                by_src: dict[int, list] = {}
                for ml in mls:
                    if ml.dst_id >= 0:
                        by_src.setdefault(ml.src_id, []).append(ml)
                launched_src = {ml.src_id for ml in mls}
                for src_id in range(P):
                    if not feats.own_mask[src_id]:
                        continue
                    res = by_src.get(src_id, [])
                    if res:
                        primary = max(res, key=lambda m: m.ships)
                        dst = primary.dst_id
                        # valid target = exists in features, not self (else the model
                        # masks it to -inf -> CE blows up; drop from supervision).
                        if 0 <= dst < P and dst != src_id and bool(feats.planet_mask[dst]):
                            ti[src_id] = dst + 1
                        else:
                            sup[src_id] = False
                    elif src_id in launched_src:
                        sup[src_id] = False  # launched but all fleets died -> don't teach a label
                    # else: genuine hold (ti stays 0)
                ex.append({
                    "pf": feats.planet_features, "gf": feats.global_features,
                    "pm": feats.planet_mask, "om": feats.own_mask,
                    "rm": feats.reachability_mask, "ti": ti, "sup": sup,
                })
        print(f"  game {g}: {len(steps)} steps, {len(ex)} examples so far")
    return ex


def _action_nonempty(e):
    a = e["action"] if not hasattr(e, "action") else e.action
    return bool(a)


def _stack(ex, key, device):
    return torch.from_numpy(np.stack([e[key] for e in ex])).to(device)


def make_batch(ex, idx, device):
    sub = [ex[i] for i in idx]
    # NOTE: deliberately NO reachability_mask. The net's reachability heuristic is
    # stricter than reality (the expert sometimes reaches targets it marks
    # unreachable); passing it masks the TRUE target to -inf and blows up the loss.
    # Producer only picks reachable targets, so the net learns reachability from the
    # data, and slice-3 execution routes through v5's exact candidate machinery.
    return {
        "planet_features": _stack(sub, "pf", device),
        "global_features": _stack(sub, "gf", device),
        "planet_mask": _stack(sub, "pm", device),
        "own_mask": _stack(sub, "om", device),
        "target_indices": _stack(sub, "ti", device),
        "supervise_mask": _stack(sub, "sup", device),
    }


@torch.no_grad()
def selection_accuracy(model, ex, device, bs=512):
    model.eval()
    n_all = c_all = n_launch = c_launch = n_hold = c_hold = 0
    for s in range(0, len(ex), bs):
        b = make_batch(ex, range(s, min(s + bs, len(ex))), device)
        out = model(b["planet_features"], b["global_features"], b["planet_mask"], b["own_mask"])
        pred = out.logits.argmax(-1)  # [B,P]
        sup = b["own_mask"] & b["supervise_mask"]
        tgt = b["target_indices"]
        correct = (pred == tgt) & sup
        launch = sup & (tgt > 0)
        hold = sup & (tgt == 0)
        n_all += int(sup.sum())
        c_all += int(correct.sum())
        n_launch += int(launch.sum())
        c_launch += int((correct & launch).sum())
        n_hold += int(hold.sum())
        c_hold += int((correct & hold).sum())

    def f(c, n):
        return c / n if n else float("nan")
    return f(c_all, n_all), f(c_launch, n_launch), f(c_hold, n_hold), n_launch, n_hold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_exit.yaml")
    ap.add_argument("--expert", default="producer")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--launch_weight", type=float, default=5.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--cache", default="outputs/macro_bc/dataset.npz")
    ap.add_argument("--out", default="outputs/checkpoints/macro_bc_producer/ckpt.pt")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = load_v2_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = Path(args.cache)
    if cache.exists() and not args.rebuild:
        print(f"loading cached dataset {cache}")
        z = np.load(cache, allow_pickle=True)
        ex = list(z["ex"])
    else:
        print(f"generating {args.games} {args.expert} self-play games...")
        ex = build_dataset(cfg, args.games, args.seed, args.expert)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ex=np.array(ex, dtype=object))
        print(f"saved {len(ex)} examples -> {cache}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ex))
    n_val = max(1, len(ex) // 10)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    train = [ex[i] for i in train_idx]
    val = [ex[i] for i in val_idx]
    print(f"train={len(train)} val={len(val)}")

    model = OrbitNet(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"OrbitNet params: {n_params}")

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        tot = 0.0
        nb = 0
        for s in range(0, len(train), args.batch):
            idx = order[s:s + args.batch]
            b = make_batch(train, idx, device)
            loss = macro_bc_loss(model, b, args.launch_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        if ep % 2 == 0 or ep == args.epochs - 1:
            acc, lacc, hacc, nl, nh = selection_accuracy(model, val, device)
            print(f"epoch {ep:3d}  loss={tot/nb:.4f}  val acc={acc:.3f}  "
                  f"launch_acc={lacc:.3f}(n={nl})  hold_acc={hacc:.3f}(n={nh})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": args.config}, out)
    print(f"saved checkpoint -> {out}")


if __name__ == "__main__":
    main()
