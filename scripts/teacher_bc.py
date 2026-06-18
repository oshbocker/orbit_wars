"""Stage 1: behavior-clone a NON-producer replay teacher's target SELECTION (pointer BC).

Builds the macro-action dataset from a real ladder teacher's CACHED replays (downloaded by
``scripts/harvest_teacher.py`` / ``replay_pulse.py`` into ``outputs/replay_pulse/cache/<date>``)
and trains a v2 OrbitNet selector, reusing the exact ``scripts/macro_bc_train`` machinery
(class-weighted CE on source->target/hold; NO fraction term — sizing is analytic at
execution in ``agents/external/bc_teacher.py``). The teacher is one or more team names; the
seat is matched per episode via ``info.TeamNames``.

    uv run python scripts/teacher_bc.py --date 2026-06-12 \
        --teams "🛰️ Low-Orbit Losers;Erfan Eshratifar;YumeNeko" \
        --epochs 50 --out outputs/checkpoints/bc_teacher/ckpt.pt

The label per (decision-obs, owned source) is the EXACT fleet-tracked destination planet
(``macro_relabel``), with the t-1 alignment (action at steps[t] decided on obs at
steps[t-1]). Holds (no launch) are supervised too. See ``rl_research/TOP_TIER_REPLAY_CORPUS.md``.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.macro_bc_train import (  # noqa: E402
    macro_bc_loss,
    make_batch,
    selection_accuracy,
)
from scripts.macro_relabel import resolve_launches_for_step  # noqa: E402
from src.game_types import parse_observation  # noqa: E402
from v2.config import load_v2_config  # noqa: E402
from v2.features import encode_features  # noqa: E402
from v2.model import OrbitNet  # noqa: E402

CACHE = ROOT / "outputs" / "replay_pulse" / "cache"


def _comet_args(obs):
    cids = obs.get("comet_planet_ids")
    cdata = obs.get("comets")
    cids = [int(x) for x in cids] if cids is not None else None
    return cids, cdata


def label_seat(steps, seat: int, cfg, P: int) -> list[dict]:
    """Per-decision-step examples for one teacher seat (ACTIVE turns only)."""
    ex: list[dict] = []
    for t in range(1, len(steps)):
        if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
            break
        prev = steps[t - 1][seat]
        if prev.get("status") != "ACTIVE":
            continue
        dec = prev.get("observation")
        if not dec or "planets" not in dec:
            continue
        mls = resolve_launches_for_step(steps, t, seat)
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
                if 0 <= dst < P and dst != src_id and bool(feats.planet_mask[dst]):
                    ti[src_id] = dst + 1
                else:
                    sup[src_id] = False
            elif src_id in launched_src:
                sup[src_id] = False  # launched but all fleets died -> no clean label
            # else: genuine hold (ti stays 0)
        ex.append({
            "pf": feats.planet_features, "gf": feats.global_features,
            "pm": feats.planet_mask, "om": feats.own_mask,
            "rm": feats.reachability_mask, "ti": ti, "sup": sup,
        })
    return ex


def build_dataset(cfg, date: str, teams: set[str]) -> list[dict]:
    P = cfg.env.max_planets
    files = sorted(glob.glob(str(CACHE / date / "*.json")))
    if not files:
        raise SystemExit(f"no cached replays at {CACHE / date} — run harvest_teacher.py first")
    ex: list[dict] = []
    n_games = matched_seats = 0
    for fp in files:
        try:
            rep = json.load(open(fp))
        except Exception:  # noqa: BLE001
            continue
        names = rep.get("info", {}).get("TeamNames", [])
        steps = rep.get("steps")
        if not steps:
            continue
        n_games += 1
        for seat in range(len(steps[0])):
            team = names[seat] if seat < len(names) else None
            if team not in teams:
                continue
            seat_ex = label_seat(steps, seat, cfg, P)
            if seat_ex:
                matched_seats += 1
                ex.extend(seat_ex)
    print(f"scanned {n_games} episodes, matched {matched_seats} teacher seats -> {len(ex)} examples")
    return ex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_exit.yaml")
    ap.add_argument("--date", default="2026-06-12")
    ap.add_argument("--teams", required=True, help="';'-separated exact team names to clone")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--launch_weight", type=float, default=5.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--cache", default="outputs/teacher_bc/dataset.npz")
    ap.add_argument("--out", default="outputs/checkpoints/bc_teacher/ckpt.pt")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = load_v2_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teams = {t.strip() for t in args.teams.split(";") if t.strip()}
    print(f"teacher teams: {sorted(teams)}")

    cache = Path(args.cache)
    if cache.exists() and not args.rebuild:
        print(f"loading cached dataset {cache}")
        ex = list(np.load(cache, allow_pickle=True)["ex"])
    else:
        ex = build_dataset(cfg, args.date, teams)
        if not ex:
            raise SystemExit("no examples — check team names match info.TeamNames exactly")
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, ex=np.array(ex, dtype=object))
        print(f"saved {len(ex)} examples -> {cache}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ex))
    n_val = max(1, len(ex) // 10)
    val = [ex[i] for i in perm[:n_val]]
    train = [ex[i] for i in perm[n_val:]]
    print(f"train={len(train)} val={len(val)}")

    model = OrbitNet(cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    print(f"OrbitNet params: {sum(p.numel() for p in model.parameters())}")

    best = -1.0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        tot = nb = 0.0
        for s in range(0, len(train), args.batch):
            b = make_batch(train, order[s:s + args.batch], device)
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
            # checkpoint on best launch accuracy (the part that matters for play)
            if lacc == lacc and lacc > best:
                best = lacc
                torch.save({"model": model.state_dict(), "config": args.config}, out)
    print(f"saved best (launch_acc={best:.3f}) -> {out}")


if __name__ == "__main__":
    main()
