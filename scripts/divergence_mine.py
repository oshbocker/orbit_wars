"""Counterfactual divergence mine: where do top-tier producer-CLONES (LB 1500+)
systematically act differently from bare `producer` / our `v5` on the IDENTICAL
observation they saw?

Pipeline (see memory/clone-residual-divergence-mine.md):
  1. For each replay seat belonging to a target team, walk the obs sequence.
  2. Counterfactual: feed the SAME obs sequence (in order, rolling) through a
     fresh `producer` and `v5` runtime and capture what each WOULD launch.
  3. Resolve every launch's target planet ANALYTICALLY with producer's own
     swept-collision physics, applied identically to actual + counterfactual, so
     resolver bias cancels in the diff.
  4. Classify the seat: clone (high action-match to producer) vs different.
  5. For clone seats, accumulate per-axis deltas (actual - cf) conditioned on
     state class (phase / contested / ahead-behind / 2P-4P).

Outputs a ranked hypothesis table. DISCOVERY ONLY — nothing shipped. Candidates
gate later at n>=100 mirror A/B (margin+steps metrics; win-rate ceilings ~99%).

  uv run python scripts/divergence_mine.py --replays '/tmp/ow_top/*.json' \
      --top-k 10 --out rl_research/CLONE_RESIDUAL_FINDINGS.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import statistics as st
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agents" / "external" / "producer"))

import torch  # noqa: E402

from agents import load_named_agent  # noqa: E402
from orbit_lite.adapter import single_obs_to_tensor  # noqa: E402
from orbit_lite.movement import (  # noqa: E402
    MovementConfig,
    PlanetMovement,
    _estimate_new_fleet_arrivals,
)

RES_H = 40  # target-resolution lookahead horizon


# ─── target resolution (producer physics, used for actual AND counterfactual) ──
def build_movement(obs: dict, player_id: int):
    """One movement cache per (obs) — shared by the actual/producer/v5 resolves."""
    ot = single_obs_to_tensor(obs, player_id=player_id)
    return PlanetMovement.from_obs_tensors(
        ot, config=MovementConfig(movement_horizon=RES_H, track_fleets=False)
    )


def resolve_targets(obs: dict, action: list, player_id: int, mv=None) -> list[int]:
    """Planet id each launch's straight ray first collides with (-1 = sun/edge/miss)."""
    if not action:
        return []
    if mv is None:
        mv = build_movement(obs, player_id)
    planets = {int(p[0]): p for p in obs["planets"]}
    L = len(action)
    rows = torch.full((L, 7), -1.0, dtype=mv.dtype)
    for i, m in enumerate(action):
        src, ang, sh = int(m[0]), float(m[1]), float(m[2])
        sp = planets.get(src)
        if sp is None:
            continue
        cx, cy, rad = float(sp[2]), float(sp[3]), float(sp[4])
        rows[i, 1] = float(player_id)
        rows[i, 2] = cx + math.cos(ang) * (rad + 0.1)
        rows[i, 3] = cy + math.sin(ang) * (rad + 0.1)
        rows[i, 4] = ang
        rows[i, 6] = sh
    est = _estimate_new_fleet_arrivals(movement=mv, obs_fleets=rows, fleet_slot=torch.arange(L))
    pid = mv.planet_ids.tolist()
    return [int(pid[int(est["target_slot"][i])]) if bool(est["has_hit"][i]) else -1 for i in range(L)]


# ─── target-feature helpers (for the disagreement profiler) ───────────────────
def inbound_owners(obs: dict, mv) -> dict:
    """planet_id -> set of owners with an IN-FLIGHT fleet resolving to it.

    Uses the SAME swept-collision resolver as launches, so contested-ness is
    measured with producer's own physics (resolver bias cancels)."""
    fleets = obs.get("fleets", []) or []
    res = defaultdict(set)
    if not fleets:
        return res
    L = len(fleets)
    rows = torch.full((L, 7), -1.0, dtype=mv.dtype)
    for i, fl in enumerate(fleets):
        rows[i, 1] = float(fl[1])   # owner
        rows[i, 2] = float(fl[2])   # x
        rows[i, 3] = float(fl[3])   # y
        rows[i, 4] = float(fl[4])   # angle
        rows[i, 6] = float(fl[6])   # ships
    est = _estimate_new_fleet_arrivals(movement=mv, obs_fleets=rows, fleet_slot=torch.arange(L))
    pid = mv.planet_ids.tolist()
    for i in range(L):
        if bool(est["has_hit"][i]):
            res[int(pid[int(est["target_slot"][i])])].add(int(fleets[i][1]))
    return res


def orbiting_set(obs: dict) -> set:
    """ids of orbiting (rotating, inner) planets — mirrors orbit_lite/obs.py:
    alive & (orb_r + radius < 50) & (orb_r > 0.5), reconstructed from initial pos."""
    init = obs.get("initial_planets") or obs.get("planets") or []
    s = set()
    for row in init:
        orb_r = math.hypot(float(row[2]) - 50.0, float(row[3]) - 50.0)
        if orb_r > 0.5 and (orb_r + float(row[4])) < 50.0:
            s.add(int(row[0]))
    return s


def src_target_map(obs: dict, action: list, player_id: int, mv=None) -> dict:
    """src_planet_id -> dominant target id (the target of that source's
    largest-ship launch). Resolved with the shared per-turn mv cache so the
    map is comparable across actual / counterfactual."""
    if not action:
        return {}
    planets = {int(p[0]): p for p in obs["planets"]}
    tgts = resolve_targets(obs, action, player_id, mv=mv)
    best = {}  # src -> (ships, tgt)
    for m, tgt in zip(action, tgts):
        src, sh = int(m[0]), int(m[2])
        if src not in planets or sh <= 0:
            continue
        if src not in best or sh > best[src][0]:
            best[src] = (sh, tgt)
    return {s: t for s, (_sh, t) in best.items()}


def target_feature_vec(obs: dict, src_id: int, tgt_id: int, player_id: int,
                       inbound: dict, orbiting: set) -> dict | None:
    """Numeric feature vector of `tgt_id` relative to `src_id` (None if either
    planet is gone, or the launch missed all planets, tgt_id == -1)."""
    planets = {int(p[0]): p for p in obs["planets"]}
    sp, tp = planets.get(src_id), planets.get(tgt_id)
    if sp is None or tp is None:
        return None
    owner = int(tp[1])
    return {
        "dist": math.hypot(float(sp[2]) - float(tp[2]), float(sp[3]) - float(tp[3])),
        "prod": float(tp[6]),
        "garrison": float(tp[5]),
        "contested": 1.0 if len(inbound.get(tgt_id, set())) >= 2 else 0.0,
        "orbiting": 1.0 if tgt_id in orbiting else 0.0,
        "is_enemy": 1.0 if (owner >= 0 and owner != player_id) else 0.0,
        "is_neutral": 1.0 if owner == -1 else 0.0,
        "is_own": 1.0 if owner == player_id else 0.0,
    }


TGT_AXES = ["dist", "prod", "garrison", "contested", "orbiting",
            "is_enemy", "is_neutral", "is_own"]


def target_disagreements(obs: dict, actual: list, cf_prod: list, player_id: int,
                         mv, inbound: dict, orbiting: set) -> tuple[int, list]:
    """On shared source planets where actual (clone) and cf_producer launch to
    DIFFERENT targets, return (n_shared_sources, [(clone_feats, prod_feats), ...]).

    Producer-vs-producer (identical actions) → 0 disagreements (self-check)."""
    a_map = src_target_map(obs, actual, player_id, mv=mv)
    p_map = src_target_map(obs, cf_prod, player_id, mv=mv)
    shared = set(a_map) & set(p_map)
    pairs = []
    for src in shared:
        at, pt = a_map[src], p_map[src]
        if at == pt:
            continue
        fa = target_feature_vec(obs, src, at, player_id, inbound, orbiting)
        fp = target_feature_vec(obs, src, pt, player_id, inbound, orbiting)
        if fa is None or fp is None:
            continue
        pairs.append((fa, fp))
    return len(shared), pairs


# ─── per-turn behavioural summary (resolution-free axes + target categories) ───
@dataclass
class TurnStats:
    n_launches: int = 0
    active: int = 0          # 1 if >=1 launch
    fracs: list = field(default_factory=list)    # ships/garrison per launch
    total_ships: int = 0
    n_sources: int = 0
    pairs: set = field(default_factory=set)      # {(src_id, tgt_id)}
    # target categories (resolved)
    n_reinforce: int = 0
    n_attack: int = 0
    n_neutral: int = 0
    tgt_prods: list = field(default_factory=list)
    tgt_dists: list = field(default_factory=list)


def summarize(obs: dict, action: list, player_id: int, mv=None) -> TurnStats:
    ts = TurnStats()
    if not action:
        return ts
    planets = {int(p[0]): p for p in obs["planets"]}
    tgts = resolve_targets(obs, action, player_id, mv=mv)
    srcs = set()
    for m, tgt in zip(action, tgts):
        src, sh = int(m[0]), int(m[2])
        sp = planets.get(src)
        if sp is None or sh <= 0:
            continue
        ts.n_launches += 1
        ts.total_ships += sh
        srcs.add(src)
        gar = int(sp[5])
        ts.fracs.append(sh / max(gar, 1))
        ts.pairs.add((src, tgt))
        if tgt >= 0:
            tp = planets.get(tgt)
            if tp is not None:
                owner = int(tp[1])
                if owner == player_id:
                    ts.n_reinforce += 1
                elif owner == -1:
                    ts.n_neutral += 1
                else:
                    ts.n_attack += 1
                ts.tgt_prods.append(int(tp[6]))
                ts.tgt_dists.append(
                    math.hypot(float(sp[2]) - float(tp[2]), float(sp[3]) - float(tp[3]))
                )
    ts.n_sources = len(srcs)
    ts.active = 1 if ts.n_launches > 0 else 0
    return ts


# ─── state classification ──────────────────────────────────────────────────
def classify_state(obs: dict, player_id: int, n_players: int, step: int) -> dict:
    phase = "open" if step < 60 else ("mid" if step < 350 else "end")
    # scores: ships on owned planets + ships in owned fleets, per owner
    score = defaultdict(float)
    owners_present = set()
    for p in obs["planets"]:
        o = int(p[1])
        if o >= 0:
            score[o] += float(p[5])
            owners_present.add(o)
    for f in obs.get("fleets", []):
        o = int(f[1])
        if o >= 0:
            score[o] += float(f[6])
    me = score.get(player_id, 0.0)
    opp = [v for k, v in score.items() if k != player_id]
    best_opp = max(opp) if opp else 0.0
    standing = "ahead" if me >= best_opp else "behind"
    # contested = >=2 distinct ENEMY owners still on the board
    n_enemy = len([o for o in owners_present if o != player_id])
    contested = "contested" if n_enemy >= 2 else "uncontested"
    fmt = "4P" if n_players >= 4 else "2P"
    return {"phase": phase, "standing": standing, "contested": contested, "fmt": fmt}


# ─── counterfactual rollout ──────────────────────────────────────────────────
def cf_rollout(steps: list, seat: int, agent_name: str) -> dict:
    """Roll a fresh agent through the seat's contiguous ACTIVE obs prefix.

    Returns {t: cf_action}. Feeds obs in order so the agent's rolling movement
    cache stays consistent (the rich-rep gotcha)."""
    ag = load_named_agent(agent_name)
    out = {}
    for t in range(1, len(steps)):
        if seat >= len(steps[t - 1]):
            break
        prev = steps[t - 1][seat]
        if prev.get("status") != "ACTIVE":
            break  # stop at first inactive — keep cache contiguous
        obs = prev.get("observation")
        if not obs or "planets" not in obs:
            break
        try:
            out[t] = ag(obs) or []
        except Exception as e:  # noqa: BLE001
            print(f"    cf {agent_name} crashed at t={t}: {e}", file=sys.stderr)
            break
    return out


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


# ─── per-seat analysis ───────────────────────────────────────────────────────
def analyze_seat(steps: list, seat: int, n_players: int) -> list[dict]:
    """One record per ACTIVE decision turn with actual + cf_producer + cf_v5 stats."""
    cf_prod = cf_rollout(steps, seat, "producer")
    cf_v5 = cf_rollout(steps, seat, "v5")
    recs = []
    for t in range(1, len(steps)):
        if t not in cf_prod or t not in cf_v5:
            continue
        prev = steps[t - 1][seat]
        obs = prev["observation"]
        actual = steps[t][seat].get("action") or []
        cls = classify_state(obs, seat, n_players, step=t - 1)
        # build the resolver movement ONCE per turn; reuse for all three actions
        mv = build_movement(obs, seat) if (actual or cf_prod[t] or cf_v5[t]) else None
        a = summarize(obs, actual, seat, mv=mv)
        p = summarize(obs, cf_prod[t], seat, mv=mv)
        v = summarize(obs, cf_v5[t], seat, mv=mv)
        # target-disagreement profile: shared sources, different targets (vs producer)
        n_shared, tgt_disagree = 0, []
        if mv is not None and (actual and cf_prod[t]):
            inbound = inbound_owners(obs, mv)
            orbiting = orbiting_set(obs)
            n_shared, tgt_disagree = target_disagreements(
                obs, actual, cf_prod[t], seat, mv, inbound, orbiting)
        recs.append({"cls": cls, "a": a, "p": p, "v": v,
                     "n_shared": n_shared, "tgt_disagree": tgt_disagree})
    return recs


# ─── axis extraction from a TurnStats ────────────────────────────────────────
def axes(ts: TurnStats) -> dict:
    nl = max(ts.n_launches, 1)
    return {
        "waves": ts.n_launches,
        "active": ts.active,
        "ships": ts.total_ships,
        "sources": ts.n_sources,
        "med_frac": st.median(ts.fracs) if ts.fracs else None,
        "mean_frac": (sum(ts.fracs) / len(ts.fracs)) if ts.fracs else None,
        "full_rate": (sum(1 for f in ts.fracs if f > 0.85) / len(ts.fracs)) if ts.fracs else None,
        "reinforce_rate": ts.n_reinforce / nl if ts.n_launches else None,
        "attack_rate": ts.n_attack / nl if ts.n_launches else None,
        "neutral_rate": ts.n_neutral / nl if ts.n_launches else None,
        "tgt_prod": (sum(ts.tgt_prods) / len(ts.tgt_prods)) if ts.tgt_prods else None,
        "tgt_dist": (sum(ts.tgt_dists) / len(ts.tgt_dists)) if ts.tgt_dists else None,
    }


AXES = ["waves", "active", "ships", "sources", "med_frac", "full_rate",
        "reinforce_rate", "attack_rate", "neutral_rate", "tgt_prod", "tgt_dist"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replays", required=True, help="glob of replay json files")
    ap.add_argument("--lb-csv", default=None, help="leaderboard csv for team ratings (latest cache if omitted)")
    ap.add_argument("--top-k", type=int, default=10, help="analyze seats whose team is rank <= top-k")
    ap.add_argument("--teams", default=None, help="comma-separated explicit team names (overrides top-k)")
    ap.add_argument("--posture-frac", type=float, default=0.9,
                    help="seat is a POSTURE-clone if its median send-fraction >= this (full-drain)")
    ap.add_argument("--wps-thr", type=float, default=1.4,
                    help="...AND mean launches-per-source <= this (no swarm/multi-wave)")
    ap.add_argument("--out", default="rl_research/CLONE_RESIDUAL_FINDINGS.md")
    ap.add_argument("--max-games", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    # ratings
    rank = {}
    lb = args.lb_csv
    if lb is None:
        cands = sorted(glob.glob(str(REPO / "outputs/replay_pulse/cache/lb/*.csv")))
        lb = cands[-1] if cands else None
    if lb:
        with open(lb, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                try:
                    rank[r["TeamName"]] = (int(r["Rank"]), float(r["Score"]))
                except (ValueError, KeyError):
                    pass
    explicit = set(s.strip() for s in args.teams.split(",")) if args.teams else None

    def is_target(team):
        if explicit is not None:
            return team in explicit
        rk = rank.get(team)
        return rk is not None and rk[0] <= args.top_k

    files = sorted(glob.glob(args.replays))
    if args.max_games:
        files = files[: args.max_games]
    print(f"{len(files)} replays; targets = {'explicit' if explicit else f'top-{args.top_k}'}")

    seat_match = []      # per-seat fingerprints + posture-clone flag
    deltas = defaultdict(list)  # (vs, axis, clskey) -> [(delta, baseline)] behavioural axes
    sel = defaultdict(list)     # (vs, metric, clskey) -> [value] selection-divergence
    tgt_deltas = defaultdict(list)  # (axis, clskey) -> [(clone-prod delta, prod baseline)]
    tgt_counts = defaultdict(lambda: [0, 0])  # clskey -> [n_shared_sources, n_disagreements]
    clone_turns = 0

    def bucket_keys(cls):
        return [
            "all=all", f"phase={cls['phase']}", f"contested={cls['contested']}",
            f"standing={cls['standing']}", f"fmt={cls['fmt']}",
            f"fmt={cls['fmt']}|phase={cls['phase']}",
            f"fmt={cls['fmt']}|standing={cls['standing']}",
            f"fmt={cls['fmt']}|contested={cls['contested']}",
        ]

    for gi, f in enumerate(files):
        try:
            rep = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        steps = rep.get("steps")
        if not steps:
            continue
        names = rep.get("info", {}).get("TeamNames", [])
        n_players = len(steps[0])
        for seat in range(n_players):
            team = names[seat] if seat < len(names) else f"seat{seat}"
            if not is_target(team):
                continue
            recs = analyze_seat(steps, seat, n_players)
            if not recs:
                continue
            # POSTURE fingerprint: median send-fraction + launches-per-source (swarm).
            afr, pfr, wps, jaccs = [], [], [], []
            for r in recs:
                afr += r["a"].fracs
                pfr += r["p"].fracs
                if r["a"].n_sources:
                    wps.append(r["a"].n_launches / r["a"].n_sources)
                if r["a"].n_launches or r["p"].n_launches:
                    jaccs.append(jaccard(r["a"].pairs, r["p"].pairs))
            med_frac = st.median(afr) if afr else 0.0
            mean_wps = (sum(wps) / len(wps)) if wps else 1.0
            is_clone = med_frac >= args.posture_frac and mean_wps <= args.wps_thr
            rk = rank.get(team)
            seat_match.append({
                "team": team, "rank": rk[0] if rk else None, "rtg": rk[1] if rk else None,
                "game": Path(f).stem, "n_turns": len(recs),
                "med_frac": med_frac, "med_frac_prod": st.median(pfr) if pfr else None,
                "wps": mean_wps, "jacc_prod": (sum(jaccs) / len(jaccs)) if jaccs else 1.0,
                "is_clone": is_clone, "fmt": recs[0]["cls"]["fmt"],
            })
            if not is_clone:
                continue
            for r in recs:
                cls = r["cls"]
                keys = bucket_keys(cls)
                clone_turns += 1
                # (1) behavioural axis deltas (actual - counterfactual)
                ax_a = axes(r["a"])
                for vs, ts in (("prod", r["p"]), ("v5", r["v"])):
                    ax_c = axes(ts)
                    for ax in AXES:
                        va, vc = ax_a[ax], ax_c[ax]
                        if va is None or vc is None:
                            continue
                        entry = (va - vc, vc)
                        for k in keys:
                            deltas[(vs, ax, k)].append(entry)
                # (2) selection-divergence: precision/recall of source & target choices
                a_pairs, a_src = r["a"].pairs, {s for s, _ in r["a"].pairs}
                for vs, ts in (("prod", r["p"]), ("v5", r["v"])):
                    b_pairs, b_src = ts.pairs, {s for s, _ in ts.pairs}
                    metrics = {}
                    if a_pairs:
                        metrics["pair_prec"] = len(a_pairs & b_pairs) / len(a_pairs)
                        metrics["src_prec"] = len(a_src & b_src) / len(a_src)
                    if b_pairs:
                        metrics["pair_rec"] = len(a_pairs & b_pairs) / len(b_pairs)
                        metrics["src_rec"] = len(a_src & b_src) / len(b_src)
                    shared = a_src & b_src
                    if shared:
                        at = {s: t for s, t in a_pairs if s in shared}
                        bt = {s: t for s, t in b_pairs if s in shared}
                        metrics["tgt_match|src"] = sum(1 for s in shared if at.get(s) == bt.get(s)) / len(shared)
                    for mname, mval in metrics.items():
                        for k in keys:
                            sel[(vs, mname, k)].append(mval)
                # (3) target-disagreement profile: clone target − producer target,
                # on shared sources where they aim DIFFERENTLY (confound-free).
                for k in keys:
                    tgt_counts[k][0] += r["n_shared"]
                    tgt_counts[k][1] += len(r["tgt_disagree"])
                for fa, fp in r["tgt_disagree"]:
                    for ax in TGT_AXES:
                        entry = (fa[ax] - fp[ax], fp[ax])
                        for k in keys:
                            tgt_deltas[(ax, k)].append(entry)
        if (gi + 1) % 20 == 0:
            print(f"  {gi+1}/{len(files)} games, {len(seat_match)} target seats, {clone_turns} clone turns")

    render(args, seat_match, deltas, sel, tgt_deltas, tgt_counts, clone_turns)


def render(args, seat_match, deltas, sel, tgt_deltas, tgt_counts, clone_turns):
    clones = [s for s in seat_match if s["is_clone"]]
    n_clone_seats = len(clones)
    print(f"\n=== {len(seat_match)} target seats, {n_clone_seats} classified POSTURE-CLONE "
          f"(med frac>={args.posture_frac}, wps<={args.wps_thr}), {clone_turns} clone turns ===")

    # rank findings by statistical robustness: |t-stat| gated by practical magnitude.
    # t-stat = mean_d / (sd/sqrt(n)) — high = systematic, not noise. pct = mean_d vs
    # the baseline's own mean = practical lever size. eff = |mean|/sd (effect size).
    rows = []
    for (vs, ax, clskey), arr in deltas.items():
        n = len(arr)
        if n < 60:
            continue
        ds = [e[0] for e in arr]
        base = [e[1] for e in arr]
        m = sum(ds) / n
        if m == 0:
            continue
        sd = st.pstdev(ds) if n > 1 else 0.0
        base_mean = sum(base) / n
        tstat = m / (sd / math.sqrt(n) + 1e-12)
        eff = abs(m) / (sd + 1e-9)
        sign = 1 if m > 0 else -1
        consistency = sum(1 for x in ds if (x > 0) == (sign > 0) and x != 0) / n
        nonzero = sum(1 for x in ds if x != 0) / n
        pct = (m / base_mean * 100.0) if abs(base_mean) > 1e-9 else float("nan")
        rows.append({
            "vs": vs, "axis": ax, "cls": clskey, "n": n, "mean_d": m,
            "base": base_mean, "pct": pct, "tstat": tstat, "eff": eff,
            "consistency": consistency, "nonzero": nonzero,
        })
    # require both statistical significance and that a non-trivial share of turns
    # actually exercise the axis (nonzero), so we don't surface tiny always-on biases.
    rows = [r for r in rows if abs(r["tstat"]) >= 3.0 and r["nonzero"] >= 0.15]
    rows.sort(key=lambda r: abs(r["tstat"]) * (0.2 + r["eff"]), reverse=True)

    # selection-divergence table: mean of each precision/recall metric by state class.
    # LOW value = clone's source/target choices diverge from the baseline on the same obs.
    sel_rows = []
    for (vs, metric, clskey), arr in sel.items():
        n = len(arr)
        if n < 60:
            continue
        m = sum(arr) / n
        sd = st.pstdev(arr) if n > 1 else 0.0
        sel_rows.append({"vs": vs, "metric": metric, "cls": clskey, "n": n, "mean": m, "sd": sd})
    sel_rows.sort(key=lambda r: (r["vs"], r["metric"], r["cls"]))

    # target-disagreement profile: clone target − producer target on shared
    # sources where they aim differently. Same gating style as `rows`.
    tgt_rows = []
    for (ax, clskey), arr in tgt_deltas.items():
        n = len(arr)
        if n < 40:  # disagreements are rarer than turns; relax the floor a touch
            continue
        ds = [e[0] for e in arr]
        base = [e[1] for e in arr]
        m = sum(ds) / n
        if m == 0:
            continue
        sd = st.pstdev(ds) if n > 1 else 0.0
        base_mean = sum(base) / n
        tstat = m / (sd / math.sqrt(n) + 1e-12)
        eff = abs(m) / (sd + 1e-9)
        sign = 1 if m > 0 else -1
        consistency = sum(1 for x in ds if (x > 0) == (sign > 0) and x != 0) / n
        nonzero = sum(1 for x in ds if x != 0) / n
        pct = (m / base_mean * 100.0) if abs(base_mean) > 1e-9 else float("nan")
        tgt_rows.append({
            "axis": ax, "cls": clskey, "n": n, "mean_d": m, "base": base_mean,
            "pct": pct, "tstat": tstat, "eff": eff, "consistency": consistency,
            "nonzero": nonzero,
        })
    tgt_rows = [r for r in tgt_rows if abs(r["tstat"]) >= 3.0]
    tgt_rows.sort(key=lambda r: abs(r["tstat"]) * (0.2 + r["eff"]), reverse=True)

    lines = []
    lines.append("# Clone-Residual Divergence Mine — Findings\n")
    lines.append("_Counterfactual action-diff of top-tier producer-family agents vs bare "
                 "`producer` and our `v5`, on the IDENTICAL observation each agent saw. "
                 "Discovery only — gate at n>=100 mirror A/B before shipping._\n")
    lines.append(f"- Target seats analyzed: **{len(seat_match)}**; classified **posture-clone** "
                 f"(median send-fraction >= {args.posture_frac} AND launches/source <= "
                 f"{args.wps_thr}): **{n_clone_seats}**; clone decision-turns: **{clone_turns}**.\n")
    lines.append("- **Headline:** the top tier is producer-family in POSTURE (full-drain, ~1 "
                 "wave/source) yet its (source,target) SELECTION diverges sharply from bare "
                 "producer — the residual is in WHAT to attack and FROM WHERE, not how much "
                 "to send. Tables below quantify where/how.\n")

    # roster
    lines.append("\n## Seat roster\n")
    lines.append("`Jaccard` = mean per-turn (src,tgt) overlap with producer's counterfactual "
                 "(LOW even for full-drain clones = the residual). `wps` = launches per source.\n")
    lines.append("| team | rank | rtg | fmt | turns | med frac | wps | Jaccard vs prod | posture-clone? |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in sorted(seat_match, key=lambda x: (x["rank"] or 9999, x["game"])):
        lines.append(f"| {s['team'][:24]} | {s['rank']} | {s['rtg']:.0f} | {s['fmt']} | "
                     f"{s['n_turns']} | {s['med_frac']:.2f} | {s['wps']:.2f} | "
                     f"{s['jacc_prod']:.2f} | {'✅' if s['is_clone'] else '—'} |")

    # selection-divergence
    lines.append("\n## Selection divergence (clone vs baseline, same obs)\n")
    lines.append("`pair_prec` = of the clone's launches, fraction producer also makes (low = "
                 "clone makes launches producer wouldn't = different/extra targets). "
                 "`pair_rec` = of producer's launches, fraction the clone also makes (low = clone "
                 "SKIPS launches producer makes). `src_*` = same at source-planet granularity. "
                 "`tgt_match|src` = on shared source planets, fraction with the SAME target. "
                 "1.0 = identical to producer; lower = more divergent.\n")
    lines.append("| vs | metric | state class | n | mean | sd |")
    lines.append("|---|---|---|---|---|---|")
    order = {"pair_prec": 0, "pair_rec": 1, "src_prec": 2, "src_rec": 3, "tgt_match|src": 4}
    for r in sorted(sel_rows, key=lambda r: (r["vs"], order.get(r["metric"], 9), r["cls"])):
        lines.append(f"| {r['vs']} | {r['metric']} | {r['cls']} | {r['n']} | "
                     f"{r['mean']:.3f} | {r['sd']:.3f} |")

    # ranked behavioural hypotheses
    lines.append("\n## Ranked behavioural divergences (clone − baseline)\n")
    lines.append("Positive `Δ` = clone does MORE of the axis than the baseline on the SAME obs. "
                 "`base` = baseline's own mean (for scale); `pct` = Δ as % of base; "
                 "`t` = t-stat (|t|>=3 kept); `eff` = |Δ|/sd; `cons` = sign-agreement; "
                 "`nz` = fraction of turns exercising the axis. Ranked by |t|·eff.\n")
    lines.append("| # | vs | axis | state class | n | Δ | base | pct | t | eff | cons | nz |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows[:60], 1):
        pct = f"{r['pct']:+.0f}%" if r["pct"] == r["pct"] else "—"
        lines.append(f"| {i} | {r['vs']} | {r['axis']} | {r['cls']} | {r['n']} | "
                     f"{r['mean_d']:+.3f} | {r['base']:.2f} | {pct} | {r['tstat']:+.1f} | "
                     f"{r['eff']:.2f} | {r['consistency']:.2f} | {r['nonzero']:.2f} |")

    # target-disagreement profile (Plan 1 — the confound-free residual)
    n_shared_all = tgt_counts.get("all=all", [0, 0])[0]
    n_disagree_all = tgt_counts.get("all=all", [0, 0])[1]
    disagree_rate = (n_disagree_all / n_shared_all) if n_shared_all else float("nan")
    lines.append("\n## Target-disagreement profile (clone target − producer target)\n")
    lines.append("_Plan 1, the confound-free residual: on the **shared source planets** where "
                 "the clone and bare `producer` BOTH launch but resolve to **different targets**, "
                 "what is systematically different about the target the clone picks vs the one "
                 "producer's flow-diff argmax picks?_\n")
    lines.append(f"- Shared-source decisions: **{n_shared_all}**; of which the clone aimed "
                 f"elsewhere than producer: **{n_disagree_all}** "
                 f"(**{disagree_rate*100:.0f}%** disagreement rate — matches `tgt_match|src`).\n")
    lines.append("Each row = a target-feature axis conditioned on a state class. `Δ` = "
                 "mean(clone_target_feature − producer_target_feature) over disagreements "
                 "(positive = clone's chosen target scores HIGHER on the axis). `base` = "
                 "producer-target mean (scale); `pct` = Δ as % of base; `t` = t-stat (|t|>=3 "
                 "kept); `eff` = |Δ|/sd; `cons` = sign-agreement; `nz` = fraction of "
                 "disagreements where the axis differs. Axes: `dist` (source→target euclidean), "
                 "`prod`, `garrison`, `contested` (>=2 owners inbound), `orbiting`, and "
                 "owner-class indicators `is_enemy`/`is_neutral`/`is_own`. Ranked by |t|·eff.\n")
    lines.append("| # | axis | state class | n | Δ | base | pct | t | eff | cons | nz |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(tgt_rows[:60], 1):
        pct = f"{r['pct']:+.0f}%" if r["pct"] == r["pct"] else "—"
        lines.append(f"| {i} | {r['axis']} | {r['cls']} | {r['n']} | "
                     f"{r['mean_d']:+.3f} | {r['base']:.2f} | {pct} | {r['tstat']:+.1f} | "
                     f"{r['eff']:.2f} | {r['consistency']:.2f} | {r['nonzero']:.2f} |")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    # also dump raw rows for the other thread
    (out.with_suffix(".json")).write_text(json.dumps(
        {"seat_match": seat_match, "rankings": rows, "selection": sel_rows,
         "target_disagreement": tgt_rows,
         "target_disagreement_counts": {k: v for k, v in tgt_counts.items()}}, indent=2))
    print(f"\nwrote {out}")
    print("\nTop 20 divergences (|t|>=3):")
    for i, r in enumerate(rows[:20], 1):
        pct = f"{r['pct']:+.0f}%" if r["pct"] == r["pct"] else "—"
        print(f"  {i:>2}. vs {r['vs']:<4} {r['axis']:<14} {r['cls']:<30} "
              f"n={r['n']:<5} Δ={r['mean_d']:+.3f} ({pct:>6}) t={r['tstat']:+.1f} eff={r['eff']:.2f}")

    print(f"\nTarget-disagreement profile ({n_disagree_all}/{n_shared_all} shared sources "
          f"= {disagree_rate*100:.0f}% disagree). Top 20 (|t|>=3):")
    for i, r in enumerate(tgt_rows[:20], 1):
        pct = f"{r['pct']:+.0f}%" if r["pct"] == r["pct"] else "—"
        print(f"  {i:>2}. {r['axis']:<11} {r['cls']:<30} "
              f"n={r['n']:<5} Δ={r['mean_d']:+.3f} ({pct:>6}) t={r['tstat']:+.1f} eff={r['eff']:.2f}")


if __name__ == "__main__":
    main()
