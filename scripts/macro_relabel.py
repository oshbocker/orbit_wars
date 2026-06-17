"""Exact macro-action relabeling for the pointer-BC pipeline (Step 0 substrate).

For each launch an agent makes, recover the EXACT destination planet by tracking
the launched fleet through the full-state replay until it lands — NO angle matching,
NO fraction bucketing. The supervised label per (state, owned-source) becomes
``(target_planet_id, posture)`` where posture is the send-fraction bucket; the exact
ship count + intercept angle are recomputed analytically at execution time, so the
clone never regresses a continuous parameter (the thing that capped prior BC at 3%).

This module is run as a script to validate destination-resolution QUALITY on
producer self-play before any training is built on top of it.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class MacroLaunch:
    src_id: int
    dst_id: int          # resolved destination planet id (-1 = died: sun/edge/unresolved)
    ships: int
    src_garrison: int    # source garrison at decision time (for send-fraction)
    angle: float
    eta: int             # steps in flight until resolution
    dst_was_owned: bool  # destination owned by the launcher at decision time (reinforce)


def _obs(step_entry):
    return step_entry["observation"] if not hasattr(step_entry, "observation") else step_entry.observation


def _action(step_entry):
    a = step_entry["action"] if not hasattr(step_entry, "action") else step_entry.action
    return a or []


def _planets_by_id(obs):
    return {int(p[0]): p for p in obs["planets"]}


def _fleets_by_id(obs):
    return {int(f[0]): f for f in obs["fleets"]}


def resolve_launches_for_step(steps, t: int, player: int) -> list[MacroLaunch]:
    """Resolve every launch player `player` made; `t` is the ACTION-STORAGE index.

    Kaggle alignment (verified): the action stored at ``steps[t].action`` was
    decided on the observation at ``steps[t-1]`` and the launched fleet appears in
    the state at ``steps[t]`` with id in ``[next_fleet_id(t-1), next_fleet_id(t))``.
    We read the source garrison from the DECISION obs (t-1), match each action row
    to its new fleet by (from_planet, ships), then follow the fleet id forward
    until it lands; destination = planet nearest its last position at disappearance.
    Returns [] if the action was empty.
    """
    if t < 1:
        return []
    actions = _action(steps[t][player])
    if not actions:
        return []

    dec_obs = _obs(steps[t - 1][player])          # decision observation
    planets_dec = _planets_by_id(dec_obs)
    cur_fleets = _fleets_by_id(_obs(steps[t][player]))   # new fleet appears here
    nfi = int(dec_obs["next_fleet_id"])
    new_fleets = {
        fid: f for fid, f in cur_fleets.items()
        if fid >= nfi and int(f[1]) == player
    }

    out: list[MacroLaunch] = []
    used: set[int] = set()
    for mv in actions:
        src_id, angle, ships = int(mv[0]), float(mv[1]), int(mv[2])
        fid = None
        for cand, f in new_fleets.items():
            if cand in used:
                continue
            if int(f[5]) == src_id and int(f[6]) == ships:
                fid = cand
                break
        if fid is None:  # fallback: match on source only
            for cand, f in new_fleets.items():
                if cand in used:
                    continue
                if int(f[5]) == src_id:
                    fid = cand
                    break
        src_p = planets_dec.get(src_id)
        src_garrison = int(src_p[5]) if src_p is not None else ships
        dst_was_owned = False
        dst_id, eta = -1, -1
        if fid is not None:
            used.add(fid)
            dst_id, eta = _track_to_destination(steps, t, player, fid)
            if dst_id >= 0:
                dst_p = planets_dec.get(dst_id)
                dst_was_owned = dst_p is not None and int(dst_p[1]) == player
        out.append(MacroLaunch(src_id, dst_id, ships, src_garrison, angle, eta, dst_was_owned))
    return out


def _track_to_destination(steps, t_start: int, player: int, fid: int) -> tuple[int, int]:
    """Follow fleet `fid` from step t_start until it vanishes; return (dst_planet_id, eta).

    Destination = the planet nearest the fleet's last seen position (at the
    disappearance step's state), within planet_radius + one fleet step. (-1, -1)
    if it dies at the sun/edge (no planet within range) or never resolves.
    """
    last_pos = None
    for k in range(t_start, len(steps)):
        fleets = _fleets_by_id(_obs(steps[k][player]))
        if fid in fleets:
            f = fleets[fid]
            last_pos = (float(f[2]), float(f[3]))
            continue
        # disappeared at step k: resolve against planets at step k (post-arrival state)
        if last_pos is None:
            return -1, -1
        obs_k = _obs(steps[k][player])
        best_id, best_d = -1, 1e9
        for p in obs_k["planets"]:
            px, py, rad = float(p[2]), float(p[3]), float(p[4])
            d = math.hypot(px - last_pos[0], py - last_pos[1])
            slack = rad + 6.5  # max fleet step ~6 units/turn + planet radius
            if d < slack and d < best_d:
                best_d, best_id = d, int(p[0])
        return best_id, (k - t_start + 1)
    return -1, -1  # game ended with fleet still in flight


# ── send-fraction posture bucketing ──────────────────────────────────────────
# capture-min (<=0.4), partial (0.4-0.85), full (>0.85). Reinforcements (own dst)
# are their own posture regardless of fraction.
POSTURES = ["capture_min", "partial", "full", "reinforce"]


def posture_of(ml: MacroLaunch) -> str:
    if ml.dst_was_owned:
        return "reinforce"
    frac = ml.ships / max(ml.src_garrison, 1)
    if frac <= 0.4:
        return "capture_min"
    if frac <= 0.85:
        return "partial"
    return "full"


if __name__ == "__main__":
    import argparse
    from collections import Counter

    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="producer")
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20000)
    args = ap.parse_args()

    from kaggle_environments import make

    from agents import load_named_agent

    tot_launches = resolved = died = unresolved = 0
    posture_counts: Counter = Counter()
    frac_samples: list[float] = []
    for g in range(args.games):
        a1, a2 = load_named_agent(args.agent), load_named_agent(args.agent)
        env = make("orbit_wars", configuration={"randomSeed": args.seed + g})
        env.run([a1, a2])
        steps = env.steps
        for t in range(1, len(steps)):
            for player in (0, 1):
                for ml in resolve_launches_for_step(steps, t, player):
                    tot_launches += 1
                    if ml.dst_id >= 0:
                        resolved += 1
                        posture_counts[posture_of(ml)] += 1
                        frac_samples.append(ml.ships / max(ml.src_garrison, 1))
                    elif ml.eta == -1 and ml.ships > 0:
                        # distinguish died-at-sun/edge (will show -1) from game-end
                        died += 1
                    else:
                        unresolved += 1
        print(f"  game {g}: {len(steps)} steps")

    print(f"\n=== {args.agent} self-play, {args.games} games ===")
    print(f"total launches      : {tot_launches}")
    print(f"resolved to planet  : {resolved} ({resolved / max(tot_launches,1):.1%})")
    print(f"died (sun/edge/end) : {died + unresolved}")
    print(f"posture mix         : {dict(posture_counts)}")
    if frac_samples:
        frac_samples.sort()
        med = frac_samples[len(frac_samples) // 2]
        print(f"send-fraction median: {med:.2f}  min={frac_samples[0]:.2f} max={frac_samples[-1]:.2f}")
