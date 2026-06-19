"""Validate the ANALYTIC target resolver (needed for counterfactual actions that
never produced a real fleet) against the ground-truth fleet-TRACKING resolver
(macro_relabel) on real replays. If they agree, the analytic resolver is safe to
use for both the top agent's actual action and the producer/v5 counterfactual.
"""
from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "agents" / "external" / "producer"))

import torch  # noqa: E402

from orbit_lite.adapter import single_obs_to_tensor  # noqa: E402
from orbit_lite.movement import (  # noqa: E402
    MovementConfig,
    PlanetMovement,
    _estimate_new_fleet_arrivals,
)
from scripts.macro_relabel import resolve_launches_for_step  # noqa: E402

H = 40  # resolver lookahead horizon


def analytic_targets(obs: dict, action: list, player_id: int) -> list[int]:
    """Resolve each launch's destination planet id analytically (producer physics).

    Returns a planet id per move (-1 = dies at sun/edge / no hit). Order matches `action`.
    """
    if not action:
        return []
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    mv = PlanetMovement.from_obs_tensors(
        obs_tensors, config=MovementConfig(movement_horizon=H, track_fleets=False)
    )
    planets = {int(p[0]): p for p in obs["planets"]}
    L = len(action)
    rows = torch.full((L, 7), -1.0, dtype=mv.dtype)
    for i, mvmove in enumerate(action):
        src_id, angle, ships = int(mvmove[0]), float(mvmove[1]), float(mvmove[2])
        sp = planets.get(src_id)
        if sp is None:
            continue
        cx, cy, rad = float(sp[2]), float(sp[3]), float(sp[4])
        sx = cx + math.cos(angle) * (rad + 0.1)
        sy = cy + math.sin(angle) * (rad + 0.1)
        rows[i, 1] = float(player_id)
        rows[i, 2] = sx
        rows[i, 3] = sy
        rows[i, 4] = angle
        rows[i, 6] = ships
    est = _estimate_new_fleet_arrivals(
        movement=mv, obs_fleets=rows, fleet_slot=torch.arange(L)
    )
    pid_by_slot = mv.planet_ids.tolist()
    out = []
    for i in range(L):
        if bool(est["has_hit"][i]):
            out.append(int(pid_by_slot[int(est["target_slot"][i])]))
        else:
            out.append(-1)
    return out


def _cat(owner, player):
    if owner == player:
        return "reinforce"
    if owner == -1:
        return "neutral"
    return "attack"


def main():
    files = sorted(glob.glob("/tmp/ow_analysis/replays/*.json"))[:15]
    agree = total = both_resolved = cat_agree = 0
    # buckets by source->target straight-line distance
    from collections import defaultdict

    by_dist = defaultdict(lambda: [0, 0])  # bucket -> [agree, n]
    for f in files:
        rep = json.load(open(f))
        steps = rep["steps"]
        nseats = len(steps[0])
        for seat in range(nseats):
            for t in range(1, len(steps)):
                if seat >= len(steps[t]):
                    break
                if steps[t - 1][seat].get("status") != "ACTIVE":
                    continue
                action = steps[t][seat].get("action") or []
                if not action:
                    continue
                dec_obs = steps[t - 1][seat]["observation"]
                planets = {int(p[0]): p for p in dec_obs["planets"]}
                tracked = resolve_launches_for_step(steps, t, seat)
                analytic = analytic_targets(dec_obs, action, seat)
                for ml, an in zip(tracked, analytic):
                    total += 1
                    if ml.dst_id >= 0 and an >= 0:
                        both_resolved += 1
                        ok = ml.dst_id == an
                        if ok:
                            agree += 1
                        # category agreement
                        pt, pa = planets.get(ml.dst_id), planets.get(an)
                        if pt is not None and pa is not None:
                            if _cat(int(pt[1]), seat) == _cat(int(pa[1]), seat):
                                cat_agree += 1
                        # distance bucket (src->tracked dst)
                        sp = planets.get(ml.src_id)
                        if sp is not None and pt is not None:
                            d = math.hypot(float(sp[2]) - float(pt[2]), float(sp[3]) - float(pt[3]))
                            b = "0-15" if d < 15 else "15-30" if d < 30 else "30+"
                            by_dist[b][0] += int(ok)
                            by_dist[b][1] += 1
    print(f"launches compared : {total}")
    print(f"both resolved     : {both_resolved} ({both_resolved/max(total,1):.1%})")
    print(f"exact-id agreement: {agree}/{both_resolved} = {agree/max(both_resolved,1):.1%}")
    print(f"category agreement: {cat_agree}/{both_resolved} = {cat_agree/max(both_resolved,1):.1%}")
    print("exact-id agreement by src->dst distance:")
    for b in ("0-15", "15-30", "30+"):
        a, n = by_dist[b]
        print(f"  {b:>6}: {a}/{n} = {a/max(n,1):.1%}")


if __name__ == "__main__":
    main()
