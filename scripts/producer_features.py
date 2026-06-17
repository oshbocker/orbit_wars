"""Producer-grade feature extraction (rich representation, Track 1 keystone).

Runs producer/v5's EXACT projection (orbit_lite_v5) on an observation via the
``_FEATURE_SINK`` hook and exposes its per-candidate quantities — most importantly
producer's own Δnet candidate score — as a dense [P, P, F] edge grid (planet-id indexed),
plus the per-planet garrison_status timeline. This is the information producer reasons over;
feeding it to the net removes the "re-derive the 18-turn projection from a snapshot" wall
that capped V2-feature BC at 3%.

Run as a script to VALIDATE that producer's actual launches are the top-scored grid entries
(proves the keystone Δnet-score feature captures producer's decision + the id mapping).
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MAXP = 40
_counter = itertools.count()


def load_v5_module():
    main_py = ROOT / "agents" / "v5" / "main.py"
    modname = f"_v5feat_{next(_counter)}"
    spec = importlib.util.spec_from_file_location(modname, main_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


class ProducerFeatureExtractor:
    """Holds one v5 module; .extract(obs) -> dense feature grids for that obs."""

    def __init__(self, max_planets: int = MAXP):
        self.mod = load_v5_module()
        self.P = max_planets

    def extract(self, obs) -> dict:
        mod = self.mod
        mod._FEATURE_SINK = {}
        mod._RUNTIME.memory.reset()        # fresh projection from THIS obs (orbits forward-predicted)
        try:
            mod.agent(obs)                  # runs run_turn -> plan_lite_waves -> fills sink
        finally:
            sink = mod._FEATURE_SINK
            mod._FEATURE_SINK = None
        return self._densify(sink)

    def _densify(self, sink: dict) -> dict:
        P = self.P
        NEG = float("-inf")
        score_grid = torch.full((P, P), NEG)        # producer Δnet per (src_pid, tgt_pid)
        eta_grid = torch.zeros((P, P))
        size_grid = torch.zeros((P, P))
        valid_grid = torch.zeros((P, P), dtype=torch.bool)
        if not sink:
            return {"score": score_grid, "eta": eta_grid, "size": size_grid,
                    "valid": valid_grid, "timeline_owner": torch.zeros((P, 0)),
                    "timeline_ships": torch.zeros((P, 0))}
        pids = sink["planet_ids"].long()                       # [P_rows] row -> planet id
        cand_src = sink["cand_src"].squeeze(-1).long()         # [C] row idx
        cand_tgt = sink["cand_tgt_slot"].long()                # [C] row idx
        valid = sink["cand_valid"].bool()                      # [C]
        score = sink["score"].float()                          # [C]
        send = sink["cand_send"].squeeze(-1).float()           # [C]
        eta = sink["cand_eta"].squeeze(-1).float()             # [C]
        src_pid = pids[cand_src.clamp(0, pids.shape[0] - 1)]
        tgt_pid = pids[cand_tgt.clamp(0, pids.shape[0] - 1)]
        for c in range(cand_src.shape[0]):
            if not bool(valid[c]):
                continue
            i, j = int(src_pid[c]), int(tgt_pid[c])
            # keep the best (highest-score) candidate per edge (cheap variants share an edge)
            if 0 <= i < P and 0 <= j < P and (
                not bool(valid_grid[i, j]) or float(score[c]) > float(score_grid[i, j])
            ):
                score_grid[i, j] = float(score[c])
                eta_grid[i, j] = float(eta[c])
                size_grid[i, j] = float(send[c])
                valid_grid[i, j] = True
        # per-planet timeline (projected owner/ships), re-indexed row -> planet id
        so = sink["status_owner"].float()      # [P_rows, H+1]
        ss = sink["status_ships"].float()       # [P_rows, H+1]
        H1 = so.shape[-1]
        t_owner = torch.zeros((P, H1))
        t_ships = torch.zeros((P, H1))
        for row in range(pids.shape[0]):
            pid = int(pids[row])
            if 0 <= pid < P:
                t_owner[pid] = so[row]
                t_ships[pid] = ss[row]
        return {"score": score_grid, "eta": eta_grid, "size": size_grid,
                "valid": valid_grid, "timeline_owner": t_owner, "timeline_ships": t_ships}


if __name__ == "__main__":
    import argparse

    from kaggle_environments import make

    from agents import load_named_agent
    from scripts.macro_relabel import resolve_launches_for_step

    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20000)
    args = ap.parse_args()

    ext = ProducerFeatureExtractor()
    n_launch = argmax_match = top3_match = valid_at_launch = 0
    for g in range(args.games):
        env = make("orbit_wars", configuration={"randomSeed": args.seed + g})
        env.run([load_named_agent("producer"), load_named_agent("producer")])
        steps = env.steps
        # sample every 8th decision step to keep it quick
        for t in range(1, len(steps), 8):
            for player in (0, 1):
                mls = [m for m in resolve_launches_for_step(steps, t, player) if m.dst_id >= 0]
                if not mls:
                    continue
                dec = steps[t - 1][player]
                obs = dec["observation"] if not hasattr(dec, "observation") else dec.observation
                grid = ext.extract(obs)["score"]
                for ml in mls:
                    i, j = ml.src_id, ml.dst_id
                    if not (0 <= i < MAXP and 0 <= j < MAXP):
                        continue
                    n_launch += 1
                    row = grid[i]
                    if bool(torch.isfinite(row[j])):
                        valid_at_launch += 1
                    finite = torch.isfinite(row)
                    if finite.any():
                        if int(row.argmax()) == j:
                            argmax_match += 1
                        top3 = set(int(x) for x in row.topk(min(3, int(finite.sum()))).indices)
                        if j in top3:
                            top3_match += 1
        print(f"  game {g} done")

    print("\n=== producer launches vs extracted Δnet score grid ===")
    print(f"launches checked        : {n_launch}")
    print(f"target valid in grid    : {valid_at_launch} ({valid_at_launch/max(n_launch,1):.1%})")
    print(f"producer tgt == argmax  : {argmax_match} ({argmax_match/max(n_launch,1):.1%})")
    print(f"producer tgt in top-3   : {top3_match} ({top3_match/max(n_launch,1):.1%})")
