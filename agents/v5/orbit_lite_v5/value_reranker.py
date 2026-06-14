"""Learned global-value tie-breaker (Axis C, LEADERBOARD_CLIMB_PLAN 2026-06-13).

A tiny global value model ``state -> P(win for the acting player)`` used ONLY to
re-rank flow-diff candidates that the exact scorer is *indifferent* between — i.e.
whose competitive scores fall inside a small ``value_rerank_eps`` band of the
about-to-be-selected best. It is a tie-breaker, never a primary scorer, and never
fires a shot the flow-diff would not (the re-rank set is restricted to candidates
that already clear ``roi_threshold``).

WHY tie-break-only (hard-won, see EXPLORED_AND_ABANDONED.md): our own Phase-2
diagnostic found learned value is grounded GLOBALLY (corr ~0.39) but too noisy to
rank near-equal siblings on its own, and every coarse signal that *second-guessed*
the exact planner (shot-validator Cluster 6, arrival-horizon Cluster 8,
defensive-symmetry Cluster 9) regressed it. So this signal only moves the choice
*within* the flow-diff's own indifference band — the exact scorer stays
authoritative everywhere else.

This is value-for-reranking, NOT policy cloning. (Producer-BC plateaus at 3% vs
producer; a value model trained on episode outcomes is a different object.)

Plumbing mirrors ``shot_validator.py``: the SAME 16-feature global encoder runs at
label-harvest time (``scripts/harvest_values.py``) and at inference, so the feature
distribution matches. The model auto-loads from ``value_model_weights.npz`` bundled
in this package; absent weights OR ``value_rerank_eps <= 0`` => pass-through,
byte-identical to plain v5.
"""

from __future__ import annotations

import numpy as np

FEATURE_DIM = 16

# Normalisation constants — MUST stay identical between the numpy harvest encoder
# (``encode_global_from_raw``) and the torch inference path (``candidate_value_scores``)
# or the model is fed an out-of-distribution feature vector.
_S = 200.0   # ship counts
_PR = 30.0   # production
_PL = 20.0   # planet counts
_F = 200.0   # in-flight ship counts
_ST = 500.0  # step
_CM = 4.0    # comet net


class ValueModel:
    """16 -> 32 -> 16 -> 1 sigmoid MLP forward pass on the npz layout
    ``{w0,b0,w2,b2,w4,b4}`` — the same artifact layout as the shot validator.

    Holds numpy weights; offers a numpy ``proba`` (offline use) and, lazily, torch
    weight tensors for the on-device ``proba_torch`` used inside the planner.
    """

    def __init__(self, npz_path):
        npz = np.load(str(npz_path))
        self.w0 = npz["w0"]
        self.b0 = npz["b0"]
        self.w2 = npz["w2"]
        self.b2 = npz["b2"]
        self.w4 = npz["w4"]
        self.b4 = npz["b4"]
        self._torch = None

    def proba(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.w0.T + self.b0)
        h = np.maximum(0.0, h @ self.w2.T + self.b2)
        z = (h @ self.w4.T + self.b4).reshape(-1)
        return 1.0 / (1.0 + np.exp(-z))

    def proba_torch(self, x):
        """P(win) for a ``[C, 16]`` torch feature batch. Returns ``[C]`` on x.device."""
        import torch

        if self._torch is None or self._torch[0].device != x.device:
            dev = x.device

            def _t(a):
                return torch.from_numpy(np.ascontiguousarray(a)).to(dev, x.dtype)

            self._torch = (_t(self.w0), _t(self.b0), _t(self.w2), _t(self.b2), _t(self.w4), _t(self.b4))
        w0, b0, w2, b2, w4, b4 = self._torch
        h = torch.relu(x @ w0.T + b0)
        h = torch.relu(h @ w2.T + b2)
        z = (h @ w4.T + b4).reshape(-1)
        return torch.sigmoid(z)


# ---------------------------------------------------------------------------
# Canonical 16-feature global encoder (harvest side — raw Kaggle obs rows)
# ---------------------------------------------------------------------------


def _prow(p, name: str, idx: int):
    """Planet/fleet field access: Struct attribute, then dict, then list index."""
    if hasattr(p, name):
        return getattr(p, name)
    if isinstance(p, dict):
        return p[name]
    return p[idx]


def encode_global_from_raw(obs, player_id: int) -> np.ndarray:
    """16-dim global feature vector for ``player_id`` from a raw observation.

    Used at harvest time on every step of every game. The acting player is
    ``player_id`` (owners in the board are absolute, so the same board encodes a
    distinct vector per seat). Mirrors the per-candidate torch assembly below.
    """
    planets = _prow(obs, "planets", 0) or []
    fleets = _prow(obs, "fleets", 1) or []
    step = float(_prow(obs, "step", 2) or 0)
    comet_ids = set()
    cids = _prow(obs, "comet_planet_ids", 0) if (hasattr(obs, "comet_planet_ids") or (isinstance(obs, dict) and "comet_planet_ids" in obs)) else None
    if cids:
        comet_ids = {int(c) for c in cids if int(c) >= 0}

    me = int(player_id)
    n_owner = 4
    ships_by = [0.0] * n_owner
    prod_by = [0.0] * n_owner
    count_by = [0] * n_owner
    neutral_planets = 0
    largest_prod = -1.0
    largest_is_mine = 0.0
    my_comets = 0
    enemy_comets = 0
    for p in planets:
        pid = int(_prow(p, "id", 0))
        if pid < 0:
            continue
        owner = int(_prow(p, "owner", 1))
        sh = float(_prow(p, "ships", 5))
        pr = float(_prow(p, "production", 6))
        if 0 <= owner < n_owner:
            ships_by[owner] += sh
            prod_by[owner] += pr
            count_by[owner] += 1
        else:
            neutral_planets += 1
        if pr > largest_prod:
            largest_prod = pr
            largest_is_mine = 1.0 if owner == me else 0.0
        if pid in comet_ids:
            if owner == me:
                my_comets += 1
            elif owner >= 0:
                enemy_comets += 1

    my_fleet = 0.0
    enemy_fleet = 0.0
    for f in fleets:
        fid = int(_prow(f, "id", 0))
        if fid < 0:
            continue
        owner = int(_prow(f, "owner", 1))
        sh = float(_prow(f, "ships", 6))
        if owner == me:
            my_fleet += sh
        elif owner >= 0:
            enemy_fleet += sh

    my_ships = ships_by[me]
    my_prod = prod_by[me]
    my_planets = count_by[me]
    enemy_ships_total = sum(ships_by[o] for o in range(n_owner) if o != me)
    enemy_prod_total = sum(prod_by[o] for o in range(n_owner) if o != me)
    enemy_planets = sum(count_by[o] for o in range(n_owner) if o != me)
    max_enemy_ships = max((ships_by[o] for o in range(n_owner) if o != me), default=0.0)
    max_enemy_prod = max((prod_by[o] for o in range(n_owner) if o != me), default=0.0)
    comet_net = float(my_comets - enemy_comets)

    return np.array(
        [
            step / _ST,
            my_ships / _S,
            my_prod / _PR,
            enemy_ships_total / _S,
            enemy_prod_total / _PR,
            max_enemy_ships / _S,
            max_enemy_prod / _PR,
            my_planets / _PL,
            enemy_planets / _PL,
            neutral_planets / _PL,
            my_fleet / _F,
            enemy_fleet / _F,
            (my_ships - max_enemy_ships) / _S,
            (my_prod - max_enemy_prod) / _PR,
            largest_is_mine,
            comet_net / _CM,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Per-candidate post-move value (inference side — torch, on-device, batched [C])
# ---------------------------------------------------------------------------


def candidate_value_scores(
    *,
    obs,                 # ParsedObs
    prod,                # [P] float tensor
    obs_tensors: dict,
    target_idx,          # [T] long — shortlist slot per shortlist position
    cand_tgt_short,      # [C] long — candidate -> shortlist position
    cand_send,           # [C, L] float — ships per contributor (0 where invalid)
    cand_active,         # [C, L] bool
    model: ValueModel,
    player_count: int,
    player_id: int,
):
    """P(win) of the **first-order projected board** after each candidate launch.

    Each candidate is a single (source, target, ships) wave. We apply its
    first-order effect to the current GLOBAL aggregates — ships leave home into
    flight; on a capture the target flips to me (its production and a surviving
    garrison accrue to me, the prior owner loses them) — and score the resulting
    aggregate state with the value MLP. This is the granularity the global model
    operates at: it discriminates candidates by *what they capture* (prod, owner,
    defender cost), which is exactly what breaks a flow-diff tie. O(C) tiny-MLP
    rows => trivially inside the 1s/step budget. Returns ``[C]`` on obs.device.
    """
    import torch

    device = obs.device
    dt = torch.float32
    A = max(2, int(player_count))
    me = int(player_id)
    C = int(cand_tgt_short.shape[0])

    owner_abs = obs.owner_abs
    ships = obs.ships.to(dt)
    prod = prod.to(dt)
    a_range = torch.arange(A, device=device)

    # base per-owner aggregates ----------------------------------------------
    valid_owner = obs.alive & (owner_abs >= 0) & (owner_abs < A)
    oa = owner_abs.long().clamp(0, A - 1)
    ships_by = torch.zeros(A, dtype=dt, device=device)
    prod_by = torch.zeros(A, dtype=dt, device=device)
    count_by = torch.zeros(A, dtype=dt, device=device)
    vf = valid_owner.to(dt)
    ships_by.scatter_add_(0, oa, ships * vf)
    prod_by.scatter_add_(0, oa, prod * vf)
    count_by.scatter_add_(0, oa, vf)
    total_alive = float(obs.alive.to(dt).sum().item())
    neutral_planets = total_alive - float(count_by.sum().item())

    base_my_ships = float(ships_by[me].item())
    base_my_prod = float(prod_by[me].item())
    base_my_planets = float(count_by[me].item())
    base_enemy_planets = float(count_by.sum().item() - count_by[me].item())

    # fleets in flight
    f_valid = obs.f_alive & (obs.f_owner >= 0)
    f_owner = obs.f_owner.long()
    f_ships = obs.f_ships.to(dt)
    my_fleet = float((f_ships * (f_valid & (f_owner == me)).to(dt)).sum().item())
    enemy_fleet = float((f_ships * (f_valid & (f_owner != me)).to(dt)).sum().item())

    # largest planet by production (matches harvest argmax-by-prod)
    if total_alive > 0:
        big = torch.where(obs.alive, prod, torch.full_like(prod, float("-inf"))).argmax()
        largest_is_mine = 1.0 if int(owner_abs[big].item()) == me else 0.0
    else:
        largest_is_mine = 0.0

    # comets (constant across candidates — targets are never comets)
    comet_net = 0.0
    comet_ids = obs_tensors.get("comet_planet_ids")
    planets_t = obs_tensors.get("planets")
    if comet_ids is not None and planets_t is not None:
        pid_col = planets_t[..., 0].long()
        my_c = en_c = 0
        for c in range(int(comet_ids.reshape(-1).shape[0])):
            cid = int(comet_ids.reshape(-1)[c].item())
            if cid < 0:
                continue
            hit = (pid_col == cid).nonzero(as_tuple=True)[0]
            if hit.numel() == 0:
                continue
            o = int(owner_abs[hit[0]].item())
            if o == me:
                my_c += 1
            elif o >= 0:
                en_c += 1
        comet_net = float(my_c - en_c)

    # per-candidate target info ----------------------------------------------
    tgt_slot = target_idx[cand_tgt_short].clamp(0, obs.P - 1)        # [C]
    tgt_owner = owner_abs[tgt_slot]                                  # [C] float (-1 neutral)
    tgt_prod = prod[tgt_slot]                                        # [C]
    tgt_ships = ships[tgt_slot]                                      # [C]
    active = cand_active.any(dim=-1)                                 # [C] bool
    committed = (cand_send * cand_active.to(dt)).sum(dim=-1)         # [C]

    is_capture = active & (tgt_owner != float(me)) & (tgt_owner >= -1)  # enemy or neutral
    # owned-target reinforcements: not a capture (floor 1); ships go home->home
    is_enemy_cap = is_capture & (tgt_owner >= 0)                     # [C]
    survivors = (committed - tgt_ships).clamp(min=1.0)              # ships that land & hold

    cap_f = is_capture.to(dt)
    en_f = is_enemy_cap.to(dt)

    # my aggregates after move
    my_ships_c = base_my_ships - committed + cap_f * survivors
    my_prod_c = base_my_prod + cap_f * tgt_prod
    my_planets_c = base_my_planets + cap_f
    enemy_planets_c = base_enemy_planets - en_f  # neutral captures don't change enemy count
    neutral_after = neutral_planets - (cap_f - en_f)  # neutral captures shrink neutral pool
    my_fleet_c = my_fleet + committed

    # per-opponent ships/prod after the capture (decrement the captured owner)
    onehot = (tgt_owner.long().clamp(0, A - 1).view(C, 1) == a_range.view(1, A)).to(dt)  # [C,A]
    sub_ships = onehot * (tgt_ships.view(C, 1) * en_f.view(C, 1))
    sub_prod = onehot * (tgt_prod.view(C, 1) * en_f.view(C, 1))
    ships_by_c = ships_by.view(1, A) - sub_ships                     # [C,A]
    prod_by_c = prod_by.view(1, A) - sub_prod
    opp_mask = (a_range.view(1, A) != me)
    neg = torch.full((1, A), float("-inf"), device=device)
    zero = torch.zeros((1, A), device=device)
    enemy_ships_total_c = torch.where(opp_mask, ships_by_c, zero).sum(dim=1)
    enemy_prod_total_c = torch.where(opp_mask, prod_by_c, zero).sum(dim=1)
    max_enemy_ships_c = torch.where(opp_mask, ships_by_c, neg).max(dim=1).values
    max_enemy_prod_c = torch.where(opp_mask, prod_by_c, neg).max(dim=1).values

    step_v = float(obs.step.reshape(-1)[0].item())
    ones = torch.ones(C, device=device, dtype=dt)
    feats = torch.stack(
        [
            ones * (step_v / _ST),
            my_ships_c / _S,
            my_prod_c / _PR,
            enemy_ships_total_c / _S,
            enemy_prod_total_c / _PR,
            max_enemy_ships_c / _S,
            max_enemy_prod_c / _PR,
            my_planets_c / _PL,
            enemy_planets_c / _PL,
            neutral_after / _PL,
            my_fleet_c / _F,
            ones * (enemy_fleet / _F),
            (my_ships_c - max_enemy_ships_c) / _S,
            (my_prod_c - max_enemy_prod_c) / _PR,
            ones * largest_is_mine,
            ones * (comet_net / _CM),
        ],
        dim=1,
    )  # [C, 16]
    return model.proba_torch(feats)
