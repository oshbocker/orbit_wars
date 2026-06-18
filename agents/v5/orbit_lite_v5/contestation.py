"""Contestation overlay: snipe the planets the opponent model predicts it will
capture-thin.

The base flow-diff planner sizes captures CAPTURE-MINIMALLY (``capture_floor`` =
ceil(projected defenders + overhead)), so a freshly captured planet holds only
~overhead + one turn of production. The engine combat is two-stage (top-second
among arrivals, then survivor-vs-garrison), so a fleet arriving a turn AFTER a
capture only has to clear that thin residual — far cheaper than the original
neutral garrison the opponent paid to clear.

This module consumes the EXACT opponent prediction already built by
``main._opponent_reactive_status`` (the engine-faithful, side-effect-free
post-opponent-capture projection) and turns it into NEW offensive candidates the
base shortlist is structurally blind to: at base-plan time the target is a
full-garrison neutral, not a 2-ship enemy. Every snipe is sized, gated and SCORED
by the same exact flow-diff the base planner uses (against the reactive world),
and drawn from the post-base-plan LEFTOVER budget — so a snipe never weakens the
base expansion (no tempo tax) and never fires a wave the exact scorer rates below
ROI.

Default OFF (``contest_waves == 0``) → ``main.run_turn`` never calls this →
byte-identical to v5.3.
"""

from __future__ import annotations

import torch

from .intercept_aim import intercept_angle
from .movement_step import LaunchEntries
from .planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _stable_argmax,
    capture_floor,
    make_launch_set,
    reachable_mask,
    safe_drain,
    score_candidates,
)


def plan_contestation_waves(
    *,
    movement,
    obs,
    obs_tensors,
    cache,
    base_status,
    reactive_status,
    prod,
    alive_by_step,
    config,
    player_count: int,
    leftover,          # [P] float — ships left on each owned planet after the base plan
    original_ships,    # [P] float — garrison at turn 0 (for safe_drain protection)
) -> LaunchEntries:
    P = obs.P
    device = obs.device
    dtype = original_ships.dtype
    pid = int(obs.player_id)

    H_axis = int(reactive_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    if H == 0:
        return _empty_entries(device, dtype)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.contest_waves))
    min_send = float(config.min_ships_to_launch)

    # --- 1. predicted thin captures -----------------------------------------
    # Planets currently mine/neutral that the reactive projection shows an
    # opponent capturing within the horizon. ``owner`` is absolute player ids.
    owner_traj = reactive_status.owner                       # [P, H+1]
    now_owner = owner_traj[:, 0]                              # [P]
    fut_owner = owner_traj[:, 1:]                             # [P, H]
    is_opp_fut = (fut_owner != pid) & (fut_owner >= 0)        # opponent-held at turn k
    now_not_opp = (now_owner == pid) | (now_owner < 0)        # mine or neutral now
    flips = now_not_opp.unsqueeze(-1) & is_opp_fut            # [P, H]
    is_thin = flips.any(dim=-1)                               # [P]
    if not bool(is_thin.any()):
        return _empty_entries(device, dtype)
    tgt_idx = torch.nonzero(is_thin, as_tuple=False).flatten()   # [T]
    T = int(tgt_idx.shape[0])
    # earliest predicted capture turn (1-based), device-stable tie-break.
    cap_turn = (_stable_argmax(flips.to(torch.long)) + 1)[tgt_idx].to(dtype)   # [T]

    # --- 2. sources: owned planets with leftover, capped by the do-nothing ---
    # safe_drain so a snipe never strips a source below its hold reserve.
    src_mask = obs.owned & obs.alive & (leftover >= min_send)
    if not bool(src_mask.any()):
        return _empty_entries(device, dtype)
    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    src_idx, src_exists = _candidate_indices(leftover, src_mask, S_cap)        # [S]
    S = int(src_idx.shape[0])
    src_safe = src_idx.clamp(0, P - 1)
    leftover_s = leftover[src_safe]
    orig_s = original_ships[src_safe]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain_s = safe_drain(
        base_status, source_idx=src_idx, source_ships=orig_s, H_eff=H_eff, player_id=pid,
    )
    committed_s = (orig_s - leftover_s).clamp(min=0.0)
    budget_s = torch.minimum(leftover_s, (drain_s - committed_s).clamp(min=0.0))  # [S]

    # --- 3. size each snipe vs the REACTIVE (post-capture, thin) garrison -----
    floor = capture_floor(
        reactive_status, target_idx=tgt_idx, k_max=K_eta,
        capture_overhead=float(config.contest_capture_overhead), player_id=pid,
    )                                                                          # [T, K]
    K = int(floor.shape[-1])
    if K == 0:
        return _empty_entries(device, dtype)
    k_pred = (cap_turn + float(config.contest_delay)).clamp(min=1.0, max=float(K))
    k_pred_idx = (k_pred.long() - 1).clamp(0, K - 1)                            # [T]
    floor_pred = floor.gather(-1, k_pred_idx.view(T, 1)).squeeze(-1)            # [T] minimal capturing size

    sizes = floor_pred.view(1, T).expand(S, T).contiguous()                    # [S, T]
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    fundable = budget_s.view(S, 1) >= sizes                                    # [S, T]
    active = reachable_mask(
        movement, source_idx=src_idx, target_idx=tgt_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1) & fundable                                                   # [S, T]
    aim = intercept_angle(
        movement, src_idx.unsqueeze(1), tgt_idx.unsqueeze(0), sizes, active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    # Re-gate the floor at the ACTUAL (size-dependent, slower) arrival turn and
    # require the fleet to land AFTER the predicted capture (so it meets the thin
    # residual, not the full neutral garrison the base plan would already target).
    k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)   # [S, T]
    floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    arrives_after = eta.ceil() >= cap_turn.view(1, T)
    clears = sizes >= floor_at_arr
    src_neq_tgt = src_idx.view(S, 1) != tgt_idx.view(1, T)
    valid = (
        viable & fundable & clears & arrives_after & (sizes >= max(1.0, min_send))
        & src_neq_tgt & src_exists.view(S, 1)
    )                                                                          # [S, T]
    if not bool(valid.any()):
        return _empty_entries(device, dtype)

    # --- 4. pack, score with the EXACT flow-diff, greedy-select --------------
    L = 1
    C = S * T
    cand_src = src_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = tgt_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = torch.zeros(C, dtype=torch.bool, device=device)

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        reactive_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )                                                                          # [C]
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    # Reserve-aware per-planet budget: the greedy debits this as it fires waves,
    # so multiple snipes sharing a source can't dip below the hold reserve.
    snipe_budget = leftover.to(dtype).clone()
    snipe_budget[src_safe] = budget_s
    target_exists = torch.ones(T, dtype=torch.bool, device=device)

    wave_entries, _ = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=snipe_budget,
        target_exists=target_exists, roi_threshold=float(config.contest_roi_threshold),
        cand_value=None, value_rerank_eps=0.0,
    )
    return wave_entries
