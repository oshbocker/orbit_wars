
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

# Make the sibling ``orbit_lite`` package importable wherever this file runs:
# loaded in place, dropped at a submission-archive root, or exec'd by
# kaggle_environments with no ``__file__`` (fall back to the working dir).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from orbit_lite_v5.adapter import single_obs_to_tensor, sparse_action_row_to_moves
from orbit_lite_v5.distance_cache import build_distance_cache
from orbit_lite_v5.geometry import fleet_speed
from orbit_lite_v5.intercept_aim import intercept_angle
from orbit_lite_v5.movement import MovementConfig, PlanetMovement
from orbit_lite_v5.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite_v5.obs import parse_obs
from orbit_lite_v5.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite_v5.shot_validator import DEFAULT_THRESHOLD, NumpyValidator, apply_veto
from orbit_lite_v5.value_reranker import ValueModel, candidate_value_scores
from torch import Tensor

# Last step at which the engine still runs production/combat (game ends at 498).
END_STEP = 498

# v5: reject-only shot validator (Phase 2.1). Auto-enabled when trained weights
# ship inside the package (the bundle build copies them); absent weights = the
# planner's moves pass through untouched, byte-identical to plain v5.
_VALIDATOR_WEIGHTS = os.path.join(_HERE, "orbit_lite_v5", "validator_weights.npz")
_VALIDATOR: NumpyValidator | None = None
_VETO_THRESHOLD = DEFAULT_THRESHOLD
if os.path.exists(_VALIDATOR_WEIGHTS):
    try:
        _VALIDATOR = NumpyValidator(_VALIDATOR_WEIGHTS)
    except Exception:
        _VALIDATOR = None


def set_validator(weights_path: str | None, threshold: float = DEFAULT_THRESHOLD) -> None:
    """Enable/disable the shot veto on this module instance (arena A/B hook)."""
    global _VALIDATOR, _VETO_THRESHOLD
    _VALIDATOR = NumpyValidator(weights_path) if weights_path else None
    _VETO_THRESHOLD = threshold


# v5.5 (Axis C): learned global-value tie-breaker. Auto-enabled only when BOTH the
# weights ship in the package AND a config sets value_rerank_eps > 0 (default 0.0,
# so a bundled model alone changes nothing — byte-identical to v5.4).
_VALUE_WEIGHTS = os.path.join(_HERE, "orbit_lite_v5", "value_model_weights.npz")
_VALUE_MODEL: ValueModel | None = None
if os.path.exists(_VALUE_WEIGHTS):
    try:
        _VALUE_MODEL = ValueModel(_VALUE_WEIGHTS)
    except Exception:
        _VALUE_MODEL = None


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs.  """

    
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12         # enemy/neutral proximity targets
    max_defensive_targets: int = 4          
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5              # fire if score > this
    min_ships_to_launch: float = 4.0
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 7.0
    regroup_pressure_delta_min: float = 0.25
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    ffa_leader_attack_bonus: float = 0.0
    ffa_target_prod_bonus: float = 0.0
    # v5: 4P nearest-opponent priority (ykhnkf distance-prioritized lineage):
    # scale candidate scores on enemy-owned targets by near/far opponent. 1.0 = off.
    ffa_near_opponent_mult: float = 1.0
    ffa_far_opponent_mult: float = 1.0
    # v5: second "just enough to capture" candidate size per (source, target):
    # size = ceil(capture floor at arrival + margin) alongside the full
    # safe_drain candidate; the exact flow-diff scorer arbitrates. Attacks the
    # single-size structural limit (full-drain captures of small neutrals strand
    # ships the scorer would rather keep home). < 0 (default) = OFF, byte-identical.
    cheap_capture_margin: float = -1.0
    # v5.3: ETA-aware reinforcement risk (The Producer V2, slawekbiel
    # 2026-06-12). Inflate the capture floor by ``reinforce_size_beta * rho(eta)
    # * C_k`` where C_k is enemy supply reachable to the target during the
    # fleet's flight, so the agent *declines* captures the enemy will reinforce
    # mid-flight instead of sinking its whole garrison into a doomed attack
    # (the flow scorer projects opponents do-nothing, so it can't see reactive
    # reinforcement). 0.0 = OFF, byte-identical to v5.2. Default ON (2.2 = V2
    # upstream) since v5.3 — gate: 75% vs v5.2 mirror @ n=120, 56% vs
    # producer_v2 @ n=60. CONFIG_4P inherits via ``dataclasses.replace``.
    reinforce_size_beta: float = 2.2
    reinforce_eta_free: float = 3.0
    reinforce_eta_scale: float = 12.0
    # v5.4 (Axis A candidate a): DEFENSIVE symmetry of the v5.3 reinforce-risk win.
    # The reinforce floor (above) inflates the *attack* send size for captures the
    # enemy can reinforce mid-flight; symmetrically, ``safe_drain`` only protects a
    # source against fleets *already in flight* (the do-nothing projection), so it
    # over-commits ships away from planets the enemy can *launch* at. This holds
    # back ``defense_size_beta * reachable-enemy-mass(source)`` ships per source —
    # reusing the exact ``cheap_enemy_pressure`` proxy that the regroup gradient and
    # the offensive floor already use (its distance decay encodes reaction timing,
    # so no separate rho ramp). 0.0 = OFF, byte-identical to v5.3.
    defense_size_beta: float = 0.0
    # v5.2: terminal-phase config swap (pilkwang ProducerLite exp59 lineage).
    # In the last ``terminal_phase_turns`` turns, ROI drops, the wave cap rises
    # and regroup stops — score banked in fleets beats ships parked at home once
    # nothing launched late can pay back. Complements the horizon clamp above
    # (clamp = exact accounting; this = aggressive spending of the exact budget).
    # 0 = OFF, byte-identical to v5.1. Default ON (40) since v5.2; applies to
    # both formats (CONFIG_4P inherits — exp59 runs it in 2P and 4P).
    terminal_phase_turns: int = 40
    terminal_roi_threshold: float = 1.0
    terminal_max_waves_per_turn: int = 8
    terminal_enable_regroup: bool = False
    # v5.5 (Axis C): learned global-value tie-breaker. When > 0, among flow-diff
    # candidates within ``value_rerank_eps`` of the about-to-be-selected best (and
    # only those that already clear roi_threshold), pick the one the value model
    # rates highest P(win) instead of the lowest slot index. Tie-break ONLY — the
    # exact flow scorer stays authoritative; never fires a wave flow-diff wouldn't.
    # 0.0 = OFF, byte-identical to v5.4. Requires value_model_weights.npz bundled
    # in the package (auto-loaded); absent weights => OFF regardless of eps.
    value_rerank_eps: float = 0.0
    # v5.4 Track 1 (level-1 opponent-aware planning). The flow-diff projects all
    # opponents as do-nothing (a one-shot best-response to a passive world = the
    # maximally *exploitable* solution-concept class for a 2-player zero-sum
    # simultaneous-move game; SOTA literature: regret/equilibrium play is less
    # exploitable). This is the next rung of the cognitive hierarchy: we ARE the
    # opponent's planner (mirror meta), so run the planner from each live enemy
    # seat (level-0, best-responding to the do-nothing baseline), inject their
    # top-N best-response ATTACK launches into the projection, then score OUR
    # candidates against that reactive world. Generalizes the v5.3 reinforce-risk
    # win (which modeled only one reactive term — mid-flight reinforcement) to the
    # opponent's full offensive response, using the EXACT planner as the opponent
    # model (not Cluster-9's coarse mass proxy) and the EXACT engine projection to
    # propagate it. 1-ply (no rollout) and side-effect-free on the rolling cache
    # (the projection is snapshotted/restored). 0 = OFF, byte-identical to v5.3;
    # N>0 injects up to N attack waves per enemy seat (the dose-response knob).
    opp_inject_waves: int = 0


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``.

    Consumed only as the **regroup gradient** (rank owned planets by how stressed
    they are, move ships up the gradient). For each planet ``t``, sums a
    distance-decayed share of every enemy source's **current** garrison that could
    straight-line reach ``t`` within ``horizon`` turns, using the step-0 centre
    distance ``cross_dist[0]``. The decay ``(1 - d/(speed·H))₊`` weights nearer
    enemies more, giving a graded frontline signal in ship-mass units.

    Approximations: ignores target orbital drift over the horizon, production
    accrued in flight, the per-owner split, and in-flight enemy fleets. Pure
    arithmetic on cached tensors
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)    # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye              # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)                       # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)                                            # [P] summed over sources


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Single-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)

    # Reachable-enemy-mass proxy ([P]) — computed ONCE and reused for ALL THREE of:
    # the offensive reinforcement-risk floor margin (below), the v5.4 defensive
    # reserve (safe_drain, right here), and the regroup gradient (further down). Its
    # decay distance-scale is the attack reach K_eta. (Producer V2 + v5.4 symmetry.)
    beta = float(config.reinforce_size_beta)
    defense_beta = float(config.defense_size_beta)
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)    # [P]
        if beta > 0.0 or defense_beta > 0.0 or bool(config.enable_regroup) else None
    )

    # v5.4 defensive symmetry: hold back ``defense_beta * enemy_mass(source)`` ships
    # on each source so the planner under-commits planets the enemy can mass on
    # (None when OFF → safe_drain byte-identical to v5.3).
    defense_reserve = (
        defense_beta * enemy_mass[source_idx.clamp(0, P - 1)]                    # [S]
        if defense_beta > 0.0 and enemy_mass is not None else None
    )
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid, reserve=defense_reserve,
    )                                                                            # [S]

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]

    # ETA-aware reinforcement risk: inflate the capture floor by ``beta * rho(k)
    # * reachable-enemy-mass(target)``. The per-arrival-turn growth comes from
    # the rho(k) timing ramp. Gated by beta > 0 (OFF = bare floor, byte-identical).
    reinforcement = None
    if beta > 0.0 and enemy_mass is not None:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]                    # [T]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange, eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )                                                                        # [K_eta]
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)      # [T, K_eta]

    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
        reinforcement=reinforcement,
    )                                                                            # [T, K]
    K = int(floor.shape[-1])

    # --- single fleet size = the max garrison launch (safe_drain) ---------------
    # Engine needs integer ship counts; floor (never exceed what's available).
    sizes = drain.view(S, 1).expand(S, T).floor()                                # [S, T]

    # Strict-superset reachability precheck (always on): defers the body screen to
    # candidates that can physically reach the target in time.
    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)                                                                # [S, T]
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),                                                 # [S, 1]
        target_idx.unsqueeze(0),                                                 # [1, T]
        sizes,                                                                    # [S, T]
        active=active,
    )
    angle = aim["angle"]                                                         # [S, T]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    # Capture-floor gate at each fleet's arrival turn (defenders grow with k). The
    # single size must clear the defender it lands on (size >= floor_at_arr). Owned
    # targets have floor 1 (reinforcement), so any positive send clears.
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)  # [S,T]
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr                                         # [S, T]

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )                                                                            # [S, T]

    # --- v5: optional second, cheaper size per (source, target) -----------------
    # Just-enough-to-capture: ceil(floor_at_arr + margin), capped at the drain.
    # The smaller fleet is slower, so its floor is re-gated at its OWN (later)
    # arrival turn; if defenders outgrow the margin in flight the variant drops
    # and the full-size candidate stays on the board. Greedy's one-wave-per-target
    # mask keeps the two sizes mutually exclusive; the scorer picks between them.
    two_sizes = float(config.cheap_capture_margin) >= 0.0
    if two_sizes:
        sizes2 = (floor_at_arr + float(config.cheap_capture_margin)).ceil().clamp(min=1.0)
        sizes2 = torch.minimum(sizes2, sizes)                                    # [S, T]
        # Only when strictly cheaper than the drain candidate, and never on owned
        # targets (their floor is 1 — a token reinforcement is junk).
        distinct = (sizes2 < sizes) & ~target_is_mine.view(1, T)
        active2 = reachable_mask(
            movement, source_idx=source_idx, target_idx=target_idx,
            fleet_sizes=sizes2.unsqueeze(-1), eta_cap=eta_cap,
        ).squeeze(-1) & distinct
        aim2 = intercept_angle(
            movement,
            source_idx.unsqueeze(1),
            target_idx.unsqueeze(0),
            sizes2,
            active=active2,
        )
        angle2 = aim2["angle"]                                                   # [S, T]
        eta2 = aim2["eta"]
        viable2 = aim2["viable"] & (eta2 <= eta_cap.view(1, T))
        if K > 0:
            k_arr2 = (eta2.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr2 = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr2.unsqueeze(-1)).squeeze(-1)
        else:
            floor_at_arr2 = torch.ones(S, T, dtype=dtype, device=device)
        valid2 = (
            viable2 & (sizes2 >= floor_at_arr2) & (sizes2 >= 1.0) & distinct
            & src_neq_tgt & source_exists.view(S, 1) & target_exists.view(1, T)
        )                                                                        # [S, T]

    # --- pack one candidate per (source, target); contributor axis L = 1 --------
    L = 1
    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]                                  # [C]
    if two_sizes:
        # Append the cheap-size variants: candidate axis C -> 2C. Everything
        # downstream (scoring, FFA bonuses, greedy) is shape-generic over C.
        cand_src = torch.cat([cand_src, cand_src], dim=0)
        cand_tgt_slot = torch.cat([cand_tgt_slot, cand_tgt_slot], dim=0)
        cand_tgt_short = torch.cat([cand_tgt_short, cand_tgt_short], dim=0)
        cand_send = torch.cat(
            [cand_send, torch.where(valid2, sizes2, torch.zeros_like(sizes2)).reshape(C, L)], dim=0
        )
        cand_angle = torch.cat([cand_angle, angle2.reshape(C, L)], dim=0)
        cand_eta = torch.cat(
            [cand_eta, torch.where(valid2, eta2, torch.ones_like(eta2)).reshape(C, L)], dim=0
        )
        cand_active = torch.cat([cand_active, valid2.reshape(C, L)], dim=0)
        cand_valid = torch.cat([cand_valid, valid2.reshape(C)], dim=0)
        cand_is_def = torch.cat([cand_is_def, cand_is_def], dim=0)
        C = 2 * C

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )                                                                            # [C]
    if int(player_count) >= 4 and (
        float(config.ffa_leader_attack_bonus) > 0.0
        or float(config.ffa_target_prod_bonus) > 0.0
    ):
        owner = obs.owner_abs.to(torch.long)
        owner_valid = (owner >= 0) & (owner < int(player_count)) & obs.alive
        owner_idx = owner.clamp(min=0, max=max(int(player_count) - 1, 0))
        prod_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        ships_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        prod_by_owner.scatter_add_(0, owner_idx, torch.where(owner_valid, prod.to(dtype), torch.zeros_like(prod.to(dtype))))
        ships_by_owner.scatter_add_(0, owner_idx, torch.where(owner_valid, obs.ships.to(dtype), torch.zeros_like(obs.ships.to(dtype))))
        strength = prod_by_owner + 0.025 * ships_by_owner
        my_strength = strength[pid].detach()

        target_owner = owner[target_idx.clamp(0, P - 1)].clamp(min=0, max=max(int(player_count) - 1, 0))
        target_owned_enemy = (
            target_exists
            & obs.is_enemy[target_idx.clamp(0, P - 1)]
            & (obs.owner_abs[target_idx.clamp(0, P - 1)] >= 0)
        )
        owner_strength = strength[target_owner]
        leader_delta = (owner_strength - my_strength).clamp(min=0.0)
        target_bonus_short = torch.where(
            target_owned_enemy,
            float(config.ffa_leader_attack_bonus) * leader_delta
            + float(config.ffa_target_prod_bonus) * prod[target_idx.clamp(0, P - 1)].to(dtype),
            torch.zeros_like(owner_strength),
        )
        score = score + target_bonus_short[cand_tgt_short]
    if int(player_count) >= 4 and (
        float(config.ffa_near_opponent_mult) != 1.0
        or float(config.ffa_far_opponent_mult) != 1.0
    ):
        # v5: nearest-opponent priority. Rank opponents by mean planet-to-planet
        # distance from our owned planets; boost scores on the nearest opponent's
        # planets, damp the others. Neutral and owned targets are untouched.
        owner = obs.owner_abs.to(torch.long)
        mine_mask = obs.owned & obs.alive
        d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt]
        opp_dist = torch.full((int(player_count),), float("inf"), dtype=dtype, device=device)
        for o in range(int(player_count)):
            if o == pid:
                continue
            om = obs.alive & (owner == o)
            if bool(mine_mask.any()) and bool(om.any()):
                opp_dist[o] = d0[mine_mask][:, om].mean()
        nearest = int(torch.argmin(opp_dist).item())
        if bool(torch.isfinite(opp_dist[nearest])):
            t_idx = target_idx.clamp(0, P - 1)
            t_owner = owner[t_idx]
            t_enemy = target_exists & obs.is_enemy[t_idx] & (t_owner >= 0)
            mult_short = torch.where(
                t_enemy & (t_owner == nearest),
                torch.full((T,), float(config.ffa_near_opponent_mult), dtype=dtype, device=device),
                torch.where(
                    t_enemy,
                    torch.full((T,), float(config.ffa_far_opponent_mult), dtype=dtype, device=device),
                    torch.ones(T, dtype=dtype, device=device),
                ),
            )
            score = score * mult_short[cand_tgt_short]
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))

    # v5.5 Axis C: learned value of each candidate's first-order resulting board,
    # consumed only as a near-tie re-rank inside the greedy. Computed once over all
    # candidates (cheap: ~C tiny-MLP rows). OFF (eps<=0 or no model) => None =>
    # greedy is byte-identical to v5.4.
    cand_value = None
    if float(config.value_rerank_eps) > 0.0 and _VALUE_MODEL is not None:
        cand_value = candidate_value_scores(
            obs=obs, prod=prod, obs_tensors=obs_tensors,
            target_idx=target_idx, cand_tgt_short=cand_tgt_short,
            cand_send=cand_send, cand_active=cand_active,
            model=_VALUE_MODEL, player_count=int(player_count), player_id=pid,
        )

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
        cand_value=cand_value, value_rerank_eps=float(config.value_rerank_eps),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    # Reuse the enemy-mass proxy already computed above (one [P, P] reduction
    # serves both the reinforcement floor and this regroup gradient).
    assert enemy_mass is not None
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


# --- v5.4 Track 1: level-1 opponent-aware planning --------------------------
# Names of the mutable projection tensors on PlanetMovement that
# ``record_fleet_arrivals`` + ``garrison_status`` touch. We snapshot/restore
# exactly these around a hypothetical opponent injection so the persistent
# rolling cache (and next turn's projection) is byte-identical to never having
# injected — i.e. the opponent model is a pure read of "what would the board
# look like if the enemy launched its best response".
_PROJECTION_STATE = (
    "fleet_buckets",
    "garrison_owner_cache",
    "garrison_ships_cache",
    "garrison_pre_combat_owner_cache",
    "garrison_pre_combat_ships_cache",
    "garrison_dirty_from",
)


def _snapshot_projection(movement) -> dict:
    return {
        name: (t.clone() if (t := getattr(movement, name, None)) is not None else None)
        for name in _PROJECTION_STATE
    }


def _restore_projection(movement, snap: dict) -> None:
    for name, t in snap.items():
        setattr(movement, name, t)


def _opponent_reactive_status(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    base_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
    H: int,
):
    """Inject each live enemy's level-0 best-response and return a reactive status.

    For every opponent seat ``o != pid`` with a launchable garrison, re-parse the
    SAME observation from ``o``'s perspective (everything but the ownership masks
    is absolute) and run the producer planner for ``o`` against the *do-nothing*
    baseline projection (``base_status`` — so each opponent is level-0, assuming
    everyone else holds). Their best-response ATTACK launches (regroup off; capped
    at ``opp_inject_waves`` waves) are injected into the projection's arrival
    buckets; ``garrison_status`` then re-resolves the engine-exact timeline with
    those fleets present. The projection is snapshotted and restored so this has
    no effect on the persistent cache. Returns ``base_status`` unchanged when no
    enemy launches (so the result is exactly the do-nothing status in that case).
    """
    pid = int(obs.player_id)
    n = int(player_count)
    opp_waves = int(config.opp_inject_waves)
    # Opponent sub-plan: attacks only (regroup is internal logistics that mostly
    # shores up enemy defense → modeling it pushes US toward passivity, the
    # Cluster-9 failure mode; the threat that should reshape our plan is the
    # enemy's offense). Faithful otherwise — same horizon / reinforce-risk / FFA
    # knobs, so the model is "the opponent is also v5.x".
    opp_config = dataclasses.replace(
        config, enable_regroup=False, max_waves_per_turn=opp_waves
    )

    tgt_chunks, own_chunks, ship_chunks, eta_chunks, valid_chunks = [], [], [], [], []
    for o in range(n):
        if o == pid:
            continue
        obs_o = parse_obs(obs_tensors, player_id=o)
        launchable = (
            obs_o.owned & obs_o.alive & (obs_o.ships >= float(opp_config.min_ships_to_launch))
        )
        if not bool(launchable.any()):
            continue
        entries_o = plan_lite_waves(
            movement=movement, obs=obs_o, obs_tensors=obs_tensors, cache=cache,
            garrison_status=base_status, prod=prod, alive_by_step=alive_by_step,
            config=opp_config, player_count=n,
        )
        launches_o = infer_planned_launches_from_entries(
            obs_tensors=obs_tensors, movement=movement, entries=entries_o, player_id=o,
        )
        if not bool(launches_o.valid.any()):
            continue
        tgt_chunks.append(launches_o.target_slots)
        own_chunks.append(torch.full_like(launches_o.target_slots, int(o)))
        ship_chunks.append(launches_o.ships)
        eta_chunks.append(launches_o.eta_turns)
        valid_chunks.append(launches_o.valid)

    if not tgt_chunks:
        return base_status

    snap = _snapshot_projection(movement)
    try:
        movement.record_fleet_arrivals(
            target_slots=torch.cat(tgt_chunks),
            owner_ids=torch.cat(own_chunks),
            ships=torch.cat(ship_chunks),
            eta=torch.cat(eta_chunks),
            valid=torch.cat(valid_chunks),
        )
        reactive = movement.garrison_status(max_horizon=H)
    finally:
        _restore_projection(movement, snap)
    return reactive


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    # v5: clamp the whole planning horizon to the turns actually remaining, so the
    # flow-diff cannot bank production past the final step. Near the end this
    # yields capture-refusal (combat losses no longer recouped) and total-war
    # drains (garrisons need not be held past game end) for free. The clamp goes
    # through config (not garrison_status(max_horizon=...), whose sub-horizon path
    # is broken upstream) so movement/cache/projection shapes stay consistent; the
    # movement cache rebuilds per turn over the final ~horizon steps only.
    step = int(obs_tensors["step"].reshape(-1)[0].item())
    h_rem = max(1, min(int(config.horizon), END_STEP - step))
    if h_rem < int(config.horizon):
        config = dataclasses.replace(config, horizon=h_rem)

    # v5.2: terminal phase (see ProducerLiteConfig) — swap to the aggressive
    # endgame knobs for the final terminal_phase_turns turns.
    if int(config.terminal_phase_turns) > 0 and step >= END_STEP - int(config.terminal_phase_turns):
        config = dataclasses.replace(
            config,
            roi_threshold=float(config.terminal_roi_threshold),
            max_waves_per_turn=int(config.terminal_max_waves_per_turn),
            enable_regroup=bool(config.terminal_enable_regroup),
        )

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    # v5.4 Track 1: replace the do-nothing projection with a level-1 reactive one
    # (each enemy's best-response attacks injected) before scoring our candidates.
    # OFF (opp_inject_waves == 0) => skipped => status unchanged => byte-identical.
    if int(config.opp_inject_waves) > 0 and int(player_count) >= 2:
        status = _opponent_reactive_status(
            movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
            base_status=status, prod=movement.planet_prod, alive_by_step=alive_by_step,
            config=config, player_count=int(player_count), H=H,
        )

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_offensive_targets=7,
    max_defensive_targets=2,
    roi_threshold=1.55,
    min_ships_to_launch=5.0,
    max_regroup_time=6.0,
    max_regroup_targets_per_source=8,
    ffa_leader_attack_bonus=0.035,
    ffa_target_prod_bonus=0.08,
    # v5: nearest-opponent priority OFF (was 1.25x/0.55x per the distance-1100
    # lineage; neutral in local 4P arena but the ladder converged v5 71 pts below
    # producer 2026-06-11 — this was the most-active delta, so reverted to parity).
    ffa_near_opponent_mult=1.0,
    ffa_far_opponent_mult=1.0,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    moves = sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
    if _VALIDATOR is not None:
        moves = apply_veto(moves, obs, _VALIDATOR, _VETO_THRESHOLD)
    return moves
