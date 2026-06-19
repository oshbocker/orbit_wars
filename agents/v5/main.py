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
from orbit_lite_v5.contestation import plan_contestation_waves
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

# RESEARCH HOOK (macro-action pointer-BC, not shipped). When set to a callable by an
# external driver, it OVERRIDES the flow-diff candidate score with a learned selector
# (so v5's EXACT candidate generation + intercept_angle + safe_drain sizing execute the
# net's source->target picks) and disables the ROI gate. None by default => the flow
# scorer is authoritative => byte-identical to the shipped agent.
# Signature: fn(obs_tensors, cand_src, cand_tgt_slot, cand_valid, score) -> score[C] or None.
# The selector receives the EXACT per-candidate flow-diff score so a producer-prior
# selector can return it unchanged (or score + a learned delta) and reproduce v5.
_SELECTOR_FN = None
# When True, KEEP v5's real ROI gate (config.roi_threshold) instead of disabling it —
# only meaningful with a selector that returns scores on the real Δnet scale (the rich
# residual design). Default False preserves the legacy "fire all picks" behavior.
_SELECTOR_KEEP_ROI = False

# RESEARCH HOOK (rich-representation feature extraction, not shipped). When set to a dict,
# plan_lite_waves writes its exact per-candidate projection tensors into it (producer's own
# Δnet score, ETA, sizes, capture floor) plus the garrison_status timelines — so the net can
# be fed the SAME information producer reasons over, instead of re-deriving it from a
# snapshot. None by default => no capture => byte-identical.
_FEATURE_SINK = None


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs."""

    # the projection window, the movement build length, AND the target ETA cap
    horizon: int = 18
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12  # enemy/neutral proximity targets
    max_defensive_targets: int = 4
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 6
    roi_threshold: float = 1.5  # fire if score > this
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
    # BUCKETED CONFIG (v5.5, phase-conditioned knobs — 2026-06-19). The terminal
    # swap above already proves a knob's optimum can differ by game PHASE; this is
    # the symmetric OPENING swap (first ``opening_phase_turns`` steps). Bucketing
    # then = opening / midgame (the base defaults) / terminal (above). Gated
    # default-OFF + byte-identical when off: B1 (2P) fires only when
    # ``opening_phase_turns > 0`` AND swaps ``roi_threshold`` to
    # ``opening_roi_threshold`` (default 1.5 == base ⇒ NO-OP even if turns>0); B2
    # (4P) additionally swaps ``ffa_target_prod_bonus`` to
    # ``opening_ffa_target_prod_bonus`` only when ``player_count >= 4`` and the
    # value is >= 0 (sentinel -1.0 ⇒ keep base). 2P and 4P NEVER share a schedule:
    # CONFIG_4P keeps opening_phase_turns=0 by default so 2P doses don't touch 4P,
    # and the ffa swap is 4P-only. Dose via arena ``v5:opening_phase_turns=40+
    # opening_roi_threshold=1.0`` (no code edit per dose). Hypothesis: opening
    # wants LOWER roi (grab uncontested land early; production compounds).
    opening_phase_turns: int = 0
    opening_roi_threshold: float = 1.5  # == base roi_threshold ⇒ no-op when equal
    opening_ffa_target_prod_bonus: float = -1.0  # <0 sentinel ⇒ keep base (4P only)
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
    # CONTESTATION OVERLAY (new). The flow-diff sizes captures CAPTURE-MINIMALLY
    # (``capture_floor`` = ceil(projected defenders + overhead)), so a freshly
    # captured planet holds only ~overhead + 1 turn of production — verified thin
    # in the engine combat (top-second, then survivor-vs-garrison). This overlay
    # runs the EXACT opponent planner (reusing ``_opponent_reactive_status``) to
    # predict which neutral/own planets an opponent will capture-thin within the
    # horizon, then generates SNIPE candidates — fleets aimed to arrive just AFTER
    # the predicted capture, sized vs the REACTIVE (post-capture, thin) garrison,
    # scored by the same exact flow-diff and greedy-selected from the post-base-plan
    # LEFTOVER budget (never weakens the base expansion → no tempo tax). These are
    # candidates the base shortlist is structurally blind to (the target is a
    # full-garrison neutral at plan time, not a 2-ship enemy). DISTINCT from the
    # INERT ``opp_inject_waves`` (Cluster 11), which fed the opponent prediction into
    # our DEFENSE; this consumes it as new OFFENSE. 0 = OFF, byte-identical to v5.3.
    # SHIPPED ON in 2P (v5.4): the Tier-1 detector below gates this so it only fires
    # vs producer-family opponents (vs anything else => exactly plain v5.3). Gate
    # (scripts/arena.py, side-alternated paired seeds): contest+detector vs producer
    # +18 (75% vs 57%, n=125), vs producer_v2 +14 (45% vs 31%, n=103), vs tamrazov
    # 100%/100% (n>=52, no regression), vs ow_proto 100%/99% (n=130). 4P keeps this OFF
    # (CONFIG_4P) until 4P has its own gate. Set 0 for plain v5.3 (the off-switch).
    contest_waves: int = 2  # max snipe waves/turn (dose-response knob)
    contest_delay: int = 1  # arrive predicted-capture-turn + this
    contest_roi_threshold: float = 1.5  # exact flow-diff ROI gate for snipes
    contest_capture_overhead: float = 1.0
    contest_opp_waves: int = 6  # opp-planner wave cap when predicting captures
    contest_verify: int = 0  # reserved: 2nd-ply re-defense filter (v2)
    # TIER-1 producer detector (gates the contestation overlay; only meaningful when
    # contest_waves > 0 — byte-identical to v5.3 when the overlay is OFF). The overlay
    # only pays off vs producer-family opponents (we run producer's exact planner as the
    # opponent model); vs a structurally-different agent a snipe sized to a *predicted*
    # capture that never happens wastes leftover ships. So we run the producer model every
    # turn, predict each enemy seat's launches, and verify next turn against the seat's
    # actually-launched fleets via source-set PRECISION (of the from_planet_ids the model
    # predicted, how many it actually fired from — robust to producer_v2's sizing
    # differences; we want to snipe it too). A per-seat precision EMA gates the overlay
    # ON only once it clears ``contest_fidelity_threshold`` with at
    # least ``contest_min_observations`` measured turns — biased toward OFF so a wrong call
    # only forgoes upside (never wastes ships on a full-garrison planet). Makes
    # contest+detector strictly >= plain v5 everywhere. contest_detect=0 disables the gate
    # (overlay always on — for un-gated A/B measurement); 1 = gated (default, the shipped
    # behavior). Detection separates producer-clones (~60-77%) from different agents (~14-27%).
    contest_detect: int = 1
    # Gate calibrated offline on per-turn precision sequences (5 games/opp, grid over
    # alpha/threshold/min_obs): at alpha=0.9 / thr=0.55 / min_obs=8 the gate is ON for
    # producer 100% of measured turns, producer_v2 43%, tamrazov 19%, ow_proto 0% —
    # biased toward OFF on non-producers (the binding "strictly >= plain v5" constraint)
    # while keeping both producer-clones snipeable. alpha is the EMA weight on HISTORY
    # (higher => more stable, separates on the well-separated per-opponent means rather
    # than noisy 3-turn spikes).
    contest_fidelity_threshold: float = 0.55
    contest_min_observations: int = 8
    contest_fidelity_alpha: float = 0.9
    # TIER-2 opponent-model ensemble (v5.6). The single Tier-1 detector models every
    # opponent as the BASE producer (reinforce_size_beta=0.0), so it OVER-predicts
    # producer_v2's launch sources: producer_v2 adds the slawekbiel V2 ETA-aware
    # reinforcement-risk capture floor (our ``reinforce_size_beta`` knob) and therefore
    # DECLINES captures the base model says it will make → source-set precision drops to
    # ~0.58 (gate ON only ~43% of producer_v2 turns vs ~99% for plain producer), forgoing
    # snipes worth a measured ~+4.5 win-rate (the gap from producer_v2 +11.7 to producer
    # +16.2). The ensemble keeps a SMALL set of opponent models — model 0 = base producer
    # (β=0), model 1 = producer_v2 (β=config.reinforce_size_beta + the two reinforce_eta_*
    # knobs kept ON) — scores each enemy seat's source-set fidelity under EACH (the tracker
    # holds a per-(seat,model) precision EMA), gates a seat ON when EITHER model clears the
    # bar, and snipes each gated seat from its BEST-matching model's reactive projection.
    # plain producer keeps scoring highest under model 0; producer_v2 now reads ~1.0 under
    # model 1 → gated ON far more often → the forgone snipes return. Adds a 2nd planner pass
    # per seat (cheap in 2P: one extra opponent seat; well under the 1s budget — confirmed).
    # 0 = OFF, byte-identical to v5.4 (detector is exactly the single base-producer model).
    # 4P keeps this OFF regardless (CONFIG_4P.contest_waves=0 → overlay never runs).
    contest_ensemble: int = 0
    # v5.5 4P FFA board-position gate. In 2P the contest snipe is pure value (the
    # thinned planet belongs to the sole opponent → taking it directly swings the
    # zero-sum). In 4P it is NOT: a snipe spends leftover ships to grab a planet that
    # becomes OUR exposed frontier in a multi-way fight, and the specific planet a
    # clone thins depends on the other live seats' moves our level-0 model can't see,
    # so snipe targets mis-fire and we overextend (measured: plain 4P enabling
    # regressed rank 1.65 vs 1.45 / win 35% vs 55% / end-score 720 vs 3027 vs
    # contest_waves=0). FFA doctrine = strike from strength: only fire snipes when our
    # board strength (prod + 0.025·ships, the same FFA strength used by the leader
    # bonus) ranks within the top ``contest_ffa_strike_rank`` of the live players —
    # when behind we hold ships and let the others fight. Consulted ONLY when
    # player_count >= 4 (2P byte-identical regardless of value). 0 = OFF (no
    # board gate — every gated seat sniped, the regressing behavior); 1 = leader/
    # tied-leader only; 2 = top-2; etc. Detection still runs every turn (fidelity EMA
    # keeps updating) so the gate is ready the moment we take the lead.
    # CLOSED 2026-06-18: rank=1 (leader-only) was the best variant but only reached
    # pooled n=180 net -3 vs off (+7 on a 2-clone table, but -4 all-clone and -6 on a
    # 1-clone "sparse" table — it fires when we're already cruising, exactly when an
    # FFA leader should consolidate). Did not clear ">= plain v5.3" → kept default-OFF
    # infra. Reopen for the "contested-leader only" refinement (gate on a SMALL lead).
    contest_ffa_strike_rank: int = 0
    # v5.4 (top-tier replay diagnostic): half-drain structural delta. The #1 ladder
    # agent (Isaiah @ Tufa Labs, 1762) sends ~half a garrison at a time (median
    # send-fraction 0.52) and beats producer-family full-drain clones; producer/v5
    # ship the full ``safe_drain`` (~1.0). This is a POST-SELECTION cap: the exact
    # flow-diff scorer still ranks full-drain candidates (so WHICH waves fire is
    # producer-identical), but each chosen full-drain wave then ships at most
    # ``(1 - reserve_frac)`` of its source's current garrison, holding the rest home
    # — UNLESS the cap is *decisive* (the trimmed fleet would no longer reach in time
    # or no longer clear the target's capture floor at its slower arrival, in which
    # case the full send goes). DISTINCT from Cluster 7 (no extra candidate is scored
    # — the scorer never sees two sizes) and Cluster 9 (a flat fraction of the
    # source, NOT a ``cheap_enemy_pressure`` mass-proxy subtraction). 0.0 = OFF,
    # byte-identical to v5.3. CONFIG_4P inherits via ``dataclasses.replace``.
    reserve_frac: float = 0.0
    # H1 4P HOLDING-PAD (v5.5, 06-18 replay diagnostic). Our own ladder replays show
    # the 4P rating leak is "capture-then-collapse": we expand to ~5.5 planets by
    # step 60, then cannot hold them under multi-opponent pressure and get eliminated
    # ~step 114 (4P win rate 30% vs 2P 51%; loss/win move-style IDENTICAL ⇒ it's not
    # sizing, it's whether captures HOLD). The do-nothing ``capture_floor`` only sizes
    # a capture to clear the projected defenders + overhead, so a fresh frontier
    # capture lands thin and a 3rd-party counter-wave (which the level-0 projection
    # can't see) retakes it. This knob inflates the capture floor of CONTESTED targets
    # (reachable by >= ``hold_min_opponents`` distinct live opponents) by
    # ``hold_margin_beta * largest-single-opponent reachable garrison`` — combat is
    # ``top - second`` then survivor-vs-garrison, so the binding counter-wave is the
    # single largest opponent, not the sum. Effect: contested captures route to
    # sources rich enough to OVER-FUND them (full ``safe_drain`` >> inflated floor ⇒
    # the capture holds the immediate counter-wave) and thin-source grabs that would
    # churn are DECLINED — "hold-quality, not fewer captures". DISTINCT from
    # reinforce_size_beta (models reinforcement arriving DURING flight to a planet the
    # enemy already owns; this models a post-capture counter-wave on a contested
    # NEUTRAL, gated on >=2 distinct opponents and sized to the largest single one),
    # from Cluster 9 defense_size_beta (reserves on the SOURCE → passivity; this sizes
    # the TARGET capture), and from Cluster 7/12 (those send LESS). 4P-ONLY (guarded by
    # player_count>=4) and gated by hold_margin_beta>0 ⇒ byte-identical to v5.4 in 2P
    # and when OFF. Gate via scripts/arena.py --players 4 + the prod-share@80 screen.
    hold_margin_beta: float = 0.0
    hold_min_opponents: int = 2


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
    d0 = cache.cross_dist[0].to(dtype)  # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))  # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)  # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye  # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)  # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)  # [P] summed over sources


def opponent_holding_pressure(
    obs,
    cache,
    *,
    horizon: float,
    player_id: int,
    player_count: int,
) -> tuple[Tensor, Tensor]:
    """Per-planet ``(largest single opponent reachable mass [P], #opponents that can
    reach it [P] long)`` — the 4P holding-pad's counter-wave model.

    Like ``cheap_enemy_pressure`` but split BY OWNER: for each live opponent seat,
    sum the distance-decayed share of its sources' current garrison that could
    straight-line reach each planet within ``horizon`` turns, then reduce across
    opponents to ``(max-over-opponents mass, count of opponents with positive mass)``.
    The MAX (not the sum) is the combat-relevant counter-wave size — a fresh capture
    meets ``top - second`` then survivor-vs-garrison, so the binding threat is the
    single largest opponent. Same approximations as ``cheap_enemy_pressure`` (ignores
    target drift / in-flight production / in-flight enemy fleets). Pure arithmetic on
    cached tensors.
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        z = torch.zeros(P, dtype=dtype, device=device)
        return z, torch.zeros(P, dtype=torch.long, device=device)
    d0 = cache.cross_dist[0].to(dtype)  # [src, tgt]
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))  # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)  # [src, 1]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)  # [src, tgt]
    owner = obs.owner_abs.to(torch.long)
    eye = torch.eye(P, device=device, dtype=torch.bool)
    largest = torch.zeros(P, dtype=dtype, device=device)
    count = torch.zeros(P, dtype=torch.long, device=device)
    for o in range(int(player_count)):
        if o == int(player_id):
            continue
        src_o = obs.alive & (owner == o)  # [P]
        if not bool(src_o.any()):
            continue
        valid = src_o.view(P, 1) & obs.alive.view(1, P) & ~eye  # [src, tgt]
        contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
        mass_o = contrib.sum(dim=0)  # [P] per target from opp o
        largest = torch.maximum(largest, mass_o)
        count = count + (mass_o > 0.0).to(torch.long)
    return largest, count


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
        obs,
        obs_tensors,
        garrison_status,
        cache,
        config=config,
        K_eta=K_eta,
        H=H,
        prod=prod,
        source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]  # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)  # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)

    # Reachable-enemy-mass proxy ([P]) — computed ONCE and reused for ALL THREE of:
    # the offensive reinforcement-risk floor margin (below), the v5.4 defensive
    # reserve (safe_drain, right here), and the regroup gradient (further down). Its
    # decay distance-scale is the attack reach K_eta. (Producer V2 + v5.4 symmetry.)
    beta = float(config.reinforce_size_beta)
    defense_beta = float(config.defense_size_beta)
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
        if beta > 0.0 or defense_beta > 0.0 or bool(config.enable_regroup)
        else None
    )

    # v5.4 defensive symmetry: hold back ``defense_beta * enemy_mass(source)`` ships
    # on each source so the planner under-commits planets the enemy can mass on
    # (None when OFF → safe_drain byte-identical to v5.3).
    defense_reserve = (
        defense_beta * enemy_mass[source_idx.clamp(0, P - 1)]  # [S]
        if defense_beta > 0.0 and enemy_mass is not None
        else None
    )
    drain = safe_drain(
        garrison_status,
        source_idx=source_idx,
        source_ships=source_ships,
        H_eff=H_eff,
        player_id=pid,
        reserve=defense_reserve,
    )  # [S]

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)  # [T]

    # ETA-aware reinforcement risk: inflate the capture floor by ``beta * rho(k)
    # * reachable-enemy-mass(target)``. The per-arrival-turn growth comes from
    # the rho(k) timing ramp. Gated by beta > 0 (OFF = bare floor, byte-identical).
    reinforcement = None
    if beta > 0.0 and enemy_mass is not None:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]  # [T]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange,
            eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )  # [K_eta]
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)  # [T, K_eta]

    # H1 4P holding-pad (see ProducerLiteConfig.hold_margin_beta): inflate the capture
    # floor of CONTESTED targets (reachable by >= hold_min_opponents distinct live
    # opponents) by ``hold_margin_beta * largest-single-opponent reachable garrison``,
    # so a fresh capture is sized to survive the immediate counter-wave (top-vs-second
    # ⇒ the largest single opponent is the binding threat). Routes contested captures
    # to sources that can over-fund them and declines thin-source grabs. Flat over k
    # (the counter-wave threat is independent of MY arrival turn). 4P-only + gated ⇒
    # byte-identical to v5.4 in 2P and when OFF; only inflates CAPTURE cells (owned
    # reinforcement cells stay floor 1 inside ``capture_floor``).
    hold_beta = float(config.hold_margin_beta)
    if int(player_count) >= 4 and hold_beta > 0.0 and K_eta > 0:
        largest_opp, opp_count = opponent_holding_pressure(
            obs,
            cache,
            horizon=float(K_eta),
            player_id=pid,
            player_count=int(player_count),
        )  # [P], [P]
        contested = opp_count >= int(config.hold_min_opponents)  # [P]
        hold_mass = torch.where(contested, largest_opp, torch.zeros_like(largest_opp))
        hold_margin = (hold_beta * hold_mass[target_idx.clamp(0, P - 1)]).view(T, 1)  # [T, 1]
        reinforcement = (
            hold_margin.expand(T, K_eta).contiguous()
            if reinforcement is None
            else reinforcement + hold_margin
        )  # [T, K_eta]

    floor = capture_floor(
        garrison_status,
        target_idx=target_idx,
        k_max=K_eta,
        capture_overhead=1.0,
        player_id=pid,
        reinforcement=reinforcement,
    )  # [T, K]
    K = int(floor.shape[-1])

    # --- single fleet size = the max garrison launch (safe_drain) ---------------
    # Engine needs integer ship counts; floor (never exceed what's available).
    sizes = drain.view(S, 1).expand(S, T).floor()  # [S, T]

    # Strict-superset reachability precheck (always on): defers the body screen to
    # candidates that can physically reach the target in time.
    active = reachable_mask(
        movement,
        source_idx=source_idx,
        target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1),
        eta_cap=eta_cap,
    ).squeeze(-1)  # [S, T]
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),  # [S, 1]
        target_idx.unsqueeze(0),  # [1, T]
        sizes,  # [S, T]
        active=active,
    )
    angle = aim["angle"]  # [S, T]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    # Capture-floor gate at each fleet's arrival turn (defenders grow with k). The
    # single size must clear the defender it lands on (size >= floor_at_arr). Owned
    # targets have floor 1 (reinforcement), so any positive send clears.
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)  # [S,T]
        floor_at_arr = (
            floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
        )
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr  # [S, T]

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable
        & clears_floor
        & (sizes >= 1.0)
        & src_neq_tgt
        & source_exists.view(S, 1)
        & target_exists.view(1, T)
    )  # [S, T]

    # --- v5: optional second, cheaper size per (source, target) -----------------
    # Just-enough-to-capture: ceil(floor_at_arr + margin), capped at the drain.
    # The smaller fleet is slower, so its floor is re-gated at its OWN (later)
    # arrival turn; if defenders outgrow the margin in flight the variant drops
    # and the full-size candidate stays on the board. Greedy's one-wave-per-target
    # mask keeps the two sizes mutually exclusive; the scorer picks between them.
    two_sizes = float(config.cheap_capture_margin) >= 0.0
    if two_sizes:
        sizes2 = (floor_at_arr + float(config.cheap_capture_margin)).ceil().clamp(min=1.0)
        sizes2 = torch.minimum(sizes2, sizes)  # [S, T]
        # Only when strictly cheaper than the drain candidate, and never on owned
        # targets (their floor is 1 — a token reinforcement is junk).
        distinct = (sizes2 < sizes) & ~target_is_mine.view(1, T)
        active2 = (
            reachable_mask(
                movement,
                source_idx=source_idx,
                target_idx=target_idx,
                fleet_sizes=sizes2.unsqueeze(-1),
                eta_cap=eta_cap,
            ).squeeze(-1)
            & distinct
        )
        aim2 = intercept_angle(
            movement,
            source_idx.unsqueeze(1),
            target_idx.unsqueeze(0),
            sizes2,
            active=active2,
        )
        angle2 = aim2["angle"]  # [S, T]
        eta2 = aim2["eta"]
        viable2 = aim2["viable"] & (eta2 <= eta_cap.view(1, T))
        if K > 0:
            k_arr2 = (eta2.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr2 = (
                floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr2.unsqueeze(-1)).squeeze(-1)
            )
        else:
            floor_at_arr2 = torch.ones(S, T, dtype=dtype, device=device)
        valid2 = (
            viable2
            & (sizes2 >= floor_at_arr2)
            & (sizes2 >= 1.0)
            & distinct
            & src_neq_tgt
            & source_exists.view(S, 1)
            & target_exists.view(1, T)
        )  # [S, T]

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
    cand_is_def = target_is_mine[cand_tgt_short]  # [C]
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
        garrison_status,
        prod=prod,
        alive_by_step=alive_by_step,
        player_count=int(player_count),
        launches=launches,
        player_id=pid,
    )  # [C]
    if int(player_count) >= 4 and (
        float(config.ffa_leader_attack_bonus) > 0.0 or float(config.ffa_target_prod_bonus) > 0.0
    ):
        owner = obs.owner_abs.to(torch.long)
        owner_valid = (owner >= 0) & (owner < int(player_count)) & obs.alive
        owner_idx = owner.clamp(min=0, max=max(int(player_count) - 1, 0))
        prod_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        ships_by_owner = torch.zeros(int(player_count), dtype=dtype, device=device)
        prod_by_owner.scatter_add_(
            0, owner_idx, torch.where(owner_valid, prod.to(dtype), torch.zeros_like(prod.to(dtype)))
        )
        ships_by_owner.scatter_add_(
            0,
            owner_idx,
            torch.where(owner_valid, obs.ships.to(dtype), torch.zeros_like(obs.ships.to(dtype))),
        )
        strength = prod_by_owner + 0.025 * ships_by_owner
        my_strength = strength[pid].detach()

        target_owner = owner[target_idx.clamp(0, P - 1)].clamp(
            min=0, max=max(int(player_count) - 1, 0)
        )
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
        float(config.ffa_near_opponent_mult) != 1.0 or float(config.ffa_far_opponent_mult) != 1.0
    ):
        # v5: nearest-opponent priority. Rank opponents by mean planet-to-planet
        # distance from our owned planets; boost scores on the nearest opponent's
        # planets, damp the others. Neutral and owned targets are untouched.
        owner = obs.owner_abs.to(torch.long)
        mine_mask = obs.owned & obs.alive
        d0 = cache.cross_dist[0].to(dtype)  # [src, tgt]
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
                    torch.full(
                        (T,), float(config.ffa_far_opponent_mult), dtype=dtype, device=device
                    ),
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
            obs=obs,
            prod=prod,
            obs_tensors=obs_tensors,
            target_idx=target_idx,
            cand_tgt_short=cand_tgt_short,
            cand_send=cand_send,
            cand_active=cand_active,
            model=_VALUE_MODEL,
            player_count=int(player_count),
            player_id=pid,
        )

    # v5.4 half-drain reserve cap (see ProducerLiteConfig.reserve_frac). Applied
    # AFTER scoring (the flow-diff ranked full-drain candidates) and BEFORE the
    # greedy ships them: trim each full-drain candidate to at most (1-reserve_frac)
    # of its source garrison, re-gating the slower trimmed fleet for reachability
    # and capture-floor clearance; where the trim would break the send (unreachable
    # or no longer captures) the full-drain send stays = "decisive". Only the
    # full-drain block (first S*T candidates) is capped — cheap-size variants, when
    # present, are already minimal. OFF (reserve_frac<=0) => skipped => byte-identical.
    reserve_frac = float(config.reserve_frac)
    if reserve_frac > 0.0:
        cap = ((1.0 - reserve_frac) * source_ships).view(S, 1)  # [S,1]
        sizes_r = torch.minimum(sizes, cap).floor().clamp(min=1.0)  # [S,T]
        trim = (sizes_r < sizes) & valid  # [S,T]
        active_r = (
            reachable_mask(
                movement,
                source_idx=source_idx,
                target_idx=target_idx,
                fleet_sizes=sizes_r.unsqueeze(-1),
                eta_cap=eta_cap,
            ).squeeze(-1)
            & trim
        )
        aim_r = intercept_angle(
            movement,
            source_idx.unsqueeze(1),
            target_idx.unsqueeze(0),
            sizes_r,
            active=active_r,
        )
        angle_r = aim_r["angle"]  # [S,T]
        eta_r = aim_r["eta"]
        viable_r = aim_r["viable"] & (eta_r <= eta_cap.view(1, T))
        if K > 0:
            k_arr_r = (eta_r.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
            floor_at_arr_r = (
                floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr_r.unsqueeze(-1)).squeeze(-1)
            )
        else:
            floor_at_arr_r = torch.ones(S, T, dtype=dtype, device=device)
        # "decisive" = the trimmed fleet can't reach or can't clear the floor at its
        # slower arrival -> keep the full-drain send.
        apply_r = trim & viable_r & (sizes_r >= floor_at_arr_r)  # [S,T]
        sizes_f = torch.where(apply_r, sizes_r, sizes)
        angle_f = torch.where(apply_r, angle_r, angle)
        eta_f = torch.where(apply_r, eta_r, eta)
        C0 = S * T
        cand_send[:C0] = torch.where(valid, sizes_f, torch.zeros_like(sizes_f)).reshape(C0, L)
        cand_angle[:C0] = angle_f.reshape(C0, L)
        cand_eta[:C0] = torch.where(valid, eta_f, torch.ones_like(eta_f)).reshape(C0, L)

    # RESEARCH HOOK (see _FEATURE_SINK): dump producer's exact per-candidate projection
    # tensors for offline feature extraction. None => skipped => byte-identical.
    if _FEATURE_SINK is not None:
        planet_ids = obs_tensors["planets"][..., 0].long()
        _FEATURE_SINK.update(
            score=score.detach().clone(),  # [C] producer Δnet (FFA-adjusted)
            cand_src=cand_src.detach().clone(),  # [C,1] source ROW idx
            cand_tgt_slot=cand_tgt_slot.detach().clone(),  # [C]   target ROW idx
            cand_valid=cand_valid.detach().clone(),  # [C]
            cand_send=cand_send.detach().clone(),  # [C,1] exact ship size
            cand_eta=cand_eta.detach().clone(),  # [C,1] exact ETA
            planet_ids=planet_ids.detach().clone(),  # [P_rows] row -> planet id
            status_ships=garrison_status.ships.detach().clone(),  # [P_rows, H+1]
            status_owner=garrison_status.owner.detach().clone(),  # [P_rows, H+1]
            player_id=int(pid),
        )

    # RESEARCH HOOK (see _SELECTOR_FN): replace the flow-diff score with a learned
    # selector over the SAME exactly-generated candidates, and fire all picks (ROI off).
    # None => unchanged => byte-identical.
    roi_thresh = float(config.roi_threshold)
    if _SELECTOR_FN is not None:
        sel = _SELECTOR_FN(obs_tensors, cand_src, cand_tgt_slot, cand_valid, score)
        if sel is not None:
            score = torch.where(
                cand_valid, sel.to(score.dtype), torch.full_like(score, float("-inf"))
            )
            if not _SELECTOR_KEEP_ROI:
                roi_thresh = float("-inf")

    wave_entries, leftover = _greedy_select(
        P=P,
        W=W,
        device=device,
        dtype=dtype,
        score=score,
        cand_src=cand_src,
        cand_send=cand_send,
        cand_angle=cand_angle,
        cand_eta=cand_eta,
        cand_active=cand_active,
        cand_tgt_slot=cand_tgt_slot,
        cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def,
        source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists,
        roi_threshold=roi_thresh,
        cand_value=cand_value,
        value_rerank_eps=float(config.value_rerank_eps),
    )

    # Capture the GREEDY-FIRED launches for self-consistent BC labels (in-grid by
    # construction). planet-id indexed. None => skipped.
    if _FEATURE_SINK is not None:
        pid_map = obs_tensors["planets"][..., 0].long()
        fs = wave_entries.source_slots.long().clamp(0, pid_map.shape[0] - 1)
        ft = wave_entries.target_slots.long().clamp(0, pid_map.shape[0] - 1)
        _FEATURE_SINK.update(
            fired_src=pid_map[fs].detach().clone(),
            fired_tgt=pid_map[ft].detach().clone(),
            fired_valid=wave_entries.valid.detach().clone(),
        )

    if not bool(config.enable_regroup):
        return wave_entries
    # Reuse the enemy-mass proxy already computed above (one [P, P] reduction
    # serves both the reinforcement floor and this regroup gradient).
    assert enemy_mass is not None
    regroup_entries = _plan_regroup(
        movement=movement,
        obs=obs,
        obs_tensors=obs_tensors,
        garrison_status=garrison_status,
        leftover=leftover,
        original_ships=obs.ships.to(dtype),
        pressure=enemy_mass,
        config=config,
        H=H,
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


# CALIBRATION HOOK (not shipped). When set to a list, ``_OpponentTracker.observe``
# appends ``(seat, n_pred, n_obs, n_inter)`` per measured turn so a calibration
# driver can compute recall/precision/Jaccard offline and pick the gate metric +
# threshold. None by default => no overhead, no behavior change.
_DETECT_DEBUG = None


def _strength_rank(obs, *, prod: Tensor, player_count: int, player_id: int) -> int:
    """1-based rank of our board strength among LIVE players (1 = strongest).

    Strength = total production + 0.025 · total ships over owned, alive planets —
    the same composite the FFA leader bonus uses. Ties share the better rank (rank =
    1 + #players strictly stronger), so ``rank == 1`` means leader or tied-leader.
    """
    dtype = obs.ships.dtype
    device = obs.device
    n = int(player_count)
    owner = obs.owner_abs.to(torch.long)
    valid = (owner >= 0) & (owner < n) & obs.alive
    idx = owner.clamp(min=0, max=max(n - 1, 0))
    prod_by = torch.zeros(n, dtype=dtype, device=device)
    ships_by = torch.zeros(n, dtype=dtype, device=device)
    zeros = torch.zeros_like(prod.to(dtype))
    prod_by.scatter_add_(0, idx, torch.where(valid, prod.to(dtype), zeros))
    ships_by.scatter_add_(
        0, idx, torch.where(valid, obs.ships.to(dtype), torch.zeros_like(obs.ships.to(dtype)))
    )
    strength = prod_by + 0.025 * ships_by
    mine = strength[player_id]
    return 1 + int((strength > mine).sum().item())


class _OpponentTracker:
    """Per-(seat, model) producer-fidelity tracker that gates the contestation overlay.

    Each turn we predict, with each producer opponent model, the set of planets each
    enemy seat will launch FROM (``set_predictions``). Next turn we read the seat's
    freshly-spawned fleets (fleet ids absent last turn, carrying ``from_planet_id``)
    and score every model's prediction by source-set PRECISION — of the planets a
    model predicted, how many the opponent actually launched from (``observe``). A
    per-(seat, model) precision EMA drives ``gated_seats`` — the overlay snipes a seat
    once ANY model's fidelity clears the threshold with enough measured turns — and
    ``best_model`` names which model to snipe that seat with. Source-set (not
    source+size) match so producer_v2 — which differs only in sizing — still reads as
    producer-family (we want to snipe it too); precision (not recall/Jaccard) because
    the capture-minimal model under-predicts, which would let recall false-positive a
    passive non-producer that fires from an obvious planet.

    ``n_models == 1`` (the Tier-1 default) is the single base-producer model and is
    byte-identical to the pre-ensemble tracker; ``n_models == 2`` adds the producer_v2
    model (Tier 2, ``contest_ensemble``). Keys are ``(seat, model_index)``.
    """

    def __init__(self, *, n_players: int, player_id: int, n_models: int = 1) -> None:
        self.n = int(n_players)
        self.pid = int(player_id)
        self.n_models = int(n_models)
        self.seats = [o for o in range(self.n) if o != self.pid]
        self.models = list(range(self.n_models))
        self.fid: dict[tuple[int, int], float] = {
            (o, m): 0.0 for o in self.seats for m in self.models
        }
        self.turns_observed: dict[tuple[int, int], int] = {
            (o, m): 0 for o in self.seats for m in self.models
        }
        self.prev_fleet_ids: dict[int, set[int]] = {o: set() for o in self.seats}
        # None until the first prediction is made; a (possibly empty) frozenset after.
        self.pred_sources: dict[tuple[int, int], frozenset[int] | None] = {
            (o, m): None for o in self.seats for m in self.models
        }

    def observe(self, obs_tensors: dict, *, alpha: float) -> None:
        """Verify last turn's predictions against this turn's freshly-spawned fleets."""
        fleets = obs_tensors["fleets"]  # [F, 7]
        ids = fleets[..., 0].long().tolist()  # fleet id (alive: >= 0)
        owners = fleets[..., 1].long().tolist()  # absolute owner
        froms = fleets[..., 5].long().tolist()  # from_planet_id
        # Turn key for offline calibration only (no behavioral effect).
        dbg_step = (
            int(obs_tensors["step"].reshape(-1)[0].item()) if _DETECT_DEBUG is not None else -1
        )
        for o in self.seats:
            cur_ids: set[int] = set()
            obs_src: set[int] = set()
            prev = self.prev_fleet_ids[o]
            for fid_, ow, fr in zip(ids, owners, froms, strict=True):
                if fid_ < 0 or ow != o:
                    continue
                cur_ids.add(fid_)
                if fid_ not in prev:  # spawned since last turn
                    obs_src.add(fr)
            # Score by PRECISION of each model's prediction: of the planets a model said
            # the opponent (if it ran that producer variant) would launch FROM, how many
            # did it actually launch from this turn? This separates clones from different
            # agents cleanly (calibration n=3 games/opp: producer 0.99, producer_v2 0.69
            # vs tamrazov 0.51, ow_proto 0.52) where recall/Jaccard don't — our model
            # UNDER-predicts (it fires a capture-minimal subset), so recall false-positives
            # a passive agent that happens to launch from an obvious planet, while precision
            # asks the discriminating question "does the opponent do what this producer
            # variant would do here?". A turn carries signal for a model only when that
            # model predicted a launch (n_pred > 0); a turn it predicted nothing is skipped.
            for m in self.models:
                pred = self.pred_sources[(o, m)]
                if pred:
                    inter = len(pred & obs_src)
                    precision = inter / len(pred)
                    self.fid[(o, m)] = alpha * self.fid[(o, m)] + (1.0 - alpha) * precision
                    self.turns_observed[(o, m)] += 1
                    if _DETECT_DEBUG is not None:
                        _DETECT_DEBUG.append((dbg_step, o, m, len(pred), len(obs_src), inter))
            self.prev_fleet_ids[o] = cur_ids

    def set_predictions(self, sources_by_model: list[dict]) -> None:
        """Store this turn's per-model predicted launch sources for next-turn verification.

        ``sources_by_model[m]`` maps seat -> ``(frozenset_sources, pairs)``.
        """
        for m in self.models:
            sm = sources_by_model[m]
            for o in self.seats:
                self.pred_sources[(o, m)] = sm.get(o, (frozenset(), ()))[0]

    def gated_seats(self, *, threshold: float, min_obs: int) -> set[int]:
        """Seats where ANY model's fidelity clears the gate — biased OFF until proven."""
        return {
            o
            for o in self.seats
            if any(
                self.turns_observed[(o, m)] >= int(min_obs) and self.fid[(o, m)] >= float(threshold)
                for m in self.models
            )
        }

    def best_model(self, o: int, *, min_obs: int) -> int:
        """Index of the highest-fidelity model for seat ``o`` (ties → lowest index).

        Restricted to models with enough measured turns; falls back to model 0 when
        none qualify (the gate would not have opened the seat, but keep it total).
        """
        eligible = [m for m in self.models if self.turns_observed[(o, m)] >= int(min_obs)]
        if not eligible:
            return 0
        return max(eligible, key=lambda m: (self.fid[(o, m)], -m))


def _producer_baseline_config(
    config: ProducerLiteConfig, *, opp_waves: int, reinforce_beta: float
) -> ProducerLiteConfig:
    """Producer flow-diff opponent model (regroup off, capped waves).

    Strips every v5-specific knob the planner reads (defensive reserve, cheap-capture
    second size, half-drain reserve, value re-rank, FFA mults/bonuses) so a real
    producer scores highest in detection and snipes are sized vs producer's own
    capture-minimal projection. ``reinforce_beta`` selects the ENSEMBLE model variant:
    0.0 = base producer (Tier-1 / model 0); ``config.reinforce_size_beta`` = producer_v2
    (Tier-2 / model 1), which keeps the ``reinforce_eta_free``/``reinforce_eta_scale``
    ramp ``dataclasses.replace`` preserves.
    """
    return dataclasses.replace(
        config,
        enable_regroup=False,
        max_waves_per_turn=opp_waves,
        reinforce_size_beta=reinforce_beta,
        defense_size_beta=0.0,
        cheap_capture_margin=-1.0,
        reserve_frac=0.0,
        value_rerank_eps=0.0,
        ffa_leader_attack_bonus=0.0,
        ffa_target_prod_bonus=0.0,
        ffa_near_opponent_mult=1.0,
        ffa_far_opponent_mult=1.0,
    )


def _entries_sources(entries_o, planet_ids: Tensor, P: int):
    """``(frozenset_source_pids, pairs)`` for a seat's predicted launches.

    ``pairs`` is the ``(planet_id, ships)`` list; the tracker uses only the source-set
    (the frozenset) — size is kept for diagnostics.
    """
    v_o = entries_o.valid
    if not bool(v_o.any()):
        return (frozenset(), ())
    ss = entries_o.source_slots[v_o].clamp(0, P - 1)
    src_pids = planet_ids[ss].long().tolist()
    src_ships = entries_o.ships[v_o].tolist()
    pairs = tuple(
        (int(p), float(s)) for p, s in zip(src_pids, src_ships, strict=True) if int(p) >= 0
    )
    return (frozenset(p for p, _ in pairs), pairs)


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
    opp_waves_override: int | None = None,
    producer_baseline: bool = False,
    inject_seats: set[int] | None = None,
    sources_out: dict | None = None,
    ensemble: bool = False,
    inject_model: dict | None = None,
    sources_out_ensemble: list[dict] | None = None,
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

    Opponent models (Tier 1 single / Tier 2 ensemble):
    - ``ensemble=False`` (default): ONE model. ``producer_baseline`` selects the
      stripped public-producer model (Tier-1 contest detector); otherwise the
      faithful "opponent is also v5.x" model (the inert ``opp_inject_waves`` path).
      ``inject_seats`` filters which seats are injected; ``sources_out`` (if given)
      records each seat's predicted sources for the detector.
    - ``ensemble=True`` (Tier 2): TWO stripped producer models — base (β=0) and
      producer_v2 (β=``config.reinforce_size_beta``). Each seat runs BOTH (a 2nd
      planner pass), recording per-model sources into ``sources_out_ensemble[m]``;
      seats in ``inject_model`` are injected using their assigned best model's
      launches, all merged into the ONE returned reactive projection.
    """
    pid = int(obs.player_id)
    n = int(player_count)
    opp_waves = (
        int(opp_waves_override) if opp_waves_override is not None else int(config.opp_inject_waves)
    )
    # Opponent sub-plan: attacks only (regroup is internal logistics that mostly
    # shores up enemy defense → modeling it pushes US toward passivity, the
    # Cluster-9 failure mode; the threat that should reshape our plan is the
    # enemy's offense). Build the model list + the per-model source-record targets.
    if ensemble:
        # TIER-2: base producer (β=0) + producer_v2 (β=config.reinforce_size_beta).
        opp_configs = [
            _producer_baseline_config(config, opp_waves=opp_waves, reinforce_beta=0.0),
            _producer_baseline_config(
                config, opp_waves=opp_waves, reinforce_beta=float(config.reinforce_size_beta)
            ),
        ]
        sources_targets = sources_out_ensemble
    elif producer_baseline:
        # TIER-1: model the opponent as the PUBLIC producer flow-diff (ladder pool is
        # producer-clones), not v5.x.
        opp_configs = [_producer_baseline_config(config, opp_waves=opp_waves, reinforce_beta=0.0)]
        sources_targets = [sources_out] if sources_out is not None else None
    else:
        opp_configs = [
            dataclasses.replace(config, enable_regroup=False, max_waves_per_turn=opp_waves)
        ]
        sources_targets = [sources_out] if sources_out is not None else None

    planet_ids = obs_tensors["planets"][..., 0]  # [P] row -> planet id
    P = int(obs.P)
    tgt_chunks, own_chunks, ship_chunks, eta_chunks, valid_chunks = [], [], [], [], []
    for o in range(n):
        if o == pid:
            continue
        obs_o = parse_obs(obs_tensors, player_id=o)
        # Which model (if any) injects this seat? Ensemble: the seat's assigned best
        # model. Single-model: model 0 iff the seat passes the inject filter.
        if ensemble:
            chosen_m = inject_model.get(o) if inject_model is not None else None
        else:
            chosen_m = 0 if (inject_seats is None or o in inject_seats) else None
        inject_entries = None
        for m, opp_config in enumerate(opp_configs):
            launchable = (
                obs_o.owned & obs_o.alive & (obs_o.ships >= float(opp_config.min_ships_to_launch))
            )
            if not bool(launchable.any()):
                if sources_targets is not None:
                    sources_targets[m][o] = (frozenset(), ())
                continue
            entries_o = plan_lite_waves(
                movement=movement,
                obs=obs_o,
                obs_tensors=obs_tensors,
                cache=cache,
                garrison_status=base_status,
                prod=prod,
                alive_by_step=alive_by_step,
                config=opp_config,
                player_count=n,
            )
            # Record this model's predicted launch SOURCES for the detector (every live
            # seat, independent of whether we inject it below).
            if sources_targets is not None:
                sources_targets[m][o] = _entries_sources(entries_o, planet_ids, P)
            if chosen_m == m:
                inject_entries = entries_o
        # Inject this seat's best-response only when a model is chosen and it launched.
        if inject_entries is None:
            continue
        launches_o = infer_planned_launches_from_entries(
            obs_tensors=obs_tensors,
            movement=movement,
            entries=inject_entries,
            player_id=o,
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

    # BUCKETED CONFIG: opening-phase swap (see ProducerLiteConfig.opening_*).
    # Applied BEFORE the terminal block so terminal wins ties at the very end of
    # short games (opening/terminal windows shouldn't overlap in practice). OFF
    # (opening_phase_turns==0) => skipped => byte-identical. B1 swaps roi only
    # (default opening_roi_threshold==base ⇒ no-op); B2 additionally swaps the FFA
    # prod bonus, 4P-only, sentinel -1.0 ⇒ keep base.
    if int(config.opening_phase_turns) > 0 and step < int(config.opening_phase_turns):
        repl = {"roi_threshold": float(config.opening_roi_threshold)}
        if int(player_count) >= 4 and float(config.opening_ffa_target_prod_bonus) >= 0.0:
            repl["ffa_target_prod_bonus"] = float(config.opening_ffa_target_prod_bonus)
        config = dataclasses.replace(config, **repl)

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
            movement=movement,
            obs=obs,
            obs_tensors=obs_tensors,
            cache=cache,
            base_status=status,
            prod=movement.planet_prod,
            alive_by_step=alive_by_step,
            config=config,
            player_count=int(player_count),
            H=H,
        )

    entries = plan_lite_waves(
        movement=movement,
        obs=obs,
        obs_tensors=obs_tensors,
        cache=cache,
        garrison_status=status,
        prod=movement.planet_prod,
        alive_by_step=alive_by_step,
        config=config,
        player_count=int(player_count),
    )

    # Contestation overlay (see ProducerLiteConfig.contest_waves). Predict the
    # opponents' captures via the EXACT planner, then snipe the freshly-thinned
    # planets from the leftover budget. OFF (contest_waves==0) => skipped =>
    # byte-identical. Runs after the base plan so the snipe source budget is what
    # the base expansion left home (no tempo tax) and BEFORE apply_private (clean
    # do-nothing projection for the opponent model).
    if int(config.contest_waves) > 0 and int(player_count) >= 2:
        n = int(player_count)
        pid = int(obs.player_id)
        detect_on = int(config.contest_detect) != 0
        # TIER-2: ensemble (base producer + producer_v2) only when detection is on —
        # it gates on per-model fidelity. OFF (contest_ensemble==0) => single base model.
        ensemble_on = detect_on and int(config.contest_ensemble) != 0
        n_models = 2 if ensemble_on else 1
        # Detector state lives on memory, reset per game (and on a model-count change).
        if (
            step == 0
            or getattr(memory, "opp_tracker", None) is None
            or memory.opp_tracker.n_models != n_models
        ):
            memory.opp_tracker = _OpponentTracker(n_players=n, player_id=pid, n_models=n_models)
        tracker = memory.opp_tracker
        if detect_on:
            # 1. verify last turn's predictions vs this turn's freshly-spawned fleets.
            tracker.observe(obs_tensors, alpha=float(config.contest_fidelity_alpha))
            # 2. gate: snipe only seats whose producer-fidelity has cleared the bar
            #    (ensemble: under EITHER model).
            gated = tracker.gated_seats(
                threshold=float(config.contest_fidelity_threshold),
                min_obs=int(config.contest_min_observations),
            )
        else:
            gated = {o for o in range(n) if o != pid}
        # FFA board-position gate (4P only): strike from strength. Suppress all snipes
        # when our board strength is outside the top contest_ffa_strike_rank live
        # players — in a multi-way fight an undersized/mis-targeted snipe exposes the
        # grabbed planet and hands tempo to a third seat (see config note). Detection
        # below still runs for every seat (fidelity EMA keeps updating), so the gate
        # re-opens the moment we lead. OFF (rank<=0) or 2P => no suppression.
        if (
            n >= 4
            and int(config.contest_ffa_strike_rank) > 0
            and gated
            and _strength_rank(obs, prod=movement.planet_prod, player_count=n, player_id=pid)
            > int(config.contest_ffa_strike_rank)
        ):
            gated = set()
        # 3. run the producer opponent model(s): record per-seat (per-model) sources for
        # next-turn fidelity (all seats), inject only gated seats into the reactive
        # projection. Ensemble: inject each gated seat with ITS best-matching model.
        if ensemble_on:
            min_obs = int(config.contest_min_observations)
            inject_model = {o: tracker.best_model(o, min_obs=min_obs) for o in gated}
            sources_out_ensemble: list[dict] = [{} for _ in range(n_models)]
            reactive = _opponent_reactive_status(
                movement=movement,
                obs=obs,
                obs_tensors=obs_tensors,
                cache=cache,
                base_status=status,
                prod=movement.planet_prod,
                alive_by_step=alive_by_step,
                config=config,
                player_count=n,
                H=H,
                opp_waves_override=int(config.contest_opp_waves),
                ensemble=True,
                inject_model=inject_model,
                sources_out_ensemble=sources_out_ensemble,
            )
            tracker.set_predictions(sources_out_ensemble)
        else:
            sources_out: dict | None = {} if detect_on else None
            reactive = _opponent_reactive_status(
                movement=movement,
                obs=obs,
                obs_tensors=obs_tensors,
                cache=cache,
                base_status=status,
                prod=movement.planet_prod,
                alive_by_step=alive_by_step,
                config=config,
                player_count=n,
                H=H,
                opp_waves_override=int(config.contest_opp_waves),
                producer_baseline=True,
                inject_seats=gated,
                sources_out=sources_out,
            )
            if detect_on and sources_out is not None:
                tracker.set_predictions([sources_out])
        # 4. snipe only when at least one seat is gated ON (else reactive == base →
        # no predicted thin captures → no snipes; skip the work entirely).
        if gated:
            original_ships = obs.ships.to(obs.ships.dtype)
            committed = torch.zeros(P, dtype=original_ships.dtype, device=device)
            if bool(entries.valid.any()):
                committed.scatter_add_(
                    0,
                    entries.source_slots.clamp(0, P - 1),
                    torch.where(entries.valid, entries.ships, torch.zeros_like(entries.ships)),
                )
            leftover = (original_ships - committed).clamp(min=0.0)
            contest_entries = plan_contestation_waves(
                movement=movement,
                obs=obs,
                obs_tensors=obs_tensors,
                cache=cache,
                base_status=status,
                reactive_status=reactive,
                prod=movement.planet_prod,
                alive_by_step=alive_by_step,
                config=config,
                player_count=int(player_count),
                leftover=leftover,
                original_ships=original_ships,
            )
            if bool(contest_entries.valid.any()):
                entries = concat_launch_entries([entries, contest_entries])

    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors,
        movement=movement,
        entries=entries,
        player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement,
        launches=launches,
        owner_id=int(obs.player_id),
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
    # Contestation overlay STAYS OFF in 4P (the 2P v5.4 default flipped it ON, but 4P
    # keeps it off). The 4P extension was fully built and measured (2026-06-18) and the
    # snipe value does NOT transfer to FFA — see rl_research/CONTESTATION_OVERLAY_FINDINGS.md
    # "4P CLOSED". The machinery is all per-seat-ready (timing was a non-issue: ~46 ms/turn,
    # 218 ms worst, zero overage; the gate transfers cleanly — clones 0.65-0.78 ON,
    # non-clones <=0.06). What fails is the strategy: in a 4-way a snipe spends leftover
    # ships to grab a planet that becomes our exposed frontier, and the specific planet a
    # clone thins depends on the other live seats' moves our level-0 model can't see, so
    # snipes mis-target and overextend. Plain enabling regressed hard (win 35% vs 55% vs
    # off, paired); the ``contest_ffa_strike_rank`` board-position gate (strike only from
    # strength) narrowed it but did not clear the bar: pooled n=180 net -3 (mixed-2-clone
    # table +7, but all-clone -4 and 1-clone "sparse" -6 — the leader-gate fires when we
    # are ALREADY winning, exactly when we should consolidate, not attack). Fails the
    # strict ">= plain v5.3" bar, so 4P = plain v5.3 (byte-identical to v5.4). Set
    # contest_waves>0 + contest_ffa_strike_rank=1 to reopen the experiment (e.g. for the
    # "contested-leader only" refinement noted in the findings doc).
    contest_waves=0,
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        self.opp_tracker: _OpponentTracker | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.opp_tracker = None


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
            obs_tensors,
            config=config,
            player_count=int(mem.cached_player_count),
            memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _agent_impl(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    moves = sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
    if _VALIDATOR is not None:
        moves = apply_veto(moves, obs, _VALIDATOR, _VETO_THRESHOLD)
    return moves


def agent(obs):
    """Single-observation entry point for local play and Kaggle.

    Crash-safe wrapper: any exception (malformed/Struct obs the parser can't
    unpack, an internal invariant ``raise``, etc.) falls back to a legal no-op
    turn (``[]``) instead of erroring the episode, which Kaggle treats as a loss
    /disqualification. Byte-identical to the bare planner on the happy path.
    """
    try:
        return _agent_impl(obs)
    except Exception:
        return []
