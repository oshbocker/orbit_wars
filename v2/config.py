from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class V2EnvConfig:
    max_planets: int = 40
    board_size: float = 100.0
    episode_steps: int = 500
    allocation_threshold: float = 0.05
    min_ships_to_send: int = 1
    # Discrete ship-fraction bins for the factored fraction head (decoupled from
    # target selection). len(ship_fractions) must equal model.n_fractions.
    ship_fractions: list[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    # ── v3 Tier-1 feature flags (default off → identical to the existing agent) ──
    # Pairwise (source->target) features: travel_time, required-ships-on-arrival,
    # intercept-valid. Feeds OrbitNet's pair head so the fraction head can size
    # fleets correctly. Adds a [P,P,pair_feat_dim] tensor to V2Features.
    use_pair_features: bool = False
    pair_feat_dim: int = 3
    # Comet targeting: predict comet future positions from their known paths,
    # include comets as viable targets in the reachability mask, and add comet
    # planet features (is_comet, steps_to_expiry). Adds 2 dims to planet features
    # (so set model.planet_feat_dim=24 when enabled).
    comet_targeting: bool = False
    # Reachability viability: a target is attackable if src.ships >=
    # takeover_margin * (effective_garrison + 1). 1.0 = "capturable by sending
    # 100%" (matches the submission agent); the old 2.0 ("capturable with 50%")
    # was over-conservative and masked many of apex's real targets.
    takeover_margin: float = 1.0
    # ── v4 ceiling flags (default off → byte-identical to v2/v3) ──────────────
    # Tier 2.2: rotate/reflect the board to a canonical frame (own home at a
    # fixed angle + fixed chirality) so symmetric starts collapse to one case.
    canonical_rotation: bool = False
    # Tier 2.3: per-planet timeline features — defense margin (garrison +
    # own_incoming − enemy_incoming), time-to-flip, and reaction-time race
    # (my_t vs enemy_t). Adds 3 planet dims.
    timeline_features: bool = False
    # Tier 2.4: flag planets whose garrison was just depleted by a launch
    # (their id appears as a recent fleet from_planet_id). Adds 1 planet dim.
    depletion_feature: bool = False
    # Tier 2.5: order the relative enemy slots by a STABLE key (total ships,
    # desc) instead of first-encounter order, so "enemy1" = biggest threat.
    stable_enemy_order: bool = False
    # Tier 1.3: richer global features — centrality, best-single-enemy framing,
    # in-flight ship share. Adds dims to the global vector (set
    # model.global_feat_dim to match: 8 + 4 = 12).
    rich_global_features: bool = False
    # Tier 2.1: size fraction bins relative to the computed required_ships
    # (capture cost) instead of as a fraction of the source garrison. Fixes the
    # "ineffective attacks" passivity. When on, the n_fractions bins are read as
    # MULTIPLES of required_ships from req_fraction_multipliers (len must match
    # model.n_fractions), capped at the source garrison.
    requirement_relative_fractions: bool = False
    req_fraction_multipliers: list[float] = field(default_factory=lambda: [1.0, 1.5, 2.0, 3.0])
    # Tier 0.4: at decode, reject a launch if the shot-success head's P(own
    # target N turns later) is below this threshold. 0.0 = never reject (the head
    # still trains as an aux signal but does not gate actions). ~0.4 ≈ konbu17.
    shot_reject_threshold: float = 0.0


@dataclass(slots=True)
class V2ModelConfig:
    embed_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    ff_dim: int = 256
    planet_feat_dim: int = 22
    global_feat_dim: int = 8
    n_fractions: int = 4  # number of discrete ship-fraction bins (factored fraction head)
    # v3: must mirror env.use_pair_features / env.pair_feat_dim (the model needs
    # them at construction; load_v2_config syncs them from the env section).
    use_pair_features: bool = False
    pair_feat_dim: int = 3
    # ── v4 ceiling flags (default off) ──
    # Tier 1.1: a second value head on the shared trunk for the PPG auxiliary
    # value phase (lets the critic train hard without dragging the actor trunk).
    aux_value_head: bool = False
    # Tier 1.2 / 0.4: per-(source→target) shot-success head predicting
    # P(we still own the target N turns after arrival). Trained as an auxiliary
    # loss and reused as a rejection filter at decode time (the "shot validator").
    shot_success_head: bool = False


@dataclass(slots=True)
class V2PPOConfig:
    rollout_steps: int = 64
    num_envs: int = 2
    total_updates: int = 3000
    epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    num_workers: int = 0  # 0 = sequential (backward-compat), >0 = parallel subprocess workers
    ent_coef_end: float = (
        -1.0
    )  # <0 = constant ent_coef; >=0 = linearly anneal ent_coef -> ent_coef_end
    value_symlog: bool = (
        False  # symlog-transform value targets (scale-robust value learning; DreamerV3)
    )
    # ── v4 ceiling flags (default off) ──
    # Tier 0.3: PopArt-style running normalization of value targets (handles the
    # DRIFTING return scale that symlog doesn't). Use INSTEAD of value_symlog.
    popart: bool = False
    popart_beta: float = 0.01  # per-UPDATE EMA rate for running return mean/var
    # Tier 1.1: PPG auxiliary value phase. After the policy phase, run aux_epochs
    # of value-only optimization on the same buffer, with a policy-clone KL term
    # (weight aux_beta_clone) freezing the action distribution. 0 = disabled.
    aux_epochs: int = 0
    aux_beta_clone: float = 1.0
    aux_every: int = 1  # run the aux phase every N updates
    # Tier 1.2: weight on the shot-success auxiliary loss (needs model.shot_success_head).
    shot_aux_coef: float = 0.0
    shot_horizon: int = 10  # label = own target this many steps after the launch decision
    shot_aux_epochs: int = 1  # passes over the shot-label set per update
    # Tier 3.1: collect rollouts via the batched fast_env instead of the Kaggle
    # harness (major throughput win; required for affordable long rollouts).
    use_batched_env: bool = False


@dataclass(slots=True)
class V2RewardConfig:
    reward_mode: str = "dense_relative"  # sparse | dense_absolute | dense_relative | pbrs
    dense_ship_coef: float = 0.002
    dense_prod_coef: float = 0.005
    early_prod_bonus: float = 9.0
    early_prod_bonus_steps: int = 50
    # PBRS (potential-based reward shaping): r = gamma*Phi(s') - Phi(s).
    # Phi rewards owning productive territory (not banked ships), so effective
    # capture is rewarded and ship-hoarding is not. Policy-invariant (Ng 1999).
    pbrs_gamma: float = 0.997  # must match ppo.gamma for invariance
    pbrs_prod_weight: float = 1.0  # weight on (own_prod - best_enemy_prod)
    pbrs_planet_weight: float = 0.5  # weight on (own_planets - best_enemy_planets)
    pbrs_scale: float = 0.01  # overall scale of the shaping term


@dataclass(slots=True)
class V2EvalConfig:
    eval_every: int = 100
    eval_games: int = 10
    eval_opponents: list[str] = field(default_factory=lambda: ["apex", "random"])
    # Base map seed for periodic eval. Shared with scripts/eval_fast.py (20000)
    # so in-training win-rates are directly comparable to the trusted scorer.
    eval_seed: int = 20000
    # Process workers for periodic eval (CPU games are slow: ~20s each). >1 fans
    # the side-alternated games across processes like scripts/eval_fast.py. 0/1 =
    # sequential. Set to ~vCPU count on Colab.
    eval_workers: int = 0


@dataclass(slots=True)
class V2ImitationConfig:
    enabled: bool = False
    bc_expert: str = "apex"
    bc_games: int = 200
    bc_demo_opponent: str = "random"
    bc_epochs: int = 50
    bc_lr: float = 1e-3
    bc_batch_size: int = 256
    coef_start: float = 0.5
    coef_decay_updates: int = 1000
    distilled_opponent: bool = True
    bc_skip_steps: int = 0
    bc_cache_path: str = ""  # if set, pickle-cache demos here and reuse across runs
    bc_match_tolerance_deg: float = (
        90.0  # angular tolerance for matching an apex launch to a target slot
    )
    # ── v4 ceiling flags (default off) ──
    # Tier 0.1: floor the imitation coefficient so the BC/expert anchor never
    # decays fully to 0. A persistent pull toward a competent reference is what
    # stops self-play from forgetting how to beat the script (DeepNash lesson).
    coef_floor: float = 0.0
    # Tier 3.3: optional path to extra BC/value demonstrations distilled from
    # top-leaderboard replays (stronger teacher than apex alone).
    bc_replay_path: str = ""


@dataclass(slots=True)
class V2ExItConfig:
    """Expert Iteration: simulator + per-planet search to distill OrbitNet."""

    enabled: bool = False
    iterations: int = 50
    games_per_iter: int = 8
    search_depth: int = 12  # forward-sim steps per candidate evaluation
    search_candidates: int = 10  # max reachable targets searched per source planet
    search_temperature: float = 1.0  # softmax temperature over candidate scores
    train_epochs: int = 4
    train_batch_size: int = 256
    train_lr: float = 3e-4
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    dataset_max_iters: int = 3  # keep this many iterations of data in the buffer
    four_player_prob: float = 0.0
    opponent: str = "apex"
    sample_collect: bool = True  # sample (vs argmax) during self-play collection
    search_workers: int = 0  # >1 = parallelize the (CPU-bound) search across processes
    collect_workers: int = 0  # >1 = play the games of each iteration in parallel across
    # processes (collection is the ExIt bottleneck, ~linear win)
    collect_fast_env: bool = False  # collect games on the standalone fast_env (engine-faithful,
    # no Kaggle harness overhead) instead of make("orbit_wars")
    # ── v4 ceiling flags (default off) ──
    # Tier 3.2: score search leaves with OrbitNet's value head instead of the
    # handcrafted heuristic eval (requires a trustworthy value head — do Tier 1
    # first). Runs candidate rollouts on the batched fast_env with an in-sim
    # opponent (two_player_search) so the search reflects real opposition.
    neural_value_leaves: bool = False
    use_batched_env: bool = False
    # Replay the opponent's turn-1 apex launch in the lookahead. Default OFF:
    # when implemented (2026-06-04) it DEGRADED the 77% agent (too-sparse one-ply
    # opponent biases the search toward passivity). Opt in only for experiments.
    two_player_search: bool = False
    # Phase 1 (2026-06-04): every-step in-sim rollout opponent — the principled
    # fix vs the sparse turn-1 version above. During each candidate's lookahead,
    # ALL players (symmetric: opponents AND our own continuation) launch via a
    # cheap geometry-light rollout policy at every sim_step, so "hold" means
    # playing on and aggression is scored against a real, persistent opponent.
    # Default OFF — keeps the 77% agent byte-identical until validated @ n=60.
    rollout_search: bool = False
    rollout_self: bool = True  # include our own continuation in the rollout
    # Phase 3 (2026-06-05): grounded learned value at search leaves — BLEND, not
    # swap. When >0 (and a value_model is supplied), each candidate leaf is scored
    # by BOTH evaluate_state (heuristic) and OrbitNet's value head; the two are
    # z-scored across the leaves of THIS source-planet decision and combined as
    # `score = (1-w)*z(heur) + w*z(neural)`. w=0.0 = pure heuristic (bit-identical,
    # the blend branch is not taken). A pure swap (neural_value_leaves) collapsed
    # the 77% agent before — the value head is grounded globally (corr w/ outcome
    # 0.389) but too noisy to rank near-equal siblings alone, so blend a small w.
    value_leaf_blend: float = 0.0


@dataclass(slots=True)
class V2Config:
    seed: int = 42
    run_name: str = "v2_default"
    device: str = "auto"
    save_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    checkpoint_every: int = 50
    log_every: int = 1
    opponent: str = "apex"
    alternate_player_sides: bool = True
    self_play_update_interval: int = 50
    self_play_deterministic: bool = False
    four_player_prob: float = 0.0
    rule_based_prob_start: float = 1.0
    rule_based_prob_end: float = 0.2
    rule_based_decay_updates: int = 2000
    # PFSP (prioritized fictitious self-play). When enabled, replaces the linear
    # rule-based decay with a pool of {apex (always kept), frozen self-snapshots}
    # sampled by win-rate so the agent trains more against what it loses to —
    # which prevents self-play from forgetting how to beat apex.
    pfsp_enabled: bool = False
    pfsp_apex_min_prob: float = 0.3  # floor on probability of facing apex each episode
    pfsp_pool_size: int = 5  # max frozen self-snapshots retained
    pfsp_snapshot_every: int = 50  # add a frozen self-snapshot every N updates
    pfsp_weighting: str = "hard"  # "hard" (favor low win-rate) or "uniform"
    env: V2EnvConfig = field(default_factory=V2EnvConfig)
    model: V2ModelConfig = field(default_factory=V2ModelConfig)
    ppo: V2PPOConfig = field(default_factory=V2PPOConfig)
    reward: V2RewardConfig = field(default_factory=V2RewardConfig)
    eval: V2EvalConfig = field(default_factory=V2EvalConfig)
    imitation: V2ImitationConfig = field(default_factory=V2ImitationConfig)
    exit: V2ExItConfig = field(default_factory=V2ExItConfig)

    def __post_init__(self) -> None:
        # OrbitNet reads pair-feature flags off cfg.model, but configs set them
        # in the env section — keep them in sync so any V2Config (constructed or
        # loaded) builds a model matching its features.
        self.model.use_pair_features = self.env.use_pair_features
        self.model.pair_feat_dim = self.env.pair_feat_dim


def load_v2_config(path: str | Path) -> V2Config:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {config_path}")
    return v2_config_from_dict(data)


def v2_config_from_dict(data: dict[str, Any]) -> V2Config:
    cfg = V2Config()
    sub = {"env", "model", "ppo", "reward", "eval", "imitation", "exit"}
    _update_dataclass(cfg, data, skip=sub)
    _update_dataclass(cfg.env, data.get("env", {}))
    _update_dataclass(cfg.model, data.get("model", {}))
    _update_dataclass(cfg.ppo, data.get("ppo", {}))
    _update_dataclass(cfg.reward, data.get("reward", {}))
    _update_dataclass(cfg.eval, data.get("eval", {}))
    _update_dataclass(cfg.imitation, data.get("imitation", {}))
    _update_dataclass(cfg.exit, data.get("exit", {}))
    # Sync v3 pair-feature flags env -> model (model needs them at construction).
    cfg.model.use_pair_features = cfg.env.use_pair_features
    cfg.model.pair_feat_dim = cfg.env.pair_feat_dim
    return cfg


def v2_config_to_dict(cfg: V2Config) -> dict[str, Any]:
    """Serialize V2Config to a plain dict (for passing to subprocess workers)."""
    from dataclasses import fields as dc_fields

    result: dict[str, Any] = {}
    sub = {"env", "model", "ppo", "reward", "eval", "imitation", "exit"}
    for f in dc_fields(cfg):
        val = getattr(cfg, f.name)
        if f.name in sub:
            result[f.name] = {sf.name: getattr(val, sf.name) for sf in dc_fields(val)}
        else:
            result[f.name] = val
    return result


def _update_dataclass(instance: Any, values: dict[str, Any], skip: set[str] | None = None) -> None:
    if not isinstance(values, dict):
        return
    skip = skip or set()
    for key, value in values.items():
        if key in skip or not hasattr(instance, key):
            continue
        default = getattr(instance, key)
        setattr(instance, key, _coerce_value(value, default))


def _coerce_value(value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, list):
        return value if isinstance(value, list) else default
    return value
