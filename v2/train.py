"""V2 training loop: OrbitNet PPO with simultaneous planet processing."""

from __future__ import annotations

import argparse
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.game_types import parse_observation
from src.logging import EvalResult, TrainLogger
from src.opponents import OpponentPolicy, build_opponent

from .actions import V2SampledAction, decode_actions, decode_sampled_actions, sample_actions
from .config import V2Config, load_v2_config
from .env import V2FastEnv, V2OrbitWarsEnv
from .features import V2Features, encode_features
from .model import OrbitNet, OrbitNetOutput
from .ppo import (
    V2TransitionBatch,
    ValueNorm,
    v2_aux_phase,
    v2_ppo_update,
    v2_shot_aux_update,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V2 OrbitNet agent")
    parser.add_argument("--config", type=str, default="configs/v2_default.yaml")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from",
    )
    parser.add_argument(
        "--bc_init",
        type=str,
        default=None,
        help="Path to a winbc checkpoint to warm-start the policy weights from "
        "(trunk + pointer/gate/frac heads load; value head trains fresh). Loaded "
        "strict=False; the config model arch must match the checkpoint's saved arch.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pair_tensor(feat: V2Features, device: torch.device):
    """Single-obs pair-feature tensor [1,P,P,pf], or None if disabled."""
    if feat.pair_features is None:
        return None
    return torch.from_numpy(feat.pair_features).unsqueeze(0).to(device)


def _current_ent_coef(cfg: V2Config, update: int) -> float:
    """Entropy coefficient with optional linear annealing.

    If cfg.ppo.ent_coef_end < 0 the coefficient is constant. Otherwise it
    interpolates linearly from ent_coef (update 1) to ent_coef_end
    (update total_updates).
    """
    end = cfg.ppo.ent_coef_end
    if end < 0:
        return cfg.ppo.ent_coef
    total = max(1, cfg.ppo.total_updates)
    frac = min(1.0, update / total)
    return cfg.ppo.ent_coef + frac * (end - cfg.ppo.ent_coef)


def _load_or_collect_demos(cfg: V2Config, log: Any) -> object:
    """Load demos from cache if available, else collect and (optionally) cache.

    Caching keeps the BC demonstration set identical and avoids re-running
    hundreds of expert games for every experiment.
    """
    import pickle

    from .imitation import collect_v2_demonstrations

    cache = cfg.imitation.bc_cache_path
    if cache and Path(cache).exists():
        with open(cache, "rb") as f:
            buf = pickle.load(f)
        log(f"  Loaded {len(buf)} cached demos from {cache}")
        return buf

    buf = collect_v2_demonstrations(
        n_games=cfg.imitation.bc_games,
        cfg=cfg,
        opponent_name=cfg.imitation.bc_demo_opponent,
    )
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(buf, f)
        log(f"  Cached {len(buf)} demos to {cache}")
    return buf


class V2SelfPlayOpponent:
    """Wraps OrbitNet for opponent use."""

    def __init__(self, cfg: V2Config, device: torch.device, deterministic: bool = True) -> None:
        self.cfg = cfg
        self.device = device
        self.deterministic = deterministic
        self.model = OrbitNet(cfg.model).to(device)
        self.model.eval()

    def sync_from(self, source_model: OrbitNet) -> None:
        self.model.load_state_dict(source_model.state_dict())
        self.model.eval()

    def act(self, observation: Any) -> list[list[float | int]]:
        return _v2_policy_act(self.model, observation, self.cfg, self.device, self.deterministic)


def _v2_policy_act(
    model: OrbitNet,
    observation: Any,
    cfg: V2Config,
    device: torch.device,
    deterministic: bool = True,
) -> list[list[float | int]]:
    """Run OrbitNet on a raw observation and return Kaggle moves."""
    state = parse_observation(observation)
    # Extract comet IDs to filter them from reachability
    comet_ids = None
    if hasattr(observation, "comet_planet_ids"):
        ids = getattr(observation, "comet_planet_ids", None)
        if ids is not None:
            comet_ids = [int(x) for x in ids]
    elif isinstance(observation, dict):
        ids = observation.get("comet_planet_ids")
        if ids is not None:
            comet_ids = [int(x) for x in ids]
    comets_data = getattr(observation, "comets", None)
    if comets_data is None and isinstance(observation, dict):
        comets_data = observation.get("comets")
    features = encode_features(state, cfg.env, comet_ids=comet_ids, comets_data=comets_data)

    with torch.inference_mode():
        pf = torch.from_numpy(features.planet_features).unsqueeze(0).to(device)
        gf = torch.from_numpy(features.global_features).unsqueeze(0).to(device)
        pm = torch.from_numpy(features.planet_mask).unsqueeze(0).to(device)
        om = torch.from_numpy(features.own_mask).unsqueeze(0).to(device)
        rm = torch.from_numpy(features.reachability_mask).unsqueeze(0).to(device)
        output = model(pf, gf, pm, om, rm, _pair_tensor(features, device))

    return decode_actions(output, features, state, cfg.env, deterministic=deterministic)


class V2MixedScheduler:
    """Blends rule-based + self-play opponents with linear decay. Supports 2p/4p."""

    def __init__(
        self,
        cfg: V2Config,
        rule_based: OpponentPolicy,
        self_play: V2SelfPlayOpponent,
    ) -> None:
        self.cfg = cfg
        self.rule_based = rule_based
        self.self_play = self_play
        self._update = 0

    def set_update(self, update: int) -> None:
        self._update = update

    def _rule_based_prob(self) -> float:
        decay = self.cfg.rule_based_decay_updates
        if decay <= 0:
            return self.cfg.rule_based_prob_end
        frac = min(1.0, self._update / decay)
        return self.cfg.rule_based_prob_start + frac * (
            self.cfg.rule_based_prob_end - self.cfg.rule_based_prob_start
        )

    def sample_episode(self) -> tuple[int, list[OpponentPolicy]]:
        is_4p = random.random() < self.cfg.four_player_prob
        n_opp = 3 if is_4p else 1
        rb_prob = self._rule_based_prob()
        opponents = []
        for _ in range(n_opp):
            if random.random() < rb_prob:
                opponents.append(self.rule_based)
            else:
                opponents.append(self.self_play)
        return (4 if is_4p else 2), opponents


class V2PFSPScheduler:
    """Prioritized Fictitious Self-Play opponent pool.

    Always keeps the rule-based reference (cfg.opponent) in the pool — with a probability
    floor — and adds frozen snapshots of the training policy over time. Opponents
    are sampled by win-rate so the agent trains more against the ones it currently
    loses to (PFSP "hard" weighting, à la AlphaStar). This directly counters the
    failure mode where self-play forgets how to beat the script.
    """

    def __init__(self, cfg: V2Config, rule_based: OpponentPolicy, device: torch.device) -> None:
        self.cfg = cfg
        self.device = device
        self._update = 0
        # Pool entries: {"name", "agent", "wins", "games"}. Apex is index 0.
        self.pool: list[dict] = [
            {"name": "rule_based", "agent": rule_based, "wins": 0.0, "games": 0.0},
        ]
        self._last_sampled: str | None = None

    def set_update(self, update: int) -> None:
        self._update = update

    def maybe_snapshot(self, model: OrbitNet) -> None:
        """Add a frozen snapshot of the current policy to the pool (FIFO-capped)."""
        if self.cfg.pfsp_snapshot_every <= 0:
            return
        if self._update == 0 or self._update % self.cfg.pfsp_snapshot_every != 0:
            return
        snap = V2SelfPlayOpponent(self.cfg, device=self.device, deterministic=False)
        snap.sync_from(model)
        self.pool.append({"name": f"self@{self._update}", "agent": snap, "wins": 0.0, "games": 0.0})
        # Cap frozen snapshots (never evict the rule-based anchor at index 0).
        frozen = self.pool[1:]
        if len(frozen) > self.cfg.pfsp_pool_size:
            self.pool = [self.pool[0]] + frozen[-self.cfg.pfsp_pool_size :]

    def _win_rate(self, entry: dict) -> float:
        return entry["wins"] / entry["games"] if entry["games"] > 0 else 0.5

    def _weights(self) -> list[float]:
        # PFSP "hard": weight ∝ (1 - win_rate) so we favor opponents we lose to.
        if self.cfg.pfsp_weighting == "uniform":
            return [1.0] * len(self.pool)
        return [max(0.05, 1.0 - self._win_rate(e)) for e in self.pool]

    def sample_episode(self) -> tuple[int, list[OpponentPolicy]]:
        is_4p = random.random() < self.cfg.four_player_prob
        n_opp = 3 if is_4p else 1
        opponents: list[OpponentPolicy] = []
        names: list[str] = []
        for _ in range(n_opp):
            # Enforce the anchor probability floor, else PFSP-weighted sample.
            if random.random() < self.cfg.pfsp_anchor_min_prob:
                idx = 0
            else:
                w = self._weights()
                total = sum(w)
                r = random.random() * total
                idx, acc = 0, 0.0
                for k, wk in enumerate(w):
                    acc += wk
                    if r <= acc:
                        idx = k
                        break
            opponents.append(self.pool[idx]["agent"])
            names.append(self.pool[idx]["name"])
        self._last_sampled = names[0] if names else None
        return (4 if is_4p else 2), opponents

    def last_name(self) -> str | None:
        return self._last_sampled

    def update_result(self, name: str | None, win: bool) -> None:
        """Record the learner's result against a specific pool opponent."""
        if name is None:
            return
        for e in self.pool:
            if e["name"] == name:
                e["games"] += 1.0
                e["wins"] += 1.0 if win else 0.0
                break

    def apply_deltas(self, deltas: dict) -> None:
        """Fold per-opponent (wins, games) deltas (e.g. from parallel workers)
        into the central pool. Unknown names are ignored (snapshot may have been
        evicted by the FIFO cap)."""
        for e in self.pool:
            d = deltas.get(e["name"])
            if d is not None:
                e["wins"] += d[0]
                e["games"] += d[1]

    def pool_summary(self) -> str:
        return ", ".join(
            f"{e['name']}:{self._win_rate(e):.0%}({int(e['games'])})" for e in self.pool
        )


def make_v2_eval_agent(
    model: OrbitNet,
    cfg: V2Config,
    device: torch.device,
) -> Callable:
    """Create a Kaggle-compatible agent(obs, config) from a V2 model."""
    eval_model = OrbitNet(cfg.model).to(device)
    eval_model.load_state_dict(model.state_dict())
    eval_model.eval()

    def agent(obs: Any, config: Any = None) -> list:
        return _v2_policy_act(eval_model, obs, cfg, device, deterministic=True)

    return agent


def collect_rollout(
    envs: list[V2OrbitWarsEnv],
    features_per_env: list[V2Features],
    model: OrbitNet,
    cfg: V2Config,
    device: torch.device,
    next_seed: int,
    scheduler: V2MixedScheduler | None = None,
    value_norm: object | None = None,
) -> tuple[V2TransitionBatch, list[V2Features], int, dict[str, float]]:
    """Collect rollout: ONE forward pass per env per step."""
    P = cfg.env.max_planets

    # Transition storage
    all_pf: list[np.ndarray] = []
    all_gf: list[np.ndarray] = []
    all_pm: list[np.ndarray] = []
    all_om: list[np.ndarray] = []
    all_rm: list[np.ndarray] = []
    all_pairf: list[np.ndarray] = []
    all_ti: list[np.ndarray] = []
    all_fi: list[np.ndarray] = []
    all_lp: list[float] = []
    all_values: list[float] = []

    # Per-env tracking for GAE
    rewards_per_env: list[list[float]] = [[] for _ in envs]
    dones_per_env: list[list[bool]] = [[] for _ in envs]
    value_indices_per_env: list[list[int]] = [[] for _ in envs]
    episode_rewards: list[float] = []
    running_rewards = [0.0 for _ in envs]
    is_pfsp = isinstance(scheduler, V2PFSPScheduler)
    # Track which PFSP opponent each env is currently facing (set at reset).
    active_opp_name: list[str | None] = [
        (scheduler.last_name() if isinstance(scheduler, V2PFSPScheduler) else None) for _ in envs
    ]

    # v4 Tier 1.2: per-step records for shot-success labeling (only if needed).
    collect_shot = model.shot_success_head is not None and cfg.ppo.shot_aux_coef > 0.0
    shot_records: list[list] = [[] for _ in envs]

    has_pair = features_per_env[0].pair_features is not None

    for _step_i in range(cfg.ppo.rollout_steps):
        next_features = []

        # Store features for every env, recording each one's buffer index.
        idx_e: list[int] = []
        for env_idx in range(len(envs)):
            feat = features_per_env[env_idx]
            idx_e.append(len(all_pf))
            all_pf.append(feat.planet_features)
            all_gf.append(feat.global_features)
            all_pm.append(feat.planet_mask)
            all_om.append(feat.own_mask)
            all_rm.append(feat.reachability_mask)
            if feat.pair_features is not None:
                all_pairf.append(feat.pair_features)
            value_indices_per_env[env_idx].append(idx_e[env_idx])

        # ONE batched forward + sample across all envs (Tier 3.1 throughput win).
        with torch.inference_mode():
            pf_b = torch.from_numpy(np.stack([f.planet_features for f in features_per_env])).to(
                device
            )
            gf_b = torch.from_numpy(np.stack([f.global_features for f in features_per_env])).to(
                device
            )
            pm_b = torch.from_numpy(np.stack([f.planet_mask for f in features_per_env])).to(device)
            om_b = torch.from_numpy(np.stack([f.own_mask for f in features_per_env])).to(device)
            rm_b = torch.from_numpy(np.stack([f.reachability_mask for f in features_per_env])).to(
                device
            )
            pair_b = (
                torch.from_numpy(np.stack([f.pair_features for f in features_per_env])).to(device)
                if has_pair
                else None
            )
            output = model(pf_b, gf_b, pm_b, om_b, rm_b, pair_b)
            sampled = sample_actions(output, om_b, deterministic=False)

        ti_cpu = sampled.target_indices.cpu().numpy()
        fi_cpu = sampled.frac_indices.cpu().numpy()
        lp_cpu = sampled.log_prob.cpu().numpy()
        val_cpu = output.value.cpu().numpy()

        for env_idx, env in enumerate(envs):
            feat = features_per_env[env_idx]
            all_ti.append(ti_cpu[env_idx])
            all_fi.append(fi_cpu[env_idx])
            all_lp.append(float(lp_cpu[env_idx]))
            all_values.append(float(val_cpu[env_idx]))

            # Per-env slice of the batched output/sample for decoding (keeps the
            # decode helpers' [0]-indexing intact).
            e = env_idx
            out_i = OrbitNetOutput(
                logits=output.logits[e : e + 1],
                value=output.value[e : e + 1],
                frac_logits=output.frac_logits[e : e + 1],
                aux_value=(output.aux_value[e : e + 1] if output.aux_value is not None else None),
                shot_logits=(
                    output.shot_logits[e : e + 1] if output.shot_logits is not None else None
                ),
            )
            sa_i = V2SampledAction(
                target_indices=sampled.target_indices[e : e + 1],
                frac_indices=sampled.frac_indices[e : e + 1],
                log_prob=sampled.log_prob[e : e + 1],
                entropy=sampled.entropy[e : e + 1],
            )
            state = env.last_state
            assert state is not None  # set by env.reset()/step() before this loop body
            moves = decode_sampled_actions(sa_i, out_i, feat, state, cfg.env)
            result = env.step(moves)

            running_rewards[env_idx] += result.reward
            rewards_per_env[env_idx].append(result.reward)
            dones_per_env[env_idx].append(result.done)

            # v4 Tier 1.2: record launches + ownership for shot-success labeling.
            if collect_shot:
                ti_np = ti_cpu[env_idx]
                launches = [
                    (i, int(ti_np[i]) - 1) for i in range(P) if feat.own_mask[i] and ti_np[i] > 0
                ]
                owners = {ps.id: ps.owner for ps in feat.planet_states if ps is not None}
                shot_records[env_idx].append(
                    (idx_e[env_idx], state.player, owners, launches, result.done)
                )

            if result.done:
                episode_rewards.append(running_rewards[env_idx])
                if is_pfsp:
                    scheduler.update_result(
                        active_opp_name[env_idx], running_rewards[env_idx] > 0.0
                    )
                running_rewards[env_idx] = 0.0
                next_seed += 1
                if scheduler is not None:
                    num_p, opps = scheduler.sample_episode()
                    if is_pfsp:
                        active_opp_name[env_idx] = scheduler.last_name()
                    new_feat = env.reset(seed=next_seed, num_players=num_p, opponents=opps)
                else:
                    new_feat = env.reset(seed=next_seed)
                next_features.append(new_feat)
            else:
                next_features.append(result.features)

        features_per_env = next_features

    # v4 Tier 1.2: build shot-success labels from the recorded trajectories.
    shot_idx: list[int] = []
    shot_src: list[int] = []
    shot_tgt: list[int] = []
    shot_lab: list[float] = []
    if collect_shot:
        H = max(1, cfg.ppo.shot_horizon)
        for recs in shot_records:
            n = len(recs)
            for t in range(n):
                t2 = t + H
                if t2 >= n:
                    continue
                if any(recs[k][4] for k in range(t, t2)):  # episode boundary in window
                    continue
                m_idx, player_t, _own, launches_t, _done = recs[t]
                owners_future = recs[t2][2]
                for s_slot, tg_slot in launches_t:
                    shot_idx.append(m_idx)
                    shot_src.append(s_slot)
                    shot_tgt.append(tg_slot)
                    shot_lab.append(1.0 if owners_future.get(tg_slot, -1) == player_t else 0.0)

    # Bootstrap final values
    next_values = _bootstrap_values(model, features_per_env, device)

    # Compute GAE. Stored values (all_values / next_values) are raw head
    # outputs; with value_symlog they live in symlog space, so map them back to
    # real return space via symexp before doing GAE arithmetic. The buffer keeps
    # the raw (symlog-space) values so the PPO clipped value loss stays consistent.
    import math as _math

    def _real(v: float) -> float:
        if value_norm is not None:
            return float(value_norm.denormalize(v))
        if not cfg.ppo.value_symlog:
            return v
        return _math.copysign(_math.expm1(abs(v)), v)

    N = len(all_pf)
    returns = [0.0] * N
    advantages = [0.0] * N
    gamma = cfg.ppo.gamma
    lam = cfg.ppo.gae_lambda

    for env_idx in range(len(envs)):
        idxs = value_indices_per_env[env_idx]
        rews = rewards_per_env[env_idx]
        dones = dones_per_env[env_idx]
        n_steps = len(idxs)
        if n_steps == 0:
            continue

        gae = 0.0
        for t in reversed(range(n_steps)):
            non_terminal = 1.0 - float(dones[t])
            if t == n_steps - 1:
                next_v = _real(next_values[env_idx]) * non_terminal
            else:
                next_v = _real(all_values[idxs[t + 1]]) * non_terminal
            delta = rews[t] + gamma * next_v - _real(all_values[idxs[t]])
            gae = delta + gamma * lam * non_terminal * gae
            i = idxs[t]
            returns[i] = gae + _real(all_values[i])
            advantages[i] = gae

    # Build batch
    if N == 0:
        batch = _empty_batch(cfg)
    else:
        batch = V2TransitionBatch(
            planet_features=torch.from_numpy(np.array(all_pf, dtype=np.float32)),
            global_features=torch.from_numpy(np.array(all_gf, dtype=np.float32)),
            planet_mask=torch.from_numpy(np.array(all_pm, dtype=bool)),
            own_mask=torch.from_numpy(np.array(all_om, dtype=bool)),
            reachability_mask=torch.from_numpy(np.array(all_rm, dtype=bool)),
            target_indices=torch.from_numpy(np.array(all_ti, dtype=np.int64)),
            frac_indices=torch.from_numpy(np.array(all_fi, dtype=np.int64)),
            log_prob=torch.tensor(all_lp, dtype=torch.float32),
            returns=torch.tensor(returns, dtype=torch.float32),
            advantages=torch.tensor(advantages, dtype=torch.float32),
            values=torch.tensor(all_values, dtype=torch.float32),
            pair_features=(
                torch.from_numpy(np.array(all_pairf, dtype=np.float32)) if all_pairf else None
            ),
            shot_idx=(torch.tensor(shot_idx, dtype=torch.long) if shot_idx else None),
            shot_src=(torch.tensor(shot_src, dtype=torch.long) if shot_idx else None),
            shot_tgt=(torch.tensor(shot_tgt, dtype=torch.long) if shot_idx else None),
            shot_label=(torch.tensor(shot_lab, dtype=torch.float32) if shot_idx else None),
        )

    stats = {
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episodes_finished": float(len(episode_rewards)),
        "samples": float(N),
    }
    return batch, features_per_env, next_seed, stats


def _bootstrap_values(
    model: OrbitNet,
    features_list: list[V2Features],
    device: torch.device,
) -> list[float]:
    values = []
    for feat in features_list:
        if not feat.own_mask.any():
            values.append(0.0)
            continue
        with torch.inference_mode():
            pf_t = torch.from_numpy(feat.planet_features).unsqueeze(0).to(device)
            gf_t = torch.from_numpy(feat.global_features).unsqueeze(0).to(device)
            pm_t = torch.from_numpy(feat.planet_mask).unsqueeze(0).to(device)
            om_t = torch.from_numpy(feat.own_mask).unsqueeze(0).to(device)
            rm_t = torch.from_numpy(feat.reachability_mask).unsqueeze(0).to(device)
            output = model(pf_t, gf_t, pm_t, om_t, rm_t, _pair_tensor(feat, device))
        values.append(float(output.value[0].cpu()))
    return values


def _empty_batch(cfg: V2Config) -> V2TransitionBatch:
    P = cfg.env.max_planets
    from .features import GLOBAL_FEAT_DIM, PLANET_FEAT_DIM

    return V2TransitionBatch(
        planet_features=torch.zeros(0, P, PLANET_FEAT_DIM),
        global_features=torch.zeros(0, GLOBAL_FEAT_DIM),
        planet_mask=torch.zeros(0, P, dtype=torch.bool),
        own_mask=torch.zeros(0, P, dtype=torch.bool),
        reachability_mask=torch.zeros(0, P, P, dtype=torch.bool),
        target_indices=torch.zeros(0, P, dtype=torch.long),
        frac_indices=torch.zeros(0, P, dtype=torch.long),
        log_prob=torch.zeros(0),
        returns=torch.zeros(0),
        advantages=torch.zeros(0),
        values=torch.zeros(0),
    )


def save_checkpoint(
    save_dir: Path,
    run_name: str,
    update: int,
    model: OrbitNet,
    optimizer: torch.optim.Optimizer,
) -> Path:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "update": update,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, run_dir / "ckpt_last.pt")
    torch.save(state, run_dir / f"ckpt_{update:06d}.pt")
    return run_dir / "ckpt_last.pt"


_PEVAL: dict = {}


def _peval_init(cfg_dict: dict, state_dict: dict, opponent: str) -> None:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from v2.config import v2_config_from_dict

    cfg = v2_config_from_dict(cfg_dict)
    model = OrbitNet(cfg.model)
    model.load_state_dict(state_dict)
    model.eval()
    _PEVAL["rl"] = make_v2_eval_agent(model, cfg, torch.device("cpu"))
    _PEVAL["opp"] = _get_eval_opponent(opponent)


def _peval_game(args: tuple[int, int]) -> str:
    """Play one (seed, side) game; return 'win'/'loss'/'tie' from RL's view."""
    from v2.fast_env import FastOrbitWars

    seed, side = args
    sim = FastOrbitWars(num_agents=2, seed=seed)
    rl, opp = _PEVAL["rl"], _PEVAL["opp"]
    while not sim.done:
        rl_moves = rl(sim.observation(side)) or []
        opp_moves = opp(sim.observation(1 - side)) or []
        acts: list = [None, None]
        acts[side] = list(rl_moves)
        acts[1 - side] = list(opp_moves)
        sim.step(acts)
    rr, orr = sim.rewards[side], sim.rewards[1 - side]
    if rr > 0 and orr > 0:
        return "tie"
    return "win" if rr > 0 else "loss"


def run_periodic_eval(
    model: OrbitNet,
    cfg: V2Config,
    device: torch.device,
) -> list[EvalResult]:
    """Side-alternated, paired-seed win-rate eval on the engine-faithful
    FastOrbitWars — mirrors scripts/eval_fast.py (and shares its game loop, so
    the in-training number is directly comparable to the trusted scorer).

    The old implementation always played the RL agent as player 0 via the
    Kaggle harness with no seed control: NOT side-alternated, NOT paired. With any
    player-0 advantage it scored the favourable side every game and inflated the
    win-rate ~2x (showed 90-95% live while eval_fast showed 33-58% on the same
    ckpts). Here each game alternates which side the RL agent plays and uses a
    fixed base seed (eval_seed=20000, shared with eval_fast), so the only variance
    is the map seed. CPU games are slow (~20s each) so cfg.eval.eval_workers>1
    fans them across processes.
    """

    n = cfg.eval.eval_games
    base_seed = cfg.eval.eval_seed
    jobs = [(base_seed + i, i % 2) for i in range(n)]  # alternate RL's side
    workers = cfg.eval.eval_workers
    results: list[EvalResult] = []

    if workers and workers > 1 and n > 1:
        from concurrent.futures import ProcessPoolExecutor

        from v2.config import v2_config_to_dict

        cfg_dict = v2_config_to_dict(cfg)
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        for opp_name in cfg.eval.eval_opponents:
            try:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_peval_init,
                    initargs=(cfg_dict, sd, opp_name),
                ) as ex:
                    res = list(ex.map(_peval_game, jobs))
            except Exception as e:  # pragma: no cover - fall back to sequential
                print(f"  parallel eval failed ({e}); falling back to sequential")
                res = _run_eval_sequential(model, cfg, device, opp_name, jobs)
            results.append(_tally(opp_name, res, n))
        return results

    for opp_name in cfg.eval.eval_opponents:
        res = _run_eval_sequential(model, cfg, device, opp_name, jobs)
        results.append(_tally(opp_name, res, n))
    return results


def _run_eval_sequential(
    model: OrbitNet,
    cfg: V2Config,
    device: torch.device,
    opp_name: str,
    jobs: list[tuple[int, int]],
) -> list[str]:
    from v2.fast_env import FastOrbitWars

    rl = make_v2_eval_agent(model, cfg, device)
    opp = _get_eval_opponent(opp_name)
    out: list[str] = []
    for seed, side in jobs:
        sim = FastOrbitWars(num_agents=2, seed=seed)
        while not sim.done:
            rl_moves = rl(sim.observation(side)) or []
            opp_moves = opp(sim.observation(1 - side)) or []
            acts: list = [None, None]
            acts[side] = list(rl_moves)
            acts[1 - side] = list(opp_moves)
            sim.step(acts)
        rr, orr = sim.rewards[side], sim.rewards[1 - side]
        out.append("tie" if (rr > 0 and orr > 0) else ("win" if rr > 0 else "loss"))
    return out


def _tally(opp_name: str, res: list[str], n: int) -> EvalResult:
    return EvalResult(
        opponent_name=opp_name,
        win_rate=res.count("win") / max(n, 1),
        loss_rate=res.count("loss") / max(n, 1),
        tie_rate=res.count("tie") / max(n, 1),
        n_games=n,
    )


def _get_eval_opponent(name: str) -> Any:
    from agents import load_named_agent

    return load_named_agent(name)


def main() -> None:
    args = parse_args()
    cfg = load_v2_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Set up text log file (append mode so resume doesn't overwrite)
    log_dir = Path(cfg.log_dir) / cfg.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    _log_path = log_dir / "train.log"
    _log_state = {"f": open(_log_path, "a"), "n": 0}

    def log(msg: str) -> None:
        """Print to stdout and append to train.log.

        On Google Drive (Colab) the FUSE mount only reliably syncs a file on
        close(), so flush()/fsync() alone can leave the synced copy stale. We
        close-and-reopen every few writes to force the data through to Drive —
        this is what keeps train.log complete across Colab session resumes.
        """
        import os as _os

        print(msg)
        f = _log_state["f"]
        f.write(msg + "\n")
        f.flush()
        try:
            _os.fsync(f.fileno())
        except (OSError, ValueError):
            pass
        _log_state["n"] += 1
        if _log_state["n"] % 10 == 0:  # periodic close/reopen -> Drive sync
            f.close()
            _log_state["f"] = open(_log_path, "a")

    # Resume-safe, Drive-synced eval history (append + close each call, so the
    # full eval curve survives Colab session restarts and is directly plottable).
    _eval_csv = log_dir / "eval_history.csv"
    if not _eval_csv.exists():
        with open(_eval_csv, "w") as ef:
            ef.write("update,opponent,win_rate,loss_rate,tie_rate,n_games\n")

    def log_eval_row(update: int, results) -> None:
        with open(_eval_csv, "a") as ef:
            for r in results:
                ef.write(
                    f"{update},{r.opponent_name},{r.win_rate},"
                    f"{r.loss_rate},{r.tie_rate},{r.n_games}\n"
                )

    log(f"V2 Config: {cfg.run_name}, device={device}, updates={cfg.ppo.total_updates}")
    log(
        f"  envs={cfg.ppo.num_envs}, rollout_steps={cfg.ppo.rollout_steps}, opponent={cfg.opponent}"
    )

    # Count parameters
    model = OrbitNet(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  OrbitNet params: {n_params:,}")

    # ── BC warm-start (lagged self-play init) ────────────────────────────────
    # Load a winbc gate-head checkpoint into the policy. The PPO action sampler
    # (v2/actions.sample_actions) decides launch via the hold column, NOT the BC
    # gate head, so the gate weights load but are unused — the trunk + target
    # pointer (the valuable learned features) transfer, and the hold/value heads
    # re-calibrate under RL. Loaded strict=False; cfg.model arch must match.
    if args.bc_init:
        bc = torch.load(args.bc_init, map_location=device, weights_only=False)
        arch = bc.get("arch", {})
        for k, v in arch.items():
            cur = getattr(cfg.model, k, None)
            if cur is not None and int(cur) != int(v):
                log(
                    f"  WARNING: bc_init arch mismatch {k}: config={cur} ckpt={v} "
                    "— mismatched weights will be dropped (check the self-play config)"
                )
        res = model.load_state_dict(bc["model"], strict=False)
        loaded = len(bc["model"]) - len(res.unexpected_keys)
        log(
            f"  BC-init from {args.bc_init}: loaded~{loaded} tensors "
            f"(missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}, "
            f"gate_head={bc.get('gate_head')})"
        )

    # Logger
    logger = TrainLogger(cfg.log_dir, cfg.run_name)

    save_dir = Path(cfg.save_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr)

    # Tier 0.3: PopArt/ValueNorm running return normalizer (sequential only).
    value_norm = None
    if cfg.ppo.popart:
        if cfg.ppo.value_symlog:
            raise ValueError("ppo.popart and ppo.value_symlog are mutually exclusive — pick one.")
        value_norm = ValueNorm(cfg.ppo.popart_beta)
        log(f"  PopArt value normalization on (beta={cfg.ppo.popart_beta})")
    if cfg.ppo.aux_epochs > 0 and not cfg.model.aux_value_head:
        log("  WARNING: ppo.aux_epochs>0 but model.aux_value_head=false — aux phase is a no-op.")

    # ── Imitation learning phases ──────────────────────────────────────────
    demo_buffer = None

    if cfg.imitation.enabled and not args.resume:
        from .imitation import v2_bc_pretrain

        # Phase 1: Collect demonstrations (or load from cache)
        log(
            f"\n=== Phase 1: Collecting {cfg.imitation.bc_games} demo games "
            f"(expert={cfg.imitation.bc_expert} vs {cfg.imitation.bc_demo_opponent}) ==="
        )
        demo_buffer = _load_or_collect_demos(cfg, log)
        log(f"  Buffer size: {len(demo_buffer)}")

        # Phase 2: BC pretrain
        log(f"\n=== Phase 2: BC pretraining ({cfg.imitation.bc_epochs} epochs) ===")
        v2_bc_pretrain(model, demo_buffer, cfg.imitation, device, logger)

        # Save BC-pretrained checkpoint (update=0)
        save_checkpoint(save_dir, cfg.run_name, 0, model, optimizer)
        log("  BC checkpoint saved (update=0)")

        # Eval the BC clone before PPO starts, so the warm-start strength is visible.
        if cfg.eval.eval_every > 0:
            log(f"  Eval of BC clone (update 0, {cfg.eval.eval_games} games)...")
            for r in run_periodic_eval(model, cfg, device):
                logger.log_eval(0, [r])
                log(
                    f"    vs {r.opponent_name}: W={r.win_rate:.0%} L={r.loss_rate:.0%} "
                    f"T={r.tie_rate:.0%} (n={r.n_games})"
                )

    # Build opponents
    rule_based_opponent = build_opponent(cfg.opponent)

    # Use distilled opponent if BC was done
    if cfg.imitation.enabled and cfg.imitation.distilled_opponent and demo_buffer is not None:
        log("  Using distilled opponent (BC-pretrained V2)")
        distilled = V2SelfPlayOpponent(cfg, device=device, deterministic=True)
        distilled.sync_from(model)
        rule_based_opponent = distilled
    sp_opponent = V2SelfPlayOpponent(cfg, device=device, deterministic=cfg.self_play_deterministic)
    sp_opponent.sync_from(model)

    # Scheduler: PFSP pool (keeps the rule-based anchor, samples by win-rate) takes precedence;
    # else the linear rule-based->self-play MixedScheduler.
    scheduler: V2MixedScheduler | V2PFSPScheduler | None = None
    if cfg.pfsp_enabled:
        scheduler = V2PFSPScheduler(cfg, rule_based_opponent, device)
        log(
            f"  PFSPScheduler: anchor_floor={cfg.pfsp_anchor_min_prob}, "
            f"pool_size={cfg.pfsp_pool_size}, snapshot_every={cfg.pfsp_snapshot_every}, "
            f"weighting={cfg.pfsp_weighting}"
        )
    elif cfg.four_player_prob > 0.0 or cfg.rule_based_prob_start < 1.0:
        scheduler = V2MixedScheduler(cfg, rule_based_opponent, sp_opponent)
        log(
            f"  MixedScheduler: 4p_prob={cfg.four_player_prob}, "
            f"rule_based={cfg.rule_based_prob_start:.1f}->{cfg.rule_based_prob_end:.1f} "
            f"over {cfg.rule_based_decay_updates} updates"
        )

    # Create envs (only needed for sequential mode)
    envs: list = []
    features_per_env: list = []
    next_seed = cfg.seed
    if cfg.ppo.num_workers == 0:
        # Tier 3.1: fast standalone sim (no Kaggle harness) when enabled.
        env_cls = V2FastEnv if cfg.ppo.use_batched_env else V2OrbitWarsEnv
        if cfg.ppo.use_batched_env:
            log("  Using V2FastEnv (standalone fast_env sim) for rollouts")
        envs = [env_cls(cfg, rule_based_opponent, env_index=idx) for idx in range(cfg.ppo.num_envs)]
        for env in envs:
            if scheduler is not None:
                num_p, opps = scheduler.sample_episode()
                features_per_env.append(
                    env.reset(seed=next_seed, num_players=num_p, opponents=opps)
                )
            else:
                features_per_env.append(env.reset(seed=next_seed))
            next_seed += 1

    # Resume from checkpoint if requested
    start_update = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            # Try as relative to save_dir/run_name
            resume_path = save_dir / cfg.run_name / args.resume
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = ckpt["update"] + 1
        # Advance seed past completed updates to avoid replaying episodes
        next_seed = cfg.seed + start_update * cfg.ppo.num_envs * cfg.ppo.rollout_steps
        # Re-sync self-play opponent with resumed model
        sp_opponent.sync_from(model)
        # Re-reset envs with new seeds
        features_per_env = []
        for env in envs:
            if scheduler is not None:
                scheduler.set_update(start_update)
                num_p, opps = scheduler.sample_episode()
                features_per_env.append(
                    env.reset(seed=next_seed, num_players=num_p, opponents=opps)
                )
            else:
                features_per_env.append(env.reset(seed=next_seed))
            next_seed += 1
        log(f"  Resumed from {resume_path} at update {ckpt['update']}")

        # Re-collect demos if imitation is active at resume point
        if cfg.imitation.enabled and demo_buffer is None:
            decay_frac = start_update / max(cfg.imitation.coef_decay_updates, 1)
            coef_at_resume = max(
                cfg.imitation.coef_floor,
                cfg.imitation.coef_start * max(0.0, 1.0 - decay_frac),
            )
            if coef_at_resume > 0.0:
                log(f"  Loading demos for imitation (coef={coef_at_resume:.3f})...")
                demo_buffer = _load_or_collect_demos(cfg, log)
                log(f"  Buffer size: {len(demo_buffer)}")

    # Training loop
    remaining = cfg.ppo.total_updates - start_update + 1
    log(
        f"\n=== PPO training (updates {start_update}..{cfg.ppo.total_updates}, "
        f"{remaining} remaining) ==="
    )

    if cfg.ppo.num_workers > 0:
        _train_parallel(
            cfg,
            model,
            optimizer,
            logger,
            save_dir,
            device,
            log,
            start_update,
            demo_buffer,
            log_eval_row,
            scheduler,
            value_norm,
            sp_opponent,
        )
    else:
        _train_sequential(
            cfg,
            model,
            optimizer,
            logger,
            save_dir,
            device,
            log,
            envs,
            features_per_env,
            next_seed,
            scheduler,
            sp_opponent,
            start_update,
            demo_buffer,
            log_eval_row,
            value_norm,
        )

    logger.close()
    _log_state["f"].close()


def _train_sequential(
    cfg: V2Config,
    model: OrbitNet,
    optimizer: torch.optim.Optimizer,
    logger: TrainLogger,
    save_dir: Path,
    device: torch.device,
    log: Any,
    envs: list,
    features_per_env: list,
    next_seed: int,
    scheduler: V2MixedScheduler | V2PFSPScheduler | None,
    sp_opponent: V2SelfPlayOpponent,
    start_update: int,
    demo_buffer: object | None = None,
    log_eval_row: Any = None,
    value_norm: object | None = None,
) -> None:
    """Original sequential training loop."""
    t_start = time.time()

    for update in range(start_update, cfg.ppo.total_updates + 1):
        t_update = time.time()

        if scheduler is not None:
            scheduler.set_update(update)

        batch, features_per_env, next_seed, stats = collect_rollout(
            envs,
            features_per_env,
            model,
            cfg,
            device,
            next_seed,
            scheduler=scheduler,
            value_norm=value_norm,
        )

        # Compute imitation coefficient (linear decay)
        imitation_coef = 0.0
        if cfg.imitation.enabled and demo_buffer is not None:
            decay_frac = update / max(cfg.imitation.coef_decay_updates, 1)
            imitation_coef = max(
                cfg.imitation.coef_floor,  # Tier 0.1: persistent anchor (never decays to 0)
                cfg.imitation.coef_start * max(0.0, 1.0 - decay_frac),
            )

        metrics = v2_ppo_update(
            model,
            optimizer,
            batch,
            clip_coef=cfg.ppo.clip_coef,
            ent_coef=_current_ent_coef(cfg, update),
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            epochs=cfg.ppo.epochs,
            minibatch_size=cfg.ppo.minibatch_size,
            device=device,
            demo_buffer=demo_buffer,
            imitation_coef=imitation_coef,
            value_symlog=cfg.ppo.value_symlog,
            value_norm=value_norm,
        )

        # Tier 1.1: PPG auxiliary value phase (no-op unless aux_epochs>0 and the
        # model has an aux value head). Runs every aux_every updates.
        if cfg.ppo.aux_epochs > 0 and update % max(1, cfg.ppo.aux_every) == 0:
            aux_metrics = v2_aux_phase(
                model,
                optimizer,
                batch,
                aux_epochs=cfg.ppo.aux_epochs,
                beta_clone=cfg.ppo.aux_beta_clone,
                minibatch_size=cfg.ppo.minibatch_size,
                device=device,
                value_symlog=cfg.ppo.value_symlog,
                value_norm=value_norm,
            )
            metrics.update(aux_metrics)

        # Tier 1.2: train the shot-success head on outcome labels (no-op unless
        # shot_aux_coef>0 and the model has a shot head).
        if cfg.ppo.shot_aux_coef > 0.0:
            shot_metrics = v2_shot_aux_update(
                model,
                optimizer,
                batch,
                coef=cfg.ppo.shot_aux_coef,
                epochs=cfg.ppo.shot_aux_epochs,
                minibatch_size=cfg.ppo.minibatch_size,
                device=device,
            )
            metrics.update(shot_metrics)

        # Sync self-play opponent periodically
        if update % cfg.self_play_update_interval == 0:
            sp_opponent.sync_from(model)

        # PFSP: periodically freeze a snapshot of the current policy into the pool.
        if isinstance(scheduler, V2PFSPScheduler):
            scheduler.maybe_snapshot(model)

        all_metrics = {**stats, **metrics}
        logger.log_update(update, all_metrics)

        if update % cfg.log_every == 0:
            elapsed = time.time() - t_start
            update_time = time.time() - t_update
            log(
                f"update={update:4d}  reward={stats['episode_reward_mean']:+.3f}  "
                f"eps={int(stats['episodes_finished'])}  samples={int(stats['samples'])}  "
                f"loss={metrics['loss']:.4f}  ploss={metrics['policy_loss']:.4f}  "
                f"vloss={metrics['value_loss']:.4f}  ent={metrics['entropy']:.3f}  "
                f"dt={update_time:.1f}s  total={elapsed:.0f}s"
            )

        # Periodic evaluation
        if cfg.eval.eval_every > 0 and update % cfg.eval.eval_every == 0:
            log(f"\n  Running eval ({cfg.eval.eval_games} games)...")
            eval_results = run_periodic_eval(model, cfg, device)
            logger.log_eval(update, eval_results)
            if log_eval_row is not None:
                log_eval_row(update, eval_results)
            for r in eval_results:
                log(
                    f"    vs {r.opponent_name}: W={r.win_rate:.0%} L={r.loss_rate:.0%} "
                    f"T={r.tie_rate:.0%} (n={r.n_games})"
                )
            if isinstance(scheduler, V2PFSPScheduler):
                log(f"    PFSP pool: {scheduler.pool_summary()}")
            log("")

        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            save_checkpoint(save_dir, cfg.run_name, update, model, optimizer)
            log(f"  -> saved checkpoint at update {update}")

    print(f"\nTraining complete. Total time: {time.time() - t_start:.0f}s")


def _train_parallel(
    cfg: V2Config,
    model: OrbitNet,
    optimizer: torch.optim.Optimizer,
    logger: TrainLogger,
    save_dir: Path,
    device: torch.device,
    log: Any,
    start_update: int,
    demo_buffer: object | None = None,
    log_eval_row: Any = None,
    scheduler: V2MixedScheduler | V2PFSPScheduler | None = None,
    value_norm: object | None = None,
    sp_opponent: V2SelfPlayOpponent | None = None,
) -> None:
    """Parallel training loop using subprocess workers (v4-complete).

    Mirrors `_train_sequential`: PopArt value-norm, the PPG aux phase, the
    shot-success aux update, self-play sync and PFSP snapshotting all run in this
    (central) process. Workers only roll out; their per-opponent win/game deltas
    are folded back into the central PFSP scheduler so the broadcast sampling
    weights stay globally consistent.
    """
    from .parallel import ParallelRolloutCollector

    num_workers = cfg.ppo.num_workers
    log(f"  Parallel mode: {num_workers} workers")

    collector = ParallelRolloutCollector(cfg, num_workers)
    # Initial broadcast: weights (+ PopArt stats + PFSP pool/stats).
    collector.sync(model, value_norm, scheduler)

    t_start = time.time()
    try:
        for update in range(start_update, cfg.ppo.total_updates + 1):
            t_update = time.time()

            if scheduler is not None:
                scheduler.set_update(update)

            batch, stats, pfsp_deltas = collector.collect(update)

            # Fold worker PFSP results into the central pool before snapshotting.
            if isinstance(scheduler, V2PFSPScheduler) and pfsp_deltas:
                scheduler.apply_deltas(pfsp_deltas)

            # Compute imitation coefficient (linear decay)
            imitation_coef = 0.0
            if cfg.imitation.enabled and demo_buffer is not None:
                decay_frac = update / max(cfg.imitation.coef_decay_updates, 1)
                imitation_coef = max(
                    cfg.imitation.coef_floor,  # Tier 0.1: persistent anchor (never decays to 0)
                    cfg.imitation.coef_start * max(0.0, 1.0 - decay_frac),
                )

            metrics = v2_ppo_update(
                model,
                optimizer,
                batch,
                clip_coef=cfg.ppo.clip_coef,
                ent_coef=_current_ent_coef(cfg, update),
                vf_coef=cfg.ppo.vf_coef,
                max_grad_norm=cfg.ppo.max_grad_norm,
                epochs=cfg.ppo.epochs,
                minibatch_size=cfg.ppo.minibatch_size,
                device=device,
                demo_buffer=demo_buffer,
                imitation_coef=imitation_coef,
                value_symlog=cfg.ppo.value_symlog,
                value_norm=value_norm,
            )

            # Tier 1.1: PPG auxiliary value phase (every aux_every updates).
            if cfg.ppo.aux_epochs > 0 and update % max(1, cfg.ppo.aux_every) == 0:
                metrics.update(
                    v2_aux_phase(
                        model,
                        optimizer,
                        batch,
                        aux_epochs=cfg.ppo.aux_epochs,
                        beta_clone=cfg.ppo.aux_beta_clone,
                        minibatch_size=cfg.ppo.minibatch_size,
                        device=device,
                        value_symlog=cfg.ppo.value_symlog,
                        value_norm=value_norm,
                    )
                )

            # Tier 1.2: train the shot-success head on outcome labels.
            if cfg.ppo.shot_aux_coef > 0.0:
                metrics.update(
                    v2_shot_aux_update(
                        model,
                        optimizer,
                        batch,
                        coef=cfg.ppo.shot_aux_coef,
                        epochs=cfg.ppo.shot_aux_epochs,
                        minibatch_size=cfg.ppo.minibatch_size,
                        device=device,
                    )
                )

            # Sync self-play opponent periodically (non-PFSP MixedScheduler path).
            if sp_opponent is not None and update % cfg.self_play_update_interval == 0:
                sp_opponent.sync_from(model)

            # PFSP: periodically freeze a snapshot of the current policy.
            if isinstance(scheduler, V2PFSPScheduler):
                scheduler.maybe_snapshot(model)

            # Broadcast updated weights (+ PopArt stats + PFSP pool/stats) to workers.
            collector.sync(model, value_norm, scheduler)

            all_metrics = {**stats, **metrics}
            logger.log_update(update, all_metrics)

            if update % cfg.log_every == 0:
                elapsed = time.time() - t_start
                update_time = time.time() - t_update
                log(
                    f"update={update:4d}  reward={stats['episode_reward_mean']:+.3f}  "
                    f"eps={int(stats['episodes_finished'])}  samples={int(stats['samples'])}  "
                    f"loss={metrics['loss']:.4f}  ploss={metrics['policy_loss']:.4f}  "
                    f"vloss={metrics['value_loss']:.4f}  ent={metrics['entropy']:.3f}  "
                    f"dt={update_time:.1f}s  total={elapsed:.0f}s"
                )

            # Periodic evaluation
            if cfg.eval.eval_every > 0 and update % cfg.eval.eval_every == 0:
                log(f"\n  Running eval ({cfg.eval.eval_games} games)...")
                eval_results = run_periodic_eval(model, cfg, device)
                logger.log_eval(update, eval_results)
                if log_eval_row is not None:
                    log_eval_row(update, eval_results)
                for r in eval_results:
                    log(
                        f"    vs {r.opponent_name}: W={r.win_rate:.0%} L={r.loss_rate:.0%} "
                        f"T={r.tie_rate:.0%} (n={r.n_games})"
                    )
                if isinstance(scheduler, V2PFSPScheduler):
                    log(f"    PFSP pool: {scheduler.pool_summary()}")
                log("")

            if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
                save_checkpoint(save_dir, cfg.run_name, update, model, optimizer)
                log(f"  -> saved checkpoint at update {update}")
    finally:
        collector.shutdown()

    print(f"\nTraining complete. Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
