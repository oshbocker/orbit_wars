"""V2 training loop: OrbitNet PPO with simultaneous planet processing."""
from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.game_types import parse_observation
from src.logging import EvalResult, TrainLogger
from src.opponents import ApexOpponent, OpponentPolicy, build_opponent

from .actions import V2SampledAction, decode_actions, decode_sampled_actions, sample_actions
from .config import V2Config, load_v2_config
from .env import V2OrbitWarsEnv
from .features import V2Features, encode_features
from .model import OrbitNet, OrbitNetOutput
from .ppo import V2TransitionBatch, v2_ppo_update


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V2 OrbitNet agent")
    parser.add_argument("--config", type=str, default="configs/v2_default.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt file to resume training from")
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
    features = encode_features(state, cfg.env, comet_ids=comet_ids)

    with torch.inference_mode():
        pf = torch.from_numpy(features.planet_features).unsqueeze(0).to(device)
        gf = torch.from_numpy(features.global_features).unsqueeze(0).to(device)
        pm = torch.from_numpy(features.planet_mask).unsqueeze(0).to(device)
        om = torch.from_numpy(features.own_mask).unsqueeze(0).to(device)
        rm = torch.from_numpy(features.reachability_mask).unsqueeze(0).to(device)
        output = model(pf, gf, pm, om, rm)

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
) -> tuple[V2TransitionBatch, list[V2Features], int, dict[str, float]]:
    """Collect rollout: ONE forward pass per env per step."""
    P = cfg.env.max_planets

    # Transition storage
    all_pf: list[np.ndarray] = []
    all_gf: list[np.ndarray] = []
    all_pm: list[np.ndarray] = []
    all_om: list[np.ndarray] = []
    all_rm: list[np.ndarray] = []
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

    for _step_i in range(cfg.ppo.rollout_steps):
        next_features = []

        for env_idx, env in enumerate(envs):
            feat = features_per_env[env_idx]

            # Store features
            idx = len(all_pf)
            all_pf.append(feat.planet_features)
            all_gf.append(feat.global_features)
            all_pm.append(feat.planet_mask)
            all_om.append(feat.own_mask)
            all_rm.append(feat.reachability_mask)

            # Forward pass
            with torch.inference_mode():
                pf_t = torch.from_numpy(feat.planet_features).unsqueeze(0).to(device)
                gf_t = torch.from_numpy(feat.global_features).unsqueeze(0).to(device)
                pm_t = torch.from_numpy(feat.planet_mask).unsqueeze(0).to(device)
                om_t = torch.from_numpy(feat.own_mask).unsqueeze(0).to(device)
                rm_t = torch.from_numpy(feat.reachability_mask).unsqueeze(0).to(device)
                output = model(pf_t, gf_t, pm_t, om_t, rm_t)

                sampled = sample_actions(output, om_t, deterministic=False)

            all_ti.append(sampled.target_indices[0].cpu().numpy())
            all_fi.append(sampled.frac_indices[0].cpu().numpy())
            all_lp.append(float(sampled.log_prob[0].cpu()))
            all_values.append(float(output.value[0].cpu()))
            value_indices_per_env[env_idx].append(idx)

            # Decode the sampled actions (not a fresh sample) so the
            # executed moves match the log_probs stored in the PPO buffer.
            state = env.last_state
            moves = decode_sampled_actions(sampled, output, feat, state, cfg.env)
            result = env.step(moves)

            running_rewards[env_idx] += result.reward
            rewards_per_env[env_idx].append(result.reward)
            dones_per_env[env_idx].append(result.done)

            if result.done:
                episode_rewards.append(running_rewards[env_idx])
                running_rewards[env_idx] = 0.0
                next_seed += 1
                if scheduler is not None:
                    num_p, opps = scheduler.sample_episode()
                    new_feat = env.reset(seed=next_seed, num_players=num_p, opponents=opps)
                else:
                    new_feat = env.reset(seed=next_seed)
                next_features.append(new_feat)
            else:
                next_features.append(result.features)

        features_per_env = next_features

    # Bootstrap final values
    next_values = _bootstrap_values(model, features_per_env, device)

    # Compute GAE. Stored values (all_values / next_values) are raw head
    # outputs; with value_symlog they live in symlog space, so map them back to
    # real return space via symexp before doing GAE arithmetic. The buffer keeps
    # the raw (symlog-space) values so the PPO clipped value loss stays consistent.
    import math as _math

    def _real(v: float) -> float:
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
            output = model(pf_t, gf_t, pm_t, om_t, rm_t)
        values.append(float(output.value[0].cpu()))
    return values


def _empty_batch(cfg: V2Config) -> V2TransitionBatch:
    P = cfg.env.max_planets
    from .features import PLANET_FEAT_DIM, GLOBAL_FEAT_DIM
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


def run_periodic_eval(
    model: OrbitNet,
    cfg: V2Config,
    device: torch.device,
) -> list[EvalResult]:
    from evaluation.evaluate import run_games

    eval_agent = make_v2_eval_agent(model, cfg, device)
    results: list[EvalResult] = []

    for opp_name in cfg.eval.eval_opponents:
        opp_callable = _get_eval_opponent(opp_name)
        raw = run_games(eval_agent, opp_callable, n_games=cfg.eval.eval_games)
        results.append(EvalResult(
            opponent_name=opp_name,
            win_rate=raw["win_rate"],
            loss_rate=raw["loss_rate"],
            tie_rate=raw["tie_rate"],
            n_games=raw["n_games"],
        ))

    return results


def _get_eval_opponent(name: str) -> Any:
    if name == "apex":
        from agents.apex import agent as apex_agent
        return apex_agent
    if name == "random":
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent
        return random_agent
    if name == "hybrid":
        from agents.hybrid import agent as hybrid_agent
        return hybrid_agent
    raise ValueError(f"Unknown eval opponent: {name}")


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
    _log_file = open(log_dir / "train.log", "a")

    def log(msg: str) -> None:
        """Print to stdout and append to train.log."""
        print(msg)
        _log_file.write(msg + "\n")
        _log_file.flush()

    log(f"V2 Config: {cfg.run_name}, device={device}, updates={cfg.ppo.total_updates}")
    log(f"  envs={cfg.ppo.num_envs}, rollout_steps={cfg.ppo.rollout_steps}, "
        f"opponent={cfg.opponent}")

    # Count parameters
    model = OrbitNet(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  OrbitNet params: {n_params:,}")

    # Logger
    logger = TrainLogger(cfg.log_dir, cfg.run_name)

    save_dir = Path(cfg.save_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr)

    # ── Imitation learning phases ──────────────────────────────────────────
    demo_buffer = None

    if cfg.imitation.enabled and not args.resume:
        from .imitation import v2_bc_pretrain

        # Phase 1: Collect demonstrations (or load from cache)
        log(f"\n=== Phase 1: Collecting {cfg.imitation.bc_games} demo games "
            f"(expert={cfg.imitation.bc_expert} vs {cfg.imitation.bc_demo_opponent}) ===")
        demo_buffer = _load_or_collect_demos(cfg, log)
        log(f"  Buffer size: {len(demo_buffer)}")

        # Phase 2: BC pretrain
        log(f"\n=== Phase 2: BC pretraining ({cfg.imitation.bc_epochs} epochs) ===")
        v2_bc_pretrain(model, demo_buffer, cfg.imitation, device, logger)

        # Save BC-pretrained checkpoint (update=0)
        save_checkpoint(save_dir, cfg.run_name, 0, model, optimizer)
        log("  BC checkpoint saved (update=0)")

    # Build opponents
    rule_based_opponent = build_opponent(cfg.opponent)

    # Use distilled opponent if BC was done
    if (cfg.imitation.enabled and cfg.imitation.distilled_opponent
            and demo_buffer is not None):
        log("  Using distilled opponent (BC-pretrained V2)")
        distilled = V2SelfPlayOpponent(cfg, device=device, deterministic=True)
        distilled.sync_from(model)
        rule_based_opponent = distilled
    sp_opponent = V2SelfPlayOpponent(cfg, device=device,
                                     deterministic=cfg.self_play_deterministic)
    sp_opponent.sync_from(model)

    # Mixed scheduler
    use_scheduler = cfg.four_player_prob > 0.0 or cfg.rule_based_prob_start < 1.0
    scheduler: V2MixedScheduler | None = None
    if use_scheduler:
        scheduler = V2MixedScheduler(cfg, rule_based_opponent, sp_opponent)
        log(f"  MixedScheduler: 4p_prob={cfg.four_player_prob}, "
            f"rule_based={cfg.rule_based_prob_start:.1f}->{cfg.rule_based_prob_end:.1f} "
            f"over {cfg.rule_based_decay_updates} updates")

    # Create envs (only needed for sequential mode)
    envs: list = []
    features_per_env: list = []
    next_seed = cfg.seed
    if cfg.ppo.num_workers == 0:
        envs = [V2OrbitWarsEnv(cfg, rule_based_opponent, env_index=idx)
                for idx in range(cfg.ppo.num_envs)]
        for env in envs:
            if scheduler is not None:
                num_p, opps = scheduler.sample_episode()
                features_per_env.append(env.reset(seed=next_seed, num_players=num_p, opponents=opps))
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
                features_per_env.append(env.reset(seed=next_seed, num_players=num_p, opponents=opps))
            else:
                features_per_env.append(env.reset(seed=next_seed))
            next_seed += 1
        log(f"  Resumed from {resume_path} at update {ckpt['update']}")

        # Re-collect demos if imitation is active at resume point
        if cfg.imitation.enabled and demo_buffer is None:
            decay_frac = start_update / max(cfg.imitation.coef_decay_updates, 1)
            coef_at_resume = cfg.imitation.coef_start * max(0.0, 1.0 - decay_frac)
            if coef_at_resume > 0.0:
                log(f"  Loading demos for imitation (coef={coef_at_resume:.3f})...")
                demo_buffer = _load_or_collect_demos(cfg, log)
                log(f"  Buffer size: {len(demo_buffer)}")

    # Training loop
    remaining = cfg.ppo.total_updates - start_update + 1
    log(f"\n=== PPO training (updates {start_update}..{cfg.ppo.total_updates}, "
        f"{remaining} remaining) ===")

    if cfg.ppo.num_workers > 0:
        _train_parallel(cfg, model, optimizer, logger, save_dir, device, log,
                        start_update, demo_buffer)
    else:
        _train_sequential(cfg, model, optimizer, logger, save_dir, device, log,
                          envs, features_per_env, next_seed, scheduler,
                          sp_opponent, start_update, demo_buffer)

    logger.close()
    _log_file.close()


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
    scheduler: V2MixedScheduler | None,
    sp_opponent: V2SelfPlayOpponent,
    start_update: int,
    demo_buffer: object | None = None,
) -> None:
    """Original sequential training loop."""
    t_start = time.time()

    for update in range(start_update, cfg.ppo.total_updates + 1):
        t_update = time.time()

        if scheduler is not None:
            scheduler.set_update(update)

        batch, features_per_env, next_seed, stats = collect_rollout(
            envs, features_per_env, model, cfg, device, next_seed,
            scheduler=scheduler,
        )

        # Compute imitation coefficient (linear decay)
        imitation_coef = 0.0
        if cfg.imitation.enabled and demo_buffer is not None:
            decay_frac = update / max(cfg.imitation.coef_decay_updates, 1)
            imitation_coef = cfg.imitation.coef_start * max(0.0, 1.0 - decay_frac)

        metrics = v2_ppo_update(
            model, optimizer, batch,
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
        )

        # Sync self-play opponent periodically
        if update % cfg.self_play_update_interval == 0:
            sp_opponent.sync_from(model)

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
            for r in eval_results:
                log(f"    vs {r.opponent_name}: W={r.win_rate:.0%} L={r.loss_rate:.0%} "
                    f"T={r.tie_rate:.0%} (n={r.n_games})")
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
) -> None:
    """Parallel training loop using subprocess workers."""
    from .parallel import ParallelRolloutCollector

    num_workers = cfg.ppo.num_workers
    log(f"  Parallel mode: {num_workers} workers")

    collector = ParallelRolloutCollector(cfg, num_workers)
    # Send initial weights to workers
    collector.sync_weights(model)

    t_start = time.time()
    try:
        for update in range(start_update, cfg.ppo.total_updates + 1):
            t_update = time.time()

            batch, stats = collector.collect(update)

            # Compute imitation coefficient (linear decay)
            imitation_coef = 0.0
            if cfg.imitation.enabled and demo_buffer is not None:
                decay_frac = update / max(cfg.imitation.coef_decay_updates, 1)
                imitation_coef = cfg.imitation.coef_start * max(0.0, 1.0 - decay_frac)

            metrics = v2_ppo_update(
                model, optimizer, batch,
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
            )

            # Sync weights to workers
            collector.sync_weights(model)

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
                for r in eval_results:
                    log(f"    vs {r.opponent_name}: W={r.win_rate:.0%} L={r.loss_rate:.0%} "
                        f"T={r.tie_rate:.0%} (n={r.n_games})")
                log("")

            if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
                save_checkpoint(save_dir, cfg.run_name, update, model, optimizer)
                log(f"  -> saved checkpoint at update {update}")
    finally:
        collector.shutdown()

    print(f"\nTraining complete. Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
