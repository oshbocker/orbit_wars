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
    features = encode_features(state, cfg.env)

    with torch.inference_mode():
        pf = torch.from_numpy(features.planet_features).unsqueeze(0).to(device)
        gf = torch.from_numpy(features.global_features).unsqueeze(0).to(device)
        pm = torch.from_numpy(features.planet_mask).unsqueeze(0).to(device)
        om = torch.from_numpy(features.own_mask).unsqueeze(0).to(device)
        output = model(pf, gf, pm, om)

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
    all_ti: list[np.ndarray] = []
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

            # Forward pass
            with torch.inference_mode():
                pf_t = torch.from_numpy(feat.planet_features).unsqueeze(0).to(device)
                gf_t = torch.from_numpy(feat.global_features).unsqueeze(0).to(device)
                pm_t = torch.from_numpy(feat.planet_mask).unsqueeze(0).to(device)
                om_t = torch.from_numpy(feat.own_mask).unsqueeze(0).to(device)
                output = model(pf_t, gf_t, pm_t, om_t)

                sampled = sample_actions(output, om_t, deterministic=False)

            all_ti.append(sampled.target_indices[0].cpu().numpy())
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

    # Compute GAE
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
                next_v = next_values[env_idx] * non_terminal
            else:
                next_v = all_values[idxs[t + 1]] * non_terminal
            delta = rews[t] + gamma * next_v - all_values[idxs[t]]
            gae = delta + gamma * lam * non_terminal * gae
            i = idxs[t]
            returns[i] = gae + all_values[i]
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
            target_indices=torch.from_numpy(np.array(all_ti, dtype=np.int64)),
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
            output = model(pf_t, gf_t, pm_t, om_t)
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
        target_indices=torch.zeros(0, P, dtype=torch.long),
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

    # Build opponents
    rule_based_opponent = build_opponent(cfg.opponent)
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

    # Create envs
    envs = [V2OrbitWarsEnv(cfg, rule_based_opponent, env_index=idx)
            for idx in range(cfg.ppo.num_envs)]

    next_seed = cfg.seed
    features_per_env = []
    for env in envs:
        if scheduler is not None:
            num_p, opps = scheduler.sample_episode()
            features_per_env.append(env.reset(seed=next_seed, num_players=num_p, opponents=opps))
        else:
            features_per_env.append(env.reset(seed=next_seed))
        next_seed += 1

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.lr)

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

    # Training loop
    remaining = cfg.ppo.total_updates - start_update + 1
    log(f"\n=== PPO training (updates {start_update}..{cfg.ppo.total_updates}, "
        f"{remaining} remaining) ===")
    t_start = time.time()

    for update in range(start_update, cfg.ppo.total_updates + 1):
        t_update = time.time()

        if scheduler is not None:
            scheduler.set_update(update)

        batch, features_per_env, next_seed, stats = collect_rollout(
            envs, features_per_env, model, cfg, device, next_seed,
            scheduler=scheduler,
        )

        metrics = v2_ppo_update(
            model, optimizer, batch,
            clip_coef=cfg.ppo.clip_coef,
            ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            epochs=cfg.ppo.epochs,
            minibatch_size=cfg.ppo.minibatch_size,
            device=device,
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

    logger.close()
    _log_file.close()
    print(f"\nTraining complete. Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
