from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .config import TrainConfig, load_train_config
from .env import OrbitWarsEnv
from .features import (
    GLOBAL_DIM,
    KNN_SCALAR_DIM,
    SOURCE_SCALAR_DIM,
    TARGET_SCALAR_DIM,
    FleetTransitState,
    SourceDecision,
    compute_fleet_transit,
    encode_source_decision,
    fleet_speed,
)
from .game_types import GameState, parse_observation
from .logging import TrainLogger, run_periodic_eval
from .opponents import DistilledOpponent, OpponentPolicy, SelfPlayOpponent, build_opponent
from .policy import TransformerPolicy
from .ppo import TransitionBatch, ppo_update, sample_actions


@dataclass(slots=True)
class StepGroup:
    """All planet decisions from one env step share the same reward."""
    indices: list[int]
    reward: float
    done: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer PPO agent")
    parser.add_argument("--config", type=str, default="configs/transformer_ppo.yaml")
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


def _forward_single(
    policy: TransformerPolicy,
    decision: SourceDecision,
    device: torch.device,
) -> tuple:
    """Forward pass for a single source planet decision. Returns (outputs, sampled)."""
    outputs = policy(
        torch.from_numpy(decision.global_features).unsqueeze(0).to(device),
        torch.from_numpy(decision.source_scalars).unsqueeze(0).to(device),
        torch.from_numpy(decision.source_position).unsqueeze(0).to(device),
        torch.from_numpy(decision.knn_scalars).unsqueeze(0).to(device),
        torch.from_numpy(decision.knn_positions).unsqueeze(0).to(device),
        torch.from_numpy(decision.target_scalars).unsqueeze(0).to(device),
        torch.from_numpy(decision.target_positions).unsqueeze(0).to(device),
        torch.from_numpy(decision.target_mask).unsqueeze(0).to(device),
    )
    sampled = sample_actions(outputs, deterministic=False)
    return outputs, sampled


class MixedScheduler:
    """Schedules 2p/4p games with a mix of rule-based and self-play opponents."""

    def __init__(
        self,
        cfg: TrainConfig,
        rule_based: OpponentPolicy,
        self_play: SelfPlayOpponent,
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
        """Return (num_players, opponents) for a new episode."""
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


def collect_rollout(
    envs: list[OrbitWarsEnv],
    raw_obs_per_env: list,
    policy: TransformerPolicy,
    cfg: TrainConfig,
    device: torch.device,
    next_seed: int,
    scheduler: MixedScheduler | None = None,
) -> tuple[TransitionBatch, list, int, dict[str, float]]:
    """Collect rollout with sequential per-planet decisions."""
    T = cfg.env.max_targets
    K = cfg.env.k_neighbors

    # Transition storage
    all_global: list[np.ndarray] = []
    all_src_scalars: list[np.ndarray] = []
    all_src_pos: list[np.ndarray] = []
    all_knn_scalars: list[np.ndarray] = []
    all_knn_pos: list[np.ndarray] = []
    all_tgt_scalars: list[np.ndarray] = []
    all_tgt_pos: list[np.ndarray] = []
    all_tgt_mask: list[np.ndarray] = []
    all_target_idx: list[int] = []
    all_frac_bin: list[int] = []
    all_log_prob: list[float] = []
    all_values: list[float] = []

    groups_per_env: list[list[StepGroup]] = [[] for _ in envs]
    episode_rewards: list[float] = []
    running_rewards = [0.0 for _ in envs]

    for _step_i in range(cfg.ppo.rollout_steps):
        next_obs_list = []

        for env_idx, env in enumerate(envs):
            obs = raw_obs_per_env[env_idx]
            state = parse_observation(obs)
            my_planets = sorted(
                [p for p in state.planets if p.owner == state.player],
                key=lambda p: -p.ships,
            )

            transit = compute_fleet_transit(state)
            moves: list[list[float | int]] = []
            group_indices: list[int] = []

            for src in my_planets:
                decision = encode_source_decision(src, state, transit, cfg.env)

                with torch.inference_mode():
                    outputs, sampled = _forward_single(policy, decision, device)

                # Store transition features
                idx = len(all_global)
                all_global.append(decision.global_features)
                all_src_scalars.append(decision.source_scalars)
                all_src_pos.append(decision.source_position)
                all_knn_scalars.append(decision.knn_scalars)
                all_knn_pos.append(decision.knn_positions)
                all_tgt_scalars.append(decision.target_scalars)
                all_tgt_pos.append(decision.target_positions)
                all_tgt_mask.append(decision.target_mask)

                tgt_idx = int(sampled.target_index.item())
                frac_bin = int(sampled.fraction_bin.item())
                all_target_idx.append(tgt_idx)
                all_frac_bin.append(frac_bin)
                all_log_prob.append(float(sampled.log_prob.item()))
                all_values.append(float(outputs.value.item()))
                group_indices.append(idx)

                # Execute action if not NoOp
                if tgt_idx > 0:
                    target_offset = tgt_idx - 1
                    if target_offset < len(decision.target_planet_ids):
                        fraction = cfg.env.ship_fractions[frac_bin]
                        ships = int(src.ships * fraction)
                        if ships > 0:
                            target_id = decision.target_planet_ids[target_offset]
                            angle = decision.target_angles[target_offset]
                            moves.append([src.id, float(angle), ships])

                            # Update transit for subsequent planet decisions
                            tgt_planet = state.planets_by_id.get(target_id)
                            if tgt_planet:
                                speed = fleet_speed(ships)
                                dist = math.hypot(src.x - tgt_planet.x, src.y - tgt_planet.y)
                                eta = dist / max(speed, 0.1)
                                transit.add_fleet(target_id, float(ships), eta, is_friendly=True)
                            src.ships = max(0, src.ships - ships)

            # Step environment
            result = env.step(moves)
            running_rewards[env_idx] += result.reward
            groups_per_env[env_idx].append(
                StepGroup(indices=group_indices, reward=result.reward, done=result.done)
            )

            if result.done:
                episode_rewards.append(running_rewards[env_idx])
                running_rewards[env_idx] = 0.0
                next_seed += 1
                if scheduler is not None:
                    num_p, opps = scheduler.sample_episode()
                    new_obs = env.reset(seed=next_seed, num_players=num_p, opponents=opps)
                else:
                    new_obs = env.reset(seed=next_seed)
                next_obs_list.append(new_obs)
            else:
                next_obs_list.append(result.obs)

        raw_obs_per_env = next_obs_list

    # Compute GAE returns and advantages
    N = len(all_global)
    returns = [0.0] * N
    advantages = [0.0] * N
    gamma = cfg.ppo.gamma
    lam = cfg.ppo.gae_lambda

    # Bootstrap final values
    next_values = _bootstrap_values(policy, raw_obs_per_env, cfg, device)

    for env_idx, groups in enumerate(groups_per_env):
        n_steps = len(groups)
        if n_steps == 0:
            continue

        # Mean value per step group (aggregate per-planet values to step-level)
        step_values = []
        for g in groups:
            if g.indices:
                step_values.append(np.mean([all_values[i] for i in g.indices]))
            else:
                step_values.append(0.0)

        # GAE(lambda)
        gae = 0.0
        for t in reversed(range(n_steps)):
            non_terminal = 1.0 - float(groups[t].done)
            if t == n_steps - 1:
                next_v = next_values[env_idx] * non_terminal
            else:
                next_v = step_values[t + 1] * non_terminal

            delta = groups[t].reward + gamma * next_v - step_values[t]
            gae = delta + gamma * lam * non_terminal * gae

            step_return = gae + step_values[t]
            for idx in groups[t].indices:
                returns[idx] = step_return
                advantages[idx] = step_return - all_values[idx]

    # Build batch
    if N == 0:
        batch = _empty_batch(cfg)
    else:
        batch = TransitionBatch(
            global_features=torch.from_numpy(np.array(all_global, dtype=np.float32)),
            source_scalars=torch.from_numpy(np.array(all_src_scalars, dtype=np.float32)),
            source_positions=torch.from_numpy(np.array(all_src_pos, dtype=np.float32)),
            knn_scalars=torch.from_numpy(np.array(all_knn_scalars, dtype=np.float32)),
            knn_positions=torch.from_numpy(np.array(all_knn_pos, dtype=np.float32)),
            target_scalars=torch.from_numpy(np.array(all_tgt_scalars, dtype=np.float32)),
            target_positions=torch.from_numpy(np.array(all_tgt_pos, dtype=np.float32)),
            target_mask=torch.from_numpy(np.array(all_tgt_mask, dtype=bool)),
            target_index=torch.tensor(all_target_idx, dtype=torch.long),
            fraction_bin=torch.tensor(all_frac_bin, dtype=torch.long),
            log_prob=torch.tensor(all_log_prob, dtype=torch.float32),
            returns=torch.tensor(returns, dtype=torch.float32),
            advantages=torch.tensor(advantages, dtype=torch.float32),
            values=torch.tensor(all_values, dtype=torch.float32),
        )

    stats = {
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episodes_finished": float(len(episode_rewards)),
        "samples": float(N),
    }
    return batch, raw_obs_per_env, next_seed, stats


def _bootstrap_values(
    policy: TransformerPolicy,
    raw_obs_list: list,
    cfg: TrainConfig,
    device: torch.device,
) -> list[float]:
    """Compute value estimates for the last observation of each env."""
    values = []
    for obs in raw_obs_list:
        state = parse_observation(obs)
        my_planets = [p for p in state.planets if p.owner == state.player]
        if not my_planets:
            values.append(0.0)
            continue
        transit = compute_fleet_transit(state)
        # Use the planet with most ships for value estimate
        src = max(my_planets, key=lambda p: p.ships)
        decision = encode_source_decision(src, state, transit, cfg.env)
        with torch.inference_mode():
            outputs = policy(
                torch.from_numpy(decision.global_features).unsqueeze(0).to(device),
                torch.from_numpy(decision.source_scalars).unsqueeze(0).to(device),
                torch.from_numpy(decision.source_position).unsqueeze(0).to(device),
                torch.from_numpy(decision.knn_scalars).unsqueeze(0).to(device),
                torch.from_numpy(decision.knn_positions).unsqueeze(0).to(device),
                torch.from_numpy(decision.target_scalars).unsqueeze(0).to(device),
                torch.from_numpy(decision.target_positions).unsqueeze(0).to(device),
                torch.from_numpy(decision.target_mask).unsqueeze(0).to(device),
            )
        values.append(float(outputs.value.item()))
    return values


def _empty_batch(cfg: TrainConfig) -> TransitionBatch:
    T = cfg.env.max_targets
    K = cfg.env.k_neighbors
    return TransitionBatch(
        global_features=torch.zeros(0, GLOBAL_DIM),
        source_scalars=torch.zeros(0, SOURCE_SCALAR_DIM),
        source_positions=torch.zeros(0, 2),
        knn_scalars=torch.zeros(0, K, KNN_SCALAR_DIM),
        knn_positions=torch.zeros(0, K, 2),
        target_scalars=torch.zeros(0, T, TARGET_SCALAR_DIM),
        target_positions=torch.zeros(0, T, 2),
        target_mask=torch.zeros(0, T + 2, dtype=torch.bool),
        target_index=torch.zeros(0, dtype=torch.long),
        fraction_bin=torch.zeros(0, dtype=torch.long),
        log_prob=torch.zeros(0),
        returns=torch.zeros(0),
        advantages=torch.zeros(0),
        values=torch.zeros(0),
    )


def save_checkpoint(
    save_dir: Path,
    run_name: str,
    update: int,
    policy: TransformerPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
) -> Path:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "update": update,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, run_dir / "ckpt_last.pt")
    torch.save(state, run_dir / f"ckpt_{update:06d}.pt")
    return run_dir / "ckpt_last.pt"


def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Config: {cfg.run_name}, device={device}, updates={cfg.ppo.total_updates}")
    print(f"  envs={cfg.ppo.num_envs}, rollout_steps={cfg.ppo.rollout_steps}, "
          f"opponent={cfg.opponent}")

    # Count parameters
    test_policy = TransformerPolicy(cfg.model, cfg.env)
    n_params = sum(p.numel() for p in test_policy.parameters())
    print(f"  model params: {n_params:,}")
    del test_policy

    # Logger
    logger = TrainLogger(cfg.log_dir, cfg.run_name)

    # Policy
    policy = TransformerPolicy(cfg.model, cfg.env).to(device)
    save_dir = Path(cfg.save_dir)

    # ── Imitation learning phases ──────────────────────────────────────────
    demo_buffer = None
    bc_checkpoint_path: Path | None = None

    if cfg.imitation.enabled:
        from .imitation import bc_pretrain, collect_demonstrations

        # Phase 1: Collect demonstrations
        print(f"\n=== Phase 1: Collecting {cfg.imitation.bc_games} demo games "
              f"(expert={cfg.imitation.bc_expert}) ===")
        demo_buffer = collect_demonstrations(
            n_games=cfg.imitation.bc_games,
            cfg=cfg,
            opponent_name=cfg.imitation.bc_demo_opponent,
        )
        print(f"  Buffer size: {len(demo_buffer)}")

        # Phase 2: BC pretrain
        print(f"\n=== Phase 2: BC pretraining ({cfg.imitation.bc_epochs} epochs) ===")
        bc_pretrain(policy, demo_buffer, cfg.imitation, device, logger)

        # Save BC-pretrained checkpoint
        bc_checkpoint_path = save_checkpoint(
            save_dir, cfg.run_name, 0, policy,
            torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr), cfg,
        )
        print(f"  BC checkpoint saved: {bc_checkpoint_path}")

    # ── Build opponents ─────────────────────────────────────────────────────
    if cfg.imitation.enabled and cfg.imitation.distilled_opponent and bc_checkpoint_path is not None:
        print("  Using distilled opponent (BC-pretrained)")
        rule_based_opponent = build_opponent(
            "distilled", cfg=cfg, device=device,
            checkpoint_path=bc_checkpoint_path,
        )
    else:
        rule_based_opponent = build_opponent(cfg.opponent, cfg=cfg, device=device)

    # Self-play opponent (always created; scheduler decides when to use it)
    sp_opponent = SelfPlayOpponent(cfg, device=device,
                                   deterministic=cfg.self_play_deterministic)
    sp_opponent.sync_from(policy)

    # Mixed scheduler for 2p/4p + rule-based/self-play mixing
    use_scheduler = cfg.four_player_prob > 0.0 or cfg.rule_based_prob_start < 1.0
    scheduler: MixedScheduler | None = None
    if use_scheduler:
        scheduler = MixedScheduler(cfg, rule_based_opponent, sp_opponent)
        print(f"  MixedScheduler: 4p_prob={cfg.four_player_prob}, "
              f"rule_based={cfg.rule_based_prob_start:.1f}→{cfg.rule_based_prob_end:.1f} "
              f"over {cfg.rule_based_decay_updates} updates")

    envs = [OrbitWarsEnv(cfg, rule_based_opponent, env_index=idx) for idx in range(cfg.ppo.num_envs)]

    next_seed = cfg.seed
    raw_obs_per_env = []
    for env in envs:
        if scheduler is not None:
            num_p, opps = scheduler.sample_episode()
            raw_obs_per_env.append(env.reset(seed=next_seed, num_players=num_p, opponents=opps))
        else:
            raw_obs_per_env.append(env.reset(seed=next_seed))
        next_seed += 1

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.ppo.lr)

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n=== Phase 3: PPO training ({cfg.ppo.total_updates} updates) ===")
    t_start = time.time()

    for update in range(1, cfg.ppo.total_updates + 1):
        t_update = time.time()

        if scheduler is not None:
            scheduler.set_update(update)

        # Compute imitation coefficient (linear decay)
        imitation_coef = 0.0
        if cfg.imitation.enabled and demo_buffer is not None:
            decay_frac = update / max(cfg.imitation.coef_decay_updates, 1)
            imitation_coef = cfg.imitation.coef_start * max(0.0, 1.0 - decay_frac)

        batch, raw_obs_per_env, next_seed, stats = collect_rollout(
            envs, raw_obs_per_env, policy, cfg, device, next_seed,
            scheduler=scheduler,
        )

        metrics = ppo_update(
            policy, optimizer, batch,
            clip_coef=cfg.ppo.clip_coef,
            ent_coef=cfg.ppo.ent_coef,
            vf_coef=cfg.ppo.vf_coef,
            max_grad_norm=cfg.ppo.max_grad_norm,
            epochs=cfg.ppo.epochs,
            minibatch_size=cfg.ppo.minibatch_size,
            device=device,
            demo_buffer=demo_buffer,
            imitation_coef=imitation_coef,
        )

        # Sync self-play opponent periodically
        if update % cfg.self_play_update_interval == 0:
            sp_opponent.sync_from(policy)

        # Merge all metrics for logging
        all_metrics = {**stats, **metrics, "imitation_coef": imitation_coef}
        logger.log_update(update, all_metrics)

        if update % cfg.log_every == 0:
            elapsed = time.time() - t_start
            update_time = time.time() - t_update
            im_str = f"  im_coef={imitation_coef:.3f}" if cfg.imitation.enabled else ""
            print(
                f"update={update:4d}  reward={stats['episode_reward_mean']:+.3f}  "
                f"eps={int(stats['episodes_finished'])}  samples={int(stats['samples'])}  "
                f"loss={metrics['loss']:.4f}  ploss={metrics['policy_loss']:.4f}  "
                f"vloss={metrics['value_loss']:.4f}  ent={metrics['entropy']:.3f}  "
                f"dt={update_time:.1f}s  total={elapsed:.0f}s{im_str}"
            )

        # Periodic evaluation
        if cfg.eval.eval_every > 0 and update % cfg.eval.eval_every == 0:
            print(f"\n  Running eval ({cfg.eval.eval_games} games)...")
            eval_results = run_periodic_eval(policy, cfg, device)
            logger.log_eval(update, eval_results)
            for r in eval_results:
                print(f"    vs {r.opponent_name}: W={r.win_rate:.0%} L={r.loss_rate:.0%} "
                      f"T={r.tie_rate:.0%} (n={r.n_games})")
            print()

        if update % cfg.checkpoint_every == 0 or update == cfg.ppo.total_updates:
            save_checkpoint(save_dir, cfg.run_name, update, policy, optimizer, cfg)
            print(f"  -> saved checkpoint at update {update}")

    logger.close()
    print(f"\nTraining complete. Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
