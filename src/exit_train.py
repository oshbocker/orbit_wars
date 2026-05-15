"""Expert Iteration (ExIt) training loop for Orbit Wars.

Instead of PPO's noisy policy gradients, ExIt uses per-planet lookahead search
to generate improved training targets, then trains the policy via supervised
learning. This gives per-decision credit assignment and accumulated data reuse.

Training loop:
  1. COLLECT: Play games with current policy, record decisions + game states
  2. SEARCH-IMPROVE: For each decision, run forward sim to produce improved targets
  3. TRAIN: Supervised learning on search-improved data + game outcome for value
  4. EVALUATE: Test vs baselines, update champion pool
"""
from __future__ import annotations

import argparse
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExItConfig, TrainConfig, load_train_config
from .env import OrbitWarsEnv
from .features import (
    FleetTransitState,
    SourceDecision,
    compute_fleet_transit,
    encode_source_decision,
    fleet_speed,
)
from .game_types import GameState, parse_observation
from .logging import TrainLogger, run_periodic_eval
from .opponents import (
    ChampionPoolOpponent,
    OpponentPolicy,
    build_opponent,
    _policy_act,
)
from .policy import TransformerPolicy
from .ppo import sample_actions
from .search import search_improve_with_player
from .simulator import SimState, build_sim_state


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class DecisionRecord:
    """One per-planet decision recorded during game play."""
    decision: SourceDecision
    game_state: GameState
    sim_state: SimState
    player: int
    step: int
    # Filled after game ends
    game_outcome: float = 0.0  # +1 win, -1 loss, 0 tie


@dataclass
class GameRecord:
    """All decisions from one game."""
    decisions: list[DecisionRecord] = field(default_factory=list)
    outcome: float = 0.0  # +1 win, -1 loss, 0 tie


@dataclass
class TrainSample:
    """One training sample: features + search targets + game outcome."""
    # Features (same as SourceDecision arrays)
    global_features: np.ndarray
    source_scalars: np.ndarray
    source_position: np.ndarray
    knn_scalars: np.ndarray
    knn_positions: np.ndarray
    target_scalars: np.ndarray
    target_positions: np.ndarray
    target_mask: np.ndarray
    # Search-improved targets
    target_probs: np.ndarray    # [1+T] improved target distribution
    fraction_probs: np.ndarray  # [T, num_fracs] improved fraction distributions
    # Game outcome for value training
    game_outcome: float


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expert Iteration training")
    parser.add_argument("--config", type=str, default="configs/expert_iteration.yaml")
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


# ── Phase 1: Game Collection ────────────────────────────────────────────────

def collect_games(
    policy: TransformerPolicy,
    opponents: list[OpponentPolicy],
    n_games: int,
    cfg: TrainConfig,
    device: torch.device,
    seed: int = 0,
) -> list[GameRecord]:
    """Play n_games using current policy and record all per-planet decisions."""
    records: list[GameRecord] = []

    for game_i in range(n_games):
        game_seed = seed + game_i
        # Choose number of players
        is_4p = random.random() < cfg.exit.four_player_prob
        num_players = 4 if is_4p else 2

        # Pick opponents
        n_opp = num_players - 1
        game_opps = [opponents[i % len(opponents)] for i in range(n_opp)]
        # Refresh champion pool opponents
        for opp in game_opps:
            if isinstance(opp, ChampionPoolOpponent):
                opp.load_random()

        env = OrbitWarsEnv(cfg, game_opps[0])
        obs = env.reset(seed=game_seed, num_players=num_players, opponents=game_opps)

        game_record = GameRecord()
        done = False

        while not done:
            state = parse_observation(obs)
            my_planets = sorted(
                [p for p in state.planets if p.owner == state.player],
                key=lambda p: -p.ships,
            )

            transit = compute_fleet_transit(state)
            sim_state = build_sim_state(state)
            moves: list[list[float | int]] = []

            for src in my_planets:
                decision = encode_source_decision(src, state, transit, cfg.env)

                # Record the decision with current sim state
                record = DecisionRecord(
                    decision=decision,
                    game_state=state,
                    sim_state=sim_state.copy(),
                    player=state.player,
                    step=state.step,
                )
                game_record.decisions.append(record)

                # Use policy to pick action
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
                    sampled = sample_actions(outputs, deterministic=False)

                tgt_idx = int(sampled.target_index.item())
                if tgt_idx > 0:
                    target_offset = tgt_idx - 1
                    if target_offset < len(decision.target_planet_ids):
                        frac_bin = int(sampled.fraction_bin.item())
                        fraction = cfg.env.ship_fractions[frac_bin]
                        ships = int(src.ships * fraction)
                        if ships > 0:
                            target_id = decision.target_planet_ids[target_offset]
                            angle = decision.target_angles[target_offset]
                            moves.append([src.id, float(angle), ships])

                            # Update transit for subsequent planets
                            tgt_planet = state.planets_by_id.get(target_id)
                            if tgt_planet:
                                speed = fleet_speed(ships)
                                dist = math.hypot(src.x - tgt_planet.x, src.y - tgt_planet.y)
                                eta = dist / max(speed, 0.1)
                                transit.add_fleet(target_id, float(ships), eta, is_friendly=True)
                            src.ships = max(0, src.ships - ships)

            result = env.step(moves)
            obs = result.obs
            done = result.done

        # Determine game outcome from terminal reward
        outcome = result.reward if result.reward != 0 else 0.0
        # Clamp to [-1, 1]
        outcome = max(-1.0, min(1.0, outcome))
        game_record.outcome = outcome

        # Propagate outcome to all decisions
        for rec in game_record.decisions:
            rec.game_outcome = outcome

        records.append(game_record)

    return records


# ── Phase 2: Search Improvement ─────────────────────────────────────────────

def search_improve_dataset(
    records: list[GameRecord],
    cfg: TrainConfig,
) -> list[TrainSample]:
    """Run search on each recorded decision to produce improved targets."""
    samples: list[TrainSample] = []

    for game_record in records:
        for rec in game_record.decisions:
            target_probs, fraction_probs = search_improve_with_player(
                decision=rec.decision,
                sim_state=rec.sim_state,
                game_state=rec.game_state,
                player=rec.player,
                step=rec.step,
                exit_cfg=cfg.exit,
                env_cfg=cfg.env,
            )

            samples.append(TrainSample(
                global_features=rec.decision.global_features,
                source_scalars=rec.decision.source_scalars,
                source_position=rec.decision.source_position,
                knn_scalars=rec.decision.knn_scalars,
                knn_positions=rec.decision.knn_positions,
                target_scalars=rec.decision.target_scalars,
                target_positions=rec.decision.target_positions,
                target_mask=rec.decision.target_mask,
                target_probs=target_probs,
                fraction_probs=fraction_probs,
                game_outcome=rec.game_outcome,
            ))

    return samples


# ── Phase 3: Supervised Training ────────────────────────────────────────────

def _build_batch(
    samples: list[TrainSample],
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build a batch of tensors from sample indices."""
    batch_samples = [samples[i] for i in indices]

    return {
        "global_features": torch.from_numpy(
            np.array([s.global_features for s in batch_samples], dtype=np.float32)
        ).to(device),
        "source_scalars": torch.from_numpy(
            np.array([s.source_scalars for s in batch_samples], dtype=np.float32)
        ).to(device),
        "source_positions": torch.from_numpy(
            np.array([s.source_position for s in batch_samples], dtype=np.float32)
        ).to(device),
        "knn_scalars": torch.from_numpy(
            np.array([s.knn_scalars for s in batch_samples], dtype=np.float32)
        ).to(device),
        "knn_positions": torch.from_numpy(
            np.array([s.knn_positions for s in batch_samples], dtype=np.float32)
        ).to(device),
        "target_scalars": torch.from_numpy(
            np.array([s.target_scalars for s in batch_samples], dtype=np.float32)
        ).to(device),
        "target_positions": torch.from_numpy(
            np.array([s.target_positions for s in batch_samples], dtype=np.float32)
        ).to(device),
        "target_mask": torch.from_numpy(
            np.array([s.target_mask for s in batch_samples], dtype=bool)
        ).to(device),
        "target_probs": torch.from_numpy(
            np.array([s.target_probs for s in batch_samples], dtype=np.float32)
        ).to(device),
        "fraction_probs": torch.from_numpy(
            np.array([s.fraction_probs for s in batch_samples], dtype=np.float32)
        ).to(device),
        "game_outcome": torch.tensor(
            [s.game_outcome for s in batch_samples], dtype=torch.float32
        ).to(device),
    }


def train_epoch(
    policy: TransformerPolicy,
    optimizer: torch.optim.Optimizer,
    samples: list[TrainSample],
    cfg: TrainConfig,
    device: torch.device,
    global_step: int,
) -> tuple[dict[str, float], int]:
    """Train one epoch on the dataset. Returns (metrics, updated_global_step)."""
    N = len(samples)
    if N < 4:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}, global_step

    batch_size = min(N, cfg.exit.train_batch_size)
    order = np.random.permutation(N)

    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_batches = 0

    for start in range(0, N, batch_size):
        idx = order[start:start + batch_size]
        if len(idx) < 4:
            continue

        batch = _build_batch(samples, idx, device)

        # Forward pass
        outputs = policy(
            batch["global_features"],
            batch["source_scalars"],
            batch["source_positions"],
            batch["knn_scalars"],
            batch["knn_positions"],
            batch["target_scalars"],
            batch["target_positions"],
            batch["target_mask"],
        )

        # Policy loss: cross-entropy with search-improved target distribution
        # target_logits: [B, 1+T], target_probs: [B, 1+T]
        log_probs = F.log_softmax(outputs.target_logits, dim=-1)
        policy_loss = -(batch["target_probs"] * log_probs).sum(dim=-1).mean()

        # Fraction loss: cross-entropy per target with search-improved fractions
        # fraction_logits: [B, T, num_fracs], fraction_probs: [B, T, num_fracs]
        B, T, F_dim = outputs.fraction_logits.shape
        frac_log_probs = F.log_softmax(outputs.fraction_logits, dim=-1)
        # Only compute fraction loss for valid targets (mask[2:])
        frac_mask = batch["target_mask"][:, 2:]  # [B, T]
        frac_loss_per_target = -(batch["fraction_probs"] * frac_log_probs).sum(dim=-1)  # [B, T]
        frac_loss_masked = frac_loss_per_target * frac_mask
        n_valid = frac_mask.sum().clamp(min=1)
        fraction_loss = frac_loss_masked.sum() / n_valid

        combined_policy_loss = policy_loss + fraction_loss

        # Value loss: MSE between predicted value and game outcome
        value_loss = F.mse_loss(outputs.value, batch["game_outcome"])

        # Entropy bonus (for exploration)
        probs = F.softmax(outputs.target_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        loss = (combined_policy_loss
                + cfg.exit.value_loss_coef * value_loss
                - cfg.exit.entropy_coef * entropy)

        # LR schedule
        _update_lr(optimizer, global_step, cfg.exit)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.exit.max_grad_norm)
        optimizer.step()

        metrics["loss"] += float(loss.detach().cpu())
        metrics["policy_loss"] += float(combined_policy_loss.detach().cpu())
        metrics["value_loss"] += float(value_loss.detach().cpu())
        metrics["entropy"] += float(entropy.detach().cpu())
        n_batches += 1
        global_step += 1

    return {k: v / max(n_batches, 1) for k, v in metrics.items()}, global_step


def _update_lr(optimizer: torch.optim.Optimizer, step: int, exit_cfg: ExItConfig) -> None:
    """Cosine schedule with linear warmup."""
    if exit_cfg.lr_schedule == "constant":
        return

    warmup = exit_cfg.lr_warmup_steps
    base_lr = exit_cfg.train_lr
    # Estimate total steps for cosine decay
    total_steps = exit_cfg.iterations * exit_cfg.train_epochs * 100  # rough estimate

    if step < warmup:
        lr = base_lr * (step + 1) / warmup
    else:
        progress = (step - warmup) / max(total_steps - warmup, 1)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    for pg in optimizer.param_groups:
        pg["lr"] = lr


# ── Checkpoint Management ───────────────────────────────────────────────────

def save_checkpoint(
    save_dir: Path,
    run_name: str,
    iteration: int,
    policy: TransformerPolicy,
    optimizer: torch.optim.Optimizer,
) -> Path:
    run_dir = save_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "update": iteration,
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state, run_dir / "ckpt_last.pt")
    torch.save(state, run_dir / f"ckpt_{iteration:06d}.pt")
    return run_dir / "ckpt_last.pt"


def _manage_champion_pool(
    save_dir: Path,
    run_name: str,
    max_size: int,
) -> None:
    """Keep only the most recent max_size checkpoints in the pool."""
    run_dir = save_dir / run_name
    ckpts = sorted(run_dir.glob("ckpt_[0-9]*.pt"))
    while len(ckpts) > max_size:
        oldest = ckpts.pop(0)
        oldest.unlink()


# ── Imitation Pretraining (optional) ────────────────────────────────────────

def _run_bc_pretrain(
    policy: TransformerPolicy,
    cfg: TrainConfig,
    device: torch.device,
    logger: TrainLogger,
    save_dir: Path,
) -> Path | None:
    """Run optional BC pretraining if imitation is enabled."""
    if not cfg.imitation.enabled:
        return None

    from .imitation import bc_pretrain, collect_demonstrations

    print(f"\n=== BC Pretraining: {cfg.imitation.bc_games} demo games "
          f"(expert={cfg.imitation.bc_expert}) ===")
    demo_buffer = collect_demonstrations(
        n_games=cfg.imitation.bc_games,
        cfg=cfg,
        opponent_name=cfg.imitation.bc_demo_opponent,
    )
    print(f"  Buffer size: {len(demo_buffer)}")

    print(f"  Training {cfg.imitation.bc_epochs} epochs...")
    bc_pretrain(policy, demo_buffer, cfg.imitation, device, logger)

    # Save BC checkpoint
    bc_path = save_checkpoint(
        save_dir, cfg.run_name, 0, policy,
        torch.optim.Adam(policy.parameters(), lr=cfg.exit.train_lr),
    )
    print(f"  BC checkpoint saved: {bc_path}")
    return bc_path


# ── Main Training Loop ──────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_train_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)

    print(f"Expert Iteration: {cfg.run_name}")
    print(f"  device={device}, iterations={cfg.exit.iterations}")
    print(f"  games/iter={cfg.exit.games_per_iter}, search_depth={cfg.exit.search_depth}")
    print(f"  train_epochs={cfg.exit.train_epochs}, batch_size={cfg.exit.train_batch_size}")
    print(f"  lr={cfg.exit.train_lr}, schedule={cfg.exit.lr_schedule}")

    # Model
    policy = TransformerPolicy(cfg.model, cfg.env).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  model params: {n_params:,}")

    # Logger
    logger = TrainLogger(cfg.log_dir, cfg.run_name)
    save_dir = Path(cfg.save_dir)

    # Optional BC pretraining
    bc_path = _run_bc_pretrain(policy, cfg, device, logger, save_dir)

    # Build opponents
    opponents: list[OpponentPolicy] = []
    opp_name = cfg.exit.opponent
    if opp_name == "champion_pool":
        pool_dir = save_dir / cfg.run_name
        pool_dir.mkdir(parents=True, exist_ok=True)
        champion_opp = ChampionPoolOpponent(pool_dir, cfg, device=device)
        opponents.append(champion_opp)
        # Also add a rule-based opponent for diversity
        opponents.append(build_opponent("apex"))
    else:
        opponents.append(build_opponent(opp_name, cfg=cfg, device=device))

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.exit.train_lr)

    # Accumulated dataset (keeps last N iterations of data)
    dataset_buffer: deque[list[TrainSample]] = deque(maxlen=cfg.exit.dataset_max_iters)
    global_step = 0
    next_seed = cfg.seed

    print(f"\n=== Expert Iteration Training ({cfg.exit.iterations} iterations) ===")
    t_start = time.time()

    for iteration in range(1, cfg.exit.iterations + 1):
        t_iter = time.time()

        # Phase 1: Collect games
        policy.eval()
        game_records = collect_games(
            policy, opponents, cfg.exit.games_per_iter, cfg, device, seed=next_seed,
        )
        next_seed += cfg.exit.games_per_iter

        n_decisions = sum(len(g.decisions) for g in game_records)
        outcomes = [g.outcome for g in game_records]
        win_rate = sum(1 for o in outcomes if o > 0) / max(len(outcomes), 1)
        t_collect = time.time() - t_iter

        # Phase 2: Search improvement
        t_search_start = time.time()
        new_samples = search_improve_dataset(game_records, cfg)
        t_search = time.time() - t_search_start

        # Add to accumulated dataset
        dataset_buffer.append(new_samples)
        all_samples = [s for batch in dataset_buffer for s in batch]

        # Phase 3: Train
        policy.train()
        epoch_metrics: dict[str, float] = {}
        for _ep in range(cfg.exit.train_epochs):
            metrics, global_step = train_epoch(
                policy, optimizer, all_samples, cfg, device, global_step,
            )
            epoch_metrics = metrics  # keep last epoch's metrics

        t_train = time.time() - t_search_start - t_search
        t_total_iter = time.time() - t_iter

        # Log
        log_metrics = {
            "episode_reward_mean": float(np.mean(outcomes)) if outcomes else 0.0,
            "win_rate": win_rate,
            "decisions": float(n_decisions),
            "dataset_size": float(len(all_samples)),
            **epoch_metrics,
        }
        logger.log_update(iteration, log_metrics)

        print(
            f"iter={iteration:4d}  win={win_rate:.0%}  decisions={n_decisions}  "
            f"dataset={len(all_samples)}  "
            f"loss={epoch_metrics.get('loss', 0):.4f}  "
            f"ploss={epoch_metrics.get('policy_loss', 0):.4f}  "
            f"vloss={epoch_metrics.get('value_loss', 0):.4f}  "
            f"collect={t_collect:.1f}s  search={t_search:.1f}s  "
            f"train={t_train:.1f}s  total={t_total_iter:.1f}s"
        )

        # Periodic evaluation
        if cfg.eval.eval_every > 0 and iteration % cfg.eval.eval_every == 0:
            print(f"\n  Running eval ({cfg.eval.eval_games} games)...")
            eval_results = run_periodic_eval(policy, cfg, device)
            logger.log_eval(iteration, eval_results)
            for r in eval_results:
                print(f"    vs {r.opponent_name}: W={r.win_rate:.0%} L={r.loss_rate:.0%} "
                      f"T={r.tie_rate:.0%} (n={r.n_games})")
            print()

        # Save checkpoint
        if iteration % cfg.checkpoint_every == 0 or iteration == cfg.exit.iterations:
            save_checkpoint(save_dir, cfg.run_name, iteration, policy, optimizer)
            _manage_champion_pool(save_dir, cfg.run_name, cfg.exit.champion_pool_size)
            print(f"  -> saved checkpoint at iteration {iteration}")

            # Refresh champion pool opponents
            for opp in opponents:
                if isinstance(opp, ChampionPoolOpponent):
                    opp.load_random()

    logger.close()
    print(f"\nTraining complete. Total time: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
