"""Parallel rollout collection using subprocess workers."""
from __future__ import annotations

import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import V2Config, v2_config_from_dict, v2_config_to_dict
from .ppo import V2TransitionBatch


@dataclass
class WorkerTransitions:
    """Serializable rollout data from one worker (numpy arrays)."""
    planet_features: np.ndarray    # [N, P, F]
    global_features: np.ndarray    # [N, G]
    planet_mask: np.ndarray        # [N, P]
    own_mask: np.ndarray           # [N, P]
    reachability_mask: np.ndarray  # [N, P, P]
    target_indices: np.ndarray     # [N, P]
    log_prob: np.ndarray           # [N]
    returns: np.ndarray            # [N]
    advantages: np.ndarray         # [N]
    values: np.ndarray             # [N]
    stats: dict[str, float]


def _worker_fn(conn: mp.connection.Connection, worker_id: int, cfg_dict: dict) -> None:
    """Subprocess entry point. Owns env + opponent + model."""
    import os
    import sys

    # Suppress stdout/stderr in workers to avoid pipe buffer deadlocks
    # (kaggle_environments prints a lot of INFO on import)
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

    # Lazy imports inside subprocess to avoid pickling issues
    from src.opponents import build_opponent

    from .config import v2_config_from_dict
    from .env import V2OrbitWarsEnv
    from .features import V2Features, encode_features
    from .model import OrbitNet
    from .train import V2MixedScheduler, V2SelfPlayOpponent, collect_rollout

    cfg = v2_config_from_dict(cfg_dict)
    device = torch.device("cpu")  # workers always on CPU

    # Seed with worker-specific offset
    worker_seed = cfg.seed + worker_id * 10000
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # Create model (for inference during rollout)
    model = OrbitNet(cfg.model).to(device)
    model.eval()

    # Create opponent
    rule_based_opponent = build_opponent(cfg.opponent)

    # Self-play opponent (for mixed scheduler)
    sp_opponent = V2SelfPlayOpponent(cfg, device=device,
                                     deterministic=cfg.self_play_deterministic)
    sp_opponent.sync_from(model)

    # Mixed scheduler
    use_scheduler = cfg.four_player_prob > 0.0 or cfg.rule_based_prob_start < 1.0
    scheduler = None
    if use_scheduler:
        scheduler = V2MixedScheduler(cfg, rule_based_opponent, sp_opponent)

    # Each worker gets 1 env (total envs = num_workers)
    envs = [V2OrbitWarsEnv(cfg, rule_based_opponent, env_index=worker_id)]
    next_seed = worker_seed

    # Initial reset
    features_per_env = []
    for env in envs:
        if scheduler is not None:
            num_p, opps = scheduler.sample_episode()
            features_per_env.append(env.reset(seed=next_seed, num_players=num_p, opponents=opps))
        else:
            features_per_env.append(env.reset(seed=next_seed))
        next_seed += 1

    # Signal that worker is ready
    conn.send("ready")

    try:
        while True:
            try:
                cmd = conn.recv()
            except (EOFError, BrokenPipeError):
                break

            if cmd[0] == "collect":
                update_num = cmd[1]
                if scheduler is not None:
                    scheduler.set_update(update_num)

                batch, features_per_env, next_seed, stats = collect_rollout(
                    envs, features_per_env, model, cfg, device, next_seed,
                    scheduler=scheduler,
                )

                # Convert batch to numpy for serialization
                wt = WorkerTransitions(
                    planet_features=batch.planet_features.numpy(),
                    global_features=batch.global_features.numpy(),
                    planet_mask=batch.planet_mask.numpy(),
                    own_mask=batch.own_mask.numpy(),
                    reachability_mask=batch.reachability_mask.numpy(),
                    target_indices=batch.target_indices.numpy(),
                    log_prob=batch.log_prob.numpy(),
                    returns=batch.returns.numpy(),
                    advantages=batch.advantages.numpy(),
                    values=batch.values.numpy(),
                    stats=stats,
                )
                conn.send(wt)

            elif cmd[0] == "sync_weights":
                state_dict = cmd[1]
                cpu_state = {k: v.cpu() for k, v in state_dict.items()}
                model.load_state_dict(cpu_state)
                model.eval()
                sp_opponent.sync_from(model)
                conn.send("ok")

            elif cmd[0] == "shutdown":
                break
    except Exception:
        # Worker crashed — send error signal so main doesn't hang
        try:
            conn.send("error")
        except (BrokenPipeError, OSError):
            pass
    finally:
        conn.close()
        devnull.close()


class ParallelRolloutCollector:
    """Manages subprocess workers for parallel rollout collection."""

    def __init__(self, cfg: V2Config, num_workers: int) -> None:
        self.num_workers = num_workers
        self.cfg_dict = v2_config_to_dict(cfg)

        # Override num_envs in worker config: each worker gets 1 env
        self.cfg_dict["ppo"]["num_envs"] = 1

        self._conns: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []

        ctx = mp.get_context("spawn")
        for i in range(num_workers):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_fn,
                args=(child_conn, i, self.cfg_dict),
                daemon=True,
            )
            proc.start()
            child_conn.close()  # parent doesn't need child's end
            self._conns.append(parent_conn)
            self._procs.append(proc)

        # Wait for all workers to be ready (env created, model loaded)
        for conn in self._conns:
            msg = conn.recv()
            if msg != "ready":
                raise RuntimeError(f"Worker failed to initialize: {msg}")

    def collect(self, update: int) -> tuple[V2TransitionBatch, dict[str, float]]:
        """Collect rollouts from all workers in parallel."""
        # Send collect command to all workers
        for conn in self._conns:
            conn.send(("collect", update))

        # Receive results from all workers
        results: list[WorkerTransitions] = []
        for conn in self._conns:
            results.append(conn.recv())

        # Concatenate transitions
        return self._merge_transitions(results)

    def sync_weights(self, model: torch.nn.Module) -> None:
        """Send updated model weights to all workers."""
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        for conn in self._conns:
            conn.send(("sync_weights", state_dict))
        # Wait for acknowledgment from all
        for conn in self._conns:
            conn.recv()

    def shutdown(self) -> None:
        """Clean shutdown of all workers."""
        for conn in self._conns:
            try:
                conn.send(("shutdown",))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    def _merge_transitions(
        self, results: list[WorkerTransitions],
    ) -> tuple[V2TransitionBatch, dict[str, float]]:
        """Concatenate worker transitions into a single batch."""
        # Filter out empty results
        non_empty = [r for r in results if r.planet_features.shape[0] > 0]

        if not non_empty:
            from .config import v2_config_from_dict
            from .train import _empty_batch
            cfg = v2_config_from_dict(self.cfg_dict)
            return _empty_batch(cfg), {"episode_reward_mean": 0.0, "episodes_finished": 0.0, "samples": 0.0}

        batch = V2TransitionBatch(
            planet_features=torch.from_numpy(np.concatenate([r.planet_features for r in non_empty])),
            global_features=torch.from_numpy(np.concatenate([r.global_features for r in non_empty])),
            planet_mask=torch.from_numpy(np.concatenate([r.planet_mask for r in non_empty])),
            own_mask=torch.from_numpy(np.concatenate([r.own_mask for r in non_empty])),
            reachability_mask=torch.from_numpy(np.concatenate([r.reachability_mask for r in non_empty])),
            target_indices=torch.from_numpy(np.concatenate([r.target_indices for r in non_empty])),
            log_prob=torch.from_numpy(np.concatenate([r.log_prob for r in non_empty])),
            returns=torch.from_numpy(np.concatenate([r.returns for r in non_empty])),
            advantages=torch.from_numpy(np.concatenate([r.advantages for r in non_empty])),
            values=torch.from_numpy(np.concatenate([r.values for r in non_empty])),
        )

        # Merge stats
        all_rewards = [r.stats.get("episode_reward_mean", 0.0) for r in results if r.stats.get("episodes_finished", 0) > 0]
        total_episodes = sum(r.stats.get("episodes_finished", 0) for r in results)
        total_samples = sum(r.stats.get("samples", 0) for r in results)

        stats = {
            "episode_reward_mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "episodes_finished": float(total_episodes),
            "samples": float(total_samples),
        }

        return batch, stats
