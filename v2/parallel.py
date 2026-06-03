"""Parallel rollout collection using subprocess workers (v4-complete).

Each worker owns 1 env + a copy of the model + (optionally) a PFSP opponent pool
synced from the main process, runs the SAME `collect_rollout` as the sequential
path, and ships back a fully-serialized v4 transition batch — including the v3/v4
`pair_features` and the v4 shot-success labels — plus its local PFSP win/game
deltas. PopArt value-norm stats are broadcast read-only for GAE denormalization.

The central process owns ALL learning: the PPO update, `value_norm.update`, the
PPG aux / shot-aux phases, and PFSP win-rate bookkeeping + snapshotting. Workers
only roll out. Snapshot weights are broadcast once (when a snapshot is first
created); thereafter only the tiny win/game stats ride along each sync.
"""
from __future__ import annotations

import multiprocessing as mp
import random
from dataclasses import dataclass

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
    frac_indices: np.ndarray       # [N, P]
    log_prob: np.ndarray           # [N]
    returns: np.ndarray            # [N]
    advantages: np.ndarray         # [N]
    values: np.ndarray             # [N]
    pair_features: np.ndarray | None    # [N, P, P, pf] or None
    shot_idx: np.ndarray | None         # [M] (row indices WITHIN this worker's batch)
    shot_src: np.ndarray | None         # [M]
    shot_tgt: np.ndarray | None         # [M]
    shot_label: np.ndarray | None       # [M]
    pfsp_delta: dict                     # name -> [wins_delta, games_delta]
    stats: dict


def _worker_fn(conn: mp.connection.Connection, worker_id: int, cfg_dict: dict) -> None:
    """Subprocess entry point. Owns env + opponent pool + model."""
    import os
    import sys

    # Suppress stdout/stderr in workers to avoid pipe buffer deadlocks
    # (kaggle_environments prints a lot of INFO on import).
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull

    # Pin each worker to a single thread to avoid BLAS/OMP oversubscription.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    # Lazy imports inside subprocess to avoid pickling issues.
    from src.opponents import build_opponent

    from .config import v2_config_from_dict
    from .env import V2FastEnv, V2OrbitWarsEnv
    from .model import OrbitNet
    from .ppo import ValueNorm
    from .train import (
        V2MixedScheduler,
        V2PFSPScheduler,
        V2SelfPlayOpponent,
        collect_rollout,
    )

    cfg = v2_config_from_dict(cfg_dict)
    device = torch.device("cpu")  # workers always on CPU

    worker_seed = cfg.seed + worker_id * 10000
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    model = OrbitNet(cfg.model).to(device)
    model.eval()

    rule_based_opponent = build_opponent(cfg.opponent)

    # Self-play opponent (for the non-PFSP MixedScheduler path).
    sp_opponent = V2SelfPlayOpponent(cfg, device=device,
                                     deterministic=cfg.self_play_deterministic)
    sp_opponent.sync_from(model)

    # PopArt value-norm (read-only in the worker; updated centrally).
    value_norm = ValueNorm(cfg.ppo.popart_beta) if cfg.ppo.popart else None

    # Scheduler: PFSP pool (synced from main) takes precedence, else MixedScheduler.
    scheduler = None
    is_pfsp = cfg.pfsp_enabled
    if is_pfsp:
        scheduler = V2PFSPScheduler(cfg, rule_based_opponent, device)
    elif cfg.four_player_prob > 0.0 or cfg.rule_based_prob_start < 1.0:
        scheduler = V2MixedScheduler(cfg, rule_based_opponent, sp_opponent)

    # Baseline PFSP (wins, games) per pool name at the last sync — used to compute
    # this collect's deltas to ship back to main.
    pfsp_baseline: dict[str, tuple[float, float]] = {}

    def _refresh_baseline() -> None:
        if scheduler is not None and is_pfsp:
            for e in scheduler.pool:
                pfsp_baseline[e["name"]] = (e["wins"], e["games"])

    # Each worker gets 1 env (fast sim when batched-env is enabled).
    env_cls = V2FastEnv if cfg.ppo.use_batched_env else V2OrbitWarsEnv
    envs = [env_cls(cfg, rule_based_opponent, env_index=worker_id)]
    next_seed = worker_seed

    features_per_env = []
    for env in envs:
        if scheduler is not None:
            num_p, opps = scheduler.sample_episode()
            features_per_env.append(env.reset(seed=next_seed, num_players=num_p, opponents=opps))
        else:
            features_per_env.append(env.reset(seed=next_seed))
        next_seed += 1

    conn.send("ready")

    def _np(t):
        return None if t is None else t.numpy()

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
                    scheduler=scheduler, value_norm=value_norm,
                )

                # PFSP win/game deltas since the last sync.
                pfsp_delta: dict[str, list[float]] = {}
                if is_pfsp and scheduler is not None:
                    for e in scheduler.pool:
                        bw, bg = pfsp_baseline.get(e["name"], (0.0, 0.0))
                        dw, dg = e["wins"] - bw, e["games"] - bg
                        if dg != 0.0 or dw != 0.0:
                            pfsp_delta[e["name"]] = [dw, dg]
                    _refresh_baseline()

                wt = WorkerTransitions(
                    planet_features=batch.planet_features.numpy(),
                    global_features=batch.global_features.numpy(),
                    planet_mask=batch.planet_mask.numpy(),
                    own_mask=batch.own_mask.numpy(),
                    reachability_mask=batch.reachability_mask.numpy(),
                    target_indices=batch.target_indices.numpy(),
                    frac_indices=batch.frac_indices.numpy(),
                    log_prob=batch.log_prob.numpy(),
                    returns=batch.returns.numpy(),
                    advantages=batch.advantages.numpy(),
                    values=batch.values.numpy(),
                    pair_features=_np(batch.pair_features),
                    shot_idx=_np(batch.shot_idx),
                    shot_src=_np(batch.shot_src),
                    shot_tgt=_np(batch.shot_tgt),
                    shot_label=_np(batch.shot_label),
                    pfsp_delta=pfsp_delta,
                    stats=stats,
                )
                conn.send(wt)

            elif cmd[0] == "sync":
                payload = cmd[1]
                cpu_state = {k: v.cpu() for k, v in payload["model"].items()}
                model.load_state_dict(cpu_state)
                model.eval()
                sp_opponent.sync_from(model)

                if value_norm is not None and payload.get("value_norm") is not None:
                    value_norm.load_state_dict(payload["value_norm"])

                if is_pfsp and scheduler is not None:
                    # Append any pool snapshots we don't yet have.
                    for name, sd in payload.get("new_snapshots", []):
                        snap = V2SelfPlayOpponent(cfg, device=device, deterministic=False)
                        snap.model.load_state_dict({k: v.cpu() for k, v in sd.items()})
                        snap.model.eval()
                        scheduler.pool.append(
                            {"name": name, "agent": snap, "wins": 0.0, "games": 0.0})
                    # Overwrite local win/game stats with authoritative global ones.
                    stats_by_name = payload.get("pfsp_stats", {})
                    for e in scheduler.pool:
                        if e["name"] in stats_by_name:
                            e["wins"], e["games"] = stats_by_name[e["name"]]
                    _refresh_baseline()

                conn.send("ok")

            elif cmd[0] == "poolinfo":
                # Diagnostic: report this worker's PFSP pool (names + local games).
                info = ([(e["name"], e["games"]) for e in scheduler.pool]
                        if scheduler is not None and is_pfsp else [])
                conn.send(info)

            elif cmd[0] == "shutdown":
                break
    except Exception:
        try:
            conn.send("error")
        except (BrokenPipeError, OSError):
            pass
    finally:
        conn.close()
        devnull.close()


class ParallelRolloutCollector:
    """Manages subprocess workers for parallel, v4-complete rollout collection."""

    def __init__(self, cfg: V2Config, num_workers: int) -> None:
        self.num_workers = num_workers
        self.cfg = cfg
        self.cfg_dict = v2_config_to_dict(cfg)
        # Each worker owns 1 env.
        self.cfg_dict["ppo"]["num_envs"] = 1
        # Track which PFSP snapshot names have already been broadcast (weights sent once).
        self._broadcast_snapshots: set[str] = set()

        self._conns: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []

        ctx = mp.get_context("spawn")
        for i in range(num_workers):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(target=_worker_fn, args=(child_conn, i, self.cfg_dict),
                               daemon=True)
            proc.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._procs.append(proc)

        for conn in self._conns:
            msg = conn.recv()
            if msg != "ready":
                raise RuntimeError(f"Worker failed to initialize: {msg}")

    def sync(self, model: torch.nn.Module, value_norm: object | None = None,
             scheduler: object | None = None) -> None:
        """Broadcast weights (+ PopArt stats + PFSP pool/stats) to all workers."""
        payload = {
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "value_norm": value_norm.state_dict() if value_norm is not None else None,
        }
        # PFSP: send authoritative win/game stats + any not-yet-broadcast snapshots.
        from .train import V2PFSPScheduler
        if isinstance(scheduler, V2PFSPScheduler):
            pfsp_stats: dict[str, tuple[float, float]] = {}
            new_snapshots: list[tuple[str, dict]] = []
            for e in scheduler.pool:
                pfsp_stats[e["name"]] = (e["wins"], e["games"])
                if e["name"] == "apex":
                    continue
                if e["name"] not in self._broadcast_snapshots:
                    sd = {k: v.cpu() for k, v in e["agent"].model.state_dict().items()}
                    new_snapshots.append((e["name"], sd))
                    self._broadcast_snapshots.add(e["name"])
            payload["pfsp_stats"] = pfsp_stats
            payload["new_snapshots"] = new_snapshots

        for conn in self._conns:
            conn.send(("sync", payload))
        for conn in self._conns:
            conn.recv()

    # Back-compat alias: old call site used sync_weights(model).
    def sync_weights(self, model: torch.nn.Module) -> None:
        self.sync(model)

    def collect(self, update: int) -> tuple[V2TransitionBatch, dict, dict]:
        """Collect rollouts from all workers in parallel.

        Returns (merged_batch, merged_stats, aggregated_pfsp_deltas).
        """
        for conn in self._conns:
            conn.send(("collect", update))
        results: list[WorkerTransitions] = []
        for conn in self._conns:
            r = conn.recv()
            if r == "error":
                raise RuntimeError("A rollout worker crashed during collect().")
            results.append(r)
        return self._merge_transitions(results)

    def debug_pools(self) -> list:
        """Diagnostic: return each worker's PFSP pool (names + local game counts)."""
        for conn in self._conns:
            conn.send(("poolinfo",))
        return [conn.recv() for conn in self._conns]

    def shutdown(self) -> None:
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
    ) -> tuple[V2TransitionBatch, dict, dict]:
        non_empty = [r for r in results if r.planet_features.shape[0] > 0]

        # Aggregate PFSP deltas across all workers (regardless of emptiness).
        pfsp_deltas: dict[str, list[float]] = {}
        for r in results:
            for name, (dw, dg) in r.pfsp_delta.items():
                acc = pfsp_deltas.setdefault(name, [0.0, 0.0])
                acc[0] += dw
                acc[1] += dg

        if not non_empty:
            from .train import _empty_batch
            return (_empty_batch(self.cfg),
                    {"episode_reward_mean": 0.0, "episodes_finished": 0.0, "samples": 0.0},
                    pfsp_deltas)

        def cat(attr):
            return torch.from_numpy(np.concatenate([getattr(r, attr) for r in non_empty]))

        # Shot labels reference rows WITHIN each worker's batch -> offset on merge.
        has_shot = all(r.shot_idx is not None for r in non_empty) and any(
            r.shot_idx is not None and r.shot_idx.shape[0] > 0 for r in non_empty)
        shot_idx = shot_src = shot_tgt = shot_label = None
        if has_shot:
            idx_parts, src_parts, tgt_parts, lab_parts = [], [], [], []
            offset = 0
            for r in non_empty:
                n = r.planet_features.shape[0]
                if r.shot_idx is not None and r.shot_idx.shape[0] > 0:
                    idx_parts.append(r.shot_idx + offset)
                    src_parts.append(r.shot_src)
                    tgt_parts.append(r.shot_tgt)
                    lab_parts.append(r.shot_label)
                offset += n
            if idx_parts:
                shot_idx = torch.from_numpy(np.concatenate(idx_parts)).long()
                shot_src = torch.from_numpy(np.concatenate(src_parts)).long()
                shot_tgt = torch.from_numpy(np.concatenate(tgt_parts)).long()
                shot_label = torch.from_numpy(np.concatenate(lab_parts)).float()

        has_pair = all(r.pair_features is not None for r in non_empty)
        pair_features = (
            torch.from_numpy(np.concatenate([r.pair_features for r in non_empty]))
            if has_pair else None)

        batch = V2TransitionBatch(
            planet_features=cat("planet_features").float(),
            global_features=cat("global_features").float(),
            planet_mask=cat("planet_mask").bool(),
            own_mask=cat("own_mask").bool(),
            reachability_mask=cat("reachability_mask").bool(),
            target_indices=cat("target_indices").long(),
            frac_indices=cat("frac_indices").long(),
            log_prob=cat("log_prob").float(),
            returns=cat("returns").float(),
            advantages=cat("advantages").float(),
            values=cat("values").float(),
            pair_features=pair_features,
            shot_idx=shot_idx,
            shot_src=shot_src,
            shot_tgt=shot_tgt,
            shot_label=shot_label,
        )

        all_rewards = [r.stats.get("episode_reward_mean", 0.0) for r in results
                       if r.stats.get("episodes_finished", 0) > 0]
        stats = {
            "episode_reward_mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "episodes_finished": float(sum(r.stats.get("episodes_finished", 0) for r in results)),
            "samples": float(sum(r.stats.get("samples", 0) for r in results)),
        }
        return batch, stats, pfsp_deltas
