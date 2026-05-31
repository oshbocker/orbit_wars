"""Expert Iteration (ExIt) for the V2 OrbitNet pipeline.

Instead of model-free PPO (whose value-scale and rollout-length pathologies
stalled progress vs apex), ExIt uses the *ground-truth* game simulator plus
per-planet lookahead search to produce improved (target, ship-fraction)
distributions, then distills them into OrbitNet by supervised learning. With a
perfect model available, planning is a far stronger and more stable source of
policy improvement than policy gradients.

Loop per iteration:
  1. COLLECT : play games with the current OrbitNet (sampled), record V2 features
               + sim states per step.
  2. SEARCH  : for every owned planet at every recorded step, forward-simulate
               candidate actions and softmax the scores -> improved targets.
  3. DISTILL : supervised cross-entropy (target + fraction) + value MSE to the
               game outcome.
  4. EVAL    : win-rate vs baselines.

Optionally BC-pretrains from apex first (imitation.enabled) for a warm start.
"""
from __future__ import annotations

import argparse
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.game_types import parse_observation
from src.logging import TrainLogger
from src.simulator import build_sim_state

from .actions import decode_sampled_actions, sample_actions
from .config import V2Config, load_v2_config
from .features import encode_features
from .model import OrbitNet
from .search import search_improve_planet
from .train import (
    make_v2_eval_agent,
    resolve_device,
    run_periodic_eval,
    save_checkpoint,
    seed_everything,
)


@dataclass
class StepRecord:
    features: Any           # V2Features
    sim_state: Any          # SimState
    game_state: Any         # GameState (for angular_velocity / step / intercept)
    player: int
    step: int
    outcome: float = 0.0    # filled after the game ends


@dataclass
class V2ExItSample:
    planet_features: np.ndarray
    global_features: np.ndarray
    planet_mask: np.ndarray
    own_mask: np.ndarray
    reachability_mask: np.ndarray
    target_probs: np.ndarray   # [P, P+1]
    frac_probs: np.ndarray     # [P, P, K]
    outcome: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="V2 Expert Iteration")
    ap.add_argument("--config", type=str, default="configs/v2_exit.yaml")
    ap.add_argument("--resume", type=str, default=None)
    return ap.parse_args()


def _extract_obs(entry):
    return entry.observation if hasattr(entry, "observation") else entry["observation"]


def _extract_status(entry) -> str:
    return entry.status if hasattr(entry, "status") else entry["status"]


def _comet_ids(obs) -> list[int] | None:
    ids = getattr(obs, "comet_planet_ids", None)
    if ids is None and isinstance(obs, dict):
        ids = obs.get("comet_planet_ids")
    return [int(x) for x in ids] if ids is not None else None


# ── Phase 1: collect games with the current policy ───────────────────────────

def collect_games(
    model: OrbitNet, cfg: V2Config, device: torch.device, n_games: int, seed: int,
) -> tuple[list[StepRecord], list[float], float]:
    from kaggle_environments import make

    from agents.apex import agent as apex_agent

    records: list[StepRecord] = []
    outcomes: list[float] = []
    sample = cfg.exit.sample_collect

    for gi in range(n_games):
        env = make("orbit_wars", configuration={"seed": seed + gi}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        done = False
        game_records: list[StepRecord] = []

        while not done:
            obs0 = _extract_obs(states[0])
            obs1 = _extract_obs(states[1])
            state = parse_observation(obs0)
            features = encode_features(state, cfg.env, comet_ids=_comet_ids(obs0))

            with torch.inference_mode():
                pf = torch.from_numpy(features.planet_features).unsqueeze(0).to(device)
                gf = torch.from_numpy(features.global_features).unsqueeze(0).to(device)
                pm = torch.from_numpy(features.planet_mask).unsqueeze(0).to(device)
                om = torch.from_numpy(features.own_mask).unsqueeze(0).to(device)
                rm = torch.from_numpy(features.reachability_mask).unsqueeze(0).to(device)
                output = model(pf, gf, pm, om, rm)
                sampled = sample_actions(output, om, deterministic=not sample)

            if features.own_mask.any():
                game_records.append(StepRecord(
                    features=features, sim_state=build_sim_state(state),
                    game_state=state, player=state.player, step=state.step,
                ))

            moves = decode_sampled_actions(sampled, output, features, state, cfg.env)
            opp_moves = apex_agent(obs1) or []
            states = env.step([moves, list(opp_moves)])
            done = _extract_status(states[0]) != "ACTIVE"

        reward = states[0]["reward"] if isinstance(states[0], dict) else states[0].reward
        outcome = max(-1.0, min(1.0, float(reward) if reward is not None else 0.0))
        outcomes.append(outcome)
        for r in game_records:
            r.outcome = outcome
            records.append(r)

    win_rate = sum(1 for o in outcomes if o > 0) / max(len(outcomes), 1)
    return records, outcomes, win_rate


# ── Phase 2: search improvement ──────────────────────────────────────────────

def _search_record(rec: StepRecord, env_cfg, exit_cfg) -> V2ExItSample:
    """Run per-planet search for one recorded step (picklable worker)."""
    feats = rec.features
    P = env_cfg.max_planets
    K = len(env_cfg.ship_fractions)
    target_probs = np.zeros((P, P + 1), dtype=np.float32)
    frac_probs = np.zeros((P, P, K), dtype=np.float32)

    for i in range(P):
        if not feats.own_mask[i]:
            continue
        tp, fp = search_improve_planet(
            state=rec.game_state, features=feats,
            sim_state=rec.sim_state, player=rec.player, source_slot=i,
            env_cfg=env_cfg, exit_cfg=exit_cfg,
        )
        target_probs[i] = tp
        frac_probs[i] = fp

    return V2ExItSample(
        planet_features=feats.planet_features, global_features=feats.global_features,
        planet_mask=feats.planet_mask, own_mask=feats.own_mask,
        reachability_mask=feats.reachability_mask,
        target_probs=target_probs, frac_probs=frac_probs,
        outcome=rec.outcome,
    )


def search_improve(records: list[StepRecord], cfg: V2Config) -> list[V2ExItSample]:
    """Search-improve all recorded decisions. Search is CPU-bound and
    embarrassingly parallel; with exit.search_workers>1 it fans out across
    processes (the main A100-box efficiency lever, since the GPU is idle here)."""
    workers = cfg.exit.search_workers
    if workers and workers > 1 and len(records) > 1:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        fn = partial(_search_record, env_cfg=cfg.env, exit_cfg=cfg.exit)
        try:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(fn, records, chunksize=4))
        except Exception as e:  # pragma: no cover - fall back to sequential
            print(f"  parallel search failed ({e}); falling back to sequential")
    return [_search_record(r, cfg.env, cfg.exit) for r in records]


# ── Phase 3: supervised distillation ─────────────────────────────────────────

def _build_batch(samples: list[V2ExItSample], idx: np.ndarray, device: torch.device):
    pick = [samples[i] for i in idx]
    t = lambda arr, dt: torch.from_numpy(np.array(arr, dtype=dt)).to(device)
    return {
        "pf": t([s.planet_features for s in pick], np.float32),
        "gf": t([s.global_features for s in pick], np.float32),
        "pm": t([s.planet_mask for s in pick], bool).bool(),
        "om": t([s.own_mask for s in pick], bool).bool(),
        "rm": t([s.reachability_mask for s in pick], bool).bool(),
        "tp": t([s.target_probs for s in pick], np.float32),
        "fp": t([s.frac_probs for s in pick], np.float32),
        "out": torch.tensor([s.outcome for s in pick], dtype=torch.float32, device=device),
    }


def train_epoch(model, optimizer, samples, cfg, device) -> dict[str, float]:
    N = len(samples)
    if N < 4:
        return {"loss": 0.0, "target_loss": 0.0, "frac_loss": 0.0, "value_loss": 0.0}
    bs = min(N, cfg.exit.train_batch_size)
    order = np.random.permutation(N)
    metrics = {"loss": 0.0, "target_loss": 0.0, "frac_loss": 0.0, "value_loss": 0.0}
    nb = 0
    for start in range(0, N, bs):
        idx = order[start:start + bs]
        if len(idx) < 4:
            continue
        b = _build_batch(samples, idx, device)
        out = model(b["pf"], b["gf"], b["pm"], b["om"], b["rm"])

        logp = F.log_softmax(out.logits.clamp(min=-1e4), dim=-1)       # [B,P,P+1]
        tgt_ce = -(b["tp"] * logp).sum(-1)                              # [B,P]
        own = b["om"].float()
        target_loss = (tgt_ce * own).sum() / own.sum().clamp(min=1)

        flogp = F.log_softmax(out.frac_logits, dim=-1)                 # [B,P,P,K]
        fce = -(b["fp"] * flogp).sum(-1)                               # [B,P,P]
        fmask = (b["rm"] & b["om"].unsqueeze(-1)).float()              # [B,P,P]
        frac_loss = (fce * fmask).sum() / fmask.sum().clamp(min=1)

        value_loss = F.mse_loss(out.value, b["out"])

        loss = target_loss + frac_loss + cfg.exit.value_loss_coef * value_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.exit.max_grad_norm)
        optimizer.step()

        metrics["loss"] += float(loss.detach())
        metrics["target_loss"] += float(target_loss.detach())
        metrics["frac_loss"] += float(frac_loss.detach())
        metrics["value_loss"] += float(value_loss.detach())
        nb += 1
    return {k: v / max(nb, 1) for k, v in metrics.items()}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    cfg = load_v2_config(args.config)
    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_dir = Path(cfg.log_dir) / cfg.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    _log_path = log_dir / "train.log"
    _log_state = {"f": open(_log_path, "a"), "n": 0}

    def log(msg: str) -> None:
        # Close-and-reopen periodically so the Google Drive FUSE mount (Colab)
        # actually syncs train.log — flush()/fsync() alone can leave it stale.
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
        if _log_state["n"] % 10 == 0:
            f.close()
            _log_state["f"] = open(_log_path, "a")

    model = OrbitNet(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"V2 ExIt: {cfg.run_name}, device={device}, params={n_params:,}")
    log(f"  iterations={cfg.exit.iterations}, games/iter={cfg.exit.games_per_iter}, "
        f"search_depth={cfg.exit.search_depth}, candidates={cfg.exit.search_candidates}")

    logger = TrainLogger(cfg.log_dir, cfg.run_name)
    save_dir = Path(cfg.save_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.exit.train_lr)

    # Optional BC warm start (clone apex first, then improve by search)
    if cfg.imitation.enabled and not args.resume:
        from .imitation import v2_bc_pretrain
        from .train import _load_or_collect_demos
        log(f"\n=== BC warm start ({cfg.imitation.bc_games} games, "
            f"{cfg.imitation.bc_epochs} epochs) ===")
        demos = _load_or_collect_demos(cfg, log)
        v2_bc_pretrain(model, demos, cfg.imitation, device, logger)
        save_checkpoint(save_dir, cfg.run_name, 0, model, optimizer)
        log("  BC checkpoint saved (iter=0)")
        if cfg.eval.eval_every > 0:
            log(f"  Eval of BC clone (iter 0, {cfg.eval.eval_games} games)...")
            for r in run_periodic_eval(model, cfg, device):
                logger.log_eval(0, [r])
                log(f"    vs {r.opponent_name}: W={r.win_rate:.0%} L={r.loss_rate:.0%} "
                    f"T={r.tie_rate:.0%} (n={r.n_games})")
    elif args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        log(f"  Resumed weights from {args.resume}")

    dataset: deque[list[V2ExItSample]] = deque(maxlen=cfg.exit.dataset_max_iters)
    next_seed = cfg.seed + 1000
    t_start = time.time()
    log(f"\n=== ExIt training ({cfg.exit.iterations} iterations) ===")

    for it in range(1, cfg.exit.iterations + 1):
        t_it = time.time()
        model.eval()
        records, outcomes, win_rate = collect_games(
            model, cfg, device, cfg.exit.games_per_iter, next_seed)
        next_seed += cfg.exit.games_per_iter
        t_collect = time.time() - t_it

        t_s = time.time()
        new_samples = search_improve(records, cfg)
        dataset.append(new_samples)
        all_samples = [s for batch in dataset for s in batch]
        t_search = time.time() - t_s

        model.train()
        m: dict[str, float] = {}
        t_tr = time.time()
        for _ in range(cfg.exit.train_epochs):
            m = train_epoch(model, optimizer, all_samples, cfg, device)
        t_train = time.time() - t_tr

        logger.log_update(it, {"win_rate": win_rate, "dataset_size": float(len(all_samples)),
                               "episode_reward_mean": float(np.mean(outcomes)) if outcomes else 0.0, **m})
        log(f"iter={it:4d}  selfwin_vs_apex={win_rate:.0%}  decisions={len(records)}  "
            f"dataset={len(all_samples)}  loss={m.get('loss', 0):.4f}  "
            f"tloss={m.get('target_loss', 0):.4f}  floss={m.get('frac_loss', 0):.4f}  "
            f"vloss={m.get('value_loss', 0):.4f}  collect={t_collect:.0f}s  "
            f"search={t_search:.0f}s  train={t_train:.0f}s")

        if cfg.eval.eval_every > 0 and it % cfg.eval.eval_every == 0:
            log(f"\n  Eval ({cfg.eval.eval_games} games)...")
            for r in run_periodic_eval(model, cfg, device):
                logger.log_eval(it, [r])
                log(f"    vs {r.opponent_name}: W={r.win_rate:.0%} L={r.loss_rate:.0%} "
                    f"T={r.tie_rate:.0%} (n={r.n_games})")
            log("")

        if it % cfg.checkpoint_every == 0 or it == cfg.exit.iterations:
            save_checkpoint(save_dir, cfg.run_name, it, model, optimizer)
            log(f"  -> checkpoint at iter {it}")

    log(f"\nExIt complete. Total time: {time.time() - t_start:.0f}s")
    logger.close()
    _log_state["f"].close()


if __name__ == "__main__":
    main()
