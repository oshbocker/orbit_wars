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
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.game_types import FleetState, parse_observation
from src.logging import TrainLogger
from src.simulator import add_fleet_event, build_sim_state

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
    # Two-player search (Tier 3.2): the opponent's (apex) launches this turn,
    # mapped to SimState schedule entries (from_id, target_id, ships, travel_time).
    # Replayed in search so the lookahead evaluates moves against a real response
    # instead of a passive opponent. Empty unless exit.two_player_search.
    opp_events: list = field(default_factory=list)


def _map_opp_moves(opp_moves: list, game_state: Any) -> list:
    """Map opponent apex moves [(from_id, angle, ships)] to SimState schedule
    entries (from_id, target_id, ships, travel_time), using true geometry from
    the (full-information) game state to ray-cast each launch to its target."""
    from .state import predict_fleet_destination

    by_id = {p.id: p for p in game_state.planets}
    out: list = []
    for mv in opp_moves:
        try:
            from_id, angle, ships = int(mv[0]), float(mv[1]), int(mv[2])
        except (TypeError, ValueError, IndexError):
            continue
        fp = by_id.get(from_id)
        if fp is None or ships <= 0:
            continue
        sx = fp.x + (fp.radius + 0.1) * math.cos(angle)
        sy = fp.y + (fp.radius + 0.1) * math.sin(angle)
        vf = FleetState(id=-1, owner=fp.owner, x=sx, y=sy, angle=angle,
                        from_planet_id=from_id, ships=ships)
        tgt, eta = predict_fleet_destination(
            vf, game_state.planets, game_state.step, game_state.angular_velocity)
        if tgt is not None and math.isfinite(eta):
            out.append((from_id, int(tgt.id), ships, float(eta)))
    return out


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

def _policy_decide(model, cfg, device, obs0, game_records: list[StepRecord]):
    """Run the policy on obs0, append a StepRecord, return player-0 moves."""
    state = parse_observation(obs0)
    features = encode_features(state, cfg.env, comet_ids=_comet_ids(obs0))
    with torch.inference_mode():
        pf = torch.from_numpy(features.planet_features).unsqueeze(0).to(device)
        gf = torch.from_numpy(features.global_features).unsqueeze(0).to(device)
        pm = torch.from_numpy(features.planet_mask).unsqueeze(0).to(device)
        om = torch.from_numpy(features.own_mask).unsqueeze(0).to(device)
        rm = torch.from_numpy(features.reachability_mask).unsqueeze(0).to(device)
        output = model(pf, gf, pm, om, rm)
        sampled = sample_actions(output, om, deterministic=not cfg.exit.sample_collect)
    if features.own_mask.any():
        game_records.append(StepRecord(
            features=features, sim_state=build_sim_state(state),
            game_state=state, player=state.player, step=state.step,
        ))
    return decode_sampled_actions(sampled, output, features, state, cfg.env)


def play_single_game(
    model: OrbitNet, cfg: V2Config, device: torch.device, seed: int,
) -> tuple[list[StepRecord], float]:
    """Play ONE game of the policy (player 0) vs apex (player 1). Returns
    (records, outcome). Backend = fast_env (cfg.exit.collect_fast_env) or Kaggle."""
    from agents.apex import agent as apex_agent

    game_records: list[StepRecord] = []
    two_player = cfg.exit.two_player_search

    def _record_opp(n_before: int, opp_moves: list) -> None:
        # If _policy_decide appended a record this turn, attach the opponent's
        # mapped launches so search can replay them (two-player one-ply).
        if two_player and len(game_records) > n_before and opp_moves:
            game_records[-1].opp_events = _map_opp_moves(opp_moves, game_records[-1].game_state)

    if cfg.exit.collect_fast_env:
        from .fast_env import FastOrbitWars
        sim = FastOrbitWars(num_agents=2, seed=seed)
        while not sim.done:
            obs0, obs1 = sim.observation(0), sim.observation(1)
            n0 = len(game_records)
            moves = _policy_decide(model, cfg, device, obs0, game_records)
            opp_moves = apex_agent(obs1) or []
            _record_opp(n0, opp_moves)
            sim.step([moves, list(opp_moves)])
        reward = sim.rewards[0]
    else:
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.reset(num_agents=2)
        states = env.step([[], []])
        done = False
        while not done:
            obs0 = _extract_obs(states[0])
            obs1 = _extract_obs(states[1])
            n0 = len(game_records)
            moves = _policy_decide(model, cfg, device, obs0, game_records)
            opp_moves = apex_agent(obs1) or []
            _record_opp(n0, opp_moves)
            states = env.step([moves, list(opp_moves)])
            done = _extract_status(states[0]) != "ACTIVE"
        reward = states[0]["reward"] if isinstance(states[0], dict) else states[0].reward

    outcome = max(-1.0, min(1.0, float(reward) if reward is not None else 0.0))
    for r in game_records:
        r.outcome = outcome
    return game_records, outcome


# ── Parallel collection workers (collection is the ExIt bottleneck) ──────────
_CW: dict = {}


def _collect_init(cfg_dict: dict, state_dict: dict) -> None:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from .config import v2_config_from_dict
    cfg = v2_config_from_dict(cfg_dict)
    model = OrbitNet(cfg.model)
    model.load_state_dict(state_dict)
    model.eval()
    _CW["model"] = model
    _CW["cfg"] = cfg
    _CW["device"] = torch.device("cpu")


def _collect_worker(seed: int) -> tuple[list[StepRecord], float]:
    return play_single_game(_CW["model"], _CW["cfg"], _CW["device"], seed)


def collect_games(
    model: OrbitNet, cfg: V2Config, device: torch.device, n_games: int, seed: int,
) -> tuple[list[StepRecord], list[float], float]:
    """Play n_games of the current policy vs apex, recording per-decision
    StepRecords for search improvement.

    With cfg.exit.collect_workers>1 the (independent) games are played in parallel
    across processes — collection is the ExIt wall-clock bottleneck, so this is
    a near-linear speedup. Falls back to sequential on any pool error.
    """
    records: list[StepRecord] = []
    outcomes: list[float] = []
    seeds = [seed + gi for gi in range(n_games)]

    workers = cfg.exit.collect_workers
    if workers and workers > 1 and n_games > 1:
        from concurrent.futures import ProcessPoolExecutor

        from .config import v2_config_to_dict
        sd = {k: v.cpu() for k, v in model.state_dict().items()}
        try:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_collect_init,
                initargs=(v2_config_to_dict(cfg), sd),
            ) as ex:
                for game_records, outcome in ex.map(_collect_worker, seeds):
                    records.extend(game_records)
                    outcomes.append(outcome)
            win_rate = sum(1 for o in outcomes if o > 0) / max(len(outcomes), 1)
            return records, outcomes, win_rate
        except Exception as e:  # pragma: no cover - fall back to sequential
            print(f"  parallel collection failed ({e}); falling back to sequential")
            records, outcomes = [], []

    for s in seeds:
        game_records, outcome = play_single_game(model, cfg, device, s)
        records.extend(game_records)
        outcomes.append(outcome)

    win_rate = sum(1 for o in outcomes if o > 0) / max(len(outcomes), 1)
    return records, outcomes, win_rate


# ── Phase 2: search improvement ──────────────────────────────────────────────

def _search_record(rec: StepRecord, env_cfg, exit_cfg, value_model=None) -> V2ExItSample:
    """Run per-planet search for one recorded step (picklable worker)."""
    feats = rec.features
    P = env_cfg.max_planets
    K = len(env_cfg.ship_fractions)
    target_probs = np.zeros((P, P + 1), dtype=np.float32)
    frac_probs = np.zeros((P, P, K), dtype=np.float32)

    # Two-player search: inject the opponent's turn-1 response (its actual apex
    # launches this step) into the base state, so EVERY candidate (and hold) is
    # evaluated against a real opponent instead of a passive one.
    base_sim = rec.sim_state
    if getattr(exit_cfg, "two_player_search", False) and getattr(rec, "opp_events", None):
        base_sim = rec.sim_state.copy()
        for from_id, tgt_id, ships, tt in rec.opp_events:
            if from_id in base_sim.planet_owner and tgt_id in base_sim.planet_owner:
                add_fleet_event(base_sim, from_id, tgt_id, ships, tt)

    for i in range(P):
        if not feats.own_mask[i]:
            continue
        tp, fp = search_improve_planet(
            state=rec.game_state, features=feats,
            sim_state=base_sim, player=rec.player, source_slot=i,
            env_cfg=env_cfg, exit_cfg=exit_cfg, value_model=value_model,
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


# ── Parallel search workers (neural-value leaves need the model in-worker) ───
_SW: dict = {}


def _search_init(cfg_dict: dict, state_dict: dict | None) -> None:
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    from .config import v2_config_from_dict
    cfg = v2_config_from_dict(cfg_dict)
    _SW["env"] = cfg.env
    _SW["exit"] = cfg.exit
    if state_dict is not None:
        model = OrbitNet(cfg.model)
        model.load_state_dict(state_dict)
        model.eval()
        _SW["model"] = model
    else:
        _SW["model"] = None


def _search_worker(rec: StepRecord) -> V2ExItSample:
    return _search_record(rec, _SW["env"], _SW["exit"], _SW.get("model"))


def search_improve(
    records: list[StepRecord], cfg: V2Config, model: OrbitNet | None = None,
) -> list[V2ExItSample]:
    """Search-improve all recorded decisions. Search is CPU-bound and
    embarrassingly parallel; with exit.search_workers>1 it fans out across
    processes (the main A100-box efficiency lever, since the GPU is idle here).

    When exit.neural_value_leaves is set, `model` is supplied so leaves are scored
    by OrbitNet's value head (Tier 3.2). In parallel mode the model's weights are
    broadcast to workers via the pool initializer; workers score leaves on CPU.
    """
    use_neural = getattr(cfg.exit, "neural_value_leaves", False) and model is not None
    workers = cfg.exit.search_workers

    if workers and workers > 1 and len(records) > 1:
        from concurrent.futures import ProcessPoolExecutor

        from .config import v2_config_to_dict
        sd = {k: v.cpu() for k, v in model.state_dict().items()} if use_neural else None
        try:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_search_init,
                initargs=(v2_config_to_dict(cfg), sd),
            ) as ex:
                return list(ex.map(_search_worker, records, chunksize=4))
        except Exception as e:  # pragma: no cover - fall back to sequential
            print(f"  parallel search failed ({e}); falling back to sequential")

    vm = model if use_neural else None
    return [_search_record(r, cfg.env, cfg.exit, vm) for r in records]


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
        new_samples = search_improve(records, cfg, model)
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
