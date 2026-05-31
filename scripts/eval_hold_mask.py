"""Hold-mask diagnostic: does forcing the agent to act fix its passivity?

Hypothesis (see rl_research / passivity analysis): PPO trains with the hold
action masked, so the hold logit is set almost entirely by BC (which cloned
apex's ~86% hold rate). At eval, decode_actions argmaxes over [hold, targets]
with hold allowed, so the BC-inflated hold logit wins and the agent is passive.

This script evaluates the SAME checkpoint vs apex two ways:
  * hold_allowed  — the normal eval path (deterministic argmax incl. hold)
  * force_act     — hold masked at eval (argmax over targets only)
and reports win-rate + average launches/step for each. A large jump from
hold_allowed -> force_act confirms the train/eval hold mismatch is the cause.

Usage:
    uv run python scripts/eval_hold_mask.py \
        --checkpoint outputs/checkpoints/v2_ppo_a100/ckpt_000750.pt \
        --config configs/v2_exit.yaml --games 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from kaggle_environments import make

from agents.apex import agent as apex_agent
from src.game_types import parse_observation
from v2.config import load_v2_config
from v2.actions import decode_actions
from v2.features import encode_features
from v2.model import OrbitNet


def make_agent(model, cfg, device, force_act: bool):
    """Build a Kaggle agent; tracks launches/step via a mutable counter."""
    stats = {"launches": 0, "steps": 0}

    def agent(obs, config=None):
        state = parse_observation(obs)
        comet_ids = None
        ids = getattr(obs, "comet_planet_ids", None)
        if ids is None and isinstance(obs, dict):
            ids = obs.get("comet_planet_ids")
        if ids is not None:
            comet_ids = [int(x) for x in ids]
        feats = encode_features(state, cfg.env, comet_ids=comet_ids)
        with torch.inference_mode():
            pf = torch.from_numpy(feats.planet_features).unsqueeze(0).to(device)
            gf = torch.from_numpy(feats.global_features).unsqueeze(0).to(device)
            pm = torch.from_numpy(feats.planet_mask).unsqueeze(0).to(device)
            om = torch.from_numpy(feats.own_mask).unsqueeze(0).to(device)
            rm = torch.from_numpy(feats.reachability_mask).unsqueeze(0).to(device)
            out = model(pf, gf, pm, om, rm)
        moves = decode_actions(out, feats, state, cfg.env,
                               deterministic=True, force_act=force_act)
        stats["launches"] += len(moves)
        stats["steps"] += 1
        return moves

    return agent, stats


def evaluate(model, cfg, device, force_act: bool, games: int) -> dict:
    wins = losses = ties = 0
    agent, stats = make_agent(model, cfg, device, force_act)
    for g in range(games):
        rl_p0 = (g % 2 == 0)
        env = make("orbit_wars", configuration={"seed": 7000 + g}, debug=False)
        pair = [agent, apex_agent] if rl_p0 else [apex_agent, agent]
        env.run(pair)
        idx = 0 if rl_p0 else 1
        final = env.steps[-1]
        r = final[idx]["reward"] if isinstance(final[idx], dict) else final[idx].reward
        if r is None or r == 0:
            ties += 1
        elif r > 0:
            wins += 1
        else:
            losses += 1
    return {
        "wins": wins, "losses": losses, "ties": ties, "games": games,
        "win_rate": wins / games,
        "launches_per_step": stats["launches"] / max(stats["steps"], 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="configs/v2_exit.yaml")
    ap.add_argument("--games", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cpu")
    cfg = load_v2_config(args.config)
    model = OrbitNet(cfg.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded {args.checkpoint} (update {ckpt.get('update', '?')})\n")

    print(f"=== hold ALLOWED (current eval path), {args.games} games vs apex ===")
    a = evaluate(model, cfg, device, force_act=False, games=args.games)
    print(f"  W={a['win_rate']:.0%}  ({a['wins']}/{a['losses']}/{a['ties']} W/L/T)  "
          f"launches/step={a['launches_per_step']:.2f}\n")

    print(f"=== hold MASKED (force act), {args.games} games vs apex ===")
    b = evaluate(model, cfg, device, force_act=True, games=args.games)
    print(f"  W={b['win_rate']:.0%}  ({b['wins']}/{b['losses']}/{b['ties']} W/L/T)  "
          f"launches/step={b['launches_per_step']:.2f}\n")

    delta = b["win_rate"] - a["win_rate"]
    print(f"=== VERDICT ===")
    print(f"  win-rate change (force_act - hold_allowed): {delta:+.0%}")
    print(f"  launches/step: {a['launches_per_step']:.2f} -> {b['launches_per_step']:.2f}")
    if delta >= 0.10:
        print("  -> Forcing action substantially helps. The train/eval HOLD MISMATCH")
        print("     is a primary cause of passivity. Fix: make hold a learnable action")
        print("     (unmask in both train and eval).")
    elif delta <= -0.05:
        print("  -> Forcing action hurts: holding was often correct. Passivity is more")
        print("     about WHICH targets / reward than the hold mechanism.")
    else:
        print("  -> Little change. Hold mechanism is not the dominant factor; look to")
        print("     reward shaping (PBRS) and opponent pool (PFSP).")


if __name__ == "__main__":
    main()
