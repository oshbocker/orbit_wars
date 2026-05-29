"""Discriminating eval: score margin vs apex for each recommendation checkpoint.

The binary win-rate is saturated (all variants 0% vs apex, 100% vs random), so we
need a finer metric. For each exp2_* final checkpoint we play K games vs apex with
side alternation and record:
  - win/loss/tie (sanity)
  - mean score margin = own_final_score - apex_final_score  (less negative = stronger)
  - mean own planets held at game end
  - mean game length (longer survival vs apex = stronger)

Score = ships on owned planets + ships in owned fleets (the game's own scoring rule).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from kaggle_environments import make

from agents.apex import agent as apex_agent
from src.game_types import parse_observation
from v2.config import load_v2_config
from v2.model import OrbitNet
from v2.train import make_v2_eval_agent

REPO_ROOT = Path(__file__).resolve().parent.parent
EXP_DIR = REPO_ROOT / "experiments"

EXPERIMENTS = ["baseline", "r1_ent_anneal", "r2_high_gamma",
               "r3_strong_value", "r4_fast_selfplay", "r5_no_prod_bonus"]


def _scores(final_obs) -> dict[int, float]:
    """Per-player score (ships on owned planets + owned fleets) from a final obs."""
    state = parse_observation(final_obs)
    s: dict[int, float] = {}
    for p in state.planets:
        if p.owner >= 0:
            s[p.owner] = s.get(p.owner, 0.0) + p.ships
    for f in state.fleets:
        if f.owner >= 0:
            s[f.owner] = s.get(f.owner, 0.0) + f.ships
    return s


def _own_planets(final_obs, player: int) -> int:
    state = parse_observation(final_obs)
    return sum(1 for p in state.planets if p.owner == player)


def build_agent(name: str, device: torch.device):
    cfg = load_v2_config(EXP_DIR / "configs" / f"exp2_{name}.yaml")
    ckpt_path = EXP_DIR / "checkpoints" / f"exp2_{name}" / "ckpt_last.pt"
    model = OrbitNet(cfg.model).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return make_v2_eval_agent(model, cfg, device)


def eval_one(name: str, k: int, device: torch.device) -> dict:
    agent = build_agent(name, device)
    margins, lengths, planets = [], [], []
    wins = losses = ties = 0
    for g in range(k):
        rl_is_p0 = (g % 2 == 0)  # alternate sides
        env = make("orbit_wars", debug=False)
        pair = [agent, apex_agent] if rl_is_p0 else [apex_agent, agent]
        env.run(pair)
        rl_idx = 0 if rl_is_p0 else 1
        final = env.steps[-1]
        final_obs = final[0]["observation"] if isinstance(final[0], dict) else final[0].observation
        sc = _scores(final_obs)
        rl_score = sc.get(rl_idx, 0.0)
        opp_score = sc.get(1 - rl_idx, 0.0)
        margins.append(rl_score - opp_score)
        lengths.append(len(env.steps))
        planets.append(_own_planets(final_obs, rl_idx))
        # terminal reward for the RL player
        r = final[rl_idx]["reward"] if isinstance(final[rl_idx], dict) else final[rl_idx].reward
        if r is None or r == 0:
            ties += 1
        elif r > 0:
            wins += 1
        else:
            losses += 1
    n = max(1, k)
    return {
        "name": name, "games": k, "wins": wins, "losses": losses, "ties": ties,
        "mean_margin": sum(margins) / n,
        "mean_length": sum(lengths) / n,
        "mean_own_planets": sum(planets) / n,
        "margins": margins,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--experiments", nargs="*", default=None)
    args = ap.parse_args()
    device = torch.device("cpu")
    exps = args.experiments or EXPERIMENTS

    results = []
    for name in exps:
        print(f"=== {name} vs apex ({args.games} games, alternating sides) ===", flush=True)
        r = eval_one(name, args.games, device)
        print(f"  W/L/T={r['wins']}/{r['losses']}/{r['ties']}  "
              f"mean_margin={r['mean_margin']:+.1f}  mean_len={r['mean_length']:.0f}  "
              f"mean_planets={r['mean_own_planets']:.1f}", flush=True)
        results.append(r)

    out = EXP_DIR / "margin_vs_apex.json"
    out.write_text(json.dumps(results, indent=2))

    # Markdown table, ranked by mean_margin (descending = strongest first)
    ranked = sorted(results, key=lambda r: r["mean_margin"], reverse=True)
    lines = ["| Rank | Experiment | W/L/T vs apex | Mean score margin | Mean survival (steps) | Mean planets held |",
             "|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        lines.append(f"| {i} | {r['name']} | {r['wins']}/{r['losses']}/{r['ties']} | "
                     f"{r['mean_margin']:+.1f} | {r['mean_length']:.0f} | {r['mean_own_planets']:.1f} |")
    table = "\n".join(lines)
    (EXP_DIR / "margin_table.md").write_text(table)
    print("\n" + table)
    print(f"\nSaved {out} and {EXP_DIR/'margin_table.md'}")


if __name__ == "__main__":
    main()
