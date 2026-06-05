# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Kaggle Orbit Wars competition.
**Primary purpose: learn reinforcement learning and win competition.** The apex rule-based agent is the current benchmark; the main effort is one or more agents using modern RL (PPO, self-play, MARL). The agents should be capable of training in Kaggle (GPU), Google Colab (GPU), or locally (no GPU). The evaluation should be capable of running in Kaggle or locally (no GPU).

## Repository Structure

> **Live effort = the v2 ExIt pipeline (`v2/`).** `src/` is now a library of shared
> building blocks that `v2/` imports, **not** a standalone pipeline. The v1 transformer-PPO
> training half and the legacy SB3 pipeline were pruned 2026-06-05 — their ideas and the
> reasons we dropped them live in `rl_research/EXPLORED_AND_ABANDONED.md`.

```
orbit_wars/
├── v2/                      # OrbitNet + ExIt pipeline (PRIMARY effort) — see MEMORY.md for the file map
│   ├── model.py             # OrbitNet (~552K params): attention over planets + pairwise head
│   ├── search.py            # Per-planet lookahead search for ExIt (Phase-3 value blend lives here)
│   ├── exit_train.py        # Expert Iteration: collect games → search-improve → distill
│   ├── train.py             # PPO loop, MixedScheduler/PFSP, side-alternated periodic eval
│   ├── env.py / fast_env.py # Kaggle env wrapper / engine-faithful standalone sim (batchable)
│   ├── features.py state.py actions.py reward.py comet.py ppo.py imitation.py parallel.py
│   ├── agent.py             # Self-contained inlined Kaggle submission agent
│   └── agent_v3.py          # Submission agent that loads real encode/decode + bundled ckpt/config
├── src/                     # Shared building blocks reused by v2/ (NOT a standalone pipeline)
│   ├── game_types.py        # PlanetState, FleetState, GameState, parse_observation()
│   ├── features.py          # Feature encoding, fleet transit, SourceDecision
│   ├── policy.py            # TransformerPolicy (v1 arch; TransformerBlock reused by v2)
│   ├── ppo.py               # sample_actions (used by opponents); v1 ppo_update kept but inert
│   ├── opponents.py         # Apex, Random, SelfPlay, Hybrid, Distilled opponents + _policy_act()
│   ├── logging.py           # TrainLogger (TensorBoard + CSV), EvalResult, periodic eval
│   ├── simulator.py         # Lightweight positional forward sim (SimState, sim_step, evaluate_state)
│   └── config.py            # TrainConfig dataclasses
├── agents/                  # Rule-based agents (benchmarks)
│   ├── apex.py              # Apex rule-based agent (THE benchmark)
│   └── hybrid.py            # Mission-based + timeline agent
├── configs/                 # YAML configs — v2_exit*.yaml are live; v2/v3/v4 PPO configs kept for reference
├── evaluation/              # evaluate.py — run_games, head_to_head, print_results (used by v2 eval)
├── notebooks/               # train_colab.ipynb (A100 v2 BC→ExIt) + explore.ipynb (scratch)
├── rl_research/             # STRONGER_EXPERT_SEARCH_PLAN.md (live) + EXPLORED_AND_ABANDONED.md (graveyard)
├── scripts/                 # eval_fast.py, replay.py, download_checkpoint.py, run_embed_ab.py, tests
├── outputs/                 # .gitignored — checkpoints/, logs/, submissions/
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Common Commands

### v2 Expert Iteration (primary pipeline — `v2/`)

```bash
# ExIt: BC pretrain from apex → collect → search-improve → distill (the live effort)
uv run python -m v2.exit_train --config configs/v2_exit.yaml

# v2 PPO (BC warm start → PPO + mixed self-play) — reference baseline
uv run python -m v2.train --config configs/v2_default.yaml

# On Colab/Kaggle (no uv)
python -m v2.exit_train --config configs/v2_exit.yaml
```

### Download checkpoints from Google Drive

After training on Colab, download checkpoints locally via rclone.

**One-time rclone setup:**
```bash
# Install rclone
sudo apt install rclone        # Debian/Ubuntu
# or: brew install rclone      # macOS
# or: curl https://rclone.org/install.sh | sudo bash

# Configure Google Drive remote
rclone config
# → New remote → name: "gdrive" → type: "drive" →
# → leave client_id/secret blank → full access →
# → opens browser for Google OAuth login → done
```

**Download checkpoints:**
```bash
# Download latest checkpoint for the default run (v2_exit_a100)
uv run python scripts/download_checkpoint.py

# Download a specific run / all its checkpoints
uv run python scripts/download_checkpoint.py --run v2_exit_a100 --all-ckpts

# List available checkpoints on Drive
uv run python scripts/download_checkpoint.py --list
```

### Evaluate a trained checkpoint locally

```bash
# Fast, side-alternated, paired-seed scorer (the reliable one — high variance, use games>=60).
# Resolves outputs/checkpoints/<run>/ckpt_<iter>.pt. Local CPU ~20s/game → prefer Colab for high n.
uv run python scripts/eval_fast.py \
    --run v2_exit_a100 --config configs/v2_exit.yaml \
    --iters 20 --opponent apex --games 60
```

### Replay a game with HTML export

```bash
# ExIt checkpoint vs apex — exports game_replay.html
uv run python scripts/replay.py --exit \
    --checkpoint outputs/checkpoints/v2_exit_a100/ckpt_000020.pt \
    --config configs/v2_exit.yaml \
    --opponent apex --seed 42 --output replay_vs_apex.html
```

### Monitor training

```bash
uv run tensorboard --logdir outputs/logs
```

## Environment Setup

**Locally, always use `uv` to run Python scripts** (e.g. `uv run python -m v2.exit_train ...`). Do not use bare `python` or `python3`.

```bash
# Local (uv manages the virtualenv):
uv run python -m v2.exit_train --config configs/v2_exit.yaml

# On Colab/Kaggle (no uv):
pip install --upgrade "kaggle-environments>=1.28.0" torch gymnasium pyyaml tensorboard
```

### Google Colab Workflow

`notebooks/train_colab.ipynb` runs the v2 BC→ExIt pipeline on Colab GPU (A100/H100) with
Google Drive persistence. The `PIPELINE` switch in the notebook selects which v2 stack to run.
Cell order: Setup (mount Drive, clone, install) → GPU check → Config (loads a v2 config + GPU
overrides) → Train → Generate Submission (`v2/agent_v3.py` bundle + checkpoint to Drive) →
Evaluate → TensorBoard (last — can block downstream cells).

**After Colab training**, download the checkpoint from Drive with
`uv run python scripts/download_checkpoint.py` (requires rclone — see "Download checkpoints
from Google Drive") and evaluate locally with `scripts/eval_fast.py`.

## Game Overview

**Orbit Wars** (`orbit_wars` v1.0.9) is a Kaggle simulation: 2 or 4 players, 500 max steps, continuous 100×100 space. Players send fleets between planets to conquer territory.

### Board & Sun
- 100×100 space, center at (50, 50), origin at top-left
- **Sun** at center with radius 10 — any fleet crossing it is instantly destroyed
- Symmetric map: 20–40 planets (5-10 symmetric groups of 4) and comets in 4-fold mirror symmetry for fair starts. At least 3 groups are guaranteed to be static, and at least one group is guaranteed to be orbiting.

### Planets
- Each planet: `[id, owner, x, y, radius, ships, production]`
  - `owner`: player index (0–3) or `-1` for neutral
  - `radius = 1 + ln(production)`; production ranges 1–5 ships/turn ; higher production planets are physically larger
- **Orbiting (inner) planets**: rotate at `angular_velocity` (0.025–0.05 rad/turn); position = `initial_angle + angular_velocity × t`; `orbital_radius + planet_radius < 50`. Moving planets can sweep up stationary fleets.
- **Static (outer) planets**: fixed (don't rotate), further from center
- **Home planets**: randomly chosen, start with 10 ships; 2-player games have diagonally opposite starts; 4-player game, each player gets one planet from the group.

### Fleets
- Each fleet: `[id, owner, x, y, angle, from_planet_id, ships]`
  - angle: Direction of travel in radians
  - ships: Number of ships in the fleet (does not change during travel)
- Straight-line travel; collision detection is **continuous**
- **Speed formula**: `speed = 1.0 + (maxSpeed − 1.0) × (log(ships) / log(1000))^1.5`
  - 1 ship → 1.0 units/turn; large fleets → up to 6.0 units/turn
- Destroyed when hitting: sun, board edge, or a planet
- A fleet of ~500 ship moves at ~5, and ~1000 ships reach the max

#### Fleet Movement
Fleets travel in a straight line at their computed speed each turn. A fleet is removed if it:

    Goes out of bounds (leaves the 100x100 playing field).
    Crosses the sun (path segment comes within the sun's radius).
    Collides with any planet (path segment comes within the planet's radius). This triggers combat.

Collision detection is continuous -- the entire path segment from old to new position is checked, not just the endpoint.

#### Fleet Launch
Each turn, your agent returns a list of moves: [from_planet_id, direction_angle, num_ships].

    You can only launch from planets you own.
    You cannot launch more ships than the planet currently has.
    The fleet spawns just outside the planet's radius in the given direction.
    You can issue multiple launches from the same or different planets in a single turn.

### Comets
- Spawn at steps 50, 150, 250, 350, 450 — one per quadrant ; Radius 1.0 (fixed)
- Highly eccentric elliptical orbits; default `cometSpeed` 4.0 units/turn; produce 1 ship/turn when captured; `comet_planet_ids` in observation ; follow normal planet rules
- The comets observation field contains comet group data including paths (the full trajectory for each comet) and path_index (current position along the path), which can be used to predict future comet positions

### Combat (per planet, per turn)
When one or more fleets collide with a planet
1. Sum arriving ships per player
2. Top attacker vs. second-largest: survivor = `top − second` (tied → both 0)
3. Survivor reinforces if same owner; captures if different and ships > garrison

### Turn Sequence
1. Expire comets → 2. Spawn comets → 3. Launch fleets → 4. Production → 5. Move fleets → 6. Advance orbits → 7. Combat

### Scoring & Termination
- Game ends at step 498 or when ≤1 player has planets/fleets
- **Score** = ships on owned planets + ships in owned fleets
- Winner: `+1` reward; losers: `−1`; ties: all tied players `+1`

## Agent Interface

```python
def agent(obs, config=None) -> list:
    # returns [[from_planet_id, angle_radians, num_ships], ...]
```

**CRITICAL**: Kaggle resolves the agent by finding the **last callable** defined at module level in the submission file. The `agent()` function must be the last `def`/`class` in the file (excluding `if __name__ == "__main__"` blocks). Any helper classes (e.g., `_Planet`, `_Fleet`) must be defined **before** `agent()`.

**Observation fields** (`obs.field` or `obs["field"]` — both work):

**IMPORTANT**: `planets` and `fleets` entries may be **Struct objects** (attribute access like `p.owner`), **dicts**, or **lists** depending on the Kaggle environment version. Parsing must try attribute access first (`hasattr`), then dict access, then list unpacking. Never use `isinstance(p, (list, tuple))` as the first check — Kaggle's `Struct` objects are iterable (yielding keys, not values) and would cause incorrect unpacking.

| Field | Description |
|-------|-------------|
| `player` | Your player ID (0–3) |
| `planets` | `[id, owner, x, y, radius, ships, production]` or dict with same keys — owner = −1 neutral |
| `fleets` | `[id, owner, x, y, angle, from_planet_id, ships]` or dict with same keys |
| `angular_velocity` | Rotation speed of inner planets (rad/turn) |
| `initial_planets` | Snapshot of starting positions |
| `comets` / `comet_planet_ids` | Active comet state |
| `step` | Current step number |
| `remainingOverageTime` | Banked time budget (60s total); disqualified at 0 |

**Timing**: `actTimeout=1s` per step; `remainingOverageTime=60s` total banked overage.

## Shared building blocks (`src/`) — legacy v1 design

> This describes the **v1 transformer-PPO** design. The v1 *training* pipeline was pruned
> (see `rl_research/EXPLORED_AND_ABANDONED.md`); what remains in `src/` is a **library of
> shared modules** imported by the live v2 pipeline (`game_types`, `features`, `policy`,
> `ppo.sample_actions`, `opponents`, `logging`, `simulator`, `config`). The **live** model is
> v2 OrbitNet (simultaneous all-planet, one forward pass per step) — its design and file map
> live in `MEMORY.md` ("Key Files — V2 OrbitNet").

### Architecture (v1, sequential — retained for the reused encoders/policy)

Per-planet sequential decisions: for each turn, iterate over owned planets (most ships first). For each source planet, a transformer processes all valid targets and outputs a factored action (target selection + ship fraction).

**Feature encoding** (`src/features.py`):
- `SourceDecision` dataclass holds all inputs for one source planet decision
- Fleet transit computation: for each in-flight fleet, ray-circle intersection determines destination planet; aggregated as `(total_ships, weighted_avg_eta)` per planet for enemy and friendly separately
- Transit state is updated sequentially as decisions are made (later planets see earlier launches)

| Component | Raw features | After embedding |
|-----------|-------------|-----------------|
| Global state | 9 scalars (step, own_ships, enemy_ships×3, own_prod, enemy_prod×3) | embed_dim |
| Source planet | (x,y) + 7 scalars (radius, prod, ships, enemy_transit, enemy_eta, own_transit, own_eta) | embed_dim |
| KNN (×3) | (x,y) + 4 scalars (radius, prod, ships, orbiting) each, mean-pooled | embed_dim |
| Target planet (×T) | (x,y) + 11 scalars (neutral, friendly, enemy, dist, ships, prod, enemy_transit, enemy_eta, own_transit, own_eta, orbiting) | embed_dim |

**Token layout**: `[CLS, NoOp, Target₁, ..., Target_T, PAD...]`
- CLS: learnable parameter → value head
- NoOp: learnable parameter → "send nothing" option
- Targets: `TokenProjection(concat(global_emb, source_knn_emb, target_emb))`

**Policy** (`src/policy.py`, ~493K params default):
- Shared PositionEncoder MLP(2 → pos_hidden → embed_dim) for all (x,y) inputs
- SourceEncoder, KNNEncoder (mean-pooled), TargetEncoder MLPs
- 2-layer pre-LayerNorm transformer (4 heads, ff_dim=256)
- Output: target_logits [B, 1+T], fraction_logits [B, T, 5], value [B]

**Factored action** (`src/ppo.py`):
- Target: Categorical over NoOp + T targets (masked)
- Fraction: 5-way Categorical [0.2, 0.4, 0.6, 0.8, 1.0] per target (zeroed for NoOp)
- `log_prob = log_prob_target + log_prob_fraction`

**Reward**: sparse terminal ±1 by default. Three modes via `reward.reward_mode`:
- `sparse`: terminal ±1 only (default)
- `dense_absolute`: Δown_ships × coef + Δown_prod × prod_coef
- `dense_relative`: Δ(own_ships − best_enemy_ships) × ship_coef + Δ(own_prod − best_enemy_prod) × prod_coef × prod_mult — rewards gaining ship and production advantage
- **Early production bonus**: prod_mult = `1 + early_prod_bonus × max(0, 1 − step/early_prod_bonus_steps)`. Default 10× at step 0, decaying linearly to 1× at step 50. Encourages early planet capture.

### Logging (`src/logging.py`)

- `TrainLogger` writes all metrics to both TensorBoard (`outputs/logs/<run_name>/`) and CSV (`metrics.csv`)
- `log_update(update, metrics)` — per-update train metrics (loss, policy_loss, value_loss, entropy, reward, imitation_coef, etc.)
- `log_eval(update, results)` — periodic eval win rates as `eval/{opponent}_win_rate`
- `make_eval_agent(policy, cfg, device)` — creates a Kaggle-compatible `agent(obs, config)` callable from a policy (snapshots weights)
- `run_periodic_eval(policy, cfg, device)` — runs eval games against all opponents in `cfg.eval.eval_opponents`, returns `list[EvalResult]`

### Imitation / BC

BC-from-apex (DAgger-style: collect expert demos → supervised pretrain → PPO with a decaying
imitation anchor) lives in the **v2** pipeline (`v2/imitation.py`). The v1 `src/imitation.py`
was removed; see `rl_research/EXPLORED_AND_ABANDONED.md`.

### Opponents (`src/opponents.py`)

- `_policy_act(policy, obs, cfg, device, deterministic)` — shared helper for all policy-based opponents (~40 lines, sequential per-planet decisions with transit updates)
- `ApexOpponent` — wraps `agents.apex.agent`
- `KaggleRandomOpponent` — wraps Kaggle's built-in `random_agent`
- `HybridOpponent` — wraps `agents.hybrid.agent` (slow, 50-800ms/step)
- `SelfPlayOpponent` — maintains a separate `TransformerPolicy`; `sync_from()` copies weights from training policy
- `DistilledOpponent` — loads a BC-pretrained `.pt` checkpoint, fast inference (~1ms/step via `_policy_act()`)
- `build_opponent(name, cfg, device, checkpoint_path)` — factory for all opponent types

### Config

Key config sections: `env`, `model`, `ppo`, `reward`, `eval`, `imitation`.

The v1 transformer configs (`transformer_ppo/dagger/mixed.yaml`) were pruned — see
`rl_research/EXPLORED_AND_ABANDONED.md`. Live configs are the **v2** ones
(`configs/v2_exit*.yaml`); the `reward`/`imitation`/`eval` knobs above carry over to the v2
config sections almost verbatim. See `MEMORY.md` for the per-config breakdown.

### Checkpoint format

```
outputs/checkpoints/<run_name>/
    ckpt_last.pt       ← latest checkpoint
    ckpt_NNNNNN.pt     ← periodic saves (every checkpoint_every updates)
```

Each `.pt` file contains `{"update": int, "policy": state_dict, "optimizer": state_dict}`.

## Key Strategic Considerations

- **Travel time**: target garrison grows during flight; send `target.ships + production × travel_time + 1`
- **Sun avoidance**: never aim through center (50, 50); check with `_passes_through_sun()`
- **Fleet speed asymmetry**: tiny fleets are slow (1 unit/turn); send enough ships to get speed
- **Orbiting targets**: predict future planet position before aiming
- **Production priority**: high-production planets compound; capture them first
- **4-player dynamics**: let opponents fight, then strike the winner

## RL Development Roadmap

Milestones reached, in order (the abandoned branches and *why* are in
`rl_research/EXPLORED_AND_ABANDONED.md`):

1. **Apex / Hybrid** (done): rule-based benchmarks in `agents/`.
2. **v1 transformer PPO + DAgger + mixed self-play** (done, then superseded by v2).
3. **v2 OrbitNet** (done): simultaneous all-planet model, one forward pass/step.
4. **Model-free PPO** (incl. v4_ceiling): **confirmed dead end** — structural credit-assignment
   stall, 0–10% vs apex regardless of capacity/reward/opponent machinery.
5. **Expert Iteration (ExIt)** (current): search → distill is the proven path; best agent =
   `v2_exit_a100/ckpt_000020.pt`.

**Now:** STRONGER EXPERT SEARCH — a *learned value blended into the search* (not a hand-coded
opponent or rollout, both of which regressed the agent). See the next section and
`rl_research/STRONGER_EXPERT_SEARCH_PLAN.md`.

## Current best agent & next-session plan: STRONGER EXPERT SEARCH (ExIt)

**State (2026-06-05).** Best agent = **heuristic ExIt `v2_exit_a100/ckpt_000020.pt` (iter 20)** (submitted). ExIt (search → distill) is the proven path. **Eval-gate fix + Phase 2 DONE (2026-06-05).** ⚠️ **The "77% vs apex @ n=60" headline does NOT reproduce on the trusted side-alternated scorer:** at seed 20000 iter-20 is ~33% P0 / 13% P1 (≈23% combined), and win-rate is HIGH-VARIANCE across map seeds (33% on the 20000 batch vs 83% P0 on the 31000 batch). The two decode paths are identical (896 steps, 0 diffs → submitted agent == training policy, NOT a bug), so the spread is genuine map variance and the old non-alternated eval was inflating. **Re-measure the true baseline with high-n multi-seed `eval_fast` on Colab before trusting any number** (local CPU eval ~20s/game → too slow for n≥60).

**Dead ends — do NOT repeat (all empirically confirmed):**
- **Model-free PPO** (incl. the full v4_ceiling stack): structural credit-assignment stall, 0–10% vs apex / 100% vs random. More representation/critic/opponent machinery does not fix it.
- **Neural-value search leaves** (`exit.neural_value_leaves`, Tier 3.2): *collapsed* the 77% agent to 0%. Cause: leaf states are reconstructed fleet-less (positionless `SimState`) → out-of-distribution for the value head, and `evaluate_state` (heuristic) already counts in-flight `fleet_events`, so the value head is a *worse* leaf scorer. Parked off. (`model.value_only` speedup kept.)
- **Two-player one-ply search** (`exit.two_player_search`, committed): replays the opponent's actual apex turn-1 launch in the lookahead. Correct + cheap but it **DEGRADED** the warm-started 77% agent (`v2_exit_2p_a100`, n=60: 40% @ iter5, 43% @ iter10, never recovering to 77%). Likely mechanism: injecting the opponent makes aggressive candidates look worse → search biases toward passivity → more passive policy → worse vs apex (the exact failure mode). The sparse turn-1 opponent is the limitation. Flag defaults on in `V2ExItConfig`; **set `two_player_search: false` for the heuristic ExIt that produced the 77% agent.**
- **Every-step in-sim rollout opponent** (`exit.rollout_search`, Phase 1, committed): the "real fix" vs turn-1 — ALL players (incl. our own continuation) launch via a cheap geometry-light rollout policy at every `sim_step`. **Also CONFIRMED NEGATIVE (2026-06-05):** `v2_exit_rollout_a100`, n=60 across 60 iters = 37,33,37,40,45,57,33,43,58,50,48 → mean ~45%, peak 58%, regressed the 77% agent and never recovered. **Diagnosis:** our own post-candidate continuation is the *weak* rollout heuristic, so candidates needing strong follow-up are undervalued → conservative/passive distilled policy → loses to apex. **AutoGo lesson: never evaluate leaves with a weak rollout — use a learned value (what AlphaZero does).** Keep `rollout_search` default OFF; keep the code (`rollout_launches`/geometry reusable for Phase 3 data gen).

**Full AutoGo-informed write-up: `rl_research/STRONGER_EXPERT_SEARCH_PLAN.md`.**

**Process bug — ✅ FIXED (2026-06-05):** in-training `run_periodic_eval` was NOT side-alternated/paired (always RL=P0 via the Kaggle harness) → inflated ~2×. Now mirrors `eval_fast`: `FastOrbitWars`, side-alternated, paired seeds (`eval_seed=20000`, shared → directly comparable), parallel via `eval_workers`. New `V2EvalConfig` fields `eval_seed`/`eval_workers`; `configs/v2_exit.yaml` → `eval_games: 40`, `eval_workers: 6`.

**The lever = a LEARNED VALUE in the search (not a hand-coded opponent/rollout).** Sequence:
1. **Eval-gate fix** — ✅ DONE (see above).
2. **Phase 2 — positional simulator** — ✅ DONE. `fleet_events` now carry geometry `(…, launch_step, sx, sy, tx, ty)`; `fleet_position_at` + `_reconstruct_leaf_state` rebuild in-distribution fleets at leaves. Heuristic path bit-identical (combat/scoring read only the first 4 fields). Readiness diagnostic on iter-20: OOD collapse FIXED (neural leaf std 0.765, not degenerate), value grounded (corr(pred,outcome)=0.389); BUT neural≈heuristic sibling-ranking corr ≈ 0.0 → value too noisy to rank near-equal candidates alone.
3. **Phase 3 — grounded learned value: BLEND, NOT SWAP.** The value head is ALREADY grounded — `play_single_game` runs to terminal and `train_epoch` already does `value_loss = MSE(out.value, terminal_outcome)` (so "~10% to terminal" is effectively 100%). Remaining work = *use* it at leaves via a **z-scored blend** `score=(1-w)·z(heur)+w·z(neural)` behind a `value_leaf_blend` knob (small w, A/B-able), warm-start iter-20, gate on the fixed eval at high n. A pure swap is the trap (sibling ranking uncorrelated). Fixes both prior failures: OOD (Phase 2) + over-trusted value (blend).
4. (later) **Joint / multi-planet search** + capacity A/B once the value is trustworthy.

**Separate track — answers the original "does a deeper net help?" question:** run the **embed-128 vs 256 capacity A/B in the ExIt regime** (capacity should help with good supervised-distillation targets, unlike PPO). Configs exist: `configs/v2_exit_embed128.yaml` / `v2_exit_embed256.yaml` (now with parallel collection); run on Colab via `scripts/run_embed_ab.py`.

**Key files:** `v2/search.py` (`search_improve_planet`, `_make_dists`, neural-value helpers, `_reconstruct_leaf_state` now rebuilds in-distribution fleets — Phase 3 blend goes here), `v2/exit_train.py` (collect→search→distill; `_search_record`; `play_single_game` is P0-only vs apex, NOT side-alternated; parallel `collect_workers`/`search_workers`, `collect_fast_env`), `src/simulator.py` (`SimState`/`fleet_events` now POSITIONAL: `…, launch_step, sx, sy, tx, ty`; `fleet_position_at`; `add_fleet_event(…, src_xy, dst_xy)`; `sim_step` still has NO opponent — Phase 4(b)), `v2/train.py` (`run_periodic_eval` side-alternated/paired/parallel; `_peval_init`/`_peval_game`), `configs/v2_exit.yaml`.

**Eval/ops tooling:** `scripts/eval_fast.py` (fast_env, side-alternated, paired seeds — the reliable scorer; ⚠️ win-rate is HIGH-VARIANCE across map seeds, use n≥60 and multiple seed batches; local CPU ~20s/game so run on Colab), `scripts/download_checkpoint.py --run <name> --all-ckpts`, `scripts/replay.py --exit`. Submission bundle: `v2/agent_v3.py` + `ckpt_last.pt` + `submission_config.yaml` + `v2/` + `src/` (built at `outputs/submissions/v2_exit_a100_bundle/`); bundle code == live (verified same).

## Config System

Live configs are the **v2** ones: plain YAML with nested sections `env`, `model`, `ppo`,
`reward`, `eval`, `imitation`, plus the ExIt `exit` block. Loaded via
`v2.config.load_v2_config()`; dataclasses live in `v2/config.py` (`V2EnvConfig`,
`V2ModelConfig`, `V2PPOConfig`, `V2RewardConfig`, `V2EvalConfig`, `V2ImitationConfig`,
`V2ExItConfig`). Override fields programmatically (no CLI `--set`). `src/config.py` is retained
only as a shared dependency of the reused `src/` modules. See `MEMORY.md` for the full
per-config breakdown.
