# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Kaggle Orbit Wars competition.
**Primary purpose: learn reinforcement learning and win competition.** The gate metric is
the local arena (`scripts/arena.py`) vs the vendored public agents (`agents/external/`;
producer = the 1287-tier flow-diff planner). The shipped competition agent is our producer
fork `agents/v5/`; the RL effort (v2 ExIt) re-anchors its teachers/opponents to that tier.
Training runs on Kaggle (GPU), Google Colab (GPU), or locally (no GPU); evaluation runs on
Kaggle or locally (no GPU). The apex/hybrid rule-based agents were retired 2026-06-11 —
design + post-mortem in `rl_research/EXPLORED_AND_ABANDONED.md` Cluster 5.

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
│   ├── opponents.py         # Random/SelfPlay/Distilled/ChampionPool + NamedAgentOpponent fallthrough
│   ├── logging.py           # TrainLogger (TensorBoard + CSV), EvalResult, periodic eval
│   ├── simulator.py         # Lightweight positional forward sim (SimState, sim_step, evaluate_state)
│   └── config.py            # TrainConfig dataclasses
├── agents/                  # __init__.load_named_agent(name) = central agent resolver (+ fast_env obs shim)
│   ├── external/            # Vendored public agents (producer 1287, tamrazov_1224, distance_1100, ...)
│   └── v5/                  # Our producer fork (SHIPPED): endgame horizon clamp + 4P nearest-opponent
├── configs/                 # v2_exit.yaml, v2_exit_embed256.yaml, v2_exit_gumbel.yaml (all live, producer-anchored)
├── evaluation/              # evaluate.py — run_games, head_to_head, print_results (used by v2 eval)
├── notebooks/               # train_colab.ipynb (A100 v2 BC→ExIt) + explore.ipynb (scratch)
├── rl_research/             # LEADERBOARD_CLIMB_PLAN.md (LIVE) + EXPLORED_AND_ABANDONED.md (graveyard)
├── scripts/                 # arena.py (THE gate), eval_fast.py, replay.py, build_v5_bundle.py, download_checkpoint.py, tests
├── outputs/                 # .gitignored — checkpoints/, logs/, submissions/
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Common Commands

### Arena (THE gate metric — real Kaggle env, side-alternated paired seeds)

```bash
# 2P round-robin matrix (resumable CSV — bump --games to extend)
uv run python scripts/arena.py \
    --agents v5,producer,ow_proto,enders_1000 --games 30 --workers 6

# 4P FFA (every 4-agent combo, seat-rotated, rank by reward + final board score)
uv run python scripts/arena.py --players 4 \
    --agents v5,producer,enders_1000,tamrazov_1224 --games 4

# v5 config A/B without code edits ("v5:key=val+key=val")
uv run python scripts/arena.py --agents "v5:roi_threshold=1.2,producer" --games 60
```

⚠️ Measurement rules (hard-won): the A/A noise floor of n=60 mirror games is ~±6%
(44% measured on IDENTICAL agents) — never act on n<100 mirror results; every
small-n outlier in this project has regressed at high n. The public pool ceilings
at ~99% for producer-level agents, so mirror A/Bs vs producer/v5 are the only
sensitive instrument for improvements to the base.

### v2 Expert Iteration (the RL pipeline — `v2/`)

```bash
# ExIt: BC pretrain from the expert (producer) → collect → search-improve → distill
uv run python -m v2.exit_train --config configs/v2_exit.yaml

# On Colab/Kaggle (no uv)
python -m v2.exit_train --config configs/v2_exit.yaml
```

### Build + verify the v5 submission bundle

```bash
# main.py + orbit_lite_v5/ at archive root; verifies via Kaggle's file loader
uv run python scripts/build_v5_bundle.py
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
# Fast, side-alternated, paired-seed scorer on fast_env (use games>=60; --opponent =
# any agents.load_named_agent name). Local CPU is slow → prefer Colab for high n.
uv run python scripts/eval_fast.py \
    --run v2_exit_embed256 --config configs/v2_exit_embed256.yaml \
    --iters last --opponent producer --games 60
```

### Replay a game with HTML export

```bash
# ExIt checkpoint vs producer — exports game_replay.html
uv run python scripts/replay.py --exit \
    --checkpoint outputs/checkpoints/v2_exit_a100/ckpt_000020.pt \
    --config configs/v2_exit.yaml \
    --opponent producer --seed 42 --output replay_vs_producer.html
```

### Monitor training

```bash
uv run tensorboard --logdir outputs/logs
```

### Lint & type-check

Ruff (lint + format) and Pyright (`basic` mode) are configured in `pyproject.toml`.

```bash
uv run ruff check .            # lint
uv run ruff check --fix .      # lint + apply safe fixes
uv run ruff format .           # format (black-compatible)
uv run pyright                 # type-check (src, v2, agents, evaluation, scripts)
```

Pyright `basic` is intentional — `torch`/`numpy`/`kaggle-environments` ship partial stubs, so
strict mode is mostly noise here; `basic` still catches None-derefs and wrong-arg bugs. There is
a known baseline of lint findings (style-only) and Pyright errors to chip away at incrementally —
fix-as-you-touch, don't bulk-rewrite research code. `v2/agent_v3.py` is E402-exempt (it mutates
`sys.path` before importing for Kaggle packaging).

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

BC-from-expert (DAgger-style: collect expert demos → supervised pretrain → train with a
decaying imitation anchor) lives in the **v2** pipeline (`v2/imitation.py`); the expert is
`imitation.bc_expert` (default `producer`, resolved via `agents.load_named_agent`). The v1
`src/imitation.py` was removed; see `rl_research/EXPLORED_AND_ABANDONED.md`.

### Opponents (`src/opponents.py`)

- `_policy_act(policy, obs, cfg, device, deterministic)` — shared helper for all policy-based opponents (~40 lines, sequential per-planet decisions with transit updates)
- `KaggleRandomOpponent` — wraps Kaggle's built-in `random_agent`
- `SelfPlayOpponent` — maintains a separate `TransformerPolicy`; `sync_from()` copies weights from training policy
- `DistilledOpponent` — loads a BC-pretrained `.pt` checkpoint, fast inference (~1ms/step via `_policy_act()`)
- `NamedAgentOpponent` — `build_opponent()` fallthrough: any `agents.load_named_agent`
  name (v5, producer, tamrazov_1224, ...) becomes an opponent
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

1. **Apex / Hybrid rule-based benchmarks** (done, RETIRED 2026-06-11 — Cluster 5 in the
   graveyard; they anchored everything at ≈ LB p55).
2. **v1 transformer PPO + DAgger + mixed self-play** (done, then superseded by v2).
3. **v2 OrbitNet** (done): simultaneous all-planet model, one forward pass/step.
4. **Model-free PPO** (incl. v4_ceiling): **confirmed dead end** — structural
   credit-assignment stall regardless of capacity/reward/opponent machinery.
5. **Expert Iteration (ExIt)** (the RL pipeline): search → distill works; Gumbel
   selection (`exit.gumbel_search`) is the one validated search improvement (+9.4%).
6. **Leaderboard re-anchor (2026-06-10, current)**: vendored public agents + arena
   replaced win-vs-apex as THE metric; producer fork `agents/v5/` shipped
   (LB ~1242, rank ~140/4212). RL track re-anchors to the producer tier.

**Now: `rl_research/LEADERBOARD_CLIMB_PLAN.md` is the live plan** (Phase 2: shot-validator
veto on v5 + ExIt re-targeted at the producer tier).

## Current state (2026-06-11)

**Active Kaggle submissions:** `producer_bundle` (LB 1242.7, rank ~140/4212) and
`v5_bundle` (our fork: endgame horizon clamp + 4P nearest-opponent priority; climbing).
Best RL checkpoint = `v2_exit_gumbel_on_a100/ckpt_000035.pt` (apex-era; ~31% vs the
public pool in 2P, last in 4P — re-anchor before investing further).

**Measurement (hard-won, do not relearn):**
- THE metric = `scripts/arena.py` (real Kaggle env, side-alternated paired seeds, 2P
  and `--players 4`). Win-vs-apex is retired.
- A/A noise floor: n=60 mirror games measured 44% on IDENTICAL agents (±6%). Never
  act on n<100 mirror results; every small-n outlier so far regressed at high n
  (77% agent → ~23–40%; 67% smoke → 31%; 57% roi sweep → 52%).
- The public pool ceilings at ~99% for producer-level agents → improvements to the
  base are only measurable via mirror A/Bs (use margin metrics; arena CSVs record
  `steps`) or the real ladder.

**ExIt dead ends — do NOT repeat (empirically confirmed, full write-ups in
`rl_research/EXPLORED_AND_ABANDONED.md` + `STRONGER_EXPERT_SEARCH_PLAN.md`):**
model-free PPO (structural stall); neural-value search leaves (OOD collapse);
two-player one-ply search (passivity bias); every-step weak-rollout opponent
(undervalues aggression); z-scored value-leaf blend (regressed 6/6 seeds, monotonic
in w); mixed collection pool + 3x data (diluted distillation); arrival-resolving
search horizon (`exit.arrival_horizon` — mechanism verified, but distilling the
passive sim's long-range predictions = overextension; h2h 17/17/3% vs champion).
**Closed pattern (6 failures): anything that leans harder on the passive sim's
long-range evaluations loses.** The one validated search win:
**Gumbel/Sequential-Halving selection** (`exit.gumbel_search`, +9.4% over control
on paired seeds) — it changed selection/anchoring, not the world model.

**Key files:** `v2/exit_train.py` (collect→search→distill; opponent =
`exit.opponent`, resolved via `agents.load_named_agent`; `play_single_game` is
P0-only, NOT side-alternated), `v2/search.py` (Gumbel + heuristic leaf eval),
`v2/imitation.py` (BC from `imitation.bc_expert`), `v2/train.py`
(`run_periodic_eval` side-alternated/paired; `make_v2_eval_agent`),
`agents/__init__.py` (`load_named_agent` + fast_env obs shim), `scripts/arena.py`,
`scripts/eval_fast.py`, `scripts/build_v5_bundle.py`.

⚠️ Producer-tier opponents are ~50-100× slower per step than the retired apex
(torch planners). ExIt collection/eval at scale belongs on Colab
(`notebooks/train_colab.ipynb`), and BC demo caches are worth keeping.

## Config System

Live configs are the **v2** ones: plain YAML with nested sections `env`, `model`, `ppo`,
`reward`, `eval`, `imitation`, plus the ExIt `exit` block. Loaded via
`v2.config.load_v2_config()`; dataclasses live in `v2/config.py` (`V2EnvConfig`,
`V2ModelConfig`, `V2PPOConfig`, `V2RewardConfig`, `V2EvalConfig`, `V2ImitationConfig`,
`V2ExItConfig`). Override fields programmatically (no CLI `--set`). `src/config.py` is retained
only as a shared dependency of the reused `src/` modules. See `MEMORY.md` for the full
per-config breakdown.
