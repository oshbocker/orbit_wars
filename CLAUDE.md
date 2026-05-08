# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Kaggle Orbit Wars competition.
**Primary purpose: learn reinforcement learning and win competition.** The competitive rule-based agent is the current benchmark; the main effort is one or more agents using modern RL (PPO, self-play, MARL). The agents should be capable of training in Kaggle (GPU), Google Colab (GPU), or locally (no GPU). The evaluation should be capable of running in Kaggle or locally (no GPU).

## Repository Structure

```
orbit_wars/
├── src/                     # Transformer PPO pipeline (primary RL effort)
│   ├── __init__.py
│   ├── config.py            # TrainConfig dataclasses (env, model, ppo, reward)
│   ├── game_types.py        # PlanetState, FleetState, GameState, parse_observation()
│   ├── features.py          # Feature encoding, fleet transit, SourceDecision
│   ├── policy.py            # TransformerPolicy (~493K params default)
│   ├── ppo.py               # Factored PPO (target + fraction), clipped update
│   ├── opponents.py         # CompetitiveOpponent, RandomOpponent, SelfPlayOpponent
│   ├── env.py               # Kaggle env wrapper (2-player, side alternation)
│   └── train.py             # Training loop with sequential per-planet decisions
├── agents/                  # Agent implementations
│   ├── competitive.py       # Net-difference rule-based agent (benchmark)
│   ├── hybrid.py            # Mission-based + timeline agent
│   └── rl_agent.py          # SB3 model wrapper + submission export
├── configs/                 # YAML experiment configs
│   ├── transformer_ppo.yaml # Transformer PPO default config
│   ├── ppo_default.yaml     # SB3 PPO vs competitive (500k steps, legacy)
│   └── ppo_selfplay.yaml    # SB3 self-play fine-tuning (legacy)
├── envs/                    # Gymnasium wrappers (legacy SB3 pipeline)
│   └── orbit_wars_env.py    # Single-agent env, obs/action encoding
├── evaluation/              # Evaluation utilities
│   └── evaluate.py          # run_games, head_to_head, benchmark
├── notebooks/               # Exploratory Jupyter notebooks
│   ├── explore.ipynb        # Main development notebook (Kaggle/Colab ready)
│   ├── orbit_wars_rl.ipynb  # Previous iteration (reference)
│   ├── train_colab.ipynb    # Google Colab training notebook (uses src.train)
│   └── orbit-wars-reinforcement-learning-tutorial.ipynb  # Kaggle RL tutorial
├── outputs/                 # .gitignored — all generated files go here
│   ├── checkpoints/         # Model .pt files (ckpt_last.pt, ckpt_NNNNNN.pt)
│   ├── logs/                # TensorBoard event files, eval results
│   └── submissions/         # Generated submission.py files
├── scripts/                 # CLI entry points (legacy SB3 pipeline)
│   ├── train.py             # Train an agent
│   ├── evaluate.py          # Head-to-head evaluation
│   └── submit.py            # Generate Kaggle submission file
├── training/                # Core training logic (legacy SB3, imported by scripts/train.py)
│   └── train.py             # train(), load_config(), resolve_opponent()
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

## Common Commands

### Transformer PPO (primary pipeline — `src/`)

```bash
# Train with default config (PPO vs competitive, 2000 updates)
uv run python -m src.train --config configs/transformer_ppo.yaml

# Quick smoke test (5 updates, ~1s on CPU)
uv run python -m src.train --config /path/to/smoke_config.yaml

# Train on Colab/Kaggle (no uv)
python -m src.train --config configs/transformer_ppo.yaml
```

### Legacy SB3 Pipeline (scripts/)

```bash
# Train (default config: PPO vs competitive, 500k steps)
uv run python scripts/train.py

# Train with a different config
uv run python scripts/train.py --config configs/ppo_selfplay.yaml

# Override specific values inline
uv run python scripts/train.py --set training.total_timesteps=1000000 env.n_envs=8 training.device=cuda

# Resume from a checkpoint
uv run python scripts/train.py --resume outputs/checkpoints/ppo_default_20260501_120000/best_model.zip

# Evaluate: trained model vs competitive and random
uv run python scripts/evaluate.py --model outputs/checkpoints/<run>/best_model.zip

# Evaluate with a custom number of games
uv run python scripts/evaluate.py --model outputs/checkpoints/<run>/best_model.zip --games 50

# Full matrix: two models + competitive + random
uv run python scripts/evaluate.py \
    --model outputs/checkpoints/run_a/best_model.zip:rl_v1 \
    --model outputs/checkpoints/run_b/best_model.zip:rl_v2 \
    --competitive --random --games 30

# Generate a submission (competitive)
uv run python scripts/submit.py --competitive

# Generate a submission (RL model) and verify it runs
uv run python scripts/submit.py --model outputs/checkpoints/<run>/best_model.zip --verify
```

## Environment Setup

**Locally, always use `uv` to run Python scripts** (e.g. `uv run python scripts/train.py`). Do not use bare `python` or `python3`.

```bash
# Local (uv manages the virtualenv):
uv run python scripts/train.py

# On Colab/Kaggle (no uv):
pip install --upgrade "kaggle-environments>=1.28.0" "stable-baselines3[extra]>=2.3" gymnasium pyyaml
```

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

## Transformer PPO Design (`src/`)

### Architecture

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

**Reward**: sparse terminal ±1 by default; optional dense shaping (Δships × 0.001 + Δproduction × 0.005) via `reward.sparse: false` in config.

### Config (`configs/transformer_ppo.yaml`)

Key config sections: `env` (max_targets, k_neighbors, ship_fractions), `model` (embed_dim, n_heads, n_layers, ff_dim), `ppo` (rollout_steps, num_envs, total_updates, lr, etc.), `reward` (sparse/dense).

### Checkpoint format

```
outputs/checkpoints/<run_name>/
    ckpt_last.pt       ← latest checkpoint
    ckpt_NNNNNN.pt     ← periodic saves (every checkpoint_every updates)
```

Each `.pt` file contains `{"update": int, "policy": state_dict, "optimizer": state_dict}`.

## Legacy Env Wrapper Design (`envs/orbit_wars_env.py`)

**Observation** — `Box(683,)` float32:
- `[0:3]` global: `step/500`, `angular_velocity/0.05`, `n_comets/4`
- `[3:283]` 40 planet slots × 7 features: `is_mine`, `is_enemy`, `is_neutral`, `x/100`, `y/100`, `log1p(ships)/10`, `production/5`
- `[283:683]` 80 fleet slots × 5 features: `is_mine`, `is_enemy`, `x/100`, `y/100`, `log1p(ships)/10`

**Action** — `MultiDiscrete([12, 40, 4])`:
- `own_slot`: which of my planets to send from (no-op if ≥ len(my_planets))
- `target_slot`: target planet index
- `frac_bin`: 0→25%, 1→50%, 2→75%, 3→100% of ships

**Reward**: dense shaping (Δships × 0.001 + Δproduction × 0.005) + terminal ±1.

## Key Strategic Considerations

- **Travel time**: target garrison grows during flight; send `target.ships + production × travel_time + 1`
- **Sun avoidance**: never aim through center (50, 50); check with `_passes_through_sun()`
- **Fleet speed asymmetry**: tiny fleets are slow (1 unit/turn); send enough ships to get speed
- **Orbiting targets**: predict future planet position before aiming
- **Production priority**: high-production planets compound; capture them first
- **4-player dynamics**: let opponents fight, then strike the winner

## RL Development Roadmap

1. **Competitive** (done): net-difference rule-based agent in `agents/competitive.py`
2. **Hybrid** (done): mission-based + timeline agent in `agents/hybrid.py`
3. **Transformer PPO vs competitive** (done): `uv run python -m src.train --config configs/transformer_ppo.yaml`
4. **Self-play**: set `opponent: self` in config
5. **Improve gradually**:

| Technique | How |
|-----------|-----|
| Larger network | Increase `model.embed_dim`, `model.n_layers`, `model.ff_dim` in YAML |
| More targets | Increase `env.max_targets` (default 30) |
| Dense reward | Set `reward.sparse: false` in config |
| Population-based training | Train a league of agents, sample opponents |
| Better features | Extend `src/features.py` (e.g. comet tracking, orbit prediction for targets) |

## Config System

### Transformer PPO configs (`src/`)

Configs are plain YAML with nested sections: `env`, `model`, `ppo`, `reward`.
Loaded via `src.config.load_train_config()`. Override fields programmatically (no CLI `--set` for `src/`).

### Legacy SB3 configs (`scripts/`)

Configs are plain YAML. Override any value with `--set key.subkey=value` on the CLI.
Values are parsed by `yaml.safe_load` so you can pass lists: `--set training.net_arch=[512,512]`.
