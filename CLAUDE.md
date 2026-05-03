# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Kaggle Orbit Wars competition.
**Primary purpose: learn reinforcement learning and win competition.** The deterministic rule-based agent is a benchmark only; the main effort is one or more agents using modern RL (PPO, self-play, MARL). The agents should be capable of training in Kaggle (GPU), Google Colab (GPU), or locally (no GPU). The evaluation should be capable of running in Kaggle or locally (no GPU).

## Repository Structure

```
orbit_wars/
├── agents/                  # Agent implementations
│   ├── baseline.py          # Deterministic rule-based agent (benchmark)
│   └── rl_agent.py          # SB3 model wrapper + submission export
├── configs/                 # YAML experiment configs
│   ├── ppo_default.yaml     # Default PPO vs baseline (500k steps)
│   └── ppo_selfplay.yaml    # Self-play fine-tuning (2M steps)
├── envs/                    # Gymnasium wrappers
│   └── orbit_wars_env.py    # Single-agent env, obs/action encoding
├── evaluation/              # Evaluation utilities
│   └── evaluate.py          # run_games, head_to_head, benchmark
├── notebooks/               # Exploratory Jupyter notebooks
│   ├── explore.ipynb        # Main development notebook (Kaggle/Colab ready)
│   ├── orbit_wars_rl.ipynb  # Previous iteration (reference)
│   └── getting-started.ipynb # Kaggle tutorial (reference, 51 MB)
├── outputs/                 # .gitignored — all generated files go here
│   ├── checkpoints/         # Model .zip files (best_model.zip, ckpt_*.zip)
│   ├── logs/                # TensorBoard event files, eval results
│   └── submissions/         # Generated submission.py files
├── scripts/                 # CLI entry points
│   ├── train.py             # Train an agent
│   ├── evaluate.py          # Head-to-head evaluation
│   └── submit.py            # Generate Kaggle submission file
├── training/                # Core training logic (imported by scripts/train.py)
│   └── train.py             # train(), load_config(), resolve_opponent()
├── requirements.txt
└── .gitignore
```

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (default config: PPO vs baseline, 500k steps)
python scripts/train.py

# Train with a different config
python scripts/train.py --config configs/ppo_selfplay.yaml

# Override specific values inline
python scripts/train.py --set training.total_timesteps=1000000 env.n_envs=8 training.device=cuda

# Resume from a checkpoint
python scripts/train.py --resume outputs/checkpoints/ppo_default_20260501_120000/best_model.zip

# Evaluate: trained model vs baseline and random
python scripts/evaluate.py --model outputs/checkpoints/<run>/best_model.zip

# Evaluate with a custom number of games
python scripts/evaluate.py --model outputs/checkpoints/<run>/best_model.zip --games 50

# Full matrix: two models + baseline + random
python scripts/evaluate.py \
    --model outputs/checkpoints/run_a/best_model.zip:rl_v1 \
    --model outputs/checkpoints/run_b/best_model.zip:rl_v2 \
    --baseline --random --games 30

# Generate a submission (baseline)
python scripts/submit.py --baseline

# Generate a submission (RL model) and verify it runs
python scripts/submit.py --model outputs/checkpoints/<run>/best_model.zip --verify
```

## Environment Setup

```bash
pip install -r requirements.txt
# or on Colab/Kaggle:
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

**Observation fields** (`obs.field` or `obs["field"]` — both work):

**IMPORTANT**: `planets` and `fleets` entries may be either **lists** (e.g., `[id, owner, x, y, ...]`) or **dicts** (e.g., `{"id": 0, "owner": 1, "x": 30, ...}`) depending on the Kaggle environment version. Always handle both formats when parsing them.

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

## Env Wrapper Design (`envs/orbit_wars_env.py`)

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

1. **Baseline** (done): deterministic rule-based agent in `agents/baseline.py`
2. **PPO vs baseline**: `python scripts/train.py --config configs/ppo_default.yaml`
3. **Self-play**: `python scripts/train.py --config configs/ppo_selfplay.yaml`
4. **Improve gradually**:

| Technique | How |
|-----------|-----|
| Larger network | `--set training.net_arch=[512,512]` |
| Action masking | Add `MaskablePPO` from `sb3-contrib` |
| Better reward shaping | Edit `OrbitWarsEnv._shape_reward()` |
| Population-based training | Train a league of agents, sample opponents |
| Recurrent policy | LSTM/GRU to handle partial observability |

## Config System

Configs are plain YAML. Override any value with `--set key.subkey=value` on the CLI.
Values are parsed by `yaml.safe_load` so you can pass lists: `--set training.net_arch=[512,512]`.

Checkpoint directory structure:
```
outputs/checkpoints/<run_name>_<timestamp>/
    best_model.zip     ← saved by EvalCallback when eval score improves
    ckpt_*.zip         ← periodic saves every 50k steps
    final_model.zip    ← saved at end of training
outputs/logs/<run_name>_<timestamp>/
    evaluations.npz    ← eval history (load with np.load)
    events.*           ← TensorBoard logs
```
View training in TensorBoard: `tensorboard --logdir outputs/logs`
