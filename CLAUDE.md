# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Kaggle Orbit Wars competition.
**Primary purpose: learn reinforcement learning and win competition.** The apex rule-based agent is the current benchmark; the main effort is one or more agents using modern RL (PPO, self-play, MARL). The agents should be capable of training in Kaggle (GPU), Google Colab (GPU), or locally (no GPU). The evaluation should be capable of running in Kaggle or locally (no GPU).

## Repository Structure

```
orbit_wars/
├── src/                     # Transformer PPO pipeline (primary RL effort)
│   ├── __init__.py
│   ├── config.py            # TrainConfig dataclasses (env, model, ppo, reward, eval, imitation)
│   ├── game_types.py        # PlanetState, FleetState, GameState, parse_observation()
│   ├── features.py          # Feature encoding, fleet transit, SourceDecision
│   ├── policy.py            # TransformerPolicy (~493K params default)
│   ├── ppo.py               # Factored PPO (target + fraction), clipped update, optional imitation loss
│   ├── opponents.py         # Apex, Random, SelfPlay, Hybrid, Distilled opponents + _policy_act()
│   ├── env.py               # Kaggle env wrapper (2/4-player, side alternation, mixed scheduling)
│   ├── logging.py           # TrainLogger (TensorBoard + CSV), EvalResult, periodic eval
│   ├── imitation.py         # DemonstrationBuffer, collect_demonstrations, BC loss, bc_pretrain
│   └── train.py             # Training loop: BC pretrain → PPO + mixed 2p/4p self-play
├── agents/                  # Agent implementations
│   ├── apex.py              # Apex rule-based agent (benchmark)
│   ├── hybrid.py            # Mission-based + timeline agent
│   └── rl_agent.py          # SB3 model wrapper + submission export
├── configs/                 # YAML experiment configs
│   ├── transformer_ppo.yaml   # Transformer PPO default config (2000 updates, apex opponent)
│   ├── transformer_dagger.yaml# DAgger: BC pretrain from apex + PPO with imitation decay (3000 updates)
│   ├── transformer_mixed.yaml # Mixed: BC from apex + dense_relative reward + 2p/4p self-play (5000 updates)
│   ├── ppo_default.yaml       # SB3 PPO vs apex (500k steps, legacy)
│   └── ppo_selfplay.yaml      # SB3 self-play fine-tuning (legacy)
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
│   ├── logs/                # TensorBoard event files, CSV metrics, eval results
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
# Train with default config (PPO vs apex, 2000 updates)
uv run python -m src.train --config configs/transformer_ppo.yaml

# DAgger: BC pretrain from apex demos + PPO with imitation decay (3000 updates)
uv run python -m src.train --config configs/transformer_dagger.yaml

# Mixed: BC from apex + dense_relative reward + 2p/4p self-play (5000 updates)
uv run python -m src.train --config configs/transformer_mixed.yaml

# Train on Colab/Kaggle (no uv)
python -m src.train --config configs/transformer_mixed.yaml
```

### Evaluate a trained transformer checkpoint locally

```bash
uv run python -c "
import torch
from src.config import load_train_config
from src.policy import TransformerPolicy
from src.logging import make_eval_agent
from evaluation.evaluate import run_games, print_results

cfg = load_train_config('configs/transformer_dagger.yaml')
device = torch.device('cpu')
policy = TransformerPolicy(cfg.model, cfg.env).to(device)
ckpt = torch.load('outputs/checkpoints/transformer_dagger/ckpt_last.pt',
                   map_location=device, weights_only=True)
policy.load_state_dict(ckpt['policy'])
policy.eval()

agent = make_eval_agent(policy, cfg, device)

from agents.apex import agent as apex
print_results('rl', 'apex', run_games(agent, apex, n_games=20, verbose=True))
"
```

### Monitor training

```bash
uv run tensorboard --logdir outputs/logs
```

### Legacy SB3 Pipeline (scripts/)

```bash
# Train (default config: PPO vs apex, 500k steps)
uv run python scripts/train.py

# Train with a different config
uv run python scripts/train.py --config configs/ppo_selfplay.yaml

# Override specific values inline
uv run python scripts/train.py --set training.total_timesteps=1000000 env.n_envs=8 training.device=cuda

# Resume from a checkpoint
uv run python scripts/train.py --resume outputs/checkpoints/ppo_default_20260501_120000/best_model.zip

# Evaluate: trained model vs apex and random
uv run python scripts/evaluate.py --model outputs/checkpoints/<run>/best_model.zip

# Evaluate with a custom number of games
uv run python scripts/evaluate.py --model outputs/checkpoints/<run>/best_model.zip --games 50

# Full matrix: two models + apex + random
uv run python scripts/evaluate.py \
    --model outputs/checkpoints/run_a/best_model.zip:rl_v1 \
    --model outputs/checkpoints/run_b/best_model.zip:rl_v2 \
    --apex --random --games 30

# Generate a submission (apex)
uv run python scripts/submit.py --apex

# Generate a submission (RL model) and verify it runs
uv run python scripts/submit.py --model outputs/checkpoints/<run>/best_model.zip --verify
```

## Environment Setup

**Locally, always use `uv` to run Python scripts** (e.g. `uv run python scripts/train.py`). Do not use bare `python` or `python3`.

```bash
# Local (uv manages the virtualenv):
uv run python scripts/train.py

# On Colab/Kaggle (no uv):
pip install --upgrade "kaggle-environments>=1.28.0" "stable-baselines3[extra]>=2.3" gymnasium pyyaml tensorboard
```

### Google Colab Workflow

`notebooks/train_colab.ipynb` runs the DAgger pipeline on Colab GPU with Google Drive persistence.

**Cell order** (designed so submission happens immediately after training):
1. Setup (mount Drive, clone repo, install deps)
2. GPU Verification
3. Experiment Config (loads `transformer_dagger.yaml`, applies H100 overrides)
4. **Train** (demo collection → BC pretrain → PPO with imitation decay)
5. **Generate Submission** (apex + hybrid + checkpoint copy to Drive)
6. **Evaluate** (trained model vs apex and random, 20 games each)
7. **TensorBoard** (last — can block downstream cell execution)

**H100 Colab overrides** in the config cell: `num_envs=4`, `rollout_steps=128`, `total_updates=5000`, `eval_every=250`.

**After Colab training**, download checkpoint from Drive and evaluate locally — see "Evaluate a trained transformer checkpoint locally" in Common Commands above.

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

**Reward**: sparse terminal ±1 by default. Three modes via `reward.reward_mode`:
- `sparse`: terminal ±1 only (default)
- `dense_absolute`: Δown_ships × coef + Δown_prod × prod_coef
- `dense_relative`: Δ(own_ships − best_enemy_ships) × coef — rewards gaining advantage

### Logging (`src/logging.py`)

- `TrainLogger` writes all metrics to both TensorBoard (`outputs/logs/<run_name>/`) and CSV (`metrics.csv`)
- `log_update(update, metrics)` — per-update train metrics (loss, policy_loss, value_loss, entropy, reward, imitation_coef, etc.)
- `log_eval(update, results)` — periodic eval win rates as `eval/{opponent}_win_rate`
- `make_eval_agent(policy, cfg, device)` — creates a Kaggle-compatible `agent(obs, config)` callable from a policy (snapshots weights)
- `run_periodic_eval(policy, cfg, device)` — runs eval games against all opponents in `cfg.eval.eval_opponents`, returns `list[EvalResult]`

### Imitation Learning (`src/imitation.py`)

**DAgger-style pipeline** for distilling an expert agent's strategy:

```
Phase 1: collect_demonstrations()  →  Expert (apex or hybrid) plays n games, records (SourceDecision, target_index, fraction_bin)
Phase 2: bc_pretrain()             →  Supervised cross-entropy on expert demos (Adam, per-epoch shuffle)
Phase 3: PPO + imitation loss      →  ppo_update() blends PPO loss + β * BC loss; β decays linearly to 0
Phase 4: Mixed self-play            →  MixedScheduler blends rule-based + self-play; rule_based_prob decays linearly
```

- `DemonstrationBuffer` — parallel lists of numpy arrays for all 8 feature fields + 2 action labels
- `collect_demonstrations(n_games, cfg, opponent_name)` — runs hybrid agent, maps moves to action space via angular matching (≤30° tolerance → NoOp if no match); validates targets against mask to avoid inf BC loss
- `compute_bc_loss(policy, batch)` — cross-entropy on target selection + cross-entropy on fraction (masked for NoOp); clamps logits to -1e4 min to avoid inf from masked positions
- `_map_to_action_space(angle, ships, src_ships, decision, env_cfg)` — finds closest target by angular difference, maps ship fraction to nearest bin

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

**`configs/transformer_ppo.yaml`** — default PPO config (2000 updates, apex opponent, eval every 100)

**`configs/transformer_dagger.yaml`** — DAgger config:
```yaml
imitation:
  enabled: true
  bc_expert: apex       # expert agent for demo collection (apex or hybrid)
  bc_games: 100         # demo games from expert agent
  bc_demo_opponent: random  # expert plays against this
  bc_epochs: 50         # supervised pretraining epochs
  bc_lr: 0.001
  bc_batch_size: 256
  coef_start: 0.5       # initial imitation loss weight
  coef_decay_updates: 1000  # linear decay to 0
  distilled_opponent: true  # use BC-pretrained model as training opponent

eval:
  eval_every: 100
  eval_games: 10
  eval_opponents: [apex, random]

ppo:
  total_updates: 3000
  lr: 0.0001            # lower since starting from BC-pretrained weights

reward:
  reward_mode: dense_relative  # rewards gaining ship advantage over enemies
  dense_ship_coef: 0.002
```

**`configs/transformer_mixed.yaml`** — Full pipeline: BC from apex + dense_relative reward + mixed 2p/4p self-play:
```yaml
four_player_prob: 0.3         # 30% of episodes are 4-player
rule_based_prob_start: 1.0    # start with all rule-based opponents
rule_based_prob_end: 0.2      # end with mostly self-play
rule_based_decay_updates: 2000  # linear decay over this many updates

reward:
  reward_mode: dense_relative
  dense_ship_coef: 0.002

imitation:
  bc_expert: apex             # clone apex behavior (faster than hybrid)
  bc_games: 100
  bc_epochs: 50

ppo:
  total_updates: 5000
```

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

1. **Apex** (done): rule-based agent in `agents/apex.py`
2. **Hybrid** (done): mission-based + timeline agent in `agents/hybrid.py`
3. **Transformer PPO vs apex** (done): `uv run python -m src.train --config configs/transformer_ppo.yaml`
4. **Logging + periodic eval** (done): TensorBoard + CSV metrics, win rate tracking against baselines
5. **DAgger / imitation learning** (done): BC pretrain from hybrid demos + PPO with decaying imitation loss: `uv run python -m src.train --config configs/transformer_dagger.yaml`
6. **Mixed self-play + 4p** (done): `MixedScheduler` blends rule-based + self-play opponents with linear decay; supports 2p and 4p games via `four_player_prob`
7. **Improve gradually**:

| Technique | How |
|-----------|-----|
| Larger network | Increase `model.embed_dim`, `model.n_layers`, `model.ff_dim` in YAML |
| More targets | Increase `env.max_targets` (default 30) |
| Dense reward | Set `reward.reward_mode: dense_relative` in config |
| Population-based training | Train a league of agents, sample opponents |
| Better features | Extend `src/features.py` (e.g. comet tracking, orbit prediction for targets) |

## Config System

### Transformer PPO configs (`src/`)

Configs are plain YAML with nested sections: `env`, `model`, `ppo`, `reward`, `eval`, `imitation`.
Loaded via `src.config.load_train_config()`. Override fields programmatically (no CLI `--set` for `src/`).

Dataclasses in `src/config.py`: `EnvConfig`, `ModelConfig`, `PPOConfig`, `RewardConfig`, `EvalConfig`, `ImitationConfig`, `TrainConfig`.

### Legacy SB3 configs (`scripts/`)

Configs are plain YAML. Override any value with `--set key.subkey=value` on the CLI.
Values are parsed by `yaml.safe_load` so you can pass lists: `--set training.net_arch=[512,512]`.
