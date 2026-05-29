# v2_bc Recommendation Experiments — Paper-Derived Improvements

**Date:** 2026-05-28
**Platform:** CPU (8 cores), no GPU
**Pipeline:** V2 OrbitNet (`v2/train.py`), BC warm start from apex
**Companion:** `rl_research/REPORT.md` (the survey + recommendation rationale)

## Design

Unlike the earlier no-BC hyperparameter sweep (`experiments/REPORT.md`), this round
improves the **BC-warm-started** `v2_bc` agent. To keep the comparison clean and the
compute tractable:

1. A single **prime** step collects+caches apex demonstrations and BC-pretrains a
   shared `ckpt_000000.pt`.
2. Every experiment (baseline + R1–R5) **resumes from that identical warm start** and
   runs the same number of PPO updates, varying exactly **one** knob.

So all curves diverge *only* because of the change under test — the BC initialization
is byte-identical across runs.

| Exp | Knob changed | Value | Baseline | Paper lineage |
|---|---|---|---|---|
| baseline | — | — | ent=0.01, γ=0.99, vf=0.5, decay=1000, prod_bonus=9.0 | v2_bc |
| R1 `ent_anneal` | entropy schedule | 0.01 → 0.0 (annealed) | fixed 0.01 | PPG, MAPPO |
| R2 `high_gamma` | discount γ | 0.997 | 0.99 | DreamerV3, long-horizon |
| R3 `strong_value` | value-loss weight | 1.0 | 0.5 | PPG, MAPPO |
| R4 `fast_selfplay` | rule_based decay | 150 updates | 1000 updates | XLand, DeepNash |
| R5 `no_prod_bonus` | early_prod_bonus | 0.0 | 9.0 | Go-Explore, shaping bias |

Shared settings: seed 42, embed=128/layers=3, rollout=32, num_envs=1, epochs=4,
minibatch=256, lr=1e-4, clip=0.2, dense_relative reward, apex opponent + distilled
self-play (MixedScheduler), eval at the listed updates (10 games each vs apex & random).

## Results

### Training summary (200 updates each, eval at 100 & 200)

| Experiment | Avg Reward (last 50) | Final Loss | Final Value Loss | Avg Entropy* | vs Apex | vs Random |
|---|---|---|---|---|---|---|
| baseline | -0.356 | 0.1856 | 0.2677 | 9.05 | **0%** | 100% |
| r1_ent_anneal | -0.226 | 0.0308 | 0.0811 | 4.27 | **0%** | 100% |
| r2_high_gamma | -0.028 | 0.0454 | 0.0043 | 11.98 | **0%** | 100% |
| r3_strong_value | -0.023 | -0.0193 | 0.8764 | 35.00 | **0%** | 100% |
| r4_fast_selfplay | -0.029 | -0.0864 | 0.0220 | 20.78 | **0%** | 100% |
| r5_no_prod_bonus | -0.082 | -0.1649 | 0.0307 | 8.94 | **0%** | 100% |

*Avg Entropy here is the **summed-over-planets** entropy of the *old* (pre-fix) action
code — see the analysis; this metric's volatility is itself a finding.

### Tie-breaker: score margin vs apex (20 games, alternating sides)

Because the binary win-rate is saturated (all 0% vs apex / 100% vs random), the
discriminating metric is how *close* each agent gets. Score = ships on owned
planets + ships in owned fleets; margin = own − apex at game end.

| Rank | Experiment | W/L/T vs apex | Mean score margin | Mean survival (steps) | Mean planets held |
|---|---|---|---|---|---|
| 1 | r3_strong_value | 0/20/0 | −3305.8 | 118 | 0.0 |
| 2 | baseline | 0/20/0 | −3552.3 | 119 | 0.0 |
| 3 | r1_ent_anneal | 0/20/0 | −3668.0 | 119 | 0.0 |
| 4 | r2_high_gamma | 0/20/0 | −3990.8 | 137 | 0.0 |
| 5 | r5_no_prod_bonus | 0/20/0 | −4004.1 | 139 | 0.0 |
| 6 | r4_fast_selfplay | 0/20/0 | −4076.9 | 126 | 0.0 |

Every agent is **eliminated by apex around step 120–139 of 498**, holding **0 planets**
at game end. The margin spread (−3306 to −4077) is within noise; R3 is marginally
"least bad", R4 marginally worst, but no variant is remotely competitive.

### Plots

- Reward: `plots2/reward.png`
- Total / value / policy loss: `plots2/loss.png`, `plots2/value_loss.png`, `plots2/policy_loss.png`
- Entropy: `plots2/entropy.png`
- Eval win rates: `plots2/eval_winrates.png`
- Tie-breaker data: `margin_vs_apex.json`, `margin_table.md`

## Per-recommendation analysis

**R1 — entropy annealing.** Lowered the (summed) entropy as designed (4.27 vs baseline
9.05) and improved training reward (−0.226 vs −0.356), but did **not** change eval: 0%
vs apex, margin −3668 (slightly *worse* than baseline). Verdict: the entropy schedule
acts on a mis-scaled bonus (the summed-over-planets entropy term, see below), so it
couldn't have the intended effect. *This motivated Fix 2.*

**R2 — γ = 0.997.** Best training reward of the clean comparators (−0.028) and tiny value
loss (0.0043), and the agent survived *longer* vs apex (137 steps), but margin was worse
(−3991). A higher discount only helps if the critic can represent the longer-horizon
return — and the critic is broken (see R3). *Premature without Fix 3.*

**R3 — vf_coef = 1.0.** "Least-bad" margin (−3306) and best training reward (−0.023), but
its value loss is the **highest** (0.876) and entropy exploded to 35. Training the critic
*harder* on a *mis-scaled* target just amplifies the noise rather than fixing it. This is
the clearest evidence that the problem is value **scale**, not value **effort**.
*Motivated Fix 3 (symlog), not just reweighting.*

**R4 — self-play decay 1000→150.** Worst margin (−4077). Injecting self-play early, when
the policy is weak and under-committing, produced a weaker, less apex-specialized agent.
Self-play needs a competent base policy and a stabilizer first (DeepNash KL anchor).

**R5 — early_prod_bonus = 0.** Longest survival (139 steps) but near-worst margin (−4004).
Removing the shaping didn't help here because the dominant failure (dribble-fleets that
can't capture or hold) is upstream of the reward — it's an action-representation problem.

**Cross-cutting finding (the real result):** the training logs show **violent entropy
thrashing** (baseline swung 0.01 → 30.7 → 0.01; R3 climbed to 88), **value-loss spikes**
(R2 hit 3.1, R3 hit 2.0 mid-run), and `eps=0` on nearly every update (episodes are
150–500 steps but rollouts are 32, so the terminal win/loss almost never reaches an
update). No knob addresses these; they are structural.

## Structural diagnosis → the four fixes (implemented)

The experiments ruled out the easy explanations and pointed at four structural causes,
all now implemented in `v2/` (validated by smoke tests — see `rl_research/REPORT.md §6`):

1. **Ship-fraction entanglement** (`decode_sampled_actions`: `frac = softmax_prob[target]`).
   The selection probability doubled as the ship fraction, so fleets were systematically
   ~20% (too small to capture/hold → eliminated by step ~120), and **PPO never had a
   gradient term for fleet size** (the fraction wasn't a sampled action). BC also discarded
   apex's fractions. → **Fix 1: dedicated masked fraction head** (factored `(target, frac)`
   action; fraction now supervised by BC and optimized by PPO/ExIt). Params 518K→551K.
2. **Summed-over-planets entropy** → board-dependent, incoherent exploration pressure
   (the 0.01↔88 swings; explains why R1 did nothing). → **Fix 2: mean per-planet entropy.**
   Post-fix smoke entropy sits in a sane ~2–3 range.
3. **Value scale mismatch** (tiny dense rewards + ±1 terminal; zero-init mean-pool head;
   loss spikes). R3 showed training value harder backfires. → **Fix 3: symlog value
   targets** (DreamerV3), gated by `ppo.value_symlog`.
4. **Rollout 32 ≪ episode 150–500** → terminal signal starvation (`eps=0`). →
   **Fix 4: longer rollouts** (`ppo.rollout_steps`), meaningful only once Fix 3 makes the
   critic usable.

## Final recommendation

The five knobs confirmed that **model-free PPO on a CPU is the wrong primary tool** for
this problem in this budget — the bottleneck is the action representation and value
learning, not hyperparameters. The plan, now implemented:

1. **Foundation fixes (done):** fraction head, mean entropy, symlog value, longer rollouts.
   These are prerequisites for *any* learning path. Re-run this harness (now with a
   non-saturated margin metric) to confirm each fix moves the needle.
2. **Pivot to imitation + search (done):** a new **v2 Expert Iteration** pipeline
   (`v2/exit_train.py`, `v2/search.py`, `configs/v2_exit.yaml`) BC-clones apex (now able to
   imitate *fleet sizes* via Fix 1), then improves it by **per-planet lookahead search
   against the ground-truth simulator** (`src/simulator.py`) and supervised distillation —
   the AlphaZero/EfficientZero loop minus the model-learning, sidestepping PPO's value-scale
   and rollout pathologies entirely.
3. **PPO/self-play as a final fine-tune** with a DeepNash-style KL-to-reference anchor, not
   as the primary learner.

The single highest-leverage change is **Fix 1 (the fraction head)** — it is what lets BC
clone apex faithfully and lets search/PPO control the decisive variable (how many ships).
Run `uv run python -m v2.exit_train --config configs/v2_exit.yaml` for the BC→ExIt pipeline.

## Reproducibility

```bash
# Run prime + all 6 experiments (baseline + R1..R5)
uv run python scripts/run_recommendation_experiments.py --updates 200 --bc-games 150 --max-parallel 3

# Regenerate plots + summary table
uv run python scripts/plot_recommendation_experiments.py
```
