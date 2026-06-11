# Explored & Abandoned — Compressed Record

*Last gardened: 2026-06-05.* This file is the graveyard. It preserves the **idea**
and the **reason we stopped** for code/experiments that were removed from the repo so
the repository stays focused on the live effort (the **v2 ExIt** pipeline) without
losing the lessons. Anything here is recoverable from git history if needed.

The live effort and its plan live in `CLAUDE.md` ("Current best agent & next-session
plan") and `rl_research/STRONGER_EXPERT_SEARCH_PLAN.md`. The reused, still-live
building blocks are `v2/` plus the `src/` leaf modules it imports
(`game_types`, `features`, `config`, `policy`, `opponents`, `ppo`, `logging`,
`simulator`).

---

## The one lesson that killed most of the below

**Pure model-free PPO from scratch does not beat a strong rule-based opponent (apex) in
this game.** Three independent streams (our own knob sweeps, a 10-paper literature
survey, and the Kaggle community/leaderboard) all converged on the same conclusion: the
ceiling path is a **trustworthy learned value function feeding search / Expert Iteration**,
not more PPO tuning. Every cluster removed below is either (a) a model-free PPO variant
that hit this wall, or (b) v1 scaffolding superseded by the v2 ExIt implementation of
that lesson.

---

## Cluster 1 — Legacy SB3 pipeline *(removed 2026-06-05)*

**What it was.** The original Stable-Baselines3 approach: a Gymnasium wrapper
(`envs/orbit_wars_env.py`, `Box(683)` obs / `MultiDiscrete([12,40,4])` action), an SB3
PPO training loop (`training/train.py`, `scripts/train.py`), an SB3 model→Kaggle
submission wrapper (`agents/rl_agent.py`, `scripts/submit.py`), an evaluation CLI
(`scripts/evaluate.py`), two early rule-based agents kept only as SB3 opponents
(`agents/ultra.py`, `agents/early_ultra.py`), and configs `ppo_default.yaml` /
`ppo_selfplay.yaml`.

**Why abandoned.** Superseded by the custom transformer PPO (then by v2). The flat
`MultiDiscrete` action head cannot express the per-turn `(source, target, fraction)`
launch structure, and SB3's single-agent assumptions fought the multi-launch / variable-
planet-set nature of the game. Fully self-contained — nothing in the live pipeline
imported it. `evaluation/evaluate.py` (`run_games`/`head_to_head`/`print_results`) was
**kept** because the live code still uses it; only its dead `benchmark()` helper's
dependency on `envs/` was removed.

## Cluster 2 — V1 transformer-PPO training half *(removed 2026-06-05)*

**What it was.** The first custom (non-SB3) pipeline: per-planet **sequential**
transformer decisions. Removed files were the *training* half —
`src/train.py` (BC pretrain → PPO + mixed 2p/4p self-play loop), `src/env.py` (Kaggle
env wrapper), `src/imitation.py` (DAgger-style BC from apex/hybrid demos),
`src/exit_train.py` (v1 Expert Iteration) and `src/search.py` (v1 per-planet lookahead) —
plus configs `transformer_ppo.yaml`, `transformer_dagger.yaml`, `transformer_mixed.yaml`,
`expert_iteration.yaml`.

**Why abandoned.** Superseded by **v2 OrbitNet** (`v2/`), which does the same job in
**one forward pass per step** (self-attention over all planets + pairwise output head)
instead of N sequential passes, and whose ExIt implementation (`v2/exit_train.py`,
`v2/search.py`) is the live effort. The v1 *leaf* modules
(`game_types`, `features`, `policy`, `opponents`, `ppo`, `logging`, `simulator`, `config`)
are still imported by v2 and were **kept**. (The dead lazy `compute_bc_loss` import inside
`src/ppo.py:ppo_update` — v1-only — was removed during this prune.)

## Cluster 3 — Hyperparameter & "paper recommendation" sweeps *(removed 2026-06-05)*

**What it was.** Two systematic CPU sweep rounds over the v2 OrbitNet PPO agent, plus
their harness/plot/eval/profile scripts (`scripts/run_hparam_experiments.py`,
`plot_experiments.py`, `run_recommendation_experiments.py`,
`plot_recommendation_experiments.py`, `eval_margin_vs_apex.py`, `eval_hold_mask.py`,
`plot_eval_curve.py`, `profile_env_step.py`) and the `experiments/` results dir.

- **Round 1 (no-BC, `experiments/REPORT.md`):** 10 configs varying one knob each
  (net size, lr, entropy, epochs, rollout, batch). **All 0% vs apex / 70–100% vs random.**
  Notably `small_net` (98K) matched/beat `large_net` (2.6M) → **capacity is not the
  bottleneck** for model-free PPO here.
- **Round 2 (BC-warm-started, `experiments/RECOMMENDATIONS_REPORT.md`):** five
  literature-derived knobs (entropy anneal, γ=0.997, value-loss=1.0, fast self-play decay,
  no prod-bonus) from a shared byte-identical BC warm start. **All still 0% vs apex.**

**Why abandoned.** Conclusively demonstrated the stall is **structural, not a
hyperparameter**: no knob in either round moved apex win-rate off 0%. Directly motivated
the pivot to ExIt/search. (The `eval_hold_mask` diagnostic separately showed passivity is
*ineffective attacks*, not the hold mechanism — fix is reward+opponent, not an idle
penalty. That lesson is carried in MEMORY.md.)

## Cluster 4 — Superseded notebooks & research docs *(removed 2026-06-05)*

**Notebooks removed:** `exit_train_colab.ipynb`, `transformer_mixed_walkthrough.ipynb`
(v1 transformer pipeline); `orbit_wars_rl.ipynb` (earliest iteration);
`v2_orbitnet_walkthrough.ipynb`, `train_v2_colab.ipynb` (superseded by the live
`train_colab.ipynb`); `orbit-wars-1000-public-score-agent.ipynb` +
`orbit_wars_heuristic_agent_scored_1000.py` (a public-LB-1000 heuristic agent — its
strategy is already captured by `agents/apex.py`, the live benchmark). Live notebooks
kept: `train_colab.ipynb` (A100 v2 BC→ExIt), `explore.ipynb` (scratch dev).

**Research docs removed (essence preserved here):**

- **`REPORT.md`** — 10-paper modern-RL survey (MAPPO, Invalid-Action-Masking, DeepNash,
  DreamerV3, EfficientZero, Go-Explore, PPG, Decision-Transformer, XLand, DouZero) mapped
  onto Orbit Wars' five difficulties (structured action space, sparse long-horizon credit,
  non-stationary opponents, delayed consequences, compute-efficiency). **Takeaway:** the
  recurring trick across all of them is a good critic + search/distillation; PPO alone is
  insufficient. Spawned the Round-2 sweep (Cluster 3), which falsified the "just tune PPO"
  hope.
- **`IMPROVEMENT_RESEARCH.md`** — diagnosed the **master bottleneck = sample throughput**
  (~5 env-steps/s on the Kaggle harness; PPO self-play needs millions–billions, cf. Lux AI
  winners). **Cure (implemented):** turn `src/simulator.py` into a fast, faithful self-play
  env → became `v2/fast_env.py` (16/16 step-for-step match vs the Kaggle engine).
- **`SIMULATOR_AUDIT.md`** — fidelity audit of `src/simulator.py` vs the Kaggle
  `interpreter()`. Documented the true engine turn order (launch → production → fleet move +
  continuous collision → rotation/comet/sweep → combat → scoring) and three gap tiers. **The
  big one (Tier A1): `sim_step` has no opponent** — enemies never launch/defend, so a policy
  trained on it learns against a passive world (root cause of the ExIt fraction head
  collapsing to uniform). This audit's findings drove the Phase-2 *positional* simulator and
  remain the spec for any in-sim opponent (Phase 4b).
- **`V3_STALL_PLAYBOOK.md`** — contingency plan written while the v3 (BC→PPO, PBRS reward,
  PFSP pool) run was cooking. **Headline:** pure PPO-from-scratch stalls vs strong rule-based;
  the observed *collapse* (60% @ u1750 → 25%) = the actor/critic shared-trunk fight (PPG) +
  an imitation anchor decaying to 0 (DeepNash R-NaD: a *persistent* reference regularizer is
  what stops self-play forgetting). Kaggle survey: the public RL author's 5 from-scratch
  attempts all ≈0% vs tier-3+ bots → pivoted to hybrid; the #1 LB entry is self-play RL **on
  top of** a critic+search; strongest *published* ML agent = **1-ply search + learned value**
  (val AUC 0.976) → LB 1000+. This doc's thesis is the spine of the live ExIt plan.
- **`PHASE5_NEURAL_VALUE_EXIT_PLAN.md`** — plan to score ExIt search leaves with OrbitNet's
  value head. Central obstacle it named: `SimState` was **positionless**, so the value head
  (which needs geometry) saw out-of-distribution leaves. **Partially executed and now
  superseded** by `STRONGER_EXPERT_SEARCH_PLAN.md`: Phase 2 made `fleet_events` positional and
  reconstructs in-distribution leaves; the pure neural-value *swap* was confirmed to collapse
  the agent (OOD + over-trusted noisy value), so the live plan is a z-scored **blend**, not a
  swap. Kept doc: `STRONGER_EXPERT_SEARCH_PLAN.md`.

---

## Cluster 5 — Apex & Hybrid rule-based agents *(removed 2026-06-11)*

**What they were.** `agents/apex.py` (~2,900 lines, "Apex v17" lineage) was THE
benchmark and training anchor for the entire RL effort 2026-05-10 → 2026-06-09: BC
teacher, ExIt opponent, and eval gate. `agents/hybrid.py` (~2,600 lines) was its
slower predecessor (50–800 ms/step) and the source of most of its machinery.

**Apex design (the compressed record).** A hand-tuned mission planner, merging the
hybrid agent with ideas from a public "1103 peaking bot":
- **World model + per-planet timeline simulation**: parse obs into a `World`;
  for each planet, replay production + every known fleet arrival with exact combat
  resolution (`_simulate_timeline`/`_resolve_combat`) out to horizon 110 — a
  hand-rolled precursor of the public tier's "arrival ledger".
- **Mission-based planning**: capture-neutral / attack / reinforce / recapture /
  snipe / multi-source swarm (ETA-synced) / crash-exploit (strike right after two
  enemy fleets collide) missions, scored by `_target_value` with ~30 hand-tuned
  multipliers (hostile 2.2×, neutral 1.4×, static-target 1.22×, comet 0.85×,
  leader-attack 1.2×, exposed-enemy bonus…).
- **Tight ship margins** (its key efficiency edge vs naive bots): send
  `needed = garrison + prod·ETA + margin` with margin caps 5/7, growing for
  long trips; `_preferred_send` balances overkill vs fleet-speed.
- **Geometry**: waypoint sun avoidance, orbit prediction with intercept search,
  exact comet path-index prediction + remaining-life checks, beam-search opening.
- **Defense**: threat lookahead 28 turns, reserves vs stacked enemy threats,
  proactive defense, time budgeting (soft deadline 0.82×actTimeout).

**Why retired.** The 2026-06-09 leaderboard reality check + the 2026-06-10 arena:
apex ≈ **757 LB ≈ p55** — a *median-tier* anchor. Every week spent optimizing
win-rate-vs-apex optimized against the middle of the field. In the vendored public
pool (n=30/pair, real Kaggle env): apex = **43% 2P, 5% outright 4P**, and the
Producer flow-diff planner beats it ~100%. Producer's *exact counterfactual
flow-diff* (engine-exact projection, `safe_drain` sizing, one unified
offense/defense scorer) structurally dominates apex's heuristic margins — it is
apex's timeline idea taken to its logical conclusion. The producer fork
(`agents/v5/`, LB ~1242, rank ~140/4212) replaced apex as base and gate; the
arena (`scripts/arena.py`) replaced win-vs-apex as the metric; ExIt teachers
re-anchor to the producer tier (see `rl_research/LEADERBOARD_CLIMB_PLAN.md`).

**What survives.** Apex's good ideas already exist, better, in the public-tier
lineage (timeline sim → arrival ledger; snipe timing; crash exploit; comet
path prediction — producer has all of them). The 4P arena disproved the last
reason to keep it (it isn't even a useful mid-tier 4P baseline: 5% outright).
Code recoverable at git history ≤ 2026-06-11. The `kaggle.com` submission
history retains apex's LB datapoints (700.8–774.7).

---

## Cluster 6 — Shot-validator veto on v5 *(closed 2026-06-11, code kept default-off)*

**What it was.** Phase 2.1 of the leaderboard climb: replicate konbu17's public
reject-only shot validator (+19pp on their v4-lineage base) on our producer fork.
24-feature MLP (24→64→32→1, ~2.4K params), trained on 274K dense per-shot labels
("did we own the ray-cast target within [arrival, arrival+10]?") harvested from
400 real-env games among {v5, producer, ow_proto, enders_1000} (incl. mirrors);
veto applied to v5's emitted attack waves, own-planet reinforcements exempt.
The model itself was good: val AUC 0.82 (game-level split), veto precision on
v5's shots 84–95% across thresholds.

**Result: CONFIRMED NEGATIVE, dose-responsive.** Mirror arena vs un-vetoed v5
(side-alternated paired seeds, n=60 per threshold): t=0.10 (19% of attacks
vetoed) → **37%**, t=0.25 (31%) → **41%**, t=0.40 (41%) → **33%**. All at or
below the 40% signal line; heavier veto = worse. Not extended to n≥120 because
the monotone dose-response is exactly the signature of removing real value, not
noise.

**Why it failed here but worked publicly.** konbu's base fires margin-heuristic
shots, many genuinely wasted — a reject-only filter prunes true errors. The
producer flow-diff only fires when the *exact counterfactual flow score* clears
ROI 1.5: its "failed-ownership" shots are mostly deliberate attrition/pressure
trades whose value the ownership label cannot see. The veto therefore
systematically deletes priced-in aggression → passivity → loss. Same failure
family as the weak-rollout opponent and value-leaf blend: **a coarse learned
signal second-guessing a stronger exact planner regresses it** (5th consecutive
instance of this lesson).

**What survives.** `agents/v5/orbit_lite_v5/shot_validator.py` (encoder + veto,
inert without a weights file — v5 byte-identical), `scripts/harvest_shots.py`
(dense per-shot label harvester, reusable for value/aux training data),
`scripts/train_shot_validator.py`, the `v5v[:threshold]` arena spec, and the
274K-shot dataset in `outputs/validator/raw/`. Do NOT ship validator weights
into `agents/v5/orbit_lite_v5/` — their presence auto-enables the veto.

---

## Cluster 7 — Second candidate size per (source, target) *(closed 2026-06-11, code kept default-off)*

**What it was.** Phase 2 Track 1 of the leaderboard climb: attack producer's
known single-size structural limit. `plan_lite_waves` builds exactly one
candidate per (source, target) with size = full `safe_drain`; we added a
config-gated second "just enough to capture" variant (`cheap_capture_margin`):
size = ceil(capture floor at the cheap fleet's own arrival turn + margin),
capped at the drain, never on owned targets. Candidate axis C = S×T → 2·S×T;
the exact flow-diff scorer arbitrates, greedy's one-wave-per-target mask keeps
the sizes mutually exclusive. Implementation clean: off-state verified
move-identical over a full game; on-state 29ms mean step (budget 1s).

**Result: STRUCTURALLY INERT.** Gate arena (margin=4, n=120 per pairing,
side-alternated paired seeds): 45.4% vs plain v5, 53.8% vs producer, steps
margins flat — i.e. an A/A measurement. Instrumented diagnostic explained why:
in a full 246-step game the greedy selector fired 81 attack waves and chose the
cheap variant **0 times**.

**Why the scorer can never prefer it.** Fleet speed grows with size, so the
full-drain candidate is a strict superset in reachability and arrives no later;
earlier capture banks more production inside the horizon; ships sent vs ships
kept home are the same ships to the flow projection; and `safe_drain` already
protects the source's defense needs, so draining has no modeled downside. Hence
score(full) ≥ score(cheap) always, and exact ties break by lower candidate
index = the full variant. The premise "the exact scorer will arbitrate" was
wrong — within producer's no-opponent projection, full drain is provably
(weakly) optimal, so the second size can only ever win when the target's
projected floor *drops* between the two arrival turns (a third-party in-flight
fleet smashing the defender first) — rare in 2P, unobserved in practice.
Margin sweeps cannot fix a candidate that never gets selected; the planned
margin=8 run was cancelled on this finding.

**What survives.** The gated code in `agents/v5/main.py` (`cheap_capture_margin
< 0` default = byte-identical), the gate CSV
(`outputs/arena/gate_cheap_capture_2p.csv` — doubles as a clean n=120 A/A
reference: 45.4%/53.8%), and the lesson: **any "alternative candidate" for the
producer planner must come with a scoring term that can actually distinguish
it** — the flow diff is indifferent to where ships sit and strictly prefers
speed, so candidate-axis extensions need either a modeled opponent response or
an explicit risk/commitment term before they can matter.

---

*Confirmed dead ends are also tracked tersely in `CLAUDE.md` and `memory/`. This file is the
long-form "why" companion to those one-liners.*
