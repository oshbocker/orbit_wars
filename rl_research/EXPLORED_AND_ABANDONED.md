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

## Cluster 8 — Arrival-resolving search horizon (ExIt) *(closed 2026-06-12, code kept default-off)*

**What:** `exit.arrival_horizon` (+ `arrival_settle_margin`, `arrival_horizon_cap`)
in `v2/search.py::_decision_depth` — per-decision sim depth = min(cap, ceil(max
candidate travel time) + margin), uniform across a decision's candidates
including hold, so every candidate's capture resolves at the leaf instead of
evaluating identical to hold (at fixed depth-12 only ~10–17% of candidates
arrived in-horizon; pi' degenerated to the prior on ~5/6 decisions, floss≈ln4).

**Mechanism verification PASSED** (`scripts/diag_arrival_horizon.py`):
hostile-candidate resolution 0%→100% @p50, capture-capable decisions with live
q-spread 35%→65%, fraction-target entropy p10 1.06→0.02. The residual flat
decisions are exact `evaluate_state` cancellations (friendly transfers conserve
ships; failed attacks on the best enemy trade 1:1 in `my − best_enemy`) — i.e.
correct, not horizon-blindness.

**Gate FAILED decisively** (run `v2_exit_producer256_v3_a100`, fresh BC + 40
iters, single variable vs the champion recipe): arena h2h vs
`v2_exit_producer256_a100/ckpt_000025.pt` at n=30/pair = **17% (iter 5), 17%
(iter 35), 3% (iter 40)** — worsening with training; eval vs ow_proto declined
72%→52–57% across iters. Loss signature: median game runs the full 499 steps
and loses on score — overextension/misallocation, not early elimination.

**Why it failed.** The sharpened sizing targets are exactly what the passive
in-sim opponent's world rewards: at depth 30–60 with nobody contesting,
`prod_weight × prod_advantage` makes greedy distant expansion look strictly
good, and the long horizon gives the sim maximal room to diverge from real
play. Distilling those predictions *more faithfully* (the whole point of the
fix) therefore transfers the sim's bias *more faithfully*. The depth-12 "blur"
the fix removed was functioning as implicit regularization: it only let the
search act on short-horizon consequences, which the sim gets right.

**The closed pattern (6 experiments, do NOT reopen the family):** neural value
leaves, every-step rollout opponent, two-player one-ply search, value-leaf
blend, mixed collection pool, arrival horizon — every change that makes the
search lean harder on the passive sim's long-range evaluations regressed the
champion. The one validated search win (Gumbel/Sequential-Halving, +9.4%)
changed *selection/anchoring to the policy prior*, not the world model. A
faithful in-sim opponent (Build 2 `net_opponent`) remains the only principled
escape, but is ~1000× slower and ungated. Salvage variants (cap 20–24) were
considered and skipped: another Colab run for a third-order knob pre-deadline.

**What survives.** The gated code (default-off, bit-identity covered by
`scripts/test_gumbel_search.py`), the probe script, the n=90 h2h CSV
(`outputs/arena/gate_v3_h2h.csv`), and the `evaluate_state` cancellation
insight (1:1 attrition trades are invisible to ship-advantage evals — the same
property that closed Cluster 6).

## Cluster 9 — Defensive symmetry of the reinforce-risk floor (v5.4 Axis A cand a) *(closed 2026-06-13, code kept default-off)*

**What:** `defense_size_beta` in `agents/v5/main.py` + `safe_drain(reserve=)` in
`orbit_lite_v5/planner_core.py`. The shipped v5.3 win (reinforce-risk) inflates
the *offensive* capture floor by `beta·ρ(eta)·enemy_mass` so the planner declines
captures the enemy will reinforce mid-flight. This was the mirror-image hypothesis:
`safe_drain` only protects a source against fleets *already in flight* (the
do-nothing projection), so it should over-commit ships away from planets the enemy
can *launch* at — fix by holding back `defense_size_beta · cheap_enemy_pressure(source)`
ships per source (the same enemy-mass proxy the offensive floor + regroup already use).

**Built clean:** byte-identity 0/555 steps vs the archived v5.3 bundle on fixed obs
streams (`/tmp/byteid_final.py`; the agent is deterministic on fixed obs, but live
games are NOT — env/opponent fp wobble diverges identical code by step 0–2, so the
check must replay a *recorded* obs stream through old and new, the method the
reinforce port used — NOT a live A/A game). Knob ON changed 145/555 steps.

**Gate FAILED decisively, dose-responsively** (`v5:defense_size_beta=X` vs `v5`
mirror, n=120 each, side-alternated paired seeds, `outputs/arena/gate_defense_b*.csv`):
beta **0.5 → 28%**, **1.5 → 10%**, **3.0 → 3%** — monotone in dose, all far below
the 50% mirror parity (A/A floor ±~4.5% @ n=120).

**Why it failed (the asymmetry is real, not a bug).** Declining a *doomed attack*
(offense) frees a known-wasted commitment for better use — a strict improvement.
Holding ships back *defensively* is speculative hoarding: producer's `safe_drain`
is already exact w.r.t. the do-nothing projection (it protects against every
in-flight threat), and `cheap_enemy_pressure` is a crude over-estimate (it credits
ALL reachable enemy garrison as if it could all converge on THIS one planet), so the
reserve over-holds and induces passivity — and our measured weak mode is 2P, where
under-deploying is precisely what loses. This is the **same pattern that closed
Clusters 6 and 8**: a coarse learned/heuristic signal second-guessing an exact
flow-diff planner regresses it. Net read on the *axis*: defensive opponent-reactivity
via reserve-on-drain is a dead direction; the offensive reinforce-risk win does not
transfer to defense.

**What survives.** The gated code (default `0.0` = OFF, byte-identical), the n=120×3
dose-response CSVs, and the byte-identity harness (`/tmp/byteid_final.py`, the fixed-
obs-stream method — reusable for any future v5 knob). Next Axis-A candidate (b) =
short-horizon 1-ply opponent-launch injection (inject the opponent's *actual* best
flow-diff sends into the projection before scoring ours — less coarse than a mass
proxy, but graveyard-risky: keep the injection horizon SHORT, never a rollout).

---

## Cluster 10 — Learned global-value tie-breaker (Axis C / v5.5 ML candidate) *(closed 2026-06-13, code kept default-off)*

**What:** `agents/v5/orbit_lite_v5/value_reranker.py` + the `value_rerank_eps` knob in
`main.py` + the near-tie re-rank branch in `planner_core._greedy_select`. The
project's designated ML/RL-learning lever made competition-relevant: a learned
**global value** model `state -> P(win for the acting player)` used ONLY to re-rank
flow-diff candidates the exact scorer is *indifferent* between — among candidates
whose competitive score is within `value_rerank_eps` of the about-to-be-selected
best (and that independently clear `roi_threshold`), pick the one the value model
rates highest instead of the lowest slot index. Tie-break only, never a primary
scorer, never fires a wave flow-diff wouldn't. Explicitly NOT policy-BC (that
plateaus at 3% vs producer — Cluster 5 / producer256 memory).

**Model + data (all built, reusable):** 16-feature global encoder (step; my/enemy
ships & prod incl. max-enemy; my/enemy/neutral planet counts; in-flight ships;
largest-planet ownership; comet net) run identically at harvest and inference.
`scripts/harvest_values.py` played 360 2P games among {v5, producer, producer_v2}
on the real Kaggle env (162K labelled states, both seats, label = did this seat win
the episode); `scripts/train_value_model.py` trained a 16→32→16→1 MLP exported in
the shot-validator npz layout. **Val AUC 0.783, well-calibrated and monotone across
all ten predicted-deciles** (decile 0 pred 0.001/actual 0.005 … decile 9 pred
1.000/actual 0.997) — the global state IS cleanly separable by win. The trained
artifact is preserved at `outputs/value/value_model_weights.npz`; the dataset at
`outputs/value/raw/`.

**Built clean (byte-identity on fixed recorded obs streams, the Cluster-9 method):**
OFF with no weights = 0/451 differ; weights present but `eps=0` = 0/451 (so a bundled
model alone changes nothing); knob ON (`eps=3.0`) changed only 8/451 steps — i.e. the
re-rank touches ~1.8% of decisions, exactly the rare-genuine-tie behaviour intended.
With the real trained model, `eps=0` control = 0/601, and after the weights were moved
out the shipped v5 is 0/894 vs the pre-Axis-C reference.

**Gate INERT** (`v5:value_rerank_eps=X` vs `v5` = v5.4 control, n=120 each,
side-alternated paired seeds, `outputs/arena/gate_value_eps{2,4,8}.0.csv`):
eps **2.0 → 46.2%**, **4.0 → 47.9%**, **8.0 → 48.3%** — all inside the A/A noise
floor (50% ± ~4.5% @ n=120), pooled ~47.5% over 360 games (≈1σ below 50). Crucially
NOT dose-responsive toward harm: the *wider* the indifference band (more model
influence), the *closer* to 50 — the signature of noise, not a real effect. No eps
gives the ≥60%/clear-margin a ship requires.

**Why it failed (predicted, not surprising).** A globally-grounded value model
(AUC 0.78) is still **too noisy to rank near-equal siblings** — the exact finding of
the Phase-2 ExIt diagnostic (`memory/phase22b-datamax-result.md`, value corr ~0.39
globally but useless for sibling ranking) and the `value_leaf_blend` regression
(Phase-3, 6/6 seeds). When the flow-diff is genuinely indifferent between two
captures, *which* you pick is dominated by downstream contingency the global
aggregates can't see; the model's pick is a coin-flip vs lowest-index. This is the
**same pattern that closed Clusters 6 (shot-validator), 8 (arrival-horizon) and 9
(defensive-symmetry)**: a coarse learned signal second-guessing the exact flow-diff
planner does not help, even when confined to its own indifference band.

**The one untried variant** (and why it's low-odds): a stronger model trained on real
**above-our-tier ladder replays** (Kaggle EpisodeService `GetEpisodeReplay` on 1300+
team episodes; aidensong's public 16-feature GBC hit AUC 0.976 with capture-aware
features). The plumbing (encoder, harvest, train, gated integration) is ready and
data-source-agnostic — swap the harvest for replay states. But the *integration*
ceiling is unchanged: eps=8.0 (most model influence) trended toward neutral, not
positive, so a better ranker is unlikely to convert the near-tie set into wins. Logged
as the only open thread, not a recommended next bet.

**What survives.** The gated code (default `value_rerank_eps=0.0` = OFF, byte-identical
to v5.4 — shipped v5 unchanged), the full harvest→train→bundle pipeline (reusable for
any future global-value experiment, incl. real-replay data), the trained artifact +
162K-state dataset, the n=120×3 gate CSVs, and the byte-identity harness. Net read on
the *axis*: learned-value-for-near-tie-reranking on top of the exact producer flow-diff
is a dead lever locally — confirming PPO/ExIt/policy-BC/value-blend/value-rerank are all
exhausted for the rule-based base. The remaining credible levers are search-free planner
deltas validated by the ladder (the v5.x line) and Axis A cand (b) (1-ply opp-launch
injection).

---

## Cluster 11 — Level-1 opponent-aware planning / 1-ply opponent-launch injection (v5.4 Track 1) *(closed 2026-06-13, code kept default-off)*

**What:** `opp_inject_waves` in `agents/v5/main.py` + `_opponent_reactive_status()`.
The first rung of the opponent-modeling / equilibrium ladder (the new research
direction — see `LEADERBOARD_CLIMB_PLAN.md` + SOTA: 2P Orbit Wars is a two-player
zero-sum **simultaneous-move** game; producer's flow-diff is a one-shot best-response
to a **do-nothing** opponent = level-0, the maximally *exploitable* class). Mirror meta
⇒ we ARE the opponent's planner, so for each live enemy seat we re-parse the obs from
their view (`parse_obs(obs_tensors, player_id=o)` — all but ownership masks is absolute),
run the EXACT producer planner as their level-0 best response to the do-nothing baseline,
inject their top-N attack launches into the projection's arrival buckets
(`record_fleet_arrivals`), re-resolve the engine-exact reactive timeline
(`garrison_status`), and score OUR candidates against THAT instead of the passive world.
The intended generalization of the v5.3 reinforce-risk win (which modeled one reactive
term — mid-flight reinforcement) to the opponent's full 1-ply offensive response.

**Built clean (the principled, graveyard-respecting form):** 1-ply, no rollout; uses the
EXACT planner as the opponent model (not Cluster-9's coarse mass proxy) propagated through
the EXACT engine projection; the 6 mutable projection tensors are snapshotted/restored so
the rolling cache is untouched. Byte-identity (subprocess-isolated, since `planner_core.py`
+ `value_reranker.py` differ from the v5.3 bundle): OFF = **0/515** steps vs the archived
v5.3 ref; ON (waves=3) changes **15/515** (~2.9%). Live smoke DONE/DONE, ~60 ms/step
(~2× planner work, well inside 1 s). Opponent sub-plan = attacks only (regroup models
enemy *defense* → induces our passivity = the Cluster-9 failure mode; the threat that
should reshape our plan is enemy *offense*).

**Gate INERT** (`v5:opp_inject_waves=X` vs `v5` mirror, n=120 each, side-alternated paired
seeds, `outputs/arena/gate_opp_inject_w{1,3,6}.csv`): **waves 1 → 46.7%, 3 → 57.5%,
6 → 56.2%**. **A/A floor for this harness = 55.4%** (byte-identical reinforce b2.2 ref,
`gate_reinforce_b2.2.csv`; the paired-seed setup has a ~+5% first-position skew). So 3/6
plateau +1–2pp over the floor (CIs all contain 55.4%); waves=1 sits *below* it (a biased
partial opponent model is worse than none). Margin metrics show a weak favorable
asymmetry (wins ~10 steps faster than losses; games ~183 steps vs A/A ~230 → genuinely
divergent play, not noise-floor) — but win-rate does not clear the bar a ladder slot
requires (the one ladder-validated win, reinforce-risk, read a clean 75% mirror first).

**Why it failed — a NEW lesson, distinct from the two existing patterns.** Not "coarse
heuristic second-guessing an exact planner" (Clusters 6/8/9/10 — opp_inject doesn't
regress; the model is the exact planner) and not "passive long-horizon sim" (Cluster 8 —
this is 1-ply). The mirror `opp_inject vs v5` IS a direct **level-1-vs-level-0
exploitability test** (we best-respond to v5's actual strategy; the opponent IS v5), and
the best-response beats the base by only ~+2pp. **The flow-diff's `Δnet_me − ΣΔnet_opp`
competitive scorer already internalizes most of the opponent's value, so an explicit
1-ply opponent model is largely redundant against it.** The dose plateau (waves 6 ≈ 3)
confirms redundancy, not insufficient injection. A secondary contributor: the injected
model is the opponent's BR-to-do-nothing, while the real v5 opponent BRs to our actual
in-flight fleets — but more-faithful injection edges toward level-2/fixed-point (the
rollout territory Cluster 8 warns against) for an effect already shown to be small.

**Implication for the rest of the ladder (the direction's main takeaway).** The
redundancy finding *lowers the EV of Track 2* (per-turn regret-matching / equilibrium
mix): if 1-ply best-response is redundant, an equilibrium mix over the same candidates is
unlikely to convert in this meta and shares the mirror-blindness. Track 3 (a direct
exploitability instrument / best-response attacker) is the only principled escalation that
could measure what the mirror can't — but the weak level-1-vs-level-0 read already hints
at low exploitability headroom. Net: the producer flow-diff base looks at/near a local
optimum for hand-buildable solution-concept deltas; the one proven lever (reinforce-risk)
changed the **world model**, and the highest-EV competition move remains meta-monitoring
for the next public *structural* idea + defending v5.3, not mining a locally-optimal base.

**What survives.** The gated code (default `opp_inject_waves=0` = OFF, byte-identical —
shipped v5 unchanged), the n=120×3 gate CSVs, the subprocess-isolated byte-id harness
(`/tmp/t1_byteid.py`), and `_opponent_reactive_status()` itself (a reusable, exact 1-ply
opponent-injection primitive — the substrate for Track 2/3 or any future opponent-aware
experiment).

**Follow-up — off-mirror gate (2026-06-16): the redundancy GENERALIZES; learned head
FOLDED.** The open escape hatch above was a *learned* head modeling the ~⅓ of the top
tier that is non-producer. Step-0 de-risk built the instrument the mirror lacked:
non-producer fixture bots (`agents/external/{half_drainer,swarmer}.py`) + an off-mirror
gate (`scripts/off_mirror_gate.py`) running `v5:opp_inject_waves=N` vs `v5`, **both vs
each archetype**, paired seeds, n=120, doses 1/3/6. v5 wins **100%** every cell (the same
~99% ceiling as the public pool → win-rate blind off-mirror too), so the primary
instrument was paired **steps-to-elimination** (not saturated, SE≈±6). Every paired Δ
(steps and score-margin) sits inside noise, non-monotonic in dose — *smaller* than the
~10-step asymmetry the mirror itself saw. Exact 1-ply injection of a non-producer
opponent's own best-response buys ≈0 → a learned approximation will not convert. Verdict:
**fold the learned opponent-prediction-head track** (caveat: v5 saturates 100% so games
aren't contested; only a contested off-mirror peer — expensive — could revive it). Full
write-up: `rl_research/OFF_MIRROR_INJECTION_FINDINGS.md`.

---

## Cluster 12 — Half-drain reserve cap (`reserve_frac`, v5.4 from the top-tier replay diagnostic) *(closed 2026-06-14, code kept default-off)*

**What:** `reserve_frac` in `agents/v5/main.py` + a post-scoring trim block in
`plan_lite_waves`. Sourced from the strongest *direct* evidence we ever had: the
top-tier replay diagnostic (`TOP_TIER_REPLAY_CORPUS.md`, `[[top-tier-replay-diagnostic]]`)
showed the **#1 ladder agent (Isaiah @ Tufa Labs, 1762) sends ~half a garrison at a time
(median send-fraction 0.52) and beats producer-family full-drain clones**, while
producer/v5 ship the full `safe_drain` (~1.0). Hypothesis: replicate the half-drain as a
gated delta. Implemented as the plan's **post-selection cap** (the form chosen
specifically to be DISTINCT from the two already-closed half-drain altitudes): the exact
flow-diff scorer still ranks **full-drain** candidates (so WHICH waves fire is
producer-identical), then each chosen full-drain wave ships at most `(1 - reserve_frac)`
of its source garrison, re-gating the slower trimmed fleet for reachability + capture-floor
clearance; a send is left full ("decisive") iff the trimmed fleet can't reach in time or no
longer clears the target floor at its later arrival.

**Built clean:** ruff/pyright clean; byte-identity (subprocess-isolated recorded-obs method,
`/tmp/byteid_reserve.py`, since `planner_core.py` differs from the v5.3 bundle): OFF =
**0/191** steps vs the archived v5.3 ref; ON (0.35) changes **74/191** (~39%).

**Gate FAILED decisively, flat across dose** (`v5:reserve_frac=X` vs `v5` mirror, n=120
each, side-alternated paired seeds, `outputs/arena/gate_reserve_frac{0.2,0.35,0.5}.csv`):
**0.2 → 29.2%** (95% CI [21,37]), **0.35 → 33.3%** ([25,42]), **0.5 → 31.7%** ([23,40]).
All three sit ~20pp below the ~55% A/A floor; **every CI is entirely under 50%**. Note the
shape: NOT monotone-down like Cluster 9 (defense reserve: 28/10/3%) — it is **flat-low
~30% at every dose**, which is *more* damning for the idea: no amount of reserve helps, the
"hold ships back at all" structural delta is the problem, dose-independent. Margin metric
confirms the passivity mechanism — the reserving agent's *losses* end faster (158–170
steps) than its *wins*; it gets out-tempoed and eliminated.

**Why it failed — the THIRD distinct closure of "send less than full drain on producer."**
Producer's flow-diff sizes every send at `safe_drain` for a reason: bigger fleets fly
faster (speed grows with `log(ships)`), so a full-drain wave both captures sooner and frees
the source's growth for the next wave; a *post-hoc* cap ships a fleet the scorer sized for
full drain, arriving later (re-gating catches the ones that now miss, but the survivors are
still slower) and stranding the reserve at home, where in a 2P mirror the opponent
out-expands the under-committed agent. The deeper lesson: **Isaiah's half-drain wins
because his *entire planner* is built around it (sizing, snipe timing, follow-up waves) — it
is not a producer with a cap.** Bolting a half-drain onto producer's full-drain scorer is
once again *a coarse modification second-guessing an exact planner* (Clusters 6/8/9/10).

**This closes the half-drain axis at all three altitudes** — the other two are already in
the graveyard: **Cluster 7** (multi-size candidates *in the scorer* — flow-diff provably
prefers full drain, 0/81 cheap picks) and **Cluster 9** (defensive reserve subtracted from
the drain via a mass proxy — 28/10/3%). Cluster 12 is the post-selection cap, the last
remaining altitude. All three fail. "Send less than `safe_drain`" is a dead direction on
the producer base, regardless of where the trim is applied.

**Consequence for v5.4 Steps 2–3 (fingerprint-gated conditioning) — also closed.** The
plan sequenced an offline opponent-type stratification (Step 2) to decide whether a
*conditional* application (apply the half-drain only vs detected passive/clone opponents,
Step 3) earns a build. That hybrid cannot be net-positive here: (i) the mirror IS the
full-drain-clone test (⅔ of the top tier) and the delta loses it at ~30%, so the correct
conditional action vs the majority type is "don't reserve" = **be vanilla v5**; (ii) the
upside would require the delta to *beat* the other ~⅓ (half-drainers/swarms) — but we have
no such opponent vendored to test, and the arithmetic is unforgiving: even a generous
0.67·0.30 + 0.33·0.70 ≈ 0.43 blended ladder rate is a net loss; (iii) for the delta to beat
Isaiah-types it would have to be Isaiah-competitive, which a degraded-producer cap is not.
Fingerprinting's proven-correct altitude (conditioning layer on a *winning* delta) has no
winning delta to condition on. Step 1's decisive kill ends the v5.4 reserve_frac line.

**What survives.** The gated code (default `reserve_frac=0.0` = OFF, byte-identical — shipped
v5 unchanged), the n=120×3 gate CSVs, and the reusable subprocess byte-id harness
(`/tmp/byteid_reserve.py`). **Fallback per plan = defend v5.3 + meta-monitor the daily
`scripts/replay_pulse.py` pulse** for the next public *structural* idea (the proven
discovery channel — how reinforce-risk was found). The diagnostic's other open thread
(real-outcome value head on 1500+ replays) is the Cluster-10 variant, Colab-scale.

---

*Confirmed dead ends are also tracked tersely in `CLAUDE.md` and `memory/`. This file is the
long-form "why" companion to those one-liners.*
