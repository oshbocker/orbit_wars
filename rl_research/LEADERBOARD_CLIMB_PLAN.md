# Leaderboard Climb Plan — 2026-06-09

**Where we are (updated 2026-06-13):** rank **325 / 4386** ("Oshbocker", team
score **1201.6** = v5.3). Active slots: **v5.3** (`v5_bundle`, sub 53615604,
**1201.6** @21:05) + **v5.2 control** (`v5_2_bundle`, sub 53615608, **1051.1**
@21:06). Final submission deadline **2026-06-23** — 10 days left.

## Update 2026-06-13 (research) — NEW DIRECTION: climb the opponent-modeling / equilibrium ladder (solution concept, not heuristics)

**Reframing (why the "out of ideas" read was wrong).** All 10 graveyard clusters
attacked the planner in exactly two ways, both now closed: (i) *additively* — bolt a
coarse learned/heuristic signal on top of the exact scorer (shot-validator C6, value
re-rank C10, defensive-reserve C9, cheap-capture C7) → "coarse signal second-guessing
an exact planner regresses it"; (ii) *extend the passive world model* — make search
lean harder on a do-nothing sim over a longer horizon (neural leaves, rollout opp,
arrival-horizon C8, value-blend) → "lean harder on the passive sim → lose." The
untouched axis is the **solution concept itself**.

**SOTA grounding (2P Orbit Wars = two-player zero-sum simultaneous-move
perfect-information game — a named, well-studied class).** Producer's flow-diff is a
one-shot **best response to a do-nothing opponent** = the *maximally exploitable* end of
the spectrum (the DUCT/greedy class). Canonical results:
- DUCT/best-response is most exploitable; **regret-matching / Online Outcome Sampling
  converge toward (coarse) correlated/Nash equilibria → provably less exploitable +
  more robust head-to-head** (Bošanský/Lisý/Lanctot/Winands, *Algorithms for Two-Player
  Simultaneous Move Games*; *Convergence of MCTS in SM Games* arXiv 1310.8613 —
  ε-Hannan-consistent selection → subgame-perfect ε-Nash).
- 2024 SOTA **NN-CCE** (arXiv 2406.10411): replace per-node UCB best-response with a
  no-regret learner (EXP-IX) approximating a coarse correlated equilibrium over the
  stage game → beats best-response/PSRO/CFR 62–91%.
- **Cognitive hierarchy / level-k** (Camerer et al.): level-k best-responds to level-(k-1).
  Producer is **level-0** (assumes do-nothing opponent).
- RL+search recipe (**ReBeL** arXiv 2007.13544; DeepMind **Player of Games**):
  depth-limited solving *with the opponent moving*, value at leaves, iterate to equilibrium.

**The bridge to our one proven win.** The reinforce-risk floor (v5.3, +150 ladder, 75%
mirror) was a *partial level-0→level-1* step: it taught the planner the opponent
reinforces during flight. In a mirror meta where everyone forks producer, "less
exploitable than the shared base" IS the differentiator — which is exactly why it won.
The rest of the level-k / equilibrium ladder is unclimbed.

### Update 2026-06-13 (later) — Track 1 BUILT + gated INERT → graveyard Cluster 11

`opp_inject_waves` shipped end-to-end (`_opponent_reactive_status()` in
`agents/v5/main.py`: per enemy seat, run the EXACT planner as their level-0 BR →
inject their attack launches into the projection → re-resolve `garrison_status` →
score ours against the reactive world). Byte-identity OFF 0/515 vs the v5.3 ref;
ON (w3) 15/515 (~2.9%); smoke ~60 ms/step. **Gate (`v5:opp_inject_waves=X` vs `v5`,
n=120 mirror): waves 1→46.7%, 3→57.5%, 6→56.2% vs the 55.4% A/A floor → INERT**
(3/6 plateau +1–2pp over floor, all CIs contain it; w1 below). Weak favorable margin
(wins ~10 steps faster; games ~183 vs A/A ~230) but no ladder-slot-worthy win.
**Key finding:** the mirror IS a level-1-vs-level-0 exploitability test, and the BR
beats the base only ~+2pp ⇒ **the flow-diff's `Δnet_me − ΣΔnet_opp` scorer already
internalizes most opponent value; a 1-ply opponent model is redundant against it.**
This **lowers Track 2's EV** (equilibrium mix over the same candidates ≈ same
redundancy + mirror-blindness) and leaves **Track 3** (exploitability instrument) as
the only principled escalation — though the weak read hints at low headroom. Net:
producer base looks ~locally-optimal for solution-concept deltas; the proven lever
(reinforce-risk) changed the world model. Full why = `EXPLORED_AND_ABANDONED.md`
Cluster 11. Code kept gated default-off; `_opponent_reactive_status()` is a reusable
exact 1-ply opponent-injection primitive.

### Update 2026-06-13 (later) — World-model gap audit → NO GAP (the base projection is faithful)

After Track 1, audited the chosen "world-model gap" axis directly
(`/tmp/proj_audit.py`, reusable): roll the v5 runtime over 10 real games vs producer,
capture what `garrison_status` (the do-nothing projection) predicts for every planet at
every horizon k, compare to the ACTUAL board k steps later (ground truth = the obs
stream). **Decisive finding: the projection is physically faithful** — owner accuracy
99.8% @k=1 → 87.8% @k=18 (β=2.2), and the *pure geometry* miss category (projection said
my fleet captures it, but it ends up neutral = my fleet vanished to sun/sweep/aim) is
**0.2%**. Orbital sweep, sun-crossing, combat ordering, intercept aim are all already
exact. The residual error is entirely **future AGENCY**: ~50–59% = my OWN future captures
the single-turn projection can't foresee (only recoverable by multi-turn lookahead =
the 6-failure search pattern); ~28–33% = ENEMY agency (reinforce-risk shipped the prime
slice; opp_inject showed the rest is scorer-redundant); ~13–17% = neutral↔enemy churn
(mostly 4P third-party races). Cross-check: reinforce ON vs OFF — ON makes the agent's
play track its own projection BETTER (owner acc +3.6pp @k=18), validating the instrument
and reinforce-risk's mechanism. **Conclusion (triangulated 3 ways — opp_inject inert,
audit faithful, whole graveyard): the producer base is at/near a LOCAL OPTIMUM for
hand-buildable + solution-concept deltas; the only proven lever changed the WORLD MODEL,
and the world model has no remaining renderable gap.** Remaining real levers: (a) defend
v5.3 + meta-monitor for the next public *structural* idea (the proven channel = how
reinforce-risk was found); (b) multi-turn search WITH a faithful opponent + learned value
(ReBeL/Player-of-Games style) to recover the ~50% own-future-agency gap — now *justified*
by the audit but a major build + graveyard-adjacent (the passive-sim search family is
closed; only a faithful in-sim opponent escapes it). 4P churn (b/c our strength) is
ladder-only measurable.

### New plan — a ladder of opponent-aware solution concepts (each a gated single variable, mirror-gateable, no rollout, no passive-long-horizon sim)

**Track 1 (PRIMARY, building now) — Level-1 opponent-aware planning** (= the old Axis A
cand (b), now SOTA-justified and top priority). Mirror meta ⇒ we ARE the opponent's
planner: run producer from each live enemy seat → their top-K best-response launches →
inject into the projection → score OUR candidates against that reactive world instead of
the do-nothing one. Single knob `opp_inject` (on/off, + K). Bounded cost (~2× planner
calls/turn; producer runs ~20–35s/*game*, budget is 1s/*step*). Sidesteps every closed
pattern: 1-ply (no rollout), uses the *exact* planner as opponent model (not Cluster-9's
coarse mass proxy). Gate: `v5:opp_inject=on` vs `v5` mirror, n≥120; dose-response on K.
Risk = over-reactive projection → Cluster-9 passivity; same gate catches it.

**Track 2 (if T1 gates >60%) — per-turn stage-game equilibrium.** Replace
"fire greedily while score>ROI" with the NN-CCE/OOS recipe at one-turn scale: my
candidate set × opponent candidate set → payoff matrix via the exact flow-diff scorer →
N rounds of **regret matching** → (possibly mixed) less-exploitable turn strategy →
select. Single-turn, no horizon extension. Gated, mirror-gateable.

**Track 3 (parallel; new measurement instrument + RL-learning capstone) — direct
exploitability.** Fixes "mirror A/B is our only, noisy instrument." Exploitability =
how well a best-response attacker does against you; a level-1 exploiter (producer that
reacts to committed moves) vs candidate measures robustness directly and orthogonally to
mirror win-rate (T1's `level-1 vs level-0` gate already IS this). Only AFTER T1/T2 does a
learned leaf value get a principled home (ReBeL-style value on the *opponent-aware* short
search — value failed every time in the *passive* setting, which the literature says is
the wrong setting). Honest: graveyard-adjacent; gate hard.

**Slot/time discipline unchanged:** ~10 days, ~4 A/B cycles. T1 gates locally before any
ladder slot; never act on n<100; every ship pairs with the incumbent resubmit. Sources in
the conversation; full reasoning in `EXPLORED_AND_ABANDONED.md` (closed patterns) +
`MEMORY.md`.

## Update 2026-06-13 — Axis 0 MEASURED (→ Axis A) + slot decision + v5.4 cand (a) built/gating

**Slot hygiene (Task 1): NO submission today — keep the warm v5.3 (1201.6)
defending.** The active-2 eviction rule auto-evicts the OLDER incumbent, and v5.2
(21:06) is *newer* than v5.3 (21:05), so any resubmit today would cold-restart our
hard-won 1201.6 v5.3 (ratings don't carry over) for zero benefit — the A/B needs
both agents cold same-epoch at v5.4 ship anyway. **Team rank is driven by the best
active sub** (LB shows us at 1201.6 = v5.3), so the passive v5.2 slot doesn't drag
rank; it stays harmlessly until v5.4. **At v5.4 ship: submit v5.4, then resubmit
v5.3** → active {v5.4, v5.3} clean same-epoch A/B (v5.2 evicted naturally). The
exact v5.3 tarball is archived as `outputs/submissions/v5_3_bundle.tar.gz` (==
on-ladder v5.3, verified contains `reinforce_size_beta=2.2`).

**Axis 0 (Task 2): measured our mode-conditional rating — VERDICT = proceed with
Axis A; Axis B (4P spec) RULED OUT.** Pulled v5.3's 86 ladder episodes via the
Kaggle EpisodeService (`/api/i/competitions.EpisodeService/ListEpisodes`,
`{"submissionId": 53615604}` — each episode carries every agent's match-time
`initialScore`, so no LB join needed). Split by `num_agents`, fit 2P Elo + 4P
Plackett-Luce winner model vs opponents' match-time ratings:

| mode | n | winrate | mean opp rating | fitted perf |
|---|---|---|---|---|
| 2P | 46 | 50% (23/23) | 1174 | **1186** |
| 4P | 39 | 41% (17/40, baseline 25%) | 1146 | **1290** |

4P − 2P = **+106** (bootstrap 90% CI [−23, +224]; P(4P ≥50 *below* 2P, the Axis-B
trigger) = **2.2%**; P(4P>2P) = 89.5%). **Our 4P is our STRENGTH, not our
weakness** — opposite of the "RL self-play agents are 2P-tilted" prior. Cross-check:
the weaker v5.2 control fits 2P 1029 / 4P 1094 (≈ its 4P baseline, as expected for a
~random-4P agent). 2P (1186) is our weaker mode — which is exactly what **Axis A
(2P opponent-reactivity)** targets. (Tooling: `/tmp/fit_modes.py`, `/tmp/boot_modes.py`,
raw `/tmp/ow_episodes_raw.json`. Mode mix 53/47, matches whymelabs' 51/49.)

**Axis A (Task 3): v5.4 candidate (a) = DEFENSIVE SYMMETRY — BUILT + byte-identity
verified + gating.** The proven v5.3 reinforce-risk inflates the *attack* capture
floor for captures the enemy can reinforce mid-flight; symmetrically, `safe_drain`
only protects a source against fleets *already in flight* (do-nothing projection),
so it over-commits ships away from planets the enemy can *launch* at. New knob
`defense_size_beta` (default **0.0 = OFF, byte-identical**) subtracts
`defense_size_beta · cheap_enemy_pressure(source)` from each source's drain —
reusing the *exact* enemy-mass proxy the offensive floor + regroup gradient already
use (its distance decay encodes reaction timing, no separate ρ). Edits:
`agents/v5/main.py` (config field + reorder enemy_mass before `safe_drain` + reserve
plumbing) and `orbit_lite_v5/planner_core.py::safe_drain` (optional `reserve` arg).
ruff/pyright clean. **Byte-identity: 0/555 steps differ** vs the archived v5.3 bundle
on fixed obs streams (`/tmp/byteid_final.py`; agent is deterministic on fixed obs but
the live game is NOT — env/opponent fp wobble — so the check replays a *recorded* obs
stream through both, the same method the reinforce port used). Knob ON (beta=3.0)
changes 145/555 steps. **Gate FAILED decisively, dose-responsively**
(`v5:defense_size_beta=X` vs `v5` mirror, n=120 each, `gate_defense_b*.csv`): beta
**0.5 → 28%**, **1.5 → 10%**, **3.0 → 3%** (monotone in dose, all ≪ 50%). **Candidate
(a) CLOSED** → graveyard Cluster 9. The asymmetry is real: declining doomed *attacks*
(offense) frees a known-wasted commitment; hoarding ships *defensively* over-reserves
(producer's `safe_drain` is already exact w.r.t. in-flight threats, and
`cheap_enemy_pressure` over-credits reachable enemy mass) → passivity, fatal in 2P
(our weak mode). Same "coarse signal second-guessing an exact planner" pattern as the
shot-validator (Cluster 6) and arrival-horizon (Cluster 8). Code kept gated default-off
(byte-identical). **Next: Axis-A candidate (b) = short-horizon 1-ply opponent-launch
injection** — inject the opponent's *actual* best flow-diff sends (we ARE the
opponent's planner) into the projection before scoring ours. Less coarse than a mass
proxy (uses the real planner, not an estimate), but graveyard-risky: keep the
injection horizon SHORT (1-ply, never a rollout). Heavier build (~run the planner from
each enemy seat + inject launches + re-score + byte-identity + gate) — recommend a
focused pass.

**reinforce_size_beta sweep (cheap diligence before cand b): NULL — keep 2.2.**
We inherited beta=2.2 from V2 upstream without sweeping it ourselves; gated
`v5:reinforce_size_beta∈{1.5,2.2,3.0,4.0}` vs `v5` (=2.2) mirror, n=120 each
(`gate_reinforce_b*.csv`). Result: 1.5→49.2%, **2.2→55.4% (A/A reference, BYTE-
IDENTICAL agents!)**, 3.0→47.5%, 4.0→40.8%. The A/A reading of 55.4% (CI
[46,64]) calibrates the noise: at n=120 in a *divergent* mirror the seat/seed
skew is ~±9% — so no candidate clears the A/A baseline (1.5/3.0 sit *below* it),
and only 4.0 is genuinely worse (over-conservative). **2.2 is near-optimal; no
v5.4 candidate here.** Confirms micro-tuning of the proven term is exhausted —
the remaining levers are cand (b) (1-ply opp injection) and Axis C.

**Axis C corpus identified (Task 4, parallel/Colab v5.5):** `kaggle kernels output
slawekbiel/am-i-in-the-top-10-replays-yet` yields only episode *summaries*
(num_players/winner/teams/rewards, `/tmp/top10_replays/cache/*.csv`), NOT replay
states — the per-step states for the 16-feature global-value dataset come from the
`kaggle/orbit-wars-episodes-YYYY-MM-DD` datasets (large, daily) that the kernel itself
downloads, or `GetEpisodeReplay` on top-team episode IDs. Confirms Axis C is a
Colab-scale build, not inline. The Kaggle EpisodeService access pattern is now proven
(see Axis 0) and reusable for harvesting top-team episode IDs.

### Update 2026-06-13 (later) — Axis C BUILT end-to-end + gated INERT → graveyard Cluster 10

**Built and gated locally without waiting on Colab/replay download** (local
strong-agent self-play is the relevant tie-break distribution): full pipeline shipped —
`orbit_lite_v5/value_reranker.py` (16-feature global encoder + numpy/torch MLP, shared
harvest+inference), `value_rerank_eps` knob (default 0.0=OFF, byte-identical), near-tie
re-rank in `_greedy_select` (fail-safe: only among candidates within eps of best that
already clear roi_threshold), `scripts/harvest_values.py` + `scripts/train_value_model.py`.
Model: 360 2P games {v5,producer,producer_v2} → 162K states → **val AUC 0.783,
calibrated/monotone** across all deciles. Byte-identity verified (OFF 0/451; eps=0 with
weights 0/451; ON eps=3.0 changes 8/451; shipped v5 0/894 vs ref after weights removed).

**Gate INERT** (`v5:value_rerank_eps=X` vs `v5`, n=120 mirror, paired seeds,
`outputs/arena/gate_value_eps{2,4,8}.0.csv`): **eps 2→46.2%, 4→47.9%, 8→48.3%** — all
inside the A/A floor (±~4.5%), pooled ~47.5%/360, NOT dose-responsive toward harm (wider
band ⇒ closer to 50 = noise). The learned global value (grounded but noisy) cannot beat
the flow-diff's own lowest-index tie-break on genuine ties — exactly the Phase-2 ExIt
diagnostic + value_leaf_blend failure, and the Cluster 6/8/9 "coarse signal second-guessing
the exact planner" pattern. **Axis C = DEAD locally → Cluster 10.** Shipped v5 unchanged
(code default-off). The pipeline + trained artifact (`outputs/value/`) survive; the only
open thread (low-odds) = retrain on real 1300+ ladder replays via EpisodeService, but
eps=8 trending to neutral (not positive) says a better ranker won't convert the tie set.
**Net: PPO/ExIt/policy-BC/value-blend/value-rerank all exhausted — the credible levers
left are ladder-validated planner deltas (v5.x) + Axis A cand (b).**

## Update 2026-06-12 (night) — after v5.3: the v5.4-vs-v6.0 decision

**Decision: v5.4, not v6.0.** Keep the shipping discipline that is working
(producer base + gated single-variable deltas + mirror gate + paired ladder
A/B) — but aim it at *structural* axes, not micro knobs (those are
exhausted: every public knob claim failed the mirror; reinforce-risk won
because it changed what the planner *models*, not a constant). A true v6.0
(new architecture / standalone RL agent) is negative-EV with 11 days left:
the graveyard holds 6 failed search surgeries + every big-swing RL attempt,
and the one validated instrument we own (producer-mirror A/B) only measures
deltas on the flow-diff base.

**Evidence pulled tonight:**
- The Producer V2 went 4 → 58 votes in one day — the clone wave is forming.
  v5.3 = parity+ with V2 (56%); the next tier requires deltas clones lack.
- Whyme Labs mode-split analysis (measured over 472 fresh episodes
  2026-06-11): the matchmaker is **51% 2P / 49% 4P** — half our rating is
  4P. Their Producer-family fork with a "player-count fix + 4P closing
  logic" fits **~1320 in 4P** vs ~1177 in 2P (+100 over its blend). Method
  is replicable: pull own episodes via the Kaggle episode API, split by
  num_agents, fit 2P Elo + 4P Plackett-Luce vs opponents' LB ratings.
  (Also: team-merger deadline 06-16 — they offer a 4P-specialist merge;
  noted, not pursued.)
- Top-10% replay corpus = daily output of kernel
  `slawekbiel/am-i-in-the-top-10-replays-yet` (refreshed tonight 23:27);
  pull via `kaggle kernels output`.

### The three v5.4+ axes, in priority order

**Axis 0 (first, cheap, decisive): measure OUR mode-conditional rating.**
Replicate the whymelabs fit on our own episodes. If our 4P fitted rating
lags 2P by ≥50 → 4P is the biggest lever and Axis B jumps the queue; if
balanced → stay on Axis A. This converts "where does rating leak?" from
guess to measurement in ~half a day.

**Axis A (default): extend the opponent-reactivity axis in 2P.** V2's
reinforce-risk is the first planner-side opponent-model term and the first
mirror win; the planner still assumes do-nothing opponents everywhere else.
Candidates, each a gated knob, each mirror-gateable vs producer_v2 (the new
reference opponent), n≥120:
  a. **Defensive symmetry**: the same reinforcement-risk logic on the
     defense side — keep_needed/defense candidates currently ignore enemy
     mass that can reach OUR planets (we decline doomed attacks now, but
     still under-garrison planets the enemy can mass on).
  b. **1-ply opponent launch injection**: compute the opponent's top
     flow-diff candidate sends (we ARE the opponent's planner — mirror
     meta) and inject them into the projection before scoring ours.
     Bounded cost; honest caveat: graveyard says don't lean on long-range
     sim — keep the injection horizon short.

**Axis B (if Axis 0 says 4P-weak): 4P specialization.** Player-count fixes
+ "4P closing logic" analogues (whymelabs' words; reverse-engineer:
elimination-order awareness, don't-feed-the-leader endgame, crash
exploitation). Ladder-only measurable → costs an A/B cycle per try; only
enter with Axis-0 evidence.

**Axis C (parallel, Colab; the ML/v5.5 candidate): learned value re-ranker
from top-10% replays.** NOT BC-the-policy (BC clones plateau below their
teacher: producer-BC hit 3% vs producer) — instead the publicly-working
pattern: global value model (aidensong: 16 features, GBC, AUC 0.976)
trained on the 1300+ replay corpus, used ONLY to re-rank near-tie flow-diff
candidates (respects our sibling-ranking finding). Gated default-off,
mirror-gateable. This is also the project's RL-learning goal made
competition-relevant.

**Slot budget: ~4 A/B cycles to 06-23.** v5.3 verdict (06-13/14) → v5.4 =
best Axis A/B winner (06-14/15) → v5.5 = re-ranker if it gates (06-16/18) →
1 reserve cycle for reverts/late finds. Every ship pairs with the incumbent
resubmit; never act on n<100.

## Update 2026-06-12 (later) — public-meta refresh #2 + the v5.3 plan

**v5.2/v5.1 in flight** (submitted 13:52/13:53, judge 06-13/14). Local mirror
check of the pair (n=60, seeds 20000–20059): 40%/60% headline, but only **7/60
games reached step 460** (terminal phase active; v5.2 won 2/7) — 53/60 games
were byte-identical A/A, so the read is noise-dominated as predicted.
Ladder remains the only judge for the terminal phase.

### New public meta (5 notebooks pulled 06-12, /tmp/kpull2/)
1. **The Producer V2 (slawekbiel, the ORIGINAL author, published 06-12)** — the
   one genuinely new structural idea since the flow-diff itself: an **ETA-aware
   enemy-reinforcement risk term on the capture floor**. `cheap_enemy_pressure`
   = distance-decayed enemy garrison mass that can reach each target within the
   horizon; capture floor inflated per arrival turn by
   `reinforce_size_beta (2.2) × ρ(eta; free=3, scale=12) × reachable_enemy_mass`.
   Fixes the known blind spot that the ROI model ignores reinforcement arriving
   *during flight*. Other V2 deltas: waves 6, roi 1.5, min_ships_to_launch 4.
   No stated LB score. **This will become the next clone base** — being early
   matters under tier compression.
2. **I'M SMARTER (tamrazov, claims 1350+):** real code = dynamic phase config
   (step<80: roi −0.2, min_launch −1; step>400: roi +0.2, def targets +2),
   time-budget degradation (shrink lanes/targets/waves when overage <5s),
   `high_prod_attack_bonus=0.20`, comet_attack_bonus 3/4. Claimed anti-overkill
   + vulture sniping are NOT in the posted code. roi 1.25 / waves 8 / horizon 20.
3. **ProducerLite Micro-Logistics (pilkwang, 63 votes):** multi-size candidates
   `[0.33, 0.66, 1.0]` — same family as our **Cluster-7 closure (keep closed)**;
   no win-rate evidence offered.
4. "No-lag 1300+" notebook = wrapper hype (external private agent). Whyme Labs
   mode-split = meta-analysis (their producer-family fork: ~1177 2P / ~1320 4P —
   confirms 4P is where producer-family rating headroom is).
5. Old GitHub issue #1047 ("critical engine bugs", May): nothing actionable —
   multi-attacker erasure is the documented combat rule; coords verified ours.

### v5.3 plan (build NOW while the ladder experiment runs)
**Primary candidate: v5.3 = (v5.2/v5.1 ladder winner) + Producer-V2
reinforcement-risk floor.** Unlike the terminal phase / ffa knobs, this delta
is **locally measurable** — it changes capture decisions from turn 1, so the
producer-mirror A/B (our only sensitive instrument) can gate it BEFORE we
spend a ladder slot. It also targets exactly the matchup that now decides the
tier (producer-family mirrors), from the strongest possible source.

1. ✅ **Vendored `producer_v2`** (2026-06-12): `agents/external/producer_v2/`
   = V2's main.py verbatim, sharing producer/'s `orbit_lite` (file set +
   imports verified identical; one sys.path entry so producer-vs-producer_v2
   games don't race on the module name). Registered in the external loader;
   `load_named_agent("producer_v2")` works; real-env smoke game DONE/DONE.
   Diff vs our vendored producer main.py is SMALL: (i) the reinforcement-risk
   wiring + 3 knobs (beta 2.2 / eta_free 3 / eta_scale 12); (ii) V2 drops
   i-m-better's ffa-bonus block and CONFIG_4P overrides (V2 4P keeps roi 1.5
   / min_ships 4; i-m-better had 1.55/5.0 + ffa bonuses). The `capture_floor
   (reinforcement=...)` plumbing + `reinforcement_timing_factor` already
   existed in our vendored orbit_lite — V2 only wires it.
2. ✅ **Ported to `agents/v5/`** (2026-06-12): `reinforce_size_beta=0.0`
   default-OFF + `reinforce_eta_free=3.0` / `reinforce_eta_scale=12.0`;
   enemy-mass proxy hoisted and shared with the regroup gradient (as in V2).
   ruff+pyright clean. **Byte-identity verified** (old-vs-new agent replayed
   on identical obs streams, 2 full games, 0/328 steps mismatched). ON-path
   smoke via arena spec `v5:reinforce_size_beta=2.2`: runs, diverges.
3. ✅ **Local gates PASSED decisively (2026-06-12, arena.csv, paired seeds,
   side-alternated):**
   a. `v5:reinforce_size_beta=2.2` vs `v5` mirror: **75% @ n=120** — the
      FIRST mirror-measurable win of the project (every prior public mirror
      knob read 42–52%); far outside the noise floor (±~4.5% @ n=120).
   b. `producer_v2` vs `v5`: **78% @ n=60** — the new public base crushes our
      shipped agent; when V2 clones flood the ladder, the current base loses
      its tier. Urgency confirmed.
   c. `producer_v2` vs `producer` (V1): **73% @ n=60** — V2 > V1 head-on;
      the three reads are mutually consistent (the reinforce term is the
      driver).
   d. `v5:reinforce_size_beta=2.2` vs `producer_v2`: **56% @ n=60** —
      parity-or-better with the new public base (within noise of 50%; the
      clamp/terminal-phase deltas are ladder-only visible, so mirror parity
      is the expected good outcome). v5.3 ≥ V2 with our ladder deltas on top.

   **v5.3 ship decision (pending only the 06-13/14 terminal-phase verdict):**
   flip `reinforce_size_beta` default 0.0 → 2.2 in `agents/v5/main.py`
   (CONFIG_4P inherits via `dataclasses.replace`, same as terminal phase) on
   top of the v5.2/v5.1 ladder winner; archive the incumbent tarball before
   rebuilding; submit v5.3 + resubmit the incumbent per the pairing rule.
   Do NOT ship early — it would evict half of the live terminal-phase A/B
   one day before its verdict.
4. **Ship decision when the terminal-phase verdict lands (06-13/14):**
   base = ladder winner; add reinforce-risk if mirror-positive (>60% @ n=120,
   or clear margin signal). ~~Fallback: 4P ffa bonuses 0.035/0.08~~ **MOOT —
   build discovery 06-12: v5's CONFIG_4P already ships
   ffa_leader_attack_bonus=0.035 / ffa_target_prod_bonus=0.08, inherited from
   the i-m-better base we vendored.** The "convergent constants in two forks"
   converged because they're IN the shared base. No fallback ship; if
   reinforce-risk fails locally, hold the slot for the RL track or the next
   meta find.
5. Deprioritized: tamrazov phase-config knobs (every public mirror knob has
   failed our producer mirror — only port if a mirror A/B is free), time-budget
   degradation (we've never been near the overage limit), micro-logistics
   multi-size (Cluster 7 stays closed).
6. **RL track unchanged and parallel:** BC from slawekbiel's top-10% replay
   dataset (1400+-tier teachers).

## Update 2026-06-12 — clamp ladder A/B VERDICT: KEEP. Next ship = v5.2

### The experiment's final conclusion
**The endgame horizon clamp is kept.** After ~24h of same-epoch episodes (both
bundles resubmitted 06-11 16:00–16:45), v5.1 (clamp-only) = **1191.1** vs the
byte-identical-base producer resubmit = **1131.3** → clamp = **+59.8** in the
only instrument that can see it. Combined with v5.0's history (clamp + 4P mult
= 1159.7, −71 vs producer), the attribution is now clean: **the 4P
nearest-opponent mult was the regression; the clamp is positive (or at worst
neutral) on the ladder.** Yesterday's 6h read (1178.7 vs 1176.1, "tied") was
pre-convergence — the spread emerged with episode volume, exactly per
Tamrazov's noise warning.

Secondary finding — **tier compression is real and fast**: the *same*
producer_bundle that converged 1230.9 in the 06-10 epoch converged ~1131 in
the 06-11 epoch (−100 in two days), while rank bands held (1230 ≈ rank 153,
1300 ≈ top 79, 1400 ≈ top 37). The producer-family population is flooding in
(ProducerLite went public 06-11) and squeezing the rating out of the shared
base. Standing still = drifting down; only deltas the clones don't have
(endgame, 4P, ML edge) hold rating.

### Decision rule updates
- **Pinned-baseline slot switches from producer to v5.1.** The clamp is the
  new incumbent base; producer's slot has served its purpose (the A/B is
  decided). Every future ship = candidate + v5.1-incumbent pair, giving each
  new delta a clean same-epoch single-variable A/B.
- Slot budget: ~5 ladder A/B cycles left before the deadline (each needs
  ~1–2 days to converge). Spend them on single-variable producer-family
  deltas, best-validated first.

### Next ships (in order)
1. **v5.2 = v5.1 + `terminal_phase_turns=40`** (port already BUILT + smoked
   2026-06-11, default-off). To ship: flip the default in
   `agents/v5/main.py` (CONFIG_4P inherits via `dataclasses.replace` — exp59
   runs it in both formats), archive the v5.1 tarball
   (`outputs/submissions/v5_bundle.tar.gz` → `v5_1_bundle.tar.gz`) BEFORE
   rebuilding, `scripts/build_v5_bundle.py`, submit v5.2, then immediately
   resubmit the archived v5.1 → active = {v5.2, v5.1}, judge 06-13/14.
   Decision rule: v5.2 ≥ v5.1 → terminal phase stays; else revert default.
2. **v5.3 = winner + 4P ffa bonuses (0.035/0.08)** — the convergent public
   constants (two independent forks), gentler than the regressed 1.25×/0.55×
   mult. Ladder-only measurable; same pairing protocol.
3. **RL track PIVOT (Phase 2.2c arrival-horizon FAILED 06-12** — h2h 17/17/3%
   vs champion, worsens with training; 6th search-surgery failure, the
   search-surgery pattern is CLOSED — only selection/anchoring (Gumbel) ever
   won). New RL bet = **BC from the top-10% replay dataset** (slawekbiel's
   daily dataset, ~4.8K replays/snapshot): filter episodes of 1400+ teams,
   map their moves into the v2 action space (existing
   `_map_expert_moves_to_v2`, 45° angular matching), BC-pretrain embed-256,
   gate in the arena vs pool + champion + v5. Rationale: imitation is the one
   ML pattern that has worked here (BC-from-producer cloned its tier); this
   swaps the teacher from producer (1230-epoch, now 1131) to the 1400–1700
   tier — above everything we can write by hand. Stretch unchanged: 16-feature
   global value from the same episodes for near-tie re-ranking.

## Update 2026-06-11 (late) — ladder check + public-meta refresh

### Ladder state
- v5.1 1178.7 vs producer 1176.1 — statistically tied mid-climb; **the clamp A/B
  verdict needs ≥1 day of episodes (judge 06-12/13)**, per Tamrazov's noise warning.
- Score → rank bands (full LB CSV, 4307 teams): **1230 ≈ rank 156, 1300 ≈ top 75,
  1400 ≈ top 37, 1500 ≈ top 13**; #1 = 1721 (Jake Will). The 1450–1580 band is
  newly crowded (12 teams) — the public tier has climbed since 06-09.

### Public-meta refresh (6 new notebooks pulled + analyzed, /tmp/kpull/)
The producer base went *mainstream*: pilkwang published **"ProducerLite: Flow-Diff
Submission"** (60 votes, variant `exp59`) on 06-11. Expect the ladder to saturate
with producer-family clones → our converged rating at that tier will compress, and
**the differentiator becomes (a) winning producer-family mirrors and (b) 4P**.

Findings, by actionability:
1. **Terminal-phase config (pilkwang exp59, NEW idea):** at step ≥ 460
   (`terminal_phase_turns=40`): `roi_threshold 1.5→1.0`, `max_waves 7→8`, regroup
   OFF. Complementary to our exact-horizon clamp (theirs loosens ROI, ours makes
   the flow-diff exact wrt termination). exp59 of a long tuning series by the
   lineage author — likely load-bearing. **Port gated default-off; ladder-only
   measurable** (mirror games end by elimination step 106–282, pre-terminal).
2. **Conservative mirror tuning (counter-producer, shionao7):** 2P
   `max_waves_per_turn=4` (vs our 6), claim = "limit unnecessary attacks, preserve
   fleet strength". Targets exactly the mirror matchup that now decides our tier.
   **Locally measurable** → Track 3 below.
3. **roi 1.55 + pressure heuristics (better-flowdiff):** safe-haven 1.5× /
   meat-grinder 0.5× scoring mods + CCW/CW directional bias 1.2/0.9 + an NN
   sidekick (30% weight). No reported score; ROI 1.55 is a cheap Track-3 knob,
   the rest is unvalidated.
4. **4P FFA bonuses — convergent constants:** `ffa_leader_attack_bonus=0.035`,
   `ffa_target_prod_bonus=0.08` appear in TWO independent forks (hybrid-v4
   lineage). Much gentler than our ladder-regressed 1.25×/0.55× mult. Locally
   unresolvable (4P noise floor ±10% @ n≈40) → **ladder A/B candidate, queued
   behind the clamp verdict**.
5. **Sun-crossing "bug fix" (hybrid-producer-v5): does NOT apply to us** —
   verified our vendored `orbit_lite` already masks sun-crossing candidates
   exactly (`movement_aiming.py`, `intercept_aim.py` swept-segment checks). They
   forked an older/lite base.
6. **Multi-size candidates (pilkwang 0.5/0.75/1.0 ×drain; anthonytherrien
   floor-matched +2/+10%):** divergent with our Cluster 7 closure (0/81 cheap
   picks, n=120 A/A). No public win-rate evidence offered → **keep closed**; our
   gated `cheap_capture_margin` code can replicate theirs in minutes if ladder
   evidence ever appears.
7. **Comet expiry guard (veto captures of comets expiring < eta+3):** our 06-10
   audit showed producer loses ~1 ship/game to expiry → immaterial, skip.
8. **Data: Slawek Biel publishes a daily-refreshed top-10% replay dataset**
   (`am-i-in-the-top-10-replays-yet`, ~4.8K replays/snapshot). Curated 1300+
   -tier episodes — upgrade for the Phase-2 stretch (learned value) AND a
   candidate BC-teacher corpus *above* producer tier for the ExIt track.

### Re-prioritized actions (in order)
1. **Track 3 (NEW, running): local 2P mirror knob A/Bs vs producer** — the one
   locally-measurable lever for the tier that now decides our rank. n≥120,
   margin metrics, one knob at a time. Gate: <40%/>60% @ n=60 to proceed,
   decision at n≥120.
   - `max_waves_per_turn=4` (counter-producer claim): **CLOSED 2026-06-11 —
     45.4% @ n=120** (A/A band 45.4–53.8%) → no signal, leaning negative; the
     "conservative waves" claim does not survive a producer mirror.
   - `roi_threshold=1.55` (better-flowdiff; also already the 4P preset value):
     **CLOSED 2026-06-11 — 52.3% ± 3.2% @ n=240** (read 55.0% in the first 120
     games, 49.6% in the second — third consecutive regression-to-the-mean on
     a roi knob). No change.
   - early-horizon 24 (hybrid-v5): **DEPRIORITIZED** — our prior sweep already
     measured global horizon 22 at 42% @ n=60 (noise-or-worse), and the
     early-only variant needs code for an unvalidated public claim. Revisit
     only if the ladder shows we lose long openings.

   **Track 3 verdict: all three public mirror-knob claims fail the producer
   mirror locally.** Producer's hand-tuned 1.5/6/18 survive (again). The
   remaining levers are ladder A/Bs (clamp → terminal-phase → ffa bonuses) and
   the RL track.
2. ✅ **Clamp ladder A/B JUDGED 2026-06-12: KEEP** (v5.1 1191.1 vs producer
   1131.3, +59.8 same-epoch — see the 06-12 update at the top).
3. ✅ **Terminal-phase port BUILT 2026-06-11** (`agents/v5/main.py`:
   `terminal_phase_turns` (0 = OFF, byte-identical) + `terminal_roi_threshold`
   1.0 / `terminal_max_waves_per_turn` 8 / `terminal_enable_regroup` False,
   swapped in `run_turn` for the final N turns; ruff+pyright clean; smoke =
   real-env game vs producer DONE/DONE with `terminal_phase_turns=40`; that
   game ended step 282 → confirms mirrors end pre-terminal, ladder-only
   measurable). Next ship = **v5.2 = (clamp if it survived) +
   `terminal_phase_turns=40` + any Track-3 winner**, paired with the mandatory
   producer resubmit. 4P preset note: exp59 uses it in BOTH formats; enable in
   both.
4. **4P ffa bonuses (0.035/0.08)** = the ladder A/B after v5.2.
5. **RL track unchanged:** launch Colab `PIPELINE='producer256_v3'`
   (arrival-horizon). New stretch: BC/value data from the top-10% replay
   dataset instead of (only) producer-mirror demos.

## The research findings that reframe everything

### 1. The leaderboard meta is far above apex — our anchor was wrong

| LB score | Meaning |
|---|---|
| 686 | median (p50) |
| **736.7** | **us (ExIt iter-20)** — rank 1762 |
| 756–780 | our own rule-based agents (inactive) |
| 931 | p75 |
| 958–1100 | strong *public* heuristic notebooks |
| 1185 | p90 |
| 1224–1287 | best public agents (Tamrazov 1224 hand-tuned; **Producer flow-diff 1287.1**) |
| 1369 | p99 |
| 1723 | #1 (Isaiah @ Tufa Labs) |

Apex ≈ 757 ≈ p55. Every week of ExIt work optimized win-rate against a median-tier
opponent. Even a hypothetical 100%-vs-apex agent could rate ~900 if it loses to the
950–1300 public tier it has never trained against.

### 2. What the 950–1300 public tier actually does (10 notebooks analyzed)

One shared rule-base lineage dominates (pilkwang structured-baseline → Tamrazov/ykhnkf
v4/v6 → forks at 958/1000/1100/1224). Its pillars, none of which apex has:

- **Arrival ledger + exact per-planet ownership timeline** (production + engine-exact
  combat replay to horizon ~110): gives `fall_turn`, `keep_needed` (binary-searched
  min garrison that survives), and `min_ships_to_own_by(target, turn)` (binary-searched
  exact send size including own planned commitments this turn). Replaces ALL
  `garrison + prod*eta + 1` heuristics.
- **Send-size↔speed fixed point** (size→speed→ETA→garrison growth→size, ~4 iters)
  — naive sizing systematically under-sends because bigger fleets fly faster.
- **Synchronized multi-source swarms** (2–3 sources, ETA spread ≤1–2, only when no
  smaller subset suffices, joint outcome re-simulated before commit).
- **Snipe timing**: land at `enemy_eta ± 1` on contested neutrals so the enemy pays
  the garrison cost.
- **Crash exploitation (4P)**: detect two different enemies' fleets arriving at the same
  planet within 2–3 turns → arrive `crash+1` and vulture the survivors.
- **Reaction-time race triage of neutrals**: my fastest legal ETA vs enemy's fastest
  → safe ×1.2 / contested ×0.7; opening avoids rotating neutrals.
- **Reserves & budgets**: attack budget = ships − max(keep_needed, ~18–28% of nearest
  enemy garrison reachable ≤12 turns); frontier planets keep deterrence garrisons.
- **Logistics**: rear planets ship 60–70% of garrison to front staging planets;
  doomed planets evacuate (ships in flight still score); endgame total-war dump.
- **4P meta**: attack the *nearest* opponent (1.25×/0.55× near/far priority — worth
  ~+100 LB by itself per the 1100 fork), elimination drive on the weakest enemy,
  leader-guard variants.
- **Time budgeting**: soft deadline 0.82×actTimeout, heavy phases gated, graceful bail.

### 3. The best public agent (Producer, 1287.1) is conceptually different — and public

Slawek Biel's "Producer" (forked by Tamrazov as `orbit-wars-i-m-better`, self-contained
payload; extracted locally to `/tmp/imbetter_payload/`): a torch-vectorized **exact
counterfactual flow-diff planner**. Per turn: project all planets+fleets 18 turns
(13 in 4P) with engine-exact combat; candidates = top-12 sources × (top-12 attack +
top-4 defend targets); ONE send size per pair = `safe_drain` (max ships the source can
shed and provably still hold itself); score = re-simulate only touched planets and take
`Δnet_me − ΣΔnet_opp`; fire greedily while score > ROI threshold 1.5; leftover ships
regroup up an enemy-pressure gradient. No hand value formula, no fractions, defense and
offense compete in one scorer. **Verified locally 2026-06-09: runs fine under
`kaggle_environments` (torch CPU, ~20–35s/game) and beat apex 4/4, side-alternated,
seeds {101, 202}** (`/tmp/producer_vs_apex.py`).

### 4. Public ML/RL findings replicate ours — and show the working pattern

- Their PPO-from-scratch, PPO+curriculum, single- and multi-teacher SFT all collapsed
  to ~0% vs strong rule-bases (mirrors our PPO dead end).
- The two ML patterns that *work* publicly:
  - **Reject-only shot validator** (konbu17): 24-feature MLP (~5K params) trained on
    dense per-shot supervised labels ("did we own the target within [arrival,
    arrival+10]?"), veto threshold 0.4, self-reinforcements exempt. **+19pp locally,
    gains concentrated vs the strongest opponents, no opponent regressed.** Fail-safe:
    can only drop bad shots from a strong rule-base.
  - **Learned global value for candidate re-ranking** (aidensong): 16-feature GBC,
    (state → P(win)) trained on top-agent replays + self-play, AUC 0.976. Global value
    is very learnable here (consistent with our Phase-2 diagnostic: value grounded
    globally, corr 0.39, but can't rank near-equal siblings).
- Episode data for training: **Meta Kaggle** `Episodes.csv` (5.6GB) /
  `EpisodeAgents.csv` (19GB), refreshed daily; per-episode replay JSON via
  `GetEpisodeReplay`.

### 5. Implications for the in-flight Colab experiments

The ckpt-gate / c_scale / embed A/Bs all optimize win-rate **vs apex** (≈ LB ~757).
Still worth collecting the results (they validate the Gumbel machinery), but shipping
their winner moves us from 736 to at best ~760-territory. The leaderboard lever is the
opponent/teacher pool, not the search hyperparameters.

## Implementation plan (1 week)

**Strategy: re-anchor everything from apex (p55) to the public 1224–1287 tier (p97), by
(a) standing on the best public agents instead of competing with them from below, and
(b) pointing our ML machinery (BC, ExIt, shot-validator) at the new tier.**

### Phase 0 — Stop the bleeding (today, ~1h)
1. ✅ DONE (2026-06-10). Vendored 7 public agents into `agents/external/` (producer
   pkg, tamrazov_1224, distance_1100, shot_validator_hybrid + decoded weights.npz,
   enders_1000, ow_proto, reinforce_958), loaded via
   `agents.external.load_agent(name)` — fresh exec per game (they keep module-level
   ledgers; note: frozen dataclasses require registering the module in `sys.modules`
   before exec). Public Apache-2.0 notebooks; forking is the norm in Kaggle sim comps.
   `agents/external` excluded from ruff/pyright.
2. ✅ DONE (2026-06-10). `scripts/arena.py`: round-robin on the REAL Kaggle env
   (external agents consume full Kaggle obs), side-alternated paired seeds, one
   process per game, incremental CSV with resume (bump --games to extend), win-rate
   matrix output, `exit:<ckpt>:<config>` specs for our checkpoints. This replaces
   "win-rate vs apex" as THE metric. ~15–35s/game, parallel.
3. ✅ DONE (2026-06-10). **Producer bundle submitted** (`outputs/submissions/
   producer_bundle.tar.gz` = main.py + orbit_lite/ at archive root, exact replica of
   the proven i-m-better packaging). Verified pre-submit by extracting the tarball and
   running a real game through Kaggle's file-agent loader (append→exec→pop sequence) —
   status DONE, won vs random. Active slots now: producer bundle (PENDING) + ExIt
   iter-20 (drifted 736.7 → 725.3). Judge the score after ≥1 day of episodes.

**First smoke matrix (2026-06-10, n=2/pair, seeds 20000–20001, side-alternated, 2P
only, 0 errors — `outputs/arena/smoke.csv`; coarse, gate decisions need n≥30):**

| agent | mean | notes |
|---|---|---|
| producer | **100% (18-0)** | swept every opponent — confirmed Phase-1 base + submission candidate |
| ow_proto | 72% | stronger locally than its LB ~1080 suggests |
| **exit embed256 last** | **67%** | beat tamrazov_1224, distance_1100, apex 2-0 each — our RL stack is already mid-public-tier |
| enders_1000 | 56% | |
| exit iter-20 (submitted, LB 736.7) | 44% | mid-pack, consistent with LB |
| tamrazov_1224 | 39% | LB 1224 yet weak here → its rating likely leans on 4P games |
| shot_validator_hybrid / reinforce_958 / apex | 33% | apex confirmed bottom-tier |
| distance_1100 | 22% | its +100 LB came from 4P features — invisible in this 2P-only arena |

**Implication:** the 2P arena under-measures 4P specialists (tamrazov_1224,
distance_1100). Add a 4-player arena mode before trusting the matrix for
submission gating — LB episodes mix both formats.

### Phase 1 — "v5 base": best public skeleton + our orthogonal fixes (days 1–3)
Pick the arena winner as the base (expected: Producer flow-diff). Add what it
measurably lacks (each gated on arena games vs the full pool):
- ~~Comet handling~~ **CLOSED 2026-06-10 — premise was wrong.** Producer does NOT
  ignore comets: `_apply_comet_paths` projects exact comet path positions + expiry
  (`alive_by_step`), and `safe_drain` + the exact flow-diff implicitly evacuate
  doomed comets (saved ships score ≈ their count). Audit (producer vs 1224, seeds
  31000/31001): producer captured 2/1 comets, lost only 1/7 ships to expiry; the
  1224 agent with "exact comet logic" lost 33. Nothing to port.
- ✅ Endgame (2026-06-10, in `agents/v5/`): planning horizon clamped to remaining
  game steps (config-level replace; `garrison_status(max_horizon<build)` is broken
  upstream). Makes the flow-diff exact wrt termination → late capture-refusal +
  total-war drains fall out for free. Gate pending.
- Snipe + crash-exploit timing missions (from the v4 lineage) if the flow-diff scorer
  doesn't already subsume them (check in arena).
- ✅ 4P preset tuning (2026-06-10, in `agents/v5/`): nearest-opponent priority
  (1.25×/0.55× score mult on enemy-owned targets, distance-1100 lineage) added on
  top of Producer's leader-guard bonuses; knobs default-off, on in the 4P preset.
  Gate pending (4P arena).
- ~~ROI threshold / horizon sweep~~ **CLOSED 2026-06-10/11, no change.** Mirror
  A/B vs producer: roi 1.8 = 42%, horizon 14 = 37%, horizon 22 = 42% (all noise
  or worse, n=60); roi 1.2 looked like 57% at n=60 but converged to **52.2% ±
  3.7% at n=180** (fresh 120 games: exactly 50.0%) — regression to the mean.
  Producer's hand-tuned 1.5/18 survive. (`outputs/arena/sweep_roi.csv`)

**v5 status 2026-06-10 (end of day): SHIPPED.** Fork = `agents/v5/` (package renamed
`orbit_lite_v5`), arena spec `v5` (+ parametrized `v5:key=val+key=val` for sweeps),
bundle builder `scripts/build_v5_bundle.py`.

**Gate results (n=60 2P / n=32 4P, 0 errors):**
- 2P: v5 44% vs producer — but every game ended by elimination at step 106–282, so
  the endgame clamp NEVER fired → this was an A/A test of identical agents. Key
  calibration: **the A/A noise floor of n=60 mirror games is ~±6% (44% measured on
  identical agents)**; only <40% / >60% is signal at this n.
- 4P: dead even (rank 1.53 vs 1.53, 47% vs 50% outright) — the nearest-opponent
  mult is active-but-neutral at producer-level tables.
- **Measurement lesson:** the pool is too weak to measure producer-level
  improvements (producer = 99% 2P / 91% outright 4P vs pool — ceiling). Mirror
  A/Bs vs producer are the only sensitive instrument, and they need margin
  metrics (steps-to-win now recorded in arena CSVs) — binary win-rate at n=60
  can't resolve <±10%.
- Shipped anyway: v5 ≈ producer locally, and the clamp/4P-mult only pay off on the
  ladder (long games vs mixed tiers); it replaced the dominated 729 ExIt slot.

**Final contender matrices (`outputs/arena/arena{,_4p}.csv`, n=30/pair 2P, 40
co-occ/pair 4P):** producer 99% 2P / 94% pairwise / 91% outright-win 4P — the
undisputed base, including 4P. ow_proto #2 both formats (68% / 57%). tamrazov_1224
and distance_1100 are NOT 4P specialists (2–5% outright) — their LB ratings come
from the weaker ladder population. **exit_embed256 = 31% 2P (below apex 43%) and
last in 4P (1% outright)** — the n=2 smoke that had it at 67% was noise; Phase-2
re-anchoring of ExIt is mandatory if the RL track is to matter.

### Phase 2 — ML edge on top (days 3–6, the RL-learning track)
In order of expected value-per-day:
1. ~~**Shot-validator veto** on the v5 base~~ **CLOSED 2026-06-11 — CONFIRMED
   NEGATIVE** (full post-mortem: EXPLORED_AND_ABANDONED.md Cluster 6). Built the
   whole pipeline (400-game harvest → 274K labels → MLP val AUC 0.82, veto
   precision 84–95% on v5's shots → `v5v:<t>` arena spec). Mirror gate vs plain
   v5: t=0.10 → 37%, t=0.25 → 41%, t=0.40 → 33% (n=60 each, dose-responsive —
   heavier veto = worse). The +19pp public pattern prunes a margin-heuristic
   base's true errors; producer's flow-diff only fires at ROI>1.5, so its
   "failed" shots are priced-in attrition trades the ownership label can't see.
   5th instance of "coarse learned signal second-guessing an exact planner
   regresses it". Infra kept (default-off; harvester reusable for value labels).
2. **Re-target the ExIt pipeline**: BC teacher and eval/opponent pool switch from apex
   to {producer, 1224, v5}. The Gumbel search machinery (our one validated search win,
   +9.4%) stays; the expert improves because the policy prior and the opponents are
   ~500 LB points stronger. This is the same pipeline we already have — only the data
   distribution changes.

   **Prep DONE 2026-06-11 (local) — ready to launch on Colab:**
   - `configs/v2_exit_producer256.yaml`: fresh BC from producer (NO warm-start —
     teacher distribution changed), embed-256, Gumbel ON, opponent/eval = producer.
   - New plumbing (smoke-tested): `exit.collect_side_alternate` (play_single_game
     was P0-only; policy seat now alternates by seed parity),
     `imitation.bc_side_alternate` (expert seat alternates by game parity),
     `imitation.bc_collect_workers` (parallel demo games),
     `imitation.bc_demo_opponent: producer` (MIRROR demos — on-distribution
     states instead of producer-crushes-random).
   - Demo cache collected locally (150 mirror games, 13,968 samples, 79%
     launch-capture) and seeded to Drive:
     `gdrive:orbit_wars_outputs/demos_producer_mirror_v1.pkl`.
   - `notebooks/train_colab.ipynb` rewritten: PIPELINE ∈ {producer256 (default),
     embed256, exit}, dead pipelines/configs/apex references pruned, submission
     cell always bundles via `v2/agent_v3.py`, eval cell evals vs producer.
   - **To launch: open the notebook on Colab (A100), run cells 1→4 with
     PIPELINE='producer256'.** Gate after: download ckpt, arena vs pool + mirror
     vs producer/v5 (n≥120).

   **RESULT 2026-06-11 — the re-anchor WORKED; new RL champion (not shipped):**
   Colab run `v2_exit_producer256_a100` (40 iters, 20 min wall-clock). In-training
   eval vs producer read 0% throughout — that's the producer ceiling, not failure.
   Real-env arena (n=30/pair): **mean 55%, #3 in the pool** — beats ow_proto 87%,
   enders_1000 87%, and the old champion exit_embed256 **97%** (old champion: 15%
   mean, last place); still 3% vs producer and v5. fast_env iter sweep vs
   ow_proto: BC clone 37% → iter-25 **97%** → iter-40 97% (ExIt added +60pp;
   plateaued by iter 25). Best ckpt = `v2_exit_producer256_a100/ckpt_000025.pt`.
   Diagnosed limits for the next run: (a) zero collection wins vs producer →
   outcome/value signal constant (vloss→0.0002) — needs a mixed/beatable opponent
   pool; (b) search fraction targets ~uniform (floss≈ln4) — sizing doesn't distill,
   and exact sizing is precisely producer's edge; (c) launch rate 0.37/step
   (passive vs producer). NOT submitted: ~1100–1180 LB tier would sacrifice one of
   the two slots (1222.8 producer / 1174.1 v5).

   **Phase 2.2b — data-maxxed follow-up, PREP DONE 2026-06-11 (ready for Colab):**
   targets diagnosed limits (a)+(c) via data/signal, capacity explicitly out of
   scope (collection inference runs on CPU workers).
   - `exit.opponent` now accepts a LIST sampled per game, deterministic by game
     seed (`v2/exit_train.py resolve_collect_opponent`; uncorrelated with the
     seed%2 side-alternation; single-name backward compat; special name `self`
     = frozen deterministic copy of the current net). The iter log prints a
     per-opponent win breakdown, e.g. `[ow_proto:1/1 producer:0/1 v5:0/2]`.
   - `configs/v2_exit_producer256_v2.yaml`: pool [producer, v5, ow_proto,
     enders_1000], games_per_iter 8→24, dataset_max_iters 3→5, iterations 60,
     eval_opponents [producer, ow_proto] (visible progress below the producer
     ceiling), NO BC — warm-start from `v2_exit_producer256_a100/ckpt_000025.pt`
     via `--resume`. Gumbel ON, embed-256.
   - Notebook: `PIPELINE='producer256_v2'` (new default) wires the config +
     WARM_START from Drive; run_name `v2_exit_producer256_v2_a100`.
   - Local smoke (2 iters, 4 games/iter, warm-start + pool): mixing works and
     the value signal is BACK — vloss 0.76/0.41 (vs 0.0002 starved), wins vs
     ow_proto/enders in collection, eval reads 0% producer / 100% ow_proto as
     designed. floss still ≈ln4 (sizing distillation = known open item).
   - **Launch = user, on Colab: run cells 1→4 with PIPELINE='producer256_v2'.**
     GATE after: download ckpts; arena must (1) beat the iter-25 champion
     head-to-head AND (2) raise the pool mean >55%, n≥30/pair; plus mirror vs
     producer/v5 at n≥60. Never act on n<100 outliers (A/A ±6% @ n=60).

   **RESULT 2026-06-11 — GATE FAILED both conditions; champion UNCHANGED
   (`v2_exit_producer256_a100/ckpt_000025.pt`).** Colab run
   `v2_exit_producer256_v2_a100` (60 iters, 87 min): the data fixes landed as
   designed (vloss alive 0.001–0.03 vs 0.0002 starved; steady wins vs
   enders/ow_proto in collection; tloss 0.35→0.27) but did NOT translate:
   - Head-to-head vs incumbent (n=30 each): iter-25 43%, iter-50 40%,
     iter-60 43% — pooled 42% over 90 games, consistently below 50%.
   - Pool (n=30/pair, same 4 opponents; incumbent = 3/3/87/87 → 45%):
     new iter-60 = 0/3/73/80 → 39%; new iter-25 = 0/3/73/90 → 42%. Every
     pair flat or down. floss stayed ≈ln4 throughout.
   Verdict: mixed-pool + 3× data mildly REGRESSED the champion. The
   "value signal" hypothesis is refuted as the lever — **nothing in the
   pipeline consumes the value head** (search leaves are heuristic
   `evaluate_state`; the head is only read by the graveyarded
   neural_value_leaves/value_leaf_blend paths), so restoring vloss could not
   improve the policy; meanwhile 75% of collection games shifted to
   sub-producer states and diluted the distillation distribution.

   **ROOT CAUSE FOUND (floss diagnostic, 2026-06-11): the search is
   horizon-blind.** Measured on collection-distribution decisions: only
   **16.8%** of enumerated candidates arrive within the depth-12 horizon
   (median travel time 28.5 turns, p75≈48). For the rest the launched fleet is
   still in flight at the leaf, and `evaluate_state` counts in-flight ships at
   full value → the leaf eval is IDENTICAL to hold (measured per-decision
   candidate q-spread: 0.0 at p90, across targets AND fractions). Gumbel's
   completed-Q then degenerates (`_minmax` → all 0.5) and pi' = softmax(prior
   logits + const) = **the prior itself**: ~5 of 6 decisions distill the prior
   back into itself. This explains floss≈ln4 (fraction siblings differ only by
   fleet speed — virtually never resolve differently in-horizon), the iter-25
   plateau, and why producer's long-range exact sizing never distills.

   **Next single-variable bet: arrival-resolving search horizon** — simulate
   each candidate to `tt + settle_margin` (capped) instead of fixed depth-12,
   so every candidate's consequence (capture succeeds/fails at THIS size)
   reaches the leaf. Directly creates the missing fraction signal. Cost is
   bounded per candidate and search is not the bottleneck (8–16s/iter vs
   40–75s collect on Colab). Graveyard caveat to respect: the passive in-sim
   opponent never reinforces, so longer horizons may overrate distant
   captures — gate vs the champion on paired seeds, flag default-off.

   **Phase 2.2c — arrival-resolving horizon, BUILT + mechanism-verified
   2026-06-11 (ready for Colab):**
   - `exit.arrival_horizon` (default OFF, bit-identity re-verified via
     `scripts/test_gumbel_search.py`) + `arrival_settle_margin` (4) +
     `arrival_horizon_cap` (60) in `v2/search.py::_decision_depth`: per-decision
     depth = min(cap, ceil(max candidate tt) + margin), UNIFORM across the
     decision's candidates INCLUDING hold (evaluate_state's production term
     grows with depth → mixed-depth leaves are not comparable). Threaded
     through `_simulate_descriptor` on both the legacy and Gumbel paths.
   - **Mechanism probe (`scripts/diag_arrival_horizon.py`, champion ckpt, 1
     game vs ow_proto, 30 records): the fix works exactly where signal is
     possible, and the residual flatness is NOT horizon-blindness.** ON vs OFF:
     hostile-candidate resolution 0%→100% @p50; hostile decisions with live
     q-spread 24%→49%; capture-capable (src≥10) 35%→65%; spread magnitude p90
     32→123; fraction-entropy p10 1.06→0.02 (decisions with signal get
     near-deterministic sizing targets). cap=90 ≡ cap=60 (cap not binding).
   - **Two exact cancellation mechanisms explain the still-flat decisions —
     both correct, neither fixable by depth:** (1) friendly transfers conserve
     ships → q ≡ hold by construction; (2) a FAILED attack on the (best) enemy
     trades ships 1:1 and `evaluate_state` scores `my − best_enemy`, so the
     trade cancels exactly (matches producer's priced-in-attrition worldview —
     the same property that killed the shot-validator veto). Net: the lever
     unlocks sizing signal on capture-capable attack decisions specifically —
     which is precisely producer's edge we failed to distill. Expect floss to
     drop below ln4 but NOT to 0; pi'=prior on consequence-free decisions is
     correct deference, not degeneracy.
   - Single-variable run: `configs/v2_exit_producer256_v3.yaml` = champion
     recipe (fresh BC from producer-mirror demos, producer opponent, NOT the
     2.2b pool) + the three arrival flags; eval adds ow_proto. Notebook
     `PIPELINE='producer256_v3'` (new default).
   - Local smoke (2 iters, 4 games, warm-start, ON vs OFF same seeds): both
     complete; search 11–13s ON vs 4–7s OFF (~2–3×, still ≪ collect 30–49s →
     off the critical path, no gumbel_sims trim needed). floss ON ≈ OFF at
     2 iters (1.36 both) — EXPECTED: floss separates only as distillation
     converges; the probe's target-entropy sharpening is the leading
     indicator. On Colab watch floss over the full 40 iters (2.2b reference:
     pinned ≈1.386 throughout) and expect a drop toward ~1.15–1.25, NOT 0
     (~78% of decisions are consequence-free → correctly keep prior targets).
   - **Launch = user, on Colab: cells 1→4, PIPELINE='producer256_v3'.** GATE
     after: download ckpts; beat `v2_exit_producer256_a100/ckpt_000025.pt`
     head-to-head AND raise the 4-opponent pool mean (>45%), n≥30/pair; mirror
     vs producer/v5 n≥60. Watch for the aggression signature (launch rate up +
     mirror losses) → lever is cap/margin, not an opponent model.

   **RESULT 2026-06-12 — GATE FAILED, decisively; champion UNCHANGED
   (`v2_exit_producer256_a100/ckpt_000025.pt`).** Colab run
   `v2_exit_producer256_v3_a100` (fresh BC + 40 iters, 26 min). The mechanism
   FIRED but made the agent worse:
   - In-run: floss spiked 2.48 at iter 1 (sharp sizing targets meeting the BC
     model — proof the targets sharpened), dipped to 1.04, settled ~1.355
     (below the 2.2b reference 1.386, above the hoped 1.15–1.25). Eval vs
     ow_proto: BC 25% → iter-5 72% → declined to 52–57% by iters 20–40.
   - Arena h2h vs champion (n=30/pair, paired seeds): **iter-5 17%, iter-35
     17%, iter-40 3%** — pooled 12% over 90 games, worsening with training.
     Pool screen skipped (condition 1 failed everywhere).
   - Loss signature: median game runs the full 499 steps and loses on score —
     overextension/misallocation, not early elimination. The sharpened targets
     are tuned to a world where NOBODY contests for 30–60 steps: the passive
     sim rewards greedy distant expansion with prod_weight × prod_advantage,
     and distilling that EXACTLY is worse than the depth-12 blur (which only
     trusted short-horizon consequences the sim gets right).
   - **Verdict: 6th leaf/horizon/opponent search experiment to regress the
     champion** (neural leaves, rollout opp, 2P one-ply, value blend, mixed
     pool, arrival horizon). The pattern is now sharp: anything that makes the
     search lean HARDER on the passive sim's long-range predictions loses; the
     one winner (Gumbel) changed selection/anchoring, not the world model.
     Possible salvage (cap/settle_margin small, e.g. cap 20–24) is another
     Colab run for a third-order knob — NOT worth it before the deadline.
     Infra stays (default-off, byte-identical); flag goes in the config
     graveyard. RL track: stop horizon-style search surgery; remaining
     credible RL lever = learning from above-producer data (Slawek top-10%
     replay dataset, action 5 in the 06-11 refresh).
3. (Stretch) **Learned value from real episodes**: pull top-team episodes via Meta
   Kaggle, train the 16-feature global value, use it for candidate re-ranking in the
   flow-diff planner (re-rank near-ties only — respects our sibling-ranking finding).

### Phase 3 — Ship & iterate (continuous)
- Every improvement that beats the incumbent in the arena (n≥60, multi-seed, both 2P
  and 4P) gets submitted same-day; keep the better of the 2 active slots pinned.
- Track LB score deltas per submission in this file.

**Submission log:**
| date | bundle | LB score | note |
|---|---|---|---|
| 2026-06-04 | v2_exit_a100 iter-20 | 736.7 → 729.4 | ExIt champion vs apex; slot freed 06-10 |
| 2026-06-10 15:23 | producer_bundle | 695.0 → **1242.7** (same day) | **rank 140/4212 (top 3.3%)**, was 1762 |
| 2026-06-10 19:48 | v5_bundle | 1110.5 → 1174.1 → **1159.7 CONVERGED LOW** | producer + endgame clamp + 4P nearest-opp mult; −71 vs producer → mult diagnosed as the suspect delta |
| 2026-06-11 16:07 | v5_bundle (v5.1) | 835.6 → **1193.4 @ 06-12** | **mult OFF** (clamp only) — **A/B verdict: clamp KEEP (+60 vs producer same-epoch)**; evicted 06-12 by the v5.2 ship |
| 2026-06-11 16:42 | producer_bundle | → **1126.0 @ 06-12** | resubmit; A/B duty done (lost to clamp), evicted 06-12 by the v5.1 resubmit — baseline role passes to v5.1 |
| 2026-06-12 13:52 | v5_bundle (v5.2) | 1147.0 @ eviction (~7h) | clamp + terminal phase ON — A/B cut short by the v5.3 ship, but led v5.1 1147 vs 1134 mid-climb → kept as the control line |
| 2026-06-12 13:53 | v5_1_bundle (v5.1 resubmit) | 1134.0 @ eviction (~7h) | trailed v5.2 by 13 mid-climb; evicted by the v5.2 control resubmit |
| 2026-06-12 21:05 | v5_bundle (v5.3) | **1201.6 @ 06-13 (rank 313/4381)** | **v5.2 + reinforce-risk floor (beta 2.2, Producer V2 lineage)** — mirror gate 75% @ n=120. **A/B VERDICT: KEEP — +150 same-epoch vs v5.2 control (1201.6 vs 1051.1); first decisive local↔ladder agreement.** New pinned baseline. |
| 2026-06-12 21:06 | v5_2_bundle (v5.2 resubmit) | 1051.1 @ 06-13 (rank ~1005) | pinned control; LOST the A/B by 150 → reinforce-risk is a real ladder win, not noise. Baseline role passes to v5.3. |

**Phase 2 local-track results (2026-06-11):**
- **Track 1 (second candidate size) CLOSED — structurally inert.** Flow-diff provably
  prefers full drain (speed grows with size; 0/81 cheap picks in an instrumented
  game); gate n=120×2 was pure A/A. Post-mortem: EXPLORED_AND_ABANDONED.md Cluster 7.
  Code kept gated default-off (`cheap_capture_margin`).
- **Track 2 knob sweep (ffa_leader_attack_bonus) — NO SIGNAL.** Scan at n=40/agent
  said 0.15 ≫ incumbent (rank 1.57 vs 1.77); focused confirm at n=48 flipped it
  (v5 1.60 vs 0.15's 1.71). Pooled head-to-head over 72 games: exactly 50.0%.
  4P-arena noise floor at n≈40 is ±10%+ — small local 4P deltas are unresolvable
  at affordable n; the ladder is the only credible 4P instrument.
- **Next real information = ladder convergence (~06-12):** v5.1 vs producer A/Bs the
  endgame clamp on the real ladder. Don't spend slots on locally-unresolvable knobs.

### Explicit non-goals this week
- No more search-hyperparameter A/Bs vs apex (collect the in-flight Colab results,
  then retire apex as the gate).
- No PPO revival; no neural leaf values inside search (4× confirmed negative).

## Risks
- **Attribution/originality**: Kaggle norms require crediting forked public work in
  writeups; fine for ranking. Our differentiation = comets + 4P tuning + shot
  validator + ExIt-on-top.
- **Public-agent saturation**: many teams run the same public bases; ties in the
  1100–1300 band are expected. The ML edge (Phase 2) is what breaks the tie — almost
  nobody's published ML actually works.
- **Ladder noise**: Tamrazov: "ladder scores are noisy, reruns move up or down" —
  judge submissions after ≥1 day of episodes, not on validation score.
