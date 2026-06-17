# Diffusion / Flow-Matching action generation — experiment plan (2026-06-16)

**Status: PROPOSED (not built).** Derived from a deep-research sweep of SOTA generative
robot-action models (Diffusion Policy, Decision Diffuser, π0, AdaFlow, Consistency Policy,
Streaming Diffusion Policy, DiffuserLite + 2024-25 surveys). This doc translates that
literature into a concrete, gated build for Orbit Wars and — critically — positions it
against the project graveyard so we don't re-run a closed pattern under a new name.

Live strategic frame: `[[strategic-direction-big-swings]]` (climb to top-50 via big swings;
rich-representation ML BC is sanctioned). De-risk discipline + slot rules: unchanged from
`LEADERBOARD_CLIMB_PLAN.md`.

---

## 0. TL;DR

Build a **multi-modal generative behavior-cloning head** (flow-matching, with diffusion as the
fallback) that generates the **full per-source allocation matrix** OrbitNet already emits
(`logits [B,P,P+1]` + `frac_logits [B,P,P,K]`), conditioned on the rich producer-grade
representation. Distill it to **few-step / variance-adaptive inference** so it stays under the
1 s/step budget. The one property worth chasing is **multi-modality**: a generative head can
clone the *heterogeneous* top tier (producer full-drain + Isaiah half-drain + swarm) **without
mode-averaging them into mush** — which is exactly the failure mode that killed naive
whole-pool BC.

**This is a NEW pattern, not a re-run of the closed one.** The just-closed Track 1
(`RICH_BC_SELECTION_FINDINGS.md`, Cluster pending) was a learned *selection delta over
producer's exact flow-diff* — it regressed the teacher (BC 18%, DAgger 22-26%, prior 46%)
because a non-zero delta second-guesses an exact planner. A generative policy does **not** ride
on producer's scorer at inference; it emits the whole action distribution itself. The shared
lesson we must respect: **prove mirror parity vs producer first**, before assuming
multi-modality buys anything.

---

## 1. What the research actually established (high-confidence, adversarially verified)

Full report + citations in the conversation; the load-bearing, 3-0-verified claims:

- **Generative > regression for actions.** SOTA has shifted from regressing one action to
  generatively modeling action chunks/trajectories (Diffusion Policy, RSS'23/IJRR'25). The head
  learns the action-distribution score field and samples it.
- **Multi-modality is the headline property** (4 independent sources). Diffusion/flow heads
  "naturally represent complex, multimodal action distributions"; GMM/EBM BC **mode-averages**
  over multiple valid demonstrations. This is the single most relevant property for Orbit Wars,
  where a turn admits many equally-good multi-fleet plans — and where our own corpus diagnostic
  proved the top tier is **heterogeneous** (`[[top-tier-replay-diagnostic]]`).
- **Diffuse-states-not-actions is a real fork** (Decision Diffuser, ICLR'23). Action sequences
  are "high-frequency and hard to denoise"; DD diffuses *state* trajectories and recovers
  actions via inverse dynamics. Directly relevant: our true output (variable-length list of
  `[from_planet, angle, ships]`) is exactly the high-frequency discrete-continuous case DD warns
  about.
- **Latency is THE constraint** and it is solved tech. Iterative sampling is the field's
  headline limitation; the fixes are all portable:
  - **AdaFlow** (ICML'24): variance-adaptive ODE — collapses to a **1-step generator when the
    action distribution is uni-modal**, spends steps only where genuinely multi-modal. Best fit
    for us: most Orbit Wars turns are forced/uni-modal.
  - **Consistency Policy**: distill a diffusion teacher to single/few-step via self-consistency.
  - **Streaming Diffusion Policy**: partially-denoised trajectory, rolled forward one step/turn
    — fits per-turn replanning over the 500-step horizon.
  - **DiffuserLite**: coarse-to-fine plan refinement, author-reported 122 Hz / ~0.5-1.2% of
    baseline runtime.
- **Flow matching scales BC over heterogeneous experts** (π0, RSS'25): flow-matching head over
  a cross-embodiment corpus. The transferable lesson is *flow-matching action head + multi-source
  BC scaling* — NOT the VLM backbone (no Internet-scale semantics here).

**Caveat the report itself flagged:** zero evidence comes from real-time adversarial RTS. Every
transfer is a hypothesis. The strongest, best-corroborated one is multi-modality; its concrete
benefit over our already-strong producer flow-diff is **unverified**.

---

## 2. Why this is NOT a closed pattern (the honest graveyard check)

| Closed work | What it was | Why it died | How this differs |
|---|---|---|---|
| Poor-rep BC (3%) | OrbitNet BC of producer from a 40×22 one-step snapshot | Representation poverty — net asked to re-derive an 18-turn projection | Use the **rich** producer-grade features (`producer_features.py`), proven to carry the info |
| Rich BC-**selection** (Cluster, 06-16) | Learned **delta over producer's exact Δnet**; net re-ranks producer's own candidates | A non-zero delta over an exact planner regresses it; DAgger confirmed it's the *mechanism*, not covariate shift | Generative head emits the **whole allocation matrix itself** — it does NOT ride on / second-guess producer's scorer at inference |
| Whole-pool BC | Naive BC of mixed top tier | **Mode-averages** producer-full-drain + half-drain + swarm into mush | A multi-modal generative head is *precisely* the tool to represent that mixture without averaging — this is the entire thesis |
| value-blend / value-rerank (C8/C10) | Learned value re-ranks search leaves / near-ties | Noisy value can't rank near-equal siblings | No value function in the BC head; this is policy generation, not value ranking |

**The residual risk this plan cannot dodge:** *you can't BC past your teacher.* A clone of
producer lands ≤ producer. The climb story depends entirely on the **multi-modal head absorbing
the top-tier replay corpus** (the ~⅓ that are structurally different and sit at the very top —
#1 Isaiah half-drains, 213tubo swarms). If multi-modality is real and capturable, the head beats
a single-mode clone; if the top is effectively a producer monoculture in disguise, this caps at
producer and we stop at the parity gate having spent a real build. That is the bet, stated
plainly.

---

## 3. Design decisions

### 3.1 What to diffuse over → the OrbitNet allocation matrix (NOT the raw action list)

Decision Diffuser's warning + our own action-format pain both point the same way: **do not
diffuse the variable-length `[from, angle, ships]` list.** Instead diffuse the **fixed-shape
per-source allocation target** the v2 model already produces:

- Target tensor = the soft allocation `A ∈ [0,1]^{P×(P+1)}` (per source planet: a distribution
  over {hold, send-to-each-target}) plus the fraction field `F ∈ [0,1]^{P×P×K}`. Fixed shape
  (`P=40`), masked to valid sources/targets exactly as today — **no variable-length problem, no
  padding hack.**
- Decode to launches with the **existing** `v2/actions.py` decode path +
  `intercept_angle`/`safe_drain` sizing. The continuous angle and integer ship count are derived
  analytically at execution, not generated — sidestepping the discrete-continuous generation
  problem entirely.
- This is the "diffuse a structured intermediate, decode actions post-hoc" pattern (DD's
  inverse-dynamics analogue), specialized to our domain.

**Open variant to weigh (Phase 0 decision):** diffuse a short **board/score trajectory** (k-step
garrison-status projection) and pick launches that realize it. Heavier; defer unless the
matrix-target head underfits multi-modality.

### 3.2 Flow matching over diffusion

Default to **conditional flow matching** (π0/AdaFlow family), not DDPM diffusion:
- Straighter probability paths → far fewer integration steps at inference (the binding
  constraint).
- AdaFlow's variance-adaptive solver gives us the **1-step-when-uni-modal** behavior for free —
  the single most important property given most turns are forced.
- Diffusion (DDPM + Consistency-distillation) is the **fallback** if flow-matching training is
  unstable on this target.

### 3.3 Why NOT "diffusion proposes, producer's exact scorer disposes" (the `0/81` receipt)

A tempting lower-risk variant: use the generative head only to *propose* a diverse multi-modal
candidate set (including half-drain / swarm modes producer's grid omits), then let producer's
**exact flow-diff scorer rank** them. This is the Diffusion-QL "generate-N, select-by-value"
pattern, and it looks graveyard-safe (the head never perturbs the score). **Reject it as the
primary bet — we already have the receipt that it fails for the case that matters.**

- The `cheap_capture_margin` knob (`agents/v5/main.py:391-426`) already builds a second,
  cheaper candidate per `(source,target)` and feeds it to the **same** exact scorer to arbitrate.
  Cluster 7 (`[[cheap-capture-second-size]]`): the scorer picked the cheap candidate **0/81**.
- Why: producer's scorer maximizes **immediate net ship-flow** and is **1-ply, single-wave,
  do-nothing-opponent**. Full-drain weakly dominates under that objective (draining is free —
  in-transit ships still count as yours; bigger fleets fly faster → arrive sooner → lower
  capture floor → robust to mid-flight reinforcement). The scorer is *correct within its model.*
- Half-drain's value is **optionality** — a source kept able to launch a 2nd wave / defend /
  branch next turn. That is multi-turn value the myopic exact scorer **structurally cannot
  price**. So widening *generation* doesn't help when the *ranker* is the wrong judge: the exact
  scorer would reject the half-drain proposal exactly as it rejected the cheap one (0/81).

**Consequence:** to capture top-tier optionality value you must clone the **policy** of the
agent that has it (full multi-modal BC, §4 Phase 2) — which carries its implicit multi-turn
rationale and bypasses producer's scorer entirely — NOT lean on producer's scorer as the ranker.
The alternative (a multi-turn value/search that prices optionality) is graveyard-adjacent (the
closed passive-sim search family; only a faithful in-sim opponent + learned value escapes it).
The `0/81` receipt is therefore a *strengthening* argument for Phase 2's full-policy BC route.

### 3.4 Real-time budget (1 s/step, 60 s overage)

Acceptance: median step < ~150 ms on the local no-GPU eval path (arena runs CPU). Levers, in
order of preference:
1. AdaFlow variance-adaptive steps (1 step on uni-modal turns).
2. Consistency-distill the trained teacher to ≤4 steps.
3. SDP-style rollover (reuse last turn's partially-denoised plan) if per-turn cost is still high.

---

## 4. Phased build with hard gates

Every gate uses `scripts/arena.py`, **n≥120 paired, side-alternated** vs the named reference;
A/A noise floor ~45-55%. Never act on n<100. Gated default-off / byte-identical-when-off, same
discipline as every v5 knob.

### Phase 0 — feasibility & target choice (cheap, ~0.5 day, no GPU needed for the decision)
- Reuse `scripts/producer_features.py` (rich edge grid + garrison timeline) as the conditioning
  input — already built and information-parity-verified.
- Reuse `scripts/macro_relabel.py` fleet-tracked relabeling to build the BC target = producer's
  realized allocation matrix on each step. **Alignment gotcha (verified):** the action at
  `steps[t]` was decided on the obs at `steps[t-1]`.
- **Coverage probe** (`scripts/coverage_probe.py`, already exists): do the top-tier agents'
  attacks fall inside the allocation-matrix support? If a different agent's moves are
  unrepresentable in `[P,P+1]`, broaden the target before harvesting.
- **Decision:** matrix-target vs board-trajectory-target. Recommend matrix-target unless coverage
  probe says otherwise.

### Phase 1 — DE-RISK GATE: flow-matching BC of producer must reach mirror PARITY
- Train the flow-matching head on **producer-only** demos (single teacher, rich rep, matrix
  target). Disjoint train/gate seed ranges (no leak — the 06-16 run's mistake-avoidance).
- **Gate:** `flowbc:<ckpt>` vs `v5`, n≥120 paired. **PASS = inside the ~45-55% A/A band
  (parity).** This proves the generative head can faithfully reproduce a single strong teacher
  through the matrix target + decode path.
- **KILL criterion:** if it lands like the closed selection-delta runs (≤ ~25%), the generative
  matrix-target reformulation did not rescue BC → **stop, write up, do NOT proceed to the
  multi-teacher harvest.** This is the same wall in new clothes.
- Latency check in parallel: confirm AdaFlow 1-step / consistency-distilled inference holds the
  step-time budget *at parity* (a parity head that's too slow is still dead).

### Phase 2 — THE BET: multi-modal BC of the heterogeneous top tier (only if Phase 1 passes)
- Harvest top-tier replays via the daily `orbit-wars-episodes-YYYY-MM-DD` datasets (access +
  schema in `TOP_TIER_REPLAY_CORPUS.md`; do NOT bulk-crawl the ListEpisodes endpoint — it 429s).
- Train the **same** head on the mixed corpus (producer-family + Isaiah half-drain + 213tubo
  swarm). The thesis: the flow head represents the *mixture* of strategies; sampling picks a
  coherent mode per turn instead of averaging.
- **This is the route, not propose-and-rank (see §3.3).** Full-policy BC is the *only* path that
  captures top-tier optionality value (e.g. half-drain): it clones the winning policy's implicit
  multi-turn rationale and bypasses producer's myopic scorer — which the `0/81` receipt proves
  would reject those moves if used as the ranker.
- **Gate A (must clear):** vs `v5`, n≥120 — must **beat parity decisively** (clear the A/A band
  upward), else the multi-modal corpus added nothing over the producer clone.
- **Gate B (the real prize):** arena vs the full external pool + producer_v2, 2P and `--players 4`.
- **Diagnostic:** sample N plans per state on held-out top-tier states; measure mode coverage
  (does it recover both full-drain AND half-drain on states where the corpus contains both?). If
  it collapses to one mode, multi-modality isn't being captured → the head or loss is wrong, not
  the thesis.

### Phase 3 — ladder ship (only if Phase 2 Gate A/B win)
- Build bundle, ship paired with the v5.3 incumbent resubmit (active-2 eviction rule). Ladder is
  the final judge for 4P deltas.

---

## 5. Kill switches / what makes us stop

- Phase 1 parity fails → stop (generative reformulation didn't beat the BC wall).
- Phase 2 Gate A ties parity → stop (no exploitable multi-modality over a producer clone; the
  "top is a monoculture" branch is true).
- Mode-collapse diagnostic shows single-mode output despite mixed corpus → the head isn't doing
  the one thing it's for; debug or abandon.
- Latency can't be distilled under budget at parity → stop.

Each stop is a clean negative with a reusable artifact (the rig is gated default-off, byte-
identical). Write up under `EXPLORED_AND_ABANDONED.md` with the phase reached.

---

## 6. Reuse map (what already exists)

- `scripts/producer_features.py` — rich conditioning rep (information-parity verified).
- `scripts/macro_relabel.py` — fleet-tracked target relabeling (alignment-correct).
- `scripts/coverage_probe.py` — candidate-support probe for non-producer teachers.
- `v2/model.py` (`OrbitNetOutput`: `logits [B,P,P+1]`, `frac_logits [B,P,P,K]`) — the matrix
  target shape + masking.
- `v2/actions.py` — decode allocation → launches (+ `intercept_angle`/`safe_drain` sizing).
- `agents/v5/main.py` `_SELECTOR_FN` / `_FEATURE_SINK` hooks — gated execution path, default-off.
- `scripts/arena.py` — the gate; `TOP_TIER_REPLAY_CORPUS.md` — replay access.

New code: a flow-matching (CFM) head + sampler module, an AdaFlow-style adaptive solver, a
consistency-distillation pass, and a `flowbc` arena agent wrapper. Training is Colab-scale
(`notebooks/train_colab.ipynb`, PIPELINE switch); local CPU is for the gate only.

---

## 7. Sources (all primary, peer-reviewed, 2022-2025; adversarially verified 24/25 claims)

- Diffusion Policy — arXiv 2303.04137 (RSS'23 / IJRR'25)
- Decision Diffuser — arXiv 2211.15657 (ICLR'23)
- π0 / pi-zero — arXiv 2410.24164 (RSS'25)
- AdaFlow — arXiv 2402.04292 (ICML'24)
- Consistency Policy — arXiv 2405.07503
- Streaming Diffusion Policy — arXiv 2406.04806
- DiffuserLite — arXiv 2401.15443 (NeurIPS'24)
- Diffuser — arXiv 2306.08810 (ICML'22)
- Diffusion-in-RL surveys — arXiv 2510.12253, Frontiers Robotics & AI 2025, Diff4RLSurvey
- Real-Time Iteration Scheme for DP — arXiv 2508.05396
