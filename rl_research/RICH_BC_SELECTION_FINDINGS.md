# Rich-representation BC of producer's selection — findings (2026-06-15)

**Status: CLOSED (graveyard-ready) as of 2026-06-16.** This documents one architecture — a
learned *selection delta* over producer's exact flow-diff scorer, BC-trained from the teacher's
own moves — measured end-to-end, including the DAgger on-policy escape hatch (§8). The escape
hatch was the discriminating test and it failed: the delta mechanism (not covariate shift)
fundamentally regresses the exact scorer. Initially logged as an open finding (the original
"not sure it's a graveyard" call); the DAgger result resolves it negative.

## 1. The question

Big-swings Track 1 ([[strategic-direction-big-swings]]) asked: can an ML net, given a
**producer-grade representation**, learn producer/v5's *target selection* well enough to (a)
reach mirror parity and then (b) — trained on top-tier replays — *improve* on it? The prior
attempt (V2 40×22 snapshot BC) capped at **3% vs producer** with ~0.59 launch-accuracy; the
hypothesis was **representation poverty** — the net was asked to re-derive producer's 18-turn
projection (fall_turn / keep_needed / min-ships / Δnet) from a one-step snapshot.

## 2. The architecture under test

- **Rich representation** (`scripts/producer_features.py`): run v5's exact `orbit_lite_v5`
  projection on the obs via the `_FEATURE_SINK` hook, expose its per-(source,target) **Δnet
  candidate score**, ETA, size, validity as a dense `[P,P,F]` edge grid + the per-planet
  garrison-status timeline. This is the information producer *itself* reasons over — guaranteed
  information-parity, same code.
- **Selector = a delta over the exact scorer** (`scripts/rich_bc_train.py::RichSelector`):
  OrbitNet with a zero-init pair head, so at init it emits **delta = 0**. The selection score
  for a candidate is `real_Δnet + delta`. The z-scored Δnet rides in `pair_features` as a net
  *input*.
- **Execution** (`scripts/rich_bc_agent.py` + `agents/v5/main.py` `_SELECTOR_FN` hook): the
  net never emits a continuous parameter; it overrides the candidate *score*, and v5's exact
  candidate generation / `intercept_angle` / `safe_drain` sizing execute the picks. Gated
  default-off → v5 byte-identical when unset.

## 3. Measurement protocol

`scripts/arena.py`, real Kaggle env, **n=120 paired, side-alternated** seeds vs `v5` (the
teacher). A/A noise floor at n=120 is ~45–55% (byte-identical agents measured 44–55% across
runs). Parity = a result inside that band; an improvement must clear it decisively.

## 4. The investigation chain

| stage | vs v5, n=120 | CSV | reading |
|---|--:|---|---|
| poor-rep BC (V2 snapshot) | 3% | — | prior baseline |
| rich BC, **old harness** (`rich_bc_v5/ckpt40`) | 21% | `gate_richbc_v5.csv` | looked like a 7× "lift" |
| untrained prior, **old harness** | 5% | `gate_richbc_untrained_v5.csv` | **prior should be ~50% — harness is broken** |
| untrained prior, **fixed harness** | **46%** | `gate_richbc_untrained_fixed_v5.csv` | parity ✓ (byte-id 0/276) |
| **retrained delta, fixed harness** | **18%** | `gate_richbc_fixed_trained_v5.csv` | learned delta **regresses** v5 −28pp |

The untrained-prior control was the load-bearing experiment: a producer-prior selector should
reproduce its teacher (~50%), so its 5% exposed a **harness bug**, not a representation result.

## 5. The harness bug (found + fixed)

The `_SELECTOR_FN` hook (`agents/v5/main.py:600-608`) **disabled v5's ROI gate**
(`roi_thresh = -inf`, "fire all picks") and let the selector's own gate decide firing —
`fire = (max target logit > hold=0)` on a **z-scored** Δnet. So even the producer prior fired
for any source whose best candidate was merely *above-average* (z>0), ignoring v5's absolute
`roi_threshold=1.5` ⇒ systematic **over-firing** ⇒ 5%. The 3%→21% "lift" was therefore
measured through a harness that lost ~95% before any learning — **void**.

**Fix (4 edits, byte-identical-when-off preserved):**
1. `agents/v5/main.py` — hook passes the **exact per-candidate Δnet** to the selector and adds
   an opt-in `_SELECTOR_KEEP_ROI` (default off ⇒ macro agent + byte-identity untouched) that
   **keeps v5's real ROI gate**.
2. `RichSelector.forward` — returns a per-edge **delta** over the real score (0 at init), not
   a z-scored residual.
3. `rich_bc_agent` — returns `cand_score + delta` on the **real scale**, no fire mask, sets
   `_SELECTOR_KEEP_ROI=True`.
4. `macro_bc_agent` — selector signature updated for the new arg (ignored; legacy path
   unchanged).

**Verification:** rich-prior vs v5 on a recorded obs stream = **0/276 steps differ** (exact
reproduction); fixed-harness untrained control = **46%** (parity). The rig is now a valid
representation test.

## 6. The decisive retrain

Made training consistent with inference so the result is attributable:
- **Dataset rebuilt** to store the raw per-candidate Δnet (`build_dataset` now saves `score`;
  the old `rich_v5_40.npz` only had the z-scored copy). 40 v5-vs-v5 games on a **disjoint**
  seed range (30000–39; gate seeds 20000–119 — no train/test leak). 18,117 examples.
- **Consistent loss** (`combine_logits`): target column `j = delta_j + real_Δnet[.,j]`, hold
  column `= delta_0 + ROI(1.5)`. At init (delta=0) the argmax is v5's *exact* fire+select
  decision — so the model **starts at parity** and learns only corrections.
- **Best-val checkpoint** (`outputs/checkpoints/rich_bc_v5_fixed/ckpt40.pt`): init/prior val
  acc **0.976** → best **0.987**; launch-acc ~0.93–0.95 (≈ the prior).

**Result: gate = 18%.** The trained delta dropped the agent from the **46% byte-identical
prior to 18%** (−28pp) **despite predicting v5's own moves at 98.7% accuracy**.

## 7. Interpretation

A near-perfect imitator (98.7%) that loses 82% of mirror games is the signature of two
(non-exclusive) mechanisms:

1. **Compounding error at contested decisions.** The ~1.3% of mis-predicted moves are not
   uniformly distributed — they concentrate at the close, contested launches that decide
   mirror games, exactly where the exact flow-diff is already optimal. A delta that perturbs
   those loses them.
2. **BC covariate shift.** Training saw only on-policy v5 states; at gate time the
   slightly-off agent visits **out-of-distribution** states where the delta was never trained
   and is unreliable → drift compounds over a ~150-step game. (This is the standard
   DAgger motivation.)

Both say the same thing for the deadline: **a single-pass BC selection-delta over the exact
scorer regresses it**, even with information-parity and 98.7% imitation. Move-accuracy is a
poor proxy for win-rate in a competitive mirror.

This is consistent with the project's recurring wall — *a coarse/learned signal
second-guessing an exact planner regresses it* (shot-validator, arrival-horizon,
defensive-symmetry). It is **worse** than the value-rerank result ([[axis-c-value-rerank-closed]],
Cluster 10): near-tie re-ranking was *inert* (neutral); a full-selection delta is *actively
harmful*. And it confirms the structural ceiling: **BC cannot pass its teacher**, and here it
does not even hold parity once a delta is learned.

## 8. DAgger on-policy correction — the escape hatch, TESTED → does NOT recover parity

The discriminating experiment (`scripts/rich_dagger.py`): roll out the learner vs v5 (its own
state distribution), relabel those states with a **clean v5 rolled over the learner's exact
obs sequence** (obs-driven projection ⇒ the Δnet grid matches what the learner saw; labels =
v5's greedy-fired targets), aggregate onto the BC set, retrain fresh, iterate. Two iters, 24
rollouts each (+3,637 on-policy examples per iter), best-val acc held ~0.987–0.989.

**Gate (`outputs/arena/gate_richbc_dagger_v5.csv`, n=120 paired):**

| model | vs v5 |
|---|--:|
| byte-identical prior (delta=0) | **46%** |
| single-pass BC | 18% |
| DAgger iter1 | **26%** |
| DAgger iter2 | **22%** |
| iter1 vs iter2 | ~50% |

**Verdict: covariate shift was a MINOR contributor (~+6pp, 18→~24%); the delta mechanism is
the DOMINANT, fundamental cause.** On-policy correction lifted the regression slightly then
**plateaued ~20pp below the prior** (iter2 did not continue climbing — it is not converging
toward parity). A learned selection delta over the exact flow-diff regresses it *regardless of
the state distribution it was trained on*. This is the project's recurring wall, now confirmed
against its strongest counter (DAgger): **you cannot bolt a learned re-ranker onto the exact
planner without losing the contested decisions it already gets right.**

## 9. Conclusion + recommendation

- **The rich-rep BC *selection-delta* architecture is closed** (graveyard-ready): the prior IS
  v5 (46% parity, byte-identical), single-pass BC regresses to 18%, and DAgger — the
  designed-in escape hatch — only reaches ~22–26%. Both the *coarse-signal-second-guessing*
  pattern (Clusters 6/8/9) and the *can't-BC-past-your-teacher* ceiling are reconfirmed.
- **The top-tier-replay harvest via this architecture is NOT worth it.** The regression is from
  the delta mechanism, not label quality, and is robust to on-policy correction — better labels
  would not escape it.
- **Pivot to Track 2** (rule-base structural deltas mined by `scripts/replay_pulse.py` — the
  only channel that has ever gained ladder rank).
- **Genuinely untried (if a future big swing is wanted), but lower-priority:** a *non-delta*
  use of the rich features — e.g. a value head over real top-tier *outcomes* used as a
  tie-breaker only (Cluster-10's open variant, but that was already INERT), or generating the
  policy *without* the exact scorer underneath (a from-scratch planner, not a delta on v5).
  Neither is a quick win.

## 10. Reusable assets (all gated default-off / byte-identical)

- **Fixed harness:** `_SELECTOR_FN` + `_SELECTOR_KEEP_ROI` in `agents/v5/main.py` (exact
  per-candidate Δnet passed in; real ROI gate optional).
- **Byte-identity method:** replay a recorded obs stream through both agents and diff actions
  (`/tmp/byteid_rich.py` pattern) — the project-standard check for gated v5 deltas.
- `scripts/rich_bc_train.py` (`RichSelector` delta model, `combine_logits`, score-storing
  `build_dataset`, best-val) · `scripts/rich_bc_agent.py` · `scripts/producer_features.py`.
- Datasets `outputs/macro_bc/rich_v5_fixed40.npz` (current) / `rich_v5_40.npz` (stale).
- Checkpoints `outputs/checkpoints/rich_bc_v5_fixed/ckpt40.pt` (current) / `rich_bc_v5/`
  (stale, old harness). Gate CSVs in `outputs/arena/gate_richbc_*`.
