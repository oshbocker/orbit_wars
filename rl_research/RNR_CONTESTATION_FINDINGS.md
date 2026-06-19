# RNR (confidence-scaled Restricted-Nash-Response) contestation — CLOSED at Step 0

**Date:** 2026-06-19
**Status:** CLOSED — de-risked NEGATIVE at Step 0 (offline EMA-separability gate). **Not built.**
**Verdict:** STOP. No monotone confidence→commitment map of the source-set precision-EMA can
separate producer_v2 from the protected non-producer population. The binary gate at
`contest_fidelity_threshold=0.55` is already the near-optimal single cut. This is the
H1/H2/Tier-2 trap the plan explicitly warned against — *don't ship a knob the offline signal
says can't separate.*

## The idea (proposed, not built)
Replace the binary contestation gate (`_OpponentTracker.gated_seats`, `agents/v5/main.py`) —
per-seat source-set precision-EMA ≥ 0.55 ⇒ overlay fully ON, else fully OFF — with a continuous
commitment ∝ confidence: scale the snipe commitment (leftover-budget fraction /
`contest_roi_threshold` aggressiveness in `plan_contestation_waves`,
`orbit_lite_v5/contestation.py`) by a monotone map of the precision-EMA, with a hard floor =
plain v5.3 (confidence below floor ⇒ zero snipe, byte-identical to v5.3). The hoped-for upside:
the ~+4.5 win-rate the Tier-2 ensemble chased — producer_v2 is gated ON only ~43% of turns by
the binary threshold (vs ~99% for plain producer), and a continuous map below 0.55 would
recover that forgone exploitation. Theory: safe-exploitation (Ganzfried–Sandholm) / RNR–DBR
(Johanson) — risk only what the modeled behavior has conceded; the overlay already draws from
post-base-plan leftover budget so the downside vs a mis-modeled opponent stays bounded.

## Step 0 method (the gate on whether to build)
`/tmp/ow_analysis/rnr_calib.py`. For each opponent, record a real seat-0 v5.4 obs stream
(contest_waves=2, detect=1, single base-producer model), replay it through a FRESH v5 runtime,
and after every step read the EXACT per-(seat,model) precision-EMA the gate consults:
`rt.memory.opp_tracker.fid[(seat,0)]` with `turns_observed`. Collect every post-warmup
(turns_observed ≥ `min_obs`=8) EMA value per opponent (alpha=0.9). **6 games/opp**, 600–830
post-warmup samples each. The headroom question: *is there a monotone ramp floor τ₀ where
producer_v2 has real EMA mass that the binary gate (≥0.55) discards, while the protected
non-producers have ~none there?*

## Result — robust NEGATIVE (n=6 games/opp)

Post-warmup precision-EMA distribution (the gate's signal):

| opponent | n | mean | p05 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|---|
| producer | 688 | 0.901 | 0.613 | 0.855 | 0.958 | 0.993 | 0.999 |
| **producer_v2** | 827 | **0.513** | 0.177 | 0.376 | **0.543** | 0.667 | 0.786 |
| tamrazov_1224 | 445 | 0.421 | 0.203 | 0.334 | 0.417 | 0.476 | 0.699 |
| ow_proto | 511 | 0.165 | 0.048 | 0.101 | 0.145 | 0.194 | 0.396 |
| distance_1100 | 601 | 0.396 | 0.147 | 0.276 | 0.425 | 0.505 | 0.713 |

The decisive finding — producer_v2's EMA **overlaps** tamrazov_1224 and distance_1100. Its
lower half (p05–p50 = 0.177–0.543) sits squarely in their territory (tamrazov mean 0.421,
distance mean 0.396). Only ow_proto separates cleanly (0.165).

**The headroom-band test (the real test).** RNR's ONLY additive region is the sub-0.55 band —
above 0.55 the binary gate already fires fully, so a confidence ramp there only *reduces*
commitment (strictly more conservative than v5.4 = pure loss). In every sub-threshold band the
protected non-producers carry MORE mass than producer_v2:

| band [τ₀, 0.55) | producer_v2 mass | max non-producer mass | net |
|---|---|---|---|
| [0.30, 0.55) | 0.346 | 0.609 | **−0.263** |
| [0.35, 0.55) | 0.306 | 0.542 | **−0.236** |
| [0.40, 0.55) | 0.224 | 0.378 | **−0.154** |
| [0.45, 0.55) | 0.150 | 0.205 | **−0.055** |

Lowering the floor below 0.55 to capture producer_v2's forgone turns captures *more*
tamrazov+distance turns than producer_v2 turns → net-negative everywhere.

## Why both ramp directions fail (the structural wall)
A monotone confidence→commitment ramp is still just a (soft) threshold on the SAME 1-D EMA
signal — it adds no separating power the hard threshold lacks.
- **τ₀ < 0.55** (to recover producer_v2's forgone ≥43%): commits non-zero snipes vs
  tamrazov+distance on more turns than vs producer_v2 (net-negative in every band). At EMA≈0.50
  you cannot tell producer_v2 from tamrazov — both have substantial mass there. A snipe vs a
  non-producer is sized to clear a *predicted* thin residual that never materializes, so it
  arrives undersized and bounces, wasting leftover ships — exactly the regression in a symmetric
  2P duel where consistency is paramount. → **regresses the protected population** (H1/H2/Tier-2 trap).
- **τ₀ ≥ 0.55** (to protect non-producers): zero commitment below 0.55 means producer_v2's
  in-[0.55,0.70] turns (~30% of its turns) and some producer turns get LESS commitment than
  v5.4's full-on binary → **forfeits the validated +16.2 (producer) / +11.7 (producer_v2)
  for nothing.**

Either way RNR cannot beat the binary gate. The cut at 0.55 is already optimal: above it
producer-family dominates and the protected set is sparse (tamrazov 0.19, distance 0.16,
ow_proto 0.02); below it the populations interleave.

> NOTE: the script's auto-heuristic printed "PROCEED at floor ~0.70" — a FALSE POSITIVE. It
> only checked that non-producers are ≤10% above a floor, ignoring that any floor ≥0.55 is
> strictly more conservative than the binary gate (which already fires fully at ≥0.55), so it
> has zero incremental upside and real downside. The correct test is the headroom-band table
> above (net mass in the sub-0.55 band), which is negative for every τ₀.

## Decision
**STOP. Do not build `contest_rnr_beta`.** The single 1-D precision-EMA signal is the same one
the binary gate thresholds; a continuous map of it cannot separate producer_v2 from the
protected non-producers. The only route to producer_v2's forgone +4.5 was a *second* opponent
model (the Tier-2 ensemble), which itself closed negative (regressed −15.8 vs plain producer).
v5.4 single-model binary-gate STANDS unchanged. No code shipped; calibration driver kept at
`/tmp/ow_analysis/rnr_calib.py` and stream cache at `/tmp/ow_analysis/streams/`.

## Reopen condition
Only a *different signal axis* (not a monotone transform of the same source-set precision-EMA)
could separate producer_v2 from tamrazov/distance — e.g. source-set *size* fidelity, or a
target-set / sizing feature that the current SOURCE-only precision throws away. That is a new
detector, not an RNR knob, and would have to clear its own offline separability gate first.
