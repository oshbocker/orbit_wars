# H2 4P target-selection (flow redirect into holdable ground) — Step-0 de-risk → NEGATIVE, NOT BUILT

**Date 2026-06-19.** H2 was the surviving 4P lever after H1 (holding-pad over-send)
closed NEGATIVE: keep expanding, but steer the SAME flow toward less-contested /
holdable targets instead of contested ones we lose. Per the build plan, Step 0 is an
offline replay de-risk *before any code* — and it is the gate on whether to build.

## The Step-0 question (the brief)

In our 4P LOSSES, at the turns where we capture a planet that later flips, was there a
LESS-contested **holdable** alternative target in the **same candidate set** (the
planner's `build_target_shortlist`) that we passed over? If that fraction is low → no
renderable gap → do not build (the H1 trap). If high → proceed.

## Method (faithful to the planner, not a numpy proxy)

`/tmp/ow_analysis/step0_h2.py` + `/tmp/ow_analysis/step0b_h2.py`. For each cached 4P
replay (v5.3 sub 53615604, our seat = `Oshbocker`): run the **real `agents.v5`
runtime** over our seat's obs stream; monkeypatch the shipped `build_target_shortlist`
to record, every turn, the shortlist planet-id set and the per-planet contestation
(`opp_count` = #distinct live opponents that can reach it) from the **shipped
`opponent_holding_pressure()`** helper. Independently build the replay ownership
timeline to detect capture events (owner→us) and flips (lose within 40 turns).
Classify each capture at its decision board (step t−1). 16 LOSS games (419 captures) +
16 WIN games (599, as control). `contested = opp_count ≥ 2`; `holdable = opp_count ≤ 1`.

## Result — the literal metric says PROCEED, but it is a saturation artifact

| metric | 4P LOSS | 4P WIN (control) |
|---|--:|--:|
| contested+flipped captures w/ a holdable alternative in shortlist | **97%** (68/70) | 100% (30/30) |
| holdable alternative available (all contested captures) | 96% | 100% |

The headline 97% is **non-discriminating**: a holdable alternative is in the shortlist
~always, in BOTH wins and losses. A feature that is ~always true regardless of outcome
cannot be the W/L lever. So the literal "high fraction → proceed" reading is a false
positive (the candidate set is simply broad).

## The decisive test — durability gap (does redirecting to holdable ground buy hold?)

H2 only helps if holdable (≤1 opp) captures actually **survive better** than contested
(≥2 opp) captures *in the games we lose*. They do not:

| flip rate by contestation | 4P LOSS | LOSS pre-elim* | 4P WIN |
|---|--:|--:|--:|
| holdable (≤1 opp) | 90% (310/343) | **88%** (236/269) | 45% (247/555) |
| contested (≥2 opp) | 92% (70/76) | **90%** (54/60) | 68% (30/44) |
| **durability gap** | **2 pts** | **2 pts** | 23 pts |
| share of our captures already holdable | **82%** | 82% | 93% |

\*pre-elim = capture's full 40-turn survival window ends before our elimination step,
so a flip is a real loss-of-hold, not the terminal wipe.

## Why H2 is a NO-GO (three independent reasons)

1. **No durability advantage for holdable ground in losses.** Holdable flips 90% vs
   contested 92% (pre-elim 88% vs 90%). Even *early* holdable captures don't hold —
   the collapse is **global**, not target-specific. Redirecting flow toward holdable
   ground buys ~nothing where we actually lose.
2. **We already pick holdable ground 82% of the time.** H2's premise — "we pass over
   holdable alternatives to grab contested ground" — is false. Contested captures are
   only 18% of losses; redirecting them to ground that also flips ~88% changes nothing.
3. **What separates W from L is not target contestation.** In WINS contestation
   genuinely predicts durability (45% vs 68%) and we win anyway; in LOSSES even
   holdable ground flips ~88–90%. The differentiator is board strength / out-expansion
   under convergent multi-opponent pressure (winners peak 16.5 planets, losses ~5.5 —
   06-18 pulse + H1), which target-redirection cannot touch.

## Disposition

**NOT BUILT.** Step 0 fails: the gap is illusory (saturated availability + ~zero
durability advantage for holdable ground in the games we lose). This is the H1 trap in
mirror image — the literal threshold said "proceed," the substantive lever is absent.
Consistent with H1 (`H1_HOLDING_PAD_FINDINGS.md`) and the 4P diagnostic
(`memory/pulse-2026-06-18-4p-holding.md`): the 4P leak is structural under-expansion /
inability to sustain hold under convergent pressure, **not** where/which targets we
commit to. Both "hold harder" (H1) and "steer to holdable" (H2) are non-levers.

Caveat on n: 16 loss games / 419 capture events (not an arena win-rate A/B, so the
n<100 mirror rule does not apply); the structural read is robust and triangulates with
H1 + the pulse + the WIN control. Analysis scripts kept under `/tmp/ow_analysis/`.

**Still genuinely distinct / untested:** the 4P leak being *expansion capacity* itself
(out-expand into more total ground earlier, not redistribute the same flow) — but that
is "send more / expand faster," adjacent to the closed over-send family, and the data
says losses under-expand because they get *pushed off*, so faster expansion alone likely
deepens the same collapse. 2P refinement (confidence-scaled RNR contestation) remains
the higher-EV open track.
