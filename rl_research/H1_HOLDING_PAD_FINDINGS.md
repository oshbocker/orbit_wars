# H1 4P holding-pad — findings (2026-06-19) → NEGATIVE, gated default-off

**Hypothesis (from the 06-18 pulse / own-replay diagnostic).** Our 4P rating leak is
"capture-then-collapse": expand to ~5.5 planets by step 60, can't hold them under
multi-opponent pressure, eliminated ~step 114 (4P win 30% vs 2P 51%). Fix = make
contested captures HOLD their immediate counter-wave.

## What shipped (default-OFF, gated, byte-identical when off / in 2P)

- `agents/v5/main.py`: knobs `hold_margin_beta` (default 0.0) + `hold_min_opponents`
  (default 2); helper `opponent_holding_pressure()` (per-target largest-single-opponent
  reachable garrison + count of distinct reachable opponents).
- Mechanism: in 4P only, inflate `capture_floor` of a target reachable by
  `>= hold_min_opponents` distinct live opponents by
  `hold_margin_beta * largest-single-opponent reachable garrison` (combat is
  `top - second` then survivor-vs-garrison ⇒ the largest single opponent is the binding
  counter-wave). Reuses the existing `capture_floor` `reinforcement` plumbing.
- `scripts/arena.py`: added `prodshare80_{i}` to the 4P CSV — the near-deterministic
  outcome predictor from the replay diagnostic, a cheap 4P screen. (KEEP — general
  instrument.)
- **Byte-identity verified** (`/tmp/byteid_h1.py`, recorded-obs replay): 2P β=8 vs off
  = 0/137; 4P off-default vs explicit-0 = 0/169; 4P β=8 vs off = 12/169 (active, sparse).

## Gate (THE metric — `scripts/arena.py --players 4`, table = [H1, plain v5, producer, producer_v2], n=160 paired seeds/dose)

| dose β | H1 win | v5 win | H1 mean rank | v5 mean rank | H2H P(H1>v5) | H1 P(prodshare@80≥.35) | v5 same |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.5 | 29% | 36% | 1.81 | 1.71 | 46% | 27% | 28% |
| 1.0 | 29% | 36% | 1.81 | 1.70 | 46% | 20% | 28% |
| 2.0 | 32% | 32% | 1.81 | 1.77 | 48% | 22% | 28% |

**Verdict: NEGATIVE at every dose.** H1 ranks at/below plain v5 (head-to-head 46–48%),
win rate equal-or-lower, and the prod-share@80 screen shows it does **not** improve our
early board position (equal/worse — β=1.0 actually drops it 28%→20%). No dose clears
parity; the cheap instrument corroborates the win-rate read.

## Why it failed (the recurring conservatism-regresses pattern)

The send size is fixed at full `safe_drain`, so inflating `capture_floor` does **not**
make us send more — it only **gates out** contested captures whose source can't over-fund
the inflated floor. That collapses to "decline contested captures" = MORE conservative on
exactly the contested ground, which **cuts expansion** — the opposite of what wins. The
replay data already showed losses *under*-expand (planets gained 20→60: 4P W +3 vs L +1)
and winners *out-expand* (peak 16.5 vs 5.5 planets), not over-fund individual captures. So
holding harder is the wrong direction: it's Cluster 9 (`defense_size_beta`) / Cluster 12
(`reserve_frac`) again — any "hold back / decline in contested ground" delta loses tempo
and regresses in 4P. The prod-share@80 drop is the mechanism made visible.

## What remains distinct (NOT tested here)

- **Literal over-send** (drain a source *below* its safe-held level to launch a fatter
  capture fleet) is a different mechanism. Low EV: over-funding a single frontier capture
  deepens the overextension that *is* the loss (winners out-expand, they don't over-fund),
  and it exposes the over-drained source — likely just moves the collapse.
- **H2 target-selection quality** (capture into *holdable / less-contested* ground; a flow
  *redirect*, not an over-send) is mechanically opposite to H1 and still open — the data
  says win by out-expanding into ground you keep, not by hoarding on contested ground.

**Disposition:** code kept gated default-off (byte-identical), `prodshare80` instrument
kept. NOT shipped. H1 = closed.
