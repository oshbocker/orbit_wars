# Contestation overlay — findings + robustness plan (2026-06-17)

**Design philosophy:** an agent built to beat producer/producer-clones by *exploiting
producer's structure* rather than out-expanding it. Producer is a greedy, single-ply,
**capture-minimal** flow planner that models all opponents as **do-nothing (level-0)**.
The contestation overlay is the level-1+ best-response: we run producer's *exact* planner
internally (we have its code; it cannot simulate us), predict which planets it will
capture-thin, and **snipe the freshly-thinned planets** for a discount.

## What shipped (default-OFF, gated, byte-identical at `contest_waves=0`)

- `agents/v5/orbit_lite_v5/contestation.py` — `plan_contestation_waves()`.
- `agents/v5/main.py` — config knobs `contest_waves` / `contest_delay` /
  `contest_roi_threshold` / `contest_capture_overhead` / `contest_opp_waves` /
  `contest_verify` (reserved); `_opponent_reactive_status` gained `opp_waves_override`;
  overlay hooked into `run_turn` after the base plan, before `apply_private`.
- Arena-tunable: `scripts/arena.py --agents "v5:contest_waves=2,producer"`.

### Mechanism (verified)
Combat is two-stage (`top − second` among arrivals, then survivor vs garrison —
confirmed in `v2/fast_env.py`). Producer sends `capture_floor = ceil(defenders + overhead)`,
so a freshly captured neutral holds **~2 ships** (overhead + 1 turn production) — verified:
capture neutral(g=5) with 7 → owns it with 2. A fleet arriving the turn **after** the
capture clears that thin residual cheaply (5 ships beats a 3-ship thinned planet, keep 2).
Same-turn contestation is *expensive* (must out-ship `producer_fleet + garrison`) → the
cheap, repeatable exploit is **counter-snipe-after-capture**, not same-turn denial.

The overlay reuses `_opponent_reactive_status` (runs producer's planner per opponent seat,
projects their captures through the engine-exact `garrison_status`, snapshot/restored →
side-effect-free), finds planets flipping mine/neutral → opponent within the horizon, sizes
snipes via `capture_floor` against the **reactive (post-capture, thin)** projection, scores
them with the **exact flow-diff** (`score_candidates`), and greedy-selects from the
**post-base-plan leftover budget** (so the base expansion never weakens → no tempo tax).
Distinct from the INERT `opp_inject_waves` (Cluster 11), which fed the opponent prediction
into our *defense*; this consumes it as new *offense* the base shortlist is blind to.

## Gate results (THE metric — `scripts/arena.py`, side-alternated paired seeds)

| matchup | plain v5 | contest_waves=2 | delta | n (paired) |
|---|--:|--:|--:|--:|
| vs **producer** | 58.7% | **69.3%** | **+10.7** | 150 |
| vs **producer_v2** | 36.0% | **46.0%** | **+10.0** | 150 |
| vs **v5** (mirror) | 50.0% | 53.3% | +3.3 (noise) | 150 |

**Confirmed robust, cross-clone +~10 pts.** Caveat learned the hard way: an early n=60
read said the v5-vs-producer baseline was 45% (→ illusory +24); the paired n=180 baseline
is 57–59% (→ real +10.7). The project's "never act on n<100 mirror" rule held again.

- The edge tracks **opponent-model fidelity**: large vs agents we simulate well (producer/
  producer_v2), noise vs v5 (which we approximate). Since `orbit_lite` is now a public
  package, the ladder pool is producer-clones → the "vs producer" number is ladder-relevant.
- Overlay fires **sparsely** (~5–6 snipe waves over a 232-turn game vs producer) yet moves
  win rate +10 — each snipe steals a high-value planet at positive tempo.
- `contest_verify` (2nd-ply re-defense filter) is a **reserved no-op** (default 0); the
  mechanism wins +10 without it, so it's a refinement, not a blocker.
- **Dose-4 vs producer** and **downside vs tamrazov** gates: see `outputs/arena/arena.csv`
  (`v5_contest_waves4`, `v5_contest_waves2` vs `tamrazov_1224`).

## The risk this raises → detection

The overlay's value is conditional on the opponent being producer-family. Against a
non-producer, a snipe sized to a *predicted* capture that never happens hits a full-garrison
planet undersized → **wasted leftover ships**. So we want to detect "am I playing producer?"
and gate the overlay.

**Detection is feasible and the signal separates** (1-game probe, producer-prediction vs
actual opponent launches, source+size match):

| opponent | match | class |
|---|--:|---|
| producer_v2 | 77.5% | producer-family |
| producer | 63.5% | producer-family |
| ow_proto | 27.3% | **different** |
| tamrazov_1224 | 14.1% | **different** |

Producer-family 60–77% vs structurally-different 14–27%. Two refinements the probe exposed
(real producer only hit 63% because the predictor used *v5's* config and a strict metric):
1. **Predict with producer's config**, not v5's (`_opponent_reactive_status` currently uses
   our config). Ladder clones run the public `orbit_lite` config.
2. **Score on source-SET overlap (Jaccard of `from_planet_id`)**, not source+size —
   producer_v2 differs only in *sizing*, and we *want* to snipe it (+10). Selection-overlap
   lumps both clones as "snipeable" while still rejecting tamrazov/ow_proto.

## Robustness plan — tiers (for the 6/23 deadline)

**Recommendation: ship Tier 1, build Tier 3 as a hedge, defer 2 & 4.**

- **Tier 1 — Detect-and-gate (SHIP FIRST; dominant, ~zero downside).** Per-seat fidelity
  EMA gates the overlay. vs producer-family → +10; vs anything else → overlay disables →
  *exactly plain v5*. Makes contest+detector **strictly ≥ plain v5 everywhere**. The only
  cost of a wrong call is a few leftover-ship snipes before fidelity converges; bias the
  gate toward OFF (false-positive wastes ships; false-negative only forgoes upside).

- **Tier 2 — Multi-model ensemble (defer; likely already covered).** The source-set
  detector lumps producer_v2 in. Only needed if replays show clones whose *selection*
  diverges; then keep a small ensemble of opponent models and snipe with the best match.

- **Tier 3 — Symmetric anti-snipe defense (cheap hedge; build alongside T1).** The exploit
  is symmetric — a swarmer/half-drainer can counter-snipe *our* thin captures. vs confirmed
  low-fidelity opponents, flip on a gated defensive knob (hold fresh captures thicker /
  small reserve). **Gated on low fidelity**, so it sidesteps the half-drain tempo-tax that's
  CLOSED vs producer in the mirror (Clusters 7/9/12) — we only pay the reserve where it pays.

- **Tier 4 — Learned archetype classifier (defer past deadline).** Train on replay
  fingerprints (the `replay_pulse` send-fraction/waves-per-turn fingerprinter exists) →
  archetype → pre-tuned config. Full opponent-modeling vision, too heavy for 6/23.

## 4P FFA extension — CLOSED (2026-06-18, snipe value does not transfer to FFA)

The hypothesis (per-seat detection is *especially* valuable in 4P: snipe the clones at the
table, base-play the rest) was built end-to-end and measured. **It does not pay** — the
contest overlay's 2P value is specific to the zero-sum 2-player structure and does not
survive in FFA. Folded: `CONFIG_4P` keeps `contest_waves=0` (4P = plain v5.3, byte-identical
to v5.4). The machinery is all retained default-off for a future reopen.

**What was NOT the problem (both cleared):**
- *Timing.* The 4P risk was that producer's full planner runs once per opponent seat (3× in
  4P). Measured: ~46 ms/turn mean, **218 ms worst single turn** across long + all-clone games
  — far under the 1 s actTimeout, zero overage drain. No per-seat-skip / cadence / lower-wave
  mitigation needed; the per-seat machinery scales fine.
- *Detection.* The 2P precision-EMA gate (alpha=0.9 / thr=0.55 / min_obs=8) **transfers
  cleanly** to 4P (re-calibrated on 6 games/opp, 3 seats each): ON-fraction producer 0.78,
  producer_v2 0.65 (clones) vs tamrazov 0.01, ow_proto 0.00, distance_1100 0.06, enders 0.00.
  clone_min 0.65 ≫ other_max 0.06 — *cleaner* than 2P (producer_v2 was 0.43 there; denser 4P
  boards give more launch signal). No 4P gate override needed.

**What IS the problem (strategy):** in a 4-way a snipe spends leftover ships to grab a planet
that becomes **our** exposed frontier, and the *specific* planet a clone thins depends on the
other two live seats' moves — which our level-0 (`do-nothing`) opponent model cannot see — so
snipe targets mis-fire and we overextend. Note the detector measures *source*-set fidelity
(which stays high, 0.78), but the snipe needs the *target* right, and that degrades in FFA.

**Gate results (`scripts/arena.py` paired + paired variant harness, our seat-0 swapped across
contest configs on identical boards + opponents):**
- *Plain enabling* (`contest_waves=2`, no board gate): clear regression — mean rank 1.65 vs
  1.45, win 35% vs 55%, end-score 720 vs 3027 vs `contest_waves=0` (paired n=24 + n=60).
- *Board-position gate* (`contest_ffa_strike_rank`, "strike only from strength" — snipe only
  when our prod+0.025·ships ranks in the top-N live players; 4P-only, 2P byte-identical):
  narrowed it but **did not clear the bar**. Best variant = rank 1 (leader-only), pooled
  n=180 vs off: **net −3** (better 32 / worse 35 / tie 113):

  | table | off win | sr1 win | paired sr1 vs off (n) |
  |---|--:|--:|---|
  | mixed (producer, producer_v2, tamrazov) | 7% | 13% | +7 (better 10 / worse 3, n=70) |
  | clones (producer, producer_v2, producer) | 29% | 27% | −4 (better 15 / worse 19, n=70) |
  | sparse (producer, tamrazov, enders) | 68% | 52% | −6 (better 7 / worse 13, n=40) |

  The leader-gate helps on contested 2-clone tables (+7) but **hurts when we're already
  cruising** (sparse: off wins 68%, sr1 drops it to 52%) — it fires precisely when an FFA
  leader should *consolidate*, not attack. `rank=2` (top-2) was worse than `rank=1`.

**Reopen idea (future):** gate snipes on a *contested* lead (leader by a SMALL margin), not
any lead — the +7 on the contested 2-clone table is the only signal worth chasing, and it's
only relevant if the real 4P ladder is clone-heavy (the local screen can't tell us; per the
project rule the ladder is the final 4P gate). Cheaper redirect: Tier 2 (producer_v2 ensemble)
lifts a *measured* 2P gap and is self-contained in 2P.
