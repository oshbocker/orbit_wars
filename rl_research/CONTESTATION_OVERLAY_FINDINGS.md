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

**4P note:** per-seat detection is *especially* valuable in 4P (snipe the clones at the
table, base-play the rest, defend only vs contest-capable seats). The overlay has only been
gated in 2P; 4P needs its own gate before shipping there.
