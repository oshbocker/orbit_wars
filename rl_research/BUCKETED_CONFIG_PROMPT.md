# Execution prompt — bucketed (phase-conditioned) config tests

Build and gate **opening-phase config swaps** to test whether tweaking producer knobs MID-GAME
(by phase) beats holding them fixed. Context + discipline: `rl_research/BUCKETED_CONFIG_EXPERIMENTS.md`
(LIVE log — append every run there), `memory/rating-delta-reframe-2026-06-19.md`,
`memory/leaderboard-throughput-fast-signal.md`. The thesis is already proven for two knobs
(endgame horizon clamp + terminal-phase ROI swap, both shipped phase-conditioned wins); this tests
breadth. A fixed-config sweep is a LOWER bound that misses schedule gains, so test the schedule
directly: compare the best phase SCHEDULE against the best CONSTANT (the schedule's all-buckets-equal
special case) — the schedule must beat constant beyond the noise floor to count.

## Mechanism (minimal generalization of the shipped terminal-phase swap)
The terminal swap at `agents/v5/main.py:1321` already does
`config = dataclasses.replace(config, roi_threshold=…, …)` for the last `terminal_phase_turns`
steps. Add a symmetric OPENING swap right after the horizon clamp (~`main.py:1318`, BEFORE the
terminal block so terminal wins ties at the very end of short games — though opening/terminal
windows shouldn't overlap in practice):

```python
if int(config.opening_phase_turns) > 0 and step < int(config.opening_phase_turns):
    config = dataclasses.replace(config, roi_threshold=float(config.opening_roi_threshold), ...)
```

Add to `ProducerLiteConfig` (all default to a value that makes the swap a NO-OP → byte-identical):
- B1 (2P): `opening_phase_turns: int = 0`, `opening_roi_threshold: float = 1.5` (= base; off when turns=0).
- B2 (4P): `opening_ffa_target_prod_bonus: float = -1.0` (sentinel: <0 ⇒ use base `ffa_target_prod_bonus`).
  Apply only when `player_count >= 4`. CONFIG_4P sets `opening_phase_turns` and the opening bonus.
Bucketing = opening / midgame(base default) / terminal(existing). Doses sweep via arena
`v5:opening_phase_turns=40+opening_roi_threshold=1.0` (NO code edit per dose).

## STEP 0 — de-risk cheapest-first (before any optimizer)
Hand-pick 2–3 doses and ask only: does an opening schedule beat the constant baseline?
- B1: `opening_phase_turns=40`, `opening_roi_threshold ∈ {0.8, 1.0, 1.2}` vs v5.4 (constant 1.5
  opening). Arena 2P paired side-alternated, **n≥100** vs producer AND v5 (mirror).
- If every dose ties v5.4 within ±6% AND ties the constant → roi×opening is inert (static and
  dynamic) → CLOSE B1, move to the next knob. If a dose shows a real lift → STEP 1/2.

## STEP 1 — build (gated, default-off, byte-identical when off)
Implement the opening swap + config fields above. Keep 2P (B1) and 4P (B2) separate; CONFIG_4P
must stay byte-identical for B1 (opening_phase_turns=0) and vice-versa.

## STEP 2 — verify + gate
- **Byte-identity** (reuse `/tmp/byteid_h1.py` pattern — record a real obs stream, replay through
  a fresh runtime under two configs, diff per-step moves): defaults (off) vs explicit-off = 0/N
  (2P and 4P); a dose with the swap ON on a matching stream MUST differ (>0).
- **Gate (THE metric, `scripts/arena.py`):**
  - B1 (2P): RNR-style table vs producer, v5 (mirror), producer_v2 (modeled), AND the protected
    field tamrazov_1224 / ow_proto / distance_1100, **n≥100/matchup**, paired side-alternated.
    Report paired win-rate AND margin (steps/end-score from the arena CSV). **Ship bar:** beats
    v5.4 vs producer/producer_v2 AND ≥ plain v5.3 vs every non-producer (no protected regression).
  - B2 (4P): arena `--players 4` screen + **leaderboard submit** (read rank/rating at ~1h/~3h; LB
    is the real 4P gate since local 4P is slow/noisy). Bounded downside (4P losses rating-cheap).
- **Dose-sweep** the winning direction; only escalate to an optimizer (grid → CMA-ES/BO over
  opening_phase_turns × opening_value) if a hand-picked dose already shows signal.

## Leaderboard protocol (parallel evaluator)
5 submissions/day; usable rank signal in ~1–3h. Spend slots on the best sweep candidates (esp.
B2/4P). Active-2 slot rule (`memory/active-two-slot-resubmit-rule.md`): a new submission evicts the
OLDER incumbent and ratings reset — keep the best-known agent protected. **Deadline June 23: with
1–2 days left, FREEZE on the best agent unless a LOCAL test gives a very strong signal.**

## Tracking (mandatory)
Append every run (incl. nulls) to `rl_research/BUCKETED_CONFIG_EXPERIMENTS.md` — exp ID, config
string, eval method, n, paired win-rate + margin, LB sub id/rank/rating, status. Update the Task
list (#4 B1, #5 B2). On a ship: build the bundle (`scripts/build_v5_bundle.py`), resubmit per the
active-2 rule, and record the new active-2 set. On a close: write the lesson into the log's
Closed/null section.

## Discipline (non-negotiable)
n<100 mirror = noise (±6% A/A floor @ n=60); every small-n outlier in this project regressed at
high n. Gated default-off + byte-identical when off. Hard floor = current v5.4 / plain v5.3 (the
protected non-producer field). Schedule must beat CONSTANT, not just the current value. Refs:
`agents/v5/main.py` (run_turn ~L1296–1327 phase swaps; ProducerLiteConfig ~L110–304; CONFIG_4P
~L1507), `scripts/arena.py`, `scripts/build_v5_bundle.py`.
