# Bucketed (phase-conditioned) config experiments — log

> **Status 2026-06-19:** B1 (2P opening-roi swap) **CLOSED** — thesis validated (schedule beats
> the constant +21 in mirror) but no field lift (TIE vs producer & producer_v2 at n=200), NOT
> shipped. B2 (4P) infra built + byte-identical, LB-gated, **deferred**. Opening-swap kept as
> gated default-off infra. See the Closed/null section for the full verdict + the mirror≠field
> measurement lesson.

**Thesis (user, 2026-06-19):** the optimal value of a producer knob may differ *within a game*
(by phase), by opponent, and by format — yet we hold knobs fixed per agent. A fixed-config
sweep is a LOWER bound that can MISS schedule gains (best static value can be a washed-out
compromise that ties current, while an early→late schedule beats both). So test
phase-bucketed values directly. Proof-point the thesis is real: the endgame horizon clamp +
terminal-phase ROI swap (roi 1.5→1.0 in the last 40 turns) are ALREADY shipped phase-conditioned
wins. Question = breadth (which OTHER knobs/phases have headroom).

**Mechanism:** generalize the existing terminal-phase swap (`main.py:1321`,
`dataclasses.replace(config, roi_threshold=..., ...)` for the last `terminal_phase_turns`). Add
an OPENING-phase swap (first `opening_phase_turns` steps). Bucketing then = opening / midgame
(base default) / terminal (existing). Gated default-off + byte-identical when the opening knobs
equal the base (or `opening_phase_turns=0`). Doses sweep via `arena.py` `v5:key=val` (no code
edits per dose).

**Discipline (hard-won, do not relearn):**
- THE metric = `scripts/arena.py` mirror A/B vs producer/v5 (public pool ceilings ~99%; only the
  mirror is sensitive). A/A noise floor ±6% @ n=60 → **never act on n<100**; every small-n
  outlier in this project regressed at high n.
- Hard floor = current v5.4. A schedule ships only if it beats constant beyond noise AND does not
  regress the protected non-producer field (tamrazov/ow_proto/distance) vs plain v5.3.
- Gated default-off, byte-identical when off. 2P and 4P NEVER share a schedule (format matters).
- Cheapest test first: hand-pick 2–3 doses before any optimizer (CMA-ES/BO).

**Leaderboard throughput (user insight 2026-06-19):** the "need 24h for LB signal" assumption is
wrong — strong agents have ranked top-10 within ~1–3h of submitting. We get **5 submissions/day**.
Use LB as a parallel high-throughput evaluator (esp. for 4P, which is slow/noisy locally), reading
rank ~1–3h post-submit. Active-2 slot rule: a new submission evicts the OLDER incumbent; ratings
reset on resubmit. **Near the deadline (1–2 days out, deadline June 23): freeze on the best agent
unless a LOCAL test gives a very strong signal.**

---

## Experiment table

| ID | track | knob × bucket | hypothesis | config (v5:…) | eval | n | result | LB rank/rating | status |
|----|-------|---------------|-----------|---------------|------|---|--------|----------------|--------|
| B1 | 2P | `roi_threshold` × opening | opening wants LOWER roi (grab uncontested land early; production compounds — reframe step-40 race). mid=1.5, terminal=1.0 already. | `opening_phase_turns=40+opening_roi_threshold=1.0` (+ doses 0.8/1.2) | move-diff offline | — | doses 1.0/1.2 INERT (0 move diffs); roi≤0.8 = same diff set | — | **CLOSED** (active dose = B1a) |
| B1a | 2P | `roi_threshold=0.8` × opening(40) | active version of B1 (0.8 = full opening-roi-relief effect; lower adds nothing) | `opening_phase_turns=40+opening_roi_threshold=0.8` | arena 2P n=200 | 200 | mirror **60.5%** vs v5.4 (+21, beats constant) BUT vs producer 64.0% = v5.4 64.5% (TIE), vs producer_v2 47.0% = v5.4 48.0% (TIE) | — | **CLOSED — no field lift** |
| B2 | 4P | `ffa_target_prod_bonus` × opening | 4P opening wants HIGH prod-bonus (win step-40 ship-mass race), decaying late. currently fixed 0.08. | `opening_phase_turns=50+opening_ffa_target_prod_bonus=0.16` | move-diff offline | — | doses 0.12–0.20 INERT (0 diffs); needs ≥0.6 to activate | — | active dose ≈0.6–1.0; LB-gated, deferred |

(Append a row per dose/run. Record paired win-rate AND margin from the arena CSV; for LB record
sub id, submit time, rank/rating at +1h/+3h/+24h.)

---

## Build (2026-06-19) — opening-phase swap SHIPPED gated default-off

`agents/v5/main.py`:
- `ProducerLiteConfig` (~L176): added `opening_phase_turns: int = 0`,
  `opening_roi_threshold: float = 1.5` (==base ⇒ no-op when equal),
  `opening_ffa_target_prod_bonus: float = -1.0` (sentinel <0 ⇒ keep base, 4P-only).
- `run_turn` (~L1319, after the horizon clamp, BEFORE the terminal block): opening swap —
  fires when `opening_phase_turns>0 and step<opening_phase_turns`; swaps `roi_threshold`
  always, and `ffa_target_prod_bonus` only when `player_count>=4 and bonus>=0`.
- `CONFIG_4P` UNCHANGED (keeps `opening_phase_turns=0`) → 2P/4P never share a schedule.
- Dose via arena `v5:opening_phase_turns=40+opening_roi_threshold=0.8` (no code edit/dose).

**Byte-identity (`/tmp/byteid_opening.py`, replay recorded obs stream through fresh runtime):**
- 2P off identity (default vs explicit `opening_phase_turns=0`): **0/N** ✓
- 2P no-op guard (`turns=40, roi=1.5==base`): **0/N** ✓ (swap fires but value==base)
- 4P off identity: **0/N** ✓
- Activation confirmed via extreme doses (roi 0.0/0.5/3.0/10.0 all differ on a 2P stream).

## STEP 0 move-level de-risk (offline, FREE) — prescribed doses are near-inert

Replayed the prescribed STEP-0 doses through fresh runtimes and counted per-step move diffs
vs v5.4 (env make() is stateful per-process so "same seed" gives different boards across
runs; within-run diffs are valid):

**B1 (2P) `roi_threshold` × opening (turns=40), diffs/game:**
| dose | seeds → diff-steps | verdict |
|------|--------------------|---------|
| roi=1.2 | 0,0,0 | fully inert (no opening candidate in (1.2,1.5] band) |
| roi=1.0 | 0,0,0 | fully inert |
| roi=0.8 | mixed: 0/1/1 on one board-set, 0/10/3/4 on another | most-active prescribed dose; admits early grabs on ~half of boards, then cascades |
| roi≤0.8 | roi 0.3=0.5=0.7=0.8 IDENTICAL diff count per board | lowering below 0.8 admits NO new waves → 0.8 captures the full opening-roi-relief effect |

→ The flow-diff scores in the opening cluster away from the (0.8, 1.5] band; the
"lower opening roi" lever is binary (admit one early sub-0.8 grab → cascade, or nothing).
**roi=0.8 is the decisive active dose** (lower adds nothing). Echoes the documented
fixed-roi regression-to-mean (`public-meta-refresh`: roi=1.55 closed; 57%→52% roi sweep).

**B2 (4P) `ffa_target_prod_bonus` × opening (turns=50), diffs/game:**
| dose | seeds → diff-steps | verdict |
|------|--------------------|---------|
| bonus=0.12/0.16/0.20 | 0,0 | FULLY inert — the prod-bonus tie-break flips no opening target |

→ prescribed B2 doses change ZERO moves (base 0.08; +0.04–0.12 is below the tie-break margin).
Activation scan (turns=50): bonus 0.3=0, **0.6=2–4, 1.0=2–5, 2.0=5–7** diffs/game. So an
ACTIVE 4P dose needs `opening_ffa_target_prod_bonus ≈ 0.6–1.0` (8–12× the base 0.08), NOT the
prescribed 0.12–0.20. B2 gate also requires an LB submit (4P local is noisy) → deferred.

## STEP 0 → STEP 2 GATE — B1 roi=0.8 × opening(40), arena n=100/pair (2026-06-19)

`outputs/arena/b1_opening_roi08.csv`. Win rate of ROW vs COL (side-alternated, paired):

| ROW \ COL | dose(roi0.8,t40) | v5.4 | producer |
|-----------|------------------|------|----------|
| **dose** | — | **58%** | 65% |
| **v5.4** | 42% | — | 68% |
| **producer** | 35% | 32% | — |

- **Schedule vs CONSTANT (mirror): dose 58% – v5.4 42% = +16** → beats the constant beyond the
  noise floor (±~5% @ n=100). **Core thesis SUPPORTED**: a phase schedule (lower opening roi)
  beats the constant even though the *fixed* roi sweep is regression-to-mean. First positive
  bucketed-schedule signal.
- **vs producer (ladder-relevant): dose 65% vs v5.4 68% = −3** → statistically TIED (n=100
  SE≈5%). Does NOT yet clear the ship bar ("beat v5.4 vs producer").
- ⚠️ n=100 roi result, the exact regime where this project's roi outliers regressed
  ("57%→52% roi sweep"). Escalated to n=200 + producer_v2 (below).

## STEP 2 FINAL GATE — B1 roi=0.8 × opening(40), arena n=200/pair + producer_v2 (2026-06-19)

`outputs/arena/b1_opening_roi08.csv`, all pairs n=200, side-alternated paired:

| matchup | dose win-rate | v5.4 win-rate | Δ (dose − v5.4) |
|---------|---------------|---------------|------------------|
| vs producer       | 64.0% | 64.5% | **−0.5 (TIE)** |
| vs producer_v2    | 47.0% | 48.0% | **−1.0 (TIE)** |
| mirror (dose vs v5.4) | **60.5%** (v5.4 39.5%) | — | **+21 head-to-head** |

(context: producer_v2 beats producer 68.5%; avg steps dose-vs-mirror 198, vs producer 153.)

**VERDICT: B1 CLOSED — schedule beats the CONSTANT head-to-head but gives NO field lift.**
- The mirror lift is robust and beyond noise (58% → 61% → 60.5% across three n=100/200 batches):
  a phase schedule (lower opening roi) genuinely out-plays its all-buckets-equal constant. The
  user's CORE THESIS is VALIDATED — schedules can carry signal the fixed roi sweep (inert,
  regression-to-mean) misses.
- BUT the mirror win is a **rock-paper-scissors artifact**, not a ladder gain: aggressive early
  expansion out-grabs a *passive clone of itself*, yet against the producer-tier field (producer
  AND producer_v2 — the ladder-relevant opponents) the dose is **statistically identical to
  v5.4** (64.0≈64.5, 47.0≈48.0). The extra marginal sub-0.8-ROI opening grabs neither help nor
  (much) hurt vs an exact flow-diff opponent; producer_v2's reinforce-risk pricing keeps both
  under 50%. Fails the ship bar ("beats v5.4 vs producer/producer_v2"). **NO SHIP.**
- **MEASUREMENT LESSON (new):** "schedule beats its constant in the MIRROR" is necessary but
  NOT sufficient — a head-to-head mirror win can be pure RPS that vanishes vs the real field.
  The binding test is dose-vs-producer/producer_v2 ≥ v5.4-vs-same, NOT dose-vs-v5.4. Future
  bucketed doses must gate on the FIELD matchup, treating the mirror only as an activation check.

## Code status
Opening-phase swap kept as gated default-off infra (byte-identical when off, lint+pyright clean);
NOT shipped into a bundle. `CONFIG_4P` untouched. Reusable for future bucketed doses.

## B2 (4P) status — infra ready, deferred (NOT submitted)
The 4P `ffa_target_prod_bonus` × opening swap is built + byte-identical when off. Prescribed
doses (0.12–0.20) are move-inert; the active range is `opening_ffa_target_prod_bonus ≥ 0.6`
(scan: 0.6→2–4, 1.0→2–5, 2.0→5–7 diffs/game). B2's only real gate is an LB submit (4P local is
noisy) and the active-2 slot rule makes a speculative submit costly. Given (a) B1's null +
the RPS lesson (a 4P mirror/arena bump likely won't transfer), (b) `rating-delta-reframe` showing
4P is already rating-net-positive, and (c) the June 23 deadline freeze rule, a 4P LB slot is NOT
recommended without a stronger prior. **Deferred to the user — do not spend a slot speculatively.**

## Closed / null
- **B1 (roi × opening, 2P)**: CLOSED 2026-06-19. Mirror +21 vs constant (thesis validated) but
  TIE vs producer (64.0 vs 64.5) AND producer_v2 (47.0 vs 48.0) at n=200 → no ladder lift. Lesson:
  mirror win ≠ field win (RPS). roi confirmed regression-to-mean as a lever in ANY bucket.
- **B1 prescribed doses (roi 1.0/1.2; ffa-bonus 0.12–0.20)**: move-INERT (0 diffs) — never gate
  these; flow-diff opening scores are bimodal (>1.5 fired / ≤0.3 marginal), the (0.3,1.5] band
  is empty, so all roi≤0.8 are the SAME agent.
