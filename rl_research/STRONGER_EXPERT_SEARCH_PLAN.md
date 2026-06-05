# Stronger Expert Search — Plan (AutoGo-informed)

**Date:** 2026-06-04
**Status of agent:** best = ExIt `v2_exit_a100/ckpt_000020.pt` (iter 20), **77% vs apex @ n=60** (submitted). Heuristic ExIt plateaus ~70–77% with the current shallow, passive-opponent expert.

This plan integrates lessons from Eric Jang's **AutoGo** (Apr 2026) — a from-scratch AlphaGo rebuild — into our ExIt loop. Sources: [github.com/ericjang/autogo](https://github.com/ericjang/autogo), [evjang.com/autogo](https://evjang.com/2026/04/28/autogo.html), [Dwarkesh × Eric Jang](https://www.dwarkesh.com/p/eric-jang).

## AutoGo lessons that bear on our stall

1. **Value grounding is make-or-break.** The value net must be trained on *real game outcomes* (terminal scores). "MCTS can be *worse* than the policy net early in training" when the value is poor or leaves are off-distribution. Fix: force ~10% of self-play games to play fully to terminal so the value head sees endgames.
2. **Search is a policy-improvement operator** — exhaustive per-move search → train the policy to imitate the improved move. One clean supervised label per move; this is the credit-assignment escape hatch that model-free PPO lacked (our PPO stalled 0–10%; ExIt reached 77%).
3. **Rollouts to terminal are unnecessary once the value is good** — a learned leaf value replaces them (KataGo dropped AlphaGo-Lee's random rollouts). A *cheap symmetric rollout policy* is a fine stepping stone before a grounded value exists.
4. **Init from supervised/BC, not tabula rasa** — we already BC from apex. ✓
5. **A small net amortizes intractable search** — capacity pays off *when fed good distillation targets*, unlike PPO where capacity wasn't the bottleneck. Validates the embed-128-vs-256 ExIt A/B track.
6. **ResNets > transformers at low data/budget** — that's grid bias for Go; attention-over-planets is the right inductive bias for our planet *set*, so keep the architecture.

## Why our two prior search experiments failed (now explained)

- **Neural-value leaves collapsed 77%→0%:** `_reconstruct_leaf_state` (v2/search.py) builds a **positionless, fleet-less** `GameState` → OOD for the value head; and `evaluate_state` already counts in-flight `fleet_events` (src/simulator.py:174-180), so the heuristic was strictly the better leaf scorer. Per lesson #1, the value was *ungrounded and leaves were OOD* — not "neural value is wrong."
- **Two-player turn-1 search degraded 77%→40%:** opponent fired only once (sparse) → lookahead still overrated aggression after step 1. AutoGo runs the opponent searching/acting *every* move. Our gap: `sim_step` (src/simulator.py:85) does production + arrivals + combat with **no opponent launches ever**.

## The plan (ordered; each gated on `scripts/eval_fast.py` n=60, side-alternated, paired seeds)

### Phase 1 — Every-step in-sim rollout opponent — ❌ CONFIRMED NEGATIVE (2026-06-05)
Built + ran 60 ExIt iters warm-started from the 77% ckpt (`v2_exit_rollout_a100`). On the trusted scorer (`eval_fast.py`, n=60, side-alternated, paired): iters 10..60 = 37,33,37,40,45,57,33,43,58,50,48 → **mean ~45%, peak 58%, regressed the 77% agent and never recovered.** Same direction as the turn-1 version (→40%).

**Diagnosis:** in the symmetric rollout, after a candidate move *our own* continuation is the cheap rollout heuristic, not our strong policy → candidates that need strong follow-up are undervalued → search biases to conservative moves → distills a more passive policy → loses to aggressive apex. The weak hand-coded rollout is the poison. **AutoGo lesson #3:** never evaluate leaves with a weak rollout — use a learned value. Phase 1 reintroduced exactly what AlphaZero deleted. Keep `rollout_search` default OFF; keep the code (`rollout_launches`/geometry reusable for Phase 3 data gen).

**Process bug found:** in-training `run_periodic_eval` (n=20) showed 90–95% while eval_fast showed 33–58% on the same ckpts — it is NOT side-alternated/paired, so it inflated ~2× and was unreadable live (cost a blind 10.6h run).

### Phase 1.5 — Fix the eval gate — ✅ DONE (2026-06-05)
`run_periodic_eval` (`v2/train.py`) now mirrors `eval_fast`: plays on `FastOrbitWars`,
side-alternated, paired seeds (base `eval_seed=20000`, shared with `eval_fast` so the
in-training number is directly comparable), parallel via `eval_workers`. Added
`eval_seed`/`eval_workers` to `V2EvalConfig`; `configs/v2_exit.yaml` → `eval_games: 40`,
`eval_workers: 6`. The game loop is character-identical to `eval_fast._eval_game`, so the
numbers match by construction (verified: random model loses to apex / ties random; parallel
path runs).

**What the fix immediately revealed (important):** the old non-alternated eval was
inflating. On the trusted side-alternated scorer at seed 20000, **iter-20 is ~33% as P0 /
13% as P1 (≈23% combined)** — NOT 77%. And win-rate is **high-variance across map seeds**:
the same deterministic agent scores 33% on the 20000 batch but 83% (P0) on the 31000 batch.
The two decode paths are *identical* (896 steps, 0 diffs: `decode_actions` ==
`decode_sampled_actions` deterministic), so the submitted agent == the training policy — the
spread is genuine map variance, not a bug. **Implication:** the "77% @ n=60" headline was
optimistic / seed-or-side-dependent; re-measure the true baseline with a high-n multi-seed
`eval_fast` run on Colab (local CPU eval is ~20s/game → too slow for n≥60). The eval-gate
fix is what makes any of this readable.

### Phase 2 — Positional simulator — ✅ DONE (2026-06-05)
`SimState`/`fleet_events` now carry per-fleet straight-line geometry
`(arrival_step, target_id, owner, ships, launch_step, sx, sy, tx, ty)` (`launch_step=-1` =
positionless sentinel). `add_fleet_event(..., src_xy, dst_xy)` records geometry; both search
call sites (`v2/search.py`, `src/search.py`) pass it. New `fleet_position_at(event, step)`
linearly interpolates a fleet's `(x, y, angle)`. `_reconstruct_leaf_state` (`v2/search.py`)
now rebuilds in-distribution fleets at leaves instead of dropping them. Combat/scoring read
only the first four fields, so the heuristic search path is **bit-identical** (verified:
`evaluate_state` geom == no-geom).

**Readiness diagnostic (offline, iter-20):** Phase 2 fixed the OOD collapse — neural leaf
scores are no longer degenerate (std 0.765, range [−1.06, 0.87]) and the value head is
grounded (corr(pred_value, terminal_outcome) = **0.389**). BUT at the fine grain of ranking
*sibling candidate moves from one position*, neural vs heuristic spearman ≈ 0.008 ± 0.421 —
i.e. the value is fine globally but too noisy to rank near-equal candidates alone.

### Phase 3 — Grounded learned value (the real win) — BLEND, NOT SWAP
The value head is **already** grounded: `play_single_game` runs to terminal and
`train_epoch` already does `value_loss = MSE(out.value, terminal_outcome)`, so the plan's
"force ~10% to terminal" is effectively at 100%. The remaining work is to *use* it at
leaves — but the readiness diagnostic says a pure heuristic→neural swap is risky (sibling
ranking ≈ uncorrelated). So: **blend** the two leaf scores (z-score each across the candidate
set, `score = (1-w)·z(heur) + w·z(neural)`, small `w`, A/B-able via a `value_leaf_blend`
knob), warm-start from iter-20, gate on the fixed eval at high n. This is the disciplined
retry of the idea that previously collapsed, now safe against both prior failure modes
(OOD leaves → fixed by Phase 2; ungrounded/over-trusted value → blend).

### Phase 4 — Genuine policy-improvement search
- (a) **Sequential multi-planet search** — update `SimState` between owned planets so coordinated multi-planet plays become visible (currently per-source greedy). Low risk.
- (b) **Shallow fixed-width tree / beam** with the opponent acting at each node and the grounded value at leaves — closest thing to real MCTS the simultaneous-move, huge-branching structure allows. Only after (a) + Phase 3.

### Separate track — embed-128 vs 256 capacity A/B in the ExIt regime
Configs `configs/v2_exit_embed128.yaml` / `v2_exit_embed256.yaml`, run via `scripts/run_embed_ab.py` on Colab once Phase 3 yields high-quality targets. Lesson #5 predicts capacity finally pays off here.

**Sequencing rationale:** 1→2→3 is a dependency chain that fixes both documented failures and installs the one non-negotiable (grounded value + in-distribution leaves). Phase 4 compounds only once the value is trustworthy — deeper search on a passive opponent / ungrounded value is exactly the trap Jang warns about.

## Key files
- `src/simulator.py` — `SimState`/`fleet_events` now POSITIONAL (Phase 2: `…, launch_step, sx, sy, tx, ty`), `fleet_position_at`, `add_fleet_event(…, src_xy, dst_xy)`, `sim_step` (still NO opponent — Phase 4(b) gap), `evaluate_state`.
- `v2/search.py` — `search_improve_planet`, `_make_dists`, neural-value helpers, `_reconstruct_leaf_state` (now rebuilds in-distribution fleets). Phase 3 blend goes here.
- `v2/exit_train.py` — collect→search→distill loop, `_search_record`, `play_single_game` (P0-only vs apex — not side-alternated), parallel workers.
- `v2/config.py` — `V2ExItConfig` (search flags; add `value_leaf_blend` for Phase 3), `V2EvalConfig` (`eval_seed`, `eval_workers`).
- `v2/train.py` — `run_periodic_eval` (now side-alternated/paired/parallel, mirrors eval_fast), `_peval_init`/`_peval_game`.
- `scripts/eval_fast.py` — the reliable scorer; run on Colab at high n (local CPU ~20s/game). NOTE: win-rate is high-variance across map seeds — use n≥60 and ideally multiple seed batches before trusting a headline number.
