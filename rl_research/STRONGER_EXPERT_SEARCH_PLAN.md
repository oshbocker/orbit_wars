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

### Phase 1 — Every-step in-sim rollout opponent (highest value ÷ risk) — IN PROGRESS
The correct version of the two-player experiment. Make BOTH sides act at every lookahead step via a cheap, **geometry-free** rollout policy that launches from `SimState` (positions supplied as a precomputed distance matrix shared from the search root). Symmetric: after the candidate's first move, our continuation *and* opponents follow the rollout policy to depth — so "hold" means *playing on*, not freezing, and aggressive candidates are evaluated against real opposition. Default OFF (new flag, keep the 77% agent byte-identical until validated).
- **Gate:** must hold ≥77% @ n=60 before keeping on.

### Phase 2 — Positional simulator (unblocks grounded neural value)
`SimState` drops fleet x/y. Track fleet positions (or reconstruct leaf `GameState`s *with* fleets) so leaf features match the training distribution — prerequisite that kills the OOD failure mode behind the neural-value collapse.

### Phase 3 — Grounded learned value (the real win)
Collect full self-play games with search on both sides, play **~10%** fully to terminal, label every visited state with the game outcome (±1 and/or final score margin), train the value head on those. Then use it at search leaves — now in-distribution (Phase 2) and grounded (this phase). Disciplined retry of the idea that previously collapsed.

### Phase 4 — Genuine policy-improvement search
- (a) **Sequential multi-planet search** — update `SimState` between owned planets so coordinated multi-planet plays become visible (currently per-source greedy). Low risk.
- (b) **Shallow fixed-width tree / beam** with the opponent acting at each node and the grounded value at leaves — closest thing to real MCTS the simultaneous-move, huge-branching structure allows. Only after (a) + Phase 3.

### Separate track — embed-128 vs 256 capacity A/B in the ExIt regime
Configs `configs/v2_exit_embed128.yaml` / `v2_exit_embed256.yaml`, run via `scripts/run_embed_ab.py` on Colab once Phase 3 yields high-quality targets. Lesson #5 predicts capacity finally pays off here.

**Sequencing rationale:** 1→2→3 is a dependency chain that fixes both documented failures and installs the one non-negotiable (grounded value + in-distribution leaves). Phase 4 compounds only once the value is trustworthy — deeper search on a passive opponent / ungrounded value is exactly the trap Jang warns about.

## Key files
- `src/simulator.py` — `SimState` (positionless), `sim_step` (NO opponent — the Phase-1/2 gap), `evaluate_state`.
- `v2/search.py` — `search_improve_planet`, `_make_dists`, neural-value helpers, `_reconstruct_leaf_state`.
- `v2/exit_train.py` — collect→search→distill loop, `_search_record`, `play_single_game`, parallel workers.
- `v2/config.py` — `V2ExItConfig` (search flags).
- `scripts/eval_fast.py` — the reliable scorer (n=60); per-iter n=20 evals too noisy.
