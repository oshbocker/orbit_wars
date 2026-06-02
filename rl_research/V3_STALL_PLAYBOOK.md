# V3 Stall Playbook — Highest-Leverage Moves If v3 Plateaus

*Synthesized 2026-06-01 from three streams: (1) a representation audit of our v3 code, (2) a
literature pass over `rl_research/` (PPG, MAPPO, DeepNash, DreamerV3, EfficientZero, Go-Explore,
Decision-Transformer, XLand, DouZero), and (3) a Kaggle Orbit Wars community/leaderboard survey.
Written as a contingency plan: the v3_a100 run (BC→PPO, PBRS reward, PFSP pool) is cooking;
this is what to reach for, in order, if it stalls.*

---

## TL;DR — the one thing that matters

**All three sources independently say the same thing: pure PPO-from-scratch stalls against strong
rule-based opponents in this game, and the ceiling path is a *trustworthy learned value function*
feeding *search / Expert Iteration*, wrapped in hybrid safety.**

- *Our logs*: knob sweeps R1–R5 all saturated at 0% vs apex; best run hit 60% @ u1750 then **collapsed to 25%**.
- *Literature*: the collapse is the textbook actor/critic shared-trunk fight (PPG) compounded by an
  imitation anchor that decays to zero (DeepNash R-NaD lesson).
- *Kaggle*: the canonical public RL author reports **5 from-scratch ML attempts all hit ~0% vs tier-3+
  bots** and pivoted to hybrid; meanwhile the **#1 leaderboard entry is a self-play RL specialist**, and
  the strongest *published* ML agent is **1-ply search + a learned value function** (GBC, val AUC 0.976) →
  LB 1000+. Self-play RL *can* win here, but only on top of a good critic + search, not raw PPO.

So if v3 stalls, **do not reach for more entropy/γ/vf knob sweeps** (proven dead). Reach for the
value head and the search loop. Everything below is ordered by that thesis.

---

## Diagnosis of the likely v3 failure modes (before changing anything)

Two concrete, high-confidence causes — both visible in the *running config* right now:

1. **The imitation anchor decays to zero.** `configs/v3_features.yaml` sets
   `imitation.coef_decay_updates: 1000` → the BC-to-apex pull is gone by u1000, right around where the
   prior run peaked and fell. DeepNash's central result: a *persistent* regularizer to a reference is
   what stops self-play from forgetting. Decaying to 0 removes the only thing holding the policy near
   "beats apex." (Blend point: `v2/ppo.py:123-128`.)
2. **Terminals rarely reach an update (`eps≈0`).** `ppo.rollout_steps: 64` with episodes of 150–498
   steps and `gamma: 0.997` means the ±1 win/loss signal is almost always bootstrapped through a value
   head that was never trained on terminals — a circular failure that starves the critic. (GAE:
   `v2/train.py:414-438`.)

These two explain both the plateau *and* the collapse. Fixing them is Tier 0.

> When you check the run tomorrow, the diagnostic to look at first is the **eval win-rate-vs-apex
> curve around the imitation-decay window (u750–u1100)** and the **value_loss / explained-variance
> trend**. A peak-then-fall there confirms cause #1; a value_loss that never drops confirms #2.

---

## Tier 0 — Anti-collapse (cheap, low-risk, do at first sign of stall)

| # | Change | Why (sources) | Where | Leverage / Cost |
|---|--------|---------------|-------|-----------------|
| 0.1 | **Floor the imitation/KL anchor — never decay to 0.** Replace linear-decay-to-zero with a persistent KL penalty `η·KL(π‖π_ref)` floored at ~0.01, where `π_ref` is periodically refreshed to our *best* checkpoint (not frozen-apex forever, so the floor rises). | Lit #4 (DeepNash R-NaD); our config is the smoking gun | `v2/ppo.py` loss, `v2/train.py` decay schedule, reuse `SelfPlayOpponent.sync_from()` for ref refresh | **High / S-M** |
| 0.2 | **Complete-episode rollouts.** Raise `rollout_steps` (64 → 256+) so a real fraction of episodes terminate per buffer; audit GAE truncation-vs-termination. | Lit #2 (`eps≈0` finding) | `configs/v3_features.yaml`; verify `v2/train.py:400-438` | **High / S** |
| 0.3 | **Keep value-target scale stable.** We already use symlog; for the *drifting* scale that PBRS + rising win-rate produce, prefer **PopArt** (running mean/var normalization of returns + adaptive output rescale). Pick one (symlog OR PopArt), don't stack. | Lit #3/#5 (MAPPO ranks this #1) | wrap `value_target` in `v2/ppo.py:108`, un-normalize in GAE | **Med-High / S** |
| 0.4 | **Rejection-only "shot validator" safety layer** (quick win, orthogonal to training). A tiny MLP predicting P(still own target 10 turns after arrival); *reject* low-confidence launches from whatever the policy proposes. Can't underperform the base policy. Community result: **+19pp win-rate (65%→84%)**, biggest gains vs hardest opponents. Label trick: exclude self-reinforcement shots (else 96% positive → no signal). | Kaggle §3d (konbu17 hybrid) | new small head/model over `decode_actions` output | **High / S-M, near-zero risk** |

---

## Tier 1 — Make the value head trustworthy (the linchpin; all 3 streams converge here)

| # | Change | Why (sources) | Where | Leverage / Cost |
|---|--------|---------------|-------|-----------------|
| 1.1 | **PPG auxiliary value phase + aux value head.** After the PPO policy phase, run extra value-only epochs on the same buffer with a `β·KL(π_old‖π)` clone term freezing the policy; add a second value head on the policy trunk. Decouples critic training from actor overfitting — the exact fix for the 60%→25% shared-trunk collapse. | Lit #1 (PPG) | `v2_ppo_update` add aux loop; `v2/model.py:87` second head; extend `action_log_prob_and_entropy` to return logits for KL | **High / M** |
| 1.2 | **Auxiliary per-shot-success head** (dense signal vs sparse reward). Predict, per launched fleet, "do we still own the target N turns later?" — a well-balanced auxiliary label that gives the trunk gradient every step. Doubles as the Tier-0.4 validator. | Kaggle §3d/§4; Lit (EfficientZero consistency spirit) | aux head in `v2/model.py`; label from rollout outcomes | **Med-High / M** |
| 1.3 | **Stronger global value features.** Add the leaderboard-proven (AUC 0.976) terms our 8-dim global vector lacks: **centrality** (Σ max(0,60−dist_to_center)/n), **best-single-enemy** framing (we mix some summed-enemy), **in-flight ship share per owner**, and explicit ship/planet/prod *lead* differences + 2P/4P one-hot. | Kaggle §4 (aidensong123) | `v2/features.py:275-284` global block | **Med-High / S** |

> **Why this tier is the priority:** every diagnostic — our collapse, the `eps≈0` finding, the
> "undertrained value head" suspicion, *and* the fact that the best public ML agent is literally a
> learned value function — points at the critic. A trustworthy value head is also the prerequisite for
> Tier 3 (search uses it to score leaves).

---

## Tier 2 — Representation upgrades (high-confidence; my audit ∩ Kaggle consensus)

| # | Change | Why (sources) | Where | Leverage / Cost |
|---|--------|---------------|-------|-----------------|
| 2.1 | **Requirement-relative ship sizing** — re-anchor fraction bins from "{0.25..1.0} of source ships" to "`required_ships` × {0.5,1,1.5,2}". We *already compute* `required_ships` (garrison + prod×eta + 1) but the action can't use it, so the net literally can't express "send exactly enough to capture." Directly fixes the diagnosed *ineffective-attacks* passivity. Community consensus: "once committed, send the solved amount / full pool." | Audit #2 + Kaggle §2 fleet-sizing | `v2/actions.py:268` + `env.ship_fractions`; pass `required_ships` from `features.py:226` | **High / M** |
| 2.2 | **Spatial canonicalization** — rotate/reflect the board so the player's home sits at a fixed angle/chirality. Starts are point-symmetric, so absolute (x,y) forces the net to learn 2–4 rotated copies of one strategy. Biggest pure sample-efficiency lever. | Audit #1 | new step before `encode_features`; invert on action output | **High / M** |
| 2.3 | **Exact-reserve + timeline features.** Replace the single blended incoming-ETA (`state.py:160-163`) with (a) a per-planet **defense margin** = garrison + own_incoming − enemy_incoming, (b) **time-to-flip**, (c) a small **multi-wave arrival histogram** (ships bucketed by ETA per team), and (d) **reaction-time** `(my_t, enemy_t)` per target. This is what apex's `_simulate_timeline` and every strong bot's binary-searched "keep-needed" encode. | Audit #3/#4 + Kaggle §4 | `v2/state.py`, `v2/features.py` planet block | **High / M-L** |
| 2.4 | **`from_planet_id` depletion feature.** A fleet's `from_planet_id` reveals a planet that just emptied itself → temporarily weak. Encode "enemy planet recently launched / depleted" as attack priority. | Kaggle §2/§4 (robust-agent) | `v2/features.py` planet block | **Med / S** |
| 2.5 | **Stable enemy ordering.** Enemies are indexed by *first-encounter order* (`features.py:106-112`, `state.py:150-152`) — permutes between steps, a non-stationary encoding in 4P. Sort by a stable key (total ships / proximity) so "enemy1" = biggest threat consistently. | Audit #5 | `v2/features.py`, `v2/state.py` | **Med / S** |
| 2.6 | **Graded sun-clearance + orbit phase/velocity** (lower priority). Replace the binary sun bool in the pair tensor with a clearance margin; add orbital radius + tangential velocity per orbiting planet so policy/value can anticipate sweeps and sun-crossings (aiming already handles this; reasoning doesn't). | Audit #6/#7 | `v2/features.py` pair + planet blocks | **Low-Med / S-M** |

---

## Tier 3 — Ceiling path (bigger bets, but this is what actually tops the board)

| # | Change | Why (sources) | Where | Leverage / Cost |
|---|--------|---------------|-------|-----------------|
| 3.1 | **Batched `fast_env` self-play** — vectorize the fidelity-verified sim to step N games at once (`[N,P]` masked ops), removing the harness/obs-dict/opponent-inference overhead that makes the scalar speedup look like only 1.5×. Unlocks Tier-0.2 (long rollouts) and 3.2 (search) in wall-clock. The throughput root-cause. | Lit #5; Kaggle (aidensong123's byte-equiv sim runs ~7000 turns/s) | new `v2/batched_env.py` from `v2/fast_env.py`; keep fidelity test as regression gate | **High / L** |
| 3.2 | **ExIt / search with a *neural* value at the leaves** — rebuild `v2/search.py` on **2-player** `fast_env` rollouts (in-sim opponent = apex/frozen-self), score leaves with OrbitNet's value head (not the handcrafted eval), sample candidate (target,fraction) from the heads, distill the improved distribution. This is the single most leaderboard-validated path (aidensong123's search+learned-value = LB 1000+; the #1 player is self-play RL). | Lit #7 + Kaggle §3e (strongest external validation) | `v2/search.py`, `v2/exit_train.py` on `fast_env`; needs Tier 1 (good value) first | **High / L** |
| 3.3 | **Train value/BC data from top-LB replays, not just apex.** Mine `penguin069/orbit-wars-local-arena` `replay_data.js` (~3MB) and top-agent episodes (reverse-engineering toolkit) to build a value-function/BC dataset from ~1500-LB bots — a far stronger teacher than our apex. | Kaggle §3e/§6 | new data-prep; feed `imitation.py` / value pretrain | **Med-High / M** |

---

## Engine-fidelity checklist (correctness, gates Tier 3)

Our simulator/features must match these exactly or search/ExIt and reward will be subtly wrong:
- **Same-turn production** — order is launch → production → move → rotate/sweep → combat, so a planet
  is never empty next turn (regains +production). Max sendable = *current* ships.
- **Two-phase combat** — arriving fleets fight *each other first* (top − second survives), then only the
  survivor fights the garrison. A 40-garrison planet survives a 60+55 split attack (60−55=5 vs 40).
  Coordinated attacks must beat this two-phase math, not just sum ships.
- **Tie → all tied players get +1.** A losing player should *race to match the leader's total* at step
  498 for a shared win — a distinct endgame target worth a reward/eval consideration.
- **Rotating-planet fleet sweep** (segment distance, captures in-transit/parked fleets) and **sun
  segment-collision** (overshoot still dies; chord must clear radius 10).
- **Last ~15 turns**: stop launching attacks that can't arrive; don't have big fleets mid-transit at 498.

---

## Recommended decision flow if v3 stalls

1. **Confirm the cause** from the eval curve (peak-then-fall at imitation-decay window → 0.1) and
   value_loss/explained-variance (flat → 0.2).
2. **Apply Tier 0 (0.1 + 0.2 first)** — cheapest, addresses the two diagnosed causes. Re-run.
   Add **0.4 (shot validator)** in parallel — it's a safe, standalone +19pp lever regardless.
3. **If still capped, add Tier 1 (1.1 PPG + 1.3 global features).** This is the value-head investment
   that everything else compounds on.
4. **Fold in Tier 2.1 + 2.3** (requirement-relative sizing + timeline/defense-margin) — the two
   representation fixes with the clearest tie to the "ineffective attacks" symptom.
5. **Commit to the ceiling path (Tier 3.1 → 3.2)** once the critic is trustworthy. This is the
   leaderboard-proven route and where the remaining headroom to ~1500+ lives.

**Do not** spend more cycles on entropy/γ/vf_coef sweeps — three independent lines of evidence say
that knob space is exhausted.

---

## Source map

- **Representation audit**: `v2/features.py`, `state.py`, `model.py`, `actions.py`, `reward.py`, `config.py`, `configs/v3_features.yaml`.
- **Literature**: `rl_research/REPORT.md`, `IMPROVEMENT_RESEARCH.md`, `SIMULATOR_AUDIT.md`, papers 01–10.
- **Kaggle** (key): leaderboard #1 Isaiah @ Tufa Labs (self-play RL); dylanxue04 mechanics deep-dive;
  thisisn0mad RL-pipeline ("5 ML attempts → hybrid"); konbu17 shot-validator hybrid (+19pp);
  **aidensong123 search + learned value (LB 1000+, AUC 0.976)**; suntzuisafteru LB-1039 launch-safety;
  penguin069 local-arena replays. Full URLs in the research-agent transcript.
</content>
</invoke>
