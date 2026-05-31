# Overcoming the Training Bottlenecks — Research Memo

**Date:** 2026-05-30
**Context:** BC clone tops out ~5% vs apex; PPO is throughput-starved on the Kaggle env
(~5 env-steps/sec aggregate); the opponent-free ExIt search can't beat apex. This memo
researches how to break each bottleneck and proposes a prioritized plan.

---

## TL;DR — the master bottleneck is **sample throughput**, and we already own the cure

Every credible path to beating apex (PPO self-play, league training, AlphaZero-style
search) needs **far more environment steps than the Kaggle env can deliver**. The closest
precedent — the **Lux AI** Kaggle RTS competitions — is explicit: winners used *BC warm
start → PPO self-play* and needed **"massive sample throughput, often billions of steps."*
We're at ~5 steps/sec. That's the gap, and it caps everything.

The cure is sitting in the repo: **`src/simulator.py` is a fast, pure-Python game model.**
The single highest-leverage investment is turning it into a **fast, faithful, vectorized
self-play training environment**. Modern GPU-vectorized envs (JAX: Gymnax/Brax/PureJaxRL/
JaxMARL) report **1000–10000× speedups** by co-locating env + policy on device. We don't
need full JAX to win big — even a batched NumPy/Torch version is 10–100× the Kaggle env.

Once throughput is solved, the rest of the research (below) becomes *applicable* rather
than aspirational.

---

## Bottleneck 1 — Throughput (the root cause)

**Evidence (ours):** Kaggle `env.step` + apex opponent + feature encode ≈ 0.5–2.3 s/step
per worker; 12 workers gave only ~5 steps/sec aggregate. PPO needs millions–billions of
steps → infeasible (5000 updates ≈ weeks).

**What the field does:**
- **GPU-vectorized envs** (Gymnax, Brax, **PureJaxRL** — "4000× speedups", JaxMARL for
  multi-agent self-play): keep environment *and* policy on the GPU, run 1k–100k envs via
  `vmap`, train on billions of steps on one device.
- Recipe to port an existing Python sim to JAX: pure functions, immutable state,
  **fixed-size padded arrays** for variable entities (planets/fleets → pad to max + mask),
  replace control flow with `jnp.where`/`lax.cond`/`lax.scan`, `jit` + `vmap`.

**Our mapping / plan (graduated by effort):**
1. **Quick (days):** make `src/simulator.py` the *training env* with an **opponent acting
   inside it** (apex or a frozen self). Even single-threaded it's ~10–100× faster per step
   than Kaggle (plain dict ops vs full sim). **Validate fidelity**: train in sim, eval in
   the real Kaggle env; check combat/movement/orbit/comet rules match.
2. **Bigger (1–2 wks):** **batch it** — fixed-size `[N, P]` planet / `[N, F]` fleet arrays,
   masked combat & movement, step N games at once in NumPy/Torch. 20–100× more.
3. **Biggest (2–4 wks):** **JAX port** for 1000×+ and on-GPU self-play.

**Risk:** simulator fidelity. Today it omits opponent moves and simplifies fleet
scheduling, so a policy trained in it can exploit the sim's gaps. Fidelity validation is
the gating task. This *also* fixes ExIt (Bottleneck 4).

---

## Bottleneck 2 — BC plateaus (~5% vs apex)

**Evidence (ours):** 60 epochs / 28k samples / loss 0.44, yet 5% vs apex (a faithful clone
ties ~50%). Dominated by the 90% "hold" majority; compounding drift over 500 steps.

**What the field says:**
- **"Pure imitation plateaus below top performance"** (Lux AI). BC is a *bootstrap*, not
  the goal — **self-play RL is what exceeds the script.** So 5% is acceptable *if* RL can
  build on it; don't over-invest in BC.
- **BC's covariate shift** → error grows ~O(εT²); **DAgger** (expert relabels the clone's
  own states) cuts it to O(εT). We already approximate this with the imitation-blend during
  PPO; a truer DAgger loop would help.
- **Filtered / weighted BC**: clone only apex's **winning** games, or weight by
  advantage/return (Decision-Transformer-style return conditioning). Removes low-value /
  losing behavior from the labels.
- **Class imbalance:** downweight the dominant "hold" class (or oversample launches) so the
  rare, game-deciding launch decisions get real gradient.
- **IQ-Learn (inverse soft-Q, NeurIPS'21)** and **offline RL (IQL/CQL)**: *dynamics-aware*
  imitation that learns a Q/value from demos, beating BC by 3–7× in low-data regimes and
  resisting compounding error. Applicable to our cached demo buffer.
- **Action chunking (ACT)** / autoregressive decoding: predict a short sequence of launches
  per turn → fewer independent decisions → less compounding error **and** cross-planet
  coordination (fixes our per-planet-independent sampling).

**Our mapping / plan:**
- Cheap now: **class-balanced + filtered BC** (downweight hold; clone apex's wins) and a
  **real DAgger** loop. Likely lifts the 5% floor.
- Medium: **autoregressive action head** (coordinate launches; reduce compounding error).
- Later: IQ-Learn/IQL on the demo buffer as a stronger-than-BC warm start.

---

## Bottleneck 3 — Sparse win signal & possibly-biased dense reward

**Evidence (ours):** outcomes are ~all losses early → value head degenerates to ≈−1; our
`dense_relative` (Δ ship/prod gap) is **not** potential-based, so it can reward
ship-hoarding over winning (consistent with the passive clone).

**What the field says (PBRS, Ng et al. 1999):** shaping `F(s,s') = γΦ(s') − Φ(s)` is the
*only* form that **provably preserves the optimal policy**. Pick a potential
`Φ(s)` = ship advantage + production advantage + planet/territory control; the shaping is
its discounted difference. Dense signal **without** biasing the true (win) objective.

**Our mapping / plan:** reformulate `dense_relative` as proper **PBRS** (`γΦ(s')−Φ(s)`),
keep the sparse ±1 terminal as the real objective. Low effort, removes a likely source of
the wrong-objective passivity. (We already have the value-symlog fix for the scale issue.)

---

## Bottleneck 4 — Self-play that actually beats the script

**Evidence (ours):** single apex opponent + linear rule-based decay; R4 (fast self-play)
was the *worst* variant — naive self-play forgets how to beat apex.

**What the field says (AlphaStar PFSP / league):**
- **Keep apex in the opponent pool** as a fixed reference (don't decay it away).
- **Prioritized Fictitious Self-Play**: sample opponents **weighted by win-rate** — train
  more against those you currently *lose* to (including apex). Prevents cycling/forgetting.
- **League**: main agent + **exploiters** (find weaknesses) + past-self checkpoints →
  robustness (important for the leaderboard vs unknown agents).
- Direct quote of the failure mode we hit: *"pure self-play can forget how to beat the
  script; keep the script in the pool (PFSP-weighted)."*

**Our mapping / plan:** replace the `MixedScheduler` decay with a **PFSP pool** = {apex,
past-self checkpoints}, sampled by `f(win_rate)`. We already have `ChampionPoolOpponent`
in the v1 code to crib from. Cheap once throughput exists.

---

## Bottleneck 5 — Search done right (the proper ExIt)

**Evidence (ours):** the current search is single-player (no opponent in the sim), uses a
myopic hand-crafted leaf eval, and washes the fraction head to uniform.

**What the field says:** the proven recipe is **policy proposes → MCTS refines → learned
value at leaves**. For our factored `(source × target × fraction)` action space,
**Sampled MuZero / Sampled AlphaZero** is the fit: *sample* candidate actions from the
policy instead of enumerating, then search. Leaf evaluation uses the **network's value
head**, not a handcrafted score.

**Our mapping / plan (gated on Bottleneck 1):** rebuild ExIt as AlphaZero-lite on the
**fast 2-player simulator** — opponent (apex or frozen self) acts in the tree; leaves
scored by OrbitNet's value; sampled actions for the big action space; freeze fraction
distillation until the search can actually rank fractions. This is the strongest long-term
plan but **only feasible after the simulator is fast and faithful.**

---

## Prioritized roadmap

| Pri | Action | Unlocks | Effort | Risk |
|---|---|---|---|---|
| **0** | **Fast, faithful self-play env from `src/simulator.py`** (opponent-in-sim → batched → JAX) | everything below | M→L | sim fidelity |
| 1 | **PBRS reward** (potential-based dense + sparse terminal) | unbiased dense signal | S | low |
| 1 | **PFSP league** (keep apex in pool, weight by win-rate) | beat-and-retain vs apex | S | low |
| 2 | **Filtered + class-balanced BC** (clone wins, downweight hold) + true DAgger | higher warm-start floor | S–M | low |
| 2 | **Autoregressive action head** | coordination, less compounding error | M | med |
| 3 | **AlphaZero-lite ExIt** (2-player sim, learned-value leaves, sampled actions) | superhuman ceiling | L | med (needs Pri-0) |
| 3 | **IQ-Learn / IQL** warm start from demos | stronger-than-BC bootstrap | M | med |

**The single most important move is Priority 0.** Our entire difficulty is that we're doing
model-free RL on a slow simulator while owning a fast one. Fix throughput and the proven
Lux-AI recipe (BC → PFSP self-play, PBRS reward) becomes directly executable; without it,
every method here is starved.

---

## Sources
- PureJaxRL / Gymnax / Brax / JaxMARL — GPU-vectorized RL envs, 1000×+ speedups.
- Lux AI (Kaggle RTS) competition writeups — BC→PPO self-play, billions of steps.
- Ng, Harada & Russell (1999) — potential-based reward shaping (policy invariance).
- Vinyals et al. (2019), AlphaStar — PFSP & league training.
- Garg et al. (2021), IQ-Learn — inverse soft-Q imitation > BC/GAIL.
- Ross et al. (2011), DAgger — covariate shift; O(εT) vs BC's O(εT²).
- Hubert et al. (2021), Sampled MuZero — search over sampled actions for large action spaces.
- Chen et al. (2021), Decision Transformer; ACT — return-conditioned / chunked imitation.
</content>
