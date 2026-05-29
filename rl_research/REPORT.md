# Reinforcement Learning for Orbit Wars — A Research Survey & Development Roadmap

**Author:** prepared for the Orbit Wars RL effort
**Date:** 2026-05-28
**Companion experiments:** `experiments/RECOMMENDATIONS_REPORT.md`

---

## 0. How to read this report

This document has three jobs:

1. **Teach** the core ideas of modern (2020–2025) reinforcement learning through ten landmark papers, chosen because each one speaks to a specific difficulty in Orbit Wars. For each paper I give the *problem it attacks*, the *key idea*, an *intuition* you can hold in your head, the *one or two equations that actually matter*, and *what we should steal*.
2. **Synthesize** — show where the papers rhyme. Ten papers look like ten ideas; they are really about five recurring problems, and the same handful of tricks keeps reappearing.
3. **Act** — turn the reading into five concrete, falsifiable changes to our `v2_bc` agent, which are then tested in the companion experiments report.

I have tried to keep the math minimal but honest: every equation here is one you should be able to *re-derive the intuition for*, not just pattern-match.

---

## 1. What makes Orbit Wars hard, in RL terms

Before the papers, let's name the enemy. Orbit Wars has five properties that each map onto a distinct, well-studied RL difficulty. Keep this table in mind — the whole report hangs off it.

| # | Property of Orbit Wars | The RL difficulty it creates | Papers that attack it |
|---|---|---|---|
| C1 | You issue **many launches per turn**, each a `(source, target, fraction)` tuple, from a variable set of owned planets | **Combinatorial / structured action space**; a naïve flat action head is hopeless | Invalid Action Masking, DouZero, (DT) |
| C2 | The real reward is **win/lose at step ~498**; ship counts in between are a proxy | **Sparse, long-horizon credit assignment** | Go-Explore, DreamerV3, EfficientZero |
| C3 | You play **against other learning/adapting agents** (apex, self-play, 4-player) | **Non-stationarity & game theory**; the "best" policy depends on the opponent; naïve self-play cycles | DeepNash, XLand, MAPPO |
| C4 | The episode is **500 steps**, fleets take many turns to arrive, early captures compound | **Delayed consequences / temporal credit** | DreamerV3, EfficientZero, (high-γ) |
| C5 | We train **on a laptop CPU**, then maybe a single GPU | **Sample- and compute-efficiency** | EfficientZero, PPG, MAPPO, BC/DT |

Our current agent (`v2_bc` = OrbitNet) is an **on-policy PPO** actor-critic with a transformer over planets, a pairwise output head for `(source → target)`, behavioral-cloning (BC) warm-start from apex, and mixed rule-based/self-play opponents. Everything below is read through that lens: *which of these papers improves the thing we already have, and which suggests a different thing entirely?*

---

## 2. The ten papers

I've grouped them into six families. Read the family intros — they carry as much of the lesson as the papers themselves.

### Family A — The on-policy core (the algorithm we actually run)

Our agent is PPO. These two papers are about making PPO work, and they are the most *immediately* actionable.

#### Paper 1 — MAPPO: *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (Yu et al., 2022) — `arXiv:2103.01955`

**Problem.** The multi-agent RL (MARL) field had largely concluded that on-policy methods like PPO were too sample-inefficient for multi-agent settings, and that you needed off-policy, value-decomposition, or specialized MARL machinery. This paper asks: is that actually true, or did we just tune PPO badly?

**Key idea.** With the right implementation details, plain PPO — under **CTDE** (Centralized Training, Decentralized Execution) — matches or beats specialized MARL algorithms on hard cooperative benchmarks (SMAC, Hanabi, MPE), while being far simpler. The "centralized" part: a single **value function that sees the global state** `V(s)`, used to compute advantages, while each agent's **policy sees only its local observation** `π(a_i | o_i)`.

**Intuition.** A decentralized critic has to guess what everyone else is doing; a centralized critic *knows*, so its value estimates have far lower variance, which makes the policy-gradient signal far cleaner. You only pay the centralization cost at training time — at deployment each agent acts on its own observation.

**The math that matters.** Same clipped objective as single-agent PPO, but the advantage uses a global-state critic:

```
L_CLIP(θ) = E_t [ min( r_t(θ) Â_t , clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]
r_t(θ) = π_θ(a_t|o_t) / π_θ_old(a_t|o_t)
Â_t  via GAE using a centralized V(s_t)  (not V(o_t))
```

The paper's lasting contribution is its **five implementation tricks**, ranked by importance:
1. **Value normalization** (normalize value targets by a running mean/std — PopArt-style). *They find this is one of the single most important factors.*
2. **Value input representation** — feed the critic a well-chosen global state (agent-specific global state beats a naïve concatenation).
3. **Training data usage** — use *fewer* epochs / smaller minibatches than you'd expect; 15 epochs is often too many, 5–10 is better; don't over-reuse data.
4. **Action masking** — mask invalid actions (see Paper 2).
5. **Death masking** — handle agents that die mid-episode correctly.

**What we steal.** (a) **Value normalization** — our returns mix tiny dense-shaping rewards with a terminal ±1; the value target's scale is a mess. This is a top-priority fix. (b) The CTDE idea legitimizes giving our critic *global* features even in self-play and 4-player. (c) "Don't over-train on each rollout" — a reason to keep PPO epochs modest.

---

#### Paper 2 — PPG: *Phasic Policy Gradient* (Cobbe et al., 2020) — `arXiv:2009.04416`

**Problem.** In actor-critic methods that **share a network trunk** between the policy and value heads (exactly OrbitNet's design), the two objectives *fight*. The value function wants to be trained hard, on lots of data, many epochs — it's a regression problem and benefits from it. The policy must *not* be trained hard on the same data or it overfits and collapses. Sharing features forces a bad compromise.

**Key idea.** **Separate training into phases.** A *policy phase* runs normal PPO for a few iterations (lightly training the value head too). Then an *auxiliary phase* trains the value function hard for many epochs — but adds a **behavioral-cloning term that anchors the policy to itself** so distilling value features doesn't wreck the policy.

**Intuition.** "Train the critic as much as it wants; protect the actor from the side effects." The clone term is the protection: it says *change your features to predict value better, but keep outputting the same action distribution you had.*

**The math that matters.** The auxiliary phase optimizes a joint objective on a value head *and* an auxiliary value head attached to the policy trunk, plus a clone penalty:

```
L_joint = L_aux_value  +  β_clone · KL( π_old ‖ π_θ )
```

where `π_old` is the policy *before* the auxiliary phase. PPG gets better sample efficiency than PPO precisely because it can reuse data aggressively for value learning without that data corrupting the policy.

**What we steal.** Two things. (a) The diagnosis: *our value head is probably undertrained because we dare not crank up epochs.* (b) A cheap proxy we can test today — **raise the value-loss weight** (`vf_coef`) so the critic gets more signal per update — and a deeper future change: a true PPG auxiliary phase. A well-fit critic is the lever that makes everything else (GAE, advantages, sparse credit) work.

> **Worth knowing (not in the ten):** Andrychowicz et al., *"What Matters for On-Policy Deep Actor-Critic Methods? A Large-Scale Study"* (2021) ran 250k+ configs and confirms much of the above: advantage normalization, careful learning-rate, modest epochs, and **a well-initialized value function** dominate exotic algorithmic changes. The lesson of Family A: *the boring details are the algorithm.*

---

### Family B — Structured action spaces (the `(source, target, fraction)` problem)

This family is the most directly relevant to C1 — our action space is the thing that makes Orbit Wars Orbit Wars.

#### Paper 3 — *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms* (Huang & Ontañón, 2022) — `arXiv:2006.14171`

**Problem.** Real-time strategy games (the paper uses **µRTS**, a close cousin of Orbit Wars) have action spaces with up to ~10⁴–10⁸ actions per step, the *vast majority of which are invalid* in any given state (you can't move a unit that doesn't exist, attack out of range, etc.). How do you do policy gradient over that?

**Key idea & the crucial theoretical point.** **Invalid action masking** — set the logits of invalid actions to −∞ before the softmax — is not just an engineering hack. The paper proves that masking corresponds to a **valid policy gradient**: the masked-out actions receive *zero gradient*, and the gradient with respect to the remaining (renormalized) distribution is exactly correct. The alternative ("invalid action penalty": let the agent pick invalid actions but punish it) is shown to be dramatically worse and to scale terribly as the fraction of invalid actions grows.

**Intuition.** Masking shrinks the decision to only the legal moves *at every step*, so the policy never wastes probability mass or gradient on the impossible. Crucially the agent never has to *learn* what's illegal — it's told. In a space that is 99.99% illegal, that is the difference between learning and not.

**The math that matters.** For logits `l` and a boolean legality mask `m`:

```
l'_i = l_i           if m_i (legal)
l'_i = −∞ (≈ −1e9)   if ¬m_i (illegal)
π(a_i) = softmax(l')_i
```

The gradient ∂log π / ∂l_i is zero for masked actions; the renormalization handles the rest. Empirically, masking lets a single PPO agent learn full-game µRTS; without it, learning stalls.

**What we steal.** *We already do this* — `reachability_mask` plus the sun/own/self masks in `model.py`. This paper is the **theoretical license** for our most important design choice, and it tells us where to invest: **the quality of the mask is a first-class lever.** Our viability check (`src.ships ≥ 2·(garrison+1)`) is a heuristic mask; tightening or loosening it directly changes the learnable action set. It also warns us *not* to switch to a penalty scheme.

---

#### Paper 4 — DouZero: *Mastering DouDizhu with Self-Play Deep RL* (Zha et al., 2021) — `arXiv:2106.06135`

**Problem.** DouDizhu is a 3-player card game with an enormous, **state-dependent legal action set** (up to ~2·10⁴ legal card combinations), imperfect information, and both competition and cooperation. Standard DQN (a Q-value head with one output per action) is impossible — you can't have 10⁴ output neurons whose meaning changes every state.

**Key idea.** **Encode the action as an *input* to the network, not an *output*.** Instead of `Q(s) → ℝ^|A|`, learn `Q(s, a) → ℝ¹` where the candidate action `a` is fed in as a feature vector. At decision time you score each *legal* action and pick the best. They train this with **Deep Monte-Carlo (DMC)**: plain Monte-Carlo returns (full-episode, no bootstrapping) regressed by a deep net, run massively in parallel via self-play.

**Intuition.** When actions are richly structured (a card combo, or "send 60% of ships from planet 7 to planet 12"), describing the action lets the network *generalize across actions* — it learns "big trumping combos are good in situations like this" rather than memorizing one output neuron per combo. Monte-Carlo, despite being "old-fashioned" and high-variance, sidesteps the deadly triad and the bootstrapping bias that plagues huge-action Q-learning.

**The math that matters.** DMC regresses the empirical return:

```
Q(s,a) ← Q(s,a) + α ( G_t − Q(s,a) ),   G_t = Σ_{k≥0} γ^k r_{t+k}   (full-episode MC)
```

with `(s,a)` both encoded as input features, and many parallel self-play actors generating `G_t`.

**What we steal.** This is the conceptual parent of **OrbitNet's pairwise head**: we already score `(source_i, target_j)` pairs by feeding both planets' embeddings into `pair_mlp` — that is "action as input." DouZero validates the architecture. It also offers a *radically simpler training signal* (full-episode Monte-Carlo) as an alternative to our GAE/PPO machinery — worth remembering if value learning stays brittle. And it underlines that **the ship-fraction should be part of the action representation**, not a softmax-probability afterthought (our current `frac = prob[target]` hack is the weak point this paper highlights by contrast).

---

### Family C — Multi-agent self-play & equilibria (the "who am I playing?" problem)

These attack C3: the opponent is not fixed.

#### Paper 5 — DeepNash: *Mastering the Game of Stratego with Model-Free Multiagent RL* (Perolat et al., 2022) — `arXiv:2206.15378`

**Problem.** Stratego is a two-player zero-sum game of *imperfect information* with an astronomically large game tree and long episodes, where bluffing matters. AlphaZero-style search fails (you can't search imperfect-information trees naïvely), and **naïve self-play cycles**: A beats B beats C beats A, and the policy chases its own tail forever (the non-transitivity problem).

**Key idea — R-NaD (Regularized Nash Dynamics).** Converge to an (approximate) **Nash equilibrium** by *regularizing the policy toward a slowly-moving reference policy*, then periodically updating the reference. This is done by **transforming the reward**: at each step, add a penalty proportional to the log-ratio between the current policy and the reference. This turns the unstable game dynamics into something that provably contracts toward a fixed point — the Nash equilibrium. **No search at test time**; it's pure model-free RL.

**Intuition.** Naïve self-play is like two people arguing in circles. The KL-to-reference penalty is a "stay close to who you recently were" rope; it damps the oscillation. Then you slowly move the anchor in the direction you've been improving. The sequence of anchors converges to the equilibrium policy that can't be exploited.

**The math that matters.** The transformed per-step reward for the learning player adds a regularization term that pulls the policy `π` toward reference `π_reg`:

```
r̃_t = r_t  −  η · log( π(a_t|s_t) / π_reg(a_t|s_t) )      (acting player)
```

(with the opposite sign for the opponent in the zero-sum setting). As `η`-regularized dynamics are iterated and `π_reg ← π` periodically, the process converges to an ε-Nash equilibrium.

**What we steal.** A principled cure for self-play instability: **regularize toward a reference policy** (e.g. our BC-from-apex policy, or a frozen recent self) via a KL penalty. This is the rigorous version of "keep the imitation/anchor term alive during PPO" — and it directly motivates *not* letting our BC coefficient decay all the way to zero too fast. It's also a different framing of PPG's clone term and TRPO/PPO's trust region: **most of modern game-playing RL is some flavor of "improve, but don't run away from your reference."**

---

#### Paper 6 — *Open-Ended Learning Leads to Generally Capable Agents* (XLand) (DeepMind, 2021) — `arXiv:2107.12808`

**Problem.** Agents trained on a fixed task (or fixed opponent) overfit to it and fail to generalize. How do you produce an agent that is *generally* capable across a vast, open-ended space of games it has never seen?

**Key idea.** **Make the curriculum itself the algorithm.** XLand procedurally generates an enormous space of multi-agent games and worlds, and uses **population-based training** with **dynamic task generation** (sampling tasks at the frontier of the agent's ability — not too easy, not impossible) and **generational distillation** (each new generation of agents starts by distilling the previous, then improves). The result is a single agent that zero-shot-handles held-out tasks (hide-and-seek, capture-the-flag variants, etc.).

**Intuition.** If you only ever face one opponent, you learn to beat *that* opponent. If you face an ever-shifting population that grows with you, you're forced to learn *robust* skills. The curriculum auto-targets your current weaknesses — like a sparring partner who always trains you at the edge of your ability.

**The math that matters.** Less a single equation, more a recipe: maintain a **population** `{π_1..π_N}`; for each agent, sample opponents/tasks weighted toward an *intermediate* win-rate band (the learnability frontier, a normalized-score/PLR idea); periodically copy-and-mutate the best, and distill across generations.

**What we steal.** Our `MixedScheduler` is an embryonic version of this (it blends rule-based and self-play with a fixed linear decay). XLand says: (a) **opponent diversity matters more than opponent strength**; (b) **prioritize opponents near 50% win-rate** rather than a fixed schedule; (c) keep a **population/league of frozen past selves**, not just the latest. This is the path from "beats apex" to "robust on the leaderboard against unknown agents."

---

### Family D — Exploration under sparse reward

#### Paper 7 — Go-Explore: *First Return, Then Explore* (Ecoffet et al., 2021, Nature) — `arXiv:2004.12919`

**Problem.** In hard-exploration, sparse-reward games (Montezuma's Revenge, Pitfall) standard ε-greedy / entropy-bonus exploration fails catastrophically. The paper names *why*: **detachment** (the agent forgets about promising frontiers it discovered earlier) and **derailment** (the exploratory noise that's supposed to get you *back* to a frontier knocks you off course before you arrive).

**Key idea.** Separate "getting back" from "exploring." Maintain an **archive of interesting states** (compressed into "cells"). Each iteration: (1) **return** to a promising archived state *reliably* (replay the trajectory, or use a goal-conditioned policy — don't rely on lucky noise to re-reach it), *then* (2) **explore** randomly from there, adding any newly-discovered cells to the archive. Finally, **robustify**: turn the discovered high-reward trajectories into a robust policy via imitation learning.

**Intuition.** Exploration fails not because the agent can't take a good random step, but because it can't *reliably get to the place where a good random step is available*. Remembering states and returning to them deterministically removes the dependence on luck. "First return, then explore."

**The math that matters.** Algorithmic, not analytic. The cell archive maps a state-descriptor `φ(s)` → best trajectory reaching that cell; selection probability favors rarely-visited or recently-promising cells. The final policy is trained by **imitation on the archived high-return trajectories** (backward algorithm / LfD).

**What we steal.** Two ideas. (a) Conceptual: our dense-reward shaping is a *band-aid* over a hard-exploration problem; Go-Explore is what you reach for when shaping isn't enough. (b) Practical and powerful: the **"robustify by imitation"** step is *exactly our BC pipeline*, and it suggests a virtuous loop — let the agent (or search) find good games, then imitate the best trajectories. This is the bridge to Family F and to Expert Iteration (which we already have in `src/`).

---

### Family E — Models, planning, and long horizons

These attack C2/C4/C5 together: learn a model, plan in it, and get long-horizon credit assignment for cheap.

#### Paper 8 — EfficientZero: *Mastering Atari Games with Limited Data* (Ye et al., 2021) — `arXiv:2111.00210`

**Problem.** MuZero (the parent method) reaches superhuman play but needs *enormous* data. Can we get the same model-based planning power with ~2 hours of data (the Atari 100k benchmark)? This matters intensely for us: **we train on a CPU.**

**MuZero in one paragraph (the parent you must understand).** Learn three networks: a **representation** `h: obs → latent s⁰`, a **dynamics** model `g: (sᵏ, aᵏ) → (sᵏ⁺¹, reward)`, and a **prediction** head `f: sᵏ → (policy, value)`. Crucially, the model is **never trained to reconstruct observations** — only to predict reward, value, and policy. Then run **MCTS in the learned latent space** to produce an improved policy target. You get planning without knowing the rules.

**EfficientZero's key idea — three additions for sample efficiency:**
1. **Self-supervised consistency loss.** Force the *predicted* next latent `g(sᵏ,aᵏ)` to match the *encoded* real next state `h(o_{t+1})` (a SimSiam-style loss). This gives the dynamics model a dense learning signal at every step instead of relying on the sparse reward/value gradient. **This is the big one.**
2. **Value prefix.** Predict the *sum* of upcoming rewards with an LSTM ("value prefix") rather than per-step rewards, which removes a brittle off-by-one state-aliasing problem in value targets.
3. **Off-policy correction.** Replay buffers contain stale targets; reweight/recompute them to account for policy drift.

**Intuition.** The headline insight is #1: *a model trained only on reward+value is starved for signal in sparse settings; make it also predict its own future latent, and it learns the dynamics densely and fast.* Self-supervision converts every transition into a free learning signal.

**The math that matters.**

```
MuZero loss:  L = Σ_k [ ℓ_reward(rᵏ, r̂ᵏ) + ℓ_value(zᵏ, v̂ᵏ) + ℓ_policy(πᵏ, p̂ᵏ) ]
EfficientZero adds:  L_consistency = − cos_sim( sg[h(o_{t+1})] , proj(g(sᵏ,aᵏ)) )
```

(`sg` = stop-gradient; the value target `zᵏ` uses the value-prefix LSTM.)

**What we steal.** (a) If we ever go model-based (we have `src/simulator.py`!), this is the recipe to make it sample-efficient on a CPU. (b) Even *without* MCTS, the **self-supervised consistency idea is an auxiliary loss we can bolt onto OrbitNet today**: predict next-step planet embeddings, get a denser gradient. (c) It validates planning-as-policy-improvement, the engine behind our Expert Iteration code.

---

#### Paper 9 — DreamerV3: *Mastering Diverse Domains through World Models* (Hafner et al., 2023) — `arXiv:2301.04104`

**Problem.** RL algorithms are notoriously finicky — you re-tune hyperparameters for every new domain. DreamerV3 aims for **one algorithm, one set of hyperparameters, 150+ tasks**, including the famously hard sparse-reward, long-horizon task of **collecting diamonds in Minecraft from scratch** (the first method to do so).

**Key idea.** Learn a **world model** (a Recurrent State-Space Model, RSSM) that compresses observations into latent states and predicts dynamics; then train an **actor and critic *entirely inside the model's imagination*** — the agent dreams thousands of rollouts and learns from them, almost never touching the real environment after each batch. The contribution is less the architecture and more a **bag of robustness tricks** that make it work everywhere without tuning.

**Intuition.** A world model is a learned simulator; once you have it, experience is cheap — you generate it. Long-horizon credit assignment becomes tractable because you can imagine far ahead in latent space. The robustness tricks all attack the same root cause: **rewards and returns vary wildly in scale across tasks and across training, and naïve losses blow up.**

**The math that matters — the robustness tricks worth memorizing:**
- **symlog squashing** of targets: `symlog(x) = sign(x)·ln(1+|x|)`, so the network predicts a compressed value and you `symexp` back. Tames huge/tiny magnitudes.
- **two-hot encoded returns + categorical regression** instead of MSE — turns value regression into a stable classification.
- **percentile return normalization** for the actor's advantage scaling — robust to outliers.
- **KL balancing / free bits** in the world-model loss so the representation neither collapses nor explodes.

**What we steal.** Even staying model-free, **symlog and value normalization are drop-in fixes** for our messy reward scale (tiny dense rewards + ±1 terminal) — the same disease MAPPO's value normalization treats. The deeper lesson: *if we want one agent robust across 2-player, 4-player, varied maps, the path is a world model + scale-robust losses.* And `src/simulator.py` means we could even use a **ground-truth** model (no need to learn dynamics), which is strictly easier than DreamerV3's setting.

---

### Family F — Sequence models & the imitation bridge

#### Paper 10 — Decision Transformer: *Reinforcement Learning via Sequence Modeling* (Chen et al., 2021) — `arXiv:2106.01345`

**Problem.** Can we drop the entire apparatus of RL — value functions, TD bootstrapping, policy-gradient — and just treat the problem as **supervised sequence prediction**, like language modeling?

**Key idea.** Feed a Transformer the trajectory as a sequence of `(return-to-go, state, action)` tokens and train it, with a plain supervised loss, to predict the next action. The trick is **return-conditioning**: each state is tagged with the *return-to-go* `R̂_t = Σ_{t'≥t} r_{t'}` you ended up achieving. At test time, you *prompt* the model with a high desired return and it produces actions consistent with achieving it.

**Intuition.** Instead of learning "what's the value of this state" and bootstrapping it forward (hard, unstable), you learn "given that I want total reward X from here, and I'm in state s, what action did high-reward trajectories take?" It's BC, but conditioned on outcome quality — so it can imitate *good* behavior selectively even from mixed data. No bootstrapping means no deadly triad.

**The math that matters.** Pure autoregressive cross-entropy over the action tokens:

```
trajectory τ = ( R̂_1, s_1, a_1, R̂_2, s_2, a_2, … )
loss = Σ_t  CE( a_t ,  Transformer(R̂_≤t, s_≤t, a_<t) )
```

**What we steal.** Three things. (a) **Our BC pretrain *is* a Decision Transformer without the return token** — DT tells us we could condition on outcome (e.g. only clone apex's *winning* games, or tag demos by margin of victory) for free quality-filtering. (b) The **autoregressive idea** is the principled fix for our independent-per-planet sampling: decode planet launches *as a sequence*, each conditioned on the previous, so the model can coordinate (not send two planets to the same target, set up pincers). (c) It legitimizes the transformer architecture we already chose and the BC→RL pipeline we already run.

---

## 3. Synthesis — where the papers rhyme

Ten papers, but only a handful of ideas keep returning. Here is the map, and then the big ideas.

### 3.1 Paper × challenge map

| Paper | C1 action space | C2 sparse reward | C3 multi-agent | C4 long horizon | C5 efficiency |
|---|:--:|:--:|:--:|:--:|:--:|
| MAPPO | ○ | | ●● | | ● |
| PPG | | | | ○ | ●● |
| Invalid Action Masking | ●● | | | | ● |
| DouZero | ●● | | ● | ● | |
| DeepNash | | | ●● | ● | |
| XLand | | | ●● | | ○ |
| Go-Explore | | ●● | | ● | |
| EfficientZero | | ● | | ●● | ●● |
| DreamerV3 | | ●● | | ●● | ● |
| Decision Transformer | ● | ○ | | ● | ● |

(●● = central contribution, ● = strong, ○ = relevant)

### 3.2 The five big ideas (the actual lessons)

**Big idea 1 — "Improve, but don't run away from a reference."**
This is the single most repeated theme. PPO's clipping, TRPO's trust region, **PPG's clone term**, **DeepNash's KL-to-reference regularization**, and **our own BC-imitation anchor** are *the same idea*: take an improvement step, but stay close to a trusted policy (your old self, your reference, the expert). It's how you get stability in a non-stationary, high-variance world. → *Don't let our BC anchor decay to zero too eagerly; consider an explicit KL-to-reference term (R-NaD-lite).*

**Big idea 2 — The value function is the bottleneck, and its **scale** is the silent killer.**
MAPPO ranks **value normalization** as a top factor; PPG exists *entirely* to train the value function harder; DreamerV3's robustness tricks (symlog, two-hot, percentile norm) are overwhelmingly about **value/return scale**; "What Matters" agrees. Our reward is a pathological mix of ~0.002-scale dense terms and ±1 terminal — a textbook case for value/return normalization. → *Normalize returns / weight value learning more; this is probably our highest-leverage, lowest-glamour fix.*

**Big idea 3 — In huge structured action spaces, mask the illegal and *describe* the action.**
Invalid Action Masking (mask → valid gradient) and DouZero (action-as-input → generalization across actions) are the two halves of how you make a `(source, target, fraction)` action space learnable. OrbitNet already does both (reachability mask + pairwise head). The unfinished half is the **fraction**: it should be a *described, masked, sampled* sub-action, not a reused softmax probability.

**Big idea 4 — Self-play is the curriculum, but it needs a stabilizer.**
XLand (population + frontier sampling), DeepNash (regularized dynamics → Nash), MAPPO (CTDE self-play): left to itself, self-play cycles and overfits. The fixes are (a) **opponent diversity / a league of past selves**, (b) **prioritizing ~50%-win-rate opponents**, and (c) **a regularizer (Big idea 1)**. Our single-opponent + linear-decay scheduler is the naïve version.

**Big idea 5 — Imitation and search/planning are the two ways to get a good policy without grinding pure RL; use both.**
Go-Explore ("robustify by imitation"), Decision Transformer (RL *as* imitation), EfficientZero/MuZero (planning *as* policy improvement), and DreamerV3 (imagination *as* cheap experience) all sidestep the slow, high-variance grind of model-free PG. We already have the two endpoints — **BC** (imitation) and **Expert Iteration + simulator** (planning). The frontier is to close the loop: imitate the expert → improve by search/self-play → imitate the improved trajectories → repeat.

### 3.3 The one-sentence synthesis

> *Modern game-playing RL is: **clip/regularize toward a reference** (1) so a **well-normalized value function** (2) can give clean credit over a **masked, described action space** (3), trained against a **diverse, stabilized self-play curriculum** (4), and **bootstrapped by imitation and sharpened by planning** (5).*

Our `v2_bc` agent already embodies (1) partially (BC anchor, PPO clip), (3) well (mask + pairwise head), (4) crudely (mixed scheduler), and (5) at the endpoints (BC + ExIt). The most neglected is **(2): value/return scale.** That observation drives the recommendations.

---

## 4. Five recommendations for `v2_bc`

Each recommendation states the **hypothesis**, the **paper lineage**, the **exact change**, and the **prediction**. They are tested one-variable-at-a-time against a BC-warm-started baseline in the companion experiments report, in the established 200-update / seed-42 / eval-vs-apex-and-random style.

> **Note on the headline structural fix.** The deepest weakness exposed by the reading (Big idea 3 + DouZero + DT) is that **ship-fraction is entangled with target-selection** (`frac = softmax_prob[target]`). Spreading one softmax over up to 41 options starves the fractions, so fleets are systematically too small. The right fix is a **dedicated, masked fraction head** (a factored action, as in our own `src/` v1 policy). This is an architecture change rather than a one-knob experiment, so it is the *primary* item in the final recommendation (§6) rather than one of the five A/B tests — but it is the most important single change we can make.

**R1 — Entropy annealing (exploration→exploitation schedule).**
- *Hypothesis:* BC gives a structured starting policy; a fixed entropy bonus of 0.01 keeps fighting it. A small early bonus that anneals to ~0 lets the policy explore around BC's blind spots, then commit.
- *Lineage:* MAPPO/PPG/"What Matters" (entropy & exploration tuning); prior in-house result that `ent=0.005` beat `0.03`.
- *Change:* anneal `ent_coef` linearly from 0.02 → 0.0 over training (new `ent_coef_end` knob).
- *Prediction:* higher win-rate and lower final entropy than fixed-0.01 baseline.

**R2 — Higher discount γ = 0.997 (long-horizon credit).**
- *Hypothesis:* with γ=0.99 the effective horizon is ~100 steps, but games run ~498 steps and the terminal win/loss is what actually matters; the terminal signal is discounted to near-nothing. A higher γ propagates it.
- *Lineage:* DreamerV3 (uses γ=0.997), Agent57/long-horizon literature, C4.
- *Change:* `gamma: 0.99 → 0.997` (λ unchanged).
- *Prediction:* better terminal-outcome alignment; possibly higher value-loss (longer credit is harder) but better win-rate.

**R3 — Stronger value learning (PPG-lite).**
- *Hypothesis:* the shared trunk under-trains the critic; a better critic → cleaner advantages → better policy. This is the cheap proxy for PPG/MAPPO's "train value harder / normalize value."
- *Lineage:* PPG (Paper 2), MAPPO value-normalization (Paper 1), Big idea 2.
- *Change:* `vf_coef: 0.5 → 1.0`.
- *Prediction:* lower value loss, more stable advantages, modest win-rate gain. (If it underperforms, that itself argues for the *full* fix — value normalization — over just reweighting.)

**R4 — Faster self-play curriculum (opponent diversity sooner).**
- *Hypothesis:* training almost entirely vs apex for 200 updates overfits to apex's quirks; injecting self-play earlier builds a more robust policy.
- *Lineage:* XLand (Paper 6), DeepNash (Paper 5), Big idea 4.
- *Change:* `rule_based_decay_updates: 1000 → 150` so self-play ramps in *within* the run.
- *Prediction:* possibly lower vs-apex win-rate (less apex-specialization) but a more robust policy; an informative trade-off either way.

**R5 — Remove the early-production reward bonus (reward-shaping bias).**
- *Hypothesis:* the 10× early-production bonus is strong non-potential shaping; it may bias the policy toward myopic early grabs rather than winning. Testing its removal isolates whether the shaping helps or distorts the true objective.
- *Lineage:* Go-Explore (Paper 7) & DreamerV3 (Paper 9) on respecting the true (sparse) objective; potential-based-shaping theory.
- *Change:* `early_prod_bonus: 9.0 → 0.0`.
- *Prediction:* uncertain — this is a genuine question. Either shaping is load-bearing (win-rate drops) or it's a distortion (win-rate holds/improves with cleaner credit).

---

## 5. Experiments — what actually happened

Full write-up, configs, plots, and per-recommendation insight: **`experiments/RECOMMENDATIONS_REPORT.md`**. The headline:

Each recommendation was run one-variable-at-a-time for 200 PPO updates from a *byte-identical* BC warm start (shared prime checkpoint + cached apex demos). Result:

> **All six variants (baseline + R1–R5) reached 0% win-rate vs apex and 100% vs random**, at both update 100 and 200.

The binary metric saturated, so a **score-margin tie-breaker** (20 games vs apex, alternating sides) was run:

| Rank | Variant | margin vs apex | survival (steps) | planets held |
|---|---|---|---|---|
| 1 | R3 vf_coef=1.0 | −3306 | 118 | 0.0 |
| 2 | baseline | −3552 | 119 | 0.0 |
| 3 | R1 ent_anneal | −3668 | 119 | 0.0 |
| 4 | R2 γ=0.997 | −3991 | 137 | 0.0 |
| 5 | R5 no_prod_bonus | −4004 | 139 | 0.0 |
| 6 | R4 fast_selfplay | −4077 | 126 | 0.0 |

Every agent is **eliminated by apex around step 120–139 of 498 with 0 planets left**; the margin spread is within noise. **No knob is remotely competitive.** The training logs additionally expose the *real* problems: entropy thrashing (0.01 ↔ 88), value-loss spikes (up to ~3.1), and `eps=0` on nearly every update (32-step rollouts vs 150–500-step episodes ⇒ the terminal win/loss almost never reaches an update).

**Interpretation against the synthesis (§3):** this is a clean negative result that *localizes* the bottleneck. The agent loses not because of γ/entropy/value-weight/self-play-timing/shaping, but because of (a) **Big idea 3** — the action representation is broken (ship-fraction was entangled with target selection, so fleets are systematically too small to capture or hold, and PPO had no gradient on fleet size), and (b) **Big idea 2** — the value function's *scale* is unmanaged, so credit assignment is noise. Knobs can't fix foundations.

---

## 6. Final recommendation (implemented)

The experiments justify a two-phase plan, **now implemented and smoke-validated** in `v2/`.

### Phase 1 — Fix the foundations (prerequisites for any learning path)

| Fix | What changed | Files | Paper lineage | Status |
|---|---|---|---|---|
| **1. Dedicated masked fraction head** | Factored action `(target, ship-fraction)`: a per-`(source→target)` categorical over fraction bins, decoupled from target selection. PPO now has a real gradient on fleet size; **BC now clones apex's fractions** (previously discarded); ExIt distills them. Params 518K→551K. | `model.py`, `actions.py`, `ppo.py`, `imitation.py`, `agent.py` | DouZero (describe the action), AlphaStar/v1 factored head | ✅ done |
| **2. Mean per-planet entropy** | Entropy bonus is the mean over owned planets, not the sum — removes board-dependent exploration pressure (the 0.01↔88 swings). Post-fix entropy sits at a sane ~2–3. | `actions.py` | MAPPO, "What Matters" | ✅ done |
| **3. Symlog value targets** | `symlog`-compress the value target (scale-robust regression); GAE consumes `symexp(value)` so returns stay in real space. Gated by `ppo.value_symlog`. | `ppo.py`, `train.py`, `config.py` | DreamerV3 | ✅ done |
| **4. Longer rollouts** | `ppo.rollout_steps` exposed/raised so episodes complete and the terminal signal reaches updates — meaningful *only* once the critic (Fix 3) is usable. | `config.py` + configs | DreamerV3 / long-horizon | ✅ done |

**Correctness validation:** all modules import; a forward→sample→recompute roundtrip confirms `log_prob` recomputed in the PPO update matches the sampled `log_prob` (the trust-region invariant — ratio = 1 on epoch 0); BC loss now combines target + fraction cross-entropy; a full training smoke runs with sane entropy; the updated submission `agent.py` plays a full 500-step game.

### Phase 2 — Pivot the main effort from PPO-grind to imitation + search (ExIt)

We own a perfect, fast forward model (`src/simulator.py`); model-free PG is the weakest tool for that situation. The new **v2 Expert Iteration** pipeline closes the Go-Explore / EfficientZero / AlphaZero loop *without the hard model-learning step*:

- `v2/search.py` — per-planet lookahead: forward-simulate candidate `(target, fraction)` actions against the ground-truth simulator, softmax the scores → improved `target_probs[P+1]` and `frac_probs[P,K]` aligned to OrbitNet's heads.
- `v2/exit_train.py` — loop: **BC-clone apex (Fix 1 makes this faithful) → collect self-play games → search-improve every owned-planet decision → supervised-distill (target CE + fraction CE + value MSE) → repeat.**
- `configs/v2_exit.yaml` — BC warm start (reuses the cached apex demos) + 50 ExIt iterations.

Run it with:
```bash
uv run python -m v2.exit_train --config configs/v2_exit.yaml
```

Smoke-validated: collects games, runs search, distills, checkpoints, and the resulting model plays valid full games.

### Phase 3 — Self-play fine-tune, stabilized (future)

Once ExIt beats apex, add a self-play/league stage with a **DeepNash R-NaD-style KL-to-reference anchor** (Big idea 1 + 4) to diversify opponents without the cycling that naïve self-play and R4 exhibited. Keep PPO (with the Phase-1 fixes) as a *fine-tuner*, never the primary learner.

### The one-sentence bottom line

> **Fix the action head so imitation can faithfully clone apex, then let lookahead search against our perfect simulator do the heavy lifting — that is the shortest path to an agent that beats apex, and the foundation every other technique in this survey builds on.**
