"""PPO algorithm for V2 pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import torch

import torch

from .actions import action_log_prob_and_entropy
from .model import OrbitNet


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric log: compresses large magnitudes, ~identity near 0 (DreamerV3)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of symlog."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


class ValueNorm:
    """Running mean/std normalization of value targets (MAPPO ValueNorm / PopArt-lite).

    Tier 0.3. The value head learns in normalized space; targets are normalized
    by running (mean, var) updated once per PPO update, and predictions are
    denormalized for GAE. Handles the DRIFTING return scale (PBRS + rising
    win-rate) that symlog's static compression does not. Use INSTEAD of symlog.
    Stats are plain Python floats so it pickles trivially for checkpoints.
    """

    def __init__(self, beta: float = 0.05) -> None:
        self.beta = beta
        self.mean = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, returns: torch.Tensor) -> None:
        bm = float(returns.mean().item())
        bv = float(returns.var(unbiased=False).item())
        self.count += 1
        if self.count == 1:
            self.mean, self.var = bm, bv
        else:
            self.mean = (1 - self.beta) * self.mean + self.beta * bm
            self.var = (1 - self.beta) * self.var + self.beta * bv

    def _std(self) -> float:
        return max(self.var, 1e-6) ** 0.5

    def normalize(self, x):
        return (x - self.mean) / self._std()

    def denormalize(self, x):
        return x * self._std() + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count, "beta": self.beta}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"]; self.var = d["var"]
        self.count = d["count"]; self.beta = d.get("beta", self.beta)


def _target_policy_kl(old_logits: torch.Tensor, new_logits: torch.Tensor,
                      own_mask: torch.Tensor) -> torch.Tensor:
    """KL(old || new) over the per-planet target distribution, own rows only.

    Used as the PPG aux-phase clone term that freezes the policy while the value
    function is trained hard. Masked logits (-inf) contribute 0 (guarded against
    0*nan). KL on the target head only — the dominant action; cheap and stable.
    """
    om = own_mask.bool()
    if not om.any():
        return new_logits.sum() * 0.0
    ol = old_logits[om]  # [K, P+1]
    nl = new_logits[om]
    logp_old = torch.log_softmax(ol, dim=-1)
    logp_new = torch.log_softmax(nl, dim=-1)
    p_old = logp_old.exp()
    term = p_old * (logp_old - logp_new)
    term = torch.where(p_old > 0, term, torch.zeros_like(term))
    return term.sum(dim=-1).mean()


@dataclass
class V2TransitionBatch:
    planet_features: torch.Tensor    # [N, P, F]
    global_features: torch.Tensor    # [N, G]
    planet_mask: torch.Tensor        # [N, P] bool
    own_mask: torch.Tensor           # [N, P] bool
    reachability_mask: torch.Tensor  # [N, P, P] bool
    target_indices: torch.Tensor     # [N, P] long
    frac_indices: torch.Tensor       # [N, P] long (ship-fraction bin per planet)
    log_prob: torch.Tensor           # [N]
    returns: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]
    values: torch.Tensor             # [N]
    pair_features: torch.Tensor | None = None  # [N, P, P, pf] (v3; None if disabled)
    # v4 Tier 1.2: shot-success labels. Parallel arrays referencing buffer rows:
    # at row shot_idx[k] the launch (shot_src[k] -> shot_tgt[k]) had outcome
    # shot_label[k] = 1 if we owned the target shot_horizon steps later.
    shot_idx: torch.Tensor | None = None     # [M] long
    shot_src: torch.Tensor | None = None     # [M] long
    shot_tgt: torch.Tensor | None = None     # [M] long
    shot_label: torch.Tensor | None = None   # [M] float


def v2_ppo_update(
    model: OrbitNet,
    optimizer: torch.optim.Optimizer,
    batch: V2TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
    demo_buffer: object | None = None,
    imitation_coef: float = 0.0,
    value_symlog: bool = False,
    value_norm: "ValueNorm | None" = None,
) -> dict[str, float]:
    """Clipped PPO update for V2 pipeline."""
    N = batch.planet_features.shape[0]
    if N < 4:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    # Move to device
    pf = batch.planet_features.to(device)
    gf = batch.global_features.to(device)
    pm = batch.planet_mask.to(device).bool()
    om = batch.own_mask.to(device).bool()
    rm = batch.reachability_mask.to(device).bool()
    ti = batch.target_indices.to(device)
    fi = batch.frac_indices.to(device)
    pairf = batch.pair_features.to(device) if batch.pair_features is not None else None
    old_log_prob = batch.log_prob.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    old_values = batch.values.to(device)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    # Tier 0.3: update the running return normalizer once per PPO update (before
    # normalizing targets). Mutually exclusive with value_symlog.
    if value_norm is not None:
        value_norm.update(returns)

    minibatch_size = min(N, max(1, minibatch_size))
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
               "imitation_loss": 0.0}
    updates = 0

    for _ in range(epochs):
        order = torch.randperm(N, device=device)
        for start in range(0, N, minibatch_size):
            idx = order[start:start + minibatch_size]

            output = model(pf[idx], gf[idx], pm[idx], om[idx], rm[idx],
                           pairf[idx] if pairf is not None else None)

            new_log_prob, entropy = action_log_prob_and_entropy(
                output, om[idx], ti[idx], fi[idx],
            )

            # Importance ratio
            ratio = (new_log_prob - old_log_prob[idx]).exp()
            adv = advantages[idx]

            # Clipped policy loss
            policy_loss = torch.maximum(
                -adv * ratio,
                -adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
            ).mean()

            # Clipped value loss. The prediction (raw head output) and old_values
            # live in whatever space the target uses: symlog-compressed,
            # PopArt-normalized, or raw. value_norm takes precedence over symlog.
            if value_norm is not None:
                value_target = value_norm.normalize(returns[idx])
            elif value_symlog:
                value_target = symlog(returns[idx])
            else:
                value_target = returns[idx]
            value_pred = output.value
            value_clipped = old_values[idx] + torch.clamp(
                value_pred - old_values[idx], -clip_coef, clip_coef,
            )
            vl_unclipped = (value_target - value_pred).pow(2)
            vl_clipped = (value_target - value_clipped).pow(2)
            value_loss = 0.5 * torch.maximum(vl_unclipped, vl_clipped).mean()

            entropy_mean = entropy.mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

            # Imitation loss blending
            im_loss_val = 0.0
            if demo_buffer is not None and imitation_coef > 0.0:
                from .imitation import compute_v2_bc_loss
                demo_batch = demo_buffer.sample_batch(len(idx), device)
                im_loss = compute_v2_bc_loss(model, demo_batch)
                loss = loss + imitation_coef * im_loss
                im_loss_val = float(im_loss.detach().cpu())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy_mean.detach().cpu())
            metrics["imitation_loss"] += im_loss_val
            updates += 1

    return {k: v / max(updates, 1) for k, v in metrics.items()}


def v2_shot_aux_update(
    model: OrbitNet,
    optimizer: torch.optim.Optimizer,
    batch: V2TransitionBatch,
    *,
    coef: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Train the shot-success head (Tier 1.2/0.4) on rollout outcome labels.

    Binary cross-entropy over (buffer-row, source, target) launches with label
    = "did we still own the target shot_horizon steps later". A dense, balanced
    signal vs sparse reward; the same head doubles as the decode-time rejection
    filter. No-op without a shot head, labels, or coef.
    """
    if (coef <= 0.0 or model.shot_success_head is None
            or batch.shot_idx is None or batch.shot_idx.numel() == 0):
        return {"shot_loss": 0.0, "shot_acc": 0.0}

    pf = batch.planet_features.to(device)
    gf = batch.global_features.to(device)
    pm = batch.planet_mask.to(device).bool()
    om = batch.own_mask.to(device).bool()
    rm = batch.reachability_mask.to(device).bool()
    pairf = batch.pair_features.to(device) if batch.pair_features is not None else None
    s_idx = batch.shot_idx.to(device)
    s_src = batch.shot_src.to(device)
    s_tgt = batch.shot_tgt.to(device)
    s_lab = batch.shot_label.to(device).float()

    M = s_idx.shape[0]
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    metrics = {"shot_loss": 0.0, "shot_acc": 0.0}
    steps = 0
    mb = min(M, max(1, minibatch_size))
    for _ in range(max(1, epochs)):
        order = torch.randperm(M, device=device)
        for start in range(0, M, mb):
            sel = order[start:start + mb]
            rows = s_idx[sel]
            # Unique buffer rows in this minibatch -> one forward each.
            uniq, inv = torch.unique(rows, return_inverse=True)
            out = model(pf[uniq], gf[uniq], pm[uniq], om[uniq], rm[uniq],
                        pairf[uniq] if pairf is not None else None)
            logit = out.shot_logits[inv, s_src[sel], s_tgt[sel]]  # [m]
            loss = coef * bce(logit, s_lab[sel])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            with torch.no_grad():
                acc = ((logit > 0).float() == s_lab[sel]).float().mean()
            metrics["shot_loss"] += float(loss.detach().cpu())
            metrics["shot_acc"] += float(acc.cpu())
            steps += 1
    return {k: v / max(steps, 1) for k, v in metrics.items()}


def v2_aux_phase(
    model: OrbitNet,
    optimizer: torch.optim.Optimizer,
    batch: V2TransitionBatch,
    *,
    aux_epochs: int,
    beta_clone: float,
    minibatch_size: int,
    device: torch.device,
    value_symlog: bool = False,
    value_norm: "ValueNorm | None" = None,
) -> dict[str, float]:
    """PPG auxiliary value phase (Tier 1.1).

    After the PPO policy phase, train the value function HARD for `aux_epochs`
    on the same buffer — through BOTH the main value head and the aux value head
    on the shared trunk — while a `beta_clone * KL(old_policy || new_policy)` term
    freezes the action distribution. This lets the critic (and the trunk's value
    features) improve without the actor overfitting, the documented fix for the
    shared-trunk actor/critic collapse. No-op if the model has no aux value head.
    """
    if aux_epochs <= 0 or model.aux_value_head is None:
        return {"aux_value_loss": 0.0, "aux_kl": 0.0}

    import copy
    N = batch.planet_features.shape[0]
    if N < 4:
        return {"aux_value_loss": 0.0, "aux_kl": 0.0}

    pf = batch.planet_features.to(device)
    gf = batch.global_features.to(device)
    pm = batch.planet_mask.to(device).bool()
    om = batch.own_mask.to(device).bool()
    rm = batch.reachability_mask.to(device).bool()
    pairf = batch.pair_features.to(device) if batch.pair_features is not None else None
    returns = batch.returns.to(device)

    # Snapshot the pre-aux policy as the clone reference.
    old_model = copy.deepcopy(model).eval()
    for p in old_model.parameters():
        p.requires_grad_(False)

    def _vtarget(r):
        if value_norm is not None:
            return value_norm.normalize(r)
        if value_symlog:
            return symlog(r)
        return r

    minibatch_size = min(N, max(1, minibatch_size))
    metrics = {"aux_value_loss": 0.0, "aux_kl": 0.0}
    steps = 0
    for _ in range(aux_epochs):
        order = torch.randperm(N, device=device)
        for start in range(0, N, minibatch_size):
            idx = order[start:start + minibatch_size]
            out = model(pf[idx], gf[idx], pm[idx], om[idx], rm[idx],
                        pairf[idx] if pairf is not None else None)
            with torch.no_grad():
                old_out = old_model(pf[idx], gf[idx], pm[idx], om[idx], rm[idx],
                                    pairf[idx] if pairf is not None else None)

            vt = _vtarget(returns[idx])
            v_loss = 0.5 * (vt - out.value).pow(2).mean()
            if out.aux_value is not None:
                v_loss = v_loss + 0.5 * (vt - out.aux_value).pow(2).mean()

            kl = _target_policy_kl(old_out.logits, out.logits, om[idx])
            loss = v_loss + beta_clone * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            metrics["aux_value_loss"] += float(v_loss.detach().cpu())
            metrics["aux_kl"] += float(kl.detach().cpu())
            steps += 1

    return {k: v / max(steps, 1) for k, v in metrics.items()}
