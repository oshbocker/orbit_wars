"""OrbitNet: simultaneous all-planet transformer with pairwise output head."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.policy import TransformerBlock

from .config import V2ModelConfig


@dataclass(slots=True)
class OrbitNetOutput:
    logits: torch.Tensor  # [B, max_planets, max_planets+1] (hold + targets)
    value: torch.Tensor  # [B]
    frac_logits: (
        torch.Tensor
    )  # [B, max_planets, max_planets, n_fractions] per (source->target) ship-fraction
    aux_value: torch.Tensor | None = None  # [B] PPG auxiliary value head (v4 Tier 1.1)
    shot_logits: torch.Tensor | None = (
        None  # [B, P, P] logit P(own target N turns after arrival) (v4 Tier 1.2/0.4)
    )
    gate_logits: torch.Tensor | None = (
        None  # [B, P] per-source launch/no-launch logit (winbc separate gate head)
    )


class OrbitNet(nn.Module):
    """Simultaneous all-planet transformer with pairwise output head.

    Input:  planet_features [B,P,F], global_features [B,G],
            planet_mask [B,P], own_mask [B,P]
    Output: logits [B,P,P+1], value [B]

    logits[:,i,0] = hold logit for planet i
    logits[:,i,j+1] = send-to-planet-j logit for planet i
    """

    def __init__(self, cfg: V2ModelConfig) -> None:
        super().__init__()
        d = cfg.embed_dim
        P = 40  # max planets (fixed)
        self.max_planets = P

        # Planet embedding: F -> d
        self.planet_embed = nn.Sequential(
            nn.Linear(cfg.planet_feat_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        # Global embedding: G -> d (broadcast-added to planet embeddings)
        self.global_embed = nn.Sequential(
            nn.Linear(cfg.global_feat_dim, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Self-attention encoder
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(d, cfg.n_heads, cfg.ff_dim) for _ in range(cfg.n_layers)]
        )
        self.final_ln = nn.LayerNorm(d)

        # Optional pairwise (source->target) input features (v3): travel_time,
        # required-ships-on-arrival, intercept-valid. Concatenated onto the pair
        # embedding so the target/fraction heads can reason about reachability
        # and fleet sizing directly. 0 = disabled (identical to the prior model).
        self.pair_feat_dim = cfg.pair_feat_dim if cfg.use_pair_features else 0
        pair_in = 2 * d + self.pair_feat_dim

        # Pairwise output head
        self.src_proj = nn.Linear(d, d)
        self.tgt_proj = nn.Linear(d, d)
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_in, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.hold_head = nn.Linear(d, 1)

        # Factored ship-fraction head: per (source->target) pair, a distribution
        # over discrete fraction bins. Decouples "how many ships" from "which
        # target", so PPO/BC/ExIt can learn fleet size directly.
        self.n_fractions = cfg.n_fractions
        self.frac_mlp = nn.Sequential(
            nn.Linear(pair_in, d),
            nn.ReLU(),
            nn.Linear(d, cfg.n_fractions),
        )

        # Value head: masked mean pool -> MLP -> scalar
        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        # v4 Tier 1.1: PPG auxiliary value head on the shared trunk. Lets the
        # aux phase train value features into the trunk while a KL-clone term
        # holds the policy fixed — decoupling critic depth from actor overfitting.
        self.aux_value_head = None
        if getattr(cfg, "aux_value_head", False):
            self.aux_value_head = nn.Sequential(
                nn.Linear(d, d),
                nn.ReLU(),
                nn.Linear(d, 1),
            )

        # v4 Tier 1.2/0.4: per-(source->target) shot-success head. Predicts
        # P(we still own the target N turns after arrival); trained as an aux
        # loss and reused as a rejection filter at decode ("shot validator").
        self.shot_success_head = None
        if getattr(cfg, "shot_success_head", False):
            self.shot_success_head = nn.Sequential(
                nn.Linear(pair_in, d),
                nn.ReLU(),
                nn.Linear(d, 1),
            )

        # winbc regime fix: separate per-source launch/no-launch GATE in front of the
        # pointer (vkhydras). When present, decode is gate(act?) -> pointer(which target),
        # rather than hold competing as a class in the target softmax.
        self.gate_head = None
        if getattr(cfg, "launch_gate_head", False):
            self.gate_head = nn.Linear(d, 1)

        self._init_output_heads()

    def _init_output_heads(self) -> None:
        """Zero-init output heads for stable early training."""
        nn.init.zeros_(self.pair_mlp[-1].weight)
        nn.init.zeros_(self.pair_mlp[-1].bias)
        nn.init.zeros_(self.hold_head.weight)
        nn.init.zeros_(self.hold_head.bias)
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)
        # Zero-init fraction head -> uniform fraction distribution at start.
        nn.init.zeros_(self.frac_mlp[-1].weight)
        nn.init.zeros_(self.frac_mlp[-1].bias)
        # Zero-init v4 aux heads (aux value -> 0; shot logit -> 0 => P=0.5 prior).
        if self.aux_value_head is not None:
            nn.init.zeros_(self.aux_value_head[-1].weight)
            nn.init.zeros_(self.aux_value_head[-1].bias)
        if self.shot_success_head is not None:
            nn.init.zeros_(self.shot_success_head[-1].weight)
            nn.init.zeros_(self.shot_success_head[-1].bias)
        if self.gate_head is not None:
            nn.init.zeros_(self.gate_head.weight)
            nn.init.zeros_(self.gate_head.bias)

    def forward(
        self,
        planet_features: torch.Tensor,  # [B, P, F]
        global_features: torch.Tensor,  # [B, G]
        planet_mask: torch.Tensor,  # [B, P] bool (True = exists)
        own_mask: torch.Tensor,  # [B, P] bool (True = we own it)
        reachability_mask: torch.Tensor | None = None,  # [B, P, P] bool (True = reachable)
        pair_features: torch.Tensor | None = None,  # [B, P, P, pair_feat_dim] (v3)
    ) -> OrbitNetOutput:
        B, P, _ = planet_features.shape

        # 1. Planet embedding
        x = self.planet_embed(planet_features)  # [B, P, d]

        # 2. Global embedding (broadcast add)
        g = self.global_embed(global_features)  # [B, d]
        x = x + g.unsqueeze(1)  # [B, P, d]

        # 3. Self-attention with masking
        key_padding_mask = ~planet_mask  # True = ignore
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.final_ln(x)  # [B, P, d]

        # 4. Pairwise output head
        src = self.src_proj(x)  # [B, P, d]
        tgt = self.tgt_proj(x)  # [B, P, d]

        # Expand for pairwise: [B, P, P, 2d]
        src_exp = src.unsqueeze(2).expand(B, P, P, -1)
        tgt_exp = tgt.unsqueeze(1).expand(B, P, P, -1)
        pair_input = torch.cat([src_exp, tgt_exp], dim=-1)  # [B, P, P, 2d]
        # Concatenate optional pairwise input features (v3 travel-time etc.).
        # If enabled but not supplied (e.g. the BC pretrain path, which has no
        # pair tensor), zero-fill so the layer shape matches — zeros = "no
        # pairwise info available" rather than crashing.
        if self.pair_feat_dim > 0:
            if pair_features is None:
                pair_features = torch.zeros(
                    B,
                    P,
                    P,
                    self.pair_feat_dim,
                    dtype=pair_input.dtype,
                    device=pair_input.device,
                )
            pair_input = torch.cat([pair_input, pair_features], dim=-1)  # [B,P,P,2d+pf]
        pair_logits = self.pair_mlp(pair_input).squeeze(-1)  # [B, P, P]

        # Ship-fraction logits per (source->target) pair
        frac_logits = self.frac_mlp(pair_input)  # [B, P, P, n_fractions]

        # Hold logits
        hold_logits = self.hold_head(x)  # [B, P, 1]

        # Concatenate: [B, P, 1+P] where [:,:,0]=hold, [:,:,1:]=targets
        logits = torch.cat([hold_logits, pair_logits], dim=-1)  # [B, P, P+1]

        # 5. Masking
        NEG_INF = torch.finfo(logits.dtype).min

        # Non-owned sources: all logits -> -inf
        source_invalid = ~own_mask  # [B, P]
        logits = logits.masked_fill(source_invalid.unsqueeze(-1), NEG_INF)

        # Non-existent targets: target logits -> -inf
        # target_mask applies to logits[:,:,1:] (the P target columns)
        target_invalid = ~planet_mask  # [B, P]
        logits[:, :, 1:] = logits[:, :, 1:].masked_fill(target_invalid.unsqueeze(1), NEG_INF)

        # Self-targeting diagonal -> -inf (can't send to self)
        diag_mask = torch.eye(P, dtype=torch.bool, device=logits.device)
        diag_mask = diag_mask.unsqueeze(0).expand(B, -1, -1)  # [B, P, P]
        logits[:, :, 1:] = logits[:, :, 1:].masked_fill(diag_mask, NEG_INF)

        # Reachability mask: block targets unreachable due to sun
        if reachability_mask is not None:
            logits[:, :, 1:] = logits[:, :, 1:].masked_fill(~reachability_mask, NEG_INF)

        # 6. Value head: masked mean pool
        mask_float = planet_mask.float().unsqueeze(-1)  # [B, P, 1]
        pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)  # [B, d]
        value = self.value_head(pooled).squeeze(-1)  # [B]

        # v4 auxiliary heads (None unless enabled in config)
        aux_value = None
        if self.aux_value_head is not None:
            aux_value = self.aux_value_head(pooled).squeeze(-1)  # [B]
        shot_logits = None
        if self.shot_success_head is not None:
            # reuse pair_input [B,P,P,2d+pf] built above for the pair/frac heads
            shot_logits = self.shot_success_head(pair_input).squeeze(-1)  # [B, P, P]

        gate_logits = None
        if self.gate_head is not None:
            gate_logits = self.gate_head(x).squeeze(-1)  # [B, P]
            # Non-owned sources can't launch -> -inf so decode/metrics ignore them.
            gate_logits = gate_logits.masked_fill(source_invalid, NEG_INF)

        return OrbitNetOutput(
            logits=logits,
            value=value,
            frac_logits=frac_logits,
            aux_value=aux_value,
            shot_logits=shot_logits,
            gate_logits=gate_logits,
        )

    def value_only(
        self,
        planet_features: torch.Tensor,  # [B, P, F]
        global_features: torch.Tensor,  # [B, G]
        planet_mask: torch.Tensor,  # [B, P] bool
    ) -> torch.Tensor:
        """Value head only (trunk -> masked mean pool -> value), skipping the
        O(P^2) pairwise output heads. ~30-100x cheaper than forward() on CPU —
        used to score ExIt search leaves (Tier 3.2) where only the value matters."""
        x = self.planet_embed(planet_features)
        g = self.global_embed(global_features)
        x = x + g.unsqueeze(1)
        key_padding_mask = ~planet_mask
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.final_ln(x)
        mask_float = planet_mask.float().unsqueeze(-1)
        pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)
        return self.value_head(pooled).squeeze(-1)
