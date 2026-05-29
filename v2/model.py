"""OrbitNet: simultaneous all-planet transformer with pairwise output head."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.policy import TransformerBlock

from .config import V2ModelConfig


@dataclass(slots=True)
class OrbitNetOutput:
    logits: torch.Tensor        # [B, max_planets, max_planets+1] (hold + targets)
    value: torch.Tensor         # [B]
    frac_logits: torch.Tensor   # [B, max_planets, max_planets, n_fractions] per (source->target) ship-fraction


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
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d, cfg.n_heads, cfg.ff_dim)
            for _ in range(cfg.n_layers)
        ])
        self.final_ln = nn.LayerNorm(d)

        # Pairwise output head
        self.src_proj = nn.Linear(d, d)
        self.tgt_proj = nn.Linear(d, d)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )
        self.hold_head = nn.Linear(d, 1)

        # Factored ship-fraction head: per (source->target) pair, a distribution
        # over discrete fraction bins. Decouples "how many ships" from "which
        # target", so PPO/BC/ExIt can learn fleet size directly.
        self.n_fractions = cfg.n_fractions
        self.frac_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Linear(d, cfg.n_fractions),
        )

        # Value head: masked mean pool -> MLP -> scalar
        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

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

    def forward(
        self,
        planet_features: torch.Tensor,    # [B, P, F]
        global_features: torch.Tensor,    # [B, G]
        planet_mask: torch.Tensor,        # [B, P] bool (True = exists)
        own_mask: torch.Tensor,           # [B, P] bool (True = we own it)
        reachability_mask: torch.Tensor | None = None,  # [B, P, P] bool (True = reachable)
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
        logits[:, :, 1:] = logits[:, :, 1:].masked_fill(
            target_invalid.unsqueeze(1), NEG_INF
        )

        # Self-targeting diagonal -> -inf (can't send to self)
        diag_mask = torch.eye(P, dtype=torch.bool, device=logits.device)
        diag_mask = diag_mask.unsqueeze(0).expand(B, -1, -1)  # [B, P, P]
        logits[:, :, 1:] = logits[:, :, 1:].masked_fill(diag_mask, NEG_INF)

        # Reachability mask: block targets unreachable due to sun
        if reachability_mask is not None:
            logits[:, :, 1:] = logits[:, :, 1:].masked_fill(
                ~reachability_mask, NEG_INF
            )

        # 6. Value head: masked mean pool
        mask_float = planet_mask.float().unsqueeze(-1)  # [B, P, 1]
        pooled = (x * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1.0)  # [B, d]
        value = self.value_head(pooled).squeeze(-1)  # [B]

        return OrbitNetOutput(logits=logits, value=value, frac_logits=frac_logits)
