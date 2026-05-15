from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import EnvConfig, ModelConfig
from .features import GLOBAL_DIM, KNN_SCALAR_DIM, SOURCE_SCALAR_DIM, TARGET_SCALAR_DIM


@dataclass(slots=True)
class PolicyOutput:
    target_logits: torch.Tensor    # [B, 1+T] over NoOp + targets
    fraction_logits: torch.Tensor  # [B, T, num_fractions] per target
    value: torch.Tensor            # [B]


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block."""

    def __init__(self, embed_dim: int, n_heads: int, ff_dim: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm attention
        normed = self.ln1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        # Pre-norm feedforward
        x = x + self.ff(self.ln2(x))
        return x


class TransformerPolicy(nn.Module):
    """Transformer policy for per-planet sequential decisions.

    Token layout: [CLS, NoOp, Target_1, ..., Target_T, PAD...]
    - CLS (token 0): learnable, used for value head
    - NoOp (token 1): learnable, represents "send nothing"
    - Targets (tokens 2..T+1): projected from global+source+target embeddings
    """

    def __init__(self, model_cfg: ModelConfig, env_cfg: EnvConfig) -> None:
        super().__init__()
        d = model_cfg.embed_dim
        self.max_targets = env_cfg.max_targets
        self.num_fractions = len(env_cfg.ship_fractions)
        self.seq_len = 2 + env_cfg.max_targets  # CLS + NoOp + targets

        # Shared position encoder for all (x,y) inputs
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, model_cfg.pos_hidden),
            nn.ReLU(),
            nn.Linear(model_cfg.pos_hidden, d),
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(GLOBAL_DIM, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Source planet encoder: pos_emb + scalars → embed
        self.source_encoder = nn.Sequential(
            nn.Linear(d + SOURCE_SCALAR_DIM, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # KNN encoder: pos_emb + scalars → embed (shared across K neighbors)
        self.knn_encoder = nn.Sequential(
            nn.Linear(d + KNN_SCALAR_DIM, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Combine source + KNN pool
        self.source_knn_combiner = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
        )

        # Target encoder: pos_emb + scalars → embed
        self.target_encoder = nn.Sequential(
            nn.Linear(d + TARGET_SCALAR_DIM, d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

        # Token projection: concat(global, source, target) → embed
        self.token_projection = nn.Linear(d * 3, d)

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(d) * 0.02)
        self.noop_token = nn.Parameter(torch.randn(d) * 0.02)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d, model_cfg.n_heads, model_cfg.ff_dim)
            for _ in range(model_cfg.n_layers)
        ])
        self.final_ln = nn.LayerNorm(d)

        # Output heads
        self.target_head = nn.Linear(d, 1)
        self.fraction_head = nn.Linear(d, self.num_fractions)
        self.value_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, 1),
        )

        self._init_output_heads()

    def _init_output_heads(self) -> None:
        """Zero-init output heads for stable early training.

        target_head → uniform policy (all logits 0)
        value_head final layer → 0.5 value (sigmoid midpoint)
        """
        nn.init.zeros_(self.target_head.weight)
        nn.init.zeros_(self.target_head.bias)
        nn.init.zeros_(self.fraction_head.weight)
        nn.init.zeros_(self.fraction_head.bias)
        # Zero-init last linear in value_head Sequential
        last_layer = self.value_head[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def forward(
        self,
        global_features: torch.Tensor,    # [B, GLOBAL_DIM]
        source_scalars: torch.Tensor,      # [B, SOURCE_SCALAR_DIM]
        source_positions: torch.Tensor,    # [B, 2]
        knn_scalars: torch.Tensor,         # [B, K, KNN_SCALAR_DIM]
        knn_positions: torch.Tensor,       # [B, K, 2]
        target_scalars: torch.Tensor,      # [B, T, TARGET_SCALAR_DIM]
        target_positions: torch.Tensor,    # [B, T, 2]
        target_mask: torch.Tensor,         # [B, T+2] bool (True = valid)
    ) -> PolicyOutput:
        B = global_features.shape[0]
        T = self.max_targets

        # 1. Encode global
        global_emb = self.global_encoder(global_features)  # [B, d]

        # 2. Encode source planet
        src_pos_emb = self.pos_encoder(source_positions)  # [B, d]
        src_emb = self.source_encoder(torch.cat([src_pos_emb, source_scalars], dim=-1))  # [B, d]

        # 3. Encode KNN neighbors and mean-pool
        K = knn_scalars.shape[1]
        knn_pos_flat = knn_positions.reshape(B * K, 2)
        knn_pos_emb = self.pos_encoder(knn_pos_flat).reshape(B, K, -1)  # [B, K, d]
        knn_input = torch.cat([knn_pos_emb, knn_scalars], dim=-1)  # [B, K, d+KNN_SCALAR_DIM]
        knn_emb = self.knn_encoder(knn_input)  # [B, K, d]
        knn_pool = knn_emb.mean(dim=1)  # [B, d]

        # 4. Combine source + KNN
        source_combined = self.source_knn_combiner(
            torch.cat([src_emb, knn_pool], dim=-1)
        )  # [B, d]

        # 5. Encode targets
        tgt_pos_flat = target_positions.reshape(B * T, 2)
        tgt_pos_emb = self.pos_encoder(tgt_pos_flat).reshape(B, T, -1)  # [B, T, d]
        tgt_input = torch.cat([tgt_pos_emb, target_scalars], dim=-1)  # [B, T, d+TARGET_SCALAR_DIM]
        tgt_emb = self.target_encoder(tgt_input)  # [B, T, d]

        # 6. Build token sequence [CLS, NoOp, Target_1, ..., Target_T]
        # Global and source are broadcast into each target token
        global_exp = global_emb.unsqueeze(1).expand(B, T, -1)  # [B, T, d]
        source_exp = source_combined.unsqueeze(1).expand(B, T, -1)  # [B, T, d]
        target_tokens = self.token_projection(
            torch.cat([global_exp, source_exp, tgt_emb], dim=-1)
        )  # [B, T, d]

        cls_tokens = self.cls_token.unsqueeze(0).expand(B, 1, -1)  # [B, 1, d]
        noop_tokens = self.noop_token.unsqueeze(0).expand(B, 1, -1)  # [B, 1, d]

        tokens = torch.cat([cls_tokens, noop_tokens, target_tokens], dim=1)  # [B, T+2, d]

        # 7. Attention mask: ~target_mask means masked (True = ignore in MHA)
        key_padding_mask = ~target_mask  # [B, T+2]

        # 8. Transformer
        x = tokens
        for block in self.transformer_blocks:
            x = block(x, key_padding_mask=key_padding_mask)
        x = self.final_ln(x)

        # 9. Outputs
        # Target selection: scores for tokens 1..T+1 (NoOp + targets)
        selection_tokens = x[:, 1:, :]  # [B, 1+T, d]
        target_logits = self.target_head(selection_tokens).squeeze(-1)  # [B, 1+T]
        # Mask invalid targets (keep NoOp = index 0 in selection, maps to token 1)
        selection_mask = target_mask[:, 1:]  # [B, 1+T]
        target_logits = target_logits.masked_fill(~selection_mask, torch.finfo(target_logits.dtype).min)

        # Fraction logits: for target tokens only (tokens 2..T+1)
        fraction_tokens = x[:, 2:, :]  # [B, T, d]
        fraction_logits = self.fraction_head(fraction_tokens)  # [B, T, num_fractions]

        # Value: from CLS token
        value = self.value_head(x[:, 0, :]).squeeze(-1)  # [B]

        return PolicyOutput(
            target_logits=target_logits,
            fraction_logits=fraction_logits,
            value=value,
        )
