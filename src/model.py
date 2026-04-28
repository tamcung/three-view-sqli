#!/usr/bin/env python3
"""
Three-view SQL injection detection model.

Architecture:
  - 3 independent transformer encoders (surface / lexical / AST)
  - Stage 1: attention pool over (z_L, z_A) → z_LA
  - Stage 2: cross-attention (Q=z_LA, K,V=H_S) → z_final
  - Stage 3: concat([z_LA, z_final]) → main classifier
  - Plus 3 aux classifiers (deep supervision per view)
  - View dropout during training
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Generic Transformer encoder (from-scratch)
# ============================================================
class TransformerViewEncoder(nn.Module):
    """4-layer Transformer encoder, learnable positional embedding, Pre-LN, GELU.

    Args:
        vocab_size: vocabulary size
        max_len: maximum sequence length (incl. CLS at position 0)
        d_model: hidden dimension
        n_layers: number of transformer layers (default 4)
        n_heads: number of attention heads
        d_ff: feedforward inner dim
        dropout: dropout rate
        pad_id: padding token id (used for embedding's padding_idx)
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int | None = None,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.emb_norm = nn.LayerNorm(d_model)
        self.emb_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> dict:
        """Returns dict with 'pooled' [B, d] (CLS) and 'full' [B, T, d]."""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.emb_norm(x)
        x = self.emb_dropout(x)

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)
        return {"pooled": x[:, 0], "full": x}


# ============================================================
# Three-view model
# ============================================================
class ThreeViewModel(nn.Module):
    """End-to-end SQL injection classifier with three views and hierarchical fusion."""

    def __init__(
        self,
        surface_vocab_size: int = 50265,
        surface_max_len: int = 513,
        surface_pad_id: int = 1,        # RoBERTa pad
        lex_vocab_size: int = 24,
        lex_max_len: int = 129,
        ast_vocab_size: int = 100,      # 94 actual + buffer
        ast_max_len: int = 257,
        d_surface: int = 384,
        d_abstract: int = 256,          # used for lexical, AST, and fusion space
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Three encoders
        self.surface_enc = TransformerViewEncoder(
            vocab_size=surface_vocab_size,
            max_len=surface_max_len,
            d_model=d_surface,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_surface * 4,
            dropout=dropout,
            pad_id=surface_pad_id,
        )
        self.lexical_enc = TransformerViewEncoder(
            vocab_size=lex_vocab_size,
            max_len=lex_max_len,
            d_model=d_abstract,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_abstract * 4,
            dropout=dropout,
            pad_id=0,
        )
        self.ast_enc = TransformerViewEncoder(
            vocab_size=ast_vocab_size,
            max_len=ast_max_len,
            d_model=d_abstract,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_abstract * 4,
            dropout=dropout,
            pad_id=0,
        )

        # Surface projection 384 → 256 (for cross-attention)
        self.surface_proj = nn.Linear(d_surface, d_abstract)

        # Stage 1: self-attention block over 2-token abstract sequence [z_L, z_A]
        # (replaces attention pool; gives mutual information exchange instead of competition)
        self.stage1_norm1 = nn.LayerNorm(d_abstract)
        self.stage1_self_attn = nn.MultiheadAttention(
            embed_dim=d_abstract,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.stage1_norm2 = nn.LayerNorm(d_abstract)
        self.stage1_ffn = nn.Sequential(
            nn.Linear(d_abstract, d_abstract * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_abstract * 4, d_abstract),
        )

        # Stage 2: cross-attention block (Q = 2-token abstract seq, K,V = surface)
        self.cross_norm_q = nn.LayerNorm(d_abstract)
        self.cross_norm_kv = nn.LayerNorm(d_abstract)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_abstract,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm_ffn = nn.LayerNorm(d_abstract)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_abstract, d_abstract * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_abstract * 4, d_abstract),
        )

        # Aux classifiers (deep supervision per view)
        self.aux_clf_S = nn.Linear(d_surface, 1)
        self.aux_clf_L = nn.Linear(d_abstract, 1)
        self.aux_clf_A = nn.Linear(d_abstract, 1)

        # Main classifier: concat dual path [z_LA; z_final] → 2 * d_abstract → ... → 1
        self.main_classifier = nn.Sequential(
            nn.Linear(d_abstract * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        surface_ids: torch.Tensor,
        surface_mask: torch.Tensor,
        lex_ids: torch.Tensor,
        lex_mask: torch.Tensor,
        ast_ids: torch.Tensor,
        ast_mask: torch.Tensor,
        ast_valid: torch.Tensor | None = None,
        view_dropout_prob: float = 0.0,
    ) -> dict:
        """Forward pass.

        Args:
            *_ids: [B, T] token id tensors
            *_mask: [B, T] attention mask (1 = valid, 0 = pad)
            ast_valid: [B] 0/1 indicating whether AST view is valid (parse succeeded)
            view_dropout_prob: per-view dropout probability (0 at eval/inference)
        """
        # Encode three views
        s_out = self.surface_enc(surface_ids, surface_mask)
        z_S = s_out["pooled"]              # [B, d_surface]
        H_S = s_out["full"]                # [B, T_S, d_surface]

        z_L = self.lexical_enc(lex_ids, lex_mask)["pooled"]  # [B, d_abstract]
        z_A = self.ast_enc(ast_ids, ast_mask)["pooled"]      # [B, d_abstract]

        # AUX predictions (use raw encoder outputs, before fusion)
        p_S = self.aux_clf_S(z_S).squeeze(-1)
        p_L = self.aux_clf_L(z_L).squeeze(-1)
        p_A = self.aux_clf_A(z_A).squeeze(-1)

        # AST validity: if parse failed, replace z_A with zero (the encoder output
        # for an AST that's just <CLS> is uninformative anyway, but we explicitly
        # zero it so the fusion handles it as "AST missing")
        if ast_valid is not None:
            ast_valid_f = ast_valid.float().unsqueeze(-1)
            z_A_eff = z_A * ast_valid_f
        else:
            z_A_eff = z_A

        # View dropout (training only)
        if self.training and view_dropout_prob > 0:
            B = z_S.size(0)
            dev = z_S.device
            keep_S = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            keep_L = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            keep_A = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            z_S_eff = z_S * keep_S
            z_L_eff = z_L * keep_L
            z_A_eff = z_A_eff * keep_A
            H_S_eff = H_S * keep_S.unsqueeze(1)
        else:
            z_S_eff = z_S
            z_L_eff = z_L
            H_S_eff = H_S

        # Project surface to fusion dim 256
        H_S_proj = self.surface_proj(H_S_eff)   # [B, T_S, d_abstract]

        # ===== Stage 1: self-attention over [z_L, z_A] =====
        # Mutual exchange between abstract views (no competition).
        abstract_seq = torch.stack([z_L_eff, z_A_eff], dim=1)  # [B, 2, d_abstract]

        # Self-attention with residual
        q1 = self.stage1_norm1(abstract_seq)
        s1_out, s1_weights = self.stage1_self_attn(q1, q1, q1, need_weights=True)
        abstract_seq = abstract_seq + s1_out

        # FFN with residual
        ff1_out = self.stage1_ffn(self.stage1_norm2(abstract_seq))
        abstract_seq = abstract_seq + ff1_out  # [B, 2, d_abstract]

        # ===== Stage 2: cross-attention block (Q = 2-token abstract seq) =====
        q2 = self.cross_norm_q(abstract_seq)               # [B, 2, d]
        kv = self.cross_norm_kv(H_S_proj)                  # [B, T_S, d]

        if surface_mask is not None:
            kv_pad_mask = ~surface_mask.bool()
        else:
            kv_pad_mask = None

        attn_out, attn_weights = self.cross_attn(
            query=q2, key=kv, value=kv,
            key_padding_mask=kv_pad_mask,
            need_weights=True,
        )
        attended_seq = abstract_seq + attn_out             # [B, 2, d]

        # FFN with residual
        ff2_out = self.cross_ffn(self.cross_norm_ffn(attended_seq))
        attended_seq = attended_seq + ff2_out              # [B, 2, d]

        # ===== Stage 3: pool both, concat dual path =====
        z_LA = abstract_seq.mean(dim=1)                    # [B, d] — pure abstract (post stage 1)
        z_final = attended_seq.mean(dim=1)                 # [B, d] — abstract + surface

        cls_input = torch.cat([z_LA, z_final], dim=-1)     # [B, 2d]
        p_main = self.main_classifier(cls_input).squeeze(-1)

        return {
            "p_main": p_main,
            "p_S": p_S,
            "p_L": p_L,
            "p_A": p_A,
            "stage1_weights": s1_weights,    # [B, 2, 2] — mutual attention between L and A
            "attn_weights": attn_weights,    # [B, 2, T_S] — abstract-to-surface attention
            "z_S": z_S, "z_L": z_L, "z_A": z_A,
            "z_LA": z_LA, "z_final": z_final,
        }

    def compute_loss(
        self,
        output: dict,
        labels: torch.Tensor,
        weights: tuple[float, float, float, float] = (0.7, 0.1, 0.1, 0.1),
    ) -> tuple[torch.Tensor, dict]:
        """Returns (total_loss, loss_components_dict)."""
        labels = labels.float()
        w_main, w_S, w_L, w_A = weights
        loss_main = F.binary_cross_entropy_with_logits(output["p_main"], labels)
        loss_S = F.binary_cross_entropy_with_logits(output["p_S"], labels)
        loss_L = F.binary_cross_entropy_with_logits(output["p_L"], labels)
        loss_A = F.binary_cross_entropy_with_logits(output["p_A"], labels)
        total = w_main * loss_main + w_S * loss_S + w_L * loss_L + w_A * loss_A
        return total, {
            "loss_total": total.item(),
            "loss_main": loss_main.item(),
            "loss_S": loss_S.item(),
            "loss_L": loss_L.item(),
            "loss_A": loss_A.item(),
        }


def count_parameters(model: nn.Module) -> dict:
    """Count parameters by module group."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    surface = sum(p.numel() for p in model.surface_enc.parameters() if p.requires_grad)
    lex = sum(p.numel() for p in model.lexical_enc.parameters() if p.requires_grad)
    ast = sum(p.numel() for p in model.ast_enc.parameters() if p.requires_grad)
    fusion = total - surface - lex - ast
    return {
        "total": total,
        "surface": surface,
        "lexical": lex,
        "ast": ast,
        "fusion_and_heads": fusion,
    }


if __name__ == "__main__":
    print("Building ThreeViewModel...")
    model = ThreeViewModel()
    params = count_parameters(model)
    print(f"\nParameter counts:")
    for k, v in params.items():
        print(f"  {k:25s} {v/1e6:>7.2f} M")

    # Dummy forward
    print("\nDummy forward (batch=4):")
    B = 4
    surface_ids = torch.randint(2, 50265, (B, 256))
    surface_ids[:, 0] = 0  # CLS
    surface_mask = torch.ones(B, 256, dtype=torch.bool)
    lex_ids = torch.randint(0, 24, (B, 64))
    lex_ids[:, 0] = 2  # CLS
    lex_mask = torch.ones(B, 64, dtype=torch.bool)
    ast_ids = torch.randint(0, 94, (B, 128))
    ast_ids[:, 0] = 2  # CLS
    ast_mask = torch.ones(B, 128, dtype=torch.bool)
    ast_valid = torch.ones(B)

    model.eval()
    with torch.no_grad():
        out = model(surface_ids, surface_mask, lex_ids, lex_mask, ast_ids, ast_mask, ast_valid)

    print(f"  p_main shape:        {out['p_main'].shape}")
    print(f"  p_S shape:           {out['p_S'].shape}")
    print(f"  stage1_weights:      {out['stage1_weights'].shape}  (B, 2, 2 = L/A mutual attn)")
    print(f"  attn_weights:        {out['attn_weights'].shape}  (B, 2, T_S = abstract→surface attn)")
    print(f"  stage1_weights[0]:")
    s1 = out['stage1_weights'][0]
    print(f"    z_L attends [L, A]: {s1[0].tolist()}")
    print(f"    z_A attends [L, A]: {s1[1].tolist()}")

    # Loss test
    labels = torch.tensor([1, 0, 1, 0])
    model.train()
    out = model(surface_ids, surface_mask, lex_ids, lex_mask, ast_ids, ast_mask,
                ast_valid, view_dropout_prob=0.1)
    loss, components = model.compute_loss(out, labels)
    print(f"\n  loss: {loss.item():.4f}")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")

    # Backward test
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    print(f"  total grad norm: {grad_norm:.4f}")
