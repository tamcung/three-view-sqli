#!/usr/bin/env python3
"""Single-view ablations of the proposed three-view fusion model,
plus the model factory used by the trainer.

Single-view ablations (one for each of the three views in 本文方法):
  - SurfaceOnlyModel    : BPE-only (matches surface_enc)
  - LexicalOnlyModel    : Lex-only (matches lex_enc)
  - CharTransformerModel: Char-only (matches char_enc inside ThreeViewFusionModel)

Char-level baselines (CharCNN / CharLSTM / CharGRU) live in baseline_models.py.
ThreeViewFusionModel (本文方法) lives in baseline_models.py.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .model import TransformerViewEncoder
except ImportError:
    from model import TransformerViewEncoder


class _SingleViewModel(nn.Module):
    def __init__(self, encoder: TransformerViewEncoder, d_in: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _forward_one(self, ids, mask) -> torch.Tensor:
        return self.encoder(ids, mask)["pooled"]

    def compute_loss(self, output, labels, weights=None, pos_weight=None):
        labels = labels.float()
        loss = F.binary_cross_entropy_with_logits(
            output["p_main"], labels, pos_weight=pos_weight,
        )
        return loss, {
            "loss_total": loss.item(), "loss_main": loss.item(),
            "loss_S": 0.0, "loss_L": 0.0, "loss_A": 0.0,
        }


class SurfaceOnlyModel(_SingleViewModel):
    def __init__(
        self,
        surface_vocab_size: int = 50265,
        surface_max_len: int = 513,
        surface_pad_id: int = 1,
        d_surface: int = 384,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        **_ignored,
    ):
        enc = TransformerViewEncoder(
            vocab_size=surface_vocab_size, max_len=surface_max_len,
            d_model=d_surface, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_surface * 4, dropout=dropout, pad_id=surface_pad_id,
        )
        super().__init__(enc, d_surface, dropout)

    def forward(self, surface_ids, surface_mask, lex_ids=None, lex_mask=None,
                char_ids=None, char_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        z = self._forward_one(surface_ids, surface_mask)
        p = self.classifier(z).squeeze(-1)
        zero = torch.zeros_like(p)
        return {"p_main": p, "p_S": p, "p_L": zero, "p_A": zero,
                "z_S": z, "z_L": zero, "z_A": zero,
                "z_LA": zero, "z_final": zero}


class LexicalOnlyModel(_SingleViewModel):
    def __init__(
        self,
        lex_vocab_size: int = 365,
        lex_max_len: int = 129,
        d_abstract: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        **_ignored,
    ):
        enc = TransformerViewEncoder(
            vocab_size=lex_vocab_size, max_len=lex_max_len,
            d_model=d_abstract, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_abstract * 4, dropout=dropout, pad_id=0,
        )
        super().__init__(enc, d_abstract, dropout)

    def forward(self, surface_ids=None, surface_mask=None, lex_ids=None, lex_mask=None,
                char_ids=None, char_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        z = self._forward_one(lex_ids, lex_mask)
        p = self.classifier(z).squeeze(-1)
        zero = torch.zeros_like(p)
        return {"p_main": p, "p_S": zero, "p_L": p, "p_A": zero,
                "z_S": zero, "z_L": z, "z_A": zero,
                "z_LA": zero, "z_final": zero}


# ============================================================
# Char-only single-view ablation (matches char_enc inside ThreeViewFusionModel)
# ============================================================
class CharTransformerModel(nn.Module):
    def __init__(
        self,
        char_vocab_size: int = 257,
        char_max_len: int = 513,
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.embed = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(char_max_len, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, char_ids=None, char_mask=None,
                surface_ids=None, surface_mask=None,
                lex_ids=None, lex_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        B, T = char_ids.shape
        pos = torch.arange(T, device=char_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(char_ids) + self.pos_emb(pos)
        key_padding_mask = ~char_mask.bool() if char_mask is not None else None
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        if char_mask is not None:
            m = char_mask.float().unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            pooled = h.mean(dim=1)
        p = self.classifier(pooled).squeeze(-1)
        zero = torch.zeros_like(p)
        return {"p_main": p, "p_S": p, "p_L": zero, "p_A": zero,
                "z_S": pooled, "z_L": zero, "z_A": zero,
                "z_LA": zero, "z_final": zero}

    def compute_loss(self, output, labels, weights=None, pos_weight=None):
        labels = labels.float()
        loss = F.binary_cross_entropy_with_logits(
            output["p_main"], labels, pos_weight=pos_weight,
        )
        return loss, {"loss_total": loss.item(), "loss_main": loss.item(),
                        "loss_S": 0.0, "loss_L": 0.0, "loss_A": 0.0}


# ============================================================
# Char + Lex two-view fusion (leave-one-out ablation: no BPE)
# ============================================================
class CharLexFusionModel(nn.Module):
    """Two-view fusion of Char + Lex (drops the BPE/surface view).

    Architecture mirrors ThreeViewFusionModel but with only 2 views:
    Char-Transformer + Lex-Transformer joined by view-type embedding,
    then a single full self-attention fusion encoder.
    """

    def __init__(
        self,
        char_vocab_size: int = 257,
        char_max_len: int = 257,
        lex_vocab_size: int = 365,
        lex_max_len: int = 129,
        d_fusion: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        fusion_layers: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()

        self.char_enc = TransformerViewEncoder(
            vocab_size=char_vocab_size, max_len=char_max_len,
            d_model=d_fusion, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_fusion * 4, dropout=dropout, pad_id=0,
        )

        self.lex_enc = TransformerViewEncoder(
            vocab_size=lex_vocab_size, max_len=lex_max_len,
            d_model=d_fusion, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_fusion * 4, dropout=dropout, pad_id=0,
        )

        self.view_emb = nn.Embedding(2, d_fusion)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_fusion, nhead=n_heads,
            dim_feedforward=d_fusion * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)

        self.aux_L = nn.Linear(d_fusion, 1)
        self.aux_C = nn.Linear(d_fusion, 1)

        self.classifier = nn.Sequential(
            nn.Linear(d_fusion, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _masked_mean(seq, mask):
        m = mask.unsqueeze(-1).float()
        s = (seq * m).sum(dim=1)
        n = m.sum(dim=1).clamp(min=1)
        return s / n

    def forward(self, surface_ids=None, surface_mask=None,
                lex_ids=None, lex_mask=None,
                char_ids=None, char_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0,
                surface_inputs_embeds=None):
        if char_mask is None:
            char_mask = (char_ids != 0).long()
        c_out = self.char_enc(char_ids, char_mask)
        z_C = c_out["pooled"]
        H_C = c_out["full"]

        l_out = self.lex_enc(lex_ids, lex_mask)
        z_L = l_out["pooled"]
        H_L = l_out["full"]

        p_L = self.aux_L(z_L).squeeze(-1)
        p_C = self.aux_C(z_C).squeeze(-1)

        dev = H_L.device
        B = H_L.size(0)
        v_L = self.view_emb(torch.tensor(0, device=dev))
        v_C = self.view_emb(torch.tensor(1, device=dev))
        H_L_t = H_L + v_L
        H_C_t = H_C + v_C

        if self.training and view_dropout_prob > 0:
            keep_L = (torch.rand(B, device=dev) > view_dropout_prob).float().view(B, 1, 1)
            keep_C = (torch.rand(B, device=dev) > view_dropout_prob).float().view(B, 1, 1)
            H_L_t = H_L_t * keep_L
            H_C_t = H_C_t * keep_C

        if lex_mask is None:
            lex_mask = (lex_ids != 0).long()

        fused = torch.cat([H_L_t, H_C_t], dim=1)
        joint_mask = torch.cat([lex_mask, char_mask], dim=1)
        joint_pad = ~joint_mask.bool()

        out = self.fusion_encoder(fused, src_key_padding_mask=joint_pad)
        z_final = self._masked_mean(out, joint_mask)
        p_main = self.classifier(z_final).squeeze(-1)

        zero = torch.zeros_like(p_main)
        return {
            "p_main": p_main,
            "p_S": zero, "p_L": p_L, "p_A": p_C,
            "z_S": zero, "z_L": z_L, "z_A": z_C,
            "z_LA": z_final, "z_final": z_final,
        }

    def compute_loss(self, output, labels,
                       weights=(0.7, 0.0, 0.15, 0.15), pos_weight=None):
        labels = labels.float()
        w_main, w_S, w_L, w_C = weights
        def bce(logits):
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)
        loss_main = bce(output["p_main"])
        loss_L = bce(output["p_L"])
        loss_C = bce(output["p_A"])
        total = w_main * loss_main + w_L * loss_L + w_C * loss_C
        return total, {
            "loss_total": total.item(),
            "loss_main": loss_main.item(),
            "loss_S": 0.0,
            "loss_L": loss_L.item(),
            "loss_A": loss_C.item(),
        }


# ============================================================
# BPE + Lex two-view fusion (leave-one-out: no Char)
# ============================================================
class BPELexFusionModel(nn.Module):
    def __init__(
        self,
        surface_vocab_size: int = 50265,
        surface_max_len: int = 257,
        surface_pad_id: int = 1,
        d_surface: int = 384,
        lex_vocab_size: int = 365,
        lex_max_len: int = 129,
        d_fusion: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        fusion_layers: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.surface_enc = TransformerViewEncoder(
            vocab_size=surface_vocab_size, max_len=surface_max_len,
            d_model=d_surface, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_surface * 4, dropout=dropout, pad_id=surface_pad_id,
        )
        self.surface_proj = nn.Linear(d_surface, d_fusion)

        self.lex_enc = TransformerViewEncoder(
            vocab_size=lex_vocab_size, max_len=lex_max_len,
            d_model=d_fusion, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_fusion * 4, dropout=dropout, pad_id=0,
        )

        self.view_emb = nn.Embedding(2, d_fusion)

        layer = nn.TransformerEncoderLayer(
            d_model=d_fusion, nhead=n_heads,
            dim_feedforward=d_fusion * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(layer, num_layers=fusion_layers)

        self.aux_S = nn.Linear(d_surface, 1)
        self.aux_L = nn.Linear(d_fusion, 1)

        self.classifier = nn.Sequential(
            nn.Linear(d_fusion, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _masked_mean(seq, mask):
        m = mask.unsqueeze(-1).float()
        return (seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward(self, surface_ids=None, surface_mask=None,
                lex_ids=None, lex_mask=None,
                char_ids=None, char_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0,
                surface_inputs_embeds=None):
        if surface_inputs_embeds is not None:
            s_out = self.surface_enc(input_ids=None, attention_mask=surface_mask,
                                      inputs_embeds=surface_inputs_embeds)
        else:
            s_out = self.surface_enc(surface_ids, surface_mask)
        z_S = s_out["pooled"]
        H_S = self.surface_proj(s_out["full"])

        l_out = self.lex_enc(lex_ids, lex_mask)
        z_L = l_out["pooled"]
        H_L = l_out["full"]

        p_S = self.aux_S(z_S).squeeze(-1)
        p_L = self.aux_L(z_L).squeeze(-1)

        dev = H_S.device
        B = H_S.size(0)
        v_S = self.view_emb(torch.tensor(0, device=dev))
        v_L = self.view_emb(torch.tensor(1, device=dev))
        H_S_t = H_S + v_S
        H_L_t = H_L + v_L

        if self.training and view_dropout_prob > 0:
            keep_S = (torch.rand(B, device=dev) > view_dropout_prob).float().view(B, 1, 1)
            keep_L = (torch.rand(B, device=dev) > view_dropout_prob).float().view(B, 1, 1)
            H_S_t = H_S_t * keep_S
            H_L_t = H_L_t * keep_L

        if lex_mask is None:
            lex_mask = (lex_ids != 0).long()

        fused = torch.cat([H_S_t, H_L_t], dim=1)
        joint_mask = torch.cat([surface_mask, lex_mask], dim=1)
        out = self.fusion_encoder(fused, src_key_padding_mask=~joint_mask.bool())
        z_final = self._masked_mean(out, joint_mask)
        p_main = self.classifier(z_final).squeeze(-1)

        zero = torch.zeros_like(p_main)
        return {"p_main": p_main, "p_S": p_S, "p_L": p_L, "p_A": zero,
                "z_S": z_S, "z_L": z_L, "z_A": zero,
                "z_LA": z_final, "z_final": z_final}

    def compute_loss(self, output, labels,
                       weights=(0.7, 0.15, 0.15, 0.0), pos_weight=None):
        labels = labels.float()
        w_main, w_S, w_L, w_C = weights
        def bce(logits):
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)
        loss_main = bce(output["p_main"])
        loss_S = bce(output["p_S"])
        loss_L = bce(output["p_L"])
        total = w_main * loss_main + w_S * loss_S + w_L * loss_L
        return total, {
            "loss_total": total.item(), "loss_main": loss_main.item(),
            "loss_S": loss_S.item(), "loss_L": loss_L.item(), "loss_A": 0.0,
        }


# ============================================================
# BPE + Char two-view fusion (leave-one-out: no Lex)
# ============================================================
class BPECharFusionModel(nn.Module):
    def __init__(
        self,
        surface_vocab_size: int = 50265,
        surface_max_len: int = 257,
        surface_pad_id: int = 1,
        d_surface: int = 384,
        char_vocab_size: int = 257,
        char_max_len: int = 257,
        d_fusion: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        fusion_layers: int = 2,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.surface_enc = TransformerViewEncoder(
            vocab_size=surface_vocab_size, max_len=surface_max_len,
            d_model=d_surface, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_surface * 4, dropout=dropout, pad_id=surface_pad_id,
        )
        self.surface_proj = nn.Linear(d_surface, d_fusion)

        self.char_enc = TransformerViewEncoder(
            vocab_size=char_vocab_size, max_len=char_max_len,
            d_model=d_fusion, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_fusion * 4, dropout=dropout, pad_id=0,
        )

        self.view_emb = nn.Embedding(2, d_fusion)

        layer = nn.TransformerEncoderLayer(
            d_model=d_fusion, nhead=n_heads,
            dim_feedforward=d_fusion * 4, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(layer, num_layers=fusion_layers)

        self.aux_S = nn.Linear(d_surface, 1)
        self.aux_C = nn.Linear(d_fusion, 1)

        self.classifier = nn.Sequential(
            nn.Linear(d_fusion, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def _masked_mean(seq, mask):
        m = mask.unsqueeze(-1).float()
        return (seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward(self, surface_ids=None, surface_mask=None,
                lex_ids=None, lex_mask=None,
                char_ids=None, char_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0,
                surface_inputs_embeds=None):
        if surface_inputs_embeds is not None:
            s_out = self.surface_enc(input_ids=None, attention_mask=surface_mask,
                                      inputs_embeds=surface_inputs_embeds)
        else:
            s_out = self.surface_enc(surface_ids, surface_mask)
        z_S = s_out["pooled"]
        H_S = self.surface_proj(s_out["full"])

        if char_mask is None:
            char_mask = (char_ids != 0).long()
        c_out = self.char_enc(char_ids, char_mask)
        z_C = c_out["pooled"]
        H_C = c_out["full"]

        p_S = self.aux_S(z_S).squeeze(-1)
        p_C = self.aux_C(z_C).squeeze(-1)

        dev = H_S.device
        B = H_S.size(0)
        v_S = self.view_emb(torch.tensor(0, device=dev))
        v_C = self.view_emb(torch.tensor(1, device=dev))
        H_S_t = H_S + v_S
        H_C_t = H_C + v_C

        if self.training and view_dropout_prob > 0:
            keep_S = (torch.rand(B, device=dev) > view_dropout_prob).float().view(B, 1, 1)
            keep_C = (torch.rand(B, device=dev) > view_dropout_prob).float().view(B, 1, 1)
            H_S_t = H_S_t * keep_S
            H_C_t = H_C_t * keep_C

        fused = torch.cat([H_S_t, H_C_t], dim=1)
        joint_mask = torch.cat([surface_mask, char_mask], dim=1)
        out = self.fusion_encoder(fused, src_key_padding_mask=~joint_mask.bool())
        z_final = self._masked_mean(out, joint_mask)
        p_main = self.classifier(z_final).squeeze(-1)

        zero = torch.zeros_like(p_main)
        return {"p_main": p_main, "p_S": p_S, "p_L": zero, "p_A": p_C,
                "z_S": z_S, "z_L": zero, "z_A": z_C,
                "z_LA": z_final, "z_final": z_final}

    def compute_loss(self, output, labels,
                       weights=(0.7, 0.15, 0.0, 0.15), pos_weight=None):
        labels = labels.float()
        w_main, w_S, w_L, w_C = weights
        def bce(logits):
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)
        loss_main = bce(output["p_main"])
        loss_S = bce(output["p_S"])
        loss_C = bce(output["p_A"])
        total = w_main * loss_main + w_S * loss_S + w_C * loss_C
        return total, {
            "loss_total": total.item(), "loss_main": loss_main.item(),
            "loss_S": loss_S.item(), "loss_L": 0.0, "loss_A": loss_C.item(),
        }


# ============================================================
# Variant factory
# ============================================================
def build_model(variant: str, model_kwargs: dict) -> nn.Module:
    """Instantiate a model variant by name."""
    variant = variant.lower()

    # Three-view fusion (本文方法) — accepts several aliases
    if variant in ("three_view", "tri_view", "fusion", "default", ""):
        try:
            from .baseline_models import ThreeViewFusionModel
        except ImportError:
            from baseline_models import ThreeViewFusionModel
        return ThreeViewFusionModel(**model_kwargs)

    # Single-view ablations
    if variant in ("surface_only", "bpe_only"):
        return SurfaceOnlyModel(**model_kwargs)
    if variant in ("lexical_only", "lex_only"):
        return LexicalOnlyModel(**model_kwargs)

    # Char-* baselines
    if variant in ("char_cnn", "charcnn"):
        try:
            from .baseline_models import CharCNNModel
        except ImportError:
            from baseline_models import CharCNNModel
        return CharCNNModel(**model_kwargs)
    if variant in ("char_lstm", "charlstm"):
        try:
            from .baseline_models import CharLSTMModel
        except ImportError:
            from baseline_models import CharLSTMModel
        return CharLSTMModel(**model_kwargs)
    if variant in ("char_gru", "chargru"):
        try:
            from .baseline_models import CharGRUModel
        except ImportError:
            from baseline_models import CharGRUModel
        return CharGRUModel(**model_kwargs)
    if variant in ("char_transformer", "chartransformer"):
        return CharTransformerModel(**model_kwargs)
    if variant in ("mvc_bicnn", "mvcbicnn"):
        try:
            from .baseline_models import MVCBiCNNModel
        except ImportError:
            from baseline_models import MVCBiCNNModel
        return MVCBiCNNModel(**model_kwargs)

    # Two-view leave-one-out ablations
    if variant in ("char_lex_fusion", "no_bpe"):
        return CharLexFusionModel(**model_kwargs)
    if variant in ("bpe_lex_fusion", "no_char"):
        return BPELexFusionModel(**model_kwargs)
    if variant in ("bpe_char_fusion", "no_lex"):
        return BPECharFusionModel(**model_kwargs)

    raise ValueError(f"Unknown model_variant: {variant}")
