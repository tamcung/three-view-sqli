#!/usr/bin/env python3
"""Baseline models from Hu Xiuwen's thesis (东南大学 2024):

  - SequenceLSTMModel: BPE-tokenized user_input → Embedding → LSTM → MLP head
                       (matches Hu's "LSTM" baseline: F1=0.9114 on his data)
  - TreeLSTMModel:     parsed AST → Child-Sum Tree-LSTM → MLP head
                       (matches Hu's "AST-LSTM" / Tree-LSTM: F1=0.9971 on his data)

Both produce the same output schema as ThreeViewModel.forward() so the trainer
can swap them in via `model_variant: sequence_lstm | tree_lstm`. For a single-
view baseline we set the unused aux logits to zero (and aux loss weight 0).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


# ============================================================
# 1. Sequence LSTM on BPE tokens of user_input
# ============================================================
class SequenceLSTMModel(nn.Module):
    """LSTM applied to surface BPE token IDs (user_input level).

    This is Hu's "LSTM" baseline (Section 3, Table 3.6). It uses only the
    surface BPE view; lexical and AST views are ignored.
    """

    def __init__(
        self,
        surface_vocab_size: int = 50265,
        surface_pad_id: int = 1,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
        **_ignored,
    ):
        super().__init__()
        self.embed = nn.Embedding(surface_vocab_size, embed_dim,
                                    padding_idx=surface_pad_id)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, surface_ids, surface_mask, lex_ids=None, lex_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        x = self.embed(surface_ids)                         # [B, T, D]
        lengths = surface_mask.sum(dim=1).clamp(min=1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True,
                                          enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        # h: [num_layers * num_directions, B, hidden]
        if self.lstm.bidirectional:
            # concat forward & backward of last layer
            h_last = torch.cat([h[-2], h[-1]], dim=-1)      # [B, 2*hidden]
        else:
            h_last = h[-1]                                    # [B, hidden]
        p = self.classifier(h_last).squeeze(-1)
        zero = torch.zeros_like(p)
        return {"p_main": p, "p_S": p, "p_L": zero, "p_A": zero,
                "z_S": h_last, "z_L": zero, "z_A": zero,
                "z_LA": zero, "z_final": zero}

    def compute_loss(self, output, labels, weights=None, pos_weight=None):
        labels = labels.float()
        loss = F.binary_cross_entropy_with_logits(
            output["p_main"], labels, pos_weight=pos_weight,
        )
        return loss, {"loss_total": loss.item(), "loss_main": loss.item(),
                        "loss_S": 0.0, "loss_L": 0.0, "loss_A": 0.0}


# ============================================================
# 1b. Char-level CNN on raw bytes of user_input
# ============================================================
class CharCNNModel(nn.Module):
    """Char-level CNN baseline (Kim 2014 adapted).

    Each utf-8 byte of user_input is mapped to id ∈ [1,256]; PAD=0.
    Embedding (256+1, embed_dim) → parallel Conv1d with multiple kernel
    sizes → ReLU → global max-pool → concat → FC head.
    """

    def __init__(
        self,
        char_vocab_size: int = 257,        # 256 byte values + PAD(0)
        embed_dim: int = 64,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        num_filters: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        **_ignored,
    ):
        super().__init__()
        # Allow YAML to pass list instead of tuple
        if isinstance(kernel_sizes, list):
            kernel_sizes = tuple(kernel_sizes)
        self.embed = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        feat_dim = num_filters * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, char_ids=None, char_mask=None,
                surface_ids=None, surface_mask=None,
                lex_ids=None, lex_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        # char_ids: [B, T]
        x = self.embed(char_ids)             # [B, T, D]
        x = x.transpose(1, 2)                # [B, D, T]
        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))              # [B, F, T]
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)  # [B, F]
            feats.append(h)
        z = torch.cat(feats, dim=-1)         # [B, F * n_kernels]
        z = self.dropout(z)
        p = self.classifier(z).squeeze(-1)
        zero = torch.zeros_like(p)
        return {"p_main": p, "p_S": p, "p_L": zero, "p_A": zero,
                "z_S": z, "z_L": zero, "z_A": zero,
                "z_LA": zero, "z_final": zero}

    def compute_loss(self, output, labels, weights=None, pos_weight=None):
        labels = labels.float()
        loss = F.binary_cross_entropy_with_logits(
            output["p_main"], labels, pos_weight=pos_weight,
        )
        return loss, {"loss_total": loss.item(), "loss_main": loss.item(),
                        "loss_S": 0.0, "loss_L": 0.0, "loss_A": 0.0}


# ============================================================
# 1b'. Char-level BiLSTM (single-view byte sequence)
# ============================================================
class CharBiLSTMModel(nn.Module):
    """Char-level Bidirectional LSTM baseline.

    Each utf-8 byte → embedding → BiLSTM → concat(forward_last, backward_last)
    → MLP head. Counterpart to CharCNN (kernel-based) with sequential
    recurrence instead of local conv windows.
    """

    def __init__(
        self,
        char_vocab_size: int = 257,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.embed = nn.Embedding(char_vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, char_ids=None, char_mask=None,
                surface_ids=None, surface_mask=None,
                lex_ids=None, lex_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        x = self.embed(char_ids)
        if char_mask is not None:
            lengths = char_mask.sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True,
                                              enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
        else:
            _, (h, _) = self.lstm(x)
        # Final layer: concat forward + backward
        h_last = torch.cat([h[-2], h[-1]], dim=-1)
        p = self.classifier(h_last).squeeze(-1)
        zero = torch.zeros_like(p)
        return {"p_main": p, "p_S": p, "p_L": zero, "p_A": zero,
                "z_S": h_last, "z_L": zero, "z_A": zero,
                "z_LA": zero, "z_final": zero}

    def compute_loss(self, output, labels, weights=None, pos_weight=None):
        labels = labels.float()
        loss = F.binary_cross_entropy_with_logits(
            output["p_main"], labels, pos_weight=pos_weight,
        )
        return loss, {"loss_total": loss.item(), "loss_main": loss.item(),
                        "loss_S": 0.0, "loss_L": 0.0, "loss_A": 0.0}


# ============================================================
# 1c. Char-CNN surface + Transformer lexical (two-view)
# ============================================================
class CharLexModel(nn.Module):
    """Two-view fusion: CharCNN on raw bytes (surface) + Transformer on
    libinjection token types (lexical). No AST view.

    Designed to combine CharCNN's robustness to byte-level noise (V5-style
    binary/octal literals, weird whitespace) with the lexical Transformer's
    natural handling of standard WAF encoding tampers (URL/HTML/hex
    entities) — both via the libinjection tokenizer's normalization.
    """

    def __init__(
        self,
        # Char surface
        char_vocab_size: int = 257,
        char_embed_dim: int = 64,
        char_kernel_sizes: tuple = (3, 5, 7),
        char_num_filters: int = 128,
        # Lex Transformer
        lex_vocab_size: int = 24,
        lex_max_len: int = 129,
        lex_d_model: int = 256,
        lex_n_layers: int = 4,
        lex_n_heads: int = 4,
        # Head
        hidden_dim: int = 128,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        if isinstance(char_kernel_sizes, list):
            char_kernel_sizes = tuple(char_kernel_sizes)

        # Surface (char CNN)
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.char_convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, char_num_filters, kernel_size=k, padding=k // 2)
            for k in char_kernel_sizes
        ])
        char_feat_dim = char_num_filters * len(char_kernel_sizes)

        # Lexical (4-layer Transformer encoder)
        try:
            from .model import TransformerViewEncoder
        except ImportError:
            from model import TransformerViewEncoder
        self.lex_enc = TransformerViewEncoder(
            vocab_size=lex_vocab_size, max_len=lex_max_len,
            d_model=lex_d_model, n_layers=lex_n_layers, n_heads=lex_n_heads,
            d_ff=lex_d_model * 4, dropout=dropout, pad_id=0,
        )

        # Fusion + classifier
        feat_dim = char_feat_dim + lex_d_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Aux deep supervision
        self.aux_S = nn.Linear(char_feat_dim, 1)
        self.aux_L = nn.Linear(lex_d_model, 1)

    def _encode_chars(self, char_ids):
        x = self.char_embed(char_ids)            # [B, T, D]
        x = x.transpose(1, 2)                    # [B, D, T]
        feats = []
        for conv in self.char_convs:
            h = F.relu(conv(x))                  # [B, F, T]
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)  # [B, F]
            feats.append(h)
        return torch.cat(feats, dim=-1)          # [B, F * n_kernels]

    def forward(self, char_ids=None, char_mask=None,
                lex_ids=None, lex_mask=None,
                surface_ids=None, surface_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        z_S = self._encode_chars(char_ids)        # [B, F*n]
        z_L = self.lex_enc(lex_ids, lex_mask)["pooled"]  # [B, d_lex]

        # Aux logits computed BEFORE view dropout (so they reflect each view's
        # own discriminative ability)
        p_S = self.aux_S(z_S).squeeze(-1)
        p_L = self.aux_L(z_L).squeeze(-1)

        # View dropout during training
        if self.training and view_dropout_prob > 0:
            B = z_S.size(0)
            dev = z_S.device
            keep_S = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            keep_L = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            z_S_eff = z_S * keep_S
            z_L_eff = z_L * keep_L
        else:
            z_S_eff = z_S
            z_L_eff = z_L

        z = torch.cat([z_S_eff, z_L_eff], dim=-1)
        z = self.dropout(z)
        p_main = self.classifier(z).squeeze(-1)

        zero = torch.zeros_like(p_main)
        return {"p_main": p_main, "p_S": p_S, "p_L": p_L, "p_A": zero,
                "z_S": z_S, "z_L": z_L, "z_A": zero,
                "z_LA": z, "z_final": z}

    def compute_loss(self, output, labels,
                       weights=(0.7, 0.15, 0.15, 0.0), pos_weight=None):
        labels = labels.float()
        w_main, w_S, w_L, _ = weights
        def bce(logits):
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)
        loss_main = bce(output["p_main"])
        loss_S = bce(output["p_S"])
        loss_L = bce(output["p_L"])
        total = w_main * loss_main + w_S * loss_S + w_L * loss_L
        return total, {
            "loss_total": total.item(),
            "loss_main": loss_main.item(),
            "loss_S": loss_S.item(),
            "loss_L": loss_L.item(),
            "loss_A": 0.0,
        }


# ============================================================
# 1d. CharCNN surface + Lex Transformer with cross-attention fusion
# ============================================================
class CharLexCrossAttnModel(nn.Module):
    """Two-view with cross-attention fusion.

    CharCNN keeps the per-position feature map [B, T, d] (no global pool yet).
    Lex Transformer outputs a single pooled CLS vector z_L.
    Cross-attention: Q = z_L (as a length-1 sequence), K, V = char features.

    The lex CLS thus learns "which positions in the byte stream provide
    structural evidence for what I observed in the libinjection token
    sequence?". Output is concat([z_L, attended z_L]) → MLP.

    This mirrors the original ThreeViewModel's Stage 2 cross-attention but
    with CharCNN replacing the surface Transformer and AST view dropped.
    """

    def __init__(
        self,
        # Char surface
        char_vocab_size: int = 257,
        char_embed_dim: int = 64,
        char_kernel_sizes: tuple = (3, 5, 7),
        char_num_filters: int = 128,
        # Lex Transformer
        lex_vocab_size: int = 24,
        lex_max_len: int = 129,
        # Fusion / shared
        d_fusion: int = 256,
        n_heads: int = 4,
        n_layers_lex: int = 4,
        # Head
        hidden_dim: int = 128,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        if isinstance(char_kernel_sizes, list):
            char_kernel_sizes = tuple(char_kernel_sizes)

        # Surface: CharCNN
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.char_convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, char_num_filters,
                       kernel_size=k, padding=k // 2)
            for k in char_kernel_sizes
        ])
        char_feat_dim = char_num_filters * len(char_kernel_sizes)
        # Project per-position char features → d_fusion
        self.char_proj = nn.Linear(char_feat_dim, d_fusion)

        # Lex Transformer (output already in d_fusion)
        try:
            from .model import TransformerViewEncoder
        except ImportError:
            from model import TransformerViewEncoder
        self.lex_enc = TransformerViewEncoder(
            vocab_size=lex_vocab_size, max_len=lex_max_len,
            d_model=d_fusion, n_layers=n_layers_lex, n_heads=n_heads,
            d_ff=d_fusion * 4, dropout=dropout, pad_id=0,
        )

        # Cross-attention block (lex CLS → attends over char sequence)
        self.cross_norm_q = nn.LayerNorm(d_fusion)
        self.cross_norm_kv = nn.LayerNorm(d_fusion)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_fusion, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_norm_ffn = nn.LayerNorm(d_fusion)
        self.cross_ffn = nn.Sequential(
            nn.Linear(d_fusion, d_fusion * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion * 4, d_fusion),
        )

        # Aux supervision per view (deep supervision)
        self.aux_S = nn.Linear(char_feat_dim, 1)   # raw char pooled feat
        self.aux_L = nn.Linear(d_fusion, 1)        # lex CLS

        # Main classifier: concat([z_L, attended_z])
        self.classifier = nn.Sequential(
            nn.Linear(d_fusion * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_char_sequence(self, char_ids):
        """Returns (per_position [B, T, d_fusion], pooled [B, char_feat_dim])."""
        x = self.char_embed(char_ids)              # [B, T, D]
        x = x.transpose(1, 2)                      # [B, D, T]
        per_kernel = []
        for conv in self.char_convs:
            h = F.relu(conv(x))                    # [B, F, T]
            per_kernel.append(h)
        # Concat along channel dim — same T across kernels
        H = torch.cat(per_kernel, dim=1)           # [B, F*n, T]
        H_seq = H.transpose(1, 2)                  # [B, T, F*n]
        H_seq = self.char_proj(H_seq)              # [B, T, d_fusion]
        # Pooled char vector for aux loss
        z_S_pool = F.adaptive_max_pool1d(H, 1).squeeze(-1)  # [B, F*n]
        return H_seq, z_S_pool

    def forward(self, char_ids=None, char_mask=None,
                lex_ids=None, lex_mask=None,
                surface_ids=None, surface_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        H_char, z_S_pool = self._encode_char_sequence(char_ids)  # [B, T, d], [B, F*n]
        z_L = self.lex_enc(lex_ids, lex_mask)["pooled"]          # [B, d]

        p_S_aux = self.aux_S(z_S_pool).squeeze(-1)
        p_L_aux = self.aux_L(z_L).squeeze(-1)

        # Optional view dropout — zero out a view's input to fusion
        if self.training and view_dropout_prob > 0:
            B = z_L.size(0)
            dev = z_L.device
            keep_S = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            keep_L = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            H_char_eff = H_char * keep_S.unsqueeze(1)
            z_L_eff = z_L * keep_L
        else:
            H_char_eff = H_char
            z_L_eff = z_L

        # Cross-attention: lex CLS as length-1 query, char sequence as KV
        q = self.cross_norm_q(z_L_eff.unsqueeze(1))               # [B, 1, d]
        kv = self.cross_norm_kv(H_char_eff)                       # [B, T, d]

        # Build key padding mask if char_mask is available
        if char_mask is not None:
            kv_pad_mask = ~char_mask.bool()                       # [B, T]
        else:
            kv_pad_mask = None

        attn_out, attn_weights = self.cross_attn(
            query=q, key=kv, value=kv,
            key_padding_mask=kv_pad_mask, need_weights=True,
        )
        z_attn = z_L_eff.unsqueeze(1) + attn_out                  # residual: [B, 1, d]
        ffn_out = self.cross_ffn(self.cross_norm_ffn(z_attn))
        z_attn = (z_attn + ffn_out).squeeze(1)                    # [B, d]

        # Concat: [original lex CLS ; cross-attn enriched]
        cls_input = torch.cat([z_L_eff, z_attn], dim=-1)          # [B, 2d]
        p_main = self.classifier(cls_input).squeeze(-1)

        zero = torch.zeros_like(p_main)
        return {"p_main": p_main, "p_S": p_S_aux, "p_L": p_L_aux, "p_A": zero,
                "z_S": z_S_pool, "z_L": z_L, "z_A": zero,
                "z_LA": z_L, "z_final": z_attn,
                "attn_weights": attn_weights}

    def compute_loss(self, output, labels,
                       weights=(0.7, 0.15, 0.15, 0.0), pos_weight=None):
        labels = labels.float()
        w_main, w_S, w_L, _ = weights
        def bce(logits):
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)
        loss_main = bce(output["p_main"])
        loss_S = bce(output["p_S"])
        loss_L = bce(output["p_L"])
        total = w_main * loss_main + w_S * loss_S + w_L * loss_L
        return total, {
            "loss_total": total.item(),
            "loss_main": loss_main.item(),
            "loss_S": loss_S.item(),
            "loss_L": loss_L.item(),
            "loss_A": 0.0,
        }


# ============================================================
# 1e. Tri-view: BPE-Surface + Char-Surface + Lex-Abstract with dual cross-attn
# ============================================================
class _CrossAttnBlock(nn.Module):
    """Pre-LN cross-attention block: Q queries (K, V); residual + FFN."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, q, kv, kv_mask=None):
        """q: [B, 1, d]  kv: [B, T, d]  kv_mask: [B, T] (1=valid). Returns [B, d]."""
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        kv_pad = ~kv_mask.bool() if kv_mask is not None else None
        attn_out, _ = self.attn(q_n, kv_n, kv_n,
                                  key_padding_mask=kv_pad, need_weights=False)
        z = q + attn_out
        ff = self.ffn(self.norm_ffn(z))
        return (z + ff).squeeze(1)


class BPECharLexModel(nn.Module):
    """Three-view: BPE Transformer (sub-word surface) + CharCNN (byte surface)
    + Lex Transformer (abstract). Lex CLS queries each surface sequence
    independently via cross-attention, then all three are concatenated for
    classification.

    Inspired by the original ThreeViewModel's Stage-2 cross-attn, but with
    AST replaced by a CharCNN surface so that byte-level mutations (V5
    style) and word-level mutations (encoded entities) are both addressable.
    """

    def __init__(
        self,
        # BPE surface
        surface_vocab_size: int = 50265,
        surface_max_len: int = 257,
        surface_pad_id: int = 1,
        d_surface: int = 384,
        # CharCNN surface
        char_vocab_size: int = 257,
        char_embed_dim: int = 64,
        char_kernel_sizes: tuple = (3, 5, 7),
        char_num_filters: int = 128,
        # Lex abstract
        lex_vocab_size: int = 24,
        lex_max_len: int = 129,
        # Fusion / shared
        d_fusion: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        if isinstance(char_kernel_sizes, list):
            char_kernel_sizes = tuple(char_kernel_sizes)

        try:
            from .model import TransformerViewEncoder
        except ImportError:
            from model import TransformerViewEncoder

        # ----- BPE surface -----
        self.surface_enc = TransformerViewEncoder(
            vocab_size=surface_vocab_size, max_len=surface_max_len,
            d_model=d_surface, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_surface * 4, dropout=dropout, pad_id=surface_pad_id,
        )
        self.bpe_proj = nn.Linear(d_surface, d_fusion)

        # ----- CharCNN surface -----
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.char_convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, char_num_filters,
                       kernel_size=k, padding=k // 2)
            for k in char_kernel_sizes
        ])
        char_feat_dim = char_num_filters * len(char_kernel_sizes)
        self.char_proj = nn.Linear(char_feat_dim, d_fusion)

        # ----- Lex abstract -----
        self.lex_enc = TransformerViewEncoder(
            vocab_size=lex_vocab_size, max_len=lex_max_len,
            d_model=d_fusion, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_fusion * 4, dropout=dropout, pad_id=0,
        )

        # ----- Two cross-attn blocks (lex_CLS → bpe_seq, lex_CLS → char_seq) -----
        self.xattn_bpe = _CrossAttnBlock(d_fusion, n_heads, dropout)
        self.xattn_char = _CrossAttnBlock(d_fusion, n_heads, dropout)

        # ----- Aux heads (deep supervision) -----
        self.aux_S_bpe = nn.Linear(d_surface, 1)        # BPE CLS
        self.aux_S_char = nn.Linear(char_feat_dim, 1)   # char max-pool
        self.aux_L = nn.Linear(d_fusion, 1)             # lex CLS

        # ----- Main classifier -----
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(d_fusion * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_char_sequence(self, char_ids):
        x = self.char_embed(char_ids)               # [B, T, D]
        x = x.transpose(1, 2)                       # [B, D, T]
        per_kernel = [F.relu(conv(x)) for conv in self.char_convs]
        H = torch.cat(per_kernel, dim=1)            # [B, F*n, T]
        H_seq = H.transpose(1, 2)                   # [B, T, F*n]
        H_proj = self.char_proj(H_seq)              # [B, T, d_fusion]
        z_pool = F.adaptive_max_pool1d(H, 1).squeeze(-1)  # [B, F*n]
        return H_proj, z_pool

    def forward(self, surface_ids=None, surface_mask=None,
                lex_ids=None, lex_mask=None,
                char_ids=None, char_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        # ---- Encode each view ----
        s_out = self.surface_enc(surface_ids, surface_mask)
        z_bpe = s_out["pooled"]                              # [B, d_surface]
        H_bpe = self.bpe_proj(s_out["full"])                 # [B, T_b, d_fusion]

        H_char, z_char_pool = self._encode_char_sequence(char_ids)  # [B, T_c, d], [B, F*n]
        z_lex = self.lex_enc(lex_ids, lex_mask)["pooled"]    # [B, d_fusion]

        # ---- Aux predictions (before view dropout) ----
        p_S_bpe = self.aux_S_bpe(z_bpe).squeeze(-1)
        p_S_char = self.aux_S_char(z_char_pool).squeeze(-1)
        p_L = self.aux_L(z_lex).squeeze(-1)

        # ---- View dropout ----
        if self.training and view_dropout_prob > 0:
            B = z_lex.size(0)
            dev = z_lex.device
            keep_S_bpe = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            keep_S_char = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            keep_L = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            H_bpe_eff = H_bpe * keep_S_bpe.unsqueeze(1)
            H_char_eff = H_char * keep_S_char.unsqueeze(1)
            z_lex_eff = z_lex * keep_L
        else:
            H_bpe_eff = H_bpe
            H_char_eff = H_char
            z_lex_eff = z_lex

        # ---- Two cross-attn queries from lex CLS ----
        q = z_lex_eff.unsqueeze(1)                           # [B, 1, d]
        attn_bpe = self.xattn_bpe(q, H_bpe_eff,
                                    kv_mask=surface_mask)    # [B, d]
        attn_char = self.xattn_char(q, H_char_eff,
                                      kv_mask=char_mask)     # [B, d]

        # ---- Three-way concat ----
        cls_input = torch.cat([z_lex_eff, attn_char, attn_bpe], dim=-1)  # [B, 3d]
        cls_input = self.dropout(cls_input)
        p_main = self.classifier(cls_input).squeeze(-1)

        return {
            "p_main": p_main,
            "p_S": (p_S_bpe + p_S_char) / 2,   # combined surface aux for back-compat
            "p_S_bpe": p_S_bpe, "p_S_char": p_S_char,
            "p_L": p_L, "p_A": torch.zeros_like(p_main),
            "z_S": z_bpe, "z_L": z_lex,
            "z_A": torch.zeros_like(z_lex),
            "attn_bpe": attn_bpe, "attn_char": attn_char,
        }

    def compute_loss(self, output, labels,
                       weights=(0.7, 0.1, 0.1, 0.1), pos_weight=None):
        """Loss = w_main · L_main + w_bpe · L_bpe + w_char · L_char + w_L · L_L.
        Caller's `weights` tuple convention preserved as (main, S_bpe, S_char, L)."""
        labels = labels.float()
        w_main, w_bpe, w_char, w_L = weights
        def bce(logits):
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)
        loss_main = bce(output["p_main"])
        loss_bpe = bce(output["p_S_bpe"])
        loss_char = bce(output["p_S_char"])
        loss_L = bce(output["p_L"])
        total = (w_main * loss_main + w_bpe * loss_bpe
                  + w_char * loss_char + w_L * loss_L)
        return total, {
            "loss_total": total.item(),
            "loss_main": loss_main.item(),
            "loss_S": (loss_bpe.item() + loss_char.item()) / 2,
            "loss_L": loss_L.item(),
            "loss_A": 0.0,
        }


# ============================================================
# 1f. Tri-view with original Stage1+Stage2 structure (CharCNN-pool replaces AST)
# ============================================================
class BPECharLexStageModel(nn.Module):
    """Tri-view: BPE Transformer (surface), CharCNN (abstract via global
    pool), Lex Transformer (abstract). Architecture mirrors the original
    ThreeViewModel: Stage 1 self-attn over the two abstract pooled vectors,
    then Stage 2 cross-attn over the BPE full sequence.

    Compared to BPECharLexModel (dual cross-attn), this preserves the
    abstract-view interaction step that proved useful in the original
    no_ast ablation, with CharCNN's pooled feature substituting for the
    AST Transformer's CLS output.
    """

    def __init__(
        self,
        # BPE surface
        surface_vocab_size: int = 50265,
        surface_max_len: int = 257,
        surface_pad_id: int = 1,
        d_surface: int = 384,
        # CharCNN (used via pooled feature, not per-position)
        char_vocab_size: int = 257,
        char_embed_dim: int = 64,
        char_kernel_sizes: tuple = (3, 5, 7),
        char_num_filters: int = 128,
        # Lex
        lex_vocab_size: int = 24,
        lex_max_len: int = 129,
        # Fusion / shared
        d_fusion: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        if isinstance(char_kernel_sizes, list):
            char_kernel_sizes = tuple(char_kernel_sizes)

        try:
            from .model import TransformerViewEncoder
        except ImportError:
            from model import TransformerViewEncoder

        # ----- BPE surface (full sequence + CLS) -----
        self.surface_enc = TransformerViewEncoder(
            vocab_size=surface_vocab_size, max_len=surface_max_len,
            d_model=d_surface, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_surface * 4, dropout=dropout, pad_id=surface_pad_id,
        )
        self.surface_proj = nn.Linear(d_surface, d_fusion)

        # ----- CharCNN (pooled abstract) -----
        self.char_embed = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.char_convs = nn.ModuleList([
            nn.Conv1d(char_embed_dim, char_num_filters,
                       kernel_size=k, padding=k // 2)
            for k in char_kernel_sizes
        ])
        char_feat_dim = char_num_filters * len(char_kernel_sizes)
        self.char_proj = nn.Linear(char_feat_dim, d_fusion)

        # ----- Lex abstract (CLS pool) -----
        self.lex_enc = TransformerViewEncoder(
            vocab_size=lex_vocab_size, max_len=lex_max_len,
            d_model=d_fusion, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_fusion * 4, dropout=dropout, pad_id=0,
        )

        # ----- Stage 1: self-attn over [z_L, z_C] (length 2 abstract seq) -----
        self.s1_norm1 = nn.LayerNorm(d_fusion)
        self.s1_self_attn = nn.MultiheadAttention(
            embed_dim=d_fusion, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.s1_norm2 = nn.LayerNorm(d_fusion)
        self.s1_ffn = nn.Sequential(
            nn.Linear(d_fusion, d_fusion * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion * 4, d_fusion),
        )

        # ----- Stage 2: cross-attn (Q = stage1 abstract seq, K,V = H_BPE) -----
        self.s2_norm_q = nn.LayerNorm(d_fusion)
        self.s2_norm_kv = nn.LayerNorm(d_fusion)
        self.s2_cross_attn = nn.MultiheadAttention(
            embed_dim=d_fusion, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.s2_norm_ffn = nn.LayerNorm(d_fusion)
        self.s2_ffn = nn.Sequential(
            nn.Linear(d_fusion, d_fusion * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_fusion * 4, d_fusion),
        )

        # ----- Aux heads (deep supervision) -----
        self.aux_S = nn.Linear(d_surface, 1)
        self.aux_L = nn.Linear(d_fusion, 1)
        self.aux_C = nn.Linear(char_feat_dim, 1)

        # ----- Main classifier: concat([z_LA, z_final]) -----
        self.classifier = nn.Sequential(
            nn.Linear(d_fusion * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _encode_char_pool(self, char_ids):
        x = self.char_embed(char_ids)               # [B, T, D]
        x = x.transpose(1, 2)                       # [B, D, T]
        per_kernel = [F.relu(conv(x)) for conv in self.char_convs]
        H = torch.cat(per_kernel, dim=1)            # [B, F*n, T]
        z_pool = F.adaptive_max_pool1d(H, 1).squeeze(-1)  # [B, F*n]
        return z_pool

    def forward(self, surface_ids=None, surface_mask=None,
                lex_ids=None, lex_mask=None,
                char_ids=None, char_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0,
                surface_inputs_embeds=None):
        """If ``surface_inputs_embeds`` (shape [B, T, d_surface]) is given,
        the BPE surface branch consumes it instead of looking up
        ``surface_ids``. This is the entry point for FreeLB-style
        embedding-space adversarial training (Zhu et al. ICLR 2020):
        perturbations are added to the BPE token embedding only, with
        char and lex paths unchanged."""
        # ---- Encoders ----
        if surface_inputs_embeds is not None:
            s_out = self.surface_enc(
                input_ids=None, attention_mask=surface_mask,
                inputs_embeds=surface_inputs_embeds,
            )
        else:
            s_out = self.surface_enc(surface_ids, surface_mask)
        z_S = s_out["pooled"]                                # [B, d_surface]
        H_S = self.surface_proj(s_out["full"])               # [B, T, d_fusion]

        z_C_raw = self._encode_char_pool(char_ids)           # [B, F*n]
        z_C = self.char_proj(z_C_raw)                        # [B, d_fusion]

        z_L = self.lex_enc(lex_ids, lex_mask)["pooled"]      # [B, d_fusion]

        # ---- Aux predictions (before view dropout) ----
        p_S = self.aux_S(z_S).squeeze(-1)
        p_L = self.aux_L(z_L).squeeze(-1)
        p_A = self.aux_C(z_C_raw).squeeze(-1)   # naming: p_A keeps original
                                                  # signature for downstream code

        # ---- View dropout ----
        if self.training and view_dropout_prob > 0:
            B = z_L.size(0)
            dev = z_L.device
            keep_S = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            keep_L = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            keep_C = (torch.rand(B, device=dev) > view_dropout_prob).float().unsqueeze(-1)
            H_S_eff = H_S * keep_S.unsqueeze(1)
            z_L_eff = z_L * keep_L
            z_C_eff = z_C * keep_C
        else:
            H_S_eff = H_S
            z_L_eff = z_L
            z_C_eff = z_C

        # ---- Stage 1: self-attn over [z_L, z_C] ----
        abstract_seq = torch.stack([z_L_eff, z_C_eff], dim=1)   # [B, 2, d_fusion]
        q1 = self.s1_norm1(abstract_seq)
        s1_out, _ = self.s1_self_attn(q1, q1, q1, need_weights=False)
        abstract_seq = abstract_seq + s1_out
        ffn1 = self.s1_ffn(self.s1_norm2(abstract_seq))
        abstract_seq = abstract_seq + ffn1                      # [B, 2, d]

        # ---- Stage 2: cross-attn (Q = abstract_seq, K,V = H_S) ----
        q2 = self.s2_norm_q(abstract_seq)
        kv = self.s2_norm_kv(H_S_eff)
        kv_pad = ~surface_mask.bool() if surface_mask is not None else None
        attn_out, _ = self.s2_cross_attn(
            query=q2, key=kv, value=kv,
            key_padding_mask=kv_pad, need_weights=False,
        )
        attended_seq = abstract_seq + attn_out
        ffn2 = self.s2_ffn(self.s2_norm_ffn(attended_seq))
        attended_seq = attended_seq + ffn2                      # [B, 2, d]

        # ---- Pool both, concat ----
        z_LA = abstract_seq.mean(dim=1)                         # [B, d] (post-Stage1)
        z_final = attended_seq.mean(dim=1)                      # [B, d] (post-Stage2)

        cls_input = torch.cat([z_LA, z_final], dim=-1)          # [B, 2d]
        p_main = self.classifier(cls_input).squeeze(-1)

        return {
            "p_main": p_main,
            "p_S": p_S, "p_L": p_L, "p_A": p_A,   # p_A = char (replaced AST)
            "z_S": z_S, "z_L": z_L, "z_A": z_C_raw,
            "z_LA": z_LA, "z_final": z_final,
        }

    def compute_loss(self, output, labels,
                       weights=(0.7, 0.1, 0.1, 0.1), pos_weight=None):
        labels = labels.float()
        w_main, w_S, w_L, w_C = weights
        def bce(logits):
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight)
        loss_main = bce(output["p_main"])
        loss_S = bce(output["p_S"])
        loss_L = bce(output["p_L"])
        loss_C = bce(output["p_A"])    # p_A is now char aux
        total = (w_main * loss_main + w_S * loss_S
                  + w_L * loss_L + w_C * loss_C)
        return total, {
            "loss_total": total.item(),
            "loss_main": loss_main.item(),
            "loss_S": loss_S.item(),
            "loss_L": loss_L.item(),
            "loss_A": loss_C.item(),
        }


# ============================================================
# 2. Child-Sum Tree-LSTM on parsed AST
# ============================================================
class TreeLSTMCell(nn.Module):
    """Child-Sum Tree-LSTM cell (Tai et al. 2015).

    Per-node update:
        h_sum = Σ_j h_j               (sum over children's hidden states)
        i = σ(W_i x + U_i h_sum + b_i)
        f_jk = σ(W_f x + U_f h_j + b_f)   (per-child forget gate)
        o = σ(W_o x + U_o h_sum + b_o)
        u = tanh(W_u x + U_u h_sum + b_u)
        c = i ⊙ u + Σ_j f_jk ⊙ c_j
        h = o ⊙ tanh(c)
    """

    def __init__(self, x_dim, h_dim):
        super().__init__()
        self.x_dim, self.h_dim = x_dim, h_dim
        # combine i,o,u into one matmul with chunk
        self.W_iou = nn.Linear(x_dim, 3 * h_dim)
        self.U_iou = nn.Linear(h_dim, 3 * h_dim, bias=False)
        # forget is per-child
        self.W_f = nn.Linear(x_dim, h_dim)
        self.U_f = nn.Linear(h_dim, h_dim, bias=False)

    def forward(self, x, children_h, children_c):
        """x: [x_dim]. children_h, children_c: [num_children, h_dim].
        Returns (h, c), each [h_dim]."""
        if children_h is None or children_h.size(0) == 0:
            h_sum = torch.zeros(self.h_dim, device=x.device, dtype=x.dtype)
        else:
            h_sum = children_h.sum(dim=0)
        iou = self.W_iou(x) + self.U_iou(h_sum)
        i, o, u = iou.chunk(3, dim=-1)
        i = torch.sigmoid(i); o = torch.sigmoid(o); u = torch.tanh(u)
        if children_h is None or children_h.size(0) == 0:
            c = i * u
        else:
            f = torch.sigmoid(self.W_f(x).unsqueeze(0) + self.U_f(children_h))  # [n_ch, h]
            c = i * u + (f * children_c).sum(dim=0)
        h = o * torch.tanh(c)
        return h, c


class TreeLSTMModel(nn.Module):
    """Hu's AST-LSTM baseline.

    Each sample provides:
        ast_node_ids:  list[int]   length N (node label token ids)
        ast_parent:    list[int]   length N (parent index, -1 for root)
                                    nodes assumed in topological / post-
                                    order so all children come before parents.

    These are passed through the dataloader as Python lists (variable length).
    Forward processes each sample one at a time (Python loop). Slow but
    correct; for our 30k train set it's still ~1-2 minutes per epoch on GPU.
    """

    def __init__(
        self,
        ast_vocab_size: int = 100,
        ast_pad_id: int = 0,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        self.embed = nn.Embedding(ast_vocab_size, embed_dim, padding_idx=ast_pad_id)
        self.cell = TreeLSTMCell(embed_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.h_dim = hidden_dim

    def _forward_single_tree(self, node_ids, parent, device):
        """Process one tree bottom-up. Returns root hidden state [h_dim]."""
        N = len(node_ids)
        h_dict = [None] * N
        c_dict = [None] * N
        # Group children per parent
        children_of = [[] for _ in range(N)]
        for i, p in enumerate(parent):
            if p >= 0 and p < N:
                children_of[p].append(i)

        # Process in post-order: nodes whose children are done. Since we
        # assume input is in post-order this is just iterate i=0..N-1.
        emb_table = self.embed(torch.tensor(node_ids, device=device))   # [N, embed]
        for i in range(N):
            children = children_of[i]
            if children:
                ch_h = torch.stack([h_dict[c] for c in children], dim=0)
                ch_c = torch.stack([c_dict[c] for c in children], dim=0)
            else:
                ch_h = torch.zeros(0, self.h_dim, device=device, dtype=emb_table.dtype)
                ch_c = torch.zeros(0, self.h_dim, device=device, dtype=emb_table.dtype)
            h, c = self.cell(emb_table[i], ch_h, ch_c)
            h_dict[i] = h
            c_dict[i] = c
        # Root is the last node in post-order
        return h_dict[N - 1]

    def forward(self, surface_ids=None, surface_mask=None, lex_ids=None, lex_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                ast_node_ids=None, ast_parent=None,
                view_dropout_prob: float = 0.0):
        """ast_node_ids, ast_parent are lists of lists (one per sample)."""
        device = next(self.parameters()).device
        B = len(ast_node_ids) if ast_node_ids is not None else surface_ids.size(0)
        roots = []
        for b in range(B):
            if ast_node_ids and ast_parent and len(ast_node_ids[b]) > 0:
                root_h = self._forward_single_tree(ast_node_ids[b], ast_parent[b], device)
            else:
                root_h = torch.zeros(self.h_dim, device=device,
                                       dtype=self.embed.weight.dtype)
            roots.append(root_h)
        H = torch.stack(roots, dim=0)                          # [B, h_dim]
        p = self.classifier(H).squeeze(-1)
        zero = torch.zeros_like(p)
        return {"p_main": p, "p_S": zero, "p_L": zero, "p_A": p,
                "z_S": zero, "z_L": zero, "z_A": H,
                "z_LA": zero, "z_final": zero}

    def compute_loss(self, output, labels, weights=None, pos_weight=None):
        labels = labels.float()
        loss = F.binary_cross_entropy_with_logits(
            output["p_main"], labels, pos_weight=pos_weight,
        )
        return loss, {"loss_total": loss.item(), "loss_main": loss.item(),
                        "loss_S": 0.0, "loss_L": 0.0, "loss_A": 0.0}
