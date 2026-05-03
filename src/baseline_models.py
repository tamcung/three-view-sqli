#!/usr/bin/env python3
"""Baseline models + proposed three-view fusion.

Char-level baselines (§3.7 main comparison):
  - CharCNNModel          : byte-level CNN
  - CharLSTMModel         : byte-level LSTM
  - CharGRUModel          : byte-level GRU

Proposed three-view fusion method (本文方法):
  - ThreeViewFusionModel  : BPE + Char + Lex, all encoded by Transformer,
                            joined by view-type embedding, then a single
                            full self-attention fusion encoder.

(Char-Transformer single-view ablation lives in ablation_models.py since
it matches the char_enc inside ThreeViewFusionModel.)
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


# ============================================================
# Char-level CNN (Kim 2014 adapted; byte vocab 257)
# ============================================================
class CharCNNModel(nn.Module):
    def __init__(
        self,
        char_vocab_size: int = 257,
        embed_dim: int = 64,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        num_filters: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.5,
        **_ignored,
    ):
        super().__init__()
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
        x = self.embed(char_ids)
        x = x.transpose(1, 2)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
            feats.append(h)
        z = torch.cat(feats, dim=-1)
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
# Char-level (uni-directional) LSTM
# ============================================================
class CharLSTMModel(nn.Module):
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
            batch_first=True, bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
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
        h_last = h[-1]
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
# Char-level (uni-directional) GRU
# ============================================================
class CharGRUModel(nn.Module):
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
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
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
            _, h = self.gru(packed)
        else:
            _, h = self.gru(x)
        h_last = h[-1]
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
# MVC-BiCNN baseline (Kakisim 2024).
# 三视图（tokenized / converted / enriched），每视图 BiLSTM + 多核 CNN。
# Late fusion: 三视图独立 sigmoid 求和阈值化（"consensus"）。
# ============================================================
class _BiCNNView(nn.Module):
    """BiLSTM + multi-kernel CNN encoder used by each MVC-BiCNN view.

    Following Kakisim 2024 §3.2 / §4.1:
    - Embedding (padding_idx-aware)
    - 1-layer bidirectional LSTM
    - Concatenation of embedding output and BiLSTM output (paper Eq. 3)
    - Multi-kernel 1D-CNN with k filter sizes
    - Per-kernel max-pool, concatenated → feature vector
    - Fully connected ReLU layer (default 250 hidden units per paper §4.1)
    - Final sigmoid head
    """

    def __init__(self, vocab_size, embed_dim=32, lstm_hidden=32,
                 num_filters=32, kernel_sizes=(2, 3, 4),
                 padding_idx=None, dropout=0.3, fc_hidden=250):
        super().__init__()
        if isinstance(kernel_sizes, list):
            kernel_sizes = tuple(kernel_sizes)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden, num_layers=1,
                              batch_first=True, bidirectional=True)
        # Per paper Eq. (3): H = [h ⊕ ε] — concatenate embedding output with
        # BiLSTM output along the feature dimension before feeding the CNN.
        cnn_in_dim = embed_dim + 2 * lstm_hidden
        self.convs = nn.ModuleList([
            nn.Conv1d(cnn_in_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        feat_dim = num_filters * len(kernel_sizes)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, 1),
        )

    def forward(self, ids, mask=None):
        x = self.embed(ids)                       # [B, T, D]
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True,
                                              enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            from torch.nn.utils.rnn import pad_packed_sequence
            h, _ = pad_packed_sequence(packed_out, batch_first=True,
                                          total_length=ids.size(1))
        else:
            h, _ = self.lstm(x)
        # Paper Eq. (3): concatenate embedding output with BiLSTM output along
        # the feature dimension, preserving both lexical and contextual cues.
        H = torch.cat([x, h], dim=-1)             # [B, T, D + 2H]
        H = H.transpose(1, 2)                     # [B, D + 2H, T]
        feats = []
        for conv in self.convs:
            f = F.relu(conv(H))
            f = F.adaptive_max_pool1d(f, 1).squeeze(-1)
            feats.append(f)
        z = torch.cat(feats, dim=-1)
        z = self.dropout(z)
        return self.classifier(z).squeeze(-1)


class MVCBiCNNModel(nn.Module):
    """MVC-BiCNN: three BiLSTM-CNN views with late fusion via sum of sigmoids.

    Faithful reimplementation of Kakisim 2024. Each view is a sqlparse-based
    sequence built by ``MVCSamplePreprocessor`` and routed through existing
    dataset slots:

        surface_ids → tokenized view  (sqlparse SQL terms after noise filter)
        lex_ids     → converted view  (21 SQL semantic tags)
        char_ids    → enriched view   (token-tag interleaved sequence)

    Each view has an independent BiLSTM-CNN encoder + sigmoid head; the
    consensus prediction is (sigmoid_T + sigmoid_C + sigmoid_E) / 3 in [0,1],
    classified as attack when above 0.5 (equivalent to the paper's threshold
    of 1.5 on the unaveraged sum).

    Default hyperparameters follow Kakisim 2024 §4.1 verbatim
    (embed_dim 32, lstm_hidden 32, num_filters 32, kernels [2,3,4], FC 250).
    """

    def __init__(
        self,
        surface_vocab_size: int = 60026,   # MVC tokenized vocab
        surface_pad_id: int = 0,
        lex_vocab_size: int = 23,           # MVC converted vocab (PAD+UNK+21)
        lex_pad_id: int = 0,
        char_vocab_size: int = 60033,       # MVC enriched vocab
        char_pad_id: int = 0,
        embed_dim: int = 32,                # paper §4.1: 32
        lstm_hidden: int = 32,              # paper §4.1: 32
        num_filters: int = 32,              # paper §4.1: 32
        kernel_sizes=(2, 3, 4),             # paper §4.1: 2,3,4
        fc_hidden: int = 250,               # paper §4.1: 250 ReLU FC
        dropout: float = 0.3,
        **_ignored,
    ):
        super().__init__()
        self.surface_pad_id = surface_pad_id

        # View 1: tokenized (sqlparse SQL terms)
        self.tokenized = _BiCNNView(
            surface_vocab_size, embed_dim, lstm_hidden,
            num_filters, kernel_sizes,
            padding_idx=surface_pad_id, dropout=dropout,
            fc_hidden=fc_hidden,
        )
        # View 2: converted (21 SQL semantic tags)
        self.converted = _BiCNNView(
            lex_vocab_size, embed_dim, lstm_hidden,
            num_filters, kernel_sizes,
            padding_idx=lex_pad_id, dropout=dropout,
            fc_hidden=fc_hidden,
        )
        # View 3: enriched (token-tag interleaved sequence)
        self.enriched = _BiCNNView(
            char_vocab_size, embed_dim, lstm_hidden,
            num_filters, kernel_sizes,
            padding_idx=char_pad_id, dropout=dropout,
            fc_hidden=fc_hidden,
        )

    def forward(self, surface_ids=None, surface_mask=None,
                lex_ids=None, lex_mask=None,
                char_ids=None, char_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0,
                surface_inputs_embeds=None):
        # Per-view logits (each view is independently encoded by its own BiCNN)
        logit_T = self.tokenized(surface_ids, surface_mask)
        logit_C = self.converted(lex_ids, lex_mask)
        logit_E = self.enriched(char_ids, char_mask)

        # Consensus = average of three sigmoids (equivalent to Kakisim's
        # sum-of-sigmoids with threshold 1.5; we use mean+threshold 0.5).
        prob_main = (torch.sigmoid(logit_T) +
                       torch.sigmoid(logit_C) +
                       torch.sigmoid(logit_E)) / 3.0
        # Convert ensemble probability back to a logit so the trainer's
        # default decision boundary `logit > 0` matches `prob > 0.5`.
        eps = 1e-7
        p_main = torch.log(prob_main.clamp(min=eps) /
                              (1 - prob_main).clamp(min=eps))

        zero = torch.zeros_like(p_main)
        return {
            "p_main": p_main,
            "p_S": logit_T,        # tokenized view logit
            "p_L": logit_C,        # converted view logit
            "p_A": logit_E,        # enriched view logit
            "z_S": zero, "z_L": zero, "z_A": zero,
            "z_LA": zero, "z_final": zero,
        }

    def compute_loss(self, output, labels, weights=None, pos_weight=None):
        """Each view trained independently with BCE; total = average of
        the three view losses. The consensus prediction (p_main) is purely
        an inference-time aggregation."""
        labels = labels.float()
        def bce(logits):
            return F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight,
            )
        loss_T = bce(output["p_S"])
        loss_C = bce(output["p_L"])
        loss_E = bce(output["p_A"])
        total = (loss_T + loss_C + loss_E) / 3.0
        return total, {
            "loss_total": total.item(),
            "loss_main": total.item(),
            "loss_S": loss_T.item(),
            "loss_L": loss_C.item(),
            "loss_A": loss_E.item(),
        }


# ============================================================
# Three-view fusion (本文方法): BPE + Char + Lex, all Transformer-encoded,
# joined by view-type embedding, then a single full self-attention fusion.
# ============================================================
class ThreeViewFusionModel(nn.Module):
    def __init__(
        self,
        surface_vocab_size: int = 50265,
        surface_max_len: int = 257,
        surface_pad_id: int = 1,
        d_surface: int = 384,
        char_vocab_size: int = 257,
        char_max_len: int = 257,
        lex_vocab_size: int = 365,
        lex_max_len: int = 129,
        d_fusion: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        fusion_layers: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        **_ignored,
    ):
        super().__init__()
        try:
            from .model import TransformerViewEncoder
        except ImportError:
            from model import TransformerViewEncoder

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

        self.lex_enc = TransformerViewEncoder(
            vocab_size=lex_vocab_size, max_len=lex_max_len,
            d_model=d_fusion, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_fusion * 4, dropout=dropout, pad_id=0,
        )

        self.view_emb = nn.Embedding(3, d_fusion)

        fusion_layer = nn.TransformerEncoderLayer(
            d_model=d_fusion, nhead=n_heads,
            dim_feedforward=d_fusion * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=fusion_layers)

        self.aux_S = nn.Linear(d_surface, 1)
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

    # ----------------------------------------------------------------
    # Two-stage forward decomposition (used by FreeLB adversarial training):
    #   stage 1: encode_views → returns H_S, H_C, H_L plus aux logits
    #   stage 2: fuse_from_views(H_S, H_C, H_L) → main logits
    # The standard ``forward`` below stitches both stages together.
    # ----------------------------------------------------------------
    def encode_views(self, surface_ids=None, surface_mask=None,
                     lex_ids=None, lex_mask=None,
                     char_ids=None, char_mask=None,
                     surface_inputs_embeds=None,
                     char_inputs_embeds=None,
                     lex_inputs_embeds=None):
        """Run the three view encoders only. Returns:
          H_S [B, L_S, d_fusion]  (post surface_proj)
          H_C [B, L_C, d_fusion]
          H_L [B, L_L, d_fusion]
          aux: {p_S, p_L, p_A, z_S, z_L, z_C}  (auxiliary classifier logits)

        Each ``*_inputs_embeds`` arg, if provided, replaces the token-id
        embedding lookup in the corresponding encoder. This is the entry
        point for token-embedding-level FreeLB perturbation.
        """
        if surface_inputs_embeds is not None:
            s_out = self.surface_enc(
                input_ids=None, attention_mask=surface_mask,
                inputs_embeds=surface_inputs_embeds,
            )
        else:
            s_out = self.surface_enc(surface_ids, surface_mask)
        z_S = s_out["pooled"]
        H_S = self.surface_proj(s_out["full"])

        if char_mask is None and char_ids is not None:
            char_mask = (char_ids != 0).long()
        if char_inputs_embeds is not None:
            c_out = self.char_enc(
                input_ids=None, attention_mask=char_mask,
                inputs_embeds=char_inputs_embeds,
            )
        else:
            c_out = self.char_enc(char_ids, char_mask)
        z_C = c_out["pooled"]
        H_C = c_out["full"]

        if lex_inputs_embeds is not None:
            l_out = self.lex_enc(
                input_ids=None, attention_mask=lex_mask,
                inputs_embeds=lex_inputs_embeds,
            )
        else:
            l_out = self.lex_enc(lex_ids, lex_mask)
        z_L = l_out["pooled"]
        H_L = l_out["full"]

        aux = {
            "p_S": self.aux_S(z_S).squeeze(-1),
            "p_L": self.aux_L(z_L).squeeze(-1),
            "p_A": self.aux_C(z_C).squeeze(-1),
            "z_S": z_S, "z_L": z_L, "z_C": z_C,
        }
        return H_S, H_C, H_L, aux

    def fuse_from_views(self, H_S, H_C, H_L,
                          surface_mask, lex_mask, char_mask,
                          view_dropout_prob: float = 0.0):
        """Run view-type embedding + fusion + classifier given the three
        view-encoded representations. Inputs ``H_S/H_C/H_L`` may be the
        unperturbed encoder outputs or perturbed via FreeLB.

        Returns the dict that ``forward`` returns (without aux logits —
        callers that need them should keep the aux dict from
        ``encode_views``).
        """
        B = H_S.size(0); dev = H_S.device
        v_S = self.view_emb(torch.tensor(0, device=dev))
        v_L = self.view_emb(torch.tensor(1, device=dev))
        v_C = self.view_emb(torch.tensor(2, device=dev))
        H_S_t = H_S + v_S
        H_L_t = H_L + v_L
        H_C_t = H_C + v_C

        if self.training and view_dropout_prob > 0:
            keep_S = (torch.rand(B, device=dev) > view_dropout_prob).float().view(B, 1, 1)
            keep_L = (torch.rand(B, device=dev) > view_dropout_prob).float().view(B, 1, 1)
            keep_C = (torch.rand(B, device=dev) > view_dropout_prob).float().view(B, 1, 1)
            H_S_t = H_S_t * keep_S
            H_L_t = H_L_t * keep_L
            H_C_t = H_C_t * keep_C

        fused = torch.cat([H_S_t, H_L_t, H_C_t], dim=1)
        joint_mask = torch.cat([surface_mask, lex_mask, char_mask], dim=1)
        joint_pad = ~joint_mask.bool()
        out = self.fusion_encoder(fused, src_key_padding_mask=joint_pad)
        z_final = self._masked_mean(out, joint_mask)
        p_main = self.classifier(z_final).squeeze(-1)
        return {"p_main": p_main, "z_final": z_final, "z_LA": z_final}

    def forward(self, surface_ids=None, surface_mask=None,
                lex_ids=None, lex_mask=None,
                char_ids=None, char_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0,
                surface_inputs_embeds=None,
                # FreeLB hooks: optional perturbations applied to each view
                # representation between encoder output and view-type embedding.
                # Each tensor has the same shape as the corresponding H_v.
                delta_S=None, delta_C=None, delta_L=None):
        H_S, H_C, H_L, aux = self.encode_views(
            surface_ids=surface_ids, surface_mask=surface_mask,
            lex_ids=lex_ids, lex_mask=lex_mask,
            char_ids=char_ids, char_mask=char_mask,
            surface_inputs_embeds=surface_inputs_embeds,
        )
        if delta_S is not None: H_S = H_S + delta_S
        if delta_C is not None: H_C = H_C + delta_C
        if delta_L is not None: H_L = H_L + delta_L

        if char_mask is None and char_ids is not None:
            char_mask = (char_ids != 0).long()
        if lex_mask is None and lex_ids is not None:
            lex_mask = (lex_ids != 0).long()

        fused = self.fuse_from_views(
            H_S, H_C, H_L,
            surface_mask=surface_mask, lex_mask=lex_mask, char_mask=char_mask,
            view_dropout_prob=view_dropout_prob,
        )
        return {
            "p_main": fused["p_main"],
            "p_S": aux["p_S"], "p_L": aux["p_L"], "p_A": aux["p_A"],
            "z_S": aux["z_S"], "z_L": aux["z_L"], "z_A": aux["z_C"],
            "z_LA": fused["z_final"], "z_final": fused["z_final"],
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
        loss_C = bce(output["p_A"])
        total = (w_main * loss_main + w_S * loss_S
                  + w_L * loss_L + w_C * loss_C)
        return total, {
            "loss_total": total.item(),
            "loss_main": loss_main.item(),
            "loss_S": loss_S.item(),
            "loss_L": loss_L.item(),
            "loss_A": loss_C.item(),
        }
