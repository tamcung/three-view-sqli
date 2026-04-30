#!/usr/bin/env python3
"""Single-view variants of ThreeViewModel for ablation studies.

All variants implement the same `forward(...)` signature as the full model,
so train.py / evaluate.py can swap them via the `model_variant` config key.

Variants:
  - SurfaceOnlyModel    : surface encoder + linear classifier
  - LexicalOnlyModel    : lexical encoder + linear classifier
  - ASTOnlyModel        : AST encoder + linear classifier
  - SurfaceLexModel     : surface + lex (no AST view), reuse ThreeView fusion
                          but force AST tokens to all-pad
  - SurfaceASTModel     : surface + AST (no lex)
  - LexASTModel         : lex + AST (no surface)

Each variant keeps the same output schema as ThreeViewModel.forward() —
{p_main, p_S, p_L, p_A, ...} — so the trainer can use the unified loss head.
For disabled views, the corresponding aux logit is held at zero and its loss
weight should be set to 0 in the config.
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
    """Common scaffolding for one-encoder ablations."""

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
        lex_vocab_size: int = 24,
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
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        z = self._forward_one(lex_ids, lex_mask)
        p = self.classifier(z).squeeze(-1)
        zero = torch.zeros_like(p)
        return {"p_main": p, "p_S": zero, "p_L": p, "p_A": zero,
                "z_S": zero, "z_L": z, "z_A": zero,
                "z_LA": zero, "z_final": zero}


class ASTOnlyModel(_SingleViewModel):
    def __init__(
        self,
        ast_vocab_size: int = 100,
        ast_max_len: int = 257,
        d_abstract: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
        **_ignored,
    ):
        enc = TransformerViewEncoder(
            vocab_size=ast_vocab_size, max_len=ast_max_len,
            d_model=d_abstract, n_layers=n_layers, n_heads=n_heads,
            d_ff=d_abstract * 4, dropout=dropout, pad_id=0,
        )
        super().__init__(enc, d_abstract, dropout)

    def forward(self, surface_ids=None, surface_mask=None, lex_ids=None, lex_mask=None,
                ast_ids=None, ast_mask=None, ast_valid=None,
                view_dropout_prob: float = 0.0):
        z = self._forward_one(ast_ids, ast_mask)
        # When AST parse failed (ast_valid=0), zero the embedding so the
        # classifier learns a sensible "missing AST" output.
        if ast_valid is not None:
            z = z * ast_valid.float().unsqueeze(-1)
        p = self.classifier(z).squeeze(-1)
        zero = torch.zeros_like(p)
        return {"p_main": p, "p_S": zero, "p_L": zero, "p_A": p,
                "z_S": zero, "z_L": zero, "z_A": z,
                "z_LA": zero, "z_final": zero}


# ============================================================
# Two-view ablations: reuse the full ThreeViewModel but mask one view
# ============================================================
class TwoViewModel(nn.Module):
    """Drops one view from the full three-view model. The dropped view's
    encoder is still allocated (to keep checkpoint shape simple) but its
    output is hard-zeroed.

    Args:
        drop: which view to disable — 'surface' / 'lexical' / 'ast'.
        **kwargs: same as ThreeViewModel constructor.
    """

    def __init__(self, drop: str, **kwargs):
        super().__init__()
        try:
            from .model import ThreeViewModel
        except ImportError:
            from model import ThreeViewModel
        assert drop in ("surface", "lexical", "ast")
        self.drop = drop
        self.inner = ThreeViewModel(**kwargs)

    def forward(self, surface_ids, surface_mask, lex_ids, lex_mask,
                ast_ids, ast_mask, ast_valid=None,
                view_dropout_prob: float = 0.0):
        # Force the dropped view's input to "empty" (CLS only).
        if self.drop == "surface":
            B = surface_ids.size(0)
            surface_ids = torch.zeros(B, 1, dtype=torch.long, device=surface_ids.device).fill_(
                self.inner.surface_enc.token_emb.padding_idx or 0
            )
            surface_ids[:, 0] = 0  # cls position
            surface_mask = torch.ones(B, 1, dtype=surface_mask.dtype, device=surface_mask.device)
        elif self.drop == "lexical":
            B = lex_ids.size(0)
            lex_ids = torch.zeros(B, 1, dtype=torch.long, device=lex_ids.device)
            lex_ids[:, 0] = 2  # CLS
            lex_mask = torch.ones(B, 1, dtype=lex_mask.dtype, device=lex_mask.device)
        elif self.drop == "ast":
            B = ast_ids.size(0)
            ast_ids = torch.zeros(B, 1, dtype=torch.long, device=ast_ids.device)
            ast_ids[:, 0] = 2  # CLS
            ast_mask = torch.ones(B, 1, dtype=ast_mask.dtype, device=ast_mask.device)
            ast_valid = torch.zeros(B, dtype=torch.float, device=ast_ids.device)

        return self.inner(
            surface_ids, surface_mask, lex_ids, lex_mask,
            ast_ids, ast_mask, ast_valid, view_dropout_prob,
        )

    def compute_loss(self, output, labels, weights=(0.7, 0.1, 0.1, 0.1), pos_weight=None):
        # Zero out the dropped view's aux loss weight
        w_main, w_S, w_L, w_A = weights
        if self.drop == "surface": w_S = 0.0
        if self.drop == "lexical": w_L = 0.0
        if self.drop == "ast":     w_A = 0.0
        return self.inner.compute_loss(
            output, labels, weights=(w_main, w_S, w_L, w_A), pos_weight=pos_weight,
        )


# ============================================================
# Variant factory
# ============================================================
def build_model(variant: str, model_kwargs: dict) -> nn.Module:
    """Instantiate a model variant by name."""
    variant = variant.lower()
    if variant in ("three_view", "full", "default", ""):
        try:
            from .model import ThreeViewModel
        except ImportError:
            from model import ThreeViewModel
        return ThreeViewModel(**model_kwargs)
    if variant == "surface_only":
        return SurfaceOnlyModel(**model_kwargs)
    if variant == "lexical_only":
        return LexicalOnlyModel(**model_kwargs)
    if variant == "ast_only":
        return ASTOnlyModel(**model_kwargs)
    if variant == "no_surface":
        return TwoViewModel(drop="surface", **model_kwargs)
    if variant == "no_lexical":
        return TwoViewModel(drop="lexical", **model_kwargs)
    if variant == "no_ast":
        return TwoViewModel(drop="ast", **model_kwargs)
    if variant == "sequence_lstm":
        try:
            from .baseline_models import SequenceLSTMModel
        except ImportError:
            from baseline_models import SequenceLSTMModel
        return SequenceLSTMModel(**model_kwargs)
    if variant == "tree_lstm":
        try:
            from .baseline_models import TreeLSTMModel
        except ImportError:
            from baseline_models import TreeLSTMModel
        return TreeLSTMModel(**model_kwargs)
    if variant in ("char_cnn", "charcnn"):
        try:
            from .baseline_models import CharCNNModel
        except ImportError:
            from baseline_models import CharCNNModel
        return CharCNNModel(**model_kwargs)
    if variant in ("char_bilstm", "charbilstm"):
        try:
            from .baseline_models import CharBiLSTMModel
        except ImportError:
            from baseline_models import CharBiLSTMModel
        return CharBiLSTMModel(**model_kwargs)
    if variant in ("char_lex", "charlex"):
        try:
            from .baseline_models import CharLexModel
        except ImportError:
            from baseline_models import CharLexModel
        return CharLexModel(**model_kwargs)
    if variant in ("char_lex_xattn", "charlexxattn"):
        try:
            from .baseline_models import CharLexCrossAttnModel
        except ImportError:
            from baseline_models import CharLexCrossAttnModel
        return CharLexCrossAttnModel(**model_kwargs)
    if variant in ("bpe_char_lex", "tri_view"):
        try:
            from .baseline_models import BPECharLexModel
        except ImportError:
            from baseline_models import BPECharLexModel
        return BPECharLexModel(**model_kwargs)
    if variant in ("bpe_char_lex_stage", "tri_view_stage"):
        try:
            from .baseline_models import BPECharLexStageModel
        except ImportError:
            from baseline_models import BPECharLexStageModel
        return BPECharLexStageModel(**model_kwargs)
    if variant in ("bpe_char_lex_full_stage", "tri_view_full_stage"):
        try:
            from .baseline_models import BPECharLexFullStageModel
        except ImportError:
            from baseline_models import BPECharLexFullStageModel
        return BPECharLexFullStageModel(**model_kwargs)
    if variant in ("bpe_char_lex_full_attn", "tri_view_full_attn"):
        try:
            from .baseline_models import BPECharLexFullAttnModel
        except ImportError:
            from baseline_models import BPECharLexFullAttnModel
        return BPECharLexFullAttnModel(**model_kwargs)
    raise ValueError(f"Unknown model_variant: {variant}")
