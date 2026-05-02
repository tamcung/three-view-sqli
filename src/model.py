#!/usr/bin/env python3
"""Generic Transformer view encoder used by all single-view and fusion models."""
from __future__ import annotations
import torch
import torch.nn as nn


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

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> dict:
        """Returns dict with 'pooled' [B, d] (CLS) and 'full' [B, T, d].

        Either ``input_ids`` (shape [B, T]) or ``inputs_embeds`` (shape
        [B, T, d_model]) must be provided. When ``inputs_embeds`` is
        passed, the token-embedding lookup is skipped — this lets
        FreeLB-style adversarial training inject a perturbed embedding
        tensor.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids / inputs_embeds")

        if inputs_embeds is None:
            B, T = input_ids.shape
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)
            x = self.token_emb(input_ids) + self.pos_emb(pos)
        else:
            B, T, _ = inputs_embeds.shape
            pos = torch.arange(T, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
            x = inputs_embeds + self.pos_emb(pos)
        x = self.emb_norm(x)
        x = self.emb_dropout(x)

        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)
        return {"pooled": x[:, 0], "full": x}
