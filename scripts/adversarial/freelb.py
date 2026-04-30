#!/usr/bin/env python3
"""FreeLB (Zhu et al. ICLR 2020) for the BPECharLexStageModel.

Implements embedding-space adversarial training that solves the inner
maximization of the min-max robust optimization

    min_θ  E_{(x,y)}  max_{‖δ‖_F ≤ ε}  L( f_θ(emb(x) + δ),  y )

via K-step PGD on the BPE token embedding. Char and lex paths receive
no perturbation — perturbing all three views simultaneously is possible
but harms convergence (the model was originally trained without view
perturbations and the lex/char vocabularies are tiny, so the embedding
manifold is brittle there).

Free-LB's "free" comes from accumulating the gradient across all K PGD
steps and using their average as the parameter update direction. So one
optimizer step costs K forward+backward passes but yields a more
adversarially-robust update than vanilla SGD.

Public entry:
    train_one_epoch_freelb(model, loader, optimizer, scheduler, scaler,
                            device, view_dropout_prob, log_every, log,
                            amp_mode, freelb_cfg)
    where freelb_cfg = FreeLBConfig(...)
"""
from __future__ import annotations
import dataclasses
import inspect
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class FreeLBConfig:
    """Hyperparameters for FreeLB. Defaults follow Zhu et al. for BERT
    fine-tuning, scaled down for our smaller surface encoder."""
    n_steps: int = 3                # K — number of PGD steps per batch
    init_norm: float = 0.05         # σ for initial δ ~ uniform(-σ, σ)
    step_size: float = 1e-2          # α — PGD step size
    max_norm: float = 0.2           # ε — Frobenius-norm constraint per token
    adv_loss_weight: float = 1.0    # weight on adv loss vs clean loss


def _move_batch_to(batch: dict, device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _filter_kwargs(model, kwargs: dict) -> dict:
    accepted = set(inspect.signature(model.forward).parameters.keys())
    return {k: v for k, v in kwargs.items() if k in accepted}


def _amp_ctx(device, amp_mode: str):
    """Match train.py's amp handling."""
    if amp_mode == "fp32":
        return torch.autocast(device_type=device.type, dtype=torch.float32, enabled=False)
    if amp_mode == "fp16":
        enabled = device.type == "cuda"
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)
    enabled = device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def train_one_epoch_freelb(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    device,
    view_dropout_prob: float,
    log_every: int,
    log,
    amp_mode: str = "bf16",
    freelb_cfg: FreeLBConfig | None = None,
) -> dict:
    """One epoch of FreeLB training.

    Per batch:
      1. Forward with clean BPE embedding → loss_clean. Backprop, scale by 1/(K+1).
      2. PGD K steps:
            forward with emb + δ → loss_adv
            backprop into δ AND model parameters (each step contributes 1/(K+1))
            update δ in the direction of its gradient, project to ε-ball.
      3. After K+1 backwards, accumulated parameter gradient is the FreeLB
         update direction. optimizer.step() once.

    The model must support `surface_inputs_embeds=` keyword (see
    BPECharLexStageModel patch).
    """
    if freelb_cfg is None:
        freelb_cfg = FreeLBConfig()
    K = freelb_cfg.n_steps

    model.train()
    accepted = set(inspect.signature(model.forward).parameters.keys())
    if "surface_inputs_embeds" not in accepted:
        raise RuntimeError(
            f"Model {type(model).__name__} does not accept "
            f"`surface_inputs_embeds`. FreeLB needs an embedding-injection "
            f"forward path. Patch the model or use vanilla train_one_epoch."
        )

    losses, main_losses, adv_losses = [], [], []
    aux_S, aux_L, aux_A = [], [], []
    n = 0
    t0 = time.time()

    # Locate the BPE token embedding once
    bpe_emb_layer = model.surface_enc.token_emb        # nn.Embedding

    for step, batch in enumerate(loader):
        batch = _move_batch_to(batch, device)
        labels = batch["labels"]

        optimizer.zero_grad(set_to_none=True)

        # ---- Clean BPE embedding (used as anchor for δ) ----
        with torch.no_grad():
            bpe_emb_clean = bpe_emb_layer(batch["surface_ids"])      # [B, T, D]
        # Mask: don't perturb pad positions
        attn = batch["surface_mask"].unsqueeze(-1).float()           # [B, T, 1]

        # ---- Initialize δ with small random noise, masked ----
        delta = torch.empty_like(bpe_emb_clean).uniform_(
            -freelb_cfg.init_norm, freelb_cfg.init_norm,
        )
        delta = (delta * attn).detach()
        delta.requires_grad_(True)

        total_loss_for_log = 0.0
        clean_loss_for_log = 0.0
        adv_loss_for_log = 0.0
        components_last = None

        # ---- Step 0: clean (no δ) forward — provides the K+1 baseline ----
        # We treat clean as one of the K+1 contributions, scaled by 1/(K+1)
        with _amp_ctx(device, amp_mode):
            kwargs0 = dict(
                surface_inputs_embeds=bpe_emb_clean,
                surface_mask=batch["surface_mask"],
                lex_ids=batch["lex_ids"], lex_mask=batch["lex_mask"],
                ast_ids=batch["ast_ids"], ast_mask=batch["ast_mask"],
                ast_valid=batch["ast_valid"],
                view_dropout_prob=view_dropout_prob,
                ast_node_ids=batch.get("ast_node_ids"),
                ast_parent=batch.get("ast_parent"),
                char_ids=batch.get("char_ids"),
                char_mask=batch.get("char_mask"),
            )
            out0 = model(**_filter_kwargs(model, kwargs0))
            loss0, comp0 = model.compute_loss(out0, labels)
        loss0_scaled = loss0 / (K + 1)
        if scaler is not None:
            scaler.scale(loss0_scaled).backward()
        else:
            loss0_scaled.backward()
        clean_loss_for_log = float(comp0["loss_main"])
        total_loss_for_log += float(comp0["loss_total"]) / (K + 1)

        # ---- PGD K steps: each accumulates parameter grad and updates δ ----
        for k in range(K):
            with _amp_ctx(device, amp_mode):
                kwargs_adv = dict(kwargs0)
                kwargs_adv["surface_inputs_embeds"] = bpe_emb_clean + delta
                out_k = model(**_filter_kwargs(model, kwargs_adv))
                loss_k, comp_k = model.compute_loss(out_k, labels)
            loss_k_scaled = (freelb_cfg.adv_loss_weight * loss_k) / (K + 1)
            if scaler is not None:
                scaler.scale(loss_k_scaled).backward()
            else:
                loss_k_scaled.backward()

            adv_loss_for_log += float(comp_k["loss_main"])
            total_loss_for_log += float(comp_k["loss_total"]) / (K + 1)
            components_last = comp_k

            # ---- Update δ with PGD: ascent on loss, project to ε-ball ----
            with torch.no_grad():
                if delta.grad is None:
                    break  # safety
                # per-token gradient norm normalization
                g = delta.grad.detach()                                # [B, T, D]
                g = g * attn                                            # zero pad-pos
                g_norm = g.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # [B, T, 1]
                g_unit = g / g_norm
                delta_new = delta + freelb_cfg.step_size * g_unit
                # Project: clamp Frobenius norm per-token to ≤ max_norm
                d_norm = delta_new.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                factor = (freelb_cfg.max_norm / d_norm).clamp(max=1.0)
                delta_new = delta_new * factor * attn
            delta = delta_new.detach()
            delta.requires_grad_(True)

        # ---- Optimizer step: accumulated grad is the FreeLB direction ----
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        losses.append(total_loss_for_log)
        main_losses.append(clean_loss_for_log)
        adv_losses.append(adv_loss_for_log / max(1, K))
        if components_last is not None:
            aux_S.append(components_last["loss_S"])
            aux_L.append(components_last["loss_L"])
            aux_A.append(components_last["loss_A"])
        else:
            aux_S.append(comp0["loss_S"])
            aux_L.append(comp0["loss_L"])
            aux_A.append(comp0["loss_A"])
        n += labels.size(0)

        if (step + 1) % log_every == 0:
            log.info(
                f"  step {step + 1}/{len(loader)}  "
                f"loss={np.mean(losses[-log_every:]):.4f}  "
                f"clean={np.mean(main_losses[-log_every:]):.4f}  "
                f"adv={np.mean(adv_losses[-log_every:]):.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"({(time.time() - t0) / (step + 1):.2f}s/step)"
            )

    return {
        "n": n,
        "loss_total": float(np.mean(losses)),
        "loss_main": float(np.mean(main_losses)),
        "loss_adv": float(np.mean(adv_losses)),
        "loss_S": float(np.mean(aux_S)),
        "loss_L": float(np.mean(aux_L)),
        "loss_A": float(np.mean(aux_A)),
        "elapsed": time.time() - t0,
    }
