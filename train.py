#!/usr/bin/env python3
"""Three-view SQL injection detection — training entrypoint.

Usage:
  python train.py --config configs/main.yaml --output results/run_001/

Loads splits from data/splits/{train,val,test}.jsonl, preprocesses (cached
under data/cache/), trains the three-view model with deep supervision and
view dropout, validates each epoch, saves best/final checkpoints, and runs
final test-set evaluation.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import SamplePreprocessor
from dataset import SQLDataset, collate_three_view, move_batch_to
from model import ThreeViewModel, count_parameters
from ablation_models import build_model


# ============================================================
# Setup helpers
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="YAML config path")
    p.add_argument("--output", type=str, required=True, help="Output dir")
    p.add_argument("--resume", type=str, default=None,
                    help="Optional checkpoint to resume from")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_warmup_cosine(opt, total_steps: int, warmup_steps: int):
    """Linear warmup, then cosine decay to 0."""
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return LambdaLR(opt, fn)


# ============================================================
# Metrics
# ============================================================
def compute_binary_metrics(y_true: np.ndarray, y_logits: np.ndarray, threshold: float = 0.5) -> dict:
    """Returns precision, recall, F1, accuracy, AUC for binary task."""
    from sklearn.metrics import (
        precision_recall_fscore_support, accuracy_score, roc_auc_score,
        confusion_matrix,
    )
    probs = 1.0 / (1.0 + np.exp(-y_logits))
    preds = (probs >= threshold).astype(int)
    P, R, F1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0,
    )
    acc = accuracy_score(y_true, preds)
    try:
        auc = roc_auc_score(y_true, probs)
    except ValueError:
        auc = float("nan")
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "precision": float(P),
        "recall": float(R),
        "f1": float(F1),
        "accuracy": float(acc),
        "auc": float(auc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ============================================================
# Train / eval loops
# ============================================================
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                     view_dropout_prob: float, log_every: int, log,
                     amp_mode: str = "bf16") -> dict:
    model.train()
    losses = []
    main_losses = []
    aux_S, aux_L, aux_A = [], [], []
    n = 0
    t0 = time.time()
    for step, batch in enumerate(loader):
        batch = move_batch_to(batch, device)
        labels = batch["labels"]

        optimizer.zero_grad(set_to_none=True)
        # amp_mode is set by main(); here we just pick dtype + enabled flag
        if amp_mode == "fp32":
            amp_dtype = torch.float32
            amp_enabled = False
        elif amp_mode == "fp16":
            amp_dtype = torch.float16
            amp_enabled = device.type == "cuda"
        else:  # bf16 default
            amp_dtype = torch.bfloat16
            amp_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            kwargs = dict(
                surface_ids=batch["surface_ids"],
                surface_mask=batch["surface_mask"],
                lex_ids=batch["lex_ids"],
                lex_mask=batch["lex_mask"],
                ast_ids=batch["ast_ids"],
                ast_mask=batch["ast_mask"],
                ast_valid=batch["ast_valid"],
                view_dropout_prob=view_dropout_prob,
                ast_node_ids=batch.get("ast_node_ids"),
                ast_parent=batch.get("ast_parent"),
                char_ids=batch.get("char_ids"),
                char_mask=batch.get("char_mask"),
            )
            import inspect
            accepted = set(inspect.signature(model.forward).parameters.keys())
            out = model(**{k: v for k, v in kwargs.items() if k in accepted})
            loss, components = model.compute_loss(out, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        losses.append(components["loss_total"])
        main_losses.append(components["loss_main"])
        aux_S.append(components["loss_S"])
        aux_L.append(components["loss_L"])
        aux_A.append(components["loss_A"])
        n += labels.size(0)

        if (step + 1) % log_every == 0:
            log.info(
                f"  step {step + 1}/{len(loader)}  "
                f"loss={np.mean(losses[-log_every:]):.4f}  "
                f"main={np.mean(main_losses[-log_every:]):.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}  "
                f"({(time.time()-t0)/(step+1):.2f}s/step)"
            )

    return {
        "n": n,
        "loss_total": float(np.mean(losses)),
        "loss_main": float(np.mean(main_losses)),
        "loss_S": float(np.mean(aux_S)),
        "loss_L": float(np.mean(aux_L)),
        "loss_A": float(np.mean(aux_A)),
        "elapsed": time.time() - t0,
    }


@torch.no_grad()
def evaluate(model, loader, device, scaler, amp_mode: str = "bf16") -> tuple[dict, np.ndarray, np.ndarray, list[dict]]:
    """Run model on loader and return (metrics_dict, logits, labels, meta_list)."""
    import inspect
    model.eval()
    all_logits = []
    all_labels = []
    all_meta = []
    if amp_mode == "fp32":
        amp_dtype = torch.float32
        amp_enabled = False
    elif amp_mode == "fp16":
        amp_dtype = torch.float16
        amp_enabled = device.type == "cuda"
    else:
        amp_dtype = torch.bfloat16
        amp_enabled = device.type == "cuda"
    accepted = set(inspect.signature(model.forward).parameters.keys())
    for batch in loader:
        batch = move_batch_to(batch, device)
        kwargs = dict(
            surface_ids=batch["surface_ids"],
            surface_mask=batch["surface_mask"],
            lex_ids=batch["lex_ids"],
            lex_mask=batch["lex_mask"],
            ast_ids=batch["ast_ids"],
            ast_mask=batch["ast_mask"],
            ast_valid=batch["ast_valid"],
            view_dropout_prob=0.0,
            ast_node_ids=batch.get("ast_node_ids"),
            ast_parent=batch.get("ast_parent"),
            char_ids=batch.get("char_ids"),
            char_mask=batch.get("char_mask"),
        )
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(**{k: v for k, v in kwargs.items() if k in accepted})
        all_logits.append(out["p_main"].float().cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
        all_meta.extend(batch["meta"])
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    metrics = compute_binary_metrics(labels, logits)
    return metrics, logits, labels, all_meta


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    out_dir = Path(args.output)
    log = setup_logger(out_dir)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    log.info(f"Config:\n{yaml.dump(cfg, sort_keys=False)}")
    # Persist resolved config
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}  CUDA: {torch.cuda.is_available()}  "
             f"BF16: {torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False}")

    # ---- Datasets ----
    pre = SamplePreprocessor()
    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading datasets...")
    train_ds = SQLDataset(
        ROOT / "data" / "splits" / "train.jsonl",
        cache_dir / "train.pkl", pre,
        max_samples=cfg.get("max_train_samples"),
    )
    val_ds = SQLDataset(
        ROOT / "data" / "splits" / "val.jsonl",
        cache_dir / "val.pkl", pre,
    )
    test_ds = SQLDataset(
        ROOT / "data" / "splits" / "test.jsonl",
        cache_dir / "test.pkl", pre,
    )
    log.info(f"  train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    bs = cfg.get("batch_size", 64)
    nw = cfg.get("num_workers", 2)
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        collate_fn=collate_three_view, num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs * 2, shuffle=False,
        collate_fn=collate_three_view, num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs * 2, shuffle=False,
        collate_fn=collate_three_view, num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0),
    )

    # ---- Model ----
    model_cfg = cfg.get("model", {})
    variant = cfg.get("model_variant", "three_view")
    model = build_model(variant, model_cfg).to(device)
    log.info(f"Model variant: {variant}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {n_params / 1e6:.2f} M")

    # ---- Optimizer / scheduler ----
    epochs = cfg.get("epochs", 5)
    lr = float(cfg.get("lr", 2e-4))
    wd = float(cfg.get("weight_decay", 0.01))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * cfg.get("warmup_frac", 0.05))
    scheduler = linear_warmup_cosine(optimizer, total_steps, warmup_steps)

    # ---- Mixed precision ----
    amp_mode = cfg.get("amp", "bf16")
    use_fp16 = amp_mode == "fp16"
    scaler = torch.cuda.amp.GradScaler() if (use_fp16 and device.type == "cuda") else None

    # ---- Resume ----
    start_epoch = 0
    best_val_f1 = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        log.info(f"Resumed from {args.resume}: epoch {start_epoch}, best_val_f1={best_val_f1:.4f}")

    # ---- Train ----
    metrics_per_epoch = []
    log_every = cfg.get("log_every_steps", 50)
    view_dropout_prob = float(cfg.get("view_dropout", 0.1))

    for epoch in range(start_epoch, epochs):
        log.info(f"\n=== Epoch {epoch + 1}/{epochs} ===")
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            view_dropout_prob=view_dropout_prob,
            log_every=log_every, log=log,
            amp_mode=amp_mode,
        )
        log.info(
            f"Train  loss_total={train_stats['loss_total']:.4f} "
            f"main={train_stats['loss_main']:.4f} "
            f"S={train_stats['loss_S']:.4f} "
            f"L={train_stats['loss_L']:.4f} "
            f"A={train_stats['loss_A']:.4f}  "
            f"({train_stats['elapsed']:.1f}s)"
        )

        val_metrics, _, _, _ = evaluate(model, val_loader, device, scaler, amp_mode=amp_mode)
        log.info(
            f"Val    f1={val_metrics['f1']:.4f} "
            f"P={val_metrics['precision']:.4f} R={val_metrics['recall']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} auc={val_metrics['auc']:.4f}  "
            f"(tp={val_metrics['tp']} fn={val_metrics['fn']} fp={val_metrics['fp']} tn={val_metrics['tn']})"
        )

        metrics_per_epoch.append({
            "epoch": epoch + 1,
            "train": train_stats,
            "val": val_metrics,
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            save_ckpt(model, optimizer, scheduler, epoch, best_val_f1,
                       out_dir / "best_checkpoint.pt")
            log.info(f"  → new best val_f1={best_val_f1:.4f}")

    # Final checkpoint
    save_ckpt(model, optimizer, scheduler, epochs - 1, best_val_f1,
               out_dir / "final_checkpoint.pt")

    # Test eval (load best)
    log.info("\n=== Final test evaluation (best checkpoint) ===")
    best = torch.load(out_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics, test_logits, test_labels, test_meta = evaluate(
        model, test_loader, device, scaler, amp_mode=amp_mode,
    )
    log.info(
        f"Test   f1={test_metrics['f1']:.4f} "
        f"P={test_metrics['precision']:.4f} R={test_metrics['recall']:.4f} "
        f"acc={test_metrics['accuracy']:.4f} auc={test_metrics['auc']:.4f}"
    )

    np.savez(out_dir / "test_predictions.npz",
              logits=test_logits, labels=test_labels)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "epochs": metrics_per_epoch,
            "final_test": test_metrics,
            "best_val_f1": best_val_f1,
            "config": cfg,
        }, f, indent=2)
    log.info(f"Wrote metrics.json. Done.")


def save_ckpt(model, opt, sched, epoch, best_f1, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "epoch": epoch,
        "best_val_f1": best_f1,
    }, path)


if __name__ == "__main__":
    main()
