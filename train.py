#!/usr/bin/env python3
"""
Production training entrypoint.

Usage:
    python train.py --config configs/medium.yaml --output results/run_001/

Outputs (in --output):
    metrics.json          per-epoch + final F1/Loss/Wilson CI
    train.log             stdout/stderr log
    config.yaml           copy of the resolved config
    best_checkpoint.pt    best val main F1 checkpoint
    final_checkpoint.pt   last epoch checkpoint
"""
from __future__ import annotations
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import SamplePreprocessor
from src.dataset import (
    build_split_files, preprocess_split_file,
    WafamoleThreeViewDataset, collate_three_view,
)
from src.model import ThreeViewModel, count_parameters


# ============================================================
# Metrics with Wilson 95% CI
# ============================================================
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def binary_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    preds = (scores > 0).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    n = len(labels)
    correct = tp + tn
    acc = correct / n
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    acc_ci = wilson_ci(correct, n)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n": n, "acc_ci_low": acc_ci[0], "acc_ci_high": acc_ci[1]}


# ============================================================
# Eval loop (returns per-view metrics)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, use_bf16: bool) -> dict:
    model.eval()
    logits = {k: [] for k in ("main", "S", "L", "A")}
    labels_all = []
    s1_all = []
    autocast_kw = {"dtype": torch.bfloat16, "enabled": use_bf16}
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", **autocast_kw):
            out = model(
                batch["surface_ids"], batch["surface_mask"],
                batch["lex_ids"], batch["lex_mask"],
                batch["ast_ids"], batch["ast_mask"],
                batch["ast_valid"],
                view_dropout_prob=0.0,
            )
        logits["main"].append(out["p_main"].float().cpu().numpy())
        logits["S"].append(out["p_S"].float().cpu().numpy())
        logits["L"].append(out["p_L"].float().cpu().numpy())
        logits["A"].append(out["p_A"].float().cpu().numpy())
        labels_all.append(batch["label"].cpu().numpy())
        s1_all.append(out["stage1_weights"].float().cpu().numpy())
    y = np.concatenate(labels_all)
    metrics = {k: binary_metrics(np.concatenate(v), y) for k, v in logits.items()}
    s1 = np.concatenate(s1_all, axis=0)
    metrics["stage1_attn"] = {
        "L_to_L": float(s1[:, 0, 0].mean()),
        "L_to_A": float(s1[:, 0, 1].mean()),
        "A_to_L": float(s1[:, 1, 0].mean()),
        "A_to_A": float(s1[:, 1, 1].mean()),
    }
    return metrics


# ============================================================
# Main training loop
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--resume", type=Path, default=None,
                    help="Resume from a checkpoint (.pt path)")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())

    # Save resolved config
    (args.output / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Tee log
    log_path = args.output / "train.log"
    log_f = open(log_path, "a", encoding="utf-8")
    def log(*objects, sep=" "):
        msg = sep.join(str(o) for o in objects)
        print(msg, flush=True)
        log_f.write(msg + "\n"); log_f.flush()

    log(f"=== {cfg['experiment_name']} ===")
    log(f"Config: {args.config}")
    log(f"Output: {args.output}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(cfg["train"].get("use_bf16", False)) and device.type == "cuda" and torch.cuda.is_bf16_supported()
    log(f"Device: {device}  BF16: {use_bf16}")

    # ---- Data ----
    pre = SamplePreprocessor()
    split_paths = build_split_files(
        n_train_per_class=cfg["n_train_per_class"],
        n_val_per_class=cfg["n_val_per_class"],
        n_test_per_class=cfg["n_test_per_class"],
        seed=cfg["seed"],
    )
    cache_paths = {s: preprocess_split_file(p, pre) for s, p in split_paths.items()}
    train_ds = WafamoleThreeViewDataset(cache_paths["train"])
    val_ds = WafamoleThreeViewDataset(cache_paths["val"])
    test_ds = WafamoleThreeViewDataset(cache_paths["test"])
    log(f"Data: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    bs = cfg["train"]["batch_size"]
    nw = cfg["train"].get("num_workers", 0)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_three_view, num_workers=nw,
                              pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            collate_fn=collate_three_view, num_workers=nw,
                            pin_memory=device.type == "cuda")
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             collate_fn=collate_three_view, num_workers=nw,
                             pin_memory=device.type == "cuda")

    # ---- Model ----
    model = ThreeViewModel(**cfg["model"]).to(device)
    params = count_parameters(model)
    log(f"Params: total={params['total']/1e6:.2f}M  surf={params['surface']/1e6:.2f}M  "
        f"lex={params['lexical']/1e6:.2f}M  ast={params['ast']/1e6:.2f}M  "
        f"fusion={params['fusion_and_heads']/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=float(cfg["train"]["lr"]),
                                   weight_decay=float(cfg["train"]["weight_decay"]))

    # Linear warmup, cosine decay (simple scheduler)
    total_steps = cfg["train"]["epochs"] * math.ceil(len(train_ds) / bs)
    warmup = cfg["train"].get("warmup_steps", 0)
    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Optional resume
    start_epoch = 1
    best_val_f1 = -1.0
    if args.resume is not None and args.resume.exists():
        log(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", -1.0)

    autocast_kw = {"dtype": torch.bfloat16, "enabled": use_bf16}
    history = []

    # ---- Train ----
    global_step = (start_epoch - 1) * math.ceil(len(train_ds) / bs)
    weights = tuple(cfg["train"]["loss_weights"])
    grad_clip = cfg["train"]["grad_clip"]
    view_dp = cfg["train"]["view_dropout_prob"]
    log_every = cfg.get("log_every", 50)

    overall_t0 = time.time()
    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        model.train()
        epoch_t0 = time.time()
        running = []

        for step_in_epoch, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", **autocast_kw):
                out = model(
                    batch["surface_ids"], batch["surface_mask"],
                    batch["lex_ids"], batch["lex_mask"],
                    batch["ast_ids"], batch["ast_mask"],
                    batch["ast_valid"],
                    view_dropout_prob=view_dp,
                )
                loss, comps = model.compute_loss(out, batch["label"], weights)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            running.append(comps["loss_total"])
            global_step += 1

            if step_in_epoch % log_every == 0:
                lr_cur = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - epoch_t0
                log(f"  ep {epoch:>2d} step {step_in_epoch:>5d}/{len(train_loader)}  "
                    f"loss={np.mean(running[-log_every:]):.4f}  lr={lr_cur:.2e}  "
                    f"elapsed={elapsed:.0f}s")

        epoch_time = time.time() - epoch_t0
        log(f"\n--- Epoch {epoch} done ({epoch_time:.0f}s) ---")
        log(f"  train mean loss: {np.mean(running):.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device, use_bf16)
        log(f"  val: main F1={val_metrics['main']['f1']:.4f}  P={val_metrics['main']['precision']:.4f}  "
            f"R={val_metrics['main']['recall']:.4f}  Acc={val_metrics['main']['accuracy']:.4f} "
            f"[CI {val_metrics['main']['acc_ci_low']:.4f}-{val_metrics['main']['acc_ci_high']:.4f}]")
        log(f"  aux: S={val_metrics['S']['f1']:.4f}  L={val_metrics['L']['f1']:.4f}  A={val_metrics['A']['f1']:.4f}")
        log(f"  stage1 attn: L->A={val_metrics['stage1_attn']['L_to_A']:.3f}  "
            f"A->L={val_metrics['stage1_attn']['A_to_L']:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(running)),
            "val": val_metrics,
            "epoch_time_sec": epoch_time,
        })
        # Save metrics after each epoch
        (args.output / "metrics.json").write_text(json.dumps(history, indent=2))

        # Checkpoint
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_f1": best_val_f1,
            "config": cfg,
        }
        torch.save(ckpt, args.output / "final_checkpoint.pt")
        if val_metrics["main"]["f1"] > best_val_f1:
            best_val_f1 = val_metrics["main"]["f1"]
            ckpt["best_val_f1"] = best_val_f1
            torch.save(ckpt, args.output / "best_checkpoint.pt")
            log(f"  ✓ new best val F1 = {best_val_f1:.4f}, saved best_checkpoint.pt")

    overall_time = time.time() - overall_t0
    log(f"\n=== Training done in {overall_time/60:.1f} min ===")
    log(f"Best val main F1: {best_val_f1:.4f}")

    # Final test set evaluation (with best checkpoint)
    log("\n=== Final test set evaluation (best checkpoint) ===")
    best_ckpt = torch.load(args.output / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(best_ckpt["model"])
    test_metrics = evaluate(model, test_loader, device, use_bf16)
    for view in ("main", "S", "L", "A"):
        m = test_metrics[view]
        log(f"  {view:6s} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
            f"Acc={m['accuracy']:.4f}  [CI {m['acc_ci_low']:.4f}-{m['acc_ci_high']:.4f}]")
    log(f"  stage1 attn: L->A={test_metrics['stage1_attn']['L_to_A']:.3f}  "
        f"A->L={test_metrics['stage1_attn']['A_to_L']:.3f}")

    # Save final metrics
    (args.output / "metrics.json").write_text(json.dumps({
        "history": history,
        "best_val_f1": best_val_f1,
        "test": test_metrics,
        "wall_time_sec": overall_time,
    }, indent=2))
    log_f.close()


if __name__ == "__main__":
    main()
