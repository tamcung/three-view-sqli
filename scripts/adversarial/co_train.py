#!/usr/bin/env python3
"""Co-evolutionary adversarial training (§4.4).

For R rounds, alternate:
    1. Generate adversarial samples with one or more attackers against the
       current model checkpoint.
    2. Merge the new adversarials with the original training set, plus an
       equal number of benign mimics to keep the 50/50 balance.
    3. Fine-tune the model for E_r epochs on the merged set.
    4. Re-evaluate on the clean test set + a held-out adversarial set
       (different seeds / attackers reserved for evaluation).

Round 0 is the seed checkpoint (the model trained on tamper-augmented data).
After round R, we have a robustified checkpoint and the per-round trace.

Usage:
    python -m scripts.adversarial.co_train \
        --seed-checkpoint results/tri_view_stage_aug/best_checkpoint.pt \
        --output results/cotrain_v1/ \
        --rounds 3 \
        --attackers search hotflip \
        --seeds-per-round 400 \
        --epochs-per-round 2

This script orchestrates calls to the existing attacker scripts (so the
attack code stays in one place) and reuses train.py's helpers via direct
import.
"""
from __future__ import annotations
import argparse
import json
import logging
import shutil
import subprocess
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import SamplePreprocessor                       # noqa: E402
from dataset import SQLDataset, collate_three_view                 # noqa: E402
from ablation_models import build_model                            # noqa: E402

# Reuse train.py utilities — they're stable and battle-tested
from train import (                                                 # noqa: E402
    train_one_epoch, evaluate, save_ckpt, set_seed,
    linear_warmup_cosine, compute_binary_metrics,
)

from scripts.adversarial.utils import (                            # noqa: E402
    load_victim, batch_predict, setup_logger,
)
from scripts.adversarial.freelb import (                            # noqa: E402
    FreeLBConfig, train_one_epoch_freelb,
)


# ============================================================
# Adversarial generation (delegates to attacker scripts)
# ============================================================
def run_search_attacker(checkpoint, output_jsonl, n_seeds, seed,
                          pop_size=24, generations=12,
                          seed_split=None, log=None):
    cmd = [
        sys.executable, "-m", "scripts.adversarial.search_attacker",
        "--checkpoint", str(checkpoint),
        "--output", str(output_jsonl),
        "--n-seeds", str(n_seeds),
        "--pop-size", str(pop_size),
        "--generations", str(generations),
        "--seed", str(seed),
        "--limit-seeds-already-broken",
    ]
    if seed_split:
        cmd += ["--seed-split", str(seed_split)]
    if log:
        log.info(f"  search_attacker: {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        if log:
            log.error(proc.stderr[-2000:])
        raise RuntimeError("search_attacker failed")
    return output_jsonl


def run_wafamole_attacker(checkpoint, output_jsonl, n_seeds, seed,
                            max_rounds=50, round_size=24,
                            seed_split=None, log=None):
    cmd = [
        sys.executable, "-m", "scripts.adversarial.wafamole_attacker",
        "--checkpoint", str(checkpoint),
        "--output", str(output_jsonl),
        "--n-seeds", str(n_seeds),
        "--max-rounds", str(max_rounds),
        "--round-size", str(round_size),
        "--seed", str(seed),
        "--limit-seeds-already-broken",
    ]
    if seed_split:
        cmd += ["--seed-split", str(seed_split)]
    if log:
        log.info(f"  wafamole_attacker: {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        if log:
            log.error(proc.stderr[-2000:])
        raise RuntimeError("wafamole_attacker failed")
    return output_jsonl


def run_hotflip_attacker(checkpoint, output_jsonl, n_seeds, seed,
                           n_flips=15, top_k=48, seed_split=None, log=None):
    cmd = [
        sys.executable, "-m", "scripts.adversarial.hotflip_attacker",
        "--checkpoint", str(checkpoint),
        "--output", str(output_jsonl),
        "--n-seeds", str(n_seeds),
        "--n-flips", str(n_flips),
        "--top-k-per-iter", str(top_k),
        "--seed", str(seed),
        "--limit-seeds-already-broken",
    ]
    if seed_split:
        cmd += ["--seed-split", str(seed_split)]
    if log:
        log.info(f"  hotflip_attacker: {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        if log:
            log.error(proc.stderr[-2000:])
        raise RuntimeError("hotflip_attacker failed")
    return output_jsonl


def run_llm_attacker(checkpoint, output_jsonl, n_seeds, seed,
                      provider="anthropic", model_name="claude-sonnet-4-5",
                      variants_per_seed=6, rounds=2,
                      seed_split=None, log=None):
    cmd = [
        sys.executable, "-m", "scripts.adversarial.llm_attacker",
        "--checkpoint", str(checkpoint),
        "--output", str(output_jsonl),
        "--n-seeds", str(n_seeds),
        "--variants-per-seed", str(variants_per_seed),
        "--rounds", str(rounds),
        "--seed", str(seed),
        "--provider", provider, "--model", model_name,
        "--limit-seeds-already-broken",
        "--keep-only-best-per-seed",
    ]
    if seed_split:
        cmd += ["--seed-split", str(seed_split)]
    if log:
        log.info(f"  llm_attacker: {' '.join(cmd[1:])}")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        if log:
            log.error(proc.stderr[-2000:])
        raise RuntimeError("llm_attacker failed")
    return output_jsonl


# ============================================================
# Dataset assembly
# ============================================================
def normalize_record(rec: dict) -> dict:
    """Coerce an adversarial record into the shape SQLDataset expects."""
    return {
        "user_input": rec["user_input"],
        "label": rec.get("label", "attack"),
        "source": rec.get("source", "adv"),
        "subtype": rec.get("subtype"),
        "technique": rec.get("technique"),
        "id": rec.get("id") or rec.get("seed_id"),
    }


def build_round_train_set(
    base_train_path: Path,
    adv_jsonls: list[Path],
    out_path: Path,
    seed: int = 42,
    cap_adv: int | None = None,
) -> dict:
    """Concatenate base train + all adversarial files. Optional cap on adv
    count per attacker. Returns counts dict."""
    import random
    rng = random.Random(seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"base": 0, "adv": {}}
    seen = set()
    with open(out_path, "w", encoding="utf-8") as fout:
        # Base
        with open(base_train_path, encoding="utf-8") as fin:
            for line in fin:
                fout.write(line)
                counts["base"] += 1
        # Adv
        for adv_path in adv_jsonls:
            if not adv_path.exists():
                continue
            with open(adv_path, encoding="utf-8") as fin:
                rows = [json.loads(l) for l in fin]
            rng.shuffle(rows)
            if cap_adv is not None:
                rows = rows[:cap_adv]
            kept = 0
            for r in rows:
                key = r["user_input"]
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps(normalize_record(r),
                                        ensure_ascii=False) + "\n")
                kept += 1
            counts["adv"][adv_path.name] = kept
    return counts


# ============================================================
# Held-out adversarial eval set (built once, reused each round)
# ============================================================
def build_holdout_adv(seed_checkpoint, out_path, n_seeds=200,
                      pop_size=20, generations=8, seed=999,
                      seed_split=None, log=None):
    """Use search attacker once on the seed checkpoint to make a frozen
    held-out adversarial test set."""
    if out_path.exists():
        if log:
            log.info(f"  Held-out adv set already exists: {out_path}")
        return out_path
    if log:
        log.info(f"  Building held-out adv set ({n_seeds} seeds, seed={seed})")
    run_search_attacker(
        seed_checkpoint, out_path, n_seeds=n_seeds, seed=seed,
        pop_size=pop_size, generations=generations,
        seed_split=seed_split, log=log,
    )
    return out_path


@torch.no_grad()
def eval_on_jsonl(model, pre, device, accepted, jsonl_path, threshold=0.5):
    """Returns (n, recall) — fraction of attacks the model still flags."""
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r.get("label", "attack") == "attack":
                rows.append(r["user_input"])
    if not rows:
        return 0, 0.0
    probs = batch_predict(model, pre, device, accepted, rows, batch_size=128)
    preds = (probs >= threshold).astype(int)
    return len(rows), float(preds.mean())


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True,
                    help="Output dir; round subdirs will be created here.")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--attackers", nargs="+",
                    choices=["search", "wafamole", "hotflip", "llm"],
                    default=["wafamole"])
    p.add_argument("--seeds-per-round", type=int, default=400)
    p.add_argument("--cap-adv-per-attacker", type=int, default=2000)
    p.add_argument("--epochs-per-round", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4,
                    help="Fine-tuning LR (lower than initial training).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--llm-provider", default="anthropic")
    p.add_argument("--llm-model", default="claude-sonnet-4-5")
    p.add_argument("--llm-variants", type=int, default=4)
    p.add_argument("--holdout-adv-n", type=int, default=200)

    # FreeLB embedding-level adversarial training
    p.add_argument("--freelb", action="store_true",
                    help="Use FreeLB embedding-space PGD during fine-tuning. "
                         "Requires the model to support surface_inputs_embeds= "
                         "(BPECharLexStageModel does).")
    p.add_argument("--freelb-steps", type=int, default=3,
                    help="K — number of PGD steps per batch.")
    p.add_argument("--freelb-init-norm", type=float, default=0.05)
    p.add_argument("--freelb-step-size", type=float, default=1e-2)
    p.add_argument("--freelb-max-norm", type=float, default=0.2)
    p.add_argument("--freelb-adv-weight", type=float, default=1.0)
    p.add_argument("--attack-seed-split", type=str,
                    default=str(ROOT / "data" / "splits" / "train.jsonl"),
                    help="Seed split for ATTACK generation (default: train, "
                         "so test.jsonl stays held-out for clean F1).")
    p.add_argument("--holdout-attack-seed-split", type=str,
                    default=str(ROOT / "data" / "splits" / "test.jsonl"),
                    help="Seed split for the FROZEN held-out adv eval set "
                         "(default: test, scored once against round-0 model).")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger("cotrain", out_dir / "cotrain.log")
    set_seed(args.seed)

    log.info(f"\n{'='*70}\n  Co-evolutionary adversarial training\n{'='*70}")
    log.info(f"  seed_checkpoint = {args.seed_checkpoint}")
    log.info(f"  rounds          = {args.rounds}")
    log.info(f"  attackers       = {args.attackers}")
    log.info(f"  seeds_per_round = {args.seeds_per_round}")
    log.info(f"  epochs_per_round= {args.epochs_per_round}")
    log.info(f"  freelb          = {args.freelb}"
              + (f" (K={args.freelb_steps}, ε={args.freelb_max_norm})"
                  if args.freelb else ""))

    freelb_cfg = (FreeLBConfig(
        n_steps=args.freelb_steps,
        init_norm=args.freelb_init_norm,
        step_size=args.freelb_step_size,
        max_norm=args.freelb_max_norm,
        adv_loss_weight=args.freelb_adv_weight,
    ) if args.freelb else None)

    # ---- Locate config + base train ----
    seed_ckpt = Path(args.seed_checkpoint)
    src_cfg_path = seed_ckpt.parent / "config.yaml"
    cfg = yaml.safe_load(open(src_cfg_path, encoding="utf-8"))
    base_train_path = ROOT / "data" / "splits" / "train.jsonl"
    val_path = ROOT / "data" / "splits" / "val.jsonl"
    test_path = ROOT / "data" / "splits" / "test.jsonl"

    # Persist resolved config + cotrain args
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump({**cfg, "cotrain": vars(args)}, f, sort_keys=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  device = {device}")

    # ---- Build held-out adversarial eval set against the seed checkpoint ----
    holdout_dir = out_dir / "holdout_adv"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    holdout_adv_path = holdout_dir / "search_holdout.jsonl"
    build_holdout_adv(
        seed_ckpt, holdout_adv_path,
        n_seeds=args.holdout_adv_n,
        seed=args.seed + 1000,
        seed_split=args.holdout_attack_seed_split,
        log=log,
    )

    # ---- Round-0 baseline eval ----
    log.info(f"\n--- Round 0 (seed checkpoint baseline) ---")
    model, pre, device, variant, accepted = load_victim(seed_ckpt, device=device.type)
    val_metrics = _eval_clean_jsonl(model, pre, device, accepted, val_path)
    test_metrics = _eval_clean_jsonl(model, pre, device, accepted, test_path)
    n_h, holdout_rec = eval_on_jsonl(model, pre, device, accepted, holdout_adv_path)
    log.info(f"  val_f1={val_metrics['f1']:.4f}  test_f1={test_metrics['f1']:.4f}  "
              f"holdout_adv_recall={holdout_rec:.4f}  ({n_h} samples)")
    round_log = [{
        "round": 0,
        "checkpoint": str(seed_ckpt),
        "val": val_metrics, "test": test_metrics,
        "holdout_adv_n": n_h,
        "holdout_adv_recall": holdout_rec,
    }]

    cur_ckpt = seed_ckpt

    # ---- Co-evolution loop ----
    for r in range(1, args.rounds + 1):
        round_dir = out_dir / f"round_{r}"
        round_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"\n{'='*60}\n  Round {r}/{args.rounds}\n{'='*60}")

        # ----- 1. attack the current model -----
        adv_jsonls = []
        attack_seed = args.seed + r * 100
        if "search" in args.attackers:
            ap = round_dir / "adv_search.jsonl"
            run_search_attacker(cur_ckpt, ap,
                                 n_seeds=args.seeds_per_round,
                                 seed=attack_seed,
                                 seed_split=args.attack_seed_split,
                                 log=log)
            adv_jsonls.append(ap)
        if "wafamole" in args.attackers:
            ap = round_dir / "adv_wafamole.jsonl"
            run_wafamole_attacker(cur_ckpt, ap,
                                    n_seeds=args.seeds_per_round,
                                    seed=attack_seed,
                                    seed_split=args.attack_seed_split,
                                    log=log)
            adv_jsonls.append(ap)
        if "hotflip" in args.attackers:
            ap = round_dir / "adv_hotflip.jsonl"
            run_hotflip_attacker(cur_ckpt, ap,
                                  n_seeds=args.seeds_per_round // 2,
                                  seed=attack_seed + 1,
                                  seed_split=args.attack_seed_split,
                                  log=log)
            adv_jsonls.append(ap)
        if "llm" in args.attackers:
            ap = round_dir / "adv_llm.jsonl"
            run_llm_attacker(cur_ckpt, ap,
                              n_seeds=args.seeds_per_round // 4,
                              seed=attack_seed + 2,
                              provider=args.llm_provider,
                              model_name=args.llm_model,
                              variants_per_seed=args.llm_variants,
                              seed_split=args.attack_seed_split,
                              log=log)
            adv_jsonls.append(ap)

        # Quick stats
        n_total_adv = 0
        for ap in adv_jsonls:
            if ap.exists():
                with open(ap, encoding="utf-8") as f:
                    n = sum(1 for _ in f)
                log.info(f"    {ap.name}: {n} adv samples")
                n_total_adv += n
        log.info(f"  Total adv this round: {n_total_adv}")

        # ----- 2. assemble fine-tune training set -----
        merged_path = round_dir / "train_merged.jsonl"
        counts = build_round_train_set(
            base_train_path, adv_jsonls, merged_path,
            seed=args.seed + r, cap_adv=args.cap_adv_per_attacker,
        )
        log.info(f"  merged train: {counts}")

        # ----- 3. fine-tune -----
        # Free the old model + recreate from the current checkpoint, with a
        # fresh optimizer so the lr schedule restarts.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model = build_model(variant, cfg.get("model", {})).to(device)
        ckpt = torch.load(cur_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

        bs = cfg.get("batch_size", 32)
        nw = cfg.get("num_workers", 2)
        # NEW: build dataset from merged file with no cache (so the new
        # adversarials are picked up). Cache the merged set for this round only.
        round_cache = round_dir / "train_merged.pkl"
        if round_cache.exists():
            round_cache.unlink()
        train_ds = SQLDataset(merged_path, cache_path=round_cache,
                                preprocessor=pre)
        val_ds = SQLDataset(val_path,
                              cache_path=ROOT / "data" / "cache" / "val.pkl",
                              preprocessor=pre)
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

        optimizer = AdamW(model.parameters(), lr=args.lr,
                            weight_decay=float(cfg.get("weight_decay", 0.01)),
                            betas=(0.9, 0.98))
        total_steps = args.epochs_per_round * len(train_loader)
        warmup_steps = int(total_steps * 0.05)
        scheduler = linear_warmup_cosine(optimizer, total_steps, warmup_steps)

        amp_mode = cfg.get("amp", "bf16")
        scaler = torch.cuda.amp.GradScaler() if (amp_mode == "fp16" and
                                                   device.type == "cuda") else None
        view_dropout_prob = float(cfg.get("view_dropout", 0.1))

        log.info(f"  fine-tuning {args.epochs_per_round} epochs, "
                  f"|train|={len(train_ds)} |val|={len(val_ds)}  lr={args.lr}"
                  + ("  [FreeLB]" if freelb_cfg is not None else ""))
        for epoch in range(args.epochs_per_round):
            log.info(f"\n  --- round {r} epoch {epoch + 1}/{args.epochs_per_round} ---")
            t0 = time.time()
            if freelb_cfg is not None:
                train_stats = train_one_epoch_freelb(
                    model, train_loader, optimizer, scheduler, scaler, device,
                    view_dropout_prob=view_dropout_prob,
                    log_every=cfg.get("log_every_steps", 100), log=log,
                    amp_mode=amp_mode, freelb_cfg=freelb_cfg,
                )
            else:
                train_stats = train_one_epoch(
                    model, train_loader, optimizer, scheduler, scaler, device,
                    view_dropout_prob=view_dropout_prob,
                    log_every=cfg.get("log_every_steps", 100), log=log,
                    amp_mode=amp_mode,
                )
            log.info(f"  train: loss={train_stats['loss_total']:.4f}  "
                      f"main={train_stats['loss_main']:.4f}  "
                      f"({train_stats['elapsed']:.0f}s)")
            val_metrics_e, _, _, _ = evaluate(model, val_loader, device, scaler,
                                                amp_mode=amp_mode)
            log.info(f"  val:   f1={val_metrics_e['f1']:.4f}  "
                      f"R={val_metrics_e['recall']:.4f}  "
                      f"P={val_metrics_e['precision']:.4f}")

        # ----- 4. save round checkpoint + eval -----
        round_ckpt = round_dir / "best_checkpoint.pt"
        save_ckpt(model, optimizer, scheduler,
                   args.epochs_per_round - 1,
                   val_metrics_e["f1"], round_ckpt)
        # also persist round config for downstream eval scripts
        with open(round_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False)

        # ----- 5. evaluate on clean test + holdout adv -----
        test_metrics = _eval_clean_jsonl(model, pre, device, accepted, test_path)
        n_h, holdout_rec = eval_on_jsonl(
            model, pre, device, accepted, holdout_adv_path,
        )
        log.info(f"\n  Round {r} summary:  test_f1={test_metrics['f1']:.4f}  "
                  f"holdout_adv_recall={holdout_rec:.4f}")
        round_log.append({
            "round": r,
            "checkpoint": str(round_ckpt),
            "val": val_metrics_e, "test": test_metrics,
            "n_total_adv_added": n_total_adv,
            "holdout_adv_n": n_h,
            "holdout_adv_recall": holdout_rec,
            "merged_counts": counts,
        })

        # next round attacks the new checkpoint
        cur_ckpt = round_ckpt

        # Persist trace each round in case we crash
        with open(out_dir / "rounds.json", "w", encoding="utf-8") as f:
            json.dump(round_log, f, indent=2)

    log.info(f"\n{'='*70}\n  DONE — final ckpt: {cur_ckpt}\n{'='*70}")
    log.info(f"  rounds.json written to {out_dir / 'rounds.json'}")


def _eval_clean_jsonl(model, pre, device, accepted, jsonl_path,
                        threshold=0.5):
    """Compute F1/P/R on a jsonl by streaming through batch_predict."""
    rows = []
    labels = []
    with open(jsonl_path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            rows.append(r["user_input"])
            labels.append(1 if r["label"] == "attack" else 0)
    probs = batch_predict(model, pre, device, accepted, rows, batch_size=128)
    return compute_binary_metrics(np.array(labels),
                                    np.log(probs / (1 - probs + 1e-12)),
                                    threshold=threshold)


if __name__ == "__main__":
    main()
