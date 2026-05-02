#!/usr/bin/env python3
"""Chapter 4 adversarial training driver.

Supports four strategies via ``--strategy``:

  - clean:    standard training (= baseline ch3 reproduction).
  - aug:      mix adversarial samples (from ``--adv-jsonl``) into the train set.
  - freelb:   embedding-perturbation training (FreeLB) on clean train set.
  - combined: aug + freelb (Chapter 4 full method).

Usage:
    python train_adv.py \
        --config configs/ch4_combined.yaml \
        --output results_adv/ch4_combined \
        --strategy combined \
        --adv-jsonl results/ch4/adv_round0.jsonl \
        --train-jsonl data/external/kaggle_sqli/jsonl/train.jsonl \
        --val-jsonl   data/external/kaggle_sqli/jsonl/val.jsonl \
        --test-jsonl  data/external/kaggle_sqli/jsonl/test.jsonl
"""
from __future__ import annotations
import argparse, inspect, json, logging, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import SamplePreprocessor
from dataset import SQLDataset, collate_three_view, move_batch_to
from ablation_models import build_model

try:
    from mvc_preprocessor import MVCSamplePreprocessor, MVCVocab
except ImportError:
    MVCSamplePreprocessor = None
    MVCVocab = None


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--strategy", required=True,
                    choices=["clean", "aug", "freelb", "combined"])
    p.add_argument("--adv-jsonl", default=None,
                    help="JSONL of adversarial samples to mix in (aug/combined)")
    p.add_argument("--train-jsonl", default=None)
    p.add_argument("--val-jsonl", default=None)
    p.add_argument("--test-jsonl", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--init-ckpt", default=None,
                    help="Optional checkpoint path to initialise model weights "
                         "(e.g. ch3 best_checkpoint.pt for warm-starting ch4)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def setup_logger(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("train_adv")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(); sh.setFormatter(fmt); log.addHandler(sh)
    fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
    fh.setFormatter(fmt); log.addHandler(fh)
    return log


def set_seed(s: int):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def linear_warmup_cosine(opt, total, warmup):
    import math
    def fn(step):
        if step < warmup: return step / max(warmup, 1)
        prog = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    return LambdaLR(opt, fn)


# ============================================================
# Augmented dataset: concatenate adv-jsonl on top of clean train
# ============================================================
class _ConcatRecordsDataset(torch.utils.data.Dataset):
    """Wrap two SQLDataset-style record lists into one iterable."""
    def __init__(self, primary_records, extra_records):
        self.records = list(primary_records) + list(extra_records)

    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]


def build_adv_records(adv_jsonl: Path, preprocessor):
    """Preprocess adversarial JSONL into the same record format as SQLDataset."""
    out = []
    if adv_jsonl is None or not Path(adv_jsonl).exists():
        return out
    with open(adv_jsonl, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            features = preprocessor(r["user_input"])
            entry = {
                **features,
                "label_int": 1 if r.get("label") == "attack" else 0,
                "meta": {"source": r.get("source", "adv_generated"),
                          "tag": r.get("tag", ""),
                          "op_chain": r.get("op_chain", [])},
            }
            out.append(entry)
    return out


# ============================================================
# FreeLB inner-PGD step
# ============================================================
def freelb_step(model, batch, K: int, epsilon: float, alpha: float,
                lambda_adv: float, optimizer, view_dropout_prob: float):
    """One FreeLB step over a batch, accumulating gradients and stepping
    optimiser at the end. Returns dict of loss components.

    Implements §4.4 Algorithm 4.2:
      - encode three views (with grad on encoders)
      - inner loop K times:
          add δ to each view, compute loss, scale by 1/(K+1) + λ, backward
          update δ along its gradient sign, project to ε-ball
      - one extra "clean" forward (δ=0) for L_clean
      - optimizer.step()
    """
    optimizer.zero_grad(set_to_none=True)

    # ---- Stage 1: encode views ONCE (grad on encoder params) ----
    H_S, H_C, H_L, aux = model.encode_views(
        surface_ids=batch["surface_ids"], surface_mask=batch["surface_mask"],
        lex_ids=batch["lex_ids"], lex_mask=batch["lex_mask"],
        char_ids=batch.get("char_ids"), char_mask=batch.get("char_mask"),
    )
    labels = batch["labels"].float()

    # ---- Clean loss (δ=0) ----
    fused_clean = model.fuse_from_views(
        H_S, H_C, H_L,
        surface_mask=batch["surface_mask"], lex_mask=batch["lex_mask"],
        char_mask=batch.get("char_mask"), view_dropout_prob=view_dropout_prob,
    )
    p_main_clean = fused_clean["p_main"]
    loss_clean = F.binary_cross_entropy_with_logits(p_main_clean, labels)

    # ---- Initialise δ as small random tensors with requires_grad ----
    def _rand_delta(H, eps):
        d = torch.randn_like(H) * eps * 0.1
        # project to ε-ball (per-sample L2)
        norm = d.view(d.size(0), -1).norm(dim=-1).view(-1, 1, 1).clamp(min=1e-8)
        d = d * (eps / norm).clamp(max=1.0)
        return d.detach().requires_grad_(True)

    delta_S = _rand_delta(H_S, epsilon)
    delta_C = _rand_delta(H_C, epsilon)
    delta_L = _rand_delta(H_L, epsilon)

    # ---- Inner PGD loop over K steps, accumulating grad on model params ----
    adv_losses = []
    for k in range(K):
        fused = model.fuse_from_views(
            H_S + delta_S, H_C + delta_C, H_L + delta_L,
            surface_mask=batch["surface_mask"], lex_mask=batch["lex_mask"],
            char_mask=batch.get("char_mask"), view_dropout_prob=view_dropout_prob,
        )
        loss_k = F.binary_cross_entropy_with_logits(fused["p_main"], labels)
        adv_losses.append(loss_k.detach().item())

        # Compute gradient w.r.t. δ (and accumulate into model params)
        grads = torch.autograd.grad(
            loss_k * (lambda_adv / max(K, 1)),
            [delta_S, delta_C, delta_L],
            retain_graph=True, create_graph=False, allow_unused=True,
        )
        # Manually trigger model-param backward for this scaled loss
        (loss_k * (lambda_adv / max(K, 1))).backward(retain_graph=False)

        # Update δ along normalised gradient sign, project to ε-ball
        with torch.no_grad():
            for d, g in zip([delta_S, delta_C, delta_L], grads):
                if g is None: continue
                gnorm = g.view(g.size(0), -1).norm(dim=-1).view(-1, 1, 1).clamp(min=1e-8)
                d.add_(alpha * g / gnorm)
                # Project
                dnorm = d.view(d.size(0), -1).norm(dim=-1).view(-1, 1, 1).clamp(min=1e-8)
                d.mul_((epsilon / dnorm).clamp(max=1.0))
                if d.grad is not None: d.grad.zero_()

    # ---- Add clean loss gradient ----
    loss_clean.backward()

    optimizer.step()

    return {
        "loss_total": float(loss_clean.item() + lambda_adv * (sum(adv_losses) / max(K, 1))),
        "loss_clean": float(loss_clean.item()),
        "loss_adv_mean": float(sum(adv_losses) / max(K, 1)),
    }


# ============================================================
# Standard (non-FreeLB) batch step
# ============================================================
def standard_step(model, batch, optimizer, view_dropout_prob):
    optimizer.zero_grad(set_to_none=True)
    out = model(
        surface_ids=batch["surface_ids"], surface_mask=batch["surface_mask"],
        lex_ids=batch["lex_ids"], lex_mask=batch["lex_mask"],
        char_ids=batch.get("char_ids"), char_mask=batch.get("char_mask"),
        view_dropout_prob=view_dropout_prob,
    )
    labels = batch["labels"].float()
    loss, comp = model.compute_loss(out, labels)
    loss.backward()
    optimizer.step()
    return comp


# ============================================================
# Eval helper (clean test or adv test)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        batch = move_batch_to(batch, device)
        out = model(
            surface_ids=batch["surface_ids"], surface_mask=batch["surface_mask"],
            lex_ids=batch["lex_ids"], lex_mask=batch["lex_mask"],
            char_ids=batch.get("char_ids"), char_mask=batch.get("char_mask"),
        )
        all_logits.append(out["p_main"].float().cpu().numpy())
        all_labels.append(batch["labels"].cpu().numpy())
    model.train()
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    preds = (logits > 0.0).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    n = len(preds)
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"f1": f1, "accuracy": acc, "precision": prec, "recall": rec,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    out_dir = Path(args.output)
    log = setup_logger(out_dir)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg["strategy"] = args.strategy
    yaml.dump(cfg, open(out_dir / "config.yaml", "w"), sort_keys=False)
    log.info(f"strategy={args.strategy}")
    log.info(f"config:\n{yaml.dump(cfg, sort_keys=False)}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device}  cuda={torch.cuda.is_available()}")

    # ---- Preprocessor ----
    if cfg.get("preprocessor") == "mvc":
        vocab_path = ROOT / "src" / "mvc_vocab.json"
        pre = MVCSamplePreprocessor(MVCVocab.from_file(vocab_path))
        cache_subdir = "cache_mvc"
    else:
        pre = SamplePreprocessor()
        cache_subdir = "cache"
    cache_dir = Path(args.cache_dir) if args.cache_dir else ROOT / "data" / cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Datasets ----
    train_path = Path(args.train_jsonl) if args.train_jsonl else ROOT / "data" / "splits" / "train.jsonl"
    val_path   = Path(args.val_jsonl)   if args.val_jsonl   else ROOT / "data" / "splits" / "val.jsonl"
    test_path  = Path(args.test_jsonl)  if args.test_jsonl  else ROOT / "data" / "splits" / "test.jsonl"

    train_ds = SQLDataset(train_path, cache_dir / "train.pkl", pre)
    val_ds   = SQLDataset(val_path,   cache_dir / "val.pkl",   pre)
    test_ds  = SQLDataset(test_path,  cache_dir / "test.pkl",  pre)

    # If aug or combined, mix adv samples into training records
    if args.strategy in ("aug", "combined"):
        if args.adv_jsonl is None:
            raise ValueError("--adv-jsonl required for aug / combined strategies")
        adv_records = build_adv_records(Path(args.adv_jsonl), pre)
        log.info(f"Loaded {len(adv_records)} adversarial samples from {args.adv_jsonl}")
        train_ds = _ConcatRecordsDataset(train_ds.records, adv_records)
    log.info(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    bs = cfg.get("batch_size", 16)
    nw = cfg.get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=collate_three_view, num_workers=nw,
                              pin_memory=True, persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                            collate_fn=collate_three_view, num_workers=nw,
                            pin_memory=True, persistent_workers=(nw > 0))
    test_loader = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                             collate_fn=collate_three_view, num_workers=nw,
                             pin_memory=True, persistent_workers=(nw > 0))

    # ---- Model ----
    variant = cfg.get("model_variant", "three_view")
    model = build_model(variant, cfg.get("model", {})).to(device)
    if args.init_ckpt and Path(args.init_ckpt).exists():
        ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        log.info(f"warm-started from {args.init_ckpt}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"model={variant}  params={n_params/1e6:.2f}M")

    # ---- Optim / sched ----
    epochs = cfg.get("epochs", 5)
    lr = float(cfg.get("lr", 2e-4))
    wd = float(cfg.get("weight_decay", 0.01))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * cfg.get("warmup_frac", 0.05))
    scheduler = linear_warmup_cosine(optimizer, total_steps, warmup_steps)
    view_dropout_prob = float(cfg.get("view_dropout", 0.1))

    # FreeLB hyperparams
    fl = cfg.get("freelb", {})
    K = int(fl.get("K", 3))
    epsilon = float(fl.get("epsilon", 0.1))
    alpha = float(fl.get("alpha", 0.03))
    lambda_adv = float(fl.get("lambda", 1.0))
    log.info(f"FreeLB: K={K} ε={epsilon} α={alpha} λ={lambda_adv}  "
             f"(active for strategy in [freelb, combined])")

    # ---- Train ----
    metrics_per_epoch = []
    best_val_f1 = 0.0
    use_freelb = args.strategy in ("freelb", "combined")
    log_every = cfg.get("log_every_steps", 100)
    for epoch in range(epochs):
        log.info(f"\n=== Epoch {epoch+1}/{epochs}  strategy={args.strategy} ===")
        model.train()
        t0 = time.time()
        loss_acc = {"loss_total": 0.0, "loss_clean": 0.0, "loss_adv_mean": 0.0,
                     "loss_main": 0.0}
        for step, batch in enumerate(train_loader):
            batch = move_batch_to(batch, device)
            if use_freelb:
                comp = freelb_step(model, batch, K, epsilon, alpha, lambda_adv,
                                    optimizer, view_dropout_prob)
            else:
                comp = standard_step(model, batch, optimizer, view_dropout_prob)
            scheduler.step()
            for k, v in comp.items():
                loss_acc[k] = loss_acc.get(k, 0.0) + v
            if (step + 1) % log_every == 0:
                avg = comp.get("loss_total") or comp.get("loss_main") or 0
                log.info(f"  step {step+1}/{len(train_loader)}  loss={avg:.4f}  "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}")
        elapsed = time.time() - t0
        log.info(f"  epoch elapsed {elapsed:.1f}s  "
                  f"mean_loss={loss_acc.get('loss_total', loss_acc.get('loss_main')) / max(len(train_loader),1):.4f}")

        val_m = evaluate(model, val_loader, device)
        log.info(f"  Val   f1={val_m['f1']:.4f}  P={val_m['precision']:.4f}  "
                  f"R={val_m['recall']:.4f}  acc={val_m['accuracy']:.4f}  "
                  f"(tp={val_m['tp']} fn={val_m['fn']} fp={val_m['fp']} tn={val_m['tn']})")
        metrics_per_epoch.append({"epoch": epoch + 1, "val": val_m,
                                    "train_loss": loss_acc})
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            torch.save({"model": model.state_dict()}, out_dir / "best_checkpoint.pt")
            log.info(f"  → new best val_f1={best_val_f1:.4f}")

    # Final test on best ckpt
    log.info("\n=== Final test (best checkpoint) ===")
    best = torch.load(out_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_m = evaluate(model, test_loader, device)
    log.info(f"Test  f1={test_m['f1']:.4f}  P={test_m['precision']:.4f}  "
              f"R={test_m['recall']:.4f}  acc={test_m['accuracy']:.4f}  "
              f"(tp={test_m['tp']} fn={test_m['fn']} fp={test_m['fp']} tn={test_m['tn']})")

    json.dump({
        "strategy": args.strategy,
        "epochs": metrics_per_epoch,
        "final_test": test_m,
        "best_val_f1": best_val_f1,
        "config": cfg,
    }, open(out_dir / "metrics.json", "w"), indent=2, default=float)
    log.info("Wrote metrics.json")


if __name__ == "__main__":
    main()
