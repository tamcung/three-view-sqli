#!/usr/bin/env python3
"""Standalone evaluation: load a trained checkpoint, run on test split, then
compare to two baselines (libinjection-only and TF-IDF + Logistic Regression).

Reports global metrics + per-stratum breakdowns:
  - per attack technique  (recall = TPR for that technique)
  - per victim_slot_context  (precision/recall/F1 within that subset)
  - per benign_subtype  (FPR for probe / plain / numeric / identifier)
  - per source_project
  - per statement_type

Usage:
  python evaluate.py --checkpoint results/run_001/best_checkpoint.pt \
                     --output  results/run_001/eval/
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import SamplePreprocessor
from dataset import SQLDataset, collate_three_view, move_batch_to
from model import ThreeViewModel
from ablation_models import build_model
from libinjection_wrapper import is_sqli as libinj_is_sqli


# ============================================================
# Generic metrics
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    from sklearn.metrics import (
        precision_recall_fscore_support, accuracy_score, roc_auc_score,
        confusion_matrix,
    )
    P, R, F1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    acc = accuracy_score(y_true, y_pred)
    auc = float("nan")
    if y_prob is not None and len(set(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "n": int(len(y_true)),
        "n_pos": int(int((y_true == 1).sum())),
        "n_neg": int(int((y_true == 0).sum())),
        "precision": float(P),
        "recall": float(R),
        "f1": float(F1),
        "accuracy": float(acc),
        "auc": float(auc),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def stratified_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    strata: list,
) -> dict:
    """For each unique value in `strata`, compute metrics on that subset."""
    breakdown = {}
    strata_arr = np.array(strata, dtype=object)
    uniq = sorted({s for s in strata if s is not None}, key=lambda s: str(s))
    for v in uniq:
        idx = np.where(strata_arr == v)[0]
        if len(idx) == 0:
            continue
        sub_true = y_true[idx]
        sub_pred = y_pred[idx]
        sub_prob = y_prob[idx] if y_prob is not None else None
        breakdown[str(v)] = compute_metrics(sub_true, sub_pred, sub_prob)
    return breakdown


# ============================================================
# Three-view inference
# ============================================================
@torch.no_grad()
def model_predict(model, loader, device):
    import inspect
    model.eval()
    logits_all, labels_all, meta_all = [], [], []
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
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            out = model(**{k: v for k, v in kwargs.items() if k in accepted})
        logits_all.append(out["p_main"].float().cpu().numpy())
        labels_all.append(batch["labels"].cpu().numpy())
        meta_all.extend(batch["meta"])
    return (
        np.concatenate(logits_all),
        np.concatenate(labels_all),
        meta_all,
    )


# ============================================================
# Baseline 1: libinjection-only
# ============================================================
def baseline_libinjection(jsonl_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """libinjection.is_sqli on raw user_input — returns (preds, labels)."""
    preds, labels = [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            res = libinj_is_sqli(r["user_input"])
            preds.append(1 if (res[0] if isinstance(res, tuple) else res) else 0)
            labels.append(1 if r["label"] == "attack" else 0)
    return np.array(preds), np.array(labels)


# ============================================================
# Baseline 2: TF-IDF + Logistic Regression on surface text
# ============================================================
def baseline_tfidf_lr(train_jsonl: Path, test_jsonl: Path):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    def load(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f]
        sqls = [r["user_input"] for r in rows]
        ys = np.array([1 if r["label"] == "attack" else 0 for r in rows])
        return sqls, ys

    train_sqls, train_y = load(train_jsonl)
    test_sqls, test_y = load(test_jsonl)

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=50_000, sublinear_tf=True,
    )
    Xtr = vec.fit_transform(train_sqls)
    Xte = vec.transform(test_sqls)
    clf = LogisticRegression(
        max_iter=300, C=1.0, solver="liblinear",
        class_weight="balanced",
    )
    clf.fit(Xtr, train_y)

    probs = clf.predict_proba(Xte)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return preds, probs, test_y


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--config", type=str, default=None,
                    help="Optional config (defaults to checkpoint dir/config.yaml)")
    p.add_argument("--baselines", action="store_true", default=True,
                    help="Run baselines (default on)")
    p.add_argument("--no-baselines", dest="baselines", action="store_false")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "evaluation.log", encoding="utf-8"),
        ],
    )
    log = logging.getLogger("eval")

    ckpt_path = Path(args.checkpoint)
    cfg_path = Path(args.config) if args.config else ckpt_path.parent / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) if cfg_path.exists() else {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Load model
    variant = cfg.get("model_variant", "three_view")
    model = build_model(variant, cfg.get("model", {})).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    log.info(f"Loaded checkpoint: {ckpt_path}  variant={variant}  "
             f"epoch={ckpt.get('epoch')}  best_val_f1={ckpt.get('best_val_f1')}")

    # Test set
    pre = SamplePreprocessor()
    test_path = ROOT / "data" / "splits" / "test.jsonl"
    train_path = ROOT / "data" / "splits" / "train.jsonl"
    cache_dir = ROOT / "data" / "cache"
    test_ds = SQLDataset(test_path, cache_dir / "test.pkl", pre)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.get("batch_size", 64) * 2, shuffle=False,
        collate_fn=collate_three_view, num_workers=cfg.get("num_workers", 2),
        pin_memory=True,
    )

    # ---- Three-view inference ----
    log.info("Running three-view model inference on test split...")
    t0 = time.time()
    logits, labels, metas = model_predict(model, test_loader, device)
    log.info(f"  done in {time.time() - t0:.1f}s ({len(labels)} samples)")

    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= args.threshold).astype(int)

    # Global metrics
    global_metrics = compute_metrics(labels, preds, probs)
    log.info(
        f"Three-view global: f1={global_metrics['f1']:.4f}  "
        f"P={global_metrics['precision']:.4f}  R={global_metrics['recall']:.4f}  "
        f"acc={global_metrics['accuracy']:.4f}  auc={global_metrics['auc']:.4f}"
    )

    # Stratified breakdowns
    strata_specs = {
        "source": [m.get("source") for m in metas],
        "subtype": [m.get("subtype") for m in metas],
        "technique": [m.get("technique") for m in metas],
    }
    breakdowns = {
        name: stratified_breakdown(labels, preds, probs, strata)
        for name, strata in strata_specs.items()
    }

    # Specifically log LLM hard-negative FPR (replaces probe)
    llm_idx = np.array([m.get("source") == "llm" for m in metas])
    if llm_idx.any():
        n_llm = int(llm_idx.sum())
        n_llm_fp = int((preds[llm_idx] == 1).sum())
        log.info(
            f"LLM hard-negative FPR: {n_llm_fp}/{n_llm} = "
            f"{n_llm_fp / max(n_llm, 1) * 100:.2f}%  "
            f"(LLM-generated SQL-keyword text misclassified as attack)"
        )

    # ---- Baselines ----
    baseline_results = {}

    if args.baselines:
        log.info("\nBaseline 1: libinjection.is_sqli ...")
        t0 = time.time()
        lib_preds, lib_labels = baseline_libinjection(test_path)
        lib_metrics = compute_metrics(lib_labels, lib_preds, lib_preds.astype(float))
        log.info(
            f"  libinjection: f1={lib_metrics['f1']:.4f}  "
            f"P={lib_metrics['precision']:.4f}  R={lib_metrics['recall']:.4f}  "
            f"acc={lib_metrics['accuracy']:.4f}  ({time.time() - t0:.1f}s)"
        )
        baseline_results["libinjection"] = {
            "global": lib_metrics,
            "by_source": stratified_breakdown(
                lib_labels, lib_preds, lib_preds.astype(float),
                [m.get("source") for m in metas]
            ),
            "by_subtype": stratified_breakdown(
                lib_labels, lib_preds, lib_preds.astype(float),
                [m.get("subtype") for m in metas]
            ),
        }

        # (TF-IDF baseline removed — replaced by Sequence-LSTM / Tree-LSTM
        # checkpoints which are evaluated separately by re-invoking
        # `evaluate.py --checkpoint results/seq_lstm/best_checkpoint.pt`.)

    # ---- Save everything ----
    np.savez(out_dir / "test_predictions.npz",
              logits=logits, labels=labels, probs=probs, preds=preds)

    result = {
        "checkpoint": str(ckpt_path),
        "threshold": args.threshold,
        "three_view": {
            "global": global_metrics,
            "by_source": breakdowns["source"],
            "by_subtype": breakdowns["subtype"],
            "by_technique": breakdowns["technique"],
        },
        "baselines": baseline_results,
    }
    with open(out_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info(f"\nWrote {out_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
