#!/usr/bin/env python3
"""
Standalone evaluation: load a checkpoint, evaluate on test set, run baselines.

Usage:
    python evaluate.py --checkpoint results/run_001/best_checkpoint.pt --output results/run_001/eval/

Reports:
    - main + 3 single-view F1/P/R/Acc + Wilson CI
    - libinjection (rule-based) baseline F1/P/R/Acc + Wilson CI on the same test set
    - TF-IDF + LR baseline (trained on train split)
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import SamplePreprocessor
from src.dataset import (
    build_split_files, preprocess_split_file,
    WafamoleThreeViewDataset, collate_three_view,
)
from src.model import ThreeViewModel
from src.libinjection_wrapper import is_sqli as libinj_is_sqli

from train import wilson_ci, binary_metrics, evaluate as model_evaluate


# ============================================================
# Strict metric: recall at a fixed false-positive rate
# ============================================================
def recall_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> dict:
    """Find the threshold that yields ≤ target_fpr on negatives, then report
    recall on positives at that threshold.

    For WAFs: business cost of false positives is high, so the operationally
    meaningful question is "if I allow at most X% benign false alarms, how many
    attacks do I catch?"
    """
    # Negative scores → find threshold
    neg = scores[labels == 0]
    pos = scores[labels == 1]
    if len(neg) == 0 or len(pos) == 0:
        return {"target_fpr": target_fpr, "actual_fpr": None,
                "recall": None, "threshold": None}
    # Threshold = the (1 - target_fpr) quantile of negative scores
    # Anything above this threshold is predicted positive (FP for negatives)
    k = max(1, int(len(neg) * target_fpr))
    threshold = float(np.partition(neg, -k)[-k])
    fp = int((neg > threshold).sum())
    tp = int((pos > threshold).sum())
    actual_fpr = fp / len(neg)
    recall = tp / len(pos)
    return {
        "target_fpr": target_fpr,
        "actual_fpr": actual_fpr,
        "threshold": threshold,
        "recall": recall,
        "tp": tp, "fn": int(len(pos) - tp),
        "fp": fp, "tn": int(len(neg) - fp),
    }


@torch.no_grad()
def collect_scores(model, loader, device, use_bf16: bool):
    """Run model and return raw logits per view + labels (for strict metrics)."""
    model.eval()
    sc = {k: [] for k in ("main", "S", "L", "A")}
    labels_all = []
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
        sc["main"].append(out["p_main"].float().cpu().numpy())
        sc["S"].append(out["p_S"].float().cpu().numpy())
        sc["L"].append(out["p_L"].float().cpu().numpy())
        sc["A"].append(out["p_A"].float().cpu().numpy())
        labels_all.append(batch["label"].cpu().numpy())
    return ({k: np.concatenate(v) for k, v in sc.items()},
            np.concatenate(labels_all))


def evaluate_libinjection(test_jsonl: Path) -> dict:
    """Run libinjection on every test sample, compute metrics."""
    import json as _json
    preds, labels = [], []
    with open(test_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = _json.loads(line)
            flag, _ = libinj_is_sqli(obj["text"])
            preds.append(int(flag))
            labels.append(obj["label"])
    return binary_metrics(np.array(preds, dtype=float) - 0.5, np.array(labels))


def evaluate_tfidf_lr(train_jsonl: Path, test_jsonl: Path) -> dict:
    """Train TF-IDF + LR on train split, evaluate on test."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    import json as _json

    print("  loading train texts...", flush=True)
    X_train, y_train = [], []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = _json.loads(line)
            X_train.append(obj["text"])
            y_train.append(obj["label"])
    X_test, y_test = [], []
    with open(test_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = _json.loads(line)
            X_test.append(obj["text"])
            y_test.append(obj["label"])

    print(f"  fitting TF-IDF on {len(X_train)} samples...", flush=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           max_features=100000, lowercase=True, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    print(f"  training LR...", flush=True)
    clf = LogisticRegression(max_iter=200, solver="liblinear")
    clf.fit(Xtr, np.array(y_train))
    scores = clf.decision_function(Xte)
    return binary_metrics(scores, np.array(y_test))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--skip-baselines", action="store_true")
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(cfg["train"].get("use_bf16", False)) and device.type == "cuda" and torch.cuda.is_bf16_supported()

    # Reload data with same split
    pre = SamplePreprocessor()
    split_paths = build_split_files(
        n_train_per_class=cfg["n_train_per_class"],
        n_val_per_class=cfg["n_val_per_class"],
        n_test_per_class=cfg["n_test_per_class"],
        seed=cfg["seed"],
    )
    cache_paths = {s: preprocess_split_file(p, pre) for s, p in split_paths.items()}

    test_ds = WafamoleThreeViewDataset(cache_paths["test"])
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                             shuffle=False, collate_fn=collate_three_view,
                             num_workers=cfg["train"].get("num_workers", 0),
                             pin_memory=device.type == "cuda")

    # Reload model
    model = ThreeViewModel(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])

    print(f"\n=== Three-view model evaluation on test set ({len(test_ds)} samples) ===")
    metrics_three = model_evaluate(model, test_loader, device, use_bf16)
    for view in ("main", "S", "L", "A"):
        m = metrics_three[view]
        print(f"  {view:6s} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"Acc={m['accuracy']:.4f}  CI[{m['acc_ci_low']:.4f}, {m['acc_ci_high']:.4f}]")

    # Strict metrics: recall at low FPR (the operationally meaningful WAF metric)
    print(f"\n=== Recall @ low FPR (strict, WAF deployment-relevant) ===")
    scores, labels = collect_scores(model, test_loader, device, use_bf16)
    strict = {}
    for fpr_target in (0.01, 0.001, 0.0001):
        print(f"\n  Target FPR = {fpr_target}:")
        strict[f"fpr_{fpr_target}"] = {}
        for view in ("main", "S", "L", "A"):
            r = recall_at_fpr(scores[view], labels, fpr_target)
            strict[f"fpr_{fpr_target}"][view] = r
            recall_str = f"{r['recall']:.4f}" if r['recall'] is not None else "N/A"
            print(f"    {view:6s} recall={recall_str}  (actual FPR={r['actual_fpr']:.5f}, "
                  f"thr={r['threshold']:.3f}, TP={r['tp']}/{r['tp']+r['fn']}, FP={r['fp']})")
    metrics_three["strict_metrics"] = strict
    results = {"three_view": metrics_three}

    if not args.skip_baselines:
        print("\n=== libinjection baseline ===")
        m_lib = evaluate_libinjection(split_paths["test"])
        print(f"  F1={m_lib['f1']:.4f}  P={m_lib['precision']:.4f}  R={m_lib['recall']:.4f}  "
              f"Acc={m_lib['accuracy']:.4f}  CI[{m_lib['acc_ci_low']:.4f}, {m_lib['acc_ci_high']:.4f}]")
        results["libinjection"] = m_lib

        print("\n=== TF-IDF + LR baseline ===")
        m_tfidf = evaluate_tfidf_lr(split_paths["train"], split_paths["test"])
        print(f"  F1={m_tfidf['f1']:.4f}  P={m_tfidf['precision']:.4f}  R={m_tfidf['recall']:.4f}  "
              f"Acc={m_tfidf['accuracy']:.4f}  CI[{m_tfidf['acc_ci_low']:.4f}, {m_tfidf['acc_ci_high']:.4f}]")
        results["tfidf_lr"] = m_tfidf

    out_file = args.output / "evaluation.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
