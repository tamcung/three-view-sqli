#!/usr/bin/env python3
"""End-to-end robustness evaluation for §4.6.

Given a checkpoint, reports a single JSON / table covering:

    1. Clean test set:        precision, recall, F1, AUC
    2. Frozen adversarial sets passed in via `--adv-jsonls`. Each is a
       jsonl of {user_input, label, ...} records. Recall is reported per
       file (since the records are all attacks by construction).
    3. Tamper-OOD sweep:       average recall across train+holdout tampers
                               (re-uses eval_tampers logic)
    4. Optional fresh attacks: regenerate `n_fresh` adversarial samples
       with the search attacker against THIS checkpoint, then check what
       fraction of those it now flags. (sanity check the model has not
       just memorized the previous round's adversarials)

Usage:
    python -m scripts.adversarial.eval_robustness \
        --checkpoint results/cotrain_v1/round_3/best_checkpoint.pt \
        --output results/cotrain_v1/round_3/robustness.json \
        --adv-jsonls results/cotrain_v1/holdout_adv/search_holdout.jsonl \
                      data/adversarial/search_pilot.jsonl
"""
from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.adversarial.utils import (
    load_victim, batch_predict, setup_logger,
)
sys.path.insert(0, str(ROOT / "src"))
from train import compute_binary_metrics                # noqa: E402


def eval_clean(model, pre, device, accepted, jsonl_path, threshold=0.5):
    rows, labels = [], []
    with open(jsonl_path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            rows.append(r["user_input"])
            labels.append(1 if r["label"] == "attack" else 0)
    probs = batch_predict(model, pre, device, accepted, rows, batch_size=128)
    logits = np.log(np.clip(probs, 1e-9, 1 - 1e-9) /
                    (1 - np.clip(probs, 1e-9, 1 - 1e-9)))
    return compute_binary_metrics(np.array(labels), logits, threshold)


def eval_attacks_only(model, pre, device, accepted, jsonl_path, threshold=0.5):
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r.get("label", "attack") == "attack":
                rows.append(r["user_input"])
    if not rows:
        return {"n": 0, "recall": float("nan")}
    probs = batch_predict(model, pre, device, accepted, rows, batch_size=128)
    preds = (probs >= threshold).astype(int)
    return {"n": len(rows), "recall": float(preds.mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--test-jsonl", type=str,
                    default=str(ROOT / "data" / "splits" / "test.jsonl"))
    p.add_argument("--adv-jsonls", nargs="*", default=[],
                    help="Frozen adversarial sets to evaluate recall on.")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--run-tamper-eval", action="store_true",
                    help="Also re-run eval_tampers.py on data/tamper_oods/")
    p.add_argument("--fresh-attack-n", type=int, default=0,
                    help="If >0, run a fresh GA attack to sanity-check robustness.")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logger("eval_robust", out_path.with_suffix(".log"))

    log.info(f"Loading {args.checkpoint}")
    model, pre, device, variant, accepted = load_victim(args.checkpoint)
    log.info(f"  variant={variant} device={device}")

    report = {"checkpoint": str(args.checkpoint), "variant": variant}

    # ---- 1. clean test ----
    log.info(f"\n--- Clean test ---")
    clean = eval_clean(model, pre, device, accepted, args.test_jsonl,
                        args.threshold)
    log.info(f"  P={clean['precision']:.4f}  R={clean['recall']:.4f}  "
              f"F1={clean['f1']:.4f}  AUC={clean['auc']:.4f}")
    report["clean_test"] = clean

    # ---- 2. frozen adversarial sets ----
    log.info(f"\n--- Frozen adversarial recall ---")
    report["adv"] = {}
    for adv_path in args.adv_jsonls:
        ap = Path(adv_path)
        if not ap.exists():
            log.warning(f"  missing: {ap}")
            continue
        r = eval_attacks_only(model, pre, device, accepted, ap, args.threshold)
        report["adv"][ap.name] = r
        log.info(f"  {ap.name:35s}  n={r['n']:>5d}  recall={r['recall']:.4f}")

    # ---- 3. tamper sweep (optional, can take 30-60s on full set) ----
    if args.run_tamper_eval:
        log.info(f"\n--- Tamper sweep ---")
        tdir = out_path.parent / "tampers"
        tdir.mkdir(exist_ok=True)
        cmd = [
            sys.executable, str(ROOT / "eval_tampers.py"),
            "--checkpoint", str(args.checkpoint),
            "--output", str(tdir),
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            log.error(proc.stderr[-1500:])
        else:
            tjson = tdir / "tamper_recalls.json"
            if tjson.exists():
                tr = json.load(open(tjson, encoding="utf-8"))
                # Mean recall across all tampers (with the current variant key)
                recalls = []
                for k, v in tr.items():
                    rec = v.get(variant)
                    if rec is not None:
                        recalls.append(rec)
                if recalls:
                    log.info(f"  mean tamper recall: "
                              f"{np.mean(recalls):.4f}  "
                              f"(n_tampers={len(recalls)})")
                report["tamper_mean_recall"] = float(np.mean(recalls)) if recalls else None
                report["tamper_per_subset"] = tr

    # ---- 4. fresh attack ----
    if args.fresh_attack_n > 0:
        log.info(f"\n--- Fresh GA attack ({args.fresh_attack_n} seeds) ---")
        fresh_path = out_path.parent / "fresh_attack.jsonl"
        cmd = [
            sys.executable, "-m", "scripts.adversarial.search_attacker",
            "--checkpoint", str(args.checkpoint),
            "--output", str(fresh_path),
            "--n-seeds", str(args.fresh_attack_n),
            "--pop-size", "20", "--generations", "10",
            "--seed", "777",
            "--limit-seeds-already-broken",
        ]
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            log.error(proc.stderr[-1500:])
        else:
            stats = fresh_path.with_suffix(".stats.json")
            if stats.exists():
                s = json.load(open(stats, encoding="utf-8"))
                log.info(f"  ASR (fresh GA): {s['asr']:.4f}  "
                          f"({s['n_success']}/{s['n_attempted']})")
                report["fresh_attack"] = {
                    "n_attempted": s["n_attempted"],
                    "n_success": s["n_success"],
                    "asr": s["asr"],
                    "n_already_broken": s["n_already_broken"],
                }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()
