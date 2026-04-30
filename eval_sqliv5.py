#!/usr/bin/env python3
"""Out-of-distribution evaluation on the SQLiV5 held-out set.

The V5 sqli rows are heavily obfuscated (binary/octal/hex literals, weird
whitespace, mixed case, random suffixes). They are NOT in our training,
val, or test splits — built explicitly disjoint via build_sqliv5_evalset.py.

Reports recall (since attacks dominate) for:
  - three-view model (loaded from a checkpoint)
  - libinjection
  - TF-IDF + LR (re-fit on training split)
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
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import SamplePreprocessor
from dataset import collate_three_view, move_batch_to
from ablation_models import build_model
from libinjection_wrapper import is_sqli as libinj_is_sqli


class MemDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        return self.recs[i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--deobfuscate", action="store_true",
                    help="Apply Hu-style de-obfuscation pre-processing")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                    logging.FileHandler(out_dir / "sqliv5.log", encoding="utf-8")],
    )
    log = logging.getLogger("v5")

    # Load V5 eval rows
    with open(ROOT / "data" / "sqliv5_eval.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    log.info(f"V5 eval set: {len(rows)} rows ({sum(1 for r in rows if r['label']=='attack')} attack, "
              f"{sum(1 for r in rows if r['label']=='benign')} benign)")

    inputs = [r["user_input"] for r in rows]
    if args.deobfuscate:
        from deobfuscation import deobfuscate
        inputs = [deobfuscate(s) for s in inputs]
        log.info(f"  applied deobfuscation to {len(inputs)} inputs")
    y = np.array([1 if r["label"] == "attack" else 0 for r in rows])

    # ---- Three-view model ----
    cfg_path = Path(args.checkpoint).parent / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.get("model_variant", "three_view"), cfg.get("model", {})).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    pre = SamplePreprocessor()

    log.info("Model inference...")
    import inspect
    accepted = set(inspect.signature(model.forward).parameters.keys())
    t0 = time.time()
    recs = []
    for s in inputs:
        f = pre(s)
        recs.append({**f, "label_int": 1, "meta": {}})
    loader = DataLoader(MemDataset(recs), batch_size=128, shuffle=False,
                          collate_fn=collate_three_view, num_workers=0)
    logits = []
    with torch.no_grad():
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
                ast_node_ids=batch.get("ast_node_ids"),
                ast_parent=batch.get("ast_parent"),
                char_ids=batch.get("char_ids"),
                char_mask=batch.get("char_mask"),
            )
            # CharCNN/etc may use float32; gracefully fall back
            try:
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                      dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model(**{k: v for k, v in kwargs.items() if k in accepted})
            except Exception:
                out = model(**{k: v for k, v in kwargs.items() if k in accepted})
            logits.append(out["p_main"].float().cpu().numpy())
    logits_tv = np.concatenate(logits)
    probs_tv = 1.0 / (1.0 + np.exp(-logits_tv))
    preds_tv = (probs_tv >= args.threshold).astype(int)
    log.info(f"  done in {time.time()-t0:.1f}s")

    # ---- Libinjection ----
    log.info("libinjection inference...")
    t0 = time.time()
    preds_lib = np.array([
        1 if (libinj_is_sqli(s)[0] if isinstance(libinj_is_sqli(s), tuple) else libinj_is_sqli(s)) else 0
        for s in inputs
    ])
    log.info(f"  done in {time.time()-t0:.1f}s")

    # ---- Metrics ----
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
    def metrics(name, y_true, y_pred, y_prob=None):
        P, R, F1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                          zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        try:
            auc = roc_auc_score(y_true, y_prob) if y_prob is not None and len(set(y_true)) > 1 else float("nan")
        except Exception:
            auc = float("nan")
        return {"f1": F1, "P": P, "R": R, "acc": acc, "auc": auc,
                "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
                "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
                "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
                "tn": int(((y_pred == 0) & (y_true == 0)).sum())}

    m_tv = metrics(cfg.get("model_variant", "three_view"), y, preds_tv, probs_tv)
    m_lib = metrics("libinjection", y, preds_lib, preds_lib.astype(float))

    log.info("\n" + "=" * 70)
    log.info(f"  SQLiV5 held-out evaluation (n_attack={int(y.sum())}, n_benign={int((y==0).sum())})")
    log.info("=" * 70)
    log.info(f"  {'model':16s}  {'F1':>6s}  {'P':>6s}  {'R':>6s}  {'acc':>6s}  {'AUC':>6s}  tp/fn")
    for name, m in [(cfg.get("model_variant", "three_view"), m_tv), ("libinjection", m_lib)]:
        log.info(f"  {name:16s}  {m['f1']:.4f}  {m['P']:.4f}  {m['R']:.4f}  {m['acc']:.4f}  "
                  f"{m['auc']:.4f}  {m['tp']}/{m['fn']}")

    # Save
    with open(out_dir / "sqliv5_results.json", "w", encoding="utf-8") as f:
        json.dump({"model": m_tv, "libinjection": m_lib,
                    "model_variant": cfg.get("model_variant", "three_view"),
                    "n_total": len(rows),
                    "n_attack": int(y.sum()),
                    "n_benign": int((y == 0).sum())}, f, indent=2)
    log.info(f"\nWrote {out_dir / 'sqliv5_results.json'}")


if __name__ == "__main__":
    main()
