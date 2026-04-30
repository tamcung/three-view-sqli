#!/usr/bin/env python3
"""Evaluate a model checkpoint on every tamper-OOD subset.

Reads `data/tamper_oods/*.jsonl` and computes recall (since each subset is
attacks-only, P=1.0 trivially when the model says attack). Reports a
recall matrix vs libinjection.

Output:
  <output_dir>/tamper_recalls.json
"""
from __future__ import annotations
import argparse
import inspect
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
from deobfuscation import deobfuscate


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
    p.add_argument("--tamper-dir", type=str, default=str(ROOT / "data" / "tamper_oods"))
    p.add_argument("--deobfuscate", action="store_true",
                    help="Apply Hu-style de-obfuscation (URL/HTML/hex/base64/etc) "
                         "before sending input to model and libinjection")
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                    logging.FileHandler(out_dir / "tamper_eval.log", encoding="utf-8")],
    )
    log = logging.getLogger("tamper")

    cfg_path = Path(args.checkpoint).parent / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.get("model_variant", "three_view"), cfg.get("model", {})).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    pre = SamplePreprocessor()
    accepted = set(inspect.signature(model.forward).parameters.keys())
    variant = cfg.get("model_variant", "three_view")

    @torch.no_grad()
    def predict(sqls):
        recs = [{**pre(s), "label_int": 1, "meta": {}} for s in sqls]
        loader = DataLoader(MemDataset(recs), batch_size=128, shuffle=False,
                              collate_fn=collate_three_view, num_workers=0)
        logits = []
        for batch in loader:
            batch = move_batch_to(batch, device)
            kwargs = dict(
                surface_ids=batch["surface_ids"], surface_mask=batch["surface_mask"],
                lex_ids=batch["lex_ids"], lex_mask=batch["lex_mask"],
                ast_ids=batch["ast_ids"], ast_mask=batch["ast_mask"],
                ast_valid=batch["ast_valid"],
                ast_node_ids=batch.get("ast_node_ids"),
                ast_parent=batch.get("ast_parent"),
                char_ids=batch.get("char_ids"),
                char_mask=batch.get("char_mask"),
            )
            try:
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                      dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    out = model(**{k: v for k, v in kwargs.items() if k in accepted})
            except Exception:
                out = model(**{k: v for k, v in kwargs.items() if k in accepted})
            logits.append(out["p_main"].float().cpu().numpy())
        return np.concatenate(logits)

    def predict_libinj(sqls):
        out = []
        for s in sqls:
            r = libinj_is_sqli(s)
            out.append(1 if (r[0] if isinstance(r, tuple) else r) else 0)
        return np.array(out)

    # Walk tamper dir
    tamper_dir = Path(args.tamper_dir)
    files = sorted(tamper_dir.glob("*.jsonl"))
    log.info(f"Found {len(files)} tamper subsets in {tamper_dir}")

    results = {}
    for f in files:
        name = f.stem.lstrip("_")
        with open(f, encoding="utf-8") as fp:
            rows = [json.loads(line) for line in fp]
        if not rows:
            log.info(f"  {name:30s} (empty, skip)")
            continue

        sqls = [r["user_input"] for r in rows]
        if args.deobfuscate:
            sqls = [deobfuscate(s) for s in sqls]
        t0 = time.time()
        logits = predict(sqls)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= args.threshold).astype(int)
        rec_model = preds.mean()
        t_model = time.time() - t0

        t0 = time.time()
        preds_lib = predict_libinj(sqls)
        rec_lib = preds_lib.mean()
        t_lib = time.time() - t0

        log.info(f"  {name:30s}  n={len(rows):>5d}  {variant}={rec_model:.4f}  libinj={rec_lib:.4f}  ({t_model:.1f}s/{t_lib:.1f}s)")

        results[name] = {
            "n": len(rows),
            variant: float(rec_model),
            "libinjection": float(rec_lib),
        }

    # Sort and pretty-print summary
    log.info("\n" + "=" * 70)
    log.info(f"  SUMMARY — recall under each tamper")
    log.info("=" * 70)
    log.info(f"  {'tamper':30s}  {'n':>5s}  {variant:>10s}  {'libinj':>10s}")
    sorted_keys = sorted(results, key=lambda k: -results[k][variant])
    for k in sorted_keys:
        r = results[k]
        log.info(f"  {k:30s}  {r['n']:>5d}  {r[variant]:>10.4f}  {r['libinjection']:>10.4f}")

    with open(out_dir / "tamper_recalls.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Wrote {out_dir / 'tamper_recalls.json'}")


if __name__ == "__main__":
    main()
