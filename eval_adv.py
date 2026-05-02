#!/usr/bin/env python3
"""Chapter 4 evaluation: clean test set + adversarial sample set.

Loads a trained checkpoint and evaluates on both:
  - the original V3 test set (Kaggle test.jsonl) → clean F1, Acc, etc.
  - an adversarial-sample JSONL (output of ``adv_generator.py``)
    → adversarial F1, ASR (= 1 - Recall on attack samples).

Usage:
    python eval_adv.py \
        --ckpt-dir results_adv/ch4_combined \
        --clean-jsonl data/external/kaggle_sqli/jsonl/test.jsonl \
        --adv-jsonl   results/ch4/adv_round0.jsonl \
        --output      results_adv/ch4_combined/eval_summary.json
"""
from __future__ import annotations
import argparse, inspect, json, sys, time
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import SamplePreprocessor
from dataset import collate_three_view, move_batch_to
from ablation_models import build_model

try:
    from mvc_preprocessor import MVCSamplePreprocessor, MVCVocab
except ImportError:
    MVCSamplePreprocessor = None
    MVCVocab = None


def load_jsonl(p):
    sqls, ys = [], []
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            sqls.append(r["user_input"])
            ys.append(1 if r.get("label") == "attack" else 0)
    return sqls, np.array(ys, dtype=np.int64)


def confusion(preds, labels):
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    n = len(preds)
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=acc, precision=prec, recall=rec, f1=f1)


@torch.no_grad()
def predict(model, sqls, pre, device, batch_size=128):
    from torch.utils.data import DataLoader, Dataset
    recs = [{**pre(s), "label_int": 1, "meta": {}} for s in sqls]

    class _D(Dataset):
        def __init__(self, recs): self.recs = recs
        def __len__(self): return len(self.recs)
        def __getitem__(self, i): return self.recs[i]

    accepted = set(inspect.signature(model.forward).parameters.keys())
    loader = DataLoader(_D(recs), batch_size=batch_size, shuffle=False,
                        collate_fn=collate_three_view, num_workers=0)
    logits = []
    for batch in loader:
        batch = move_batch_to(batch, device)
        kwargs = dict(
            surface_ids=batch["surface_ids"], surface_mask=batch["surface_mask"],
            lex_ids=batch["lex_ids"], lex_mask=batch["lex_mask"],
            ast_ids=batch["ast_ids"], ast_mask=batch["ast_mask"],
            ast_valid=batch["ast_valid"],
            char_ids=batch.get("char_ids"),
            char_mask=batch.get("char_mask"),
        )
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        try:
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                out = model(**kwargs)
        except Exception:
            out = model(**kwargs)
        logits.append(out["p_main"].float().cpu().numpy())
    return (np.concatenate(logits) > 0.0).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--clean-jsonl", required=True)
    ap.add_argument("--adv-jsonl", default=None)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    cfg = yaml.safe_load(open(ckpt_dir / "config.yaml", encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    model = build_model(cfg.get("model_variant", "three_view"),
                        cfg.get("model", {})).to(device)
    ckpt = torch.load(ckpt_dir / "best_checkpoint.pt", map_location=device,
                      weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if cfg.get("preprocessor") == "mvc":
        vocab_path = ROOT / "src" / "mvc_vocab.json"
        pre = MVCSamplePreprocessor(MVCVocab.from_file(vocab_path))
    else:
        pre = SamplePreprocessor()

    summary = {"ckpt_dir": str(ckpt_dir), "strategy": cfg.get("strategy", "unknown")}

    # Clean
    print(f"\nClean test: {args.clean_jsonl}", flush=True)
    sqls, y = load_jsonl(args.clean_jsonl)
    print(f"  {len(sqls)} samples ({int(y.sum())} attack / "
          f"{len(y)-int(y.sum())} benign)", flush=True)
    t0 = time.time()
    preds = predict(model, sqls, pre, device)
    m = confusion(preds, y)
    print(f"  F1={m['f1']:.4f} Acc={m['accuracy']:.4f} P={m['precision']:.4f} "
          f"R={m['recall']:.4f}  ({time.time()-t0:.1f}s)", flush=True)
    summary["clean"] = m

    # Adversarial
    if args.adv_jsonl and Path(args.adv_jsonl).exists():
        print(f"\nAdversarial test: {args.adv_jsonl}", flush=True)
        sqls, y = load_jsonl(args.adv_jsonl)
        print(f"  {len(sqls)} samples ({int(y.sum())} attack / "
              f"{len(y)-int(y.sum())} benign)", flush=True)
        if int(y.sum()) == 0:
            print("  (skip — no attack samples)", flush=True)
        else:
            t0 = time.time()
            preds = predict(model, sqls, pre, device)
            m = confusion(preds, y)
            # ASR = fraction of attacks predicted as benign = FN / (FN + TP)
            asr = m["fn"] / max(m["fn"] + m["tp"], 1)
            print(f"  F1={m['f1']:.4f} Acc={m['accuracy']:.4f} P={m['precision']:.4f} "
                  f"R={m['recall']:.4f} ASR={asr:.4f}  ({time.time()-t0:.1f}s)",
                  flush=True)
            m["asr"] = asr
            summary["adversarial"] = m

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.output, "w"), indent=2, default=float)
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
