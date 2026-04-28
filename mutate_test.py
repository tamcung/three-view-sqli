#!/usr/bin/env python3
"""
Robustness probe: apply simple semantics-preserving mutations to test SQLi
samples and re-evaluate.

This is NOT an adversarial search (no beam search, no query budget). It's a
deterministic mutator that applies common WAF bypass tactics:
  - random case flipping (UNION → UnIoN)
  - comment insertion between keywords (UNION SELECT → UNION/**/SELECT)
  - whitespace-to-tab/newline substitution
  - URL-percent-encoding of special chars

Goal: see whether three-view fusion (main) holds up better than surface-only
when surface-level features are perturbed.

Usage:
    python mutate_test.py --checkpoint results/run_full/best_checkpoint.pt \
                          --output results/run_full/mutate/
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import SamplePreprocessor
from src.dataset import (
    build_split_files, preprocess_split_file,
    WafamoleThreeViewDataset, collate_three_view, CACHE_ROOT,
)
from src.model import ThreeViewModel
from train import binary_metrics
from evaluate import collect_scores, recall_at_fpr


# ============================================================
# Semantics-preserving mutations (simple, deterministic)
# ============================================================
SQL_KEYWORDS = re.compile(
    r'\b(SELECT|FROM|WHERE|JOIN|UNION|INSERT|UPDATE|DELETE|AND|OR|NOT|'
    r'NULL|TRUE|FALSE|LIKE|IN|BETWEEN|IS|GROUP|ORDER|BY|HAVING|LIMIT|'
    r'OFFSET|VALUES|INTO|SET|AS|ON|DISTINCT|ALL|EXISTS|CASE|WHEN|THEN|'
    r'ELSE|END|TABLE|DROP|CREATE|ALTER|INDEX)\b',
    re.IGNORECASE,
)

URL_ENCODE_TARGETS = "<>'\"()&|;"


def mutation_case(text: str, p: float = 0.5) -> str:
    """Randomly flip case of each ASCII letter with probability p."""
    out = []
    for c in text:
        if c.isalpha() and random.random() < p:
            out.append(c.upper() if c.islower() else c.lower())
        else:
            out.append(c)
    return ''.join(out)


def mutation_comment(text: str, p: float = 0.7) -> str:
    """After each SQL keyword, insert /**/ with probability p."""
    def repl(m):
        kw = m.group(0)
        return f"{kw}/**/" if random.random() < p else kw
    return SQL_KEYWORDS.sub(repl, text)


def mutation_whitespace(text: str, p: float = 0.5) -> str:
    """Substitute spaces with tab/newline/comment with probability p."""
    out = []
    for c in text:
        if c == ' ' and random.random() < p:
            out.append(random.choice(['\t', '\n', ' ', '/**/']))
        else:
            out.append(c)
    return ''.join(out)


def mutation_urlencode(text: str, p: float = 0.4) -> str:
    """URL-encode some special characters with probability p."""
    out = []
    for c in text:
        if c in URL_ENCODE_TARGETS and random.random() < p:
            out.append(f"%{ord(c):02X}")
        else:
            out.append(c)
    return ''.join(out)


MUTATION_RECIPES = {
    "clean":      lambda s: s,                          # no mutation
    "case":       mutation_case,
    "comment":    mutation_comment,
    "whitespace": mutation_whitespace,
    "urlenc":     mutation_urlencode,
    "all":        lambda s: mutation_urlencode(
                              mutation_whitespace(
                                  mutation_comment(
                                      mutation_case(s, 0.5),
                                      0.7),
                                  0.5),
                              0.3),
}


# ============================================================
# Build mutated test cache
# ============================================================
def build_mutated_test(test_jsonl: Path, mutation_name: str, seed: int = 42) -> Path:
    """Build a new JSONL file where every SQLi sample is mutated.
    Benign samples are NOT mutated (they're not under attack).
    """
    out_path = test_jsonl.with_name(test_jsonl.stem + f"__mut_{mutation_name}.jsonl")
    if out_path.exists():
        return out_path
    rng = random.Random(seed)
    random.seed(seed)
    mutator = MUTATION_RECIPES[mutation_name]
    print(f"  Building mutated test: {out_path.name}", flush=True)
    n_mutated = 0
    n_total = 0
    with open(test_jsonl, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as o:
        for line in f:
            obj = json.loads(line)
            n_total += 1
            if obj["label"] == 1:
                obj["text"] = mutator(obj["text"])
                n_mutated += 1
            o.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"    mutated {n_mutated}/{n_total} samples")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--mutations", nargs="+",
                    default=["clean", "case", "comment", "whitespace", "urlenc", "all"])
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(cfg["train"].get("use_bf16", False)) and device.type == "cuda" and torch.cuda.is_bf16_supported()

    pre = SamplePreprocessor()
    split_paths = build_split_files(
        n_train_per_class=cfg["n_train_per_class"],
        n_val_per_class=cfg["n_val_per_class"],
        n_test_per_class=cfg["n_test_per_class"],
        seed=cfg["seed"],
    )
    test_jsonl = split_paths["test"]

    # Reload model
    model = ThreeViewModel(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])

    results = {}
    for mut_name in args.mutations:
        print(f"\n=== Mutation: {mut_name} ===")
        mut_jsonl = build_mutated_test(test_jsonl, mut_name)
        cache_path = preprocess_split_file(mut_jsonl, pre)
        ds = WafamoleThreeViewDataset(cache_path)
        loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            collate_fn=collate_three_view,
                            num_workers=cfg["train"].get("num_workers", 0),
                            pin_memory=device.type == "cuda")

        # Get raw scores per view
        scores, labels = collect_scores(model, loader, device, use_bf16)

        view_metrics = {}
        for view in ("main", "S", "L", "A"):
            m = binary_metrics(scores[view], labels)
            r_strict = recall_at_fpr(scores[view], labels, 0.001)
            view_metrics[view] = {**m, "recall_at_fpr_001": r_strict["recall"]}
            print(f"  {view:6s} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"R@FPR=0.001:{r_strict['recall']:.4f if r_strict['recall'] is not None else 0:.4f}")
        results[mut_name] = view_metrics

    out_file = args.output / "mutation_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_file}")

    # Summary table
    print("\n=== Summary: F1 across mutations ===")
    print(f"{'mutation':12s}  {'main':>8s}  {'S':>8s}  {'L':>8s}  {'A':>8s}")
    for mut_name in args.mutations:
        r = results[mut_name]
        print(f"{mut_name:12s}  "
              f"{r['main']['f1']:>8.4f}  {r['S']['f1']:>8.4f}  "
              f"{r['L']['f1']:>8.4f}  {r['A']['f1']:>8.4f}")

    print("\n=== Summary: Recall@FPR=0.001 across mutations ===")
    print(f"{'mutation':12s}  {'main':>8s}  {'S':>8s}  {'L':>8s}  {'A':>8s}")
    for mut_name in args.mutations:
        r = results[mut_name]
        def fmt(x): return f"{x:.4f}" if x is not None else "  N/A "
        print(f"{mut_name:12s}  "
              f"{fmt(r['main']['recall_at_fpr_001']):>8s}  "
              f"{fmt(r['S']['recall_at_fpr_001']):>8s}  "
              f"{fmt(r['L']['recall_at_fpr_001']):>8s}  "
              f"{fmt(r['A']['recall_at_fpr_001']):>8s}")


if __name__ == "__main__":
    main()
