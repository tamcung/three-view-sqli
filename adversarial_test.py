#!/usr/bin/env python3
"""Payload-level adversarial robustness probe.

Apply lexical mutations to attack `user_input` strings (random case, extra
whitespace, inline /**/ comments, tab substitution, mixed) and re-evaluate.
Measures how much each detector's recall drops under each mutation type.

Three detectors compared on the same mutated inputs:
  - three-view model (loaded from a checkpoint)
  - libinjection.is_sqli
  - TF-IDF + LogisticRegression (re-fit on training split)

Output:
  results/<run>/eval/adversarial.json   per-mutation recall + drop
"""
from __future__ import annotations
import argparse
import json
import logging
import random
import re
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


# ============================================================
# Mutations on user_input strings
# ============================================================
SQL_KEYWORDS = {
    "SELECT", "UNION", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "LIKE",
    "BETWEEN", "IS", "NULL", "TRUE", "FALSE", "INSERT", "UPDATE", "DELETE",
    "VALUES", "SET", "INTO", "TABLE", "DROP", "CREATE", "ALTER", "EXEC",
    "EXECUTE", "DECLARE", "BEGIN", "END", "IF", "THEN", "ELSE", "CASE",
    "WHEN", "GROUP", "BY", "ORDER", "HAVING", "DISTINCT", "AS", "JOIN",
    "INNER", "OUTER", "LEFT", "RIGHT", "FULL", "ON", "LIMIT", "OFFSET",
    "ALL", "ANY", "SOME", "EXISTS", "WAITFOR", "DELAY", "SLEEP",
    "BENCHMARK", "EXTRACTVALUE", "UPDATEXML", "VERSION", "USER", "DATABASE",
    "CONCAT", "SUBSTRING", "ASCII", "CHAR", "HEX", "UNHEX",
}


def mut_random_case(s, rng):
    out, i = [], 0
    while i < len(s):
        matched = False
        for kw in sorted(SQL_KEYWORDS, key=len, reverse=True):
            if s[i:i + len(kw)].upper() == kw and (i == 0 or not s[i - 1].isalnum()):
                end = i + len(kw)
                if end == len(s) or not s[end].isalnum():
                    out.append("".join(c.upper() if rng.random() > 0.5 else c.lower() for c in s[i:end]))
                    i = end; matched = True; break
        if not matched:
            out.append(s[i]); i += 1
    return "".join(out)


def mut_extra_space(s, rng):
    parts = s.split(" ")
    return "  ".join(p + (" " if rng.random() > 0.5 else "") for p in parts)


def mut_inline_comment(s, rng):
    out = s
    for kw in sorted(SQL_KEYWORDS, key=len, reverse=True):
        pat = re.compile(rf"\b({kw})\b\s+", flags=re.IGNORECASE)
        out = pat.sub(rf"\1/**/", out, count=1)
    return out


def mut_tab_for_space(s, rng):
    return "".join("\t" if c == " " and rng.random() > 0.5 else c for c in s)


def mut_url_encode(s, rng):
    """Percent-encode common SQL operators randomly."""
    table = {"=": "%3D", "<": "%3C", ">": "%3E", "(": "%28", ")": "%29",
             " ": "%20", "'": "%27", "\"": "%22"}
    return "".join(table.get(c, c) if rng.random() > 0.4 else c for c in s)


def mut_mixed(s, rng):
    s = mut_random_case(s, rng)
    s = mut_extra_space(s, rng)
    return mut_inline_comment(s, rng)


MUTATIONS = {
    "random_case": mut_random_case,
    "extra_space": mut_extra_space,
    "inline_comment": mut_inline_comment,
    "tab_for_space": mut_tab_for_space,
    "url_encode": mut_url_encode,
    "mixed": mut_mixed,
}


# ============================================================
# In-memory dataset for mutated user_inputs
# ============================================================
class MemDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        return self.recs[i]


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(),
                    logging.FileHandler(out_dir / "adversarial.log", encoding="utf-8")],
    )
    log = logging.getLogger("adv")

    # ---- Load test attacks ----
    log.info("Loading test split, filtering to attacks...")
    with open(ROOT / "data" / "splits" / "test.jsonl", encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    attacks = [r for r in rows if r["label"] == "attack"]
    if args.max_samples:
        attacks = rng.sample(attacks, min(args.max_samples, len(attacks)))
    log.info(f"  {len(attacks)} attack samples")

    # ---- Load three-view model ----
    cfg_path = Path(args.checkpoint).parent / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.get("model_variant", "three_view"), cfg.get("model", {})).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    pre = SamplePreprocessor()

    # ---- Train TF-IDF baseline ----
    log.info("Fitting TF-IDF baseline on train split...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    with open(ROOT / "data" / "splits" / "train.jsonl", encoding="utf-8") as f:
        tr = [json.loads(l) for l in f]
    tr_inputs = [r["user_input"] for r in tr]
    tr_y = np.array([1 if r["label"] == "attack" else 0 for r in tr])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                            max_features=50_000, sublinear_tf=True)
    Xtr = vec.fit_transform(tr_inputs)
    clf = LogisticRegression(max_iter=300, C=1.0, solver="liblinear",
                              class_weight="balanced")
    clf.fit(Xtr, tr_y)
    log.info(f"  TF-IDF: {Xtr.shape[1]} features, fit done")

    # ---- Predictors ----
    @torch.no_grad()
    def pred_tv(inputs):
        recs = []
        for s in inputs:
            f = pre(s)
            recs.append({**f, "label_int": 1, "meta": {}})
        loader = DataLoader(MemDataset(recs), batch_size=128, shuffle=False,
                              collate_fn=collate_three_view, num_workers=0)
        logits = []
        for batch in loader:
            batch = move_batch_to(batch, device)
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                  dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(batch["surface_ids"], batch["surface_mask"],
                              batch["lex_ids"], batch["lex_mask"],
                              batch["ast_ids"], batch["ast_mask"],
                              ast_valid=batch["ast_valid"])
            logits.append(out["p_main"].float().cpu().numpy())
        return np.concatenate(logits)

    def pred_libinj(inputs):
        out = []
        for s in inputs:
            r = libinj_is_sqli(s)
            out.append(1 if (r[0] if isinstance(r, tuple) else r) else 0)
        return np.array(out)

    def pred_tfidf(inputs):
        return clf.predict_proba(vec.transform(inputs))[:, 1]

    # ---- Baseline (no mutation) ----
    results = {}
    log.info("\n=== Baseline (no mutation) ===")
    base_inputs = [a["user_input"] for a in attacks]

    t0 = time.time()
    base_logits_tv = pred_tv(base_inputs)
    base_probs_tv = 1.0 / (1.0 + np.exp(-base_logits_tv))
    base_preds_tv = (base_probs_tv >= args.threshold).astype(int)
    base_rec_tv = base_preds_tv.mean()
    log.info(f"  3-view recall: {base_rec_tv:.4f}  ({base_preds_tv.sum()}/{len(attacks)})  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    base_preds_lib = pred_libinj(base_inputs)
    base_rec_lib = base_preds_lib.mean()
    log.info(f"  libinj recall: {base_rec_lib:.4f}  ({base_preds_lib.sum()}/{len(attacks)})  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    base_probs_tf = pred_tfidf(base_inputs)
    base_preds_tf = (base_probs_tf >= 0.5).astype(int)
    base_rec_tf = base_preds_tf.mean()
    log.info(f"  tfidf  recall: {base_rec_tf:.4f}  ({base_preds_tf.sum()}/{len(attacks)})  ({time.time()-t0:.1f}s)")

    results["no_mutation"] = {
        "three_view": float(base_rec_tv),
        "libinjection": float(base_rec_lib),
        "tfidf": float(base_rec_tf),
    }

    # ---- Per-mutation eval ----
    for mut_name, mut_fn in MUTATIONS.items():
        log.info(f"\n=== Mutation: {mut_name} ===")
        mutated = [mut_fn(a["user_input"], rng) for a in attacks]
        n_changed = sum(1 for a, m in zip(attacks, mutated) if a["user_input"] != m)
        log.info(f"  changed inputs: {n_changed}/{len(attacks)}")

        t0 = time.time()
        logits = pred_tv(mutated)
        probs = 1.0 / (1.0 + np.exp(-logits))
        preds = (probs >= args.threshold).astype(int)
        rec_tv = preds.mean()

        t0 = time.time()
        rec_lib = pred_libinj(mutated).mean()

        t0 = time.time()
        probs_tf = pred_tfidf(mutated)
        rec_tf = (probs_tf >= 0.5).astype(int).mean()

        log.info(f"  3-view recall: {rec_tv:.4f}  drop: {base_rec_tv - rec_tv:+.4f}")
        log.info(f"  libinj recall: {rec_lib:.4f}  drop: {base_rec_lib - rec_lib:+.4f}")
        log.info(f"  tfidf  recall: {rec_tf:.4f}  drop: {base_rec_tf - rec_tf:+.4f}")

        results[mut_name] = {
            "three_view": float(rec_tv),
            "three_view_drop": float(base_rec_tv - rec_tv),
            "libinjection": float(rec_lib),
            "libinjection_drop": float(base_rec_lib - rec_lib),
            "tfidf": float(rec_tf),
            "tfidf_drop": float(base_rec_tf - rec_tf),
        }

    # ---- Summary ----
    log.info("\n" + "=" * 70)
    log.info(f"  SUMMARY: recall under each mutation")
    log.info("=" * 70)
    log.info(f"  {'mutation':18s}  {'3-view':>8s}  {'libinj':>8s}  {'tfidf':>8s}")
    for k, v in results.items():
        log.info(f"  {k:18s}  {v['three_view']:>8.4f}  "
                  f"{v['libinjection']:>8.4f}  {v['tfidf']:>8.4f}")

    with open(out_dir / "adversarial.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log.info(f"\n  Wrote {out_dir / 'adversarial.json'}")


if __name__ == "__main__":
    main()
