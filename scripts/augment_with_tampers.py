#!/usr/bin/env python3
"""Augment the attack pool with a subset of tamper-generated variants.

Splits the 36+ non-empty tamper subsets into:
  - TRAIN_TAMPERS: 26 (structural / operator / function / single-layer encoding)
  - HOLDOUT_TAMPERS: 10 (heavy encoding — kept as held-out OOD)

For each TRAIN_TAMPERS subset we sample K variants and add them as new
records into `data/attack_pool.json` (with source = "tamper_aug").

Run order after this:
  python scripts/synthesize_dataset.py
  python scripts/split_dataset.py --mode random
  python scripts/preprocess_dataset.py
  python train.py --config configs/<...>.yaml --output results/<...>/
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
ATTACK_POOL = ROOT / "data" / "attack_pool.json"
TAMPER_DIR = ROOT / "data" / "tamper_oods"

# Held-out: heavy / unusual encoding only (model has no chance unless it sees them)
HOLDOUT_TAMPERS = {
    "base64encode",
    "chardoubleencode",
    "charunicodeencode",
    "charunicodeescape",
    "decentities",
    "hexentities",
    "htmlencode",
    "overlongutf8",
    "overlongutf8more",
    "percentage",
}

# Training augmentation: everything else with non-zero OOD output, plus
# `_no_tamper` is excluded (already overlaps with attack_pool entries).
EXPLICIT_EXCLUDE = {"_no_tamper"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples-per-tamper", type=int, default=500,
                    help="Max variants to sample per training tamper.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=str(ATTACK_POOL),
                    help="Where to write the augmented attack pool. "
                         "Defaults to in-place overwrite of attack_pool.json.")
    args = p.parse_args()
    rng = random.Random(args.seed)

    # ---- Load existing attack pool ----
    with open(ATTACK_POOL, encoding="utf-8") as f:
        attack_pool = json.load(f)
    base_count = len(attack_pool)
    base_strs = set(r["payload"] for r in attack_pool)
    print(f"Loaded base attack_pool: {base_count} payloads")

    # ---- Discover tamper jsonls ----
    if not TAMPER_DIR.exists():
        raise FileNotFoundError(f"Run scripts/apply_tampers.py first")
    files = sorted(TAMPER_DIR.glob("*.jsonl"))
    train_tampers = []
    holdout_tampers = []
    for f in files:
        name = f.stem
        if name in EXPLICIT_EXCLUDE:
            continue
        if name in HOLDOUT_TAMPERS:
            holdout_tampers.append(name)
        else:
            train_tampers.append(name)

    print(f"\n  Training-side tampers ({len(train_tampers)}):")
    for n in train_tampers:
        print(f"    {n}")
    print(f"\n  Held-out tampers ({len(holdout_tampers)}):")
    for n in holdout_tampers:
        print(f"    {n}")

    # ---- Sample augmentation from training tampers ----
    print(f"\n  Sampling {args.samples_per_tamper} variants per training tamper...")
    aug_records = []
    aug_counter = Counter()
    for tamper in train_tampers:
        path = TAMPER_DIR / f"{tamper}.jsonl"
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        if not rows:
            continue
        rng.shuffle(rows)
        sampled = rows[: args.samples_per_tamper]
        for r in sampled:
            payload = r["user_input"]
            if payload in base_strs:
                # Don't duplicate what's already in the original pool
                continue
            base_strs.add(payload)
            aug_records.append({
                "payload": payload,
                "source": "tamper_aug",
                "technique": r.get("technique"),
                "tamper": tamper,
                "id": "tamp_" + hashlib.md5(payload.encode("utf-8")).hexdigest()[:12],
                "length": len(payload),
            })
            aug_counter[tamper] += 1
        print(f"    {tamper:35s}  added {aug_counter[tamper]}")

    print(f"\n  Total augmented records: {len(aug_records)}")
    print(f"  Total attack pool after augment: {base_count + len(aug_records)}")

    # ---- Write augmented attack pool ----
    augmented = attack_pool + aug_records
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(augmented, f, ensure_ascii=False, indent=2)
    print(f"\n  Wrote {args.out}: {len(augmented)} payloads")

    # ---- Save the train/holdout split list (so eval scripts can read it) ----
    split_file = ROOT / "data" / "tamper_split.json"
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump({
            "train_tampers": train_tampers,
            "holdout_tampers": holdout_tampers,
            "samples_per_tamper": args.samples_per_tamper,
            "n_augmented": len(aug_records),
        }, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {split_file}")


if __name__ == "__main__":
    main()
