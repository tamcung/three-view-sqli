#!/usr/bin/env python3
"""Phase 6: AST-equivalence-class disjoint train/val/test splits.

For each sample, we compute the AST signature of its SQL (using the
constant-collapse normalization from validate_payloads.ast_signature). Samples
whose SQL fails to parse fall into a special "PARSE_FAIL_<hash>" bucket so
that twins of attacks that broke parsing still group together by user_input.

Splitting unit is the AST signature, not the individual sample. We greedily
assign sig groups to train/val/test to hit target proportions (70/15/15),
respecting label balance within each split.

Outputs:
  data/splits/train.jsonl
  data/splits/val.jsonl
  data/splits/test.jsonl
  data/splits/split_meta.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
logging.getLogger("sqlglot").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from sql_utils import parse_strict, ast_signature, wrap

INPUT_DATASET = ROOT / "data" / "synthesized_dataset.jsonl"
OUT_DIR = ROOT / "data" / "splits"


def compute_sig(user_input: str) -> str:
    """Wrap user_input in a canonical mini-template, parse, signature.
    Falls back to text-hash bucket on parse failure."""
    wrapped = wrap(user_input, slot_context=None)  # default numeric wrapper
    tree = parse_strict(wrapped)
    if tree is not None:
        sig = ast_signature(tree)
        return f"AST_{hashlib.md5(repr(sig).encode('utf-8')).hexdigest()[:16]}"
    norm = " ".join(user_input.lower().split())
    return f"PF_{hashlib.md5(norm.encode('utf-8')).hexdigest()[:16]}"


def compute_sig_batch(samples_chunk):
    out = []
    for idx, user_input in samples_chunk:
        out.append((idx, compute_sig(user_input)))
    return out


def run_random_split(samples: list, args) -> None:
    """Stratified random split: shuffle within each (label, source, subtype/technique)
    stratum and partition 70/15/15. Independent samples — no AST-level
    disjointness constraint."""
    rng = random.Random(args.seed)

    def stratum_key(s):
        if s["label"] == "attack":
            return ("attack", s.get("source") or "unk", s.get("technique") or "unk")
        return ("benign", s.get("source") or "unk", s.get("subtype") or "unk")

    by_stratum = defaultdict(list)
    for i, s in enumerate(samples):
        by_stratum[stratum_key(s)].append(i)

    print(f"\n  {len(by_stratum)} strata")
    split_assignments = defaultdict(list)
    for stratum, indices in by_stratum.items():
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        # remainder goes to test (avoids rounding shrinking train)
        n_test = n - n_train - n_val
        split_assignments["train"].extend(indices[:n_train])
        split_assignments["val"].extend(indices[n_train:n_train + n_val])
        split_assignments["test"].extend(indices[n_train + n_val:])

    # Shuffle within each split for ordering randomness
    for k in split_assignments:
        rng.shuffle(split_assignments[k])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\n  Split assignment results\n{'='*70}")
    for split_name in ("train", "val", "test"):
        idxs = split_assignments[split_name]
        n_atk = sum(1 for i in idxs if samples[i]["label"] == "attack")
        n_ben = sum(1 for i in idxs if samples[i]["label"] == "benign")
        print(f"  {split_name:6s}: total={len(idxs):>6d}  attack={n_atk:>6d}  benign={n_ben:>6d}  "
              f"({n_atk / max(len(idxs), 1) * 100:.1f}%/{n_ben / max(len(idxs), 1) * 100:.1f}%)")
        path = OUT_DIR / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for idx in idxs:
                f.write(json.dumps(samples[idx], ensure_ascii=False) + "\n")
        print(f"    wrote {path}")

    # Source / subtype / technique breakdown per split
    print(f"\n{'='*70}\n  Per-split detail\n{'='*70}")
    for split_name in ("train", "val", "test"):
        idxs = split_assignments[split_name]
        srcs = Counter(samples[i]["source"] for i in idxs)
        techs = Counter(samples[i].get("technique") for i in idxs if samples[i]["label"] == "attack")
        subs = Counter(samples[i].get("subtype") for i in idxs if samples[i]["label"] == "benign")
        print(f"\n  {split_name}:")
        print(f"    Sources:         {dict(srcs)}")
        print(f"    Techniques:      {dict(techs)}")
        print(f"    Benign subtypes: {dict(subs)}")

    meta = {
        "input_dataset": str(INPUT_DATASET),
        "total_samples": len(samples),
        "split_mode": "random",
        "n_strata": len(by_stratum),
        "split_proportions": {"train": args.train, "val": args.val, "test": args.test},
        "split_sizes": {k: len(v) for k, v in split_assignments.items()},
        "seed": args.seed,
    }
    with open(OUT_DIR / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  Wrote {OUT_DIR / 'split_meta.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mode", choices=("random", "ast_disjoint"),
                          default="random",
                          help="random = shuffle + stratify by label; "
                               "ast_disjoint = same AST signature can't appear "
                               "in train and test")
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, \
        "Splits must sum to 1.0"

    print(f"Loading {INPUT_DATASET} ...")
    with open(INPUT_DATASET, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f]
    print(f"  {len(samples)} samples loaded")
    print(f"  split mode: {args.mode}")

    if args.mode == "random":
        run_random_split(samples, args)
        return

    # Compute sigs in parallel
    print(f"\nComputing AST signatures with {args.workers} workers...")
    payload_for_workers = [(i, s["user_input"]) for i, s in enumerate(samples)]
    chunk_size = max(1, len(payload_for_workers) // (args.workers * 4))
    chunks = [
        payload_for_workers[i:i + chunk_size]
        for i in range(0, len(payload_for_workers), chunk_size)
    ]

    sig_for_idx = {}
    if args.workers > 1:
        with mp.Pool(processes=args.workers) as pool:
            done = 0
            for batch in pool.imap_unordered(compute_sig_batch, chunks):
                for idx, sig in batch:
                    sig_for_idx[idx] = sig
                done += len(batch)
                if done % 20000 == 0 or done == len(samples):
                    print(f"  {done}/{len(samples)} done")
    else:
        for batch in chunks:
            for idx, sig in compute_sig_batch(batch):
                sig_for_idx[idx] = sig

    # Annotate samples with sig
    for i, s in enumerate(samples):
        s["ast_sig"] = sig_for_idx[i]

    # Group samples by sig
    sig_to_indices = defaultdict(list)
    for i, s in enumerate(samples):
        sig_to_indices[s["ast_sig"]].append(i)
    print(f"\nUnique AST signatures: {len(sig_to_indices)}")
    sig_size_counter = Counter(len(v) for v in sig_to_indices.values())
    print(f"  Top sig group sizes: {sig_size_counter.most_common(5)}")

    # Distinguish parseable vs parse_fail sigs
    n_pf_sigs = sum(1 for sig in sig_to_indices if sig.startswith("PF_"))
    n_ast_sigs = sum(1 for sig in sig_to_indices if sig.startswith("AST_"))
    n_pf_samples = sum(len(v) for sig, v in sig_to_indices.items() if sig.startswith("PF_"))
    n_ast_samples = sum(len(v) for sig, v in sig_to_indices.items() if sig.startswith("AST_"))
    print(f"  AST sigs:  {n_ast_sigs}  ({n_ast_samples} samples)")
    print(f"  PF sigs:   {n_pf_sigs}   ({n_pf_samples} samples)")

    # Assign sig groups to splits with greedy balanced allocation
    rng = random.Random(args.seed)
    n_total = len(samples)
    targets = {
        "train": int(n_total * args.train),
        "val":   int(n_total * args.val),
        "test":  int(n_total * args.test),
    }
    # Adjust train to get exact total
    targets["train"] = n_total - targets["val"] - targets["test"]

    print(f"\nTarget split sizes: {targets}")

    # Each sig group gets a stratum signature: (label, context, subtype/technique).
    # We then run a per-stratum greedy AST-disjoint allocation. If a sig
    # spans multiple strata (rare, due to constant collapse), it is owned by
    # its dominant stratum but its full sample list moves together.

    def sample_stratum(s):
        if s["label"] == "attack":
            return ("attack", s.get("source") or "unk", s.get("technique") or "unk")
        return ("benign", s.get("source") or "unk", s.get("subtype") or "unk")

    # Per-sig dominant stratum
    sig_dominant_stratum = {}
    for sig, indices in sig_to_indices.items():
        strata = Counter(sample_stratum(samples[i]) for i in indices)
        sig_dominant_stratum[sig] = strata.most_common(1)[0][0]

    # Group sigs by their dominant stratum
    stratum_to_sigs = defaultdict(list)
    for sig, strat in sig_dominant_stratum.items():
        stratum_to_sigs[strat].append(sig)

    # Per-split tracking
    split_label_counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    split_assignments = defaultdict(list)

    print(f"\n  Stratifying split across {len(stratum_to_sigs)} strata...")
    for stratum, sigs_in_stratum in stratum_to_sigs.items():
        # Shuffle sigs within stratum for randomized partition
        rng.shuffle(sigs_in_stratum)
        # Sort by size desc (so larger groups are allocated first; smaller
        # groups can compensate)
        sigs_in_stratum.sort(key=lambda sg: -len(sig_to_indices[sg]))

        n_total_in_stratum = sum(len(sig_to_indices[sg]) for sg in sigs_in_stratum)
        per_split_target = {
            "train": int(n_total_in_stratum * args.train),
            "val":   int(n_total_in_stratum * args.val),
            "test":  int(n_total_in_stratum * args.test),
        }
        per_split_target["train"] = (
            n_total_in_stratum - per_split_target["val"] - per_split_target["test"]
        )
        per_split_current = {"train": 0, "val": 0, "test": 0}

        for sig in sigs_in_stratum:
            indices = sig_to_indices[sig]
            scores = {}
            for split_name in ("train", "val", "test"):
                t = max(per_split_target[split_name], 1)
                scores[split_name] = per_split_current[split_name] / t
            chosen = min(scores.keys(), key=lambda k: scores[k])
            for idx in indices:
                label = samples[idx]["label"]
                split_label_counts[chosen][label] += 1
                split_assignments[chosen].append(idx)
            per_split_current[chosen] += len(indices)

    # Report
    print(f"\n{'='*70}")
    print(f"  Split assignment results")
    print(f"{'='*70}")
    for split_name in ("train", "val", "test"):
        n = sum(split_label_counts[split_name].values())
        labels = dict(split_label_counts[split_name])
        attack = labels.get("attack", 0)
        benign = labels.get("benign", 0)
        print(f"  {split_name:6s}: total={n:>6d}  attack={attack:>6d}  benign={benign:>6d}  "
              f"({attack / max(n, 1) * 100:.1f}%/{benign / max(n, 1) * 100:.1f}%)")

    # Verify disjointness: AST sigs in train should not appear in test
    train_sigs = {samples[i]["ast_sig"] for i in split_assignments["train"]}
    val_sigs   = {samples[i]["ast_sig"] for i in split_assignments["val"]}
    test_sigs  = {samples[i]["ast_sig"] for i in split_assignments["test"]}
    print(f"\n  Disjointness check:")
    print(f"    train ∩ val:   {len(train_sigs & val_sigs)}")
    print(f"    train ∩ test:  {len(train_sigs & test_sigs)}")
    print(f"    val ∩ test:    {len(val_sigs & test_sigs)}")

    # Write splits
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        path = OUT_DIR / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for idx in split_assignments[split_name]:
                f.write(json.dumps(samples[idx], ensure_ascii=False) + "\n")
        print(f"  Wrote {path}: {len(split_assignments[split_name])} samples")

    # Save metadata
    meta = {
        "input_dataset": str(INPUT_DATASET),
        "total_samples": len(samples),
        "n_unique_sigs": len(sig_to_indices),
        "n_ast_sigs": n_ast_sigs,
        "n_pf_sigs": n_pf_sigs,
        "split_proportions": {"train": args.train, "val": args.val, "test": args.test},
        "split_sizes": {k: sum(split_label_counts[k].values()) for k in split_label_counts},
        "split_label_counts": {k: dict(split_label_counts[k]) for k in split_label_counts},
        "seed": args.seed,
        "disjointness": {
            "train_val_overlap": len(train_sigs & val_sigs),
            "train_test_overlap": len(train_sigs & test_sigs),
            "val_test_overlap": len(val_sigs & test_sigs),
        },
    }
    with open(OUT_DIR / "split_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n  Wrote split_meta.json")

    # Per-split detail stats
    print(f"\n{'='*70}")
    print(f"  Per-split detail")
    print(f"{'='*70}")
    for split_name in ("train", "val", "test"):
        indices = split_assignments[split_name]
        print(f"\n  {split_name}:")
        srcs = Counter(samples[i]["source"] for i in indices)
        techs = Counter(samples[i].get("technique") for i in indices if samples[i]["label"] == "attack")
        subtypes = Counter(samples[i].get("subtype") for i in indices if samples[i]["label"] == "benign")
        print(f"    Sources:         {dict(srcs)}")
        print(f"    Techniques:      {dict(techs)}")
        print(f"    Benign subtypes: {dict(subtypes)}")


if __name__ == "__main__":
    main()
