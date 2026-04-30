#!/usr/bin/env python3
"""Phase 5 (payload-level): build the final dataset from attack + benign pools.

Reads `data/attack_pool.json` and `data/benign_pool.json` (produced by
`build_pools.py`), balances attack vs benign 50/50, up-samples the LLM
hard-negative subset to a target rate, and emits one record per sample.

Each record:
  {
    "user_input": str,
    "label": "attack" | "benign",
    "source": str,           e.g. "httpparams", "sqliv3", "sqlmap", "llm"
    "subtype": str | null,   benign subtype (real_param, llm_keyword_in_text, ...)
    "technique": str | null, attack technique if known (sqlmap entries only)
    "id": str,
  }

Output: data/synthesized_dataset.jsonl
"""
from __future__ import annotations
import argparse
import json
import logging
import random
import sys
import warnings
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
ATTACK_POOL = ROOT / "data" / "attack_pool.json"
BENIGN_POOL = ROOT / "data" / "benign_pool.json"
OUT = ROOT / "data" / "synthesized_dataset.jsonl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-attack", type=int, default=None,
                    help="Cap on attack samples. Default: keep all.")
    p.add_argument("--llm-benign-rate", type=float, default=0.15,
                    help="Within benigns, target fraction from LLM pool. "
                         "Ignored when --no-oversample is set (uses natural rate).")
    p.add_argument("--no-oversample", action="store_true",
                    help="Use each unique benign / attack at most once. If "
                         "pools are unbalanced after dedup, the larger side "
                         "is subsampled to match the smaller side.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading {ATTACK_POOL} ...")
    attacks = json.load(open(ATTACK_POOL, encoding="utf-8"))
    print(f"  {len(attacks)} unique attacks")

    print(f"Loading {BENIGN_POOL} ...")
    benigns = json.load(open(BENIGN_POOL, encoding="utf-8"))
    print(f"  {len(benigns)} unique benigns")

    # ---- Subset attacks ----
    if args.target_attack and args.target_attack < len(attacks):
        rng.shuffle(attacks)
        attacks = attacks[:args.target_attack]
        print(f"  capped attacks at {len(attacks)}")
    n_attack = len(attacks)

    # ---- Build benign side ----
    llm_benigns = [b for b in benigns if b.get("source") == "llm"]
    other_benigns = [b for b in benigns if b.get("source") != "llm"]
    rng.shuffle(other_benigns)
    rng.shuffle(llm_benigns)

    if args.no_oversample:
        # Use every unique benign at most once. Total benigns = pool size.
        chosen_other = other_benigns
        chosen_llm = llm_benigns
        chosen_benigns = chosen_other + chosen_llm
        # Balance to 50/50 by subsampling the larger side
        if len(chosen_benigns) < n_attack:
            rng.shuffle(attacks)
            attacks = attacks[:len(chosen_benigns)]
            print(f"  subsampled attacks to {len(attacks)} to match benigns")
            n_attack = len(attacks)
        elif len(chosen_benigns) > n_attack:
            # benign pool larger — subsample but try to keep LLM share intact
            keep_other = max(0, min(len(chosen_other), n_attack - len(chosen_llm)))
            chosen_other = chosen_other[:keep_other]
            keep_llm = max(0, n_attack - len(chosen_other))
            chosen_llm = chosen_llm[:keep_llm]
            chosen_benigns = chosen_other + chosen_llm
            print(f"  subsampled benigns to {len(chosen_benigns)} to match attacks")
    else:
        # Original behavior: enforce LLM rate via with-replacement up-sampling
        n_target_llm = int(n_attack * args.llm_benign_rate)
        n_target_other = n_attack - n_target_llm
        chosen_other = other_benigns[:n_target_other]
        if len(chosen_other) < n_target_other:
            deficit = n_target_other - len(chosen_other)
            chosen_other.extend(rng.choices(other_benigns, k=deficit))
        chosen_llm = []
        if not llm_benigns:
            print(f"  WARN: no LLM benigns in pool")
        else:
            chosen_llm = rng.choices(llm_benigns, k=n_target_llm)
        chosen_benigns = chosen_other + chosen_llm

    rng.shuffle(chosen_benigns)
    print(f"  benigns: {len(chosen_benigns)}  ({len(chosen_other)} other + {len(chosen_llm)} llm)")

    # ---- Emit records ----
    samples = []
    for a in attacks:
        samples.append({
            "user_input": a["payload"],
            "label": "attack",
            "source": a["source"],
            "subtype": None,
            "technique": a.get("technique"),
            "id": a["id"],
        })
    for b in chosen_benigns:
        samples.append({
            "user_input": b["payload"],
            "label": "benign",
            "source": b["source"],
            "subtype": b.get("subtype"),
            "technique": None,
            "id": b["id"],
        })

    rng.shuffle(samples)

    # ---- Write ----
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- Stats ----
    print(f"\n{'='*60}")
    print(f"  Final dataset")
    print(f"{'='*60}")
    print(f"  Total: {len(samples)}")
    print(f"  Attack: {sum(1 for s in samples if s['label']=='attack')}")
    print(f"  Benign: {sum(1 for s in samples if s['label']=='benign')}")

    by_src = Counter((s["label"], s["source"]) for s in samples)
    print(f"\n  By (label, source):")
    for k, n in by_src.most_common():
        print(f"    {k}  {n}")

    # length stats
    import statistics
    L = [len(s["user_input"]) for s in samples]
    print(f"\n  user_input length: min={min(L)} max={max(L)} median={statistics.median(L):.0f} mean={statistics.mean(L):.1f}")
    print(f"  Wrote {OUT}")


if __name__ == "__main__":
    main()
