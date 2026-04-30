#!/usr/bin/env python3
"""Build a held-out evaluation set from SQLiV5.

SQLiV5 is the latest version of nidnogg/sqliv5-dataset and is a SUPERSET of
SQLiV3 (which was already used in training). We extract:
  - SQLi rows from V5 that do NOT appear in our existing train+val+test pool
  - Some V5 valid rows that are also held-out

Output: data/sqliv5_eval.jsonl   {"user_input", "label", "source"}
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
V5_PATH = ROOT.parent / "external" / "payload_sources" / "sqliv5.json"
OUT = ROOT / "data" / "sqliv5_eval.jsonl"


def norm(p: str) -> str:
    p = p.strip()
    p = re.sub(r"  +", " ", p)
    if p.startswith('"') and p.endswith('"') and len(p) > 2:
        inner = p[1:-1]
        if '"' not in inner:
            p = inner
    return p


def main():
    # Load known training pool (attacks + benigns)
    attacks = json.load(open(ROOT / "data" / "attack_pool.json", encoding="utf-8"))
    benigns = json.load(open(ROOT / "data" / "benign_pool.json", encoding="utf-8"))
    known = set(a["payload"] for a in attacks) | set(b["payload"] for b in benigns)
    print(f"Known pool size: {len(known)}")

    # Load V5
    v5 = json.load(open(V5_PATH, encoding="utf-8"))
    print(f"SQLiV5 entries: {len(v5)}")

    novel_sqli = []
    novel_valid = []
    overlap_sqli = 0
    overlap_valid = 0
    for d in v5:
        t = d.get("type")
        p = d.get("pattern", "")
        if not p:
            continue
        p = norm(p)
        if not p:
            continue
        if t == "sqli":
            if p in known:
                overlap_sqli += 1
            else:
                novel_sqli.append(p)
        elif t == "valid":
            if p in known:
                overlap_valid += 1
            else:
                novel_valid.append(p)

    print(f"\nV5 sqli   — novel: {len(novel_sqli):>5d}, overlap: {overlap_sqli:>5d}")
    print(f"V5 valid  — novel: {len(novel_valid):>5d}, overlap: {overlap_valid:>5d}")

    # Dedup
    novel_sqli = list(dict.fromkeys(novel_sqli))
    novel_valid = list(dict.fromkeys(novel_valid))
    print(f"After dedup — sqli: {len(novel_sqli)}, valid: {len(novel_valid)}")

    # Show samples
    print(f"\nSample novel sqli (first 5):")
    for p in novel_sqli[:5]:
        print(f"  {p[:120]!r}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for p in novel_sqli:
            f.write(json.dumps({"user_input": p, "label": "attack",
                                  "source": "sqliv5_novel", "subtype": None,
                                  "technique": None}, ensure_ascii=False) + "\n")
        for p in novel_valid:
            f.write(json.dumps({"user_input": p, "label": "benign",
                                  "source": "sqliv5_novel", "subtype": "real_param",
                                  "technique": None}, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUT}: {len(novel_sqli)} sqli + {len(novel_valid)} valid")


if __name__ == "__main__":
    main()
