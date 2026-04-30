#!/usr/bin/env python3
"""Build unified attack and benign pools from three external sources.

Sources:
  - HttpParamsDataset:  attack_type ∈ {sqli, norm} → attack and benign rows
  - SQLiV3:             type ∈ {sqli, valid}      → attack and benign rows
  - sqlmap XML payloads: <test> → attack templates (with placeholder expansion)

Output:
  data/attack_pool.json  list of {payload, source, technique, parse_ok}
  data/benign_pool.json  list of {payload, source, subtype}

LLM-generated hard-negative benigns are added in a separate step
(scripts/generate_llm_benigns.py) and merged here on second pass.
"""
from __future__ import annotations
import csv
import hashlib
import json
import logging
import random
import re
import sys
import warnings
from collections import Counter
from pathlib import Path
from urllib.parse import unquote

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
logging.getLogger("sqlglot").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT.parent / "external" / "payload_sources"

HPD_CSV = EXT / "httpparams_full.csv"
SQLIV3_JSON = EXT / "sqliv3.json"
SQLMAP_XML_DIR = EXT / "sqlmap_xml"
LLM_BENIGN_FILE = ROOT / "data" / "llm_benigns.json"  # produced by generate_llm_benigns.py

OUT_ATTACK = ROOT / "data" / "attack_pool.json"
OUT_BENIGN = ROOT / "data" / "benign_pool.json"


# ============================================================
# 1. HttpParamsDataset
# ============================================================
def parse_httpparams() -> tuple[list[dict], list[dict]]:
    """Returns (sqli_rows, norm_rows). Strips a 'p=' prefix and URL-decodes."""
    attack_rows, benign_rows = [], []
    with open(HPD_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            payload = r["payload"]
            # HPD payloads sometimes have 'p=' prefix from URL form encoding;
            # strip it and decode (a few of the rows are URL encoded).
            if payload.startswith("p="):
                payload = payload[2:]
            try:
                decoded = unquote(payload)
                if "%" in payload and len(decoded) < len(payload):
                    payload = decoded
            except Exception:
                pass
            payload = payload.strip()
            if not payload:
                continue
            if r["attack_type"] == "sqli":
                attack_rows.append({"payload": payload, "source": "httpparams"})
            elif r["attack_type"] == "norm":
                benign_rows.append({"payload": payload, "source": "httpparams_norm",
                                     "subtype": "real_param"})
            # ignore xss / cmdi / path-traversal
    return attack_rows, benign_rows


# ============================================================
# 2. SQLiV3
# ============================================================
def parse_sqliv3() -> tuple[list[dict], list[dict]]:
    """Returns (sqli, valid). Normalizes excessive whitespace from the corpus."""
    with open(SQLIV3_JSON, encoding="utf-8") as f:
        data = json.load(f)
    attack_rows, benign_rows = [], []
    for d in data:
        payload = d.get("pattern", "").strip()
        if not payload:
            continue
        # SQLiV3 has weird extra spaces — collapse runs of 2+ spaces to 1
        payload = re.sub(r"  +", " ", payload)
        # Also strip wrapping quotes if duplicated
        if payload.startswith('"') and payload.endswith('"') and len(payload) > 2:
            # only strip if truly a wrapping quote pair — heuristic
            inner = payload[1:-1]
            if '"' not in inner:
                payload = inner
        if d.get("type") == "sqli":
            attack_rows.append({"payload": payload, "source": "sqliv3"})
        elif d.get("type") == "valid":
            benign_rows.append({"payload": payload, "source": "sqliv3_valid",
                                 "subtype": "real_param"})
    return attack_rows, benign_rows


# ============================================================
# 3. sqlmap XML — placeholder expansion
# ============================================================
import xml.etree.ElementTree as ET

# sqlmap technique codes → readable name
SQLMAP_TECHNIQUE = {
    "1": "boolean_blind",
    "2": "error_based",
    "3": "inline_query",
    "4": "stacked_queries",
    "5": "time_blind",
    "6": "union_query",
}

# Known placeholders in sqlmap payloads
PLACEHOLDER_RE = re.compile(r"\[(RANDNUM|RANDSTR|SLEEPTIME|RANDNUM\d+|"
                             r"INFERENCE|ORIGINAL_VALUE|PAYLOAD|UNION|GENERIC_SQL_COMMENT|"
                             r"DELIMITER_START|DELIMITER_END|"
                             r"DBMS_DELIMITER|MYSQL_DELIMITER|"
                             r"DOLLAR_TOKEN_START|DOLLAR_TOKEN_END|"
                             r"AT_REPLACE|ASTERISK|CHAR|"
                             r"DBMS_FUNCTION|RAND_FUNCTION)\]")


def _expand_payload(template: str, rng: random.Random) -> str:
    """Replace placeholders with concrete values. Multiple expansions per
    template are produced by callers via different rng seeds."""
    def repl(m):
        ph = m.group(1)
        if ph == "RANDNUM" or ph.startswith("RANDNUM"):
            return str(rng.randint(1, 99999))
        if ph == "RANDSTR":
            return "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=6))
        if ph == "SLEEPTIME":
            return "5"
        if ph == "INFERENCE":
            return "1=1"
        if ph == "ORIGINAL_VALUE":
            return "1"
        if ph == "PAYLOAD":
            return ""           # the inner payload — empty replacement
        if ph == "GENERIC_SQL_COMMENT":
            return "-- "
        if ph in ("DELIMITER_START", "DELIMITER_END",
                   "DBMS_DELIMITER", "MYSQL_DELIMITER"):
            return rng.choice(["'", "\"", ""])
        if ph in ("DOLLAR_TOKEN_START", "DOLLAR_TOKEN_END"):
            return "$"
        if ph == "AT_REPLACE":
            return "@"
        if ph == "ASTERISK":
            return "*"
        if ph == "CHAR":
            return "CHAR"
        if ph in ("DBMS_FUNCTION", "RAND_FUNCTION"):
            return "RAND()"
        if ph == "UNION":
            return "UNION"
        return m.group(0)
    return PLACEHOLDER_RE.sub(repl, template)


def parse_sqlmap(expansions_per_template: int = 5, seed: int = 42) -> list[dict]:
    """Load real-exploit payloads produced by build_sqlmap_exploits.py.

    The old XML-template expansion has been replaced: those payloads were
    sqlmap's DETECTION probes (e.g. `AND [RANDNUM]=[RANDNUM]`), not actual
    exploitation. The new pipeline reads the <vector> field of each test
    and substitutes [INFERENCE]/[QUERY]/[UNION] with realistic exfiltration
    expressions. See scripts/build_sqlmap_exploits.py.
    """
    exploits_path = ROOT / "data" / "sqlmap_exploits.json"
    if not exploits_path.exists():
        print(f"  WARN: {exploits_path} not present — run scripts/build_sqlmap_exploits.py first")
        return []
    with open(exploits_path, encoding="utf-8") as f:
        rows = json.load(f)
    out = []
    for r in rows:
        out.append({
            "payload": r["payload"],
            "source": "sqlmap",   # keep source name to preserve downstream filters
            "technique": r.get("technique", "unknown"),
            "sqlmap_title": r.get("sqlmap_title", ""),
        })
    return out


# ============================================================
# 4. Optional LLM-generated benign hard negatives
# ============================================================
def parse_llm_benigns() -> list[dict]:
    if not LLM_BENIGN_FILE.exists():
        print(f"  ({LLM_BENIGN_FILE} not present yet — skip LLM benigns)")
        return []
    with open(LLM_BENIGN_FILE, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for d in data:
        text = d.get("text") or d.get("payload")
        sub = d.get("subtype", "llm_hard_negative")
        if text:
            rows.append({"payload": text, "source": "llm", "subtype": sub})
    return rows


# ============================================================
# Dedup helper
# ============================================================
def dedupe(rows: list[dict], key: str = "payload") -> list[dict]:
    seen = set()
    out = []
    for r in rows:
        k = r[key]
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  Building unified pools from external sources")
    print("=" * 70)

    print("\n--- HttpParamsDataset ---")
    hpd_atk, hpd_ben = parse_httpparams()
    print(f"  sqli: {len(hpd_atk)}   norm: {len(hpd_ben)}")

    print("\n--- SQLiV3 ---")
    sv3_atk, sv3_ben = parse_sqliv3()
    print(f"  sqli: {len(sv3_atk)}   valid: {len(sv3_ben)}")

    print("\n--- sqlmap XML ---")
    sqm_atk = parse_sqlmap(expansions_per_template=8)
    print(f"  expanded payloads: {len(sqm_atk)}")
    print(f"  by technique:")
    for t, n in Counter(p["technique"] for p in sqm_atk).most_common():
        print(f"    {t:25s} {n}")

    print("\n--- LLM benigns ---")
    llm_ben = parse_llm_benigns()
    print(f"  rows: {len(llm_ben)}")

    # Combine
    all_attacks = hpd_atk + sv3_atk + sqm_atk
    all_benigns = hpd_ben + sv3_ben + llm_ben

    # Dedup on payload string
    print("\n--- Combine + dedupe ---")
    print(f"  before: attacks={len(all_attacks)}  benigns={len(all_benigns)}")
    all_attacks = dedupe(all_attacks)
    all_benigns = dedupe(all_benigns)
    print(f"  after:  attacks={len(all_attacks)}  benigns={len(all_benigns)}")

    # Cross-pool dedup: a string can't be both attack AND benign — drop overlaps
    benign_strs = {r["payload"] for r in all_benigns}
    attack_strs = {r["payload"] for r in all_attacks}
    overlap = benign_strs & attack_strs
    if overlap:
        print(f"  cross-pool overlap: {len(overlap)} strings appear in both → kept as attacks (drop from benign)")
        all_benigns = [r for r in all_benigns if r["payload"] not in overlap]

    # Length filter — drop empty / extremely long (>2000 chars likely garbage)
    def length_ok(r):
        n = len(r["payload"])
        return 1 <= n <= 2000
    n_atk0, n_ben0 = len(all_attacks), len(all_benigns)
    all_attacks = [r for r in all_attacks if length_ok(r)]
    all_benigns = [r for r in all_benigns if length_ok(r)]
    print(f"  length filter: attacks {n_atk0}→{len(all_attacks)}, benigns {n_ben0}→{len(all_benigns)}")

    # Annotate each with id
    for r in all_attacks:
        r["id"] = "atk_" + hashlib.md5(r["payload"].encode("utf-8")).hexdigest()[:12]
        r["length"] = len(r["payload"])
    for r in all_benigns:
        r["id"] = "ben_" + hashlib.md5(r["payload"].encode("utf-8")).hexdigest()[:12]
        r["length"] = len(r["payload"])

    # Per-source stats
    print(f"\n  Attack pool source breakdown:")
    for s, n in Counter(r["source"] for r in all_attacks).most_common():
        print(f"    {s:18s} {n}")
    print(f"  Benign pool source breakdown:")
    for s, n in Counter(r["source"] for r in all_benigns).most_common():
        print(f"    {s:18s} {n}")

    # Write
    OUT_ATTACK.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_ATTACK, "w", encoding="utf-8") as f:
        json.dump(all_attacks, f, ensure_ascii=False, indent=2)
    with open(OUT_BENIGN, "w", encoding="utf-8") as f:
        json.dump(all_benigns, f, ensure_ascii=False, indent=2)
    print(f"\n  Wrote {OUT_ATTACK}: {len(all_attacks)} attacks")
    print(f"  Wrote {OUT_BENIGN}: {len(all_benigns)} benigns")


if __name__ == "__main__":
    main()
