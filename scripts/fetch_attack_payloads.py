#!/usr/bin/env python3
"""
Fetch and curate canonical SQLi payloads from SecLists + PayloadsAllTheThings.

Output:
  data/attack_payloads.json
    [{"payload": str, "source": str, "category": str, "length": int}, ...]

Process:
  1. Download each source file
  2. Strip comment/blank lines and surrounding whitespace
  3. Tag each payload with source filename and high-level category
  4. Deduplicate across sources (preserve first-seen source)
  5. Save with metadata
"""
from __future__ import annotations
import json
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# Source definitions
# ============================================================
SECLISTS_BASE = "https://raw.githubusercontent.com/danielmiessler/SecLists/master/Fuzzing/Databases/SQLi"
PATTHINGS_BASE = "https://raw.githubusercontent.com/swisskyrepo/PayloadsAllTheThings/master/SQL%20Injection/Intruder"

SOURCES = [
    # SecLists
    (f"{SECLISTS_BASE}/Generic-BlindSQLi.fuzzdb.txt",        "seclists_generic_blind",       "blind"),
    (f"{SECLISTS_BASE}/Generic-SQLi.txt",                    "seclists_generic",             "mixed"),
    (f"{SECLISTS_BASE}/MSSQL.fuzzdb.txt",                    "seclists_mssql",               "mssql_specific"),
    (f"{SECLISTS_BASE}/MySQL.fuzzdb.txt",                    "seclists_mysql",               "mysql_specific"),
    (f"{SECLISTS_BASE}/MySQL-SQLi-Login-Bypass.fuzzdb.txt",  "seclists_mysql_login_bypass",  "auth_bypass"),
    (f"{SECLISTS_BASE}/Oracle.fuzzdb.txt",                   "seclists_oracle",              "oracle_specific"),
    (f"{SECLISTS_BASE}/SQLi-Polyglots.txt",                  "seclists_polyglots",           "polyglot"),
    (f"{SECLISTS_BASE}/quick-SQLi.txt",                      "seclists_quick",               "quick_short"),
    # PayloadsAllTheThings
    (f"{PATTHINGS_BASE}/Auth_Bypass.txt",                    "patthings_auth_bypass",        "auth_bypass"),
    (f"{PATTHINGS_BASE}/Auth_Bypass2.txt",                   "patthings_auth_bypass2",       "auth_bypass"),
    (f"{PATTHINGS_BASE}/Generic_ErrorBased.txt",             "patthings_error_based",        "error_based"),
    (f"{PATTHINGS_BASE}/Generic_TimeBased.txt",              "patthings_time_based",         "time_based"),
    (f"{PATTHINGS_BASE}/Generic_UnionSelect.txt",            "patthings_union_select",       "union_based"),
    (f"{PATTHINGS_BASE}/SQLi_Polyglots.txt",                 "patthings_polyglots",          "polyglot"),
]

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "attack_payloads.json"


# ============================================================
# Cleaning rules
# ============================================================
def is_comment_or_meta(line: str) -> bool:
    """Skip comment lines, regex notes, etc."""
    s = line.strip()
    if not s:
        return True
    if s.startswith("#"):
        return True
    # PayloadsAllTheThings uses some inline notes with leading spaces
    return False


def is_too_short_or_trivial(payload: str) -> bool:
    """Skip payloads that are too short to be a real attack."""
    s = payload.strip()
    if len(s) < 2:
        return True
    # Single-character "shock" payloads like '-', ' ', '&' are too generic
    if len(s) == 1:
        return True
    return False


def normalize(payload: str) -> str:
    """Light normalization: strip leading/trailing whitespace and trailing tabs."""
    # Remove inline comments after a tab (some files have "<payload>\t<comment>")
    if "\t" in payload:
        payload = payload.split("\t", 1)[0]
    return payload.strip()


# ============================================================
# Fetcher
# ============================================================
def fetch(url: str) -> list[str]:
    print(f"  fetching {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "three-view-sqli/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace").splitlines()


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_payloads = []
    seen = set()  # deduplication
    source_counts = Counter()
    category_counts = Counter()
    skipped_counts = Counter()

    for url, source_name, category in SOURCES:
        try:
            lines = fetch(url)
        except Exception as e:
            print(f"    ERROR fetching {source_name}: {e}")
            continue

        kept = 0
        for line in lines:
            if is_comment_or_meta(line):
                skipped_counts["meta"] += 1
                continue
            payload = normalize(line)
            if not payload:
                skipped_counts["empty"] += 1
                continue
            if is_too_short_or_trivial(payload):
                skipped_counts["too_short"] += 1
                continue
            if payload in seen:
                skipped_counts["duplicate"] += 1
                continue
            seen.add(payload)
            all_payloads.append({
                "payload": payload,
                "source": source_name,
                "category": category,
                "length": len(payload),
            })
            kept += 1
            source_counts[source_name] += 1
            category_counts[category] += 1
        print(f"    kept {kept} from {source_name}")

    print(f"\nTotal unique payloads: {len(all_payloads)}")
    print(f"\nBy source:")
    for src, cnt in source_counts.most_common():
        print(f"  {src:35s} {cnt:>5d}")
    print(f"\nBy category:")
    for cat, cnt in category_counts.most_common():
        print(f"  {cat:25s} {cnt:>5d}")
    print(f"\nSkipped:")
    for reason, cnt in skipped_counts.most_common():
        print(f"  {reason:15s} {cnt:>5d}")

    # Length distribution
    lengths = [p["length"] for p in all_payloads]
    if lengths:
        print(f"\nPayload length: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} "
              f"p95={sorted(lengths)[int(len(lengths)*0.95)]} max={max(lengths)}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_payloads, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
