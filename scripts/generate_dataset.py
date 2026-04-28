#!/usr/bin/env python3
"""
Generate SQLi dataset by combining templates × payloads, with sqlglot
AST-based label assignment.

Algorithm:
  For each template T:
    1. Compute the "expected" AST signature by filling T with safe placeholders.
    2. For each (benign payload P_b):
         sql = T.fill(P_b)
         if parses OK and ast_sig(sql) == expected_sig:
             label = 0 (benign — structure preserved)
         else: skip (boundary case)
    3. For each (attack payload P_a):
         sql = T.fill(P_a)
         try parse:
             if parse fails:
                 label = 1, class = 'attack_breaks_parsing'
             elif ast_sig(sql) == expected_sig:
                 label = 0, class = 'failed_attack_aka_benign'
                 (attack didn't escape string context)
             elif structurally_dangerous(sql, expected_sig):
                 label = 1, class = 'successful_attack'
             else: skip (ambiguous)

Output: data/generated_dataset.jsonl
  one sample per line: {sql, label, template_id, payload_class, ...}
"""
from __future__ import annotations
import json
import logging
import random
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")
logging.getLogger("sqlglot").setLevel(logging.ERROR)

import sqlglot
from sqlglot import expressions as exp

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_PATH = ROOT / "data" / "templates.yaml"
ATTACK_POOL_PATH = ROOT / "data" / "attack_payloads.json"
BENIGN_POOL_PATH = ROOT / "data" / "benign_payloads.json"
OUT_PATH = ROOT / "data" / "generated_dataset.jsonl"

random.seed(42)


# ============================================================
# Safe placeholders for computing expected AST signature
# ============================================================
SAFE_PLACEHOLDERS = {
    "string": "SAFE_STRING",
    "number": "1",
    "identifier": "id",
    "date": "2024-01-01",
}


def safe_fill(template: dict) -> str:
    """Fill template slots with safe placeholders to compute expected AST."""
    sql = template["sql"]
    for slot_name, slot_meta in template["slots"].items():
        ptype = slot_meta["payload_type"]
        if slot_meta["context"] == "date_quoted":
            ph = "2024-01-01"
        elif ptype == "number":
            ph = "1"
        elif ptype == "identifier":
            ph = "id"
        else:
            ph = "safe_str"
        sql = sql.replace("{" + slot_name + "}", ph)
    return sql


# ============================================================
# AST signature: structural fingerprint that ignores literal content
# ============================================================
def ast_signature(tree: exp.Expression) -> tuple:
    """Hashable structural signature: (top_kind, frozen node-type tree, function calls)."""
    if tree is None:
        return ("None",)

    def walk(node):
        if not isinstance(node, exp.Expression):
            return None
        kind = type(node).__name__
        # For terminal-like nodes, just use kind
        if isinstance(node, (exp.Literal, exp.Identifier, exp.Boolean, exp.Null, exp.Star)):
            return kind
        children = []
        for arg in node.args.values():
            if arg is None:
                continue
            if isinstance(arg, list):
                for a in arg:
                    c = walk(a)
                    if c is not None:
                        children.append(c)
            else:
                c = walk(arg)
                if c is not None:
                    children.append(c)
        return (kind, tuple(children))

    skeleton = walk(tree)
    funcs = tuple(sorted({type(n).__name__ for n in tree.walk() if isinstance(n, exp.Func)}))
    return (skeleton, funcs)


# ============================================================
# Structural danger detection (when AST differs from expected)
# ============================================================
DANGEROUS_NODE_TYPES = {
    "Block",       # multi-statement (stacked)
    "Drop",        # DDL injection
    "Create",      # DDL injection
    "Alter",       # DDL injection
    "Truncate",    # DDL injection
    "Union",       # union-based (if not in template)
    "If",          # conditional time-based
    "IfBlock",
    "Subquery",    # if not in template
    "Case",        # if not in template
}

DANGEROUS_FUNCTIONS = {
    # Time-based blind
    "Sleep", "Benchmark", "WaitFor",
    # Info leak
    "ExtractValue", "UpdateXml", "LoadFile", "Version",
    "User", "Database", "CurrentUser", "Schema",
    "ConnectionId", "LastInsertId",
    # File / OOB
    "Outfile", "DumpFile",
}


def structurally_dangerous(actual_tree: exp.Expression, expected_sig: tuple) -> tuple[bool, list]:
    """Returns (is_dangerous, reasons).

    Dangerous if actual tree contains node types or functions that the expected
    template did not have.
    """
    expected_node_types = set()
    expected_funcs = set()

    def collect_from_sig(sig):
        if isinstance(sig, str):
            expected_node_types.add(sig)
        elif isinstance(sig, tuple) and len(sig) == 2:
            kind, children = sig
            expected_node_types.add(kind)
            for c in children:
                collect_from_sig(c)

    if expected_sig and len(expected_sig) >= 1:
        collect_from_sig(expected_sig[0])
    if expected_sig and len(expected_sig) >= 2:
        expected_funcs = set(expected_sig[1])

    reasons = []
    actual_node_types = {type(n).__name__ for n in actual_tree.walk() if isinstance(n, exp.Expression)}
    actual_funcs = {type(n).__name__ for n in actual_tree.walk() if isinstance(n, exp.Func)}

    for nt in DANGEROUS_NODE_TYPES:
        if nt in actual_node_types and nt not in expected_node_types:
            reasons.append(f"new_node:{nt}")

    for fn in DANGEROUS_FUNCTIONS:
        if fn in actual_funcs and fn not in expected_funcs:
            reasons.append(f"new_function:{fn}")

    return (len(reasons) > 0, reasons)


def parse_strict(sql: str):
    """Parse SQL with multi-dialect attempts. Returns tree or None."""
    for d in ("mysql", "postgres", "tsql"):
        try:
            tree = sqlglot.parse_one(sql, read=d, error_level=sqlglot.ErrorLevel.IGNORE)
        except Exception:
            continue
        if tree is None or isinstance(tree, exp.Command):
            continue
        return tree
    return None


# ============================================================
# Payload pool helpers
# ============================================================
def filter_payloads_by_type(payloads: list[dict], payload_type: str) -> list[dict]:
    """For benign payload pool, keep only payloads suitable for the slot type."""
    if payload_type == "string":
        # Anything works in a string context
        return payloads
    if payload_type == "number":
        # Need numeric-looking payloads
        out = []
        for p in payloads:
            text = p["payload"].strip()
            try:
                float(text)
                out.append(p)
            except ValueError:
                pass
        return out
    if payload_type == "identifier":
        # Use a fixed list of column-like identifiers
        IDENT_LIST = ["id", "name", "created_at", "updated_at", "status", "email",
                      "username", "price", "quantity", "title", "category",
                      "user_id", "product_id", "rating", "score"]
        return [{"payload": x, "source": "identifier_pool", "category": "identifier"} for x in IDENT_LIST]
    return payloads


# ============================================================
# Main generator
# ============================================================
def main():
    print("Loading templates...")
    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        templates = yaml.safe_load(f)["templates"]
    print(f"  {len(templates)} templates")

    print("Loading attack payloads...")
    with open(ATTACK_POOL_PATH, encoding="utf-8") as f:
        attack_pool = json.load(f)
    print(f"  {len(attack_pool)} attack payloads")

    print("Loading benign payloads...")
    with open(BENIGN_POOL_PATH, encoding="utf-8") as f:
        benign_pool = json.load(f)
    print(f"  {len(benign_pool)} benign payloads")

    # Per template, decide how many samples to generate
    BENIGN_PER_TEMPLATE = 250
    ATTACK_PER_TEMPLATE = 300

    samples = []
    skipped = Counter()
    label_counts = Counter()
    payload_class_counts = Counter()

    t0 = time.time()
    for i, template in enumerate(templates):
        if i % 10 == 0:
            print(f"  template {i+1}/{len(templates)}: {template['id']} ({time.time()-t0:.0f}s)")

        # Compute expected AST signature
        safe_sql = safe_fill(template)
        expected_tree = parse_strict(safe_sql)
        if expected_tree is None:
            print(f"    SKIP {template['id']}: safe-fill cannot parse: {safe_sql!r}")
            continue
        expected_sig = ast_signature(expected_tree)

        # ---- Benign samples ----
        # Sample appropriate-typed payloads per slot
        for _ in range(BENIGN_PER_TEMPLATE):
            sql = template["sql"]
            payloads_used = {}
            ok = True
            for slot_name, slot_meta in template["slots"].items():
                ptype = slot_meta["payload_type"]
                pool = filter_payloads_by_type(benign_pool, ptype)
                if not pool:
                    ok = False
                    break
                p = random.choice(pool)
                payloads_used[slot_name] = p
                sql = sql.replace("{" + slot_name + "}", p["payload"])
            if not ok:
                skipped["no_typed_benign_payload"] += 1
                continue

            tree = parse_strict(sql)
            if tree is None:
                skipped["benign_parse_fail"] += 1
                continue
            actual_sig = ast_signature(tree)
            if actual_sig == expected_sig:
                # Structure preserved → benign
                # Determine subclass based on payload categories used
                pclasses = sorted({pl["category"] for pl in payloads_used.values()})
                subclass = "benign_" + ("_".join(pclasses) if pclasses else "unknown")
                samples.append({
                    "sql": sql, "label": 0,
                    "template_id": template["id"],
                    "template_category": template["category"],
                    "payload_class": subclass,
                })
                label_counts[0] += 1
                payload_class_counts[subclass] += 1
            else:
                skipped["benign_structure_changed"] += 1

        # ---- Attack samples ----
        for _ in range(ATTACK_PER_TEMPLATE):
            sql = template["sql"]
            attack_p = random.choice(attack_pool)
            # Pick ONE slot to inject attack into; others get benign typed payloads
            slot_names = list(template["slots"].keys())
            inject_slot = random.choice(slot_names)
            for slot_name in slot_names:
                if slot_name == inject_slot:
                    sql = sql.replace("{" + slot_name + "}", attack_p["payload"])
                else:
                    smeta = template["slots"][slot_name]
                    ptype = smeta["payload_type"]
                    pool = filter_payloads_by_type(benign_pool, ptype)
                    if not pool:
                        sql = sql.replace("{" + slot_name + "}", "1")
                    else:
                        sql = sql.replace("{" + slot_name + "}", random.choice(pool)["payload"])

            tree = parse_strict(sql)
            if tree is None:
                # Attack made SQL unparseable — count as attack (would also fail in DB,
                # but WAF should still block this kind of malformed input)
                samples.append({
                    "sql": sql, "label": 1,
                    "template_id": template["id"],
                    "template_category": template["category"],
                    "payload_class": "attack_breaks_parsing",
                    "attack_category": attack_p["category"],
                })
                label_counts[1] += 1
                payload_class_counts["attack_breaks_parsing"] += 1
                continue

            actual_sig = ast_signature(tree)
            if actual_sig == expected_sig:
                # Attack payload didn't escape its slot → harmless, label benign
                samples.append({
                    "sql": sql, "label": 0,
                    "template_id": template["id"],
                    "template_category": template["category"],
                    "payload_class": "failed_attack_aka_benign",
                    "attack_category": attack_p["category"],
                })
                label_counts[0] += 1
                payload_class_counts["failed_attack_aka_benign"] += 1
            else:
                # Any structural change from the template's expected signature
                # means the payload broke out of its intended slot → attack.
                is_dangerous, reasons = structurally_dangerous(tree, expected_sig)
                subclass = "successful_attack" if is_dangerous else "structural_attack"
                samples.append({
                    "sql": sql, "label": 1,
                    "template_id": template["id"],
                    "template_category": template["category"],
                    "payload_class": subclass,
                    "attack_category": attack_p["category"],
                    "structural_diff": reasons,
                })
                label_counts[1] += 1
                payload_class_counts[subclass] += 1

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"\nLabel counts:")
    for label, cnt in sorted(label_counts.items()):
        print(f"  label={label}: {cnt:,}")
    print(f"\nPayload class breakdown:")
    for pc, cnt in payload_class_counts.most_common():
        print(f"  {pc:40s} {cnt:>6,}")
    print(f"\nSkipped:")
    for reason, cnt in skipped.most_common():
        print(f"  {reason:40s} {cnt:>6,}")

    # Dedupe by SQL
    seen = set()
    unique = []
    for s in samples:
        if s["sql"] in seen:
            continue
        seen.add(s["sql"])
        unique.append(s)
    print(f"\nTotal samples: {len(samples)}")
    print(f"Unique SQLs:   {len(unique)}")

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for s in unique:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nWrote {OUT_PATH} ({len(unique):,} samples)")

    # Final stats
    final_label = Counter(s["label"] for s in unique)
    final_pclass = Counter(s["payload_class"] for s in unique)
    print(f"\nFinal label distribution: {dict(final_label)}")
    print(f"Final class distribution:")
    for pc, cnt in final_pclass.most_common():
        print(f"  {pc:40s} {cnt:>6,}")


if __name__ == "__main__":
    main()
