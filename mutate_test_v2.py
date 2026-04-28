#!/usr/bin/env python3
"""
Payload-only adversarial robustness probe (deployment-realistic).

Threat model: an attacker controls only the parameter VALUE submitted via HTTP.
The surrounding SQL skeleton (SELECT ... FROM ... WHERE col = '?') is fixed by
the application code. Hence WAF-A-MoLE mutations should apply ONLY to the
payload, not to the entire SQL statement.

Method:
  1. Curate canonical SQLi payloads (15 patterns covering union / blind /
     stacked / time / auth-bypass / etc.)
  2. For each payload, apply WAF-A-MoLE SqlFuzzer N times (chain of N
     mutations) for N in {0, 1, 3, 5, 10}
  3. Embed each (mutated) payload into each deployment template (6 templates)
  4. Run the trained 3-view model on each final SQL
  5. Report per-view detection rate vs mutation count

This isolates two questions:
  a) Does fusion (main) hold up better than surface alone under payload mutation?
  b) Does the AST view stay informative when the payload portion is perturbed?
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "WAF-A-MoLE"))  # vendored

from src.preprocessing import SamplePreprocessor
from src.dataset import collate_three_view
from src.model import ThreeViewModel
from train import binary_metrics
from evaluate import collect_scores, recall_at_fpr

from wafamole.payloadfuzzer.sqlfuzzer import SqlFuzzer


# ============================================================
# Canonical SQLi payloads (curated from SecLists + PayloadsAllTheThings)
# Covers 8 categories of SQL injection: auth bypass, tautology, union,
# stacked queries, time-based blind, boolean blind, error-based, polyglot.
# ============================================================
CANONICAL_PAYLOADS = [
    # === Auth bypass ===
    "admin' -- ",
    "admin' #",
    "admin' or '1'='1' -- ",
    "admin') or ('1'='1' -- ",
    "' OR 1=1 -- ",
    # === Tautology (various quoting / comment styles) ===
    "1' OR '1'='1",
    "1\") or (\"1\"=\"1",
    "1' OR 1=1#",
    "' or true -- ",
    "1') OR ('1'='1",
    # === Union-based (data extraction) ===
    "1' UNION SELECT NULL,NULL,NULL -- ",
    "' UNION ALL SELECT user,password FROM users -- ",
    "1') UNION SELECT NULL,version() -- ",
    "-1 UNION SELECT 1,2,3,4 -- ",
    # === Stacked queries (multi-statement) ===
    "1; DROP TABLE users -- ",
    "'; UPDATE users SET password='x' WHERE 1=1 -- ",
    # === Time-based blind ===
    "1' AND SLEEP(5) -- ",
    "1' AND BENCHMARK(10000000, MD5('a')) -- ",
    "1') OR pg_sleep(5) -- ",
    "1; WAITFOR DELAY '0:0:5' -- ",
    # === Boolean blind / data extraction ===
    "1' AND ASCII(SUBSTRING(database(),1,1)) > 64 -- ",
    "1' AND (SELECT COUNT(*) FROM users) > 0 -- ",
    "1' AND SUBSTRING(@@version,1,1)='5' -- ",
    # === Error-based info leak ===
    "1' AND EXTRACTVALUE(1, CONCAT(0x7e, USER())) -- ",
    "1' AND UPDATEXML(1, CONCAT(0x7e, VERSION()), 1) -- ",
    # === Out-of-band / file access ===
    "1' UNION SELECT LOAD_FILE('/etc/passwd'),NULL,NULL -- ",
    "1' INTO OUTFILE '/tmp/x.txt' -- ",
    # === Multi-DB polyglot ===
    "SLEEP(1) /*' or SLEEP(1) or '\" or SLEEP(1) or \"*/",
    # === WAF-bypass tricks (already obfuscated forms) ===
    "1' /*!OR*/ '1'='1",
    "1' UnIoN/**/SeLeCt NULL,NULL,NULL -- ",
]

# Deployment templates (mirrors src/preprocessing or system deployment)
DEPLOY_TEMPLATES = [
    "SELECT * FROM tab WHERE col1 = '{payload}'",
    "SELECT * FROM tab WHERE col1 = {payload}",
    "SELECT * FROM tab WHERE col1 LIKE '{payload}'",
    "INSERT INTO tab (col1) VALUES ('{payload}')",
    "UPDATE tab SET col1 = '{payload}' WHERE col2 = 'x'",
    "DELETE FROM tab WHERE col1 = '{payload}'",
]

MUTATION_COUNTS = [0, 1, 3, 5, 10]


def mutate_payload(payload: str, n_mutations: int, seed: int = 0) -> str:
    """Apply n_mutations chained WAF-A-MoLE mutations to a payload."""
    if n_mutations == 0:
        return payload
    random.seed(seed)
    fuzzer = SqlFuzzer(payload)
    for _ in range(n_mutations):
        fuzzer.fuzz()
    return fuzzer.current()


def safe_format(template: str, payload: str) -> str:
    """Embed payload into template, escaping single quotes only when the
    payload is inside a quoted slot (heuristic by surrounding chars)."""
    # If the {payload} slot is surrounded by single quotes, escape any single
    # quotes the payload contains so the surrounding quotes still close.
    # Otherwise embed raw.
    idx = template.index("{payload}")
    before = template[:idx]
    after = template[idx + len("{payload}"):]
    if before.endswith("'") and after.startswith("'"):
        # Quoted slot — but for an injection test we WANT the payload to
        # break out, so embed raw without escaping.
        pass
    return template.replace("{payload}", payload)


# ============================================================
# Build test set: (payload_id, n_muts, template_id, full_sql)
# ============================================================
def build_test_cases(seeds_per_payload: int = 5) -> list[dict]:
    """Generate the full test grid.
    Returns a list of {payload_id, n_muts, template_id, mutated_payload, full_sql, label}.
    Label is always 1 (these are all attacks).
    """
    cases = []
    for p_idx, payload in enumerate(CANONICAL_PAYLOADS):
        for n_muts in MUTATION_COUNTS:
            # Multiple seeds per (payload, n_muts) to average over randomness
            seeds = [0] if n_muts == 0 else list(range(seeds_per_payload))
            for seed in seeds:
                mutated = mutate_payload(payload, n_muts, seed=seed)
                for t_idx, tpl in enumerate(DEPLOY_TEMPLATES):
                    full = safe_format(tpl, mutated)
                    cases.append({
                        "payload_id": p_idx,
                        "payload_text": payload,
                        "n_muts": n_muts,
                        "seed": seed,
                        "template_id": t_idx,
                        "mutated_payload": mutated,
                        "full_sql": full,
                        "label": 1,
                    })
    return cases


# ============================================================
# Model inference
# ============================================================
@torch.no_grad()
def score_cases(model, preprocessor, cases, device, use_bf16, batch_size=64):
    """Run model on every full_sql, return per-view scores aligned with cases."""
    all_scores = {k: [] for k in ("main", "S", "L", "A")}
    autocast_kw = {"dtype": torch.bfloat16, "enabled": use_bf16}

    # Preprocess into one big batch list
    pre_samples = [preprocessor(c["full_sql"]) for c in cases]
    for s in pre_samples:
        s["label"] = 1

    # Batched inference
    for i in range(0, len(pre_samples), batch_size):
        batch_list = pre_samples[i:i+batch_size]
        batch = collate_three_view(batch_list)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", **autocast_kw):
            out = model(
                batch["surface_ids"], batch["surface_mask"],
                batch["lex_ids"], batch["lex_mask"],
                batch["ast_ids"], batch["ast_mask"],
                batch["ast_valid"],
                view_dropout_prob=0.0,
            )
        for k in ("main", "S", "L", "A"):
            all_scores[k].append(out[f"p_{k}"].float().cpu().numpy() if k != "main" else out["p_main"].float().cpu().numpy())

    return {k: np.concatenate(v) for k, v in all_scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--seeds-per-payload", type=int, default=5)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bool(cfg["train"].get("use_bf16", False)) and device.type == "cuda" and torch.cuda.is_bf16_supported()

    pre = SamplePreprocessor()
    model = ThreeViewModel(**cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])

    print(f"\nBuilding test cases...")
    cases = build_test_cases(seeds_per_payload=args.seeds_per_payload)
    print(f"  {len(cases)} cases ({len(CANONICAL_PAYLOADS)} payloads × "
          f"{len(MUTATION_COUNTS)} mutation levels × {len(DEPLOY_TEMPLATES)} templates × "
          f"{args.seeds_per_payload} seeds (1 for n_muts=0))")

    print("\nRunning inference...")
    t0 = time.time()
    scores = score_cases(model, pre, cases, device, use_bf16)
    print(f"  done in {time.time()-t0:.1f}s")

    # Aggregate by mutation count
    summary = defaultdict(lambda: {"main": [], "S": [], "L": [], "A": []})
    for i, c in enumerate(cases):
        for view in ("main", "S", "L", "A"):
            summary[c["n_muts"]][view].append(scores[view][i])

    print("\n=== Detection rate (score > 0 i.e. p > 0.5) by mutation count ===")
    print(f"{'n_muts':>8s}  {'n_cases':>8s}  {'main':>10s}  {'S':>10s}  {'L':>10s}  {'A':>10s}")
    table = []
    for n_muts in MUTATION_COUNTS:
        row = {"n_muts": n_muts, "n_cases": len(summary[n_muts]["main"])}
        for view in ("main", "S", "L", "A"):
            arr = np.array(summary[n_muts][view])
            det_rate = float((arr > 0).mean())
            mean_prob = float(1 / (1 + np.exp(-arr)).mean())
            row[f"{view}_det"] = det_rate
            row[f"{view}_meanp"] = mean_prob
        table.append(row)
        print(f"{n_muts:>8d}  {row['n_cases']:>8d}  "
              f"{row['main_det']:>10.4f}  {row['S_det']:>10.4f}  "
              f"{row['L_det']:>10.4f}  {row['A_det']:>10.4f}")

    # Drop in detection from n=0 to n=10
    print("\n=== Robustness: detection drop from n_muts=0 to n_muts=10 ===")
    base = next(r for r in table if r["n_muts"] == 0)
    last = next(r for r in table if r["n_muts"] == 10)
    print(f"{'view':>6s}  {'n=0 det':>10s}  {'n=10 det':>10s}  {'drop':>10s}")
    for view in ("main", "S", "L", "A"):
        d0 = base[f"{view}_det"]
        d10 = last[f"{view}_det"]
        drop = d0 - d10
        marker = " ← BEST" if view == "main" and drop == min(base[f"{v}_det"] - last[f"{v}_det"] for v in ("main","S","L","A")) else ""
        print(f"{view:>6s}  {d0:>10.4f}  {d10:>10.4f}  {drop:>10.4f}{marker}")

    # Save
    out = args.output / "mutate_v2_results.json"
    out.write_text(json.dumps({
        "n_payloads": len(CANONICAL_PAYLOADS),
        "n_templates": len(DEPLOY_TEMPLATES),
        "mutation_counts": MUTATION_COUNTS,
        "table": table,
        "raw_per_n_muts": {str(k): {view: list(map(float, v)) for view, v in d.items()}
                            for k, d in summary.items()},
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
