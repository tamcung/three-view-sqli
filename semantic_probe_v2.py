#!/usr/bin/env python3
"""
Semantic understanding probe v2: hand-crafted minimal-pair test set.

Pairs of SQL statements that are SURFACE-similar (share keywords, similar
length) but SEMANTICALLY different — one is a real injection, the other is
a benign use of the same keywords.

Goal: see whether the model can distinguish "OR 1=1 inside a WHERE clause
that actually executes" from "OR 1=1 inside a string literal that is just data".

This is the cleanest test of "does the model understand SQL or just match
character patterns?"
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import SamplePreprocessor
from src.dataset import collate_three_view
from src.model import ThreeViewModel


# ============================================================
# CLASS A: Truly BENIGN — SQL is structurally benign at execution
# ============================================================
TRULY_BENIGN = [
    # Plain CRUD with completely innocent content
    ("clean-1",  "INSERT INTO comments (text) VALUES ('Great product, fast shipping!')"),
    ("clean-2",  "SELECT * FROM users WHERE name = 'OBrien'"),
    ("clean-3",  "UPDATE settings SET locale = 'en-US' WHERE user_id = 42"),
    ("clean-4",  "SELECT COUNT(*) FROM orders WHERE status = 'shipped'"),
    ("clean-5",  "INSERT INTO login_log (msg) VALUES ('user42 login successful')"),
    # Benign that MENTIONS SQL injection text in string literal data
    ("text-mention-1",  "INSERT INTO articles (content) VALUES ('Today I learned about SQL injection')"),
    ("text-mention-2",  "INSERT INTO faq (q, a) VALUES ('What is SQL?', 'SQL is a query language')"),
    ("text-mention-3",  "UPDATE help SET content = 'Use SELECT to query data' WHERE topic = 'basics'"),
    ("text-mention-4",  "INSERT INTO posts (body) VALUES ('I love OR jokes!')"),
    ("text-mention-5",  "INSERT INTO blog (text) VALUES ('Beware of admin\\' OR 1=1 in user input')"),
    ("text-mention-6",  "INSERT INTO security_log VALUES ('Blocked attempt: UNION SELECT password')"),
    ("text-mention-7",  "UPDATE notes SET body = 'WAF flagged: SLEEP(5) timing attack'"),
    ("text-mention-8",  "INSERT INTO chat (msg) VALUES ('Hey, what does \\'OR 1=1\\' mean?')"),
    # Benign use of SQL keywords in legitimate clauses
    ("legit-or-1",      "SELECT * FROM events WHERE year = 2024 OR year = 2023"),
    ("legit-or-2",      "SELECT name FROM users WHERE country = 'US' OR country = 'CA'"),
    ("legit-union-1",   "SELECT name FROM customers UNION SELECT name FROM suppliers"),
    ("legit-comment-1", "SELECT * FROM users -- daily report query"),
    ("legit-drop-1",    "DROP TABLE temp_data IF EXISTS"),
    ("legit-and-1",     "SELECT * FROM users WHERE active = 1 AND created_at > '2024-01-01'"),
    ("legit-update-1",  "UPDATE users SET last_login = NOW() WHERE id = 100"),
    # Tricky: legitimate equality between same column / value
    ("legit-eq-1",      "SELECT id FROM products WHERE category = category AND in_stock = 1"),
    # Identifier names that look attack-y
    ("ident-1",         "SELECT * FROM sleep_log WHERE duration > 5"),
    ("ident-2",         "INSERT INTO drop_records (status) VALUES ('archived')"),
    ("ident-3",         "SELECT id, password_hash FROM users WHERE active = 1"),
    ("ident-4",         "SELECT version FROM software_inventory ORDER BY name"),
]


# ============================================================
# CLASS B: Truly ATTACK — SQL is structurally malicious at execution
# ============================================================
TRULY_ATTACK = [
    # Classical structural injections (OR clause attached to WHERE)
    ("attack-or-1",     "SELECT * FROM users WHERE id = 1 OR 1=1"),
    ("attack-or-2",     "SELECT * FROM users WHERE name = 'admin' OR '1'='1'"),
    ("attack-or-3",     "SELECT * FROM users WHERE id = 1 OR 'x'='x'"),
    # Stacked queries (multi-statement)
    ("attack-stacked-1","SELECT * FROM users WHERE id = 1; DROP TABLE users; --"),
    ("attack-stacked-2","SELECT * FROM users WHERE name = 'a'; DELETE FROM users; --"),
    ("attack-stacked-3","INSERT INTO logs VALUES ('test'); DROP TABLE users; --"),
    ("attack-stacked-4","UPDATE products SET price = 0; UPDATE users SET role = 'admin' WHERE id = 1; --"),
    # UNION-based (data exfiltration)
    ("attack-union-1",  "SELECT name FROM users WHERE id = 1 UNION SELECT password FROM admins"),
    ("attack-union-2",  "SELECT name FROM products WHERE id = 1 UNION SELECT username || ':' || password FROM users"),
    ("attack-union-3",  "SELECT id FROM articles WHERE category = 'a' UNION ALL SELECT password FROM users WHERE 1=1"),
    # Always-true tautologies
    ("attack-taut-1",   "SELECT * FROM users WHERE name = '' OR '' = ''"),
    ("attack-taut-2",   "SELECT * FROM users WHERE id = 1 OR (SELECT 1) = 1"),
    # Blind boolean/time injection structurally embedded
    ("attack-blind-1",  "SELECT * FROM users WHERE id = 1 AND ASCII(SUBSTRING(database(),1,1)) > 64"),
    ("attack-blind-2",  "SELECT * FROM users WHERE id = 1 AND SLEEP(5)"),
    ("attack-blind-3",  "SELECT * FROM users WHERE id = 1 AND BENCHMARK(1000000, MD5('a'))"),
    # Subquery exfiltration
    ("attack-sub-1",    "SELECT * FROM users WHERE id = 1 AND (SELECT COUNT(*) FROM users WHERE password LIKE 'a%') > 0"),
    ("attack-sub-2",    "SELECT * FROM products WHERE id = 1 AND EXISTS(SELECT 1 FROM information_schema.tables)"),
    # Error-based info leak
    ("attack-error-1",  "SELECT * FROM users WHERE id = 1 AND EXTRACTVALUE(1, CONCAT(0x7e, USER()))"),
    ("attack-error-2",  "SELECT * FROM users WHERE id = 1 AND UPDATEXML(1, CONCAT(0x7e, VERSION()), 1)"),
    # Out-of-band / file
    ("attack-oob-1",    "SELECT * FROM users WHERE id = 1 UNION SELECT LOAD_FILE('/etc/passwd'), NULL, NULL"),
    ("attack-oob-2",    "SELECT * FROM users WHERE id = 1 INTO OUTFILE '/tmp/dump.txt'"),
    # Auth bypass forms
    ("attack-auth-1",   "SELECT * FROM users WHERE username = 'admin' AND password = 'x' OR '1'='1'"),
    ("attack-auth-2",   "SELECT * FROM users WHERE username = 'admin' --' AND password = 'whatever'"),
    # Quote-escape style
    ("attack-quote-1",  "SELECT * FROM users WHERE name = ''; DROP TABLE users; --"),
    ("attack-quote-2",  "SELECT * FROM logs WHERE msg = '' UNION SELECT password FROM users; --"),
]


# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def score_batch(model, preprocessor, texts, device, use_bf16, batch_size=32):
    samples = [preprocessor(t) for t in texts]
    for s in samples:
        s["label"] = 0
    autocast_kw = {"dtype": torch.bfloat16, "enabled": use_bf16}
    out_scores = {k: [] for k in ("main", "S", "L", "A")}
    for i in range(0, len(samples), batch_size):
        batch_list = samples[i:i+batch_size]
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
            out_scores[k].append(out[f"p_{k}"].float().cpu().numpy() if k != "main" else out["p_main"].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out_scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
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

    benign_texts = [t for _, t in TRULY_BENIGN]
    attack_texts = [t for _, t in TRULY_ATTACK]

    benign_scores = score_batch(model, pre, benign_texts, device, use_bf16)
    attack_scores = score_batch(model, pre, attack_texts, device, use_bf16)

    def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

    # ---- Detailed table per sample ----
    def print_section(title, samples, scores):
        print(f"\n{title}")
        print(f"{'tag':>20s}  {'main':>5s} {'S':>5s} {'L':>5s} {'A':>5s}  text")
        for i, (tag, txt) in enumerate(samples):
            mp = sigmoid(scores["main"][i])
            sp = sigmoid(scores["S"][i])
            lp = sigmoid(scores["L"][i])
            ap = sigmoid(scores["A"][i])
            print(f"{tag:>20s}  {mp:>5.2f} {sp:>5.2f} {lp:>5.2f} {ap:>5.2f}  {txt[:70]}")

    print_section("=== Class A: TRULY BENIGN (lower score = correct) ===", TRULY_BENIGN, benign_scores)
    print_section("=== Class B: TRULY ATTACK (higher score = correct) ===", TRULY_ATTACK, attack_scores)

    # ---- Aggregate ----
    print(f"\n{'='*60}")
    print("AGGREGATE")
    print(f"{'='*60}")
    print(f"\nClass A — TRULY BENIGN ({len(benign_texts)} samples):")
    print(f"  view    FPR (predicted as attack)")
    for view in ("main", "S", "L", "A"):
        fpr = float((benign_scores[view] > 0).mean())
        print(f"  {view:6s} {fpr*100:>6.2f}%")
    print(f"\nClass B — TRULY ATTACK ({len(attack_texts)} samples):")
    print(f"  view    Detection rate")
    for view in ("main", "S", "L", "A"):
        det = float((attack_scores[view] > 0).mean())
        print(f"  {view:6s} {det*100:>6.2f}%")

    # ---- Verdict ----
    print(f"\n{'='*60}")
    print("VERDICT (per view)")
    print(f"{'='*60}")
    for view in ("main", "S", "L", "A"):
        fpr = float((benign_scores[view] > 0).mean())
        det = float((attack_scores[view] > 0).mean())
        # Balanced accuracy
        bal_acc = (det + (1 - fpr)) / 2
        if fpr < 0.2 and det > 0.85:
            verdict = "GOOD: separates structural attack from text mention"
        elif det > 0.85 and fpr > 0.5:
            verdict = "POOR: catches attacks but flags benign-with-attack-text (surface matching)"
        elif det < 0.7:
            verdict = "POOR: misses real structural attacks"
        else:
            verdict = "MODERATE"
        print(f"  {view:6s}  FPR={fpr*100:>5.2f}%  Det={det*100:>5.2f}%  BalAcc={bal_acc*100:>5.2f}%  → {verdict}")

    # Save
    out = {
        "class_a_benign": {tag: {
            "text": txt,
            "main_prob": float(sigmoid(benign_scores["main"][i])),
            "S_prob": float(sigmoid(benign_scores["S"][i])),
            "L_prob": float(sigmoid(benign_scores["L"][i])),
            "A_prob": float(sigmoid(benign_scores["A"][i])),
        } for i, (tag, txt) in enumerate(TRULY_BENIGN)},
        "class_b_attack": {tag: {
            "text": txt,
            "main_prob": float(sigmoid(attack_scores["main"][i])),
            "S_prob": float(sigmoid(attack_scores["S"][i])),
            "L_prob": float(sigmoid(attack_scores["L"][i])),
            "A_prob": float(sigmoid(attack_scores["A"][i])),
        } for i, (tag, txt) in enumerate(TRULY_ATTACK)},
    }
    out_file = args.output / "semantic_probe_v2.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()
