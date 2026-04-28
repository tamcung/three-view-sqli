#!/usr/bin/env python3
"""
Semantic understanding probe: does the model truly understand SQL semantics,
or is it pattern-matching on surface character sequences?

Two diagnostics:

  Probe 1 (BENIGN containing attack-like text):
    Legitimate SQL where a string literal / comment / column data contains
    text that looks like an attack ("OR 1=1", "UNION SELECT" etc.). A model
    that semantically understands SQL must NOT flag these as malicious.

  Probe 2 (ATTACK with novel equivalent syntax):
    SQL injection payloads written in semantically-equivalent forms that the
    model has likely NOT seen verbatim in training (e.g. "OR 'a'='a'" instead
    of "OR 1=1", "OR (SELECT 1)>0" instead of tautology). A model that learned
    semantics should still detect these.

A model that fails Probe 1 (high FPR on disguised benign) or Probe 2 (low
detection on novel attacks) is doing surface pattern matching, not semantic
understanding.
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
# Probe 1: BENIGN with attack-like string content
# ============================================================
PROBE_1_BENIGN_LOOKALIKE = [
    # Logs / messages mentioning attacks
    "INSERT INTO logs (msg) VALUES ('User input: admin OR 1=1 detected')",
    "INSERT INTO security_log (event) VALUES ('SQL injection attempt: UNION SELECT password')",
    "INSERT INTO audit (action) VALUES ('Blocked attack: \" OR \"x\"=\"x\"')",
    "INSERT INTO events (description) VALUES ('Suspicious payload: \\' OR 1=1--')",
    # Help / tutorial / docs
    "SELECT * FROM articles WHERE title = 'How to prevent UNION SELECT attacks'",
    "UPDATE help SET content = 'The OR 1=1 trick is a classic SQLi pattern'",
    "INSERT INTO faq (question) VALUES ('What does '\\' OR \\'1\\'=\\'1' mean?')",
    "SELECT * FROM blog_posts WHERE tags LIKE '%sqli-prevention%' AND title LIKE '%DROP TABLE%'",
    # User-generated content with embedded SQLi-looking text
    "INSERT INTO comments (text) VALUES ('My friend told me about admin\\' OR 1=1 hack')",
    "UPDATE notes SET body = 'Reminder: never use SLEEP(5) in production code'",
    "INSERT INTO posts (content) VALUES ('Read about SELECT user, password FROM users vulnerability')",
    # Column / table names that look attack-y
    "SELECT id, name, password_hash FROM users WHERE active = 1",
    "SELECT name, version FROM software_inventory ORDER BY name",
    "INSERT INTO sleep_log (duration) VALUES (5)",
    "UPDATE drop_records SET status = 'archived' WHERE created_at < NOW()",
    # Legitimate analytic queries with suspicious-looking conditions
    "SELECT COUNT(*) FROM transactions WHERE amount > 0 AND status = 'completed'",
    "SELECT user_id FROM sessions WHERE expires_at > NOW() AND active = 1",
    "SELECT a.id, b.name FROM accounts a JOIN balances b ON a.id = b.account_id WHERE a.country = 'US'",
    # Innocent UNION usage
    "SELECT name FROM customers UNION SELECT name FROM suppliers ORDER BY name",
    "SELECT '2024-Q1' AS quarter, revenue FROM sales WHERE year = 2024 UNION SELECT '2024-Q2', revenue FROM sales WHERE year = 2024",
    # Innocent comments
    "SELECT * FROM users -- get all active users for monthly report",
    "SELECT product, price FROM catalog /* exclude discontinued */",
]


# ============================================================
# Probe 2: ATTACK with novel equivalent syntax
# ============================================================
PROBE_2_NOVEL_ATTACK = [
    # Tautology with non-numeric equivalents
    "1' OR 'a'='a' -- ",
    "1' OR 'cat'<>'dog' -- ",
    "1' OR LENGTH('x')=1 -- ",
    "1' OR ASCII('A')=65 -- ",
    "1' OR (SELECT 1)>0 -- ",
    "1' OR (SELECT 'x')='x' -- ",
    "1' OR ('a' BETWEEN 'a' AND 'z') -- ",
    "1' OR 'abc' LIKE 'a%' -- ",
    "1' OR REVERSE('ab')='ba' -- ",
    "1' OR CONCAT('a','b')='ab' -- ",
    # Tautology hidden in arithmetic / comparison
    "1' OR 5+5=10 -- ",
    "1' OR 100/100=1 -- ",
    "1' OR ABS(-1)=1 -- ",
    "1' OR ROUND(3.14)=3 -- ",
    "1' OR FLOOR(2.5)=2 -- ",
    # Subquery-based injections (less obvious)
    "1' AND (SELECT username FROM users WHERE id=1) IS NOT NULL -- ",
    "1' AND EXISTS(SELECT 1 FROM information_schema.tables) -- ",
    "1' UNION SELECT NULL,NULL,(SELECT GROUP_CONCAT(table_name) FROM information_schema.tables) -- ",
    # Function-based info leak (not the standard ones)
    "1' AND CHAR_LENGTH(USER())>0 -- ",
    "1' AND OCTET_LENGTH(DATABASE())>0 -- ",
    "1' AND SOUNDEX('test')=SOUNDEX('test') -- ",
    "1' AND HEX(USER()) LIKE '%' -- ",
    # Boolean injection with novel functions
    "1' AND IF(USER()='root', 1, 0)=1 -- ",
    "1' AND CASE WHEN 1=1 THEN 1 ELSE 0 END=1 -- ",
    "1' AND COALESCE(NULL, 1, 2)=1 -- ",
    # Time-based with non-standard sleep functions
    "1' AND IF((SELECT COUNT(*) FROM users)>0, BENCHMARK(1000000, SHA1('x')), 0) -- ",
    "1' AND GET_LOCK('x', 5)=1 -- ",
    "1; SELECT pg_sleep(2.5)+pg_sleep(2.5) -- ",
    # Stacked / batch attacks with novel ordering
    "1; CREATE TABLE x AS SELECT * FROM users -- ",
    "1; INSERT INTO admins SELECT * FROM users WHERE role='user' -- ",
]


# ============================================================
# Inference helpers
# ============================================================
@torch.no_grad()
def score_batch(model, preprocessor, texts, device, use_bf16, batch_size=32):
    samples = [preprocessor(t) for t in texts]
    for s in samples:
        s["label"] = 0  # placeholder, not used for inference
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

    # ---- Probe 1: Benign with attack-like content ----
    print(f"\n=== Probe 1: BENIGN with attack-like content ({len(PROBE_1_BENIGN_LOOKALIKE)} samples) ===")
    print("(true label = benign; lower score = better; FPR = false positive rate)")
    scores_p1 = score_batch(model, pre, PROBE_1_BENIGN_LOOKALIKE, device, use_bf16)

    print(f"\n{'idx':>3s} {'main':>7s} {'S':>7s} {'L':>7s} {'A':>7s}  text")
    for i, txt in enumerate(PROBE_1_BENIGN_LOOKALIKE):
        m_p = 1 / (1 + np.exp(-scores_p1["main"][i]))
        s_p = 1 / (1 + np.exp(-scores_p1["S"][i]))
        l_p = 1 / (1 + np.exp(-scores_p1["L"][i]))
        a_p = 1 / (1 + np.exp(-scores_p1["A"][i]))
        print(f"{i:>3d} {m_p:>7.3f} {s_p:>7.3f} {l_p:>7.3f} {a_p:>7.3f}  {txt[:80]}")

    # FPR per view
    print(f"\n{'view':>6s}  {'FPR (predicted as attack)':>30s}")
    for view in ("main", "S", "L", "A"):
        fpr = float((scores_p1[view] > 0).mean())
        print(f"{view:>6s}  {fpr*100:>26.2f}%")

    # ---- Probe 2: Attack with novel equivalent syntax ----
    print(f"\n=== Probe 2: ATTACK with novel equivalent syntax ({len(PROBE_2_NOVEL_ATTACK)} samples) ===")
    print("(true label = attack; higher score = better; missed = predicted benign)")

    # Wrap each payload in templates to be realistic
    DEPLOY_TEMPLATES = [
        "SELECT * FROM tab WHERE col1 = '{p}'",
        "INSERT INTO tab (col1) VALUES ('{p}')",
        "UPDATE tab SET col1 = '{p}' WHERE col2 = 'x'",
    ]
    wrapped = []
    for payload in PROBE_2_NOVEL_ATTACK:
        for tpl in DEPLOY_TEMPLATES:
            wrapped.append(tpl.format(p=payload))
    print(f"  {len(PROBE_2_NOVEL_ATTACK)} payloads × {len(DEPLOY_TEMPLATES)} templates = {len(wrapped)} cases")
    scores_p2 = score_batch(model, pre, wrapped, device, use_bf16)

    print(f"\n{'view':>6s}  {'detection rate':>20s}  {'mean prob':>15s}")
    p2_metrics = {}
    for view in ("main", "S", "L", "A"):
        det = float((scores_p2[view] > 0).mean())
        mean_p = float((1 / (1 + np.exp(-scores_p2[view]))).mean())
        p2_metrics[view] = {"detection_rate": det, "mean_prob": mean_p}
        print(f"{view:>6s}  {det*100:>18.2f}%  {mean_p:>15.3f}")

    # Per-payload breakdown (averaged over 3 templates)
    print(f"\nPer-payload detection (averaged over {len(DEPLOY_TEMPLATES)} templates):")
    print(f"{'main':>5s} {'S':>5s} {'L':>5s} {'A':>5s}  payload")
    n_per = len(DEPLOY_TEMPLATES)
    for p_idx, payload in enumerate(PROBE_2_NOVEL_ATTACK):
        idxs = [p_idx * n_per + j for j in range(n_per)]
        m = (scores_p2["main"][idxs] > 0).mean()
        s = (scores_p2["S"][idxs] > 0).mean()
        l = (scores_p2["L"][idxs] > 0).mean()
        a = (scores_p2["A"][idxs] > 0).mean()
        print(f"{m:>5.2f} {s:>5.2f} {l:>5.2f} {a:>5.2f}  {payload[:80]}")

    # Save
    out = {
        "probe_1_benign_lookalike": {
            "n_samples": len(PROBE_1_BENIGN_LOOKALIKE),
            "fpr_per_view": {view: float((scores_p1[view] > 0).mean()) for view in ("main", "S", "L", "A")},
            "raw_probs_per_view": {view: [float(1/(1+np.exp(-scores_p1[view][i]))) for i in range(len(PROBE_1_BENIGN_LOOKALIKE))]
                                    for view in ("main", "S", "L", "A")},
            "samples": PROBE_1_BENIGN_LOOKALIKE,
        },
        "probe_2_novel_attack": {
            "n_payloads": len(PROBE_2_NOVEL_ATTACK),
            "n_templates": len(DEPLOY_TEMPLATES),
            "n_cases": len(wrapped),
            "metrics_per_view": p2_metrics,
            "samples": PROBE_2_NOVEL_ATTACK,
        },
    }
    out_file = args.output / "semantic_probe.json"
    out_file.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_file}")

    # Summary verdict
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("Probe 1 (BENIGN with attack-like text) — FPR per view:")
    for view in ("main", "S", "L", "A"):
        fpr = float((scores_p1[view] > 0).mean())
        verdict = "GOOD" if fpr < 0.1 else "MODERATE" if fpr < 0.3 else "POOR (surface pattern matching!)"
        print(f"  {view:6s} {fpr*100:>6.2f}%  → {verdict}")
    print("\nProbe 2 (NOVEL attack syntax) — Detection rate per view:")
    for view in ("main", "S", "L", "A"):
        det = float((scores_p2[view] > 0).mean())
        verdict = "GOOD (semantic understanding)" if det > 0.9 else "MODERATE" if det > 0.7 else "POOR (missing novel patterns)"
        print(f"  {view:6s} {det*100:>6.2f}%  → {verdict}")


if __name__ == "__main__":
    main()
