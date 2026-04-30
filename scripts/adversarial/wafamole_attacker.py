#!/usr/bin/env python3
"""WAF-A-MoLE-based adversarial attacker (§4.3).

Reuses WAF-A-MoLE's published `SqlFuzzer` (Demetrio et al., USENIX Security
2020) as the mutation engine, plugging our trained tri-view model in as
the target classifier. Three of WAF-A-MoLE's nine mutators are
out-of-distribution w.r.t. our training tamper set:

    swap_int_repr        digit → hex / (SELECT digit) / etc.
    change_tautologies   1=1 → 'x'='x' / 2=2 / 'a'<>'b'
    logical_invariant    insert " AND True" / " OR False"

These produce semantic-equivalent SQL with structures the model has not
seen during training. The other six (random_case, spaces_to_comments,
swap_keywords, etc.) overlap with sqlmap tampers.

Search procedure: hill-climbing — each round applies `round_size` random
mutations, picks the one whose model confidence is lowest, repeats from
that mutated payload. Same as the original WAF-A-MoLE engine, but uses
our `batch_predict` so we score `round_size` candidates in one GPU pass.

Usage:
    python -m scripts.adversarial.wafamole_attacker \
        --checkpoint results/tri_view_stage_aug/best_checkpoint.pt \
        --output data/adversarial/wafamole_pilot.jsonl \
        --n-seeds 200 --max-rounds 50 --round-size 20
"""
from __future__ import annotations
import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# Make WAF-A-MoLE importable without installing it.
# Only SqlFuzzer is needed; the evasion engine pulls in keras / tf and we
# replace it anyway with our batched scoring loop.
WAFAMOLE_PATH = ROOT.parent / "external" / "WAF-A-MoLE"
if str(WAFAMOLE_PATH) not in sys.path:
    sys.path.insert(0, str(WAFAMOLE_PATH))

from wafamole.payloadfuzzer.sqlfuzzer import SqlFuzzer  # noqa: E402

# Mutator name registry — for stats / reporting which fired
MUTATOR_FNS = {fn.__name__: fn for fn in SqlFuzzer.strategies}

from scripts.adversarial.utils import (
    load_victim, batch_predict, is_functional_sqli,
    load_seed_attacks, save_adv_records, setup_logger,
)


# ============================================================
# Hill-climbing per seed (WAF-A-MoLE engine, batched)
# ============================================================
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))


def attack_one_seed(
    seed: str,
    score_logits_fn,
    rng: random.Random,
    max_rounds: int = 50,
    round_size: int = 20,
    threshold: float = 0.5,
    require_functional: bool = True,
) -> tuple[str, float, list[dict], list[str]]:
    """Hill-climb over WAF-A-MoLE mutations.

    Operates in logit space because sigmoid saturates near 1.0 for the
    well-trained victim (sigmoid(30) and sigmoid(50) are both ~1.0 in
    float32 — using probs would freeze hill-climbing immediately).

    Returns (best_payload, best_prob, history, applied_mutators).
    """
    history = []
    applied_mutators: list[str] = []

    fuzzer = SqlFuzzer(seed)
    base_logit = float(score_logits_fn([seed])[0])
    cur_payload, cur_logit = seed, base_logit
    best_payload, best_logit = seed, base_logit

    threshold_logit = float(np.log(threshold / (1 - threshold)))

    for r in range(max_rounds):
        if cur_logit < threshold_logit and r >= 2:
            break
        # Generate `round_size` candidate mutations
        candidates: list[str] = []
        mutators_used: list[str] = []
        seen = set()
        attempts = 0
        while len(candidates) < round_size and attempts < round_size * 4:
            fuzzer.payload = cur_payload
            mutated = fuzzer.fuzz()
            attempts += 1
            if mutated in seen or mutated == cur_payload:
                continue
            seen.add(mutated)
            candidates.append(mutated)
            mutators_used.append("")
        if not candidates:
            break

        cand_logits = score_logits_fn(candidates)
        order = np.argsort(cand_logits)   # ascending: smallest logit first
        improved = False
        for j in order:
            cand = candidates[j]
            cand_logit = float(cand_logits[j])
            if cand_logit >= cur_logit:
                continue
            if require_functional and not is_functional_sqli(cand):
                continue
            history.append({
                "round": r,
                "before_logit": float(cur_logit),
                "after_logit": cand_logit,
                "before_prob": float(_sigmoid(cur_logit)),
                "after_prob": float(_sigmoid(cand_logit)),
                "payload_len": len(cand),
            })
            cur_payload, cur_logit = cand, cand_logit
            applied_mutators.append(mutators_used[j])
            improved = True
            if cand_logit < best_logit:
                best_payload, best_logit = cand, cand_logit
            break
        if not improved:
            history.append({"round": r, "no_improvement": True,
                            "cur_logit": float(cur_logit)})
            # Restart from seed occasionally to escape local minima
            if rng.random() < 0.3:
                cur_payload, cur_logit = seed, base_logit

    best_prob = float(_sigmoid(best_logit))
    return best_payload, best_prob, history, applied_mutators


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed-split", type=str,
                    default=str(ROOT / "data" / "splits" / "test.jsonl"))
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--max-rounds", type=int, default=50,
                    help="Max hill-climb iterations per seed.")
    p.add_argument("--round-size", type=int, default=20,
                    help="Candidates per iteration (one batch).")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-seeds-already-broken", action="store_true")
    p.add_argument("--no-require-functional", action="store_true")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logger("adv.wafamole", out_path.with_suffix(".log"))
    rng = random.Random(args.seed)
    random.seed(args.seed)  # SqlFuzzer uses module-level random

    # ---- victim ----
    log.info(f"Loading victim from {args.checkpoint}")
    model, pre, device, variant, accepted = load_victim(args.checkpoint)
    log.info(f"  variant={variant} device={device}")

    def score_fn(sqls):
        return batch_predict(model, pre, device, accepted, sqls, batch_size=64)
    def score_logits_fn(sqls):
        return batch_predict(model, pre, device, accepted, sqls,
                              batch_size=64, return_logits=True)

    # ---- seeds ----
    seeds = load_seed_attacks(args.seed_split, n=args.n_seeds, seed=args.seed)
    base_probs = score_fn([s["user_input"] for s in seeds])
    n_already_broken = int((base_probs < args.threshold).sum())
    log.info(f"  loaded {len(seeds)} seeds  pre-broken={n_already_broken}")
    log.info(f"\n{'='*70}\n  WAF-A-MoLE hill-climb: max_rounds={args.max_rounds} "
              f"round_size={args.round_size}  threshold={args.threshold}\n{'='*70}")

    # ---- attack ----
    adv_records = []
    per_seed = []
    n_success, n_skip = 0, 0
    t0 = time.time()
    for idx, seed_row in enumerate(seeds):
        seed_text = seed_row["user_input"]
        if args.limit_seeds_already_broken and base_probs[idx] < args.threshold:
            n_skip += 1
            continue

        best_payload, best_prob, history, mutators = attack_one_seed(
            seed_text, score_logits_fn, rng,
            max_rounds=args.max_rounds, round_size=args.round_size,
            threshold=args.threshold,
            require_functional=not args.no_require_functional,
        )

        per_seed.append({
            "id": seed_row["id"],
            "base_prob": float(base_probs[idx]),
            "best_prob": float(best_prob),
            "n_rounds": len(history),
            "len_seed": len(seed_text),
            "len_adv": len(best_payload),
        })

        if best_prob < args.threshold and best_payload != seed_text:
            n_success += 1
            adv_records.append({
                "user_input": best_payload,
                "label": "attack",
                "source": "adv_wafamole",
                "seed_id": seed_row["id"],
                "seed_input": seed_text,
                "n_rounds": len(history),
                "model_prob": float(best_prob),
                "technique": seed_row.get("technique"),
            })

        if (idx + 1) % 20 == 0 or idx == len(seeds) - 1:
            attempted = idx + 1 - n_skip
            asr = n_success / max(1, attempted)
            log.info(f"  [{idx+1:>4d}/{len(seeds)}]  succ={n_success}  "
                      f"ASR={asr:.2%}  elapsed={time.time()-t0:.0f}s")

    n_written = save_adv_records(out_path, adv_records)
    stats_path = out_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "checkpoint": str(args.checkpoint),
                "n_seeds": len(seeds),
                "max_rounds": args.max_rounds,
                "round_size": args.round_size,
                "threshold": args.threshold,
            },
            "n_already_broken": n_already_broken,
            "n_skipped": n_skip,
            "n_attempted": len(seeds) - n_skip,
            "n_success": n_success,
            "asr": n_success / max(1, len(seeds) - n_skip),
            "per_seed": per_seed,
        }, f, indent=2, ensure_ascii=False)

    log.info(f"\n  Wrote {n_written} adv records to {out_path}")
    log.info(f"  ASR: {n_success}/{len(seeds) - n_skip} = "
              f"{n_success / max(1, len(seeds) - n_skip):.2%}")


if __name__ == "__main__":
    main()
