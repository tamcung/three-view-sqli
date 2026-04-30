#!/usr/bin/env python3
"""Search-based adversarial attacker (genetic algorithm over mutation chains).

For each seed attack we evolve a population of mutation chains. Each chain
is a list of operator names from `mutations.ALL_OPERATORS`. Fitness rewards
chains that drive the victim model's `attack` probability below threshold,
provided the resulting string is still functionally SQLi.

Usage:
  python -m scripts.adversarial.search_attacker \
      --checkpoint results/tri_view_stage_aug/best_checkpoint.pt \
      --seed-split data/splits/test.jsonl \
      --n-seeds 200 \
      --output data/adversarial/search_v1.jsonl

Outputs:
  - <output>.jsonl   one record per successful adversarial sample, with
                     fields {user_input, label="attack", source="adv_search",
                             seed_id, seed_input, ops, model_prob}
  - <output>.stats.json   per-seed best-prob trace + ASR (attack success rate)

This is the threat model implementation for §4.3 (search-based) and the
"AdvSEARCH" attacker compared in §4.6.
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

from scripts.adversarial.utils import (
    load_victim, batch_predict, is_functional_sqli,
    load_seed_attacks, save_adv_records, setup_logger,
)
from scripts.adversarial.mutations import (
    ALL_OPERATORS, TERMINAL_OPERATORS, apply_chain,
)

OP_NAMES = list(ALL_OPERATORS.keys())
NON_TERMINAL = [o for o in OP_NAMES if o not in TERMINAL_OPERATORS]
TERMINAL = list(TERMINAL_OPERATORS)


# ============================================================
# Chromosome ops
# ============================================================
def random_chromosome(rng: random.Random, max_len: int = 4) -> list[str]:
    n = rng.randint(1, max_len)
    chain = rng.sample(NON_TERMINAL, k=min(n, len(NON_TERMINAL)))
    if TERMINAL and rng.random() < 0.25:
        chain.append(rng.choice(TERMINAL))
    return chain


def crossover(a: list[str], b: list[str], rng: random.Random) -> list[str]:
    """Take a prefix of a and a suffix of b. Move terminal op (if any) to end."""
    cut_a = rng.randint(0, len(a))
    cut_b = rng.randint(0, len(b))
    child = a[:cut_a] + b[cut_b:]
    # dedup adjacent (keeps diversity)
    out = []
    for op in child:
        if not out or out[-1] != op:
            out.append(op)
    # enforce at most one terminal at end
    out_nt = [o for o in out if o not in TERMINAL_OPERATORS]
    out_t  = [o for o in out if o in TERMINAL_OPERATORS]
    return out_nt[:5] + out_t[:1]


def mutate(chain: list[str], rng: random.Random) -> list[str]:
    chain = list(chain)
    r = rng.random()
    has_terminal = bool(TERMINAL)
    if r < 0.4 and len(chain) < 6:
        # insert
        op = rng.choice(NON_TERMINAL)
        pos = rng.randint(0, len(chain))
        chain.insert(pos, op)
    elif r < 0.7 and len(chain) > 1:
        # delete
        del chain[rng.randint(0, len(chain) - 1)]
    elif r < 0.85 and chain:
        # swap one operator
        i = rng.randint(0, len(chain) - 1)
        if chain[i] in TERMINAL_OPERATORS and has_terminal:
            chain[i] = rng.choice(TERMINAL)
        else:
            chain[i] = rng.choice(NON_TERMINAL)
    elif has_terminal:
        # toggle terminal at the end (only when TERMINAL is non-empty)
        if chain and chain[-1] in TERMINAL_OPERATORS:
            chain.pop()
        else:
            chain.append(rng.choice(TERMINAL))
    else:
        # No terminals available — shuffle order to explore composition orderings
        rng.shuffle(chain)
    # cleanup
    chain_nt = [o for o in chain if o not in TERMINAL_OPERATORS]
    chain_t  = [o for o in chain if o in TERMINAL_OPERATORS]
    return chain_nt[:5] + chain_t[:1] if has_terminal else chain_nt[:5]


# ============================================================
# GA driver
# ============================================================
def evolve_for_seed(
    seed: str,
    pop_size: int,
    n_generations: int,
    threshold: float,
    score_fn,                # callable(list[str]) -> np.ndarray of probs
    rng: random.Random,
    elitism: int = 4,
    require_functional: bool = True,
) -> tuple[list[str], float, str, list[float]]:
    """Run GA for one seed payload.

    Returns (best_chain, best_prob, best_payload, prob_trace).
    `prob_trace[g]` = best (lowest) attack-prob seen by end of generation g.
    """
    population = [random_chromosome(rng) for _ in range(pop_size)]
    best_chain, best_prob, best_payload = [], 1.0, seed
    trace = []

    for g in range(n_generations):
        # decode + score
        payloads = [apply_chain(seed, c, rng) for c in population]
        probs = score_fn(payloads)

        # successful = prob low AND still SQLi
        for i, (c, p, pay) in enumerate(zip(population, probs, payloads)):
            if p >= threshold:
                continue
            if pay == seed:
                continue
            if require_functional and not is_functional_sqli(pay):
                # invalid attack — penalize
                probs[i] = 1.0
                continue
            if p < best_prob:
                best_prob = float(p)
                best_chain = list(c)
                best_payload = pay
        trace.append(best_prob)
        if best_prob < threshold and g >= 2:
            # found something cheap — keep evolving but at half budget
            if g >= max(3, n_generations // 2):
                break

        # selection: keep top `elitism`, fill rest by tournament
        order = np.argsort(probs)
        elites = [population[i] for i in order[:elitism]]
        next_pop = list(elites)
        while len(next_pop) < pop_size:
            i = rng.randint(0, pop_size - 1)
            j = rng.randint(0, pop_size - 1)
            parent_a = population[i] if probs[i] < probs[j] else population[j]
            i = rng.randint(0, pop_size - 1)
            j = rng.randint(0, pop_size - 1)
            parent_b = population[i] if probs[i] < probs[j] else population[j]
            child = crossover(parent_a, parent_b, rng)
            if rng.random() < 0.5:
                child = mutate(child, rng)
            if not child:
                child = random_chromosome(rng)
            next_pop.append(child)
        population = next_pop

    return best_chain, best_prob, best_payload, trace


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True,
                    help="Output .jsonl path. .stats.json sibling is also written.")
    p.add_argument("--seed-split", type=str,
                    default=str(ROOT / "data" / "splits" / "test.jsonl"))
    p.add_argument("--n-seeds", type=int, default=200)
    p.add_argument("--pop-size", type=int, default=24)
    p.add_argument("--generations", type=int, default=12)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-require-functional", action="store_true",
                    help="Don't require libinjection to flag the perturbed string. "
                         "Use only for diagnostic runs.")
    p.add_argument("--limit-seeds-already-broken", action="store_true",
                    help="Skip seeds the model already misclassifies (saves budget).")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logger("adv.search", out_path.with_suffix(".log"))
    rng = random.Random(args.seed)

    # ---- victim ----
    log.info(f"Loading victim from {args.checkpoint}")
    model, pre, device, variant, accepted = load_victim(args.checkpoint)
    log.info(f"  variant={variant} device={device}")

    def score_fn(sqls):
        return batch_predict(model, pre, device, accepted, sqls,
                              batch_size=128, deobfuscate_first=False)

    # ---- seeds ----
    seeds = load_seed_attacks(args.seed_split, n=args.n_seeds, seed=args.seed)
    log.info(f"  loaded {len(seeds)} seed attacks from {args.seed_split}")

    # baseline: which seeds does the model already get wrong?
    base_probs = score_fn([s["user_input"] for s in seeds])
    n_already_broken = int((base_probs < args.threshold).sum())
    log.info(f"  baseline misclassified seeds: {n_already_broken}/{len(seeds)} "
              f"({n_already_broken/len(seeds):.1%})")

    # ---- per-seed GA ----
    log.info(f"\n{'='*70}\n  GA: pop={args.pop_size} gens={args.generations} "
              f"threshold={args.threshold}\n{'='*70}")
    adv_records = []
    per_seed_trace = []
    n_success, n_skip = 0, 0
    t0 = time.time()
    for idx, seed in enumerate(seeds):
        seed_text = seed["user_input"]
        if args.limit_seeds_already_broken and base_probs[idx] < args.threshold:
            n_skip += 1
            per_seed_trace.append({"id": seed["id"], "skipped": True,
                                    "base_prob": float(base_probs[idx])})
            continue

        best_chain, best_prob, best_payload, trace = evolve_for_seed(
            seed_text, args.pop_size, args.generations, args.threshold,
            score_fn, rng,
            require_functional=not args.no_require_functional,
        )

        record = {
            "id": seed["id"],
            "base_prob": float(base_probs[idx]),
            "best_prob": float(best_prob),
            "ops": best_chain,
            "trace": [float(x) for x in trace],
            "skipped": False,
        }
        per_seed_trace.append(record)

        if best_prob < args.threshold and best_payload != seed_text:
            n_success += 1
            adv_records.append({
                "user_input": best_payload,
                "label": "attack",
                "source": "adv_search",
                "seed_id": seed["id"],
                "seed_input": seed_text,
                "ops": best_chain,
                "model_prob": float(best_prob),
                "technique": seed.get("technique"),
            })

        if (idx + 1) % 20 == 0 or idx == len(seeds) - 1:
            asr = n_success / max(1, idx + 1 - n_skip)
            log.info(f"  [{idx+1:>4d}/{len(seeds)}]  success={n_success}  "
                      f"skip={n_skip}  ASR={asr:.2%}  elapsed={time.time()-t0:.0f}s")

    # ---- write ----
    n_written = save_adv_records(out_path, adv_records)
    stats_path = out_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "checkpoint": str(args.checkpoint),
                "n_seeds": len(seeds),
                "pop_size": args.pop_size,
                "generations": args.generations,
                "threshold": args.threshold,
                "seed_split": str(args.seed_split),
            },
            "n_already_broken": n_already_broken,
            "n_skipped": n_skip,
            "n_attempted": len(seeds) - n_skip,
            "n_success": n_success,
            "asr": n_success / max(1, len(seeds) - n_skip),
            "per_seed": per_seed_trace,
        }, f, indent=2, ensure_ascii=False)

    log.info(f"\n  Wrote {n_written} adversarial samples to {out_path}")
    log.info(f"  Wrote stats to {stats_path}")
    log.info(f"  ASR (excluding pre-broken): "
              f"{n_success}/{len(seeds) - n_skip} = "
              f"{n_success / max(1, len(seeds) - n_skip):.2%}")


if __name__ == "__main__":
    main()
