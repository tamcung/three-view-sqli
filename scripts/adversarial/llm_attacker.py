#!/usr/bin/env python3
"""LLM-based adversarial attacker (§4.5).

Differentiates from Hu Xiuwen 2024's RL approach: instead of training a
policy network to emit token-edit actions, we use a frozen LLM as the
attacker. The LLM is prompted with (a) a seed SQLi payload, (b) the
victim's confidence on the seed, (c) a list of mutation strategies, and
asked to emit K candidate variants in JSON.

Each candidate is then:
  - run through the victim model
  - validated by `is_functional_sqli` (libinjection on raw or deobfuscated)
  - kept if model prob < threshold AND validates

The advantage over GA is semantic awareness: the LLM can reason about
what "looks SQL-like to the model but encodes attack" rather than blind
random mutation.

Provider abstraction: the LLM call is delegated to `LLM_CLIENT.generate`,
which can be backed by:
  - Anthropic SDK         (set ANTHROPIC_API_KEY env, model=claude-sonnet)
  - OpenAI-compat HTTP    (any local vLLM/LM Studio, --provider openai
                            --base-url http://...)
  - 'echo'                (deterministic stub for offline tests)

The attacker is round-aware: at round r, prior failed/successful chains
are folded into the prompt as in-context examples, so each successive
round explores the model's blind spots more aggressively.

Usage:
  export ANTHROPIC_API_KEY=...
  python -m scripts.adversarial.llm_attacker \
      --checkpoint results/tri_view_stage_aug/best_checkpoint.pt \
      --output data/adversarial/llm_v1.jsonl \
      --n-seeds 100 --variants-per-seed 6 --provider anthropic
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.adversarial.utils import (
    load_victim, batch_predict, is_functional_sqli,
    load_seed_attacks, save_adv_records, setup_logger,
)


# ============================================================
# LLM client abstraction
# ============================================================
class LLMClient:
    """Minimal interface so we can swap providers."""
    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        raise NotImplementedError


class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-5",
                 api_key: str | None = None):
        try:
            import anthropic
        except ImportError:
            raise RuntimeError(
                "Install `anthropic` package: pip install anthropic"
            )
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.model = model

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if b.type == "text")


class OpenAICompatClient(LLMClient):
    def __init__(self, base_url: str, model: str, api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Install `openai` package: pip install openai")
        self._client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
        self.model = model

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                       {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


class EchoClient(LLMClient):
    """Deterministic stub. Returns hand-written variants for offline runs."""
    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        # Extract payload from prompt and apply a couple deterministic mutations
        m = re.search(r"PAYLOAD:\s*```\s*(.*?)\s*```", user, flags=re.DOTALL)
        if not m:
            return json.dumps({"variants": []})
        p = m.group(1)
        out = [
            {"strategy": "url-encode", "payload": "".join(f"%{ord(c):02X}" if ord(c)<128 else c for c in p)},
            {"strategy": "case-mix", "payload": "".join(c.swapcase() if c.isalpha() and i%3==0 else c for i,c in enumerate(p))},
            {"strategy": "comment-pad", "payload": p.replace(" ", "/**/")},
        ]
        return json.dumps({"variants": out})


def make_client(provider: str, model: str, base_url: str | None = None) -> LLMClient:
    p = provider.lower()
    if p == "anthropic":
        return AnthropicClient(model=model)
    if p == "openai":
        if not base_url:
            raise RuntimeError("--base-url required for openai-compat provider")
        return OpenAICompatClient(base_url=base_url, model=model)
    if p == "echo":
        return EchoClient()
    raise ValueError(f"Unknown provider: {provider}")


# ============================================================
# Prompt
# ============================================================
SYSTEM_PROMPT = """You are an offensive security researcher testing a SQL injection detector.
Your task is to produce semantic-preserving variants of a known SQLi payload that may evade the detector.

A "semantic-preserving variant" means:
  - The variant must still execute the same logical injection if delivered to a vulnerable backend.
  - You may apply: case mixing, comment insertion, whitespace changes, equivalent operators
    (e.g. = → LIKE, OR → ||, AND → &&), URL/HTML/hex/unicode encoding, base64 wrapping if
    the application decodes it, splitting keywords with /**/ , adding trailing comments.
  - You may NOT replace the attack with a benign string. The goal of the payload (extract data,
    bypass auth, time-delay, etc.) must remain intact.

You are given:
  - the original payload
  - the detector's current confidence (1.0 = sure attack, 0.0 = sure benign)
  - examples of past variants that did or didn't fool the detector (may be empty)

Return a single JSON object:
  {"variants": [
      {"strategy": "<short name>", "payload": "<mutated string>"},
      ...
  ]}

Produce K candidate variants, each with a distinct strategy. Strategies should diversify across
encoding, syntax-equivalence, and structural mutation. Do not include explanations outside the
JSON object."""


def build_user_prompt(
    seed: str,
    seed_prob: float,
    k: int,
    history: list[dict] | None = None,
) -> str:
    parts = [f"PAYLOAD:\n```\n{seed}\n```\n",
             f"DETECTOR_CONFIDENCE: {seed_prob:.4f}",
             f"K: {k}"]
    if history:
        # Keep only last 6 history entries to bound prompt
        h = history[-6:]
        ex = []
        for r in h:
            ex.append(f"- strategy={r['strategy']!r}  payload={r['payload']!r}  "
                      f"detector={r['model_prob']:.3f}  "
                      f"functional={'yes' if r['functional'] else 'no'}")
        parts.append("PAST_VARIANTS_THIS_PAYLOAD:\n" + "\n".join(ex))
    parts.append(f"\nReturn JSON with exactly {k} variants.")
    return "\n\n".join(parts)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_variants(text: str) -> list[dict]:
    """Parse the LLM response. Accepts either a raw JSON object or one
    fenced in markdown."""
    m = _JSON_RE.search(text)
    if not m:
        return []
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # Attempt to fix common trailing-comma issues
        try:
            cleaned = re.sub(r",(\s*[\]\}])", r"\1", m.group(0))
            obj = json.loads(cleaned)
        except Exception:
            return []
    variants = obj.get("variants", []) if isinstance(obj, dict) else []
    out = []
    for v in variants:
        if isinstance(v, dict) and "payload" in v:
            out.append({"strategy": str(v.get("strategy", "")),
                        "payload": str(v["payload"])})
    return out


# ============================================================
# Per-seed attack loop
# ============================================================
def attack_one_seed(
    seed: str,
    seed_id: str,
    seed_prob: float,
    client: LLMClient,
    score_fn,
    rng: random.Random,
    threshold: float = 0.5,
    n_rounds: int = 2,
    variants_per_round: int = 6,
    require_functional: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Run multi-round attack. Returns (successes, full_history)."""
    history: list[dict] = []
    successes: list[dict] = []

    for r in range(n_rounds):
        user = build_user_prompt(seed, seed_prob, variants_per_round, history)
        try:
            text = client.generate(SYSTEM_PROMPT, user, max_tokens=2048)
        except Exception as e:
            logging.warning(f"  LLM call failed (seed={seed_id} round={r}): {e}")
            break

        variants = parse_variants(text)
        if not variants:
            logging.warning(f"  no parsable variants (seed={seed_id} round={r})")
            break

        payloads = [v["payload"] for v in variants]
        probs = score_fn(payloads)

        for v, p, pay in zip(variants, probs, payloads):
            functional = is_functional_sqli(pay) if require_functional else True
            entry = {
                "strategy": v["strategy"],
                "payload": pay,
                "model_prob": float(p),
                "functional": bool(functional),
                "round": r,
            }
            history.append(entry)
            if p < threshold and functional and pay != seed:
                successes.append(entry)

        # Early stop if we already have several successes
        if len(successes) >= variants_per_round:
            break

    return successes, history


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--seed-split", type=str,
                    default=str(ROOT / "data" / "splits" / "test.jsonl"))
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--variants-per-seed", type=int, default=6,
                    help="Variants per round.")
    p.add_argument("--rounds", type=int, default=2,
                    help="Multi-round refinement (later rounds see prior failures).")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--provider", choices=["anthropic", "openai", "echo"],
                    default="anthropic")
    p.add_argument("--model", type=str, default="claude-sonnet-4-5",
                    help="Anthropic: claude-sonnet-4-5 / claude-haiku-4-5 / "
                         "OpenAI-compat: model name on the endpoint.")
    p.add_argument("--base-url", type=str, default=None,
                    help="OpenAI-compat endpoint, e.g. http://localhost:8000/v1")

    p.add_argument("--limit-seeds-already-broken", action="store_true")
    p.add_argument("--no-require-functional", action="store_true")
    p.add_argument("--keep-only-best-per-seed", action="store_true",
                    help="Emit only the lowest-prob successful variant per seed.")
    args = p.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log = setup_logger("adv.llm", out_path.with_suffix(".log"))
    rng = random.Random(args.seed)

    # ---- victim ----
    log.info(f"Loading victim from {args.checkpoint}")
    model, pre, device, variant, accepted = load_victim(args.checkpoint)
    log.info(f"  variant={variant} device={device}")

    def score_fn(sqls: Sequence[str]):
        return batch_predict(model, pre, device, accepted, sqls,
                              batch_size=64, deobfuscate_first=False)

    # ---- LLM ----
    log.info(f"Initializing LLM client: {args.provider} / {args.model}")
    client = make_client(args.provider, args.model, args.base_url)

    # ---- seeds ----
    seeds = load_seed_attacks(args.seed_split, n=args.n_seeds, seed=args.seed)
    log.info(f"  loaded {len(seeds)} seed attacks")
    base_probs = score_fn([s["user_input"] for s in seeds])
    n_already_broken = int((base_probs < args.threshold).sum())
    log.info(f"  baseline misclassified: {n_already_broken}/{len(seeds)} "
              f"({n_already_broken/len(seeds):.1%})")

    # ---- attack loop ----
    log.info(f"\n{'='*70}\n  LLM attack: {args.rounds} rounds × "
              f"{args.variants_per_seed} variants  threshold={args.threshold}\n"
              f"{'='*70}")
    adv_records = []
    full_history = []
    n_success_seeds, n_skip = 0, 0
    total_variants_tried = 0
    t0 = time.time()
    for idx, seed_row in enumerate(seeds):
        seed_text = seed_row["user_input"]
        seed_id = seed_row["id"]
        if args.limit_seeds_already_broken and base_probs[idx] < args.threshold:
            n_skip += 1
            continue

        successes, history = attack_one_seed(
            seed_text, seed_id, float(base_probs[idx]), client, score_fn, rng,
            threshold=args.threshold, n_rounds=args.rounds,
            variants_per_round=args.variants_per_seed,
            require_functional=not args.no_require_functional,
        )
        full_history.append({
            "id": seed_id, "seed_input": seed_text,
            "base_prob": float(base_probs[idx]),
            "history": history,
            "n_successes": len(successes),
        })
        total_variants_tried += len(history)

        if successes:
            n_success_seeds += 1
            chosen = sorted(successes, key=lambda r: r["model_prob"])
            if args.keep_only_best_per_seed:
                chosen = chosen[:1]
            for s in chosen:
                adv_records.append({
                    "user_input": s["payload"],
                    "label": "attack",
                    "source": "adv_llm",
                    "seed_id": seed_id,
                    "seed_input": seed_text,
                    "strategy": s["strategy"],
                    "model_prob": s["model_prob"],
                    "round": s["round"],
                    "technique": seed_row.get("technique"),
                })

        if (idx + 1) % 10 == 0 or idx == len(seeds) - 1:
            attempted = idx + 1 - n_skip
            asr_seed = n_success_seeds / max(1, attempted)
            log.info(f"  [{idx+1:>4d}/{len(seeds)}]  success_seeds={n_success_seeds}  "
                      f"variants_tried={total_variants_tried}  ASR_seed={asr_seed:.2%}  "
                      f"elapsed={time.time()-t0:.0f}s")

    # ---- write ----
    n_written = save_adv_records(out_path, adv_records)
    stats_path = out_path.with_suffix(".stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "checkpoint": str(args.checkpoint),
                "provider": args.provider,
                "model": args.model,
                "n_seeds": len(seeds),
                "variants_per_seed": args.variants_per_seed,
                "rounds": args.rounds,
                "threshold": args.threshold,
            },
            "n_already_broken": n_already_broken,
            "n_skipped": n_skip,
            "n_attempted": len(seeds) - n_skip,
            "n_success_seeds": n_success_seeds,
            "n_total_variants_tried": total_variants_tried,
            "asr_seed": n_success_seeds / max(1, len(seeds) - n_skip),
            "per_seed": full_history,
        }, f, indent=2, ensure_ascii=False)

    log.info(f"\n  Wrote {n_written} adv records to {out_path}")
    log.info(f"  Wrote stats to {stats_path}")
    log.info(f"  Per-seed ASR: {n_success_seeds}/{len(seeds) - n_skip} = "
              f"{n_success_seeds / max(1, len(seeds) - n_skip):.2%}")


if __name__ == "__main__":
    main()
