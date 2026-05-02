#!/usr/bin/env python3
"""Beam-search adversarial SQLi sample generator (Algorithm 4.1).

Given a target detection model and a set of mutation operators, this module
searches for variants of an attack payload that make the model output a
low malicious probability. Variants that successfully flip the prediction
to benign are kept as adversarial samples; near-miss high-perturbation
variants are kept as hard samples.

Usage (CLI):
    python -m src.adv_generator \
        --ckpt-dir results_kaggle/three_view_d128_L2 \
        --input data/external/kaggle_sqli/jsonl/train.jsonl \
        --output results/ch4/adv_round0.jsonl \
        --beam 5 --max-steps 10 --query-budget 200 \
        --max-samples 5000

Usage (programmatic):
    from adv_generator import generate_adversarial
    A, H = generate_adversarial(payload, model, preprocessor, ...)
"""
from __future__ import annotations
import argparse
import inspect
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocessing import SamplePreprocessor  # noqa: E402
from dataset import collate_three_view, move_batch_to  # noqa: E402
from ablation_models import build_model  # noqa: E402
from adv_operators import OPERATORS  # noqa: E402

# Note: no semantic-preservation gate. WAFamole operators are
# semantic-preserving by construction (see Demetrio 2020 §3); any external
# SQLi-classifier-based gate (e.g. libinjection) would filter out exactly
# the high-value adversarial samples adversarial training is meant to learn
# from, defeating the purpose. Variants that mangle the payload into noise
# are extremely rare and tolerated as label noise.


# ============================================================
# Result containers
# ============================================================
@dataclass
class BeamItem:
    payload: str
    score: float          # current model's malicious probability (lower = better adv)
    op_chain: list[str] = field(default_factory=list)


@dataclass
class GenResult:
    original: str
    adversarials: list[BeamItem]   # successful (model says benign)
    hard_cases: list[BeamItem]     # not-quite-success but high probability drop
    queries_used: int


# ============================================================
# Oracle: batched malicious-probability prediction
# ============================================================
class ModelOracle:
    """Wraps a trained detector for batched prob queries."""

    def __init__(self, ckpt_dir: Path, device: torch.device | None = None,
                 batch_size: int = 64, amp: bool = True):
        cfg_path = ckpt_dir / "config.yaml"
        ckpt_path = ckpt_dir / "best_checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
        self.cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        variant = self.cfg.get("model_variant", "three_view")
        self.model = build_model(variant, self.cfg.get("model", {})).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # Pick preprocessor by config
        if self.cfg.get("preprocessor") == "mvc":
            from mvc_preprocessor import MVCSamplePreprocessor, MVCVocab
            vocab_path = ROOT / "src" / "mvc_vocab.json"
            self.pre = MVCSamplePreprocessor(MVCVocab.from_file(vocab_path))
        else:
            self.pre = SamplePreprocessor()

        self._accepted = set(
            inspect.signature(self.model.forward).parameters.keys()
        )
        self.batch_size = batch_size
        self.amp = amp and self.device.type == "cuda"

    @torch.no_grad()
    def query(self, payloads: list[str]) -> np.ndarray:
        """Return malicious probability ∈ [0,1] for each payload."""
        from torch.utils.data import DataLoader, Dataset

        recs = [
            {**self.pre(p), "label_int": 1, "meta": {}}
            for p in payloads
        ]

        class _D(Dataset):
            def __init__(self, recs): self.recs = recs
            def __len__(self): return len(self.recs)
            def __getitem__(self, i): return self.recs[i]

        loader = DataLoader(
            _D(recs), batch_size=self.batch_size, shuffle=False,
            collate_fn=collate_three_view, num_workers=0,
        )
        probs = []
        for batch in loader:
            batch = move_batch_to(batch, self.device)
            kwargs = dict(
                surface_ids=batch["surface_ids"], surface_mask=batch["surface_mask"],
                lex_ids=batch["lex_ids"], lex_mask=batch["lex_mask"],
                ast_ids=batch["ast_ids"], ast_mask=batch["ast_mask"],
                ast_valid=batch["ast_valid"],
                char_ids=batch.get("char_ids"),
                char_mask=batch.get("char_mask"),
            )
            kwargs = {k: v for k, v in kwargs.items() if k in self._accepted}
            try:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16, enabled=self.amp,
                ):
                    out = self.model(**kwargs)
            except Exception:
                out = self.model(**kwargs)
            logits = out["p_main"].float().cpu().numpy()
            probs.append(1.0 / (1.0 + np.exp(-logits)))
        return np.concatenate(probs)


# ============================================================
# Beam search generator (Algorithm 4.1)
# ============================================================
def generate_adversarial(
    original: str,
    oracle: ModelOracle,
    beam: int = 5,
    max_steps: int = 10,
    query_budget: int = 200,
    success_threshold: float = 0.5,
    hard_drop_threshold: float = 0.3,
    operator_repeats: int = 2,
) -> GenResult:
    """Beam search for adversarial variants of ``original``.

    Returns adversarials (model malicious_prob < success_threshold) and
    hard cases (any beam item whose score dropped by ≥ hard_drop_threshold
    relative to the original probability) found during the search.
    """
    # Initial query
    probs = oracle.query([original])
    queries = 1
    init_score = float(probs[0])
    beams = [BeamItem(payload=original, score=init_score)]
    adversarials: list[BeamItem] = []
    hard_cases: list[BeamItem] = []

    for step in range(max_steps):
        if queries >= query_budget:
            break

        # Expand: apply each operator multiple times to each beam item
        candidates: list[BeamItem] = []
        for b in beams:
            for op_name, op_fn in OPERATORS.items():
                for _ in range(operator_repeats):
                    try:
                        v = op_fn(b.payload)
                    except Exception:
                        continue
                    if not v or v == b.payload:
                        continue
                    candidates.append(BeamItem(
                        payload=v, score=1.0,  # placeholder, refilled below
                        op_chain=b.op_chain + [op_name],
                    ))

        if not candidates:
            break

        # Cap candidates against remaining query budget
        budget_remaining = query_budget - queries
        if len(candidates) > budget_remaining:
            candidates = random.sample(candidates, budget_remaining)

        # Score candidates in batch
        cand_probs = oracle.query([c.payload for c in candidates])
        queries += len(candidates)
        for c, p in zip(candidates, cand_probs):
            c.score = float(p)

        # Collect successes
        for c in candidates:
            if c.score < success_threshold:
                adversarials.append(c)
            elif init_score - c.score >= hard_drop_threshold:
                hard_cases.append(c)

        # Early stop if we have any success
        if adversarials:
            break

        # Keep top-beam (lowest score = closest to flipping)
        candidates.sort(key=lambda x: x.score)
        beams = candidates[:beam]

    return GenResult(
        original=original,
        adversarials=adversarials,
        hard_cases=hard_cases,
        queries_used=queries,
    )


# ============================================================
# CLI entry: generate from a JSONL of attack payloads
# ============================================================
def _record_to_dict(r: GenResult, source_id: str | None) -> list[dict]:
    """Flatten a GenResult into JSONL output records."""
    out = []
    for tag, items in [("adversarial", r.adversarials), ("hard", r.hard_cases)]:
        for it in items:
            out.append({
                "user_input": it.payload,
                "label": "attack",
                "source": "adv_generated",
                "tag": tag,
                "op_chain": it.op_chain,
                "model_score": it.score,
                "original_id": source_id,
                "original_payload": r.original,
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="Trained model dir")
    ap.add_argument("--input", required=True, help="Input JSONL with attack payloads")
    ap.add_argument("--output", required=True, help="Output JSONL")
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--query-budget", type=int, default=200)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap on number of input payloads to process")
    ap.add_argument("--success-threshold", type=float, default=0.5)
    ap.add_argument("--hard-drop-threshold", type=float, default=0.3)
    ap.add_argument("--operator-repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[adv_generator] Loading oracle from {args.ckpt_dir} ...", flush=True)
    oracle = ModelOracle(Path(args.ckpt_dir))
    print(f"[adv_generator] device={oracle.device}", flush=True)

    # Load attack payloads
    payloads = []
    for line in open(args.input, encoding="utf-8"):
        r = json.loads(line)
        if r.get("label") == "attack":
            payloads.append((r.get("id", ""), r["user_input"]))
        if args.max_samples and len(payloads) >= args.max_samples:
            break
    print(f"[adv_generator] Processing {len(payloads)} attack payloads ...", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_adv = n_hard = 0
    t0 = time.time()
    with open(args.output, "w", encoding="utf-8") as fout:
        for i, (pid, payload) in enumerate(payloads):
            r = generate_adversarial(
                payload, oracle,
                beam=args.beam,
                max_steps=args.max_steps,
                query_budget=args.query_budget,
                success_threshold=args.success_threshold,
                hard_drop_threshold=args.hard_drop_threshold,
                operator_repeats=args.operator_repeats,
            )
            for rec in _record_to_dict(r, pid):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_adv += len(r.adversarials)
            n_hard += len(r.hard_cases)
            if (i + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(payloads) - i - 1) / rate
                print(f"  [{i+1}/{len(payloads)}] adv={n_adv} hard={n_hard}  "
                      f"rate={rate:.2f}/s  ETA={eta/60:.1f} min", flush=True)
    print(f"[adv_generator] Done. Wrote {n_adv} adv + {n_hard} hard to {args.output}",
          flush=True)


if __name__ == "__main__":
    main()
