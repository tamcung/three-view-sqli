#!/usr/bin/env python3
"""Shared utilities for the Chapter 4 adversarial training pipeline.

Provides:
  load_victim(ckpt_path)         -> (model, preprocessor, device, variant_name)
  batch_predict(model, sqls)     -> np.ndarray of sigmoid probabilities
  is_functional_sqli(payload)    -> bool   (libinjection-on-deobfuscated)
  save_adv_records(path, records)
  load_seed_attacks(split, ...)  -> list of {"user_input", "id", "technique", ...}

A "successful adversarial sample" is one that:
  - the victim model classifies as benign (prob < threshold)
  - libinjection's `is_sqli` flags positive AFTER our Hu-style deobfuscation,
    OR (optionally) sqlglot can parse it as a SELECT/UPDATE/etc

This decouples textual perturbation (what attackers do) from semantic
preservation (what we require for a sample to count as an attack).
"""
from __future__ import annotations
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import SamplePreprocessor                    # noqa: E402
from dataset import collate_three_view, move_batch_to           # noqa: E402
from ablation_models import build_model                         # noqa: E402
from libinjection_wrapper import is_sqli as _libinj_is_sqli      # noqa: E402
from deobfuscation import deobfuscate                            # noqa: E402


log = logging.getLogger("adversarial.utils")


# ============================================================
# Model loading
# ============================================================
class _MemDataset(Dataset):
    def __init__(self, recs):
        self.recs = recs
    def __len__(self):
        return len(self.recs)
    def __getitem__(self, i):
        return self.recs[i]


def load_victim(ckpt_path: str | Path, device: str | None = None):
    """Load a trained model + its preprocessor.

    Returns (model, preprocessor, device, variant_name, accepted_kwargs).
    """
    ckpt_path = Path(ckpt_path)
    cfg_path = ckpt_path.parent / "config.yaml"
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    variant = cfg.get("model_variant", "three_view")
    model = build_model(variant, cfg.get("model", {})).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    pre = SamplePreprocessor()
    accepted = set(inspect.signature(model.forward).parameters.keys())
    return model, pre, device, variant, accepted


@torch.no_grad()
def batch_predict(
    model,
    pre,
    device,
    accepted: set[str],
    sqls: Sequence[str],
    batch_size: int = 128,
    deobfuscate_first: bool = False,
    return_logits: bool = False,
) -> np.ndarray:
    """Predict per-sample attack scores.

    Returns sigmoid probabilities by default. With return_logits=True returns
    raw logits — preferable when the model saturates (sigmoid(30) == sigmoid(50)
    in float32 even though gradients differ).
    """
    if deobfuscate_first:
        sqls = [deobfuscate(s) for s in sqls]
    recs = [{**pre(s), "label_int": 1, "meta": {}} for s in sqls]
    loader = DataLoader(
        _MemDataset(recs), batch_size=batch_size, shuffle=False,
        collate_fn=collate_three_view, num_workers=0,
    )
    logits = []
    for batch in loader:
        batch = move_batch_to(batch, device)
        kwargs = dict(
            surface_ids=batch["surface_ids"], surface_mask=batch["surface_mask"],
            lex_ids=batch["lex_ids"], lex_mask=batch["lex_mask"],
            ast_ids=batch["ast_ids"], ast_mask=batch["ast_mask"],
            ast_valid=batch["ast_valid"],
            ast_node_ids=batch.get("ast_node_ids"),
            ast_parent=batch.get("ast_parent"),
            char_ids=batch.get("char_ids"),
            char_mask=batch.get("char_mask"),
        )
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        try:
            with torch.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16, enabled=device.type == "cuda",
            ):
                out = model(**kwargs)
        except Exception:
            out = model(**kwargs)
        logits.append(out["p_main"].float().cpu().numpy())
    logits = np.concatenate(logits)
    if return_logits:
        return logits
    return 1.0 / (1.0 + np.exp(-logits))


# ============================================================
# Validation: is the perturbed string still a functional SQLi?
# ============================================================
def is_functional_sqli(payload: str, allow_deobf: bool = True) -> bool:
    """Returns True iff the payload can plausibly execute as SQLi.

    The check first tries libinjection on the raw payload; if that fails and
    `allow_deobf` is True, it retries on the Hu-style deobfuscated form.
    Encoded payloads (base64, hex, charunicode) thus still count as SQLi
    even though libinjection cannot directly parse them — the operational
    threat model is "WAF receives encoded payload, application decodes".
    """
    if not payload:
        return False
    try:
        r = _libinj_is_sqli(payload)
        flag = r[0] if isinstance(r, tuple) else r
        if flag:
            return True
    except Exception:
        pass
    if not allow_deobf:
        return False
    try:
        decoded = deobfuscate(payload)
    except Exception:
        return False
    if decoded == payload:
        return False
    try:
        r = _libinj_is_sqli(decoded)
        flag = r[0] if isinstance(r, tuple) else r
        return bool(flag)
    except Exception:
        return False


def filter_successful_adversarials(
    seeds: Sequence[str],
    candidates: Sequence[str],
    probs: np.ndarray,
    threshold: float = 0.5,
    require_functional: bool = True,
) -> list[int]:
    """Indices of `candidates` that count as successful attacks: model says
    benign AND payload is still functionally SQLi AND it actually changed."""
    out = []
    for i, c in enumerate(candidates):
        if probs[i] >= threshold:
            continue
        if c == seeds[i]:
            continue
        if require_functional and not is_functional_sqli(c):
            continue
        out.append(i)
    return out


# ============================================================
# Seed sampling
# ============================================================
def load_seed_attacks(
    split_path: str | Path,
    n: int | None = None,
    seed: int = 42,
    only_techniques: set[str] | None = None,
) -> list[dict]:
    """Sample attack records from a {train,val,test}.jsonl split."""
    import random
    rows = []
    with open(split_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("label") != "attack":
                continue
            if only_techniques and r.get("technique") not in only_techniques:
                continue
            rows.append(r)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if n is not None and n < len(rows):
        rows = rows[:n]
    return rows


def save_adv_records(path: str | Path, records: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# ============================================================
# Misc
# ============================================================
def setup_logger(name: str, log_path: Path | None = None) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
        handlers=[h for h in [
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8") if log_path else None,
        ] if h is not None],
        force=True,
    )
    return logging.getLogger(name)
