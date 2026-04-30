#!/usr/bin/env python3
"""GA operator set for the search-based attacker.

The operator set is the **52 sqlmap tampers used in training augmentation**
(read from `data/tamper_split.json::train_tampers`). Each operator is the
raw sqlmap tamper function `tamper(payload)` from `external/sqlmap/tamper/`.

Why this choice (vs hand-crafted operators):
  Each individual operator was already in the model's training distribution
  (Chapter 3 augmented training with one-shot tampered payloads from each).
  The compositional generalization claim is therefore: **even though every
  atomic operator is in-distribution, GA-discovered chains of 2-5 of them
  are NOT** — and the model fails on those chains.

The 10 holdout tampers (heavy encoding: base64encode, charunicodeencode,
etc.) are deliberately NOT in this operator set. If GA could pick those,
it would trivially win in one step (we already know the model fails on
each holdout tamper alone). Restricting to in-distribution operators
forces GA to find genuinely compositional attacks.

Public interface (consumed by search_attacker.py):
    ALL_OPERATORS         : dict[name -> callable(s, rng) -> str]
    TERMINAL_OPERATORS    : set of operator names that may only appear last
                            (empty here — all sqlmap tampers compose freely)
    apply_chain(seed, ops, rng) -> str
"""
from __future__ import annotations
import importlib.util
import json
import logging
import random
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SQLMAP_REPO = ROOT.parent / "external" / "sqlmap"

# Tampers import lib.core.* — make sqlmap root importable
warnings.filterwarnings("ignore")
if str(SQLMAP_REPO) not in sys.path:
    sys.path.insert(0, str(SQLMAP_REPO))

_log = logging.getLogger("adversarial.mutations")

TAMPER_SPLIT_FILE = ROOT / "data" / "tamper_split.json"


def _load_tamper_fn(name: str):
    """Return the sqlmap `tamper(payload)` function for a tamper name, or None."""
    path = SQLMAP_REPO / "tamper" / f"{name}.py"
    if not path.exists():
        _log.warning(f"  tamper file missing: {path}")
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"_tamper_{name}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as e:
        _log.warning(f"  tamper {name} failed to import: {e}")
        return None
    fn = getattr(mod, "tamper", None)
    if fn is None:
        _log.warning(f"  tamper {name} has no tamper() function")
    return fn


def _wrap_tamper(name: str, fn):
    """Adapt sqlmap's `tamper(payload, **kwargs)` to our `(s, rng) -> str`.

    Signature wrapper:
      - sqlmap tampers don't take rng — randomness is internal (e.g.
        `randomcase` uses Python's `random` module). Different calls already
        produce different outputs, which is what GA needs.
      - On exception or None return, fall back to the input string (so a
        tamper that doesn't apply just acts as identity).
    """
    def wrapped(s: str, rng: random.Random) -> str:
        try:
            out = fn(s)
        except Exception:
            return s
        if out is None or not isinstance(out, str):
            return s
        return out
    wrapped.__name__ = f"sqlmap_{name}"
    return wrapped


def _build_operator_set() -> tuple[dict, list[str]]:
    """Read tamper_split.json, return (operators, missing_names)."""
    if not TAMPER_SPLIT_FILE.exists():
        raise FileNotFoundError(
            f"{TAMPER_SPLIT_FILE} not found. Run scripts/augment_with_tampers.py first."
        )
    with open(TAMPER_SPLIT_FILE, encoding="utf-8") as f:
        split = json.load(f)
    train_names = list(split.get("train_tampers", []))

    operators = {}
    missing = []
    for name in train_names:
        fn = _load_tamper_fn(name)
        if fn is None:
            missing.append(name)
            continue
        operators[name] = _wrap_tamper(name, fn)
    return operators, missing


# Build at import time so search_attacker can just import.
ALL_OPERATORS, _MISSING = _build_operator_set()
TERMINAL_OPERATORS: set[str] = set()  # all sqlmap tampers compose freely

if _MISSING:
    _log.info(f"  {len(_MISSING)} tampers failed to load: {_MISSING}")
_log.info(f"  loaded {len(ALL_OPERATORS)} sqlmap tampers as GA operators")


def apply_chain(seed: str, ops: list[str], rng: random.Random) -> str:
    """Apply a sequence of operator names in order. Unknown names are skipped."""
    s = seed
    for name in ops:
        fn = ALL_OPERATORS.get(name)
        if fn is None:
            continue
        s = fn(s, rng)
    return s
