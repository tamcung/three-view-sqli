#!/usr/bin/env python3
"""Adapter around WAFamole's 9 semantic-preserving SQL mutation operators.

WAFamole [Demetrio 2020] provides battle-tested implementations of the
9 operators used in Table 4.1 of Chapter 4. This module imports them and
exposes a uniform ``OPERATORS`` dict for the beam-search adversarial sample
generator (``adv_generator.py``).

The 9 operators are:
    reset_inline_comments, logical_invariant, change_tautologies,
    spaces_to_comments, spaces_to_whitespaces_alternatives, random_case,
    comment_rewriting, swap_int_repr, swap_keywords

WAFamole on RunPod (one-time setup):
    git clone https://github.com/AvalZ/WAF-A-MoLE third_party/wafamole
    pip install -e third_party/wafamole sqlparse networkx

Or copy a local checkout into ``third_party/wafamole`` (or
``third_party/waf_a_mole``) — this module probes both paths.
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable

# Probe known WAFamole install locations (project-local first, then dev box).
_WAFAMOLE_PATHS = [
    Path(__file__).resolve().parent.parent / "third_party" / "wafamole",
    Path(__file__).resolve().parent.parent / "third_party" / "waf_a_mole",
    Path("D:/rebuild/third_party/waf_a_mole"),
]
for _p in _WAFAMOLE_PATHS:
    if _p.exists():
        sys.path.insert(0, str(_p.resolve()))
        break

from wafamole.payloadfuzzer.sqlfuzzer import (
    reset_inline_comments,
    logical_invariant,
    change_tautologies,
    spaces_to_comments,
    spaces_to_whitespaces_alternatives,
    random_case,
    comment_rewriting,
    swap_int_repr,
    swap_keywords,
)

OPERATORS: dict[str, Callable[[str], str]] = {
    "reset_inline_comments": reset_inline_comments,
    "logical_invariant": logical_invariant,
    "change_tautologies": change_tautologies,
    "spaces_to_comments": spaces_to_comments,
    "spaces_to_whitespaces_alternatives": spaces_to_whitespaces_alternatives,
    "random_case": random_case,
    "comment_rewriting": comment_rewriting,
    "swap_int_repr": swap_int_repr,
    "swap_keywords": swap_keywords,
}


def apply_one(op_name: str, payload: str) -> str | None:
    """Apply a single operator once. Returns the variant string, or None
    if the operator did not apply or raised an exception."""
    fn = OPERATORS.get(op_name)
    if fn is None:
        raise KeyError(f"Unknown operator: {op_name}")
    try:
        out = fn(payload)
    except Exception:
        return None
    if not out or out == payload:
        return None
    return out


def all_variants(payload: str) -> list[tuple[str, str]]:
    """Apply every operator once to ``payload``; return [(op_name, variant)].

    WAFamole operators are non-deterministic — repeated calls may yield
    different variants. The beam-search generator can call this multiple
    times to expand the candidate pool.
    """
    out: list[tuple[str, str]] = []
    for name in OPERATORS:
        v = apply_one(name, payload)
        if v is not None:
            out.append((name, v))
    return out


if __name__ == "__main__":
    import random
    random.seed(0)
    samples = [
        "1' OR 1=1 --",
        "union select username,password from users",
        "1 AND sleep(5) #",
        "1' UNION SELECT NULL,version(),database() -- a",
    ]
    for s in samples:
        print(f">>> {s}")
        for name, v in all_variants(s):
            print(f"  [{name:40s}] {v}")
        print()
