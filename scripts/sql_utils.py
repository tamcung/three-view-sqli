#!/usr/bin/env python3
"""Shared SQL parsing / AST utilities used by split_dataset.py and the
preprocessing layer.

  - parse_strict(sql)      MySQL-strict parse (returns None on failure)
  - ast_signature(tree)    Hashable structural signature with pure-constant
                           subtree collapse (so that 1, 2, 'foo', NULL all
                           reduce to a single 'Literal' token).
  - MINI_TEMPLATES         per-slot-context wrappers used to embed a raw
                           user_input into a canonical SQL skeleton for
                           AST view computation.
"""
from __future__ import annotations
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("sqlglot").setLevel(logging.ERROR)

import sqlglot
from sqlglot import expressions as exp


# ============================================================
# Mini wrappers per slot-context
# ============================================================
MINI_TEMPLATES = {
    "bare_numeric":            "SELECT * FROM t WHERE id = {slot}",
    "bare_string":             "SELECT * FROM t WHERE name = '{slot}'",
    "bare_identifier":         "SELECT {slot} FROM t",
    "directly_quoted_string":  "SELECT * FROM t WHERE name = '{slot}'",
    "sql_fragment":            "SELECT * FROM t WHERE {slot}",
}

# Default wrapper for payload-only mode (when slot_context is unknown):
# numeric is the most permissive — string injections still parse since
# they end up unquoted; bare numbers parse as Literal.
DEFAULT_WRAPPER = MINI_TEMPLATES["bare_numeric"]


def wrap(payload: str, slot_context: str | None = None) -> str:
    tpl = MINI_TEMPLATES.get(slot_context, DEFAULT_WRAPPER) if slot_context else DEFAULT_WRAPPER
    return tpl.format(slot=payload)


# ============================================================
# Strict MySQL parser
# ============================================================
def parse_strict(sql: str):
    """Returns the parsed tree, or None if MySQL strict parsing fails."""
    if not sql:
        return None
    try:
        # error_level=RAISE means strict; the default IGNORE silently
        # recovers from invalid SQL which we don't want here.
        tree = sqlglot.parse_one(sql, read="mysql", error_level=sqlglot.ErrorLevel.RAISE)
    except Exception:
        return None
    if tree is None or isinstance(tree, exp.Command):
        return None
    return tree


# ============================================================
# Constant-collapse AST signature
# ============================================================
_CONST_ARITH_TYPES = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)


def is_pure_constant_expr(node) -> bool:
    """A subtree is "pure constant" if every leaf is a Literal/Null/Boolean
    and every internal node is an arithmetic operator over pure-constants
    (or Neg / Paren of one).
    """
    if isinstance(node, (exp.Literal, exp.Null, exp.Boolean)):
        return True
    if isinstance(node, exp.Neg):
        return is_pure_constant_expr(node.this)
    if isinstance(node, exp.Paren):
        return is_pure_constant_expr(node.this)
    if isinstance(node, _CONST_ARITH_TYPES):
        left = node.args.get("this")
        right = node.args.get("expression")
        return (left is not None and right is not None
                and is_pure_constant_expr(left) and is_pure_constant_expr(right))
    return False


def ast_signature(node) -> tuple:
    """Produce a hashable structural signature.

    - Pure-constant subtrees collapse to the single token "Literal"
    - Identifier nodes collapse to "Identifier"
    - Other nodes recurse with their children's signatures
    """
    if not isinstance(node, exp.Expression):
        return ()

    if is_pure_constant_expr(node):
        return ("Literal",)

    if isinstance(node, exp.Identifier):
        return ("Identifier",)

    if isinstance(node, exp.Column):
        return ("Column",)  # collapse table.col vs col

    name = type(node).__name__
    children = []
    for arg in node.args.values():
        if isinstance(arg, list):
            for a in arg:
                if isinstance(a, exp.Expression):
                    children.append(ast_signature(a))
        elif isinstance(arg, exp.Expression):
            children.append(ast_signature(arg))
    return (name, tuple(children))


if __name__ == "__main__":
    samples = [
        "SELECT * FROM t WHERE id = 1",
        "SELECT * FROM t WHERE id = 999",
        "SELECT * FROM t WHERE id = -1 + 5",
        "SELECT * FROM t WHERE id = 1 OR 1=1",
        "SELECT * FROM t WHERE name = 'admin'",
    ]
    for s in samples:
        tree = parse_strict(s)
        sig = ast_signature(tree) if tree else None
        print(f"  {s}")
        print(f"    sig: {sig}")
