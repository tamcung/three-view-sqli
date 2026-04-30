#!/usr/bin/env python3
"""
SQL → 三视图 token ids.

Provides:
  - lexicalize(sql) → list[str]    (libinjection type codes, drop comments)
  - serialize_ast(sql) → list[str] | None    (sqlglot pre-order brackets)
  - SamplePreprocessor class       (combines all three views)
"""
from __future__ import annotations
import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("sqlglot").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Use absolute import that works whether `src` is a package or scripts CWD
try:
    from .libinjection_wrapper import tokenize as libinj_tokenize, FLAG_QUOTE_NONE, FLAG_SQL_MYSQL
except ImportError:
    from libinjection_wrapper import tokenize as libinj_tokenize, FLAG_QUOTE_NONE, FLAG_SQL_MYSQL

import sqlglot
from sqlglot import expressions as exp


# ============================================================
# Lexical view: libinjection type codes (drop comments)
# ============================================================
LEX_DROP = frozenset({"c"})

LEX_VOCAB_TOKENS = [
    "<PAD>", "<UNK>", "<CLS>",
    "s", "n", "1", "k", "E", "U", "f", "o", "&", "v", "B", "t", "T", "X",
    "(", ")", ",", ";", ":", "\\", "?",
]
LEX_VOCAB = {tok: i for i, tok in enumerate(LEX_VOCAB_TOKENS)}


def lexicalize(sql: str) -> list[str]:
    """SQL → libinjection type-code sequence (comments dropped)."""
    raw = libinj_tokenize(sql, flags=FLAG_QUOTE_NONE | FLAG_SQL_MYSQL)
    return [t for t, _ in raw if t and t not in LEX_DROP]


# ============================================================
# AST view: sqlglot pre-order with brackets
# ============================================================
AST_VOCAB_TOKENS = [
    "<PAD>", "<UNK>", "<CLS>",
    "[", "]",
    "<ID>", "<STR>", "<NUM>", "<FUNC>",
    "TRUE", "FALSE", "NULL", "Star",
    # 73 sqlglot node types from full WAF-A-MoLE audit
    "Column", "Table", "Where", "EQ", "Select", "Update", "From",
    "Insert", "Delete", "Like", "Limit", "Neg", "Tuple", "Values",
    "Schema", "LT", "GTE", "Offset", "TableAlias", "Alias", "Paren",
    "Subquery", "Block", "Mod", "Mul", "BitwiseXor", "GT", "LTE",
    "NEQ", "Div", "Add", "Sub", "And", "Or", "Not", "In", "Between",
    "Is", "Order", "Group", "Having", "Distinct", "All", "Union",
    "Intersect", "Except", "Case", "When", "If", "Cast", "TryCast",
    "Convert", "Concat", "BitwiseAnd", "BitwiseOr", "BitwiseLeftShift",
    "BitwiseRightShift", "BitwiseNot", "Anonymous", "Window", "Lambda",
    "Bracket", "Dot", "Default", "Reference", "Properties",
    "Identifier", "DataType", "AlterTable", "Use", "DropTable",
    "Create", "Drop", "Truncate", "Pivot", "Unpivot", "Lateral",
    "Group", "Trim", "Substring", "JoinHint", "Join",
]
AST_VOCAB = {tok: i for i, tok in enumerate(AST_VOCAB_TOKENS)}


def serialize_ast_tree(node, out: list[str]) -> None:
    """Pre-order + brackets serialization with placeholders for content."""
    if not isinstance(node, exp.Expression):
        return
    if isinstance(node, exp.Identifier):
        out.append("<ID>"); return
    if isinstance(node, exp.Literal):
        out.append("<STR>" if node.is_string else "<NUM>"); return
    if isinstance(node, exp.Boolean):
        out.append("TRUE" if node.this else "FALSE"); return
    if isinstance(node, exp.Null):
        out.append("NULL"); return
    if isinstance(node, exp.Star):
        out.append("Star"); return
    if isinstance(node, exp.Func):
        out.append("[")
        out.append("<FUNC>")
        for arg in node.args.values():
            if isinstance(arg, list):
                for a in arg:
                    serialize_ast_tree(a, out)
            else:
                serialize_ast_tree(arg, out)
        out.append("]")
        return
    type_name = type(node).__name__
    out.append("[")
    out.append(type_name)
    for arg in node.args.values():
        if isinstance(arg, list):
            for a in arg:
                serialize_ast_tree(a, out)
        else:
            serialize_ast_tree(arg, out)
    out.append("]")


def serialize_ast(sql: str) -> list[str] | None:
    """SQL → bracketed pre-order token sequence. None if parse fails."""
    for d in ("mysql", "postgres", "tsql"):
        try:
            tree = sqlglot.parse_one(sql, read=d, error_level=sqlglot.ErrorLevel.IGNORE)
        except Exception:
            continue
        if tree is None or isinstance(tree, exp.Command):
            continue
        try:
            out: list[str] = []
            serialize_ast_tree(tree, out)
            return out
        except Exception:
            return None
    return None


# ============================================================
# Tree structure extraction (for Tree-LSTM baseline)
# ============================================================
def _node_label(node) -> str:
    """Map an exp node to a single token label matching AST_VOCAB."""
    if isinstance(node, exp.Identifier):
        return "<ID>"
    if isinstance(node, exp.Literal):
        return "<STR>" if node.is_string else "<NUM>"
    if isinstance(node, exp.Boolean):
        return "TRUE" if node.this else "FALSE"
    if isinstance(node, exp.Null):
        return "NULL"
    if isinstance(node, exp.Star):
        return "Star"
    if isinstance(node, exp.Func):
        return "<FUNC>"
    return type(node).__name__


def serialize_ast_tree_struct(sql: str):
    """Parse SQL and return (node_label_ids, parent_indices) in post-order
    so that all children precede their parent (root is last).

    Returns (None, None) on parse failure.
    """
    tree = None
    for d in ("mysql", "postgres", "tsql"):
        try:
            tree = sqlglot.parse_one(sql, read=d,
                                       error_level=sqlglot.ErrorLevel.IGNORE)
        except Exception:
            tree = None
            continue
        if tree is not None and not isinstance(tree, exp.Command):
            break
        tree = None
    if tree is None:
        return None, None

    # Post-order DFS, assign incremental indices
    labels: list[str] = []
    parents: list[int] = []
    # Stack-based post-order
    stack = [(tree, False, -1)]
    pending_parent_label_index_map = {}
    # We use recursive Python — tree depth is bounded by SQL complexity (<100)
    def visit(node, parent_idx):
        if not isinstance(node, exp.Expression):
            return None
        # Visit children first
        child_indices = []
        for arg in node.args.values():
            if isinstance(arg, list):
                for a in arg:
                    if isinstance(a, exp.Expression):
                        ci = visit(a, None)  # parent set later
                        if ci is not None:
                            child_indices.append(ci)
            elif isinstance(arg, exp.Expression):
                ci = visit(arg, None)
                if ci is not None:
                    child_indices.append(ci)
        # Now emit this node
        my_idx = len(labels)
        labels.append(_node_label(node))
        parents.append(parent_idx)
        # Patch children to point to me
        for c in child_indices:
            parents[c] = my_idx
        return my_idx

    try:
        visit(tree, -1)
    except RecursionError:
        return None, None

    # Convert labels to AST_VOCAB ids
    label_ids = [AST_VOCAB.get(l, AST_VOCAB["<UNK>"]) for l in labels]
    return label_ids, parents


# ============================================================
# Surface view: CodeBERT BPE tokenizer (lazy load)
# ============================================================
_BPE_TOKENIZER = None


def get_bpe_tokenizer():
    global _BPE_TOKENIZER
    if _BPE_TOKENIZER is None:
        from transformers import AutoTokenizer
        _BPE_TOKENIZER = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    return _BPE_TOKENIZER


# ============================================================
# AST wrapper: try multiple mini-templates, take the first that parses
# ============================================================
# These are intentionally short/simple so the resulting AST is dominated
# by what the user_input contributes structurally.
AST_WRAPPERS = [
    "SELECT * FROM t WHERE id = {slot}",        # numeric / generic
    "SELECT * FROM t WHERE name = '{slot}'",    # string-quoted
    "SELECT {slot} FROM t",                     # identifier
    "SELECT * FROM t WHERE {slot}",             # SQL fragment
]


def serialize_ast_payload(user_input: str) -> tuple[list[str] | None, str | None]:
    """Try wrapping user_input in each mini-template; return the first
    serialization that succeeds, plus which wrapper worked."""
    for wrap in AST_WRAPPERS:
        wrapped = wrap.replace("{slot}", user_input)
        toks = serialize_ast(wrapped)
        if toks is not None:
            return toks, wrap
    return None, None


# ============================================================
# Payload-level three-view preprocessor
# ============================================================
class SamplePreprocessor:
    """Convert one user_input string into three-view token id arrays.

    Surface: BPE on raw user_input (max 257 tokens — payloads are short).
    Lexical: libinjection on raw user_input (libinjection's intended use).
    AST:     parse user_input wrapped in a canonical mini-template; try
             multiple wrappers and use the first that parses.
    """

    SURFACE_MAX = 257     # user_input rarely exceeds 200 chars
    LEX_MAX = 129
    AST_MAX = 257
    CHAR_MAX = 257        # char-level baseline encoding length

    def __init__(self):
        self.bpe = get_bpe_tokenizer()
        self.surface_pad = self.bpe.pad_token_id
        self.surface_cls = self.bpe.cls_token_id
        self.surface_sep = self.bpe.sep_token_id

    def __call__(self, user_input: str) -> dict:
        # ---- Surface ----
        surface_enc = self.bpe(
            user_input, add_special_tokens=True,
            max_length=self.SURFACE_MAX, truncation=True,
            return_attention_mask=True,
        )
        surface_ids = surface_enc["input_ids"]
        surface_mask = surface_enc["attention_mask"]

        # ---- Lexical ----
        lex_tokens = lexicalize(user_input)[: self.LEX_MAX - 1]
        lex_ids = [LEX_VOCAB["<CLS>"]] + [LEX_VOCAB.get(t, LEX_VOCAB["<UNK>"]) for t in lex_tokens]
        lex_mask = [1] * len(lex_ids)

        # ---- AST (wrap → parse) ----
        ast_tokens, wrapper_used = serialize_ast_payload(user_input)
        if ast_tokens is None:
            ast_ids = [AST_VOCAB["<CLS>"]]
            ast_mask = [1]
            ast_valid = 0
            ast_node_ids: list[int] = []
            ast_parent: list[int] = []
        else:
            ast_tokens = ast_tokens[: self.AST_MAX - 1]
            ast_ids = [AST_VOCAB["<CLS>"]] + [AST_VOCAB.get(t, AST_VOCAB["<UNK>"]) for t in ast_tokens]
            ast_mask = [1] * len(ast_ids)
            ast_valid = 1
            # Tree structure for Tree-LSTM (use the same wrapper)
            wrapped = (wrapper_used or AST_WRAPPERS[0]).replace("{slot}", user_input)
            node_ids, parents = serialize_ast_tree_struct(wrapped)
            if node_ids is None:
                ast_node_ids, ast_parent = [], []
            else:
                ast_node_ids = node_ids[: self.AST_MAX - 1]
                ast_parent = parents[: self.AST_MAX - 1]
                # Make sure orphaned parent indices (out of truncated range)
                # are clamped to -1 so the tree stays well-formed
                for i, p in enumerate(ast_parent):
                    if p >= len(ast_parent):
                        ast_parent[i] = -1

        # ---- Char-level (for CharCNN baseline) ----
        # Each utf-8 byte → id in [1, 256], 0 reserved for PAD.
        raw = user_input.encode("utf-8")[: self.CHAR_MAX]
        char_ids = [b + 1 for b in raw]  # shifts 0-255 → 1-256
        if not char_ids:
            char_ids = [1]  # at least one token to keep collate happy
        char_mask = [1] * len(char_ids)

        return {
            "surface_ids": surface_ids,
            "surface_mask": surface_mask,
            "lex_ids": lex_ids,
            "lex_mask": lex_mask,
            "ast_ids": ast_ids,
            "ast_mask": ast_mask,
            "ast_valid": ast_valid,
            "ast_node_ids": ast_node_ids,
            "ast_parent": ast_parent,
            "char_ids": char_ids,
            "char_mask": char_mask,
        }


if __name__ == "__main__":
    pre = SamplePreprocessor()
    samples = [
        "SELECT * FROM users",
        "1 OR ASCII(SUBSTRING(version(),1,1))=53",
        "INSERT INTO tab (col1) VALUES ('foo')",
        "1 UNION/*!50000SELECT*/ NULL,version()",
    ]
    for s in samples:
        out = pre(s)
        print(f"\nSQL: {s}")
        print(f"  surface_ids ({len(out['surface_ids'])}): {out['surface_ids'][:10]}...")
        print(f"  lex_ids ({len(out['lex_ids'])}): {out['lex_ids']}")
        print(f"  ast_ids ({len(out['ast_ids'])}): {out['ast_ids'][:15]}...")
        print(f"  ast_valid: {out['ast_valid']}")
    print(f"\nLEX vocab size:  {len(LEX_VOCAB)}")
    print(f"AST vocab size:  {len(AST_VOCAB)}")
    print(f"BPE vocab size:  {pre.bpe.vocab_size}")
