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
# Combined preprocessor
# ============================================================
class SamplePreprocessor:
    """Convert one SQL string into three-view token id arrays.

    Surface uses CodeBERT BPE (50k vocab), lengths capped at 513 (1 CLS + 512).
    Lexical uses libinjection type codes (24 vocab), capped at 129 (1 CLS + 128).
    AST uses bracketed pre-order with sqlglot (~85 vocab), capped at 257 (1 CLS + 256).
    """

    SURFACE_MAX = 513
    LEX_MAX = 129
    AST_MAX = 257

    def __init__(self):
        self.bpe = get_bpe_tokenizer()
        # CodeBERT (RoBERTa) tokenizer pads with id 1 by default; we'll align our
        # PAD to its token (it has cls_token_id=0, pad_token_id=1).
        self.surface_pad = self.bpe.pad_token_id
        self.surface_cls = self.bpe.cls_token_id
        self.surface_sep = self.bpe.sep_token_id

    def __call__(self, sql: str) -> dict:
        # ---- Surface ----
        surface_enc = self.bpe(
            sql, add_special_tokens=True,
            max_length=self.SURFACE_MAX, truncation=True,
            return_attention_mask=True,
        )
        surface_ids = surface_enc["input_ids"]
        surface_mask = surface_enc["attention_mask"]

        # ---- Lexical ----
        lex_tokens = lexicalize(sql)[: self.LEX_MAX - 1]
        lex_ids = [LEX_VOCAB["<CLS>"]] + [LEX_VOCAB.get(t, LEX_VOCAB["<UNK>"]) for t in lex_tokens]
        lex_mask = [1] * len(lex_ids)

        # ---- AST ----
        ast_tokens = serialize_ast(sql)
        if ast_tokens is None:
            ast_ids = [AST_VOCAB["<CLS>"]]
            ast_mask = [1]
            ast_valid = 0
        else:
            ast_tokens = ast_tokens[: self.AST_MAX - 1]
            ast_ids = [AST_VOCAB["<CLS>"]] + [AST_VOCAB.get(t, AST_VOCAB["<UNK>"]) for t in ast_tokens]
            ast_mask = [1] * len(ast_ids)
            ast_valid = 1

        return {
            "surface_ids": surface_ids,
            "surface_mask": surface_mask,
            "lex_ids": lex_ids,
            "lex_mask": lex_mask,
            "ast_ids": ast_ids,
            "ast_mask": ast_mask,
            "ast_valid": ast_valid,
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
