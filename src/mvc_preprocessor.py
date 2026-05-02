#!/usr/bin/env python3
"""MVC-BiCNN preprocessor — faithful reimplementation of Kakisim 2024 §3.1.

Generates three views per payload using sqlparse-based tokenization and a
21-class semantic SQL tag dictionary (Table 1 of the original paper):

  - tokenized: SQL term sequence after sqlparse tokenization, with
               numbers / hex / low-frequency identifiers filtered out
  - converted: sequence of 21-class semantic tags (DDL / DML / DCL / Keyword
               / Integer / Hexadecimal / Punctuation / Identifier /
               Identifierlist / Quotes / Comparison / Wildcard / Builtin /
               Error / Where / Parenthesis / Function / CTE / Order /
               Operator / Escape)
  - enriched:  CODDLE-style interleaved [token, tag, token, tag, ...] with
               numbers / punctuation / parentheses removed (per §3.1.3)

All three views are truncated/padded to length 50 (per §4.1).

This preprocessor reuses the existing dataset / collate fields:
  surface_ids → MVC tokenized
  lex_ids     → MVC converted
  char_ids    → MVC enriched
so the training loop does not need MVC-specific code paths.
"""
from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path

import sqlparse
from sqlparse.tokens import Token


# ============================================================
# 21 SQL semantic tags (Kakisim 2024 Table 1)
# ============================================================
TAGS = [
    "DDL", "DML", "DCL",
    "Keyword",
    "Integer", "Hexadecimal",
    "Escape", "Punctuation",
    "Identifier", "Identifierlist",
    "Quotes", "Comparison",
    "Wildcard", "Builtin",
    "Error", "Where",
    "Parenthesis", "Function",
    "CTE", "Order", "Operator",
]
assert len(TAGS) == 21

# Filtering: tokens with these tags are dropped from the respective views
NOISE_TAGS_TOK = {"Integer", "Hexadecimal"}
NOISE_TAGS_ENR = {"Integer", "Hexadecimal", "Punctuation", "Parenthesis"}


# ============================================================
# Lexical resources for tag classification
# ============================================================
DDL_KW = {"create", "alter", "drop", "truncate", "rename"}
DML_KW = {
    "select", "insert", "update", "delete", "replace", "merge",
    "from", "into", "values", "set", "returning",
}
DCL_KW = {"grant", "revoke", "deny"}
CTE_KW = {"with"}
WHERE_KW = {"where"}
ORDER_KW = {"order", "by", "asc", "desc", "limit", "offset", "having"}
GENERAL_KW = {
    "or", "and", "xor", "not", "in", "between", "like", "rlike", "regexp",
    "is", "null", "exists", "case", "when", "then", "else", "end", "as",
    "join", "inner", "outer", "left", "right", "cross", "full", "natural",
    "on", "using", "all", "any", "some", "distinct", "union", "intersect",
    "except", "group", "exec", "execute", "begin", "declare", "rollback",
    "commit", "if", "while", "for", "loop", "return", "true", "false",
    "lateral",
}
BUILTIN_KW = {
    "date", "datetime", "time", "timestamp", "year",
    "varchar", "char", "text", "blob", "binary", "varbinary",
    "int", "integer", "bigint", "smallint", "tinyint",
    "float", "double", "decimal", "numeric", "real",
    "bool", "boolean", "bit", "money",
    "unsigned", "signed", "nvarchar", "nchar",
}
FUNCTION_KW = {
    "sleep", "benchmark", "substr", "substring", "concat", "concat_ws",
    "char", "chr", "ascii", "ord", "version", "user", "current_user",
    "system_user", "session_user", "database", "schema", "load_file",
    "extractvalue", "updatexml", "if", "ifnull", "isnull", "coalesce",
    "nullif", "cast", "convert", "count", "max", "min", "sum", "avg",
    "length", "char_length", "lower", "upper", "now", "current_timestamp",
    "md5", "sha1", "sha", "rand", "round", "floor", "ceil", "abs",
    "exp", "log", "pow", "power", "sqrt", "mod", "trim", "ltrim", "rtrim",
    "replace", "instr", "locate", "position", "reverse", "repeat",
    "soundex", "format", "hex", "unhex", "bin",
    "extract", "datediff", "date_add", "date_sub", "date_format",
    "pg_sleep", "dbms_pipe.receive_message", "wait", "waitfor",
}

COMPARISON_OPS = {"<", ">", "=", "<=", ">=", "!=", "<>", "<=>"}
WILDCARD_OPS = {"*"}
PARENTHESIS_OPS = {"(", ")"}
PUNCTUATION_OPS = {".", ";", ",", ":"}
QUOTES_OPS = {"'", "`"}
ESCAPE_OPS = {"\\"}
ERROR_OPS = {'"', "$"}
ARITH_OPS = {"+", "-", "/", "%", "@", "#", "&", "|", "^", "~", "!"}


_NUM_RE = re.compile(r"-?\d+(\.\d+)?([eE][+\-]?\d+)?")
_HEX_RE = re.compile(r"0[xX][0-9a-fA-F]+")


def _classify(text: str, ttype) -> str | None:
    """Map a (token text, sqlparse Token type) pair to one of the 21 tags."""
    txt = text.strip()
    if not txt:
        return None

    low = txt.lower()

    # 1. Exact text in keyword groups (these dominate over sqlparse types)
    if low in DDL_KW: return "DDL"
    if low in DML_KW: return "DML"
    if low in DCL_KW: return "DCL"
    if low in CTE_KW: return "CTE"
    if low in ORDER_KW: return "Order"
    if low in WHERE_KW: return "Where"
    if low in BUILTIN_KW: return "Builtin"
    if low in FUNCTION_KW: return "Function"
    if low in GENERAL_KW: return "Keyword"

    # 2. Symbols
    if txt in COMPARISON_OPS: return "Comparison"
    if txt in WILDCARD_OPS: return "Wildcard"
    if txt in PARENTHESIS_OPS: return "Parenthesis"
    if txt in PUNCTUATION_OPS: return "Punctuation"
    if txt in QUOTES_OPS: return "Quotes"
    if txt in ESCAPE_OPS: return "Escape"
    if txt in ERROR_OPS: return "Error"
    if txt in ARITH_OPS: return "Operator"

    # 3. Hex / number by regex (more reliable than sqlparse sometimes)
    if _HEX_RE.fullmatch(txt):
        return "Hexadecimal"
    if _NUM_RE.fullmatch(txt):
        return "Integer"

    # 4. sqlparse ttype hints
    if ttype is not None:
        if ttype in Token.Number.Hexadecimal:
            return "Hexadecimal"
        if ttype in Token.Number:
            return "Integer"
        if ttype in Token.String:
            return "Quotes"
        if ttype in Token.Punctuation:
            return "Punctuation"
        if ttype in Token.Operator:
            return "Operator"

    # 5. Default: treat as identifier
    return "Identifier"


_FALLBACK_RE = re.compile(r"""
    [a-zA-Z_][a-zA-Z0-9_.]*       # identifier / keyword
    | 0[xX][0-9a-fA-F]+            # hex literal
    | -?\d+(?:\.\d+)?              # integer / float
    | '(?:[^'\\]|\\.)*'            # single-quoted string
    | "(?:[^"\\]|\\.)*"            # double-quoted string
    | <=>|<>|!=|<=|>=|\|\||&&      # multi-char operators
    | --|/\*|\*/                   # comment markers
    | \S                           # any other single non-space char
""", re.VERBOSE)


def _tokenize_fallback(sql: str) -> list[tuple[str, str]]:
    """Regex-based fallback when sqlparse fails on pathological payloads."""
    out: list[tuple[str, str]] = []
    for m in _FALLBACK_RE.finditer(sql):
        text = m.group(0)
        # Drop comment-marker artefacts (we never want comments in views)
        if text in ("--", "/*", "*/"):
            continue
        tag = _classify(text, None)
        if tag is not None:
            out.append((text.strip(), tag))
    return out


def tokenize_sql(sql: str) -> list[tuple[str, str]]:
    """sqlparse split + tag classification, with regex fallback.

    Returns a list of (token_text, semantic_tag). Whitespace and comments
    are skipped. Quoted strings are returned as a single ``Quotes`` token
    spanning the entire literal. If sqlparse raises (e.g. on deeply nested
    parentheses common in noisy injection payloads), a regex fallback
    tokenizer is used so the pipeline does not lose samples.
    """
    try:
        parsed = sqlparse.parse(sql)
    except Exception:
        return _tokenize_fallback(sql)
    if not parsed:
        return []

    out: list[tuple[str, str]] = []

    def walk(node):
        for tok in node.tokens:
            if tok.is_group:
                walk(tok)
                continue
            if tok.ttype in (
                Token.Whitespace, Token.Comment,
                Token.Comment.Single, Token.Comment.Multiline,
            ):
                continue
            text = str(tok)
            if not text.strip():
                continue
            tag = _classify(text, tok.ttype)
            if tag is not None:
                out.append((text.strip(), tag))

    try:
        for stmt in parsed:
            walk(stmt)
    except Exception:
        return _tokenize_fallback(sql)
    return out


def build_views(sql: str) -> tuple[list[str], list[str], list[str]]:
    """Generate the three MVC views as string sequences.

    Returns
    -------
    tokenized : list[str]
        Filtered SQL term list (numbers / hex removed; lower-cased).
    converted : list[str]
        21-class tag list (one tag per token, in source order).
    enriched : list[str]
        Interleaved [token1, tag1, token2, tag2, ...] sequence with
        numbers / punctuation / parens dropped from both tokens and tags.
    """
    raw = tokenize_sql(sql)
    tokenized: list[str] = []
    converted: list[str] = []
    enriched: list[str] = []

    for text, tag in raw:
        text_lc = text.lower()

        if tag not in NOISE_TAGS_TOK:
            tokenized.append(text_lc)

        converted.append(tag)

        if tag not in NOISE_TAGS_ENR:
            enriched.append(text_lc)
            enriched.append(tag)

    return tokenized, converted, enriched


# ============================================================
# Vocabulary
# ============================================================
class MVCVocab:
    """Three vocab maps for the three MVC views.

    The converted vocab is fixed at <PAD> + <UNK> + 21 tags = 23 entries.
    The tokenized and enriched vocabs are built from the training set with a
    minimum frequency cutoff (default 2), which approximates Kakisim's
    "low-frequency identifiers are removed" rule.
    """
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(
        self,
        tok_vocab: dict[str, int],
        cnv_vocab: dict[str, int],
        enr_vocab: dict[str, int],
    ):
        self.tok_vocab = tok_vocab
        self.cnv_vocab = cnv_vocab
        self.enr_vocab = enr_vocab
        self.tok_pad = tok_vocab[self.PAD]
        self.cnv_pad = cnv_vocab[self.PAD]
        self.enr_pad = enr_vocab[self.PAD]
        self.tok_unk = tok_vocab[self.UNK]
        self.cnv_unk = cnv_vocab[self.UNK]
        self.enr_unk = enr_vocab[self.UNK]

    def encode_tokenized(self, tokens):
        return [self.tok_vocab.get(t, self.tok_unk) for t in tokens]

    def encode_converted(self, tags):
        return [self.cnv_vocab.get(t, self.cnv_unk) for t in tags]

    def encode_enriched(self, items):
        return [self.enr_vocab.get(t, self.enr_unk) for t in items]

    @classmethod
    def from_file(cls, path):
        data = json.load(open(path, encoding="utf-8"))
        return cls(data["tokenized"], data["converted"], data["enriched"])

    def save(self, path):
        json.dump(
            {
                "tokenized": self.tok_vocab,
                "converted": self.cnv_vocab,
                "enriched": self.enr_vocab,
            },
            open(path, "w", encoding="utf-8"),
            ensure_ascii=False,
            indent=2,
        )

    @property
    def tok_size(self): return len(self.tok_vocab)

    @property
    def cnv_size(self): return len(self.cnv_vocab)

    @property
    def enr_size(self): return len(self.enr_vocab)


def build_vocab_from_jsonl(jsonl_path: str | Path, min_freq: int = 2) -> MVCVocab:
    """Build the three MVC vocabs from a training-set JSONL file.

    The vocab construction follows Kakisim §3.1.1: tokens (after the
    tag-based noise filter) are kept only if their frequency is ≥ `min_freq`.
    Rare identifiers fall back to ``<UNK>`` at runtime.
    """
    tok_counter: Counter = Counter()
    enr_counter: Counter = Counter()

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            payload = rec.get("user_input", "")
            tokens, _, enriched = build_views(payload)
            tok_counter.update(tokens)
            enr_counter.update(enriched)  # mixes tokens and tag strings

    tok_vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for tok, c in sorted(tok_counter.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq:
            tok_vocab[tok] = len(tok_vocab)

    cnv_vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for tag in TAGS:
        cnv_vocab[tag] = len(cnv_vocab)

    enr_vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for item, c in sorted(enr_counter.items(), key=lambda x: (-x[1], x[0])):
        if c >= min_freq:
            enr_vocab[item] = len(enr_vocab)

    return MVCVocab(tok_vocab, cnv_vocab, enr_vocab)


# ============================================================
# Sample preprocessor (drop-in replacement for SamplePreprocessor in MVC mode)
# ============================================================
class MVCSamplePreprocessor:
    """Convert one user_input string into MVC-BiCNN three-view token id arrays.

    Reuses existing field names so the dataset / collate / trainer code does
    not need MVC-specific branches:

        surface_ids / surface_mask  ← MVC tokenized view
        lex_ids     / lex_mask      ← MVC converted view
        char_ids    / char_mask     ← MVC enriched view

    All views are truncated to ``MAX_LEN = 50`` per Kakisim §4.1.
    """

    MAX_LEN = 50

    def __init__(self, vocab: MVCVocab):
        self.vocab = vocab
        # PAD ids exposed for the model side (which sets embedding padding_idx)
        self.surface_pad = vocab.tok_pad
        self.lex_pad = vocab.cnv_pad
        self.char_pad = vocab.enr_pad

    def __call__(self, user_input: str) -> dict:
        tokens, tags, enriched = build_views(user_input)

        tok_ids = self.vocab.encode_tokenized(tokens)[: self.MAX_LEN]
        cnv_ids = self.vocab.encode_converted(tags)[: self.MAX_LEN]
        enr_ids = self.vocab.encode_enriched(enriched)[: self.MAX_LEN]

        # Make sure no view is empty (collate needs at least one token per row)
        if not tok_ids: tok_ids = [self.vocab.tok_pad]
        if not cnv_ids: cnv_ids = [self.vocab.cnv_pad]
        if not enr_ids: enr_ids = [self.vocab.enr_pad]

        return {
            "surface_ids": tok_ids,
            "surface_mask": [1] * len(tok_ids),
            "lex_ids": cnv_ids,
            "lex_mask": [1] * len(cnv_ids),
            "char_ids": enr_ids,
            "char_mask": [1] * len(enr_ids),
            # Inactive slots (kept for collate compatibility)
            "ast_ids": [0],
            "ast_mask": [1],
            "ast_valid": 0,
            "ast_node_ids": [],
            "ast_parent": [],
        }


# ============================================================
# CLI helpers
# ============================================================
def _build_and_save(train_jsonl: str, output_path: str, min_freq: int):
    vocab = build_vocab_from_jsonl(train_jsonl, min_freq=min_freq)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    vocab.save(output_path)
    print(f"Saved MVC vocab to {output_path}")
    print(f"  tokenized: {vocab.tok_size} entries")
    print(f"  converted: {vocab.cnv_size} entries (fixed: PAD + UNK + 21 tags)")
    print(f"  enriched:  {vocab.enr_size} entries")
    return vocab


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build / inspect MVC-BiCNN vocab.")
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output", default="src/mvc_vocab.json")
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--probe", action="store_true",
                   help="After building vocab, print 5 sample decompositions.")
    args = p.parse_args()

    vocab = _build_and_save(args.train_jsonl, args.output, args.min_freq)

    if args.probe:
        pre = MVCSamplePreprocessor(vocab)
        samples = [
            "1' OR 1=1--",
            "admin' or '1'='1",
            "union select username,password from users",
            "1; DROP TABLE users--",
            "1 UNION SELECT NULL,NULL,NULL,version()",
        ]
        for s in samples:
            tokens, tags, enriched = build_views(s)
            print(f"\n>>> {s}")
            print(f"  tokenized ({len(tokens)}): {tokens}")
            print(f"  converted ({len(tags)}): {tags}")
            print(f"  enriched ({len(enriched)}): {enriched}")
