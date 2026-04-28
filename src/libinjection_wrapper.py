#!/usr/bin/env python3
"""
ctypes wrapper around libinjection (Windows DLL or Linux shared object).

The native library is compiled by `scripts/build_libinjection.{sh,bat}` from
`external/libinjection/src/`. The wrapper auto-detects platform and loads
the correct file.

Exposes:
  tokenize(sql, flags) -> list[(type_char, text)]
  is_sqli(sql) -> (bool, fingerprint_str)
  get_version() -> str
"""
from __future__ import annotations
import ctypes
import platform
from ctypes import (
    POINTER, Structure, c_char, c_char_p, c_int, c_size_t, c_void_p,
    byref, cdll
)
from pathlib import Path

# Locate the compiled native library
_LIB_DIR = Path(__file__).resolve().parent.parent / "lib"
_SYS = platform.system()
if _SYS == "Windows":
    _LIB_NAME = "libinjection.dll"
elif _SYS == "Darwin":
    _LIB_NAME = "libinjection.dylib"
else:  # Linux and BSDs
    _LIB_NAME = "libinjection.so"

_LIB_PATH = _LIB_DIR / _LIB_NAME
if not _LIB_PATH.exists():
    raise FileNotFoundError(
        f"libinjection native library not found at {_LIB_PATH}.\n"
        f"On Linux: run `bash scripts/build_libinjection.sh`\n"
        f"On Windows: run `scripts\\build_libinjection.bat` (requires MSVC Build Tools)"
    )
_lib = cdll.LoadLibrary(str(_LIB_PATH))


class StokenT(Structure):
    _fields_ = [
        ("pos", c_size_t),
        ("len", c_size_t),
        ("count", c_int),
        ("type", c_char),
        ("str_open", c_char),
        ("str_close", c_char),
        ("val", c_char * 32),
    ]


class SqliState(Structure):
    _fields_ = [
        ("s", c_char_p),
        ("slen", c_size_t),
        ("lookup", c_void_p),
        ("userdata", c_void_p),
        ("flags", c_int),
        ("pos", c_size_t),
        ("tokenvec", StokenT * 8),
        ("current", POINTER(StokenT)),
        ("fingerprint", c_char * 8),
        ("reason", c_int),
        ("stats_comment_ddw", c_int),
        ("stats_comment_ddx", c_int),
        ("stats_comment_c", c_int),
        ("stats_comment_hash", c_int),
        ("stats_folds", c_int),
        ("stats_tokens", c_int),
    ]


_lib.libinjection_sqli_init.argtypes = [POINTER(SqliState), c_char_p, c_size_t, c_int]
_lib.libinjection_sqli_init.restype = None
_lib.libinjection_sqli_tokenize.argtypes = [POINTER(SqliState)]
_lib.libinjection_sqli_tokenize.restype = c_int
_lib.libinjection_is_sqli.argtypes = [POINTER(SqliState)]
_lib.libinjection_is_sqli.restype = c_int
_lib.libinjection_version.argtypes = []
_lib.libinjection_version.restype = c_char_p

FLAG_NONE = 0
FLAG_QUOTE_NONE = 1
FLAG_QUOTE_SINGLE = 2
FLAG_QUOTE_DOUBLE = 4
FLAG_SQL_ANSI = 8
FLAG_SQL_MYSQL = 16


def get_version() -> str:
    return _lib.libinjection_version().decode("ascii")


def tokenize(sql: str, flags: int = FLAG_QUOTE_NONE | FLAG_SQL_MYSQL) -> list[tuple[str, str]]:
    """Return list of (type_char, text) per token from libinjection's tokenizer."""
    state = SqliState()
    sql_b = sql.encode("utf-8", errors="replace")
    _lib.libinjection_sqli_init(byref(state), sql_b, len(sql_b), flags)

    tokens = []
    safety_limit = 4096
    while safety_limit > 0:
        safety_limit -= 1
        ret = _lib.libinjection_sqli_tokenize(byref(state))
        if ret == 0:
            break
        cur = state.current
        if not cur:
            break
        tok = cur.contents
        type_char = tok.type.decode("latin-1") if tok.type else ""
        try:
            text = tok.val.decode("utf-8", errors="replace").rstrip("\x00")
        except Exception:
            text = ""
        tokens.append((type_char, text))
    return tokens


def is_sqli(sql: str) -> tuple[bool, str]:
    """Run libinjection's full SQLi check; returns (is_sqli, fingerprint)."""
    state = SqliState()
    sql_b = sql.encode("utf-8", errors="replace")
    _lib.libinjection_sqli_init(byref(state), sql_b, len(sql_b), FLAG_NONE)
    result = _lib.libinjection_is_sqli(byref(state))
    fp = bytes(state.fingerprint).rstrip(b"\x00").decode("ascii", errors="replace")
    return bool(result), fp


if __name__ == "__main__":
    print(f"libinjection version: {get_version()}")
    samples = ["SELECT * FROM users", "' OR 1=1--", "1 UNION SELECT NULL"]
    for s in samples:
        toks = tokenize(s)
        print(f"  {s!r:40s} → {''.join(t[0] for t in toks)}")
