#!/usr/bin/env python3
"""De-obfuscation pre-processing for SQL injection inputs.

Implements common WAF-bypass un-doings, applied iteratively until fixed point:

  1. HTML entity decode  (&#x27; → ' , &#39; → ' , &amp; → &)
  2. URL decode           (%27 → ' , %20 → space)
  3. Unicode escape       (%u0027 → ')
  4. Hex byte escape      (\\x27 → ')
  5. Base64 decode        (whole-string heuristic)
  6. Comment removal      (/* ... */)
  7. Whitespace normalize (\\t \\xa0 \\v \\f \\r \\n → space; collapse)
  8. Operator alternatives (`||` → OR, `&&` → AND)

Used to bring tamper-mutated payloads back into the model's training
distribution before classification.
"""
from __future__ import annotations
import base64
import html
import re
from urllib.parse import unquote_plus


_HEX_ESCAPE_RE = re.compile(r"\\x([0-9a-fA-F]{2})")
_UNI_ESCAPE_RE = re.compile(r"%u([0-9a-fA-F]{4})")
_BSLASH_U_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_BASE64_SHAPE_RE = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
_WS_RE = re.compile(r" +")
_OR_RE = re.compile(r"\|\|")
_AND_RE = re.compile(r"(?<![&])&&(?![&])")


def _maybe_base64_decode(s: str) -> str:
    """If `s` is shaped like base64 and decodes to text containing SQLi
    fingerprints, return the decoded string; otherwise return `s` unchanged."""
    stripped = s.strip()
    if not (8 < len(stripped) < 5000):
        return s
    if not _BASE64_SHAPE_RE.match(stripped):
        return s
    try:
        raw = base64.b64decode(stripped, validate=True)
    except Exception:
        return s
    try:
        decoded = raw.decode("utf-8", errors="replace")
    except Exception:
        return s
    # Only adopt the decoded form if it looks SQLi-flavored
    needles = ("select", "union", "or 1=1", "and 1=", "--", "' or", "/*")
    if any(n in decoded.lower() for n in needles):
        return decoded
    return s


def deobfuscate(s: str, max_iter: int = 5) -> str:
    """Iteratively un-encode common WAF-bypass tricks. Idempotent at fixed point."""
    if not s:
        return s

    prev = None
    iterations = 0
    while s != prev and iterations < max_iter:
        prev = s
        # HTML entity decode (handles hexentities / decentities / &amp;)
        try:
            s = html.unescape(s)
        except Exception:
            pass
        # URL decode (handles charencode / chardoubleencode after two passes)
        try:
            s = unquote_plus(s)
        except Exception:
            pass
        # %uXXXX → unicode char  (charunicodeencode)
        s = _UNI_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), s)
        # \uXXXX → unicode char  (charunicodeescape)
        s = _BSLASH_U_RE.sub(lambda m: chr(int(m.group(1), 16)), s)
        # \xHH → byte
        s = _HEX_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), s)
        iterations += 1

    # Base64 (single pass — only applies to whole-string base64)
    s = _maybe_base64_decode(s)

    # Comment removal — replace inline comments with single space
    s = _BLOCK_COMMENT_RE.sub(" ", s)

    # Normalize whitespace variants → single space
    for ws in ("\t", "\xa0", "\v", "\f", "\r", "\n"):
        s = s.replace(ws, " ")
    s = _WS_RE.sub(" ", s)

    # Operator equivalents
    s = _OR_RE.sub(" OR ", s)
    s = _AND_RE.sub(" AND ", s)

    return s.strip()


if __name__ == "__main__":
    samples = [
        "%27%20OR%201%3D1%20--",           # url encoded
        "%2527%2520OR%25201%253D1%2520--", # double url encoded
        "&#x27; OR 1=1 --",                # hex html entity
        "&#39; OR 1=1 --",                 # decimal html entity
        "%u0027 OR %u0031%u003D%u0031",    # %uXXXX
        "JyBPUiAxPTEgLS0=",                 # base64 of "' OR 1=1 --"
        "1' /*evil*/ OR 1=1 --",           # block comment
        "1\xa0OR\t1=1\v--",                # weird whitespace
        "1' || 1=1 --",                    # || operator
        "AND 1=1",                          # negative control
    ]
    for s in samples:
        print(f"  in : {s!r}")
        print(f"  out: {deobfuscate(s)!r}")
        print()
