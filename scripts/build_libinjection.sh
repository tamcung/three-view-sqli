#!/bin/bash
# Build libinjection.so on Linux (or .dylib on macOS) using gcc/clang.
# Outputs to lib/libinjection.so

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/external/libinjection/src"
OUT_DIR="$ROOT/lib"
mkdir -p "$OUT_DIR"

UNAME="$(uname -s)"
case "$UNAME" in
  Darwin) OUT="$OUT_DIR/libinjection.dylib"; SO_FLAGS="-dynamiclib";;
  *)      OUT="$OUT_DIR/libinjection.so";    SO_FLAGS="-shared";;
esac

CC="${CC:-gcc}"

echo "Compiling libinjection → $OUT"
"$CC" $SO_FLAGS -fPIC -O2 \
  -I "$SRC" \
  "$SRC/libinjection_sqli.c" \
  -o "$OUT"

echo "Build OK: $OUT"
ls -la "$OUT"
