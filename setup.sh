#!/bin/bash
# One-shot RunPod / Linux setup.
#
# Idempotent: safe to re-run.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "============================================================"
echo "1. Compiling libinjection native library"
echo "============================================================"
bash scripts/build_libinjection.sh

echo ""
echo "============================================================"
echo "2. Installing Python dependencies"
echo "============================================================"
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "============================================================"
echo "3. Pre-downloading CodeBERT tokenizer (cached for offline use)"
echo "============================================================"
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('microsoft/codebert-base')"

echo ""
echo "============================================================"
echo "4. Sanity checks"
echo "============================================================"
python -c "
import torch
print('  torch version       :', torch.__version__)
print('  CUDA available      :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU                  :', torch.cuda.get_device_name(0))
    print('  CUDA version        :', torch.version.cuda)
    print('  BF16 supported      :', torch.cuda.is_bf16_supported())
"
python -c "
import sys; sys.path.insert(0, 'src')
from libinjection_wrapper import get_version, tokenize
print('  libinjection version:', get_version())
print('  test tokenize       :', tokenize(\"' OR 1=1\"))
"
python -c "
import sys; sys.path.insert(0, 'src')
from preprocessing import SamplePreprocessor
p = SamplePreprocessor()
out = p('SELECT * FROM users WHERE id = 1')
print('  preprocessor surface_ids count:', len(out['surface_ids']))
print('  preprocessor lex_ids:           ', out['lex_ids'])
print('  preprocessor ast_ids count:     ', len(out['ast_ids']))
"

echo ""
echo "============================================================"
echo "Setup complete. Next steps:"
echo "  1. Place WAF-A-MoLE data files in data/wafamole/"
echo "       attacks.sql.statements.jsonl"
echo "       sane.sql.statements.jsonl"
echo "  2. python train.py --config configs/medium.yaml --output results/run_001"
echo "============================================================"
