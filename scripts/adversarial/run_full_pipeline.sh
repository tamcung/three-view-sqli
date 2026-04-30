#!/usr/bin/env bash
# Chapter 4 — Full adversarial training pipeline (WAF-A-MoLE + FreeLB)
#
# Stage 1 (~10-20 min): pilot threat models on the seed checkpoint
#   - WAF-A-MoLE (black-box, hill-climb, 3 OOD mutators)
#   - HotFlip    (white-box, byte-level gradient — diagnostic only)
#   - LLM        (black-box, free-form, optional, needs ANTHROPIC_API_KEY)
#
# Stage 2 (~30-45 min/round * 3 rounds): co-evolutionary fine-tuning
#   - Outer loop: alternate { attack with WAF-A-MoLE [+LLM] } and { fine-tune }
#   - Inner loop: each fine-tuning epoch uses FreeLB (embedding-space PGD)
#
# Stage 3 (~5 min): final robustness evaluation table
#
# Recommended hardware: A6000 / 4090 24 GB.
# Adjust SEED_CKPT / OUT_DIR if your paths differ.

set -euo pipefail
cd "$(dirname "$0")/../.."

SEED_CKPT="results/tri_view_stage_aug/best_checkpoint.pt"
OUT_DIR="results/cotrain_v1"
ADV_DIR="data/adversarial"

mkdir -p "$ADV_DIR"

echo "================================================================"
echo "  Stage 1: pilot threat models against seed checkpoint"
echo "================================================================"

# 1a. WAF-A-MoLE (Demetrio et al. 2020) — primary search-based attacker
python -m scripts.adversarial.wafamole_attacker \
    --checkpoint "$SEED_CKPT" \
    --output "$ADV_DIR/wafamole_pilot.jsonl" \
    --n-seeds 400 --max-rounds 50 --round-size 24 \
    --seed-split data/splits/test.jsonl \
    --limit-seeds-already-broken --seed 42

# 1b. HotFlip (Ebrahimi'18 inspired) — white-box diagnostic
python -m scripts.adversarial.hotflip_attacker \
    --checkpoint "$SEED_CKPT" \
    --output "$ADV_DIR/hotflip_pilot.jsonl" \
    --n-seeds 200 --n-flips 18 --top-k-per-iter 48 \
    --seed-split data/splits/test.jsonl \
    --limit-seeds-already-broken --seed 42

# 1c. LLM-based (Claude) — optional, needs ANTHROPIC_API_KEY
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    python -m scripts.adversarial.llm_attacker \
        --checkpoint "$SEED_CKPT" \
        --output "$ADV_DIR/llm_pilot.jsonl" \
        --n-seeds 80 --variants-per-seed 6 --rounds 2 \
        --provider anthropic --model claude-sonnet-4-5 \
        --seed-split data/splits/test.jsonl \
        --limit-seeds-already-broken --keep-only-best-per-seed --seed 42
else
    echo "  (Skipping LLM pilot — set ANTHROPIC_API_KEY to enable)"
fi

echo
echo "================================================================"
echo "  Stage 2: co-evolutionary fine-tuning (3 rounds)"
echo "    Outer: WAF-A-MoLE [+ LLM] adversarial sample collection"
echo "    Inner: FreeLB embedding-space PGD during each epoch"
echo "================================================================"

ATTACKERS=("wafamole")
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    ATTACKERS+=("llm")
fi

python -m scripts.adversarial.co_train \
    --seed-checkpoint "$SEED_CKPT" \
    --output "$OUT_DIR/" \
    --rounds 3 \
    --attackers "${ATTACKERS[@]}" \
    --seeds-per-round 400 \
    --cap-adv-per-attacker 2000 \
    --epochs-per-round 2 \
    --lr 1e-4 \
    --holdout-adv-n 200 \
    --attack-seed-split data/splits/train.jsonl \
    --holdout-attack-seed-split data/splits/test.jsonl \
    --freelb \
    --freelb-steps 3 \
    --freelb-init-norm 0.05 \
    --freelb-step-size 0.01 \
    --freelb-max-norm 0.2 \
    --freelb-adv-weight 1.0 \
    --seed 42

echo
echo "================================================================"
echo "  Stage 3: final robustness evaluation"
echo "================================================================"

ADV_LIST=("$OUT_DIR/holdout_adv/search_holdout.jsonl"
          "$ADV_DIR/wafamole_pilot.jsonl"
          "$ADV_DIR/hotflip_pilot.jsonl")
if [ -f "$ADV_DIR/llm_pilot.jsonl" ]; then
    ADV_LIST+=("$ADV_DIR/llm_pilot.jsonl")
fi

# Round 0 (seed)
python -m scripts.adversarial.eval_robustness \
    --checkpoint "$SEED_CKPT" \
    --output "$OUT_DIR/round_0/robustness.json" \
    --adv-jsonls "${ADV_LIST[@]}" \
    --run-tamper-eval --fresh-attack-n 100

# Each cotrain round
for r in 1 2 3; do
    if [ -f "$OUT_DIR/round_${r}/best_checkpoint.pt" ]; then
        python -m scripts.adversarial.eval_robustness \
            --checkpoint "$OUT_DIR/round_${r}/best_checkpoint.pt" \
            --output "$OUT_DIR/round_${r}/robustness.json" \
            --adv-jsonls "${ADV_LIST[@]}" \
            --run-tamper-eval --fresh-attack-n 100
    fi
done

echo
echo "================================================================"
echo "  DONE. Per-round robustness JSONs:"
echo "================================================================"
for r in 0 1 2 3; do
    f="$OUT_DIR/round_${r}/robustness.json"
    [ -f "$f" ] && echo "  $f"
done
