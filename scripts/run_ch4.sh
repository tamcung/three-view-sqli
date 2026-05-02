#!/usr/bin/env bash
# Chapter 4 end-to-end pipeline (designed for RunPod execution).
#
# Stages:
#   stage 1  generate adversarial samples from the §3 base model (round 0)
#   stage 2  train 4 strategies (clean / aug / freelb / combined)
#   stage 3  evaluate each strategy on clean test + adversarial test
#   stage 4  iterative loop (R-1 more rounds: re-attack -> re-train)
#
# Run all stages:                bash scripts/run_ch4.sh all
# Run a single stage:            bash scripts/run_ch4.sh stage1
#                                bash scripts/run_ch4.sh stage1b   # eval adv set
#                                bash scripts/run_ch4.sh stage2
#                                bash scripts/run_ch4.sh stage3
#                                bash scripts/run_ch4.sh stage4
# Run ε ablation only:           bash scripts/run_ch4.sh ablate-eps
#
# Two adversarial sets are kept disjoint by construction to avoid evaluation
# leakage (aug/combined strategies train on round0.jsonl):
#   round0.jsonl       — generated from TRAIN seeds, used for training augmentation
#   round0_eval.jsonl  — generated from TEST seeds, used only for evaluation
#
# Pre-requisites on RunPod:
#   - The §3 best checkpoint must already exist:
#       results_kaggle/three_view_d128_L2/best_checkpoint.pt
#   - WAFamole installed:
#       git clone https://github.com/AvalZ/WAF-A-MoLE third_party/wafamole
#       pip install -e third_party/wafamole sqlparse networkx
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# -----------------------------------------------------------
# Configuration (edit if needed)
# -----------------------------------------------------------
BASE_CKPT="results_kaggle/three_view_d128_L2"
TRAIN_JSONL="data/external/kaggle_sqli/jsonl/train.jsonl"
VAL_JSONL="data/external/kaggle_sqli/jsonl/val.jsonl"
TEST_JSONL="data/external/kaggle_sqli/jsonl/test.jsonl"

CONFIG="configs/ch4_combined.yaml"

ADV_DIR="results/ch4/adv_samples"
RESULTS_DIR="results_adv"
mkdir -p "$ADV_DIR" "$RESULTS_DIR"

ROUNDS=3
ATTACK_MAX_SAMPLES=5000   # cap adversarial generation to this many seed payloads

ATTACK_ARGS=(
  --beam 5
  --max-steps 10
  --query-budget 200
  --max-samples "$ATTACK_MAX_SAMPLES"
)


# -----------------------------------------------------------
# Stage 1: generate training-side adversarial samples (round 0) from §3 model
# Seeds: train.jsonl  →  used by aug / combined as augmentation data.
# -----------------------------------------------------------
stage1() {
  echo "=== Stage 1: generate round-0 adversarial samples (TRAIN seeds) ==="
  python -u src/adv_generator.py \
    --ckpt-dir "$BASE_CKPT" \
    --input "$TRAIN_JSONL" \
    --output "$ADV_DIR/round0.jsonl" \
    "${ATTACK_ARGS[@]}"
}


# -----------------------------------------------------------
# Stage 1b: generate evaluation-only adversarial samples from §3 model
# Seeds: test.jsonl  →  used ONLY by stage 3 evaluation. Disjoint from
# round0.jsonl by construction (different seed pool), so aug/combined
# never see these during training. This is a one-shot fixed eval set
# that all four strategies (and stage-4 round-r models) are scored on.
# -----------------------------------------------------------
stage1b() {
  echo "=== Stage 1b: generate evaluation adversarial samples (TEST seeds) ==="
  python -u src/adv_generator.py \
    --ckpt-dir "$BASE_CKPT" \
    --input "$TEST_JSONL" \
    --output "$ADV_DIR/round0_eval.jsonl" \
    "${ATTACK_ARGS[@]}"
}


# -----------------------------------------------------------
# Stage 2: train 4 strategies
# -----------------------------------------------------------
train_one() {
  local strategy="$1"
  local out_dir="$RESULTS_DIR/$strategy"
  echo "=== Stage 2: train strategy=$strategy → $out_dir ==="
  local extra=()
  if [[ "$strategy" == "aug" || "$strategy" == "combined" ]]; then
    extra+=(--adv-jsonl "$ADV_DIR/round0.jsonl")
  fi
  python -u train_adv.py \
    --config "$CONFIG" \
    --output "$out_dir" \
    --strategy "$strategy" \
    --train-jsonl "$TRAIN_JSONL" \
    --val-jsonl "$VAL_JSONL" \
    --test-jsonl "$TEST_JSONL" \
    --init-ckpt "$BASE_CKPT/best_checkpoint.pt" \
    "${extra[@]}"
}

stage2() {
  for s in clean aug freelb combined; do
    train_one "$s"
  done
}


# -----------------------------------------------------------
# Stage 3: evaluate each trained strategy
# -----------------------------------------------------------
eval_one() {
  local out_dir="$1"
  local adv_jsonl="$2"
  python -u eval_adv.py \
    --ckpt-dir "$out_dir" \
    --clean-jsonl "$TEST_JSONL" \
    --adv-jsonl "$adv_jsonl" \
    --output "$out_dir/eval_summary.json"
}

stage3() {
  echo "=== Stage 3: evaluate 4 strategies on clean + adv test ==="
  if [[ ! -f "$ADV_DIR/round0_eval.jsonl" ]]; then
    echo "ERROR: $ADV_DIR/round0_eval.jsonl missing — run stage1b first" >&2
    exit 1
  fi
  for s in clean aug freelb combined; do
    eval_one "$RESULTS_DIR/$s" "$ADV_DIR/round0_eval.jsonl"
  done
}


# -----------------------------------------------------------
# Stage 4: iterative loop (rounds 1..R-1)
# -----------------------------------------------------------
stage4() {
  local current="$RESULTS_DIR/combined"
  echo "=== Stage 4: iterative loop, $((ROUNDS-1)) more rounds ==="
  for r in $(seq 1 $((ROUNDS-1))); do
    echo "--- Round $r ---"
    # 4.1 attack current model with TRAIN seeds → training-side adv set
    python -u src/adv_generator.py \
      --ckpt-dir "$current" \
      --input "$TRAIN_JSONL" \
      --output "$ADV_DIR/round${r}.jsonl" \
      "${ATTACK_ARGS[@]}"
    # 4.2 attack current model with TEST seeds → evaluation-only adv set
    #     (used both for ASR_r curve and for the round-r model's adv F1)
    python -u src/adv_generator.py \
      --ckpt-dir "$current" \
      --input "$TEST_JSONL" \
      --output "$ADV_DIR/round${r}_eval.jsonl" \
      "${ATTACK_ARGS[@]}"
    # 4.3 train next-round model from current ckpt using train-side adv
    local next_dir="$RESULTS_DIR/combined_round${r}"
    python -u train_adv.py \
      --config "$CONFIG" \
      --output "$next_dir" \
      --strategy combined \
      --adv-jsonl "$ADV_DIR/round${r}.jsonl" \
      --train-jsonl "$TRAIN_JSONL" \
      --val-jsonl "$VAL_JSONL" \
      --test-jsonl "$TEST_JSONL" \
      --init-ckpt "$current/best_checkpoint.pt"
    # 4.4 evaluate the new round-r model on its own held-out eval adv set
    eval_one "$next_dir" "$ADV_DIR/round${r}_eval.jsonl"
    current="$next_dir"
  done
}


# -----------------------------------------------------------
# Optional: ε ablation (overrides freelb.epsilon and trains combined)
# -----------------------------------------------------------
ablate_eps() {
  echo "=== ε ablation: 0.05 / 0.10 / 0.20 / 0.50 ==="
  for eps in 0.05 0.10 0.20 0.50; do
    local cfg_file="configs/ch4_eps_${eps/./p}.yaml"
    python -c "
import yaml
c = yaml.safe_load(open('$CONFIG'))
c.setdefault('freelb', {})['epsilon'] = float('$eps')
yaml.dump(c, open('$cfg_file', 'w'), sort_keys=False)
print('wrote', '$cfg_file')
"
    local out_dir="$RESULTS_DIR/eps_${eps/./p}"
    python -u train_adv.py \
      --config "$cfg_file" \
      --output "$out_dir" \
      --strategy combined \
      --adv-jsonl "$ADV_DIR/round0.jsonl" \
      --train-jsonl "$TRAIN_JSONL" \
      --val-jsonl "$VAL_JSONL" \
      --test-jsonl "$TEST_JSONL" \
      --init-ckpt "$BASE_CKPT/best_checkpoint.pt"
    eval_one "$out_dir" "$ADV_DIR/round0_eval.jsonl"
  done
}


# -----------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------
case "${1:-all}" in
  stage1) stage1 ;;
  stage1b) stage1b ;;
  stage2) stage2 ;;
  stage3) stage3 ;;
  stage4) stage4 ;;
  ablate-eps) ablate_eps ;;
  all) stage1; stage1b; stage2; stage3; stage4 ;;
  *) echo "Usage: $0 {stage1|stage1b|stage2|stage3|stage4|ablate-eps|all}"; exit 1 ;;
esac

echo "=== Done ==="
