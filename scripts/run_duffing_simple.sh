#!/usr/bin/env bash
set -euo pipefail

# Simple baseline: LISTA sparse autoencoder on Duffing oscillator

TARGET_SIZE="${1:-128}"
SPARSITY="${2:-1.0}"
STEPS="${3:-8000}"
BS="${4:-512}"
RUN_DIR="${5:-runs/duffing_lista_simple}"

echo "Running Duffing simple baseline"
echo "  target:     $TARGET_SIZE"
echo "  sparsity:   $SPARSITY"
echo "  steps:      $STEPS"
echo "  batch:      $BS"
echo "  run_dir:    $RUN_DIR"

uv run python tools/train.py \
  --config lista_nonlinear \
  --env duffing \
  --num_steps "$STEPS" \
  --batch_size "$BS" \
  --target_size "$TARGET_SIZE" \
  --sparsity_coeff "$SPARSITY" \
  --log_dir "$RUN_DIR"

uv run python tools/evaluate_support_uniqueness.py \
  --checkpoint "$RUN_DIR/last.pt" \
  --system duffing \
  --output_dir "$RUN_DIR/support_eval"
