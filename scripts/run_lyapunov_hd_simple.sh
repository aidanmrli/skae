#!/usr/bin/env bash
set -euo pipefail

# Simple baseline: LISTA sparse autoencoder on higher-dimensional Lyapunov

DIM="${1:-8}"
TARGET_SIZE="${2:-256}"
SPARSITY="${3:-1.0}"
STEPS="${4:-10000}"
BS="${5:-512}"
RUN_DIR="${6:-runs/lyapunov_hd_dim${DIM}_lista_simple}"

echo "Running Lyapunov-HD simple baseline"
echo "  dim:        $DIM"
echo "  target:     $TARGET_SIZE"
echo "  sparsity:   $SPARSITY"
echo "  steps:      $STEPS"
echo "  batch:      $BS"
echo "  run_dir:    $RUN_DIR"

uv run python tools/train.py \
  --config lista_nonlinear \
  --env lyapunov \
  --lyapunov_dim "$DIM" \
  --lyapunov_extend_mode embed \
  --lyapunov_points_mode fixed \
  --num_steps "$STEPS" \
  --batch_size "$BS" \
  --target_size "$TARGET_SIZE" \
  --sparsity_coeff "$SPARSITY" \
  --log_dir "$RUN_DIR"

uv run python tools/evaluate_support_uniqueness.py \
  --checkpoint "$RUN_DIR/last.pt" \
  --system lyapunov \
  --output_dir "$RUN_DIR/support_eval"
