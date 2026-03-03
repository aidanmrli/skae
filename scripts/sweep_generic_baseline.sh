#!/bin/bash

#SBATCH --job-name=generic_base
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH -o /network/scratch/l/lia/skae/generic_base-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-11

# ============================================================================
# GenericKM Baseline with Various Dimensions
# ============================================================================
# Purpose: Establish strong baselines for comparison
# GenericKM previously achieved 88% linear classifier accuracy with 64 dims
# Testing: larger dimensions to see if MLP encoder can do better
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="/network/scratch/l/lia/skae/generic_baseline_lyapunov"
mkdir -p "$BASE_OUT"

echo "============================================="
echo "GenericKM Baseline Sweep"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"
echo "============================================="

BATCH_SIZE=1024  # GenericKM is fast, use large batch
NUM_STEPS=10000

# Grid: target_size x sparsity_coeff
# target_size: [64, 128, 256, 512]
# sparsity_coeff: [0.0, 0.01, 0.1]
# 4 * 3 = 12 configs

TARGET_SIZE_OPTIONS=(64 128 256 512 64 128 256 512 64 128 256 512)
SPARSITY_OPTIONS=(0.0 0.0 0.0 0.0 0.01 0.01 0.01 0.01 0.1 0.1 0.1 0.1)

TARGET_SIZE=${TARGET_SIZE_OPTIONS[$SLURM_ARRAY_TASK_ID]}
SPARSITY_COEFF=${SPARSITY_OPTIONS[$SLURM_ARRAY_TASK_ID]}

SEED=42
EXP_NAME="dim${TARGET_SIZE}_sp${SPARSITY_COEFF}"

echo "Config: target_size=$TARGET_SIZE, sparsity=$SPARSITY_COEFF"
echo "============================================="

LOG_DIR="${BASE_OUT}/${EXP_NAME}"

uv run python tools/train.py \
    --config generic_sparse \
    --env lyapunov \
    --target_size $TARGET_SIZE \
    --sparsity_coeff $SPARSITY_COEFF \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --reconst_coeff 0.5 \
    --pred_coeff 1.0 \
    --sequence_length 1 \
    --seed $SEED \
    --device cuda \
    --log_dir "$LOG_DIR"

TRAIN_EXIT=$?

# Evaluate
CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
    echo ">>> Running latent basin clustering evaluation..."
    EVAL_DIR="${LOG_DIR}/latent_eval"
    mkdir -p "$EVAL_DIR"

    uv run python tools/evaluate_latent_basin_clustering.py \
        --checkpoint "$CKPT" \
        --num_trajectories 100 \
        --output_dir "$EVAL_DIR" \
        --device cuda

    results="${EVAL_DIR}/analysis_results.json"
    if [ -f "$results" ]; then
        echo "=== Results: $EXP_NAME ==="
        uv run python -c "
import json
with open('$results') as f:
    r = json.load(f)
print(f'Config: dim=$TARGET_SIZE, sparsity=$SPARSITY_COEFF')
print(f'Linear Classifier Accuracy: {r.get(\"linear_classifier_accuracy\", 0):.4f}')
print(f'Silhouette Score: {r.get(\"silhouette_score\", 0):.4f}')
print(f'Final Pred Error: {r.get(\"final_pred_error\", 0):.4f}')
"
    fi
fi

echo "Done: $EXP_NAME, Exit: $TRAIN_EXIT"
