#!/bin/bash

#SBATCH --job-name=lista_large
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -o /network/scratch/l/lia/skae/lista_large-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-23

# ============================================================================
# Unstructured LISTAKM with Large Dimensions
# ============================================================================
# Hypothesis: LISTA's sparse support patterns naturally encode basin structure
# Without explicit partitioning, let sparsity create basin-specific supports
# Testing: target_size in [256, 512, 1024], various sparsity/alpha settings
# GPU: A100 40GB - use larger batch sizes
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="/network/scratch/l/lia/skae/unstructured_lista_lyapunov"
mkdir -p "$BASE_OUT"

echo "============================================="
echo "Unstructured LISTAKM Large Dimensions Sweep"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"
echo "============================================="

# Optimized for A100
BATCH_SIZE=512
LISTA_NUM_LOOPS=5

# Grid: target_size x sparsity_coeff x lista_alpha x reconst_coeff
# target_size: [256, 512, 1024]
# sparsity_coeff: [0.5, 1.0, 2.0]
# lista_alpha: [0.2, 0.35]
# 3 * 3 * 2 = 18 configs + 6 extra for larger dims = 24 jobs

TARGET_SIZE_OPTIONS=(256 512 1024 256 512 1024 256 512 1024 256 512 1024 256 512 1024 256 512 1024 2048 2048 2048 2048 2048 2048)
SPARSITY_OPTIONS=(0.5 0.5 0.5 1.0 1.0 1.0 2.0 2.0 2.0 0.5 0.5 0.5 1.0 1.0 1.0 2.0 2.0 2.0 0.5 1.0 2.0 0.5 1.0 2.0)
ALPHA_OPTIONS=(0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.2 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.35 0.25 0.25 0.25 0.35 0.35 0.35)

TARGET_SIZE=${TARGET_SIZE_OPTIONS[$SLURM_ARRAY_TASK_ID]}
SPARSITY_COEFF=${SPARSITY_OPTIONS[$SLURM_ARRAY_TASK_ID]}
LISTA_ALPHA=${ALPHA_OPTIONS[$SLURM_ARRAY_TASK_ID]}

# Adjust settings based on model size
if [ $TARGET_SIZE -ge 1024 ]; then
    NUM_STEPS=8000
    RECONST_COEFF=0.2
    PRED_COEFF=1.0
    BATCH_SIZE=256  # Reduce for larger models
elif [ $TARGET_SIZE -ge 512 ]; then
    NUM_STEPS=10000
    RECONST_COEFF=0.3
    PRED_COEFF=1.0
else
    NUM_STEPS=10000
    RECONST_COEFF=0.5
    PRED_COEFF=1.0
fi

SEED=42
EXP_NAME="dim${TARGET_SIZE}_sp${SPARSITY_COEFF}_alpha${LISTA_ALPHA}"

echo "Config: target_size=$TARGET_SIZE, sparsity=$SPARSITY_COEFF, alpha=$LISTA_ALPHA"
echo "Batch size: $BATCH_SIZE, Steps: $NUM_STEPS"
echo "============================================="

LOG_DIR="${BASE_OUT}/${EXP_NAME}"

uv run python train.py \
    --config lista_nonlinear \
    --env lyapunov \
    --target_size $TARGET_SIZE \
    --sparsity_coeff $SPARSITY_COEFF \
    --lista_alpha $LISTA_ALPHA \
    --lista_num_loops $LISTA_NUM_LOOPS \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --reconst_coeff $RECONST_COEFF \
    --pred_coeff $PRED_COEFF \
    --pairwise \
    --seed $SEED \
    --device cuda \
    --log_dir "$LOG_DIR"

TRAIN_EXIT=$?

# Evaluate using latent clustering (works for any model)
CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
    echo ">>> Running latent basin clustering evaluation..."
    EVAL_DIR="${LOG_DIR}/latent_eval"
    mkdir -p "$EVAL_DIR"

    uv run python evaluate_latent_basin_clustering.py \
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
print(f'Config: dim=$TARGET_SIZE, sparsity=$SPARSITY_COEFF, alpha=$LISTA_ALPHA')
print(f'Linear Classifier Accuracy: {r.get(\"linear_classifier_accuracy\", 0):.4f}')
print(f'Silhouette Score: {r.get(\"silhouette_score\", 0):.4f}')
print(f'Adjusted Rand Index: {r.get(\"adjusted_rand_index\", 0):.4f}')
print(f'Sparsity Ratio: {r.get(\"sparsity_ratio\", 0):.4f}')
print(f'Final Pred Error: {r.get(\"final_pred_error\", 0):.4f}')
"
    fi
fi

echo "Done: $EXP_NAME, Exit: $TRAIN_EXIT"
