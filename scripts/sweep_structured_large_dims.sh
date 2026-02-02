#!/bin/bash

#SBATCH --job-name=struct_large
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH -o /network/scratch/l/lia/skae/struct_large-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-23

# ============================================================================
# StructuredLISTAKM with LARGE Basin Dimensions
# ============================================================================
# Hypothesis: Small d_basin (4) causes basin collapse due to limited capacity
# Testing: d_basin in [16, 32, 64], d_global in [16, 32]
# GPU: A100 40GB - use larger batch sizes for better utilization
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="/network/scratch/l/lia/skae/structured_large_dims_lyapunov"
mkdir -p "$BASE_OUT"

echo "============================================="
echo "StructuredLISTAKM Large Dimensions Sweep"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"
echo "============================================="

# Optimized for A100: larger batch size
BATCH_SIZE=512
SEED=42
LISTA_NUM_LOOPS=5
NUM_BASINS=20  # Over-specified for 13 GT basins

# Grid: d_global x d_basin x lambda_excl x lista_alpha
# d_global: [16, 32]
# d_basin: [16, 32, 64]
# lambda_excl: [0.01, 0.05]
# 2 * 3 * 2 = 12 configs, run 2 seeds each = 24 jobs

D_GLOBAL_OPTIONS=(16 32)
D_BASIN_OPTIONS=(16 32 64)
LAMBDA_EXCL_OPTIONS=(0.01 0.05)

# Compute indices
config_idx=$((SLURM_ARRAY_TASK_ID / 2))  # 0-11 for configs
seed_offset=$((SLURM_ARRAY_TASK_ID % 2))  # 0 or 1 for seed variation

d_global_idx=$((config_idx / 6))
remainder=$((config_idx % 6))
d_basin_idx=$((remainder / 2))
lambda_idx=$((remainder % 2))

D_GLOBAL=${D_GLOBAL_OPTIONS[$d_global_idx]}
D_BASIN=${D_BASIN_OPTIONS[$d_basin_idx]}
LAMBDA_EXCL=${LAMBDA_EXCL_OPTIONS[$lambda_idx]}
SEED=$((42 + seed_offset * 100))

# Compute dimensions
TOTAL_DIM=$((D_GLOBAL + NUM_BASINS * D_BASIN))

# Adjust training based on model size
if [ $TOTAL_DIM -gt 500 ]; then
    NUM_STEPS=8000
    RECONST_COEFF=0.2
    PRED_COEFF=1.0
    LISTA_ALPHA=0.25
else
    NUM_STEPS=10000
    RECONST_COEFF=0.3
    PRED_COEFF=1.0
    LISTA_ALPHA=0.3
fi

EXP_NAME="dg${D_GLOBAL}_db${D_BASIN}_excl${LAMBDA_EXCL}_s${SEED}"

echo "Config: d_global=$D_GLOBAL, d_basin=$D_BASIN, lambda_excl=$LAMBDA_EXCL"
echo "Total dim: $TOTAL_DIM, Seed: $SEED"
echo "Batch size: $BATCH_SIZE, Steps: $NUM_STEPS"
echo "============================================="

LOG_DIR="${BASE_OUT}/${EXP_NAME}"

uv run python tools/train.py \
    --config lista_nonlinear \
    --env lyapunov \
    --structured \
    --d_global $D_GLOBAL \
    --num_basins $NUM_BASINS \
    --d_basin $D_BASIN \
    --lambda_global 1e-4 \
    --lambda_local 1e-3 \
    --lambda_exclusivity $LAMBDA_EXCL \
    --lambda_sparsity 0.3 \
    --excl_warmup_steps 2000 \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --reconst_coeff $RECONST_COEFF \
    --pred_coeff $PRED_COEFF \
    --lista_alpha $LISTA_ALPHA \
    --lista_num_loops $LISTA_NUM_LOOPS \
    --pairwise \
    --seed $SEED \
    --device cuda \
    --log_dir "$LOG_DIR"

TRAIN_EXIT=$?

# Evaluate
CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
    echo ">>> Running basin structure evaluation..."
    EVAL_DIR="${LOG_DIR}/basin_eval"
    mkdir -p "$EVAL_DIR"

    uv run python tools/evaluate_basin_structure.py \
        --checkpoint "$CKPT" \
        --system lyapunov \
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
print(f'Config: d_g=$D_GLOBAL, d_b=$D_BASIN, λ_excl=$LAMBDA_EXCL, seed=$SEED')
print(f'Basin Assignment Accuracy: {r.get(\"basin_assignment_accuracy\", 0):.4f}')
print(f'Temporal Consistency: {r.get(\"temporal_consistency\", 0):.4f}')
print(f'Mean Activation Entropy: {r.get(\"mean_activation_entropy\", 0):.4f}')
"
    fi
fi

echo "Done: $EXP_NAME, Exit: $TRAIN_EXIT"
