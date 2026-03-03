#!/bin/bash

#SBATCH --job-name=struct_support_unique
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/struct_support_unique-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-23

# ============================================================================
# StructuredLISTAKM Support Uniqueness Sweep (Lyapunov)
# ============================================================================
# Focused around best config: d_g=16, d_b=16, B=20
# Sweep: lambda_exclusivity x lambda_sparsity x lambda_temporal (1 seed)
# ============================================================================ 

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="/network/scratch/l/lia/skae/structured_support_unique_lyapunov"
mkdir -p "$BASE_OUT"

echo "============================================="
echo "StructuredLISTAKM Support Uniqueness Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================="

# Fixed architecture (best known)
D_GLOBAL=16
D_BASIN=16
NUM_BASINS=20

# Training params
BATCH_SIZE=256
NUM_STEPS=10000
RECONST_COEFF=0.5
PRED_COEFF=1.0
LISTA_ALPHA=0.3
LISTA_NUM_LOOPS=5
EXCL_WARMUP=2000
SEED=42

# Sweep grid (4 * 3 * 2 = 24)
LAMBDA_EXCLS=(0.005 0.01 0.02 0.05)
LAMBDA_SPARSES=(0.1 0.3 0.5)
LAMBDA_TEMPORALS=(0.0 0.05)

idx=$SLURM_ARRAY_TASK_ID
temp_idx=$((idx % 2))
idx=$((idx / 2))
sparse_idx=$((idx % 3))
idx=$((idx / 3))
excl_idx=$((idx % 4))

LAMBDA_EXCL=${LAMBDA_EXCLS[$excl_idx]}
LAMBDA_SPARSITY=${LAMBDA_SPARSES[$sparse_idx]}
LAMBDA_TEMPORAL=${LAMBDA_TEMPORALS[$temp_idx]}

EXP_NAME="excl${LAMBDA_EXCL}_sp${LAMBDA_SPARSITY}_temp${LAMBDA_TEMPORAL}"
LOG_DIR="${BASE_OUT}/${EXP_NAME}"

echo "Experiment: $EXP_NAME"
echo "  d_global: $D_GLOBAL"
echo "  d_basin: $D_BASIN"
echo "  num_basins: $NUM_BASINS"
echo "  lambda_exclusivity: $LAMBDA_EXCL"
echo "  lambda_sparsity: $LAMBDA_SPARSITY"
echo "  lambda_temporal: $LAMBDA_TEMPORAL"
echo "  seed: $SEED"
echo "============================================="

python tools/train.py \
    --config lista_nonlinear \
    --env lyapunov \
    --structured \
    --d_global $D_GLOBAL \
    --num_basins $NUM_BASINS \
    --d_basin $D_BASIN \
    --lambda_global 1e-4 \
    --lambda_local 1e-3 \
    --lambda_exclusivity $LAMBDA_EXCL \
    --lambda_sparsity $LAMBDA_SPARSITY \
    --lambda_temporal $LAMBDA_TEMPORAL \
    --excl_warmup_steps $EXCL_WARMUP \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --reconst_coeff $RECONST_COEFF \
    --pred_coeff $PRED_COEFF \
    --lista_alpha $LISTA_ALPHA \
    --lista_num_loops $LISTA_NUM_LOOPS \
    --sequence_length 1 \
    --monitor_support \
    --support_monitor_every 500 \
    --support_threshold 1e-3 \
    --seed $SEED \
    --device cuda \
    --log_dir "$LOG_DIR"

TRAIN_EXIT=$?

CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo ">>> Running support uniqueness evaluation..."
    EVAL_DIR="${LOG_DIR}/support_eval"
    mkdir -p "$EVAL_DIR"

    python tools/evaluate_support_uniqueness.py \
        --checkpoint "$CKPT" \
        --system lyapunov \
        --support_threshold 1e-3 \
        --support_mode mean \
        --output_dir "$EVAL_DIR" \
        --device cpu

    results="${EVAL_DIR}/support_uniqueness.json"
    if [ -f "$results" ]; then
        echo ""
        echo "=== Support Results: $EXP_NAME ==="
        python - <<PY
import json
with open("$results") as f:
    r = json.load(f)
print(f"Unique supports: {r.get('unique_mode_supports', 0)}/{r.get('num_basins', 0)}")
print(f"Mean pairwise Jaccard: {r.get('mean_pairwise_jaccard', 0):.4f}")
print(f"Mean basin consistency: {r.get('mean_basin_consistency', 0):.4f}")
print(f"Mean support size: {r.get('mean_mode_support_size', 0):.2f}")
PY
    fi
fi

echo "============================================="
echo "Experiment: $EXP_NAME"
echo "End Time: $(date)"
echo "Exit Code: $TRAIN_EXIT"
echo "============================================="
