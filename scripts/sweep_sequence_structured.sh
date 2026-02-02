#!/bin/bash

#SBATCH --job-name=seq_struct_sweep
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/seq_struct_sweep-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-23

# ============================================================================
# StructuredLISTAKM Sequence Training Sweep for Basin-Block Correspondence
# ============================================================================
# Goal: Improve basin assignment accuracy from 76.6% (pairwise baseline)
# Key parameters: sequence_length, d_basin, lambda_exclusivity, lambda_temporal
# ============================================================================

# Load modules
module load cuda/12.6.0

# Activate environment
source .venv/bin/activate

# Base output directory
BASE_OUT="/network/scratch/l/lia/skae/structured_sequence_sweep_lyapunov"
mkdir -p "$BASE_OUT"

# Log job info
echo "============================================="
echo "StructuredLISTAKM Sequence Training Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================="

# Common parameters (from best pairwise config)
D_GLOBAL=16
NUM_BASINS=20
BATCH_SIZE=256
RECONST_COEFF=0.5
PRED_COEFF=1.0
LISTA_ALPHA=0.3
LISTA_NUM_LOOPS=5
NUM_STEPS=10000
EXCL_WARMUP=2000
LAMBDA_SPARSITY=0.3

# Define experiment configurations
# Sweep: sequence_length x d_basin x lambda_excl x seed
# sequence_length: {10, 20, 50}
# d_basin: {16, 32}
# lambda_excl: {0.01, 0.05}
# seeds: {42, 142}
# Total: 3 * 2 * 2 * 2 = 24 experiments

SEQ_LENGTHS=(10 20 50)
D_BASINS=(16 32)
LAMBDA_EXCLS=(0.01 0.05)
SEEDS=(42 142)

# Compute indices (4D array flattening)
idx=$SLURM_ARRAY_TASK_ID
seed_idx=$((idx % 2))
idx=$((idx / 2))
excl_idx=$((idx % 2))
idx=$((idx / 2))
dbasin_idx=$((idx % 2))
idx=$((idx / 2))
seqlen_idx=$((idx % 3))

SEQ_LEN=${SEQ_LENGTHS[$seqlen_idx]}
D_BASIN=${D_BASINS[$dbasin_idx]}
LAMBDA_EXCL=${LAMBDA_EXCLS[$excl_idx]}
SEED=${SEEDS[$seed_idx]}

# Compute total latent dimension for logging
TOTAL_DIM=$((D_GLOBAL + NUM_BASINS * D_BASIN))

EXP_NAME="seq${SEQ_LEN}_db${D_BASIN}_excl${LAMBDA_EXCL}_seed${SEED}"

echo "Experiment: $EXP_NAME"
echo "  sequence_length: $SEQ_LEN"
echo "  d_basin: $D_BASIN"
echo "  num_basins: $NUM_BASINS"
echo "  lambda_exclusivity: $LAMBDA_EXCL"
echo "  lambda_sparsity: $LAMBDA_SPARSITY"
echo "  seed: $SEED"
echo "  total_latent_dim: $TOTAL_DIM"
echo "============================================="

LOG_DIR="${BASE_OUT}/${EXP_NAME}"

# Train StructuredLISTAKM with sequence training
uv run python tools/train.py \
    --config lista_nonlinear \
    --env lyapunov \
    --structured \
    --sequence \
    --sequence_length $SEQ_LEN \
    --d_global $D_GLOBAL \
    --num_basins $NUM_BASINS \
    --d_basin $D_BASIN \
    --lambda_global 1e-4 \
    --lambda_local 1e-3 \
    --lambda_exclusivity $LAMBDA_EXCL \
    --lambda_sparsity $LAMBDA_SPARSITY \
    --excl_warmup_steps $EXCL_WARMUP \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --reconst_coeff $RECONST_COEFF \
    --pred_coeff $PRED_COEFF \
    --lista_alpha $LISTA_ALPHA \
    --lista_num_loops $LISTA_NUM_LOOPS \
    --seed $SEED \
    --device cuda \
    --log_dir "$LOG_DIR"

TRAIN_EXIT=$?

# Run basin structure evaluation
CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
    echo ""
    echo ">>> Running basin structure evaluation..."

    EVAL_DIR="${LOG_DIR}/basin_eval"
    mkdir -p "$EVAL_DIR"

    uv run python tools/evaluate_basin_structure.py \
        --checkpoint "$CKPT" \
        --system lyapunov \
        --num_trajectories 100 \
        --trajectory_length 500 \
        --output_dir "$EVAL_DIR" \
        --device cuda

    # Print summary
    results="${EVAL_DIR}/analysis_results.json"
    if [ -f "$results" ]; then
        echo ""
        echo "=== Basin Structure Results for $EXP_NAME ==="
        uv run python -c "
import json
with open('$results') as f:
    r = json.load(f)
print(f'Experiment: $EXP_NAME')
print(f'  Basin Assignment Accuracy: {r.get(\"basin_assignment_accuracy\", 0):.4f}')
print(f'  Temporal Consistency:      {r.get(\"temporal_consistency\", 0):.4f}')
print(f'  Mean Activation Entropy:   {r.get(\"mean_activation_entropy\", 0):.4f}')
print(f'  Within-Basin Similarity:   {r.get(\"within_basin_similarity\", 0):.4f}')
print(f'  Cross-Basin Separation:    {r.get(\"cross_basin_separation\", 0):.4f}')
"
    fi
fi

echo "============================================="
echo "Experiment: $EXP_NAME"
echo "End Time: $(date)"
echo "Exit Code: $TRAIN_EXIT"
echo "============================================="
