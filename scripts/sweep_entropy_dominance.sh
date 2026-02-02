#!/bin/bash

#SBATCH --job-name=struct_entropy_sweep
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=6:00:00
#SBATCH -o /network/scratch/l/lia/skae/struct_entropy_sweep-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-11

# ============================================================================
# StructuredLISTAKM Entropy/Dominance Loss Sweep
# ============================================================================
# Building on best config from exclusivity sweep: B=20 over-specified basins
# Goal: Push basin assignment accuracy from 61.47% toward >80%
# Key parameters: lambda_entropy, lambda_dominance
# ============================================================================

# Load modules
module load cuda/12.6.0

# Activate environment
source .venv/bin/activate

# Base output directory
BASE_OUT="/network/scratch/l/lia/skae/structured_entropy_sweep_lyapunov"
mkdir -p "$BASE_OUT"

# Log job info
echo "============================================="
echo "StructuredLISTAKM Entropy/Dominance Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================="

# Best config from first sweep: B=20, d_basin=4, lambda_excl=0.05
# Common parameters
BATCH_SIZE=256
SEED=42
RECONST_COEFF=0.5
PRED_COEFF=1.0
LISTA_ALPHA=0.3
LISTA_NUM_LOOPS=5
D_GLOBAL=8
NUM_BASINS=20  # Over-specified (13 GT basins)
D_BASIN=4
NUM_STEPS=10000

# Define experiment configurations
# Format: lambda_excl, lambda_entropy, lambda_dominance, experiment_name
case $SLURM_ARRAY_TASK_ID in
    0)
        # Baseline: best from first sweep (no entropy/dominance)
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.0
        LAMBDA_DOMINANCE=0.0
        EXP_NAME="baseline_excl0.05"
        ;;
    1)
        # Add entropy loss only (low)
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.01
        LAMBDA_DOMINANCE=0.0
        EXP_NAME="entropy_0.01"
        ;;
    2)
        # Add entropy loss only (medium)
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.05
        LAMBDA_DOMINANCE=0.0
        EXP_NAME="entropy_0.05"
        ;;
    3)
        # Add entropy loss only (high)
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.1
        LAMBDA_DOMINANCE=0.0
        EXP_NAME="entropy_0.1"
        ;;
    4)
        # Add dominance loss only (low)
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.0
        LAMBDA_DOMINANCE=0.01
        EXP_NAME="dominance_0.01"
        ;;
    5)
        # Add dominance loss only (medium)
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.0
        LAMBDA_DOMINANCE=0.05
        EXP_NAME="dominance_0.05"
        ;;
    6)
        # Add dominance loss only (high)
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.0
        LAMBDA_DOMINANCE=0.1
        EXP_NAME="dominance_0.1"
        ;;
    7)
        # Combined: entropy + dominance (low-low)
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.01
        LAMBDA_DOMINANCE=0.01
        EXP_NAME="combined_0.01_0.01"
        ;;
    8)
        # Combined: entropy + dominance (medium-medium)
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.05
        LAMBDA_DOMINANCE=0.05
        EXP_NAME="combined_0.05_0.05"
        ;;
    9)
        # Combined: entropy + dominance (high-high)
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.1
        LAMBDA_DOMINANCE=0.1
        EXP_NAME="combined_0.1_0.1"
        ;;
    10)
        # Entropy-focused: high entropy, low dominance
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.1
        LAMBDA_DOMINANCE=0.01
        EXP_NAME="entropy_focus_0.1_0.01"
        ;;
    11)
        # Dominance-focused: low entropy, high dominance
        LAMBDA_EXCL=0.05
        LAMBDA_ENTROPY=0.01
        LAMBDA_DOMINANCE=0.1
        EXP_NAME="dominance_focus_0.01_0.1"
        ;;
esac

# Compute total latent dimension for logging
TOTAL_DIM=$((D_GLOBAL + NUM_BASINS * D_BASIN))

echo "Experiment: $EXP_NAME"
echo "  lambda_exclusivity: $LAMBDA_EXCL"
echo "  lambda_entropy: $LAMBDA_ENTROPY"
echo "  lambda_dominance: $LAMBDA_DOMINANCE"
echo "  d_basin: $D_BASIN"
echo "  num_basins: $NUM_BASINS"
echo "  num_steps: $NUM_STEPS"
echo "  total_latent_dim: $TOTAL_DIM"
echo "============================================="

LOG_DIR="${BASE_OUT}/${EXP_NAME}"

# Train StructuredLISTAKM
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
    --lambda_entropy $LAMBDA_ENTROPY \
    --lambda_dominance $LAMBDA_DOMINANCE \
    --lambda_sparsity 0.5 \
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
print(f'  lambda_entropy: $LAMBDA_ENTROPY')
print(f'  lambda_dominance: $LAMBDA_DOMINANCE')
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
