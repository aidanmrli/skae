#!/bin/bash

#SBATCH --job-name=struct_excl_sweep
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=6:00:00
#SBATCH -o /network/scratch/l/lia/skae/struct_excl_sweep-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-7

# ============================================================================
# StructuredLISTAKM Hyperparameter Sweep for Basin Specialization
# ============================================================================
# Goal: Improve basin assignment accuracy from 37.9% to >80%
# Key parameters: lambda_exclusivity, d_basin, num_basins, training steps
# ============================================================================

# Load modules
module load cuda/12.6.0

# Activate environment
source .venv/bin/activate

# Base output directory
BASE_OUT="/network/scratch/l/lia/skae/structured_excl_sweep_lyapunov"
mkdir -p "$BASE_OUT"

# Log job info
echo "============================================="
echo "StructuredLISTAKM Exclusivity Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================="

# Common parameters
BATCH_SIZE=256
SEED=42
RECONST_COEFF=0.5
PRED_COEFF=1.0
LISTA_ALPHA=0.3
LISTA_NUM_LOOPS=5

# Define experiment configurations
# Format: lambda_excl, d_basin, num_basins, num_steps, experiment_name
case $SLURM_ARRAY_TASK_ID in
    0)
        # Baseline with 10x stronger exclusivity
        LAMBDA_EXCL=0.01
        D_BASIN=4
        NUM_BASINS=13
        NUM_STEPS=10000
        EXP_NAME="excl_0.01_db4_nb13_10k"
        ;;
    1)
        # 50x stronger exclusivity
        LAMBDA_EXCL=0.05
        D_BASIN=4
        NUM_BASINS=13
        NUM_STEPS=10000
        EXP_NAME="excl_0.05_db4_nb13_10k"
        ;;
    2)
        # 100x stronger exclusivity
        LAMBDA_EXCL=0.1
        D_BASIN=4
        NUM_BASINS=13
        NUM_STEPS=10000
        EXP_NAME="excl_0.1_db4_nb13_10k"
        ;;
    3)
        # Larger basin capacity (d_basin=8)
        LAMBDA_EXCL=0.05
        D_BASIN=8
        NUM_BASINS=13
        NUM_STEPS=10000
        EXP_NAME="excl_0.05_db8_nb13_10k"
        ;;
    4)
        # Larger basin capacity + strong exclusivity
        LAMBDA_EXCL=0.1
        D_BASIN=8
        NUM_BASINS=13
        NUM_STEPS=10000
        EXP_NAME="excl_0.1_db8_nb13_10k"
        ;;
    5)
        # Longer training
        LAMBDA_EXCL=0.05
        D_BASIN=4
        NUM_BASINS=13
        NUM_STEPS=20000
        EXP_NAME="excl_0.05_db4_nb13_20k"
        ;;
    6)
        # Strong exclusivity + longer training
        LAMBDA_EXCL=0.1
        D_BASIN=4
        NUM_BASINS=13
        NUM_STEPS=20000
        EXP_NAME="excl_0.1_db4_nb13_20k"
        ;;
    7)
        # Over-specify basins (20 model basins for 13 GT basins)
        LAMBDA_EXCL=0.05
        D_BASIN=4
        NUM_BASINS=20
        NUM_STEPS=10000
        EXP_NAME="excl_0.05_db4_nb20_10k"
        ;;
esac

# Compute total latent dimension for logging
D_GLOBAL=8
TOTAL_DIM=$((D_GLOBAL + NUM_BASINS * D_BASIN))

echo "Experiment: $EXP_NAME"
echo "  lambda_exclusivity: $LAMBDA_EXCL"
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
