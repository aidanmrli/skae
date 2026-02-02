#!/bin/bash

#SBATCH --job-name=basin_comparison
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH -o /network/scratch/l/lia/skae/basin_comparison-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-2

# Load modules
module load cuda/12.6.0

# Activate environment
source .venv/bin/activate

# Create output directory
BASE_OUT="/network/scratch/l/lia/skae/basin_comparison_lyapunov"
mkdir -p "$BASE_OUT"

# Log job info
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================="

# Common training parameters
NUM_STEPS=5000
BATCH_SIZE=256
TARGET_SIZE=64  # Match latent dimensions across models
SEED=42

case $SLURM_ARRAY_TASK_ID in
    0)
        # GenericKM (MLP encoder) - baseline
        MODEL_NAME="generic_sparse"
        LOG_DIR="${BASE_OUT}/generic_km"

        uv run python tools/train.py \
            --config generic_sparse \
            --env lyapunov \
            --num_steps $NUM_STEPS \
            --batch_size $BATCH_SIZE \
            --target_size $TARGET_SIZE \
            --reconst_coeff 0.02 \
            --pred_coeff 1.0 \
            --sparsity_coeff 0.001 \
            --pairwise \
            --seed $SEED \
            --device cuda \
            --log_dir "$LOG_DIR"
        ;;
    1)
        # LISTAKM (LISTA encoder, no structure)
        MODEL_NAME="lista_nonlinear"
        LOG_DIR="${BASE_OUT}/lista_km"

        uv run python tools/train.py \
            --config lista_nonlinear \
            --env lyapunov \
            --num_steps $NUM_STEPS \
            --batch_size $BATCH_SIZE \
            --target_size $TARGET_SIZE \
            --reconst_coeff 0.5 \
            --pred_coeff 1.0 \
            --sparsity_coeff 0.5 \
            --lista_alpha 0.3 \
            --lista_num_loops 5 \
            --pairwise \
            --seed $SEED \
            --device cuda \
            --log_dir "$LOG_DIR"
        ;;
    2)
        # StructuredLISTAKM (LISTA encoder with basin structure)
        # Lyapunov has 13 attractors, so use 13 basins
        # Total dim: d_global + num_basins * d_basin = 8 + 13*4 = 60 (close to 64)
        MODEL_NAME="structured_lista"
        LOG_DIR="${BASE_OUT}/structured_lista_km"

        uv run python tools/train.py \
            --config lista_nonlinear \
            --env lyapunov \
            --structured \
            --d_global 8 \
            --num_basins 13 \
            --d_basin 4 \
            --lambda_global 1e-4 \
            --lambda_local 1e-3 \
            --lambda_exclusivity 1e-3 \
            --lambda_sparsity 0.5 \
            --excl_warmup_steps 2000 \
            --num_steps $NUM_STEPS \
            --batch_size $BATCH_SIZE \
            --reconst_coeff 0.5 \
            --pred_coeff 1.0 \
            --lista_alpha 0.3 \
            --lista_num_loops 5 \
            --pairwise \
            --seed $SEED \
            --device cuda \
            --log_dir "$LOG_DIR"
        ;;
esac

echo "============================================="
echo "Model: $MODEL_NAME"
echo "End Time: $(date)"
echo "Exit Code: $?"
echo "============================================="
