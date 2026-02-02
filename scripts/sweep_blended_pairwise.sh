#!/bin/bash

#SBATCH --job-name=blended_pair
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=6:00:00
#SBATCH -o /network/scratch/l/lia/skae/blended_pair_sweep-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-15

# ============================================================================
# StructuredLISTAKM Pairwise Baseline on BlendedLinearSystem
# ============================================================================
# Baseline comparison for sequence training experiments
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="/network/scratch/l/lia/skae/blended_pairwise_sweep"
mkdir -p "$BASE_OUT"

echo "============================================="
echo "Blended System Pairwise Baseline Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"
echo "============================================="

# Common parameters
D_GLOBAL=8
NUM_BASINS=5
BATCH_SIZE=256
NUM_STEPS=10000
EXCL_WARMUP=2000
LAMBDA_SPARSITY=0.3

# Sweep: d_basin x lambda_excl x seed
D_BASINS=(8 16)
LAMBDA_EXCLS=(0.01 0.05 0.1 0.2)
SEEDS=(42 142)

idx=$SLURM_ARRAY_TASK_ID
seed_idx=$((idx % 2))
idx=$((idx / 2))
excl_idx=$((idx % 4))
idx=$((idx / 4))
dbasin_idx=$((idx % 2))

D_BASIN=${D_BASINS[$dbasin_idx]}
LAMBDA_EXCL=${LAMBDA_EXCLS[$excl_idx]}
SEED=${SEEDS[$seed_idx]}

TOTAL_DIM=$((D_GLOBAL + NUM_BASINS * D_BASIN))
EXP_NAME="pairwise_db${D_BASIN}_excl${LAMBDA_EXCL}_seed${SEED}"

echo "Experiment: $EXP_NAME"
echo "  d_basin: $D_BASIN, num_basins: $NUM_BASINS"
echo "  lambda_exclusivity: $LAMBDA_EXCL"
echo "  total_latent_dim: $TOTAL_DIM"
echo "============================================="

LOG_DIR="${BASE_OUT}/${EXP_NAME}"

uv run python tools/train.py \
    --config lista_nonlinear \
    --env blended \
    --structured \
    --pairwise \
    --d_global $D_GLOBAL \
    --num_basins $NUM_BASINS \
    --d_basin $D_BASIN \
    --lambda_exclusivity $LAMBDA_EXCL \
    --lambda_sparsity $LAMBDA_SPARSITY \
    --excl_warmup_steps $EXCL_WARMUP \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --device cuda \
    --log_dir "$LOG_DIR"

echo "============================================="
echo "End Time: $(date)"
echo "============================================="
