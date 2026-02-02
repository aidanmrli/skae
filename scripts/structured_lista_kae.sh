#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH -o /network/scratch/l/lia/skae/structured-slurm-%j.out
#SBATCH --requeue

# Load modules
module load cuda/12.6.0

# Activate environment
source .venv/bin/activate

# Create output directory
mkdir -p /network/scratch/l/lia/skae

# Log job info
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================="

# ============================================
# StructuredLISTAKM Hyperparameters
# ============================================
# Structured latent space config
D_GLOBAL=32              # Global block dimension
NUM_BASINS=20           # Number of basin slots
D_BASIN=32               # Per-basin block dimension
# Total latent dim = D_GLOBAL + NUM_BASINS * D_BASIN

# Structured loss weights
LAMBDA_GLOBAL=1e-3      # Global sparsity weight
LAMBDA_LOCAL=1e-3       # Local/basin sparsity weight
LAMBDA_EXCL=1e-3        # Exclusivity penalty weight
LAMBDA_SPARSITY=1e-3    # Explicit L1 sparsity on full z
EXCL_WARMUP=1000        # Steps to ramp exclusivity/sparsity from 0 to final

# LISTA encoder config
LISTA_ALPHA=0.35        # Soft-threshold parameter
LISTA_NUM_LOOPS=5       # LISTA iterations

# Training config
NUM_STEPS=5000
BATCH_SIZE=256
RECONST_COEFF=0.05
PRED_COEFF=1.0

# Output directory with structure info
LOG_DIR="/network/scratch/l/lia/skae/structured_lista_g${D_GLOBAL}_b${NUM_BASINS}x${D_BASIN}/lyapunov"
mkdir -p "$LOG_DIR"

# Run training
uv run python tools/train.py \
  --config lista_nonlinear \
  --env lyapunov \
  --structured \
  --d_global $D_GLOBAL \
  --num_basins $NUM_BASINS \
  --d_basin $D_BASIN \
  --lambda_global $LAMBDA_GLOBAL \
  --lambda_local $LAMBDA_LOCAL \
  --lambda_exclusivity $LAMBDA_EXCL \
  --lambda_sparsity $LAMBDA_SPARSITY \
  --excl_warmup_steps $EXCL_WARMUP \
  --num_steps $NUM_STEPS \
  --batch_size $BATCH_SIZE \
  --reconst_coeff $RECONST_COEFF \
  --pred_coeff $PRED_COEFF \
  --lista_alpha $LISTA_ALPHA \
  --lista_num_loops $LISTA_NUM_LOOPS \
  --pairwise \
  --seed 42 \
  --device cuda \
  --log_dir "$LOG_DIR"

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $?"
echo "============================================="
