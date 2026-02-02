#!/bin/bash

#SBATCH --job-name=blended_seq
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/blended_seq_sweep-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-23

# ============================================================================
# StructuredLISTAKM Sequence Training on BlendedLinearSystem (3 distinct basins)
# ============================================================================
# Tests whether model learns dynamical regimes vs geometric features
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="/network/scratch/l/lia/skae/blended_sequence_sweep"
mkdir -p "$BASE_OUT"

echo "============================================="
echo "Blended System Sequence Training Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================="

# Common parameters
D_GLOBAL=8
NUM_BASINS=5  # Over-specify: 5 model basins for 3 GT basins
BATCH_SIZE=256
NUM_STEPS=10000
EXCL_WARMUP=2000
LAMBDA_SPARSITY=0.3

# Sweep: sequence_length x d_basin x lambda_excl x seed
SEQ_LENGTHS=(10 20 50)
D_BASINS=(8 16)
LAMBDA_EXCLS=(0.01 0.05)
SEEDS=(42 142)

# Compute indices
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

TOTAL_DIM=$((D_GLOBAL + NUM_BASINS * D_BASIN))
EXP_NAME="seq${SEQ_LEN}_db${D_BASIN}_excl${LAMBDA_EXCL}_seed${SEED}"

echo "Experiment: $EXP_NAME"
echo "  sequence_length: $SEQ_LEN"
echo "  d_basin: $D_BASIN, num_basins: $NUM_BASINS"
echo "  lambda_exclusivity: $LAMBDA_EXCL"
echo "  total_latent_dim: $TOTAL_DIM"
echo "============================================="

LOG_DIR="${BASE_OUT}/${EXP_NAME}"

uv run python tools/train.py \
    --config lista_nonlinear \
    --env blended \
    --structured \
    --sequence \
    --sequence_length $SEQ_LEN \
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
echo "Experiment: $EXP_NAME"
echo "End Time: $(date)"
echo "============================================="
