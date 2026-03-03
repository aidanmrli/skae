#!/bin/bash

#SBATCH --job-name=blended_temp
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/blended_temp_sweep-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-11

# ============================================================================
# StructuredLISTAKM Temporal Consistency Loss on BlendedLinearSystem
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="/network/scratch/l/lia/skae/blended_temporal_sweep"
mkdir -p "$BASE_OUT"

echo "============================================="
echo "Blended System Temporal Consistency Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Start Time: $(date)"
echo "============================================="

# Common parameters
D_GLOBAL=8
D_BASIN=16
NUM_BASINS=5
BATCH_SIZE=256
NUM_STEPS=10000
EXCL_WARMUP=2000
LAMBDA_SPARSITY=0.3
LAMBDA_EXCL=0.05

# Sweep: sequence_length x lambda_temporal x seed
SEQ_LENGTHS=(10 20)
LAMBDA_TEMPORALS=(0.0 0.01 0.1)
SEEDS=(42 142)

idx=$SLURM_ARRAY_TASK_ID
seed_idx=$((idx % 2))
idx=$((idx / 2))
temporal_idx=$((idx % 3))
idx=$((idx / 3))
seqlen_idx=$((idx % 2))

SEQ_LEN=${SEQ_LENGTHS[$seqlen_idx]}
LAMBDA_TEMPORAL=${LAMBDA_TEMPORALS[$temporal_idx]}
SEED=${SEEDS[$seed_idx]}

EXP_NAME="seq${SEQ_LEN}_temporal${LAMBDA_TEMPORAL}_seed${SEED}"

echo "Experiment: $EXP_NAME"
echo "  sequence_length: $SEQ_LEN"
echo "  lambda_temporal: $LAMBDA_TEMPORAL"
echo "  seed: $SEED"
echo "============================================="

LOG_DIR="${BASE_OUT}/${EXP_NAME}"

uv run python tools/train.py \
    --config lista_nonlinear \
    --env blended \
    --structured \
    --sequence_length $SEQ_LEN \
    --d_global $D_GLOBAL \
    --num_basins $NUM_BASINS \
    --d_basin $D_BASIN \
    --lambda_exclusivity $LAMBDA_EXCL \
    --lambda_sparsity $LAMBDA_SPARSITY \
    --lambda_temporal $LAMBDA_TEMPORAL \
    --excl_warmup_steps $EXCL_WARMUP \
    --num_steps $NUM_STEPS \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --device cuda \
    --log_dir "$LOG_DIR"

echo "============================================="
echo "End Time: $(date)"
echo "============================================="
