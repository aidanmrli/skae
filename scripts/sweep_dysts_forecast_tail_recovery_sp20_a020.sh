#!/bin/bash
#
#SBATCH --job-name=dysts_fcast_tailrec
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/dysts-fcast-tailrec-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-17

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/dysts_forecast_tail_recovery_sp20_a020_ts256}"
mkdir -p "$BASE_OUT"

SYSTEMS=(
  LorenzCoupled
  SprottTorus
  MultiChua
  Hadley
  Dadras
  LuChenCheng
)
SEEDS=(0 1 2)

NUM_SEEDS=${#SEEDS[@]}
SYS_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

SYSTEM=${SYSTEMS[$SYS_IDX]}
SEED=${SEEDS[$SEED_IDX]}

if [ -z "$SYSTEM" ] || [ -z "$SEED" ]; then
  echo "Invalid task mapping for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
  exit 1
fi

echo "============================================="
echo "Dysts Forecast Tail Recovery (LISTA ReLU)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "System: $SYSTEM"
echo "Seed: $SEED"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "BASE_OUT: $BASE_OUT"
echo "============================================="

uv run python tools/train.py \
  --config lista_nonlinear \
  --env "dysts:${SYSTEM}" \
  --num_steps 10000 \
  --batch_size 256 \
  --target_size 256 \
  --reconst_coeff 0.5 \
  --pred_coeff 1.0 \
  --sparsity_coeff 2.0 \
  --lista_alpha 0.2 \
  --lista_num_loops 5 \
  --lista_final_op relu \
  --sequence_length 1 \
  --standardize \
  --dysts_ic_noise_scale 0.2 \
  --dysts_native_cache \
  --dysts_cache_warmup 2000 \
  --seed "$SEED" \
  --device cuda \
  --log_dir "${BASE_OUT}/${SYSTEM}/seed_${SEED}"

EXIT_CODE=$?

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "============================================="
exit $EXIT_CODE
