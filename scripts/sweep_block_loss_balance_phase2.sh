#!/bin/bash

#SBATCH --job-name=lyap_blk_bal_p2
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=8:00:00
#SBATCH -o /network/scratch/l/lia/skae/lyap_blk_bal_p2-%A_%a.out
#SBATCH --requeue

# ============================================================================
# Phase 2: Block Loss Balance Sweep (generalize across target sizes)
# ============================================================================
# Provide CONFIGS as a semicolon-separated list:
#   BAL_LOSS,ONE_WEIGHT,BAL_WEIGHT,TOP1_MARGIN
# Example:
#   CONFIGS="usage_entropy,0.3,0.3,0.1;kl_uniform,0.3,0.3,0.1"
#
# Grid: len(CONFIGS) x target sizes {64,128,256,512} x seeds {0,1,2}
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

TARGET_SIZES=(64 128 256 512)
SEEDS=(0 1 2)

DIM="${DIM:-8}"
NUM_BASINS="${NUM_BASINS:-13}"
NUM_BLOCKS="${NUM_BLOCKS:-20}"
SPARSITY="${SPARSITY:-1.0}"
NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-1e-3}"

if [ -z "$CONFIGS" ]; then
  echo "CONFIGS not set. Provide semicolon-separated list:"
  echo "  BAL_LOSS,ONE_WEIGHT,BAL_WEIGHT,TOP1_MARGIN"
  exit 1
fi

IFS=';' read -r -a CONFIG_LIST <<< "$CONFIGS"
NUM_CONFIGS=${#CONFIG_LIST[@]}
TOTAL_JOBS=$(( NUM_CONFIGS * ${#TARGET_SIZES[@]} * ${#SEEDS[@]} ))

idx=$SLURM_ARRAY_TASK_ID
seed_idx=$(( idx % ${#SEEDS[@]} ))
idx=$(( idx / ${#SEEDS[@]} ))
ts_idx=$(( idx % ${#TARGET_SIZES[@]} ))
idx=$(( idx / ${#TARGET_SIZES[@]} ))
config_idx=$(( idx % NUM_CONFIGS ))

SEED="${SEEDS[$seed_idx]}"
TARGET_SIZE="${TARGET_SIZES[$ts_idx]}"

CONFIG_SPEC="${CONFIG_LIST[$config_idx]}"
IFS=',' read -r BAL_LOSS ONE_WEIGHT BAL_WEIGHT TOP1_MARGIN <<< "$CONFIG_SPEC"

# Block-diagonal K with block size derived from NUM_BLOCKS (not GT basins)
K_BLOCK_SIZE=$(( TARGET_SIZE / NUM_BLOCKS ))
if [ "$K_BLOCK_SIZE" -lt 1 ]; then
  K_BLOCK_SIZE=1
fi

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lyapunov_block_loss_balance_phase2}"
EXP_NAME="dim${DIM}_nb${NUM_BASINS}_ts${TARGET_SIZE}_blkdiag_top1m${TOP1_MARGIN}_one${ONE_WEIGHT}_bal${BAL_WEIGHT}_${BAL_LOSS}_seed${SEED}"
LOG_DIR="${BASE_OUT}/${EXP_NAME}"

echo "============================================="
echo "Block Loss Balance Sweep (Phase 2)"
echo "Job ID: $SLURM_JOB_ID  Array Task: $SLURM_ARRAY_TASK_ID"
echo "DIM: $DIM  NUM_BASINS: $NUM_BASINS"
echo "TARGET_SIZE: $TARGET_SIZE  K_STRUCTURE: block_diagonal"
echo "NUM_BLOCKS: $NUM_BLOCKS  K_BLOCK_SIZE: $K_BLOCK_SIZE"
echo "ONE_BLOCK_LOSS: top1_margin (w=$ONE_WEIGHT, margin=$TOP1_MARGIN)"
echo "BALANCE_LOSS: $BAL_LOSS (w=$BAL_WEIGHT)"
echo "SEED: $SEED"
echo "TOTAL_CONFIGS: $NUM_CONFIGS  TOTAL_JOBS: $TOTAL_JOBS"
echo "LOG_DIR: $LOG_DIR"
echo "============================================="

uv run python tools/train.py \
  --config lista_nonlinear \
  --env lyapunov \
  --lyapunov_dim "$DIM" \
  --lyapunov_num_basins "$NUM_BASINS" \
  --lyapunov_extend_mode embed \
  --lyapunov_points_mode fixed \
  --num_steps "$NUM_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --target_size "$TARGET_SIZE" \
  --sparsity_coeff "$SPARSITY" \
  --seed "$SEED" \
  --k_structure block_diagonal \
  --k_block_size "$K_BLOCK_SIZE" \
  --pairwise \
  --monitor_support \
  --support_monitor_every 500 \
  --support_threshold "$SUPPORT_THRESHOLD" \
  --block_loss \
  --block_one_block_loss top1_margin \
  --block_one_block_weight "$ONE_WEIGHT" \
  --block_top1_margin "$TOP1_MARGIN" \
  --block_balance_loss "$BAL_LOSS" \
  --block_balance_weight "$BAL_WEIGHT" \
  --block_energy_norm l2 \
  --device cuda \
  --log_dir "$LOG_DIR"

TRAIN_EXIT=$?

CKPT=$(ls -t "${LOG_DIR}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ] && [ $TRAIN_EXIT -eq 0 ]; then
  echo ""
  echo ">>> Running cosine + threshold sweep evaluation..."
  EVAL_DIR="${LOG_DIR}/support_eval"
  mkdir -p "$EVAL_DIR"
  uv run python tools/evaluate_support_uniqueness.py \
    --checkpoint "$CKPT" \
    --system lyapunov \
    --support_mode mean \
    --cosine_diag \
    --threshold_sweep \
    --output_dir "$EVAL_DIR" \
    --device cuda
fi

echo "============================================="
echo "Experiment: $EXP_NAME"
echo "End Time: $(date)"
echo "Exit Code: $TRAIN_EXIT"
echo "============================================="
