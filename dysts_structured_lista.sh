#!/bin/bash
#
#SBATCH --job-name=dysts_structured
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/dysts-structured-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-14

# Load modules
module load cuda/12.6.0

# Activate environment
source .venv/bin/activate

# Create output directory
BASE_OUT="/network/scratch/l/lia/skae/dysts_structured_lista"
mkdir -p "$BASE_OUT"

# Multi-basin systems (from benchmarks/system_catalog.py)
SYSTEMS=(
  Dadras
  Duffing
  QiChen
  Sakarya
  SprottTorus
  Chua
  MultiChua
  DequanLi
  LuChenCheng
  SanUmSrisuchinwong
  WangSun
  ShimizuMorioka
  LorenzCoupled
  RikitakeDynamo
  Hadley
)

SYSTEM=${SYSTEMS[$SLURM_ARRAY_TASK_ID]}
if [ -z "$SYSTEM" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
  exit 1
fi

# Log job info
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "System: $SYSTEM"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================="

# ============================================
# StructuredLISTAKM Hyperparameters
# ============================================
# Structured latent space config
D_GLOBAL=8              # Global block dimension
NUM_BASINS=20           # Number of basin slots
D_BASIN=8               # Per-basin block dimension
# Total latent dim = D_GLOBAL + NUM_BASINS * D_BASIN = 168

# Structured loss weights
LAMBDA_GLOBAL=1e-4      # Global sparsity weight (small to allow dense global)
LAMBDA_LOCAL=1e-3       # Local/basin sparsity weight
LAMBDA_EXCL=1e-2        # Exclusivity penalty weight
EXCL_WARMUP=1000        # Steps to ramp exclusivity from 0 to final

# LISTA encoder config
LISTA_ALPHA=0.35        # Soft-threshold parameter
LISTA_NUM_LOOPS=5       # LISTA iterations

# Training config
NUM_STEPS=20000
BATCH_SIZE=256
RECONST_COEFF=1.0
PRED_COEFF=10.0

# Run training
uv run python train.py \
  --config lista_nonlinear \
  --env "dysts:${SYSTEM}" \
  --structured \
  --d_global $D_GLOBAL \
  --num_basins $NUM_BASINS \
  --d_basin $D_BASIN \
  --lambda_global $LAMBDA_GLOBAL \
  --lambda_local $LAMBDA_LOCAL \
  --lambda_exclusivity $LAMBDA_EXCL \
  --excl_warmup_steps $EXCL_WARMUP \
  --num_steps $NUM_STEPS \
  --batch_size $BATCH_SIZE \
  --reconst_coeff $RECONST_COEFF \
  --pred_coeff $PRED_COEFF \
  --lista_alpha $LISTA_ALPHA \
  --lista_num_loops $LISTA_NUM_LOOPS \
  --pairwise \
  --standardize \
  --dysts_ic_noise_scale 0.2 \
  --dysts_native_cache \
  --dysts_cache_warmup 2000 \
  --seed 42 \
  --device cuda \
  --log_dir "${BASE_OUT}/${SYSTEM}"

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $?"
echo "============================================="
