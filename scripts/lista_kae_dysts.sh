#!/bin/bash
#
#SBATCH --job-name=dysts_multi_basin
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH -o /network/scratch/l/lia/skae/dysts-multibasin-%A_%a.out
#SBATCH --requeue

# Load modules
module load cuda/12.6.0

# Activate environment
source .venv/bin/activate

# Multi-basin systems (from benchmarks/system_catalog.py)
# (
#   Dadras
#   Duffing
#   QiChen
#   Sakarya
#   SprottTorus
#   Chua
#   MultiChua
#   DequanLi
#   LuChenCheng
#   SanUmSrisuchinwong
#   WangSun
#   ShimizuMorioka
  # LorenzCoupled
#   RikitakeDynamo
#   Hadley
# )
# Create output directory
BASE_OUT="/network/scratch/l/lia/skae/dysts_multi_basin_lista_nonlinear"
mkdir -p "$BASE_OUT"
SYSTEM=Duffing
# Log job info
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "System: $SYSTEM"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "============================================="

# Run training
uv run python tools/train.py \
  --config lista_nonlinear \
  --env "dysts:${SYSTEM}" \
  --num_steps 5000 \
  --batch_size 256 \
  --target_size 128 \
  --reconst_coeff 0.5 \
  --pred_coeff 1.0 \
  --sparsity_coeff 1.0 \
  --lista_alpha 0.30 \
  --lista_num_loops 5 \
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
