#!/bin/bash
#
#SBATCH --job-name=reeval_multibasin
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH -o /network/scratch/l/lia/skae/reeval-multibasin-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-14

# Load modules
module load cuda/12.6.0

# Activate environment
source .venv/bin/activate

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

# Re-evaluate checkpoints for this system across both model roots
uv run python dysts_experiments/reevaluate_multibasin_runs.py \
  --runs_roots runs/dysts_multi_basin_lista_nonlinear runs/dysts_multi_basin_generic_sparse \
  --systems "$SYSTEM" \
  --phase_portrait_length 30000 \
  --phase_portrait_batch_size 32 \
  --phase_portrait_dims 2,3 \
  --output_tag phase_30000 \
  --skip_existing \
  --device cuda

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $?"
echo "============================================="
