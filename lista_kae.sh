#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH -o /network/scratch/l/lia/skae/slurm-%j.out
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

# Run training
uv run python train.py \
  --config lista_nonlinear \
  --env lyapunov \
  --num_steps 20000 \
  --batch_size 256 \
  --target_size 512 \
  --reconst_coeff 1.0 \
  --pred_coeff 10.0 \
  --sparsity_coeff 1.5 \
  --lista_alpha 0.35 \
  --lista_num_loops 5 \
  --pairwise \
  --seed 42 \
  --device cuda

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $?"
echo "============================================="
