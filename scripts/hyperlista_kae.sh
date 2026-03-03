#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=3:00:00
#SBATCH -o /network/scratch/l/lia/skae/hyperlista-%j.out
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
echo "Training: HyperLISTA Koopman Autoencoder"
echo "============================================="

# Run training with HyperLISTA
# Key differences from LISTA:
# - Uses analytically-derived encoder weights from dictionary
# - Only 3 learnable hyperparams (c_theta, c_beta, c_ss)
# - Instance-adaptive threshold and momentum
uv run python tools/train.py \
  --config hyperlista \
  --env lyapunov \
  --num_steps 3000 \
  --batch_size 256 \
  --target_size 256 \
  --reconst_coeff 1.0 \
  --pred_coeff 1.0 \
  --sparsity_coeff 1.0 \
  --lista_num_loops 5 \
  --hyperlista_c_theta 1e-2 \
  --hyperlista_c_beta 1e-4 \
  --hyperlista_c_ss 0.5 \
  --sequence_length 1 \
  --lr 5e-5 \
  --seed 42 \
  --device cuda

# HyperLISTA scalars:
# - hyperlista_c_theta: threshold scaling (C_THETA)
# - hyperlista_c_beta: momentum scaling (C_BETA)
# - hyperlista_c_ss: support-selection scaling (C_SS)

echo "============================================="
echo "End Time: $(date)"
echo "Exit Code: $?"
echo "============================================="
