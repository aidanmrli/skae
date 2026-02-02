#!/bin/bash

#SBATCH --job-name=lista_alpha_sweep
#SBATCH --ntasks=1
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=2:00:00
#SBATCH -o /network/scratch/l/lia/skae/lista_alpha_sweep-%A_%a.out
#SBATCH --requeue
#SBATCH --array=0-4

# Load modules
module load cuda/12.6.0

# Activate environment
source .venv/bin/activate

# Alpha values to sweep
ALPHAS=(0.1 0.2 0.3 0.4 0.5)
ALPHA=${ALPHAS[$SLURM_ARRAY_TASK_ID]}

BASE_OUT="/network/scratch/l/lia/skae/lista_alpha_sweep_lyapunov"
mkdir -p "$BASE_OUT"

echo "============================================="
echo "LISTA Alpha Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Alpha: $ALPHA"
echo "Start Time: $(date)"
echo "============================================="

# Train LISTAKM with this alpha
uv run python tools/train.py \
    --config lista_nonlinear \
    --env lyapunov \
    --num_steps 3000 \
    --batch_size 256 \
    --target_size 64 \
    --reconst_coeff 0.5 \
    --pred_coeff 1.0 \
    --sparsity_coeff 0.5 \
    --lista_alpha $ALPHA \
    --lista_num_loops 5 \
    --pairwise \
    --seed 42 \
    --device cuda \
    --log_dir "${BASE_OUT}/alpha_${ALPHA}"

# Run quick evaluation
CKPT=$(ls -t "${BASE_OUT}/alpha_${ALPHA}"/*/checkpoint.pt 2>/dev/null | head -1)
if [ -n "$CKPT" ]; then
    echo ""
    echo ">>> Evaluating alpha=$ALPHA"
    uv run python tools/evaluate_latent_basin_clustering.py \
        --checkpoint "$CKPT" \
        --system lyapunov \
        --num_trajectories 100 \
        --trajectory_length 200 \
        --output_dir "${BASE_OUT}/eval_alpha_${ALPHA}" \
        --device cuda

    # Print key metrics
    results="${BASE_OUT}/eval_alpha_${ALPHA}/analysis_results.json"
    if [ -f "$results" ]; then
        echo ""
        echo "=== Results for alpha=$ALPHA ==="
        uv run python -c "
import json
with open('$results') as f:
    r = json.load(f)
print(f'Alpha: $ALPHA')
print(f'  Linear Classifier Acc: {r.get(\"linear_classifier_accuracy\", 0):.4f}')
print(f'  Mean Sparsity:         {r.get(\"mean_sparsity\", 0):.4f}')
print(f'  Silhouette Score:      {r.get(\"silhouette_score\", 0):.4f}')
"
    fi
fi

echo "============================================="
echo "End Time: $(date)"
echo "============================================="
