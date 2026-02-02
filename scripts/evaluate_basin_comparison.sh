#!/bin/bash

#SBATCH --job-name=basin_eval
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=1:00:00
#SBATCH -o /network/scratch/l/lia/skae/basin_eval-%j.out
#SBATCH --requeue

# Load modules
module load cuda/12.6.0

# Activate environment
source .venv/bin/activate

# Base directories
BASE_TRAIN="/network/scratch/l/lia/skae/basin_comparison_lyapunov"
BASE_EVAL="/network/scratch/l/lia/skae/basin_comparison_eval"
mkdir -p "$BASE_EVAL"

echo "============================================="
echo "Basin Comparison Evaluation"
echo "Start Time: $(date)"
echo "============================================="

# Find the most recent checkpoint for each model
find_latest_checkpoint() {
    local model_dir="$1"
    # Find the most recent run directory and get checkpoint.pt
    local latest_run=$(ls -t "$model_dir" 2>/dev/null | head -1)
    if [ -n "$latest_run" ] && [ -f "$model_dir/$latest_run/checkpoint.pt" ]; then
        echo "$model_dir/$latest_run/checkpoint.pt"
    else
        echo ""
    fi
}

# Evaluate GenericKM
GENERIC_CKPT=$(find_latest_checkpoint "$BASE_TRAIN/generic_km")
if [ -n "$GENERIC_CKPT" ]; then
    echo ""
    echo ">>> Evaluating GenericKM: $GENERIC_CKPT"
    uv run python tools/evaluate_latent_basin_clustering.py \
        --checkpoint "$GENERIC_CKPT" \
        --system lyapunov \
        --num_trajectories 200 \
        --trajectory_length 300 \
        --output_dir "$BASE_EVAL/generic_km" \
        --device cuda
else
    echo "WARNING: GenericKM checkpoint not found"
fi

# Evaluate LISTAKM
LISTA_CKPT=$(find_latest_checkpoint "$BASE_TRAIN/lista_km")
if [ -n "$LISTA_CKPT" ]; then
    echo ""
    echo ">>> Evaluating LISTAKM: $LISTA_CKPT"
    uv run python tools/evaluate_latent_basin_clustering.py \
        --checkpoint "$LISTA_CKPT" \
        --system lyapunov \
        --num_trajectories 200 \
        --trajectory_length 300 \
        --output_dir "$BASE_EVAL/lista_km" \
        --device cuda
else
    echo "WARNING: LISTAKM checkpoint not found"
fi

# Evaluate StructuredLISTAKM (both general and structured-specific)
STRUCTURED_CKPT=$(find_latest_checkpoint "$BASE_TRAIN/structured_lista_km")
if [ -n "$STRUCTURED_CKPT" ]; then
    echo ""
    echo ">>> Evaluating StructuredLISTAKM (general): $STRUCTURED_CKPT"
    uv run python tools/evaluate_latent_basin_clustering.py \
        --checkpoint "$STRUCTURED_CKPT" \
        --system lyapunov \
        --num_trajectories 200 \
        --trajectory_length 300 \
        --output_dir "$BASE_EVAL/structured_lista_km_general" \
        --device cuda

    echo ""
    echo ">>> Evaluating StructuredLISTAKM (basin-specific): $STRUCTURED_CKPT"
    uv run python tools/evaluate_basin_structure.py \
        --checkpoint "$STRUCTURED_CKPT" \
        --system lyapunov \
        --num_trajectories 200 \
        --trajectory_length 300 \
        --output_dir "$BASE_EVAL/structured_lista_km_basins" \
        --device cuda
else
    echo "WARNING: StructuredLISTAKM checkpoint not found"
fi

# Generate comparison summary
echo ""
echo "============================================="
echo "COMPARISON SUMMARY"
echo "============================================="

for model in generic_km lista_km structured_lista_km_general; do
    results_file="$BASE_EVAL/$model/analysis_results.json"
    if [ -f "$results_file" ]; then
        echo ""
        echo "=== $model ==="
        # Extract key metrics using python
        uv run python -c "
import json
with open('$results_file') as f:
    r = json.load(f)
print(f'  Linear Classifier Acc: {r.get(\"linear_classifier_accuracy\", \"N/A\"):.4f}')
print(f'  Silhouette Score:      {r.get(\"silhouette_score\", \"N/A\"):.4f}')
print(f'  K-means Purity:        {r.get(\"kmeans_purity\", \"N/A\"):.4f}')
print(f'  Mean Sparsity:         {r.get(\"mean_sparsity\", \"N/A\"):.4f}')
"
    fi
done

# StructuredLISTAKM basin-specific metrics
structured_basins="$BASE_EVAL/structured_lista_km_basins/analysis_results.json"
if [ -f "$structured_basins" ]; then
    echo ""
    echo "=== structured_lista_km (basin-specific) ==="
    uv run python -c "
import json
with open('$structured_basins') as f:
    r = json.load(f)
print(f'  Basin Assignment Acc:  {r.get(\"basin_assignment_accuracy\", \"N/A\"):.4f}')
print(f'  Temporal Consistency:  {r.get(\"temporal_consistency\", \"N/A\"):.4f}')
print(f'  Mean Entropy:          {r.get(\"mean_activation_entropy\", \"N/A\"):.4f}')
"
fi

echo ""
echo "============================================="
echo "End Time: $(date)"
echo "Results saved to: $BASE_EVAL"
echo "============================================="
