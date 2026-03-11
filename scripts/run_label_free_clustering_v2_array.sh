#!/bin/bash
#SBATCH --job-name=lfc_v2
#SBATCH --partition=long
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/network/scratch/l/lia/skae/lfc-v2-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/lfc-v2-%A_%a.err

set -euo pipefail

TASK_TSV="${TASK_TSV:?TASK_TSV not set}"
BASE_OUT="${BASE_OUT:?BASE_OUT not set}"
TASK_IDX="${SLURM_ARRAY_TASK_ID:?Not running as array job}"

cd /home/mila/l/lia/skae

# Read the task row (header + task_idx line)
HEADER=$(head -1 "${TASK_TSV}")
LINE=$(awk -v idx="${TASK_IDX}" 'NR == idx + 2' "${TASK_TSV}")

if [[ -z "${LINE}" ]]; then
    echo "ERROR: No task at index ${TASK_IDX}"
    exit 1
fi

# Parse TSV fields
IFS=$'\t' read -r TASK_ID SYSTEM FAMILY ROOT_LABEL SEED CHECKPOINT <<< "${LINE}"

OUTPUT_DIR="${BASE_OUT}/eval/${SYSTEM}/${FAMILY}/seed_${SEED}"
echo "Task ${TASK_ID}: system=${SYSTEM} family=${FAMILY} seed=${SEED}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"

uv run python tools/evaluate_label_free_clustering_v2.py \
    --checkpoint "${CHECKPOINT}" \
    --system "${SYSTEM}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_trajectories 256 \
    --trajectory_length 256 \
    --long_rollout_steps 5000 \
    --support_threshold 1e-3 \
    --pca_dim 20 \
    --seed 42 \
    --device cpu
