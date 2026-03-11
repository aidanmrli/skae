#!/bin/bash

#SBATCH --job-name=pp_h_lf_cluster
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/slurm-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/slurm-%A_%a.err

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: sbatch --array=0-N $0 <task_tsv>" >&2
  exit 2
fi

TASK_TSV="$1"
if [[ ! -f "$TASK_TSV" ]]; then
  echo "Task TSV not found: $TASK_TSV" >&2
  exit 1
fi

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"

module load cuda/12.6.0

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Git commit: $(git rev-parse HEAD)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

TASK_LINE="$(awk -F $'\t' -v idx="$SLURM_ARRAY_TASK_ID" 'NR == idx + 2 { print; exit }' "$TASK_TSV")"
if [[ -z "$TASK_LINE" ]]; then
  echo "No task line found for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

IFS=$'\t' read -r \
  task_id \
  system \
  family \
  root_label \
  seed \
  source_phase \
  checkpoint \
  output_dir \
  feature_view \
  num_trajectories \
  trajectory_length \
  long_rollout_steps \
  support_threshold <<<"$TASK_LINE"

echo "Task: $task_id system=$system family=$family root=$root_label seed=$seed"
echo "Checkpoint: $checkpoint"
echo "Output dir: $output_dir"

mkdir -p "$output_dir"
export UV_LINK_MODE=copy

uv run python tools/paper_parallel_20260309_h_evaluate_label_free_clustering.py \
  --checkpoint "$checkpoint" \
  --system "$system" \
  --output_dir "$output_dir" \
  --feature_view "$feature_view" \
  --support_threshold "$support_threshold" \
  --num_trajectories "$num_trajectories" \
  --trajectory_length "$trajectory_length" \
  --long_rollout_steps "$long_rollout_steps" \
  --seed 42 \
  --device cuda
