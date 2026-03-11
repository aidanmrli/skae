#!/bin/bash

#SBATCH --job-name=kuramoto_mode_audit
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/network/scratch/l/lia/skae/kuramoto_mode_support_audit/slurm-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/kuramoto_mode_support_audit/slurm-%A_%a.err

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

mapfile -t TASK_FIELDS < <(
  awk -F '\t' -v idx="${SLURM_ARRAY_TASK_ID}" '
    NR == idx + 2 {
      for (i = 1; i <= 21; ++i) {
        print $i
      }
      exit
    }
  ' "${TASK_TSV}"
)

if [[ ${#TASK_FIELDS[@]} -eq 0 ]]; then
  echo "No task line found for array index $SLURM_ARRAY_TASK_ID" >&2
  exit 1
fi

for i in "${!TASK_FIELDS[@]}"; do
  TASK_FIELDS[$i]="${TASK_FIELDS[$i]%$'\r'}"
done

task_id="${TASK_FIELDS[0]-}"
phase_label="${TASK_FIELDS[1]-}"
system="${TASK_FIELDS[2]-}"
family="${TASK_FIELDS[3]-}"
root_label="${TASK_FIELDS[4]-}"
model_variant="${TASK_FIELDS[5]-}"
seed="${TASK_FIELDS[6]-}"
checkpoint="${TASK_FIELDS[7]-}"
output_dir="${TASK_FIELDS[8]-}"
sampling_strategy="${TASK_FIELDS[9]-}"
num_trajectories="${TASK_FIELDS[10]-}"
trajectories_per_basin="${TASK_FIELDS[11]-}"
target_raw_labels_csv="${TASK_FIELDS[12]-}"
trajectory_length="${TASK_FIELDS[13]-}"
long_rollout_steps="${TASK_FIELDS[14]-}"
support_threshold="${TASK_FIELDS[15]-}"
support_modes_csv="${TASK_FIELDS[16]-}"
threshold_sweep_modes_csv="${TASK_FIELDS[17]-}"
thresholds_csv="${TASK_FIELDS[18]-}"
max_attempts="${TASK_FIELDS[19]-}"
device="${TASK_FIELDS[20]-}"

echo "Task: $task_id phase=$phase_label family=$family seed=$seed sampling=$sampling_strategy"
echo "Checkpoint: $checkpoint"
echo "Output dir: $output_dir"

mkdir -p "$output_dir"
export UV_LINK_MODE=copy

CMD=(uv run python tools/evaluate_kuramoto_mode_support_audit.py
  --checkpoint "$checkpoint"
  --system "$system"
  --output_dir "$output_dir"
  --sampling_strategy "$sampling_strategy"
  --trajectory_length "$trajectory_length"
  --long_rollout_steps "$long_rollout_steps"
  --support_threshold "$support_threshold"
  --support_modes_csv "$support_modes_csv"
  --threshold_sweep_modes_csv "$threshold_sweep_modes_csv"
  --thresholds_csv "$thresholds_csv"
  --seed 42
  --device "$device"
)

if [[ -n "$num_trajectories" ]]; then
  CMD+=(--num_trajectories "$num_trajectories")
fi
if [[ -n "$trajectories_per_basin" ]]; then
  CMD+=(--trajectories_per_basin "$trajectories_per_basin")
fi
if [[ -n "$target_raw_labels_csv" ]]; then
  CMD+=(--target_raw_labels_csv="$target_raw_labels_csv")
fi
if [[ -n "$max_attempts" ]]; then
  CMD+=(--max_attempts "$max_attempts")
fi

"${CMD[@]}"
