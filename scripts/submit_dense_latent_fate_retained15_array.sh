#!/usr/bin/env bash
# Submit dense/zero-sparsity MLP continuous latent-fate clustering controls.

#SBATCH --job-name=dense-zfate15
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=results/latent_fate_dense_control_retained15_sbatch_20260515/logs/%A_%a.out
#SBATCH --error=results/latent_fate_dense_control_retained15_sbatch_20260515/logs/%A_%a.err

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-results/latent_fate_dense_control_retained15_sbatch_20260515}"
TASKS_TSV="${TASKS_TSV:-${OUT_ROOT}/tasks.tsv}"
MAX_PARALLEL="${MAX_PARALLEL:-30}"
DEVICE="${DEVICE:-cpu}"
ROWS_CSV="results/transition_rich_table2_controls_p256_compact_20260502/self_routed_controls/self_routed_forecasting_rows.csv"
ROOT_LABEL="mlp_zero_sparse_hardinit_basin_partition_control"

SYSTEMS=(
  "claude:arrested_spiral"
  "claude:cal_asymmetric_3"
  "claude:cal_hexagon_6"
  "claude:cal_high_cross_3"
  "claude:cal_octagon_8"
  "claude:cal_pentagon_5"
  "claude:cal_square_4"
  "claude:duffing_triple_well"
  "claude:snic_multi"
  "claude:transition_routes_4"
  "claude:var_depth_gradient_4"
  "claude:var_diamond_4"
  "claude:var_l_shape_5"
  "gated_local_linear"
  "gated_transfer_linear"
)

submit_array() {
  mkdir -p "${OUT_ROOT}/logs"
  : > "${TASKS_TSV}"

  local system
  local seed
  for system in "${SYSTEMS[@]}"; do
    for seed in 0 1; do
      printf '%s\t%s\n' "${system}" "${seed}" >> "${TASKS_TSV}"
    done
  done

  local task_count
  task_count="$(wc -l < "${TASKS_TSV}")"
  echo "Submitting ${task_count} dense latent-fate tasks"
  echo "Task table: ${TASKS_TSV}"
  echo "Output root: ${OUT_ROOT}"
  sbatch --array="0-$((task_count - 1))%${MAX_PARALLEL}" "$0"
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  submit_array
  exit 0
fi

task_line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS_TSV}")"
if [[ -z "${task_line}" ]]; then
  echo "No task at index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

IFS=$'\t' read -r system_key seed <<< "${task_line}"
safe_system="${system_key//:/_}"
output_dir="${OUT_ROOT}/shards/${safe_system}/seed_${seed}"
mkdir -p "${output_dir}"

echo "root=${ROOT_LABEL}"
echo "system=${system_key}"
echo "seed=${seed}"
echo "output_dir=${output_dir}"
echo "device=${DEVICE}"

uv run python tools/evaluate_latent_fate_components.py \
  --rows_csv "${ROWS_CSV}" \
  --output_dir "${output_dir}" \
  --root_labels "${ROOT_LABEL}" \
  --systems "${system_key}" \
  --seeds "${seed}" \
  --num_trajectories 64 \
  --trajectory_length 96 \
  --tail_window 16 \
  --max_clusters 12 \
  --min_silhouette 0.05 \
  --device "${DEVICE}"
