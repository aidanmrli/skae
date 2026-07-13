#!/usr/bin/env bash
# Submit or run a retained-15 stable-support-component evaluation shard.
#
# Run without SLURM variables to create the task table and submit the array:
#   bash scripts/submit_stable_support_retained15_sparse_array.sh
#
# The array evaluates one (root, system, seed) tuple per task. Outputs are
# written as independent shards under OUT_ROOT for later aggregation.

#SBATCH --job-name=cstab-ret15
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=results/stable_support_components_retained15_sparse_sbatch_20260515/logs/%A_%a.out
#SBATCH --error=results/stable_support_components_retained15_sparse_sbatch_20260515/logs/%A_%a.err

set -euo pipefail

OUT_ROOT="${OUT_ROOT:-results/stable_support_components_retained15_sparse_sbatch_20260515}"
TASKS_TSV="${TASKS_TSV:-${OUT_ROOT}/tasks.tsv}"
MAX_PARALLEL="${MAX_PARALLEL:-50}"
DEVICE="${DEVICE:-cpu}"

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

add_root_tasks() {
  local slug="$1"
  local root_label="$2"
  local rows_csv="$3"
  local system
  local seed
  for system in "${SYSTEMS[@]}"; do
    for seed in 0 1; do
      printf '%s\t%s\t%s\t%s\t%s\n' "${slug}" "${root_label}" "${rows_csv}" "${system}" "${seed}" >> "${TASKS_TSV}"
    done
  done
}

submit_array() {
  mkdir -p "${OUT_ROOT}/logs"
  : > "${TASKS_TSV}"

  add_root_tasks \
    "lista_dense_p256" \
    "lista_dense_signsplit_p256_hardinit_basin_partition" \
    "results/transition_rich_lista_dense_p256_hardinit_table123_20260430/collect_pass0/forecasting_rows.csv"

  add_root_tasks \
    "lista_dense_softblock_p256" \
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition" \
    "results/transition_rich_lista_sb_p256_hardinit_fairness_seed15_20260428/collect_pass0/forecasting_rows.csv"

  add_root_tasks \
    "lista_blockdiag" \
    "lista_blockdiag_signsplit_hardinit_basin_partition" \
    "results/transition_rich_table2_5model_seed15_backfill_20260428/collect_pass0/forecasting_rows.csv"

  add_root_tasks \
    "sparse_mlp" \
    "mlp_sparse_hardinit_basin_partition_control" \
    "results/transition_rich_table2_5model_seed15_backfill_20260428/collect_pass0/forecasting_rows.csv"

  add_root_tasks \
    "sparse_mlp_blockdiag_repaired" \
    "mlp_sparse_blockdiag_hardinit_basin_partition_control" \
    "results/transition_rich_sparse_mlp_bd_repaired_table1_20260506/collect_pass0/forecasting_rows.csv"

  local task_count
  task_count="$(wc -l < "${TASKS_TSV}")"
  if [[ "${task_count}" -le 0 ]]; then
    echo "No tasks written to ${TASKS_TSV}" >&2
    exit 1
  fi

  echo "Submitting ${task_count} stable-support-component tasks"
  echo "Task table: ${TASKS_TSV}"
  echo "Output root: ${OUT_ROOT}"
  sbatch --array="0-$((task_count - 1))%${MAX_PARALLEL}" "$0"
}

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  submit_array
  exit 0
fi

if [[ ! -f "${TASKS_TSV}" ]]; then
  echo "Missing task table: ${TASKS_TSV}" >&2
  exit 1
fi

task_line="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS_TSV}")"
if [[ -z "${task_line}" ]]; then
  echo "No task at index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

IFS=$'\t' read -r slug root_label rows_csv system_key seed <<< "${task_line}"
safe_system="${system_key//:/_}"
output_dir="${OUT_ROOT}/shards/${slug}/${safe_system}/seed_${seed}"
mkdir -p "${output_dir}"

echo "root=${root_label}"
echo "system=${system_key}"
echo "seed=${seed}"
echo "rows_csv=${rows_csv}"
echo "output_dir=${output_dir}"
echo "device=${DEVICE}"

uv run python tools/evaluate_stable_support_components.py \
  --rows_csv "${rows_csv}" \
  --output_dir "${output_dir}" \
  --root_labels "${root_label}" \
  --systems "${system_key}" \
  --seeds "${seed}" \
  --support_definition absolute:0.001 \
  --base_object family \
  --base_family_jaccard 0.8 \
  --comparison_family_jaccard 0.5 \
  --num_trajectories 64 \
  --trajectory_length 96 \
  --tail_window 16 \
  --min_absorption_confidence 0.8 \
  --device "${DEVICE}"
