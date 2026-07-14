#!/bin/bash
#
# Submit one support-alignment reducer shard per model row, then merge them.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<final interpretability output directory>
#
# Optional env vars:
#   ROOT_LABELS_CSV=<comma-separated root labels>
#   ROOT_LABELS_FILE=<text file with MODEL_VARIANT=PATH per line>
#   SYSTEMS_CSV=<comma-separated system keys>
#   SEEDS_CSV=<comma-separated seeds>
#   LOG_DIR=<directory for shard and merge logs>
#   REDUCE_WALLTIME=12:00:00
#   REDUCE_CPUS=4
#   REDUCE_MEM=16G
#   PROGRESS_EVERY_RUNS=1
#   FLUSH_EVERY_RUNS=5
#   MERGE_WALLTIME=00:30:00
#   MERGE_MEM=4G
#   QUEUE_MANIFEST_JSON=<optional path for a submission manifest>

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

ROWS_CSV="${ROWS_CSV:?ROWS_CSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:-}"
ROOT_LABELS_FILE="${ROOT_LABELS_FILE:-}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
REDUCE_WALLTIME="${REDUCE_WALLTIME:-12:00:00}"
REDUCE_CPUS="${REDUCE_CPUS:-4}"
REDUCE_MEM="${REDUCE_MEM:-16G}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-5}"
MERGE_WALLTIME="${MERGE_WALLTIME:-00:30:00}"
MERGE_MEM="${MERGE_MEM:-4G}"
QUEUE_MANIFEST_JSON="${QUEUE_MANIFEST_JSON:-}"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/shards" "${LOG_DIR}"

ROOT_LABELS=()
if [[ -n "${ROOT_LABELS_CSV}" ]]; then
  IFS=',' read -r -a ROOT_LABELS <<< "${ROOT_LABELS_CSV}"
elif [[ -n "${ROOT_LABELS_FILE}" ]]; then
  mapfile -t ROOT_LABELS < <(awk -F= 'NF>=1 && $1!="" && !seen[$1]++ {print $1}' "${ROOT_LABELS_FILE}")
else
  ROOT_LABELS=(
    "lista_dense_signsplit_p256_hardinit_basin_partition"
    "lista_blockdiag_signsplit_hardinit_basin_partition"
    "lista_dense_softblock_signsplit_p256_hardinit_basin_partition"
    "mlp_sparse_blockdiag_hardinit_basin_partition_control"
    "mlp_sparse_hardinit_basin_partition_control"
    "mlp_zero_sparse_hardinit_basin_partition_control"
  )
fi

if (( ${#ROOT_LABELS[@]} == 0 )); then
  echo "No root labels resolved for interpretability sharding." >&2
  exit 1
fi

sanitize_name() {
  echo "$1" | tr -cs '[:alnum:]_-' '_'
}

SHARD_JOB_IDS=()
for root_label in "${ROOT_LABELS[@]}"; do
  [[ -z "${root_label}" ]] && continue
  slug="$(sanitize_name "${root_label}")"
  shard_out_dir="${OUT_DIR}/shards/${root_label}"
  mkdir -p "${shard_out_dir}"
  shard_job_id=$(
    ROWS_CSV="${ROWS_CSV}" \
    OUT_DIR="${shard_out_dir}" \
    ROOT_LABELS_CSV="${root_label}" \
    SYSTEMS_CSV="${SYSTEMS_CSV}" \
    SEEDS_CSV="${SEEDS_CSV}" \
    PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS}" \
    FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS}" \
      sbatch \
        --job-name="tr_interp_${slug}" \
        --time="${REDUCE_WALLTIME}" \
        --cpus-per-task="${REDUCE_CPUS}" \
        --mem="${REDUCE_MEM}" \
        --output="${LOG_DIR}/${slug}-%A.out" \
        --error="${LOG_DIR}/${slug}-%A.err" \
        scripts/neurips_2026/controlled/reduce_alignment.sh | awk '{print $4}'
  )
  SHARD_JOB_IDS+=("${shard_job_id}")
done

SHARD_DEPENDENCY="$(IFS=:; echo "${SHARD_JOB_IDS[*]}")"
MERGE_JOB_ID=$(
  SHARDS_DIR="${OUT_DIR}/shards" \
  OUT_DIR="${OUT_DIR}" \
  ROWS_CSV="${ROWS_CSV}" \
  ROOT_LABELS_CSV="$(IFS=,; echo "${ROOT_LABELS[*]}")" \
  SYSTEMS_CSV="${SYSTEMS_CSV}" \
  SEEDS_CSV="${SEEDS_CSV}" \
    sbatch \
      --dependency="afterok:${SHARD_DEPENDENCY}" \
      --time="${MERGE_WALLTIME}" \
      --mem="${MERGE_MEM}" \
      --output="${LOG_DIR}/merge-%A.out" \
      --error="${LOG_DIR}/merge-%A.err" \
      scripts/neurips_2026/controlled/merge_alignment.sh | awk '{print $4}'
)

if [[ -n "${QUEUE_MANIFEST_JSON}" ]]; then
  mkdir -p "$(dirname "${QUEUE_MANIFEST_JSON}")"
  cat > "${QUEUE_MANIFEST_JSON}" <<EOF
{
  "rows_csv": "${ROWS_CSV}",
  "out_dir": "${OUT_DIR}",
  "root_labels": [$(printf '"%s",' "${ROOT_LABELS[@]}" | sed 's/,$//')],
  "shard_job_ids": [$(printf '"%s",' "${SHARD_JOB_IDS[@]}" | sed 's/,$//')],
  "merge_job_id": "${MERGE_JOB_ID}",
  "log_dir": "${LOG_DIR}",
  "alignment_protocol": {
    "support_scheme": "absolute:0.001",
    "family_fit_scope": "all_generated_evaluation_trajectory_states",
    "scoring_scope": "per_observed_label_high_margin_tie_inclusive",
    "family_jaccard_threshold": 0.50,
    "mask_visit_order": "descending_frequency_then_ascending_packbits_bytes",
    "family_assignment_tie_break": "earliest_created_family",
    "num_evaluation_trajectories": 128,
    "trajectory_transitions": 128,
    "states_per_trajectory": 129,
    "evaluation_seed": 42,
    "native_label_systems": ["gated_local_linear", "gated_transfer_linear"],
    "native_label_source": "env.basin_label",
    "native_center_source": "env.points",
    "proxy_label_systems": ["claude:arrested_spiral", "claude:cal_asymmetric_3", "claude:cal_high_cross_3", "claude:cal_hexagon_6", "claude:cal_octagon_8", "claude:cal_pentagon_5", "claude:cal_square_4", "claude:duffing_triple_well", "claude:snic_multi", "claude:transition_routes_4", "claude:var_depth_gradient_4", "claude:var_diamond_4", "claude:var_l_shape_5"],
    "proxy_basin_count_source": "known_benchmark_count_for_evaluation_only",
    "proxy_endpoint_rollout_steps": 5000,
    "proxy_center_estimator": "deterministic_farthest_first_kmeans_on_advanced_endpoints",
    "kmeans_initial_center": "first_advanced_endpoint",
    "kmeans_farthest_tie_break": "first_endpoint_index",
    "kmeans_assignment_tie_break": "first_center_index",
    "kmeans_empty_cluster_rule": "retain_previous_center",
    "kmeans_max_iterations": 25,
    "proxy_state_label_rule": "nearest_estimated_center",
    "center_margin_definition": "second_nearest_center_distance_minus_nearest",
    "center_margin_quantile": 0.75,
    "center_margin_selection_rule": "margin_greater_than_or_equal_to_empirical_q75_tie_inclusive",
    "center_margin_tie_semantics": "retain_margin_greater_than_or_equal_to_q75; ties_can_make_the_scored_slice_larger_than_25_percent",
    "entropy_units": "nats",
    "family_count_semantics": "observed_family_ids_on_scored_high_margin_slice"
  }
}
EOF
fi

printf 'ROOT_LABELS_CSV=%q\n' "$(IFS=,; echo "${ROOT_LABELS[*]}")"
printf 'SHARD_JOB_IDS_CSV=%q\n' "$(IFS=,; echo "${SHARD_JOB_IDS[*]}")"
printf 'MERGE_JOB_ID=%q\n' "${MERGE_JOB_ID}"
printf 'LOG_DIR=%q\n' "${LOG_DIR}"
printf 'INTERPRETABILITY_ROWS_CSV=%q\n' "${OUT_DIR}/interpretability_rows.csv"
