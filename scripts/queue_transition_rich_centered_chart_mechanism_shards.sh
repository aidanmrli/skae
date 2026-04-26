#!/bin/bash
#
# Submit one centered-chart-mechanism shard per root label, then a merge job.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<final centered-chart-mechanism output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#
# Optional env vars:
#   SYSTEMS_CSV=<comma-separated system keys>
#   SEEDS_CSV=<comma-separated seeds>
#   SUPPORT_DEFINITIONS=relative:0.1,topk:8
#   DEPTH_STRATA=all,q1,q2,q3,q4
#   TRANSITION_REGIMES=all_current,persistent_current
#   PARTITION_KINDS=basin,family,support
#   LOG_DIR=<directory for shard / merge logs>
#   REDUCE_WALLTIME=12:00:00
#   REDUCE_CPUS=4
#   REDUCE_MEM=24G
#   MERGE_WALLTIME=00:30:00
#   MERGE_MEM=4G
#   NUM_TRAJECTORIES=256
#   TRAJECTORY_LENGTH=256
#   ENDPOINT_ROLLOUT_STEPS=5000
#   DEVICE=cpu
#   LABEL_MODE=auto
#   RIDGE_LAMBDA=1e-4
#   MIN_OPERATOR_TRANSITIONS=128
#   FAMILY_JACCARD_THRESHOLD=0.5
#   TRAIN_FRACTION=0.5
#   NUM_RANDOM_PARTITIONS=8
#   LATENT_KMEANS_MAX_CLASSES=16
#   MAX_PARTITION_CLASSES=256
#   PROGRESS_EVERY_RUNS=1
#   FLUSH_EVERY_RUNS=0
#   QUEUE_MANIFEST_JSON=<optional path for a submission manifest>

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

ROWS_CSVS="${ROWS_CSVS:?ROWS_CSVS is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:?ROOT_LABELS_CSV is required}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS:-relative:0.1,topk:8}"
DEPTH_STRATA="${DEPTH_STRATA:-all,q1,q2,q3,q4}"
TRANSITION_REGIMES="${TRANSITION_REGIMES:-all_current,persistent_current}"
PARTITION_KINDS="${PARTITION_KINDS:-basin,family,support}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
REDUCE_WALLTIME="${REDUCE_WALLTIME:-12:00:00}"
REDUCE_CPUS="${REDUCE_CPUS:-4}"
REDUCE_MEM="${REDUCE_MEM:-24G}"
MERGE_WALLTIME="${MERGE_WALLTIME:-00:30:00}"
MERGE_MEM="${MERGE_MEM:-4G}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-256}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-256}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}"
DEVICE="${DEVICE:-cpu}"
LABEL_MODE="${LABEL_MODE:-auto}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-4}"
MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS:-128}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.5}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.5}"
NUM_RANDOM_PARTITIONS="${NUM_RANDOM_PARTITIONS:-8}"
LATENT_KMEANS_MAX_CLASSES="${LATENT_KMEANS_MAX_CLASSES:-16}"
MAX_PARTITION_CLASSES="${MAX_PARTITION_CLASSES:-256}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-0}"
QUEUE_MANIFEST_JSON="${QUEUE_MANIFEST_JSON:-}"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/shards" "${LOG_DIR}"
IFS=',' read -r -a ROOT_LABELS <<< "${ROOT_LABELS_CSV}"
if (( ${#ROOT_LABELS[@]} == 0 )); then
  echo "No root labels resolved for centered-chart-mechanism sharding." >&2
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
    ROWS_CSVS="${ROWS_CSVS}" \
    OUT_DIR="${shard_out_dir}" \
    ROOT_LABELS_CSV="${root_label}" \
    SYSTEMS_CSV="${SYSTEMS_CSV}" \
    SEEDS_CSV="${SEEDS_CSV}" \
    SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS}" \
    DEPTH_STRATA="${DEPTH_STRATA}" \
    TRANSITION_REGIMES="${TRANSITION_REGIMES}" \
    PARTITION_KINDS="${PARTITION_KINDS}" \
    NUM_TRAJECTORIES="${NUM_TRAJECTORIES}" \
    TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH}" \
    ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS}" \
    DEVICE="${DEVICE}" \
    LABEL_MODE="${LABEL_MODE}" \
    RIDGE_LAMBDA="${RIDGE_LAMBDA}" \
    MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS}" \
    FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
    TRAIN_FRACTION="${TRAIN_FRACTION}" \
    NUM_RANDOM_PARTITIONS="${NUM_RANDOM_PARTITIONS}" \
    LATENT_KMEANS_MAX_CLASSES="${LATENT_KMEANS_MAX_CLASSES}" \
    MAX_PARTITION_CLASSES="${MAX_PARTITION_CLASSES}" \
    PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS}" \
    FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS}" \
      sbatch \
        --job-name="tr_ccm_${slug}" \
        --time="${REDUCE_WALLTIME}" \
        --cpus-per-task="${REDUCE_CPUS}" \
        --mem="${REDUCE_MEM}" \
        --output="${LOG_DIR}/${slug}-%A.out" \
        --error="${LOG_DIR}/${slug}-%A.err" \
        scripts/run_transition_rich_centered_chart_mechanism.sh | awk '{print $4}'
  )
  SHARD_JOB_IDS+=("${shard_job_id}")
done

SHARD_DEPENDENCY="$(IFS=:; echo "${SHARD_JOB_IDS[*]}")"
MERGE_JOB_ID=$(
  SHARDS_DIR="${OUT_DIR}/shards" \
  OUT_DIR="${OUT_DIR}" \
  ROWS_CSVS="${ROWS_CSVS}" \
  ROOT_LABELS_CSV="${ROOT_LABELS_CSV}" \
  SYSTEMS_CSV="${SYSTEMS_CSV}" \
  SEEDS_CSV="${SEEDS_CSV}" \
    sbatch \
      --dependency="afterok:${SHARD_DEPENDENCY}" \
      --time="${MERGE_WALLTIME}" \
      --mem="${MERGE_MEM}" \
      --output="${LOG_DIR}/merge-%A.out" \
      --error="${LOG_DIR}/merge-%A.err" \
      scripts/merge_transition_rich_centered_chart_mechanism_shards.sh | awk '{print $4}'
)

if [[ -n "${QUEUE_MANIFEST_JSON}" ]]; then
  mkdir -p "$(dirname "${QUEUE_MANIFEST_JSON}")"
  cat > "${QUEUE_MANIFEST_JSON}" <<EOF
{
  "rows_csvs": [$(printf '"%s",' ${ROWS_CSVS//,/ } | sed 's/,$//')],
  "out_dir": "${OUT_DIR}",
  "root_labels": [$(printf '"%s",' "${ROOT_LABELS[@]}" | sed 's/,$//')],
  "label_mode": "${LABEL_MODE}",
  "support_definitions": [$(printf '"%s",' ${SUPPORT_DEFINITIONS//,/ } | sed 's/,$//')],
  "depth_strata": [$(printf '"%s",' ${DEPTH_STRATA//,/ } | sed 's/,$//')],
  "transition_regimes": [$(printf '"%s",' ${TRANSITION_REGIMES//,/ } | sed 's/,$//')],
  "shard_job_ids": [$(printf '"%s",' "${SHARD_JOB_IDS[@]}" | sed 's/,$//')],
  "merge_job_id": "${MERGE_JOB_ID}",
  "log_dir": "${LOG_DIR}"
}
EOF
fi

printf 'ROOT_LABELS_CSV=%q\n' "${ROOT_LABELS_CSV}"
printf 'SHARD_JOB_IDS_CSV=%q\n' "$(IFS=,; echo "${SHARD_JOB_IDS[*]}")"
printf 'MERGE_JOB_ID=%q\n' "${MERGE_JOB_ID}"
printf 'LOG_DIR=%q\n' "${LOG_DIR}"
printf 'CENTERED_CHART_MECHANISM_ROWS_CSV=%q\n' "${OUT_DIR}/centered_chart_mechanism_rows.csv"
