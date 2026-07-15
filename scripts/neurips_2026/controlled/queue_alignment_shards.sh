#!/bin/bash
#
# Submit one support-alignment reducer shard per model row, then merge them.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<final interpretability output directory>
#
# Optional env vars:
#   ROOT_LABELS_CSV=<comma-separated root labels, required unless file is set>
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
  echo "ROOT_LABELS_CSV or ROOT_LABELS_FILE is required." >&2
  echo "Use queue_alignment.sh to submit the frozen six-row campaign." >&2
  exit 2
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
  "protocol_source": "Each reducer shard and the merged manifest record experiments.neurips_2026.alignment.alignment_protocol_metadata"
}
EOF
fi

printf 'ROOT_LABELS_CSV=%q\n' "$(IFS=,; echo "${ROOT_LABELS[*]}")"
printf 'SHARD_JOB_IDS_CSV=%q\n' "$(IFS=,; echo "${SHARD_JOB_IDS[*]}")"
printf 'MERGE_JOB_ID=%q\n' "${MERGE_JOB_ID}"
printf 'LOG_DIR=%q\n' "${LOG_DIR}"
printf 'INTERPRETABILITY_ROWS_CSV=%q\n' "${OUT_DIR}/interpretability_rows.csv"
