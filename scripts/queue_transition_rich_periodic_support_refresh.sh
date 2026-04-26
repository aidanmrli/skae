#!/bin/bash
#
# Submit post-entry periodic support-refresh shards.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<final output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#
# Optional env vars:
#   SYSTEMS_CSV=<comma-separated system keys>
#   SEEDS_CSV=<comma-separated seeds>
#   SEED_SPLITS_SEMICOLON=<semicolon-separated seed CSV groups, e.g. 0;1;2>
#   SUPPORT_DEFINITIONS=absolute:0.001,topk:8
#   LOG_DIR=<directory for logs>
#   WALLTIME=12:00:00
#   CPUS=4
#   MEM=24G
#   QUEUE_MANIFEST_JSON=<optional submission manifest path>

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
SEED_SPLITS_SEMICOLON="${SEED_SPLITS_SEMICOLON:-}"
SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS:-absolute:0.001,topk:8}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
WALLTIME="${WALLTIME:-12:00:00}"
CPUS="${CPUS:-4}"
MEM="${MEM:-24G}"
QUEUE_MANIFEST_JSON="${QUEUE_MANIFEST_JSON:-}"

mkdir -p "${OUT_DIR}/shards" "${LOG_DIR}"

sanitize_name() {
  echo "$1" | tr -cs '[:alnum:]_-' '_'
}

IFS=',' read -r -a ROOT_LABELS <<< "${ROOT_LABELS_CSV}"
if [[ -n "${SEED_SPLITS_SEMICOLON}" ]]; then
  IFS=';' read -r -a SEED_SPLITS <<< "${SEED_SPLITS_SEMICOLON}"
else
  SEED_SPLITS=("${SEEDS_CSV}")
fi

JOB_IDS=()
for root_label in "${ROOT_LABELS[@]}"; do
  [[ -z "${root_label}" ]] && continue
  for seed_split in "${SEED_SPLITS[@]}"; do
    slug="$(sanitize_name "${root_label}")"
    split_suffix=""
    if [[ -n "${seed_split}" ]]; then
      split_suffix="__seeds_$(sanitize_name "${seed_split//,/_}")"
    fi
    shard_slug="${slug}${split_suffix}"
    shard_out_dir="${OUT_DIR}/shards/${shard_slug}"
    mkdir -p "${shard_out_dir}"
    job_id=$(
      ROWS_CSVS="${ROWS_CSVS}" \
      OUT_DIR="${shard_out_dir}" \
      ROOT_LABELS_CSV="${root_label}" \
      SYSTEMS_CSV="${SYSTEMS_CSV}" \
      SEEDS_CSV="${seed_split}" \
      SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS}" \
      NUM_TRANSFERS_PER_PAIR="${NUM_TRANSFERS_PER_PAIR:-2}" \
      MAX_PAIRS_PER_SYSTEM="${MAX_PAIRS_PER_SYSTEM:-0}" \
      PRE_STEPS="${PRE_STEPS:-32}" \
      BRIDGE_STEPS="${BRIDGE_STEPS:-32}" \
      POST_STEPS="${POST_STEPS:-128}" \
      CONTINUATION_HORIZON="${CONTINUATION_HORIZON:-64}" \
      REENCODE_PERIODS="${REENCODE_PERIODS:-1,8}" \
      START_MODES="${START_MODES:-target_entry,post_start}" \
      ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}" \
      DEVICE="${DEVICE:-cpu}" \
      USE_DYNAMICS_PRIOR="${USE_DYNAMICS_PRIOR:-0}" \
      MAX_SPECS="${MAX_SPECS:-0}" \
      SMOKE="${SMOKE:-0}" \
        sbatch \
          --job-name="tr_refresh_${shard_slug}" \
          --time="${WALLTIME}" \
          --cpus-per-task="${CPUS}" \
          --mem="${MEM}" \
          --output="${LOG_DIR}/${shard_slug}-%A.out" \
          --error="${LOG_DIR}/${shard_slug}-%A.err" \
          scripts/run_transition_rich_periodic_support_refresh.sh | awk '{print $4}'
    )
    JOB_IDS+=("${job_id}")
  done
done

if [[ -n "${QUEUE_MANIFEST_JSON}" ]]; then
  mkdir -p "$(dirname "${QUEUE_MANIFEST_JSON}")"
  cat > "${QUEUE_MANIFEST_JSON}" <<EOF
{
  "rows_csvs": "${ROWS_CSVS}",
  "out_dir": "${OUT_DIR}",
  "root_labels_csv": "${ROOT_LABELS_CSV}",
  "systems_csv": "${SYSTEMS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "seed_splits_semicolon": "${SEED_SPLITS_SEMICOLON}",
  "support_definitions": "${SUPPORT_DEFINITIONS}",
  "walltime": "${WALLTIME}",
  "cpus": "${CPUS}",
  "mem": "${MEM}",
  "job_ids": "$(IFS=,; echo "${JOB_IDS[*]}")",
  "log_dir": "${LOG_DIR}"
}
EOF
fi

printf 'PERIODIC_SUPPORT_REFRESH_JOB_IDS_CSV=%q\n' "$(IFS=,; echo "${JOB_IDS[*]}")"
printf 'LOG_DIR=%q\n' "${LOG_DIR}"
printf 'SHARDS_DIR=%q\n' "${OUT_DIR}/shards"
