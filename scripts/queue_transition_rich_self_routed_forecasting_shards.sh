#!/bin/bash
#
# Submit one self-routed-forecasting shard per root label / seed split, then a merge job.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<final self-routed-forecasting output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#
# Optional env vars:
#   SYSTEMS_CSV=<comma-separated system keys>
#   SEEDS_CSV=<comma-separated seeds>
#   SEED_SPLITS_SEMICOLON=<semicolon-separated seed CSV groups, e.g. 0,1;2,3;4>
#   SYSTEM_SPLITS_SEMICOLON=<semicolon-separated system CSV groups>
#   SUPPORT_DEFINITIONS=relative:0.1,topk:8
#   DEPTH_STRATA=all,q1,q2,q3,q4
#   ROLLOUT_MODES=global_k,support_gated_k,support_block_gated_k,support_local_centered,family_local_centered
#   REENCODE_PERIODS=0
#   HORIZONS=100,500,1000
#   LOG_DIR=<directory for shard / merge logs>
#   REDUCE_PARTITION=long
#   REDUCE_DEPENDENCY=<optional SLURM dependency string, e.g. afterok:1:2>
#   REDUCE_WALLTIME=12:00:00
#   REDUCE_CPUS=4
#   REDUCE_MEM=24G
#   MERGE_PARTITION=long
#   MERGE_WALLTIME=00:30:00
#   MERGE_MEM=4G
#   FIT_NUM_TRAJECTORIES=256
#   FIT_TRAJECTORY_LENGTH=256
#   FIT_EVAL_SEED=42
#   FORECAST_NUM_TRAJECTORIES=128
#   FORECAST_EVAL_SEED=314
#   ENDPOINT_ROLLOUT_STEPS=5000
#   DEVICE=cpu
#   LABEL_MODE=auto
#   FIT_DYSTS_CACHE_SPLIT=
#   FORECAST_DYSTS_CACHE_SPLIT=
#   DYSTS_CACHE_PROFILE=full
#   DYSTS_CACHE_DIR=
#   DYSTS_CACHE_NUM_WORKERS=2
#   RIDGE_LAMBDA=1e-4
#   MIN_OPERATOR_TRANSITIONS=128
#   FAMILY_JACCARD_THRESHOLD=0.5
#   MAX_PARTITION_CLASSES=256
#   PROGRESS_EVERY_RUNS=1
#   FLUSH_EVERY_RUNS=1
#   MAX_RUNTIME_SECONDS=0
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
SEED_SPLITS_SEMICOLON="${SEED_SPLITS_SEMICOLON:-}"
SYSTEM_SPLITS_SEMICOLON="${SYSTEM_SPLITS_SEMICOLON:-}"
SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS:-relative:0.1,topk:8}"
DEPTH_STRATA="${DEPTH_STRATA:-all,q1,q2,q3,q4}"
ROLLOUT_MODES="${ROLLOUT_MODES:-global_k,support_gated_k,support_block_gated_k,support_local_centered,family_local_centered}"
REENCODE_PERIODS="${REENCODE_PERIODS:-0}"
HORIZONS="${HORIZONS:-100,500,1000}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
REDUCE_PARTITION="${REDUCE_PARTITION:-long}"
REDUCE_DEPENDENCY="${REDUCE_DEPENDENCY:-}"
REDUCE_WALLTIME="${REDUCE_WALLTIME:-12:00:00}"
REDUCE_CPUS="${REDUCE_CPUS:-4}"
REDUCE_MEM="${REDUCE_MEM:-24G}"
MERGE_PARTITION="${MERGE_PARTITION:-long}"
MERGE_WALLTIME="${MERGE_WALLTIME:-00:30:00}"
MERGE_MEM="${MERGE_MEM:-4G}"
FIT_NUM_TRAJECTORIES="${FIT_NUM_TRAJECTORIES:-256}"
FIT_TRAJECTORY_LENGTH="${FIT_TRAJECTORY_LENGTH:-256}"
FIT_EVAL_SEED="${FIT_EVAL_SEED:-42}"
FORECAST_NUM_TRAJECTORIES="${FORECAST_NUM_TRAJECTORIES:-128}"
FORECAST_EVAL_SEED="${FORECAST_EVAL_SEED:-314}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}"
DEVICE="${DEVICE:-cpu}"
LABEL_MODE="${LABEL_MODE:-auto}"
FIT_DYSTS_CACHE_SPLIT="${FIT_DYSTS_CACHE_SPLIT:-}"
FORECAST_DYSTS_CACHE_SPLIT="${FORECAST_DYSTS_CACHE_SPLIT:-}"
DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-full}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-}"
DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS:-2}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-4}"
MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS:-128}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.5}"
MAX_PARTITION_CLASSES="${MAX_PARTITION_CLASSES:-256}"
PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS:-1}"
FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS:-1}"
MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-0}"
QUEUE_MANIFEST_JSON="${QUEUE_MANIFEST_JSON:-}"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/shards" "${LOG_DIR}"
REDUCE_DEPENDENCY_ARGS=()
if [[ -n "${REDUCE_DEPENDENCY}" ]]; then
  REDUCE_DEPENDENCY_ARGS=(--dependency="${REDUCE_DEPENDENCY}")
fi
IFS=',' read -r -a ROOT_LABELS <<< "${ROOT_LABELS_CSV}"
if (( ${#ROOT_LABELS[@]} == 0 )); then
  echo "No root labels resolved for self-routed-forecasting sharding." >&2
  exit 1
fi

sanitize_name() {
  echo "$1" | tr -cs '[:alnum:]_-' '_'
}

SHARD_JOB_IDS=()
SEED_SPLITS=()
SYSTEM_SPLITS=()

if [[ -n "${SEED_SPLITS_SEMICOLON}" ]]; then
  IFS=';' read -r -a SEED_SPLITS <<< "${SEED_SPLITS_SEMICOLON}"
else
  SEED_SPLITS=("${SEEDS_CSV}")
fi
if [[ -n "${SYSTEM_SPLITS_SEMICOLON}" ]]; then
  IFS=';' read -r -a SYSTEM_SPLITS <<< "${SYSTEM_SPLITS_SEMICOLON}"
else
  SYSTEM_SPLITS=("${SYSTEMS_CSV}")
fi

for root_label in "${ROOT_LABELS[@]}"; do
  [[ -z "${root_label}" ]] && continue
  slug="$(sanitize_name "${root_label}")"
  if (( ${#SEED_SPLITS[@]} == 0 )); then
    SEED_SPLITS=("")
  fi
  if (( ${#SYSTEM_SPLITS[@]} == 0 )); then
    SYSTEM_SPLITS=("")
  fi
  for seed_split in "${SEED_SPLITS[@]}"; do
    for system_split in "${SYSTEM_SPLITS[@]}"; do
      split_suffix=""
      split_seeds="${seed_split}"
      split_systems="${system_split}"
      if [[ -n "${split_seeds}" ]]; then
        split_suffix="${split_suffix}__seeds_$(sanitize_name "${split_seeds//,/_}")"
      fi
      if [[ -n "${split_systems}" ]]; then
        split_suffix="${split_suffix}__systems_$(sanitize_name "${split_systems//,/_}")"
      fi
      shard_slug="${slug}${split_suffix}"
      shard_out_dir="${OUT_DIR}/shards/${root_label}${split_suffix}"
      mkdir -p "${shard_out_dir}"
      shard_job_id=$(
        ROWS_CSVS="${ROWS_CSVS}" \
        OUT_DIR="${shard_out_dir}" \
        ROOT_LABELS_CSV="${root_label}" \
        SYSTEMS_CSV="${split_systems}" \
        SEEDS_CSV="${split_seeds}" \
        SUPPORT_DEFINITIONS="${SUPPORT_DEFINITIONS}" \
        DEPTH_STRATA="${DEPTH_STRATA}" \
        ROLLOUT_MODES="${ROLLOUT_MODES}" \
        REENCODE_PERIODS="${REENCODE_PERIODS}" \
        HORIZONS="${HORIZONS}" \
        FIT_NUM_TRAJECTORIES="${FIT_NUM_TRAJECTORIES}" \
        FIT_TRAJECTORY_LENGTH="${FIT_TRAJECTORY_LENGTH}" \
        FIT_EVAL_SEED="${FIT_EVAL_SEED}" \
        FORECAST_NUM_TRAJECTORIES="${FORECAST_NUM_TRAJECTORIES}" \
        FORECAST_EVAL_SEED="${FORECAST_EVAL_SEED}" \
        ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS}" \
        DEVICE="${DEVICE}" \
        LABEL_MODE="${LABEL_MODE}" \
        FIT_DYSTS_CACHE_SPLIT="${FIT_DYSTS_CACHE_SPLIT}" \
        FORECAST_DYSTS_CACHE_SPLIT="${FORECAST_DYSTS_CACHE_SPLIT}" \
        DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE}" \
        DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR}" \
        DYSTS_CACHE_NUM_WORKERS="${DYSTS_CACHE_NUM_WORKERS}" \
        RIDGE_LAMBDA="${RIDGE_LAMBDA}" \
        MIN_OPERATOR_TRANSITIONS="${MIN_OPERATOR_TRANSITIONS}" \
        FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
        MAX_PARTITION_CLASSES="${MAX_PARTITION_CLASSES}" \
        PROGRESS_EVERY_RUNS="${PROGRESS_EVERY_RUNS}" \
        FLUSH_EVERY_RUNS="${FLUSH_EVERY_RUNS}" \
        MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS}" \
          sbatch \
            "${REDUCE_DEPENDENCY_ARGS[@]}" \
            --job-name="tr_srf_${shard_slug}" \
            --partition="${REDUCE_PARTITION}" \
            --time="${REDUCE_WALLTIME}" \
            --signal="B:USR1@300" \
            --cpus-per-task="${REDUCE_CPUS}" \
            --mem="${REDUCE_MEM}" \
            --output="${LOG_DIR}/${shard_slug}-%A.out" \
            --error="${LOG_DIR}/${shard_slug}-%A.err" \
            scripts/run_transition_rich_self_routed_forecasting.sh | awk '{print $4}'
      )
      SHARD_JOB_IDS+=("${shard_job_id}")
    done
  done
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
      --partition="${MERGE_PARTITION}" \
      --time="${MERGE_WALLTIME}" \
      --mem="${MERGE_MEM}" \
      --output="${LOG_DIR}/merge-%A.out" \
      --error="${LOG_DIR}/merge-%A.err" \
      scripts/merge_transition_rich_self_routed_forecasting_shards.sh | awk '{print $4}'
)

if [[ -n "${QUEUE_MANIFEST_JSON}" ]]; then
  mkdir -p "$(dirname "${QUEUE_MANIFEST_JSON}")"
  cat > "${QUEUE_MANIFEST_JSON}" <<EOF
{
  "rows_csvs": [$(printf '"%s",' ${ROWS_CSVS//,/ } | sed 's/,$//')],
  "out_dir": "${OUT_DIR}",
  "root_labels": [$(printf '"%s",' "${ROOT_LABELS[@]}" | sed 's/,$//')],
  "seed_splits": [$(printf '"%s",' "${SEED_SPLITS[@]}" | sed 's/,$//')],
  "system_splits": [$(printf '"%s",' "${SYSTEM_SPLITS[@]}" | sed 's/,$//')],
  "support_definitions": [$(printf '"%s",' ${SUPPORT_DEFINITIONS//,/ } | sed 's/,$//')],
  "depth_strata": [$(printf '"%s",' ${DEPTH_STRATA//,/ } | sed 's/,$//')],
  "rollout_modes": [$(printf '"%s",' ${ROLLOUT_MODES//,/ } | sed 's/,$//')],
  "reencode_periods": [$(printf '"%s",' ${REENCODE_PERIODS//,/ } | sed 's/,$//')],
  "label_mode": "${LABEL_MODE}",
  "fit_dysts_cache_split": "${FIT_DYSTS_CACHE_SPLIT}",
  "forecast_dysts_cache_split": "${FORECAST_DYSTS_CACHE_SPLIT}",
  "dysts_cache_profile": "${DYSTS_CACHE_PROFILE}",
  "reduce_partition": "${REDUCE_PARTITION}",
  "reduce_dependency": "${REDUCE_DEPENDENCY}",
  "reduce_walltime": "${REDUCE_WALLTIME}",
  "reduce_cpus": "${REDUCE_CPUS}",
  "reduce_mem": "${REDUCE_MEM}",
  "merge_partition": "${MERGE_PARTITION}",
  "merge_walltime": "${MERGE_WALLTIME}",
  "merge_mem": "${MERGE_MEM}",
  "progress_every_runs": "${PROGRESS_EVERY_RUNS}",
  "flush_every_runs": "${FLUSH_EVERY_RUNS}",
  "max_runtime_seconds": "${MAX_RUNTIME_SECONDS}",
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
printf 'SELF_ROUTED_FORECASTING_ROWS_CSV=%q\n' "${OUT_DIR}/self_routed_forecasting_rows.csv"
