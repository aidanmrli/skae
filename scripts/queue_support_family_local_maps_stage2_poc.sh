#!/bin/bash
#
# Queue stage-2 support-family local-map jobs, or the calibrated global-map
# ablation, for existing checkpoints.
#
# Required env vars:
#   ROWS_CSVS=<comma-separated forecasting_rows.csv files>
#   OUT_DIR=<output directory>
#   ROOT_LABELS_CSV=<comma-separated root labels>
#   SYSTEMS_CSV=<comma-separated system keys>
#
# Optional env vars:
#   SEEDS_CSV=0
#   REENCODE_PERIODS=1,2,5,10
#   ROUTE_FREEZE_MODES=reroute_each_step,freeze_within_segment
#   TRAIN_STEPS=20000
#   HORIZONS=100,500,1000
#   SUPPORT_DEFINITION=topk:8
#   FAMILY_JACCARD_THRESHOLD=0.4
#   STAGE2_MAP_MODE=family_local_centered # or global_dense_calibrated
#   WALLTIME=03:00:00
#   PARTITION=long
#   CPUS=4
#   MEM=24G
#   DEVICE=cpu
#   GRES=           # set to gpu:1 for GPU workers
#   LABEL_MODE=auto # use none for Dysts
#   MAX_RUNTIME_SECONDS=0
#   RESUME_FROM_OUTPUT_DIRS=
#   WORKER_DEPENDENCY= # e.g. afterok:12345 to hold all workers
#   MERGE_PARTITION=${PARTITION}
#   MERGE_WALLTIME=00:30:00
#   MERGE_MEM=4G
#   AGGREGATE=1
#   AGG_PARTITION=${MERGE_PARTITION}
#   AGG_WALLTIME=00:30:00
#   AGG_MEM=8G
#   AGG_DATASETS=multibasin
#   SKIP_COMPLETED=0
#   LOG_DIR=${OUT_DIR}/logs
#   QUEUE_MANIFEST_JSON=${OUT_DIR}/queue_manifest.json

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
SYSTEMS_CSV="${SYSTEMS_CSV:?SYSTEMS_CSV is required}"
SEEDS_CSV="${SEEDS_CSV:-0}"
REENCODE_PERIODS="${REENCODE_PERIODS:-1,2,5,10}"
ROUTE_FREEZE_MODES="${ROUTE_FREEZE_MODES:-reroute_each_step,freeze_within_segment}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
HORIZONS="${HORIZONS:-100,500,1000}"
SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-topk:8}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.4}"
STAGE2_MAP_MODE="${STAGE2_MAP_MODE:-family_local_centered}"
WALLTIME="${WALLTIME:-03:00:00}"
PARTITION="${PARTITION:-long}"
CPUS="${CPUS:-4}"
MEM="${MEM:-24G}"
DEVICE="${DEVICE:-cpu}"
GRES="${GRES:-}"
LABEL_MODE="${LABEL_MODE:-auto}"
RESUME_FROM_OUTPUT_DIRS="${RESUME_FROM_OUTPUT_DIRS:-}"
WORKER_DEPENDENCY="${WORKER_DEPENDENCY:-}"
MERGE_PARTITION="${MERGE_PARTITION:-${PARTITION}}"
MERGE_WALLTIME="${MERGE_WALLTIME:-00:30:00}"
MERGE_MEM="${MERGE_MEM:-4G}"
AGGREGATE="${AGGREGATE:-1}"
AGG_PARTITION="${AGG_PARTITION:-${MERGE_PARTITION}}"
AGG_WALLTIME="${AGG_WALLTIME:-00:30:00}"
AGG_MEM="${AGG_MEM:-8G}"
AGG_DATASETS="${AGG_DATASETS:-multibasin}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
LOG_DIR="${LOG_DIR:-${OUT_DIR}/logs}"
QUEUE_MANIFEST_JSON="${QUEUE_MANIFEST_JSON:-${OUT_DIR}/queue_manifest.json}"

mkdir -p "${OUT_DIR}" "${OUT_DIR}/shards" "${LOG_DIR}"

sanitize_name() {
  echo "$1" | tr -cs '[:alnum:]_-' '_'
}

is_completed_shard() {
  local shard_dir="$1"
  [[ -s "${shard_dir}/self_routed_forecasting_rows.csv" ]] || return 1
  [[ -f "${shard_dir}/failures.json" ]] || return 1
  if tr -d '[:space:]' < "${shard_dir}/failures.json" | grep -qx '\[\]'; then
    return 0
  fi
  return 1
}

IFS=',' read -r -a ROOT_LABELS <<< "${ROOT_LABELS_CSV}"
IFS=',' read -r -a SYSTEMS <<< "${SYSTEMS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IFS=',' read -r -a PERIODS <<< "${REENCODE_PERIODS}"
IFS=',' read -r -a FREEZE_MODES <<< "${ROUTE_FREEZE_MODES}"

JOB_IDS=()
SKIPPED_SLUGS=()
for root_label in "${ROOT_LABELS[@]}"; do
  [[ -z "${root_label}" ]] && continue
  for system_key in "${SYSTEMS[@]}"; do
    [[ -z "${system_key}" ]] && continue
    for seed in "${SEEDS[@]}"; do
      [[ -z "${seed}" ]] && continue
      for period in "${PERIODS[@]}"; do
          [[ -z "${period}" ]] && continue
        for freeze_mode in "${FREEZE_MODES[@]}"; do
          [[ -z "${freeze_mode}" ]] && continue
          slug_key="${root_label}__${system_key}__seed${seed}__p${period}__${freeze_mode}"
          if [[ "${STAGE2_MAP_MODE}" != "family_local_centered" ]]; then
            slug_key="${slug_key}__${STAGE2_MAP_MODE}"
          fi
          slug="$(sanitize_name "${slug_key}")"
          shard_dir="${OUT_DIR}/shards/${slug}"
          mkdir -p "${shard_dir}"
          if [[ "${SKIP_COMPLETED}" == "1" ]] && is_completed_shard "${shard_dir}"; then
            SKIPPED_SLUGS+=("${slug}")
            continue
          fi
          sbatch_args=(
            --job-name="sf_lm_${slug}"
            --partition="${PARTITION}"
            --time="${WALLTIME}"
            --signal="B:USR1@300"
            --cpus-per-task="${CPUS}"
            --mem="${MEM}"
            --output="${LOG_DIR}/${slug}-%A.out"
            --error="${LOG_DIR}/${slug}-%A.err"
          )
          if [[ -n "${WORKER_DEPENDENCY}" ]]; then
            sbatch_args=(--dependency="${WORKER_DEPENDENCY}" "${sbatch_args[@]}")
          fi
          if [[ -n "${GRES}" ]]; then
            sbatch_args+=(--gres="${GRES}")
          fi
          job_id=$(
            ROWS_CSVS="${ROWS_CSVS}" \
            OUT_DIR="${shard_dir}" \
            ROOT_LABELS_CSV="${root_label}" \
            SYSTEMS_CSV="${system_key}" \
            SEEDS_CSV="${seed}" \
            REENCODE_PERIODS="${period}" \
            ROUTE_FREEZE_MODES="${freeze_mode}" \
            TRAIN_STEPS="${TRAIN_STEPS}" \
            HORIZONS="${HORIZONS}" \
            SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
            FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
            STAGE2_MAP_MODE="${STAGE2_MAP_MODE}" \
            DEVICE="${DEVICE}" \
            LABEL_MODE="${LABEL_MODE}" \
            MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-0}" \
            RESUME_FROM_OUTPUT_DIRS="${RESUME_FROM_OUTPUT_DIRS}" \
              sbatch "${sbatch_args[@]}" \
                scripts/run_support_family_local_maps_stage2.sh | awk '{print $4}'
          )
          JOB_IDS+=("${job_id}")
        done
      done
    done
  done
done

JOB_DEPENDENCY="$(IFS=:; echo "${JOB_IDS[*]}")"
merge_sbatch_args=(
  --job-name="sf_lm_merge"
  --partition="${MERGE_PARTITION}"
  --time="${MERGE_WALLTIME}"
  --mem="${MERGE_MEM}"
  --output="${LOG_DIR}/merge-%A.out"
  --error="${LOG_DIR}/merge-%A.err"
)
if [[ -n "${JOB_DEPENDENCY}" ]]; then
  merge_sbatch_args=(--dependency="afterok:${JOB_DEPENDENCY}" "${merge_sbatch_args[@]}")
fi
MERGE_JOB_ID=$(
  SHARDS_DIR="${OUT_DIR}/shards" \
  OUT_DIR="${OUT_DIR}" \
    sbatch "${merge_sbatch_args[@]}" \
      scripts/merge_support_family_local_maps_stage2_shards.sh | awk '{print $4}'
)

AGG_JOB_ID=""
if [[ "${AGGREGATE}" == "1" ]]; then
  AGG_JOB_ID=$(
    sbatch \
      --dependency="afterok:${MERGE_JOB_ID}" \
      --job-name="sf_lm_agg" \
      --partition="${AGG_PARTITION}" \
      --time="${AGG_WALLTIME}" \
      --mem="${AGG_MEM}" \
      --output="${LOG_DIR}/aggregation-%A.out" \
      --error="${LOG_DIR}/aggregation-%A.err" \
      --wrap="cd '${ROOT_DIR}' && . .venv/bin/activate && uv run python tools/analyze_routed_forecasting_mse.py --multibasin_routed_csv '${OUT_DIR}/self_routed_forecasting_rows.csv' --dysts_routed_csv '${OUT_DIR}/self_routed_forecasting_rows.csv' --output_dir '${OUT_DIR}/aggregation' --datasets '${AGG_DATASETS}' --support_definition '${SUPPORT_DEFINITION}'" | awk '{print $4}'
  )
fi

mkdir -p "$(dirname "${QUEUE_MANIFEST_JSON}")"
cat > "${QUEUE_MANIFEST_JSON}" <<EOF
{
  "rows_csvs": [$(printf '"%s",' ${ROWS_CSVS//,/ } | sed 's/,$//')],
  "out_dir": "${OUT_DIR}",
  "root_labels": [$(printf '"%s",' "${ROOT_LABELS[@]}" | sed 's/,$//')],
  "systems": [$(printf '"%s",' "${SYSTEMS[@]}" | sed 's/,$//')],
  "seeds": [$(printf '"%s",' "${SEEDS[@]}" | sed 's/,$//')],
  "reencode_periods": [$(printf '"%s",' "${PERIODS[@]}" | sed 's/,$//')],
  "route_freeze_modes": [$(printf '"%s",' "${FREEZE_MODES[@]}" | sed 's/,$//')],
  "train_steps": "${TRAIN_STEPS}",
  "horizons": "${HORIZONS}",
  "support_definition": "${SUPPORT_DEFINITION}",
  "family_jaccard_threshold": "${FAMILY_JACCARD_THRESHOLD}",
  "stage2_map_mode": "${STAGE2_MAP_MODE}",
  "partition": "${PARTITION}",
  "merge_partition": "${MERGE_PARTITION}",
  "aggregation_partition": "${AGG_PARTITION}",
  "aggregation_datasets": "${AGG_DATASETS}",
  "walltime": "${WALLTIME}",
  "cpus": "${CPUS}",
  "mem": "${MEM}",
  "device": "${DEVICE}",
  "gres": "${GRES}",
  "label_mode": "${LABEL_MODE}",
  "resume_from_output_dirs": [$(printf '"%s",' ${RESUME_FROM_OUTPUT_DIRS//,/ } | sed 's/,$//')],
  "worker_dependency": "${WORKER_DEPENDENCY}",
  "skip_completed": "${SKIP_COMPLETED}",
  "skipped_count": "${#SKIPPED_SLUGS[@]}",
  "submitted_count": "${#JOB_IDS[@]}",
  "skipped_slugs": [$(printf '"%s",' "${SKIPPED_SLUGS[@]}" | sed 's/,$//')],
  "job_ids": [$(printf '"%s",' "${JOB_IDS[@]}" | sed 's/,$//')],
  "merge_job_id": "${MERGE_JOB_ID}",
  "aggregation_job_id": "${AGG_JOB_ID}",
  "log_dir": "${LOG_DIR}"
}
EOF

printf 'JOB_IDS_CSV=%q\n' "$(IFS=,; echo "${JOB_IDS[*]}")"
printf 'MERGE_JOB_ID=%q\n' "${MERGE_JOB_ID}"
printf 'AGG_JOB_ID=%q\n' "${AGG_JOB_ID}"
printf 'SKIPPED_COUNT=%q\n' "${#SKIPPED_SLUGS[@]}"
printf 'LOG_DIR=%q\n' "${LOG_DIR}"
printf 'QUEUE_MANIFEST_JSON=%q\n' "${QUEUE_MANIFEST_JSON}"
