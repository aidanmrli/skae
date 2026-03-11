#!/bin/bash

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-kuramoto_mode_support_audit_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-kuramoto_mode_support_audit}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
TASK_TSV="${TASK_TSV:-${TASK_DIR}/kuramoto_mode_support_audit.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${TASK_DIR}/kuramoto_mode_support_audit_manifest.json}"
SUMMARY_DIR="${SUMMARY_DIR:-${BASE_OUT}/summary}"
SOURCE_ROOT="${SOURCE_ROOT:-/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308/kuramoto_dt00625_200k}"

MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-generic_sparse,lista_dense,lista_blockdiag}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4}"
SAMPLING_STRATEGIES_CSV="${SAMPLING_STRATEGIES_CSV:-random,balanced}"
ENV_DT="${ENV_DT:-0.00625}"
RANDOM_NUM_TRAJECTORIES="${RANDOM_NUM_TRAJECTORIES:-256}"
BALANCED_TRAJECTORIES_PER_BASIN="${BALANCED_TRAJECTORIES_PER_BASIN:-16}"
BALANCED_TARGET_RAW_LABELS_CSV="${BALANCED_TARGET_RAW_LABELS_CSV:--2,-1,0,1,2}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-256}"
LONG_ROLLOUT_STEPS="${LONG_ROLLOUT_STEPS:-5000}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.001}"
SUPPORT_MODES_CSV="${SUPPORT_MODES_CSV:-mean,majority,modal}"
THRESHOLD_SWEEP_MODES_CSV="${THRESHOLD_SWEEP_MODES_CSV:-mean,modal}"
THRESHOLDS_CSV="${THRESHOLDS_CSV:-1e-4,5e-4,1e-3,5e-3,1e-2,5e-2,1e-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20000}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${TASK_DIR}" "${SUMMARY_DIR}"

uv run python tools/build_kuramoto_mode_support_audit_tasks.py \
  --phase_label "${PHASE_LABEL}" \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --source_root "${SOURCE_ROOT}" \
  --scratch_root "${BASE_OUT}" \
  --model_variants_csv "${MODEL_VARIANTS_CSV}" \
  --seeds_csv "${SEEDS_CSV}" \
  --sampling_strategies_csv "${SAMPLING_STRATEGIES_CSV}" \
  --env_dt "${ENV_DT}" \
  --random_num_trajectories "${RANDOM_NUM_TRAJECTORIES}" \
  --balanced_trajectories_per_basin "${BALANCED_TRAJECTORIES_PER_BASIN}" \
  --balanced_target_raw_labels_csv="${BALANCED_TARGET_RAW_LABELS_CSV}" \
  --trajectory_length "${TRAJECTORY_LENGTH}" \
  --long_rollout_steps "${LONG_ROLLOUT_STEPS}" \
  --support_threshold "${SUPPORT_THRESHOLD}" \
  --support_modes_csv "${SUPPORT_MODES_CSV}" \
  --threshold_sweep_modes_csv "${THRESHOLD_SWEEP_MODES_CSV}" \
  --thresholds_csv "${THRESHOLDS_CSV}" \
  --max_attempts "${MAX_ATTEMPTS}" \
  --device "${DEVICE}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

ARRAY_JOB_ID=$(sbatch --array=0-$((TASK_COUNT - 1)) scripts/run_kuramoto_mode_support_audit_array.sh "${TASK_TSV}" | awk '{print $4}')
COLLECT_JOB_ID=$(TASK_TSV="${TASK_TSV}" SUMMARY_DIR="${SUMMARY_DIR}" sbatch --dependency=afterany:${ARRAY_JOB_ID} scripts/run_kuramoto_mode_support_audit_collect.sh | awk '{print $4}')

echo "Queued Kuramoto mode-support audit."
echo "Array job: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Task TSV: ${TASK_TSV}"
echo "Summary dir: ${SUMMARY_DIR}"
