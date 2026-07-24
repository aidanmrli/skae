#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/mila/l/lia/skae"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_direct_baseline_v1_20260721"
TASK_LOCK="${REPO_ROOT}/experiments/neurips_2026/allen_cahn_direct_baseline/task_lock.json"
EXPECTED_TASK_LOCK_SHA256="${EXPECTED_TASK_LOCK_SHA256:?Set the independently recorded frozen task-lock SHA-256}"
OBSERVED_TASK_LOCK_SHA256="$(sha256sum "${TASK_LOCK}" | awk '{print $1}')"
if [[ "${OBSERVED_TASK_LOCK_SHA256}" != "${EXPECTED_TASK_LOCK_SHA256}" ]]; then
  echo "Task-lock mismatch: ${OBSERVED_TASK_LOCK_SHA256} != ${EXPECTED_TASK_LOCK_SHA256}" >&2
  exit 1
fi
FULL_LAUNCH_AUTHORIZED="$(jq -r '.source_locked_command_graph.full_launch_authorized' "${TASK_LOCK}")"
if [[ "${FULL_LAUNCH_AUTHORIZED}" != "true" ]]; then
  echo "Full scientific launch is not authorized by the frozen task lock" >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}/training" || -e "${OUTPUT_ROOT}/evaluation" || -e "${OUTPUT_ROOT}/summary" ]]; then
  echo "Refusing to launch into an existing scientific output root" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_ROOT}"

TRAIN_JOB="$(sbatch --parsable --export=ALL,TASK_LOCK_SHA256="${EXPECTED_TASK_LOCK_SHA256}" scripts/neurips_2026/allen_cahn_direct_baseline/run_train_array.sh)"
EVAL_JOB="$(sbatch --parsable --dependency=afterok:"${TRAIN_JOB}" --export=ALL,TASK_LOCK_SHA256="${EXPECTED_TASK_LOCK_SHA256}" scripts/neurips_2026/allen_cahn_direct_baseline/run_evaluate_array.sh)"
SUMMARY_JOB="$(sbatch --parsable --dependency=afterok:"${EVAL_JOB}" --export=ALL,TASK_LOCK_SHA256="${EXPECTED_TASK_LOCK_SHA256}" scripts/neurips_2026/allen_cahn_direct_baseline/run_summary.sh)"
printf 'training_job=%s\nevaluation_job=%s\nsummary_job=%s\n' "${TRAIN_JOB}" "${EVAL_JOB}" "${SUMMARY_JOB}"
