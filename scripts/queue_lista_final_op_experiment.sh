#!/usr/bin/env bash
set -euo pipefail

# Submit all LISTA final-op experiment phases plus final collection job.

BASE_ROOT="${BASE_ROOT:-/network/scratch/l/lia/skae/lista_final_op_experiment}"
DEVICE="${DEVICE:-cuda}"

submit_job() {
  local cmd_output
  cmd_output="$("$@")"
  echo "${cmd_output}" >&2
  echo "${cmd_output}" | awk '{print $4}'
}

echo "Queueing LISTA final-op experiment"
echo "Base root: ${BASE_ROOT}"
echo "Device: ${DEVICE}"

JOB0=$(submit_job sbatch \
  --export=ALL,BASE_OUT="${BASE_ROOT}/phase0_smoke",DEVICE="${DEVICE}" \
  scripts/sweep_lista_final_op_phase0_smoke.sh)

JOB1=$(submit_job sbatch \
  --export=ALL,BASE_OUT="${BASE_ROOT}/phase1_core",DEVICE="${DEVICE}" \
  scripts/sweep_lista_final_op_phase1_core.sh)

JOB2=$(submit_job sbatch \
  --export=ALL,BASE_OUT="${BASE_ROOT}/phase2_sparsity_match",DEVICE="${DEVICE}" \
  scripts/sweep_lista_final_op_phase2_sparsity_match.sh)

JOB3=$(submit_job sbatch \
  --export=ALL,BASE_OUT="${BASE_ROOT}/phase3_structured_transfer",DEVICE="${DEVICE}" \
  scripts/sweep_lista_final_op_phase3_structured_transfer.sh)

COLLECT_DEP="afterany:${JOB0}:${JOB1}:${JOB2}:${JOB3}"
JOBC=$(submit_job sbatch \
  --dependency="${COLLECT_DEP}" \
  --export=ALL,BASE_DIR="${BASE_ROOT}",OUTPUT_DIR="${BASE_ROOT}/results" \
  scripts/collect_lista_final_op_results.sh)

echo ""
echo "Queued jobs:"
echo "  Phase 0 (smoke):            ${JOB0}"
echo "  Phase 1 (core):             ${JOB1}"
echo "  Phase 2 (sparsity match):   ${JOB2}"
echo "  Phase 3 (structured):       ${JOB3}"
echo "  Collect (dependency):       ${JOBC}"
