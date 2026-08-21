#!/usr/bin/env bash
# CPU-only validation allocation for the utilization pilot contracts.

#SBATCH --job-name=skae-util-pilot-tests
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@60
# Slurm opens these before the script starts; the stable account scratch log
# directory must exist before submission.
#SBATCH --output=/network/scratch/l/lia/skae/slurm_logs/%x-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/slurm_logs/%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export UV_PROJECT="${SKAE_UV_PROJECT:-/home/mila/l/lia/skae}"
export UV_NO_SYNC=1
JOB_ID="${SLURM_JOB_ID:?missing SLURM_JOB_ID}"
SCRATCH_BASE="${SCRATCH:?SCRATCH must be set by the allocation}"
TEST_ROOT="${SCRATCH_BASE}/skae/utilization_pilot_tests/${JOB_ID}"
ATTEMPT="${SLURM_RESTART_COUNT:-0}"
mkdir -p "${TEST_ROOT}"

if [[ -f "${TEST_ROOT}/final.json" ]]; then
  set +e
  uv run python -m experiments.neurips_2026.utilization_pilot.test_receipt \
    --repo-root "${ROOT_DIR}" --output-root "${TEST_ROOT}" --phase validate-final
  FINAL_STATUS=$?
  set -e
  if (( FINAL_STATUS == 0 )); then
    exit 0
  fi
fi

uv run python -m experiments.neurips_2026.utilization_pilot.test_receipt \
  --repo-root "${ROOT_DIR}" --output-root "${TEST_ROOT}" \
  --phase started --attempt "${ATTEMPT}"

TERM_FORWARDED=0
ACTIVE_PID=""
ACTIVE_PHASE=""
TERM_TRIGGER_PID=""
on_term() {
  if (( TERM_FORWARDED != 0 )); then
    exit 0
  fi
  TERM_FORWARDED=1
  set +e
  PHASE="${ACTIVE_PHASE}"
  CHILD_STATUS=1
  if [[ -n "${ACTIVE_PID}" ]] && kill -0 "${ACTIVE_PID}" 2>/dev/null; then
    kill -TERM "${ACTIVE_PID}" 2>/dev/null || true
    wait "${ACTIVE_PID}"
    CHILD_STATUS=$?
    ACTIVE_PID=""
  fi
  ACTIVE_PHASE=""
  if [[ "${PHASE}" == "hold" ]]; then
    if (( CHILD_STATUS == 75 )); then
      uv run python -m experiments.neurips_2026.utilization_pilot.test_receipt \
        --repo-root "${ROOT_DIR}" --output-root "${TEST_ROOT}" --phase validate-hold
      RECORD_STATUS=$?
    else
      RECORD_STATUS=1
    fi
    if (( RECORD_STATUS == 0 )); then
      uv run python -m experiments.neurips_2026.utilization_pilot.test_receipt \
        --repo-root "${ROOT_DIR}" --output-root "${TEST_ROOT}" \
        --phase forced-term --attempt "${ATTEMPT}"
      RECORD_STATUS=$?
    fi
  else
    uv run python -m experiments.neurips_2026.utilization_pilot.test_receipt \
      --repo-root "${ROOT_DIR}" --output-root "${TEST_ROOT}" \
      --phase interrupted --attempt "${ATTEMPT}"
    RECORD_STATUS=$?
  fi
  if (( RECORD_STATUS == 0 )); then
    scontrol requeue "${JOB_ID}"
    REQUEUE_STATUS=$?
  else
    REQUEUE_STATUS=1
  fi
  set -e
  if (( RECORD_STATUS == 0 && REQUEUE_STATUS == 0 )); then
    exit 0
  fi
  exit 1
}
trap on_term TERM INT

if [[ "${ATTEMPT}" == "0" && ! -f "${TEST_ROOT}/forced_term.json" ]]; then
  set +e
  srun --exact --nodes=1 --ntasks=1 --cpus-per-task=1 \
    uv run python -m experiments.neurips_2026.utilization_pilot.test_receipt \
    --repo-root "${ROOT_DIR}" --output-root "${TEST_ROOT}" \
    --phase hold --attempt "${ATTEMPT}" --seconds 120 &
  ACTIVE_PID=$!
  ACTIVE_PHASE="hold"
  (
    sleep 2
    kill -TERM "$$" 2>/dev/null || true
  ) &
  TERM_TRIGGER_PID=$!
  wait "${ACTIVE_PID}"
  HOLD_STATUS=$?
  ACTIVE_PID=""
  ACTIVE_PHASE=""
  kill "${TERM_TRIGGER_PID}" 2>/dev/null || true
  set -e
  if (( HOLD_STATUS != 75 )); then
    echo "hold child exited without the batch-shell TERM path: ${HOLD_STATUS}" >&2
    exit 1
  fi
fi

uv run python -m experiments.neurips_2026.utilization_pilot.test_receipt \
  --repo-root "${ROOT_DIR}" --output-root "${TEST_ROOT}" \
  --phase resume --attempt "${ATTEMPT}"

set +e
srun --exact --nodes=1 --ntasks=1 --cpus-per-task=4 \
  uv run pytest tests/test_utilization_pilot.py &
ACTIVE_PID=$!
ACTIVE_PHASE="pytest"
wait "${ACTIVE_PID}"
TEST_STATUS=$?
ACTIVE_PID=""
ACTIVE_PHASE=""
set -e
if (( TEST_STATUS == 0 )); then
  uv run python -m experiments.neurips_2026.utilization_pilot.test_receipt \
    --repo-root "${ROOT_DIR}" --output-root "${TEST_ROOT}" \
    --phase complete --attempt "${ATTEMPT}"
  uv run python -m experiments.neurips_2026.utilization_pilot.test_receipt \
    --repo-root "${ROOT_DIR}" --output-root "${TEST_ROOT}" --phase validate-final
else
  uv run python -m experiments.neurips_2026.utilization_pilot.test_receipt \
    --repo-root "${ROOT_DIR}" --output-root "${TEST_ROOT}" \
    --phase failed --attempt "${ATTEMPT}" || true
fi
exit "${TEST_STATUS}"
