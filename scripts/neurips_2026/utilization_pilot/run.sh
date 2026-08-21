#!/usr/bin/env bash
# Diagnostic-only exact-shape utilization pilot.
#
# Submit with:
#   sbatch --export=ALL,PILOT_LABEL=base,PILOT_VARIANT=base \
#     scripts/neurips_2026/utilization_pilot/run.sh
#
# A candidate implementation uses the same fixed command and shape; only the
# non-scientific label/variant environment values differ.

#SBATCH --job-name=skae-util-pilot
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mem=16G
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
if [[ -f scripts/common/cluster_env.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/common/cluster_env.sh
fi
if command -v module >/dev/null 2>&1; then
  module load cuda/12.6.0 >/dev/null 2>&1 || true
fi
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

JOB_ID="${SLURM_JOB_ID:?missing SLURM_JOB_ID}"
SCRATCH_BASE="${SKAE_SCRATCH_ROOT:-/network/scratch/l/lia/skae}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRATCH_BASE}/utilization_pilot/${PILOT_LABEL:-base}/${JOB_ID}}"
mkdir -p "${OUTPUT_ROOT}"

command -v uv >/dev/null
command -v nvidia-smi >/dev/null

PILOT_PID=""
TERM_FORWARDED=0
forward_term() {
  if (( TERM_FORWARDED == 0 )); then
    TERM_FORWARDED=1
    if [[ -n "${PILOT_PID}" ]] && kill -0 "${PILOT_PID}" 2>/dev/null; then
      kill -TERM "${PILOT_PID}" 2>/dev/null || true
    fi
  fi
}
trap forward_term TERM INT

set +e
srun --exact --nodes=1 --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-6}" \
  uv run python -m experiments.neurips_2026.utilization_pilot.run_pilot \
  --repo-root "${ROOT_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --label "${PILOT_LABEL:-base}" \
  --variant "${PILOT_VARIANT:-base}" \
  --restart-count "${SLURM_RESTART_COUNT:-0}" &
PILOT_PID=$!
wait "${PILOT_PID}"
PILOT_STATUS=$?
set -e
PILOT_PID=""

if (( PILOT_STATUS == 75 || TERM_FORWARDED == 1 )); then
  # The Python runner has already atomically written progress.json and the
  # attempt receipt before requesting continuation.
  scontrol requeue "${JOB_ID}"
  exit 0
fi
exit "${PILOT_STATUS}"
