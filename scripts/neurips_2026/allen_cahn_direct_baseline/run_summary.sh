#!/usr/bin/env bash
#SBATCH --job-name=ac-direct-packet
#SBATCH --output=/network/scratch/l/lia/skae/allen_cahn_direct_baseline_v1_20260721/logs/summary-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/allen_cahn_direct_baseline_v1_20260721/logs/summary-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

REPO_ROOT="/home/mila/l/lia/skae"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_direct_baseline_v1_20260721"
TASK_LOCK="${REPO_ROOT}/experiments/neurips_2026/allen_cahn_direct_baseline/task_lock.json"
TASK_LOCK_SHA256="${TASK_LOCK_SHA256:?TASK_LOCK_SHA256 must be exported by the authenticated launcher}"
SUMMARY="${OUTPUT_ROOT}/summary/decision.json"

if [[ -e "${SUMMARY}" ]]; then
  echo "Refusing to overwrite ${SUMMARY}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_ROOT}/logs" "$(dirname "${SUMMARY}")"
cd "${REPO_ROOT}"
echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID}"
sha256sum "${TASK_LOCK}"

uv run python -m experiments.neurips_2026.allen_cahn_direct_baseline.report \
  --task-lock "${TASK_LOCK}" \
  --expected-task-lock-sha256 "${TASK_LOCK_SHA256}" \
  --training-root "${OUTPUT_ROOT}/training" \
  --evaluation-root "${OUTPUT_ROOT}/evaluation" \
  --output "${SUMMARY}"

sha256sum "${SUMMARY}"
