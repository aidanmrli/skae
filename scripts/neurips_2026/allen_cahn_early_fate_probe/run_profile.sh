#!/usr/bin/env bash
#SBATCH --job-name=ac-fate-profile
#SBATCH --output=/network/scratch/l/lia/skae/ac-fate-profile-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-fate-profile-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G

set -euo pipefail

PROJECT_DIR="/home/mila/l/lia/skae"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_early_fate_probe_20260721_v1"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_TASK_MANIFEST_SHA256="${EXPECTED_TASK_MANIFEST_SHA256:?required}"
AUTHORIZATION="${AUTHORIZE_ALLEN_EARLY_FATE:?required}"
EXPECTED_AUTHORIZATION="root-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}:${EXPECTED_TASK_MANIFEST_SHA256}"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Launch authorization does not match exact roots." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing pre-existing output root." >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
  echo "Exactly one visible GPU is required." >&2
  exit 1
fi

RAW_TELEMETRY="$(mktemp /network/scratch/l/lia/skae/ac-fate-profile-${SLURM_JOB_ID}-XXXXXX.csv)"
nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits --loop=1 > "${RAW_TELEMETRY}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

cd "${PROJECT_DIR}"
uv run python -m experiments.neurips_2026.allen_cahn_early_fate_probe.profile \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected-task-manifest-sha256 "${EXPECTED_TASK_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}"

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT
uv run python -m experiments.neurips_2026.allen_cahn_early_fate_probe.telemetry profile \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected-task-manifest-sha256 "${EXPECTED_TASK_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}" \
  --raw-telemetry "${RAW_TELEMETRY}"
