#!/usr/bin/env bash
#SBATCH --job-name=ac-newic-repl
#SBATCH --output=/network/scratch/l/lia/skae/ac-newic-repl-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-newic-repl-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail

PROJECT_DIR="/home/mila/l/lia/skae"
CARD_PATH="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_forecast_replication/prediction_card.json"
SOURCE_MANIFEST="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_forecast_replication/source_manifest.sha256"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_forecast_replication_v1_20260720"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?EXPECTED_CARD_SHA256 is required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?EXPECTED_SOURCE_MANIFEST_SHA256 is required}"
AUTHORIZATION="${AUTHORIZE_ALLEN_FORECAST_REPLICATION:?explicit root authorization is required}"
EXPECTED_AUTHORIZATION="root-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Scientific launch authorization token does not match the exact roots." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing pre-existing output root ${OUTPUT_ROOT}." >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
  echo "Exactly one visible GPU is required." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
export SKAE_GIT_COMMIT
SKAE_GIT_COMMIT="$(git rev-parse HEAD)"
uv run python -m experiments.neurips_2026.allen_cahn_forecast_replication.source_lock \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"

RAW_TELEMETRY="$(mktemp /network/scratch/l/lia/skae/ac-newic-gpu-${SLURM_JOB_ID}-XXXXXX.csv)"
nvidia-smi \
  --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv \
  --loop=1 > "${RAW_TELEMETRY}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

uv run python -m experiments.neurips_2026.allen_cahn_forecast_replication.run \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}"

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT

uv run python -m experiments.neurips_2026.allen_cahn_forecast_replication.telemetry \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}" \
  --raw-telemetry "${RAW_TELEMETRY}"
