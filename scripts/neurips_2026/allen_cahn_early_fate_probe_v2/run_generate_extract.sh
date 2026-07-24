#!/usr/bin/env bash
#SBATCH --job-name=ac-fate-v2-gpu
#SBATCH --output=/network/scratch/l/lia/skae/ac-fate-v2-gpu-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-fate-v2-gpu-%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

PROJECT_DIR="/home/mila/l/lia/skae"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_early_fate_probe_20260721_v2"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_TASK_MANIFEST_SHA256="${EXPECTED_TASK_MANIFEST_SHA256:?required}"
AUTHORIZATION="${AUTHORIZE_ALLEN_EARLY_FATE_V2:?required}"
EXPECTED_AUTHORIZATION="root-redteam-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}:${EXPECTED_TASK_MANIFEST_SHA256}"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Independent-redteam launch authorization does not match exact roots." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing pre-existing V2 output root." >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
  echo "Exactly one visible GPU is required." >&2
  exit 1
fi

RAW_TELEMETRY="$(mktemp /network/scratch/l/lia/skae/ac-fate-v2-${SLURM_JOB_ID}-XXXXXX.csv)"
nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits --loop=1 > "${RAW_TELEMETRY}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

cd "${PROJECT_DIR}"
uv run python -m experiments.neurips_2026.allen_cahn_early_fate_probe_v2.generate_extract \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected-task-manifest-sha256 "${EXPECTED_TASK_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}"

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT
UNVERIFIED_TRACE="${OUTPUT_ROOT}/field_only/raw_telemetry_unverified.csv"
if [[ -e "${UNVERIFIED_TRACE}" ]]; then
  echo "Refusing pre-existing unverified telemetry trace." >&2
  exit 1
fi
mv "${RAW_TELEMETRY}" "${UNVERIFIED_TRACE}"
