#!/usr/bin/env bash
#SBATCH --job-name=ac-fate-v2-tel
#SBATCH --output=/network/scratch/l/lia/skae/ac-fate-v2-tel-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-fate-v2-tel-%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

PROJECT_DIR="/home/mila/l/lia/skae"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_early_fate_probe_20260721_v2"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_TASK_MANIFEST_SHA256="${EXPECTED_TASK_MANIFEST_SHA256:?required}"
EXPECTED_GPU_JOB_ID="${EXPECTED_GPU_JOB_ID:?required}"
AUTHORIZATION="${AUTHORIZE_ALLEN_EARLY_FATE_V2:?required}"
EXPECTED_AUTHORIZATION="root-redteam-approved:${EXPECTED_CARD_SHA256}:${EXPECTED_SOURCE_MANIFEST_SHA256}:${EXPECTED_TASK_MANIFEST_SHA256}"
RAW_TELEMETRY="${OUTPUT_ROOT}/field_only/raw_telemetry_unverified.csv"

if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "Independent-redteam authorization does not match exact roots." >&2
  exit 1
fi
if [[ ! -f "${RAW_TELEMETRY}" ]]; then
  echo "Unverified GPU telemetry trace is missing." >&2
  exit 1
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
  echo "Telemetry authentication must be CPU-only." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
uv run python -m experiments.neurips_2026.allen_cahn_early_fate_probe_v2.telemetry \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --expected-task-manifest-sha256 "${EXPECTED_TASK_MANIFEST_SHA256}" \
  --expected-gpu-slurm-job-id "${EXPECTED_GPU_JOB_ID}" \
  --output-root "${OUTPUT_ROOT}" \
  --raw-telemetry "${RAW_TELEMETRY}"
