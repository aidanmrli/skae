#!/usr/bin/env bash
#SBATCH --job-name=ac-periodic-v5-gpu
#SBATCH --output=/network/scratch/l/lia/skae/ac-periodic-v5-gpu-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-periodic-v5-gpu-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail
umask 077

PROJECT_DIR="/home/mila/l/lia/skae"
PACKAGE_DIR="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_periodic_reencoding_v5"
CARD_PATH="${PACKAGE_DIR}/prediction_card.json"
SOURCE_MANIFEST="${PACKAGE_DIR}/source_manifest.sha256"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_periodic_reencoding_confirmation_20260721_v5"
SMOKE_ROOT="/network/scratch/l/lia/skae/allen_cahn_periodic_reencoding_smoke_20260721_v5"
SMOKE_RECEIPT="${SMOKE_ROOT}/smoke_receipt.json"
UUID_PROBE="${SMOKE_ROOT}/lineage_uuid_probe.json"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_SMOKE_RECEIPT_SHA256="${EXPECTED_SMOKE_RECEIPT_SHA256:?required}"
EXPECTED_UUID_PROBE_SHA256="${EXPECTED_UUID_PROBE_SHA256:?required}"
AUTHORIZATION="${SKAE_PERIODIC_REENCODING_V5_AUTHORIZE:-}"
EXPECTED_AUTHORIZATION="YES_I_AUTHORIZE_FROZEN_PERIODIC_REENCODING_V5"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this scientific launcher with sbatch." >&2
  exit 1
fi
if [[ "${AUTHORIZATION}" != "${EXPECTED_AUTHORIZATION}" ]]; then
  echo "The frozen periodic-reencoding launch remains dormant." >&2
  exit 1
fi
if [[ "$(sha256sum "${CARD_PATH}" | awk '{print $1}')" != "${EXPECTED_CARD_SHA256}" ]] || \
   [[ "$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "The card or source manifest differs from the authorized freeze." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing pre-existing scientific output root: ${OUTPUT_ROOT}" >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
  echo "Exactly one visible GPU is required." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
sha256sum --check --strict --status "${SOURCE_MANIFEST}"
uv run python -m \
  experiments.neurips_2026.allen_cahn_periodic_reencoding_v5.guard smoke \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --output-root "${SMOKE_ROOT}" \
  --expected-smoke-receipt-sha256 "${EXPECTED_SMOKE_RECEIPT_SHA256}" \
  --expected-uuid-probe-sha256 "${EXPECTED_UUID_PROBE_SHA256}"
VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)"
if [[ "${VISIBLE_GPU_COUNT}" -ne 1 ]]; then
  echo "The scientific job must see exactly one allocated GPU." >&2
  exit 1
fi
export SKAE_GIT_COMMIT
SKAE_GIT_COMMIT="$(git rev-parse HEAD)"

RAW_TELEMETRY="$(mktemp /network/scratch/l/lia/skae/ac-periodic-v5-gpu-${SLURM_JOB_ID}-XXXXXX.csv)"
nvidia-smi \
  --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits \
  --loop=60 > "${RAW_TELEMETRY}" &
MONITOR_PID=$!
cleanup_monitoring() {
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup_monitoring EXIT

uv run python -m experiments.neurips_2026.allen_cahn_periodic_reencoding_v3.run \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --smoke-receipt "${SMOKE_RECEIPT}" \
  --expected-smoke-receipt-sha256 "${EXPECTED_SMOKE_RECEIPT_SHA256}" \
  --output-root "${OUTPUT_ROOT}"

cleanup_monitoring
trap - EXIT

uv run python -m experiments.neurips_2026.allen_cahn_periodic_reencoding_v5.telemetry \
  --root "${OUTPUT_ROOT}" \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --raw-telemetry "${RAW_TELEMETRY}"
