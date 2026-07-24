#!/usr/bin/env bash
#SBATCH --job-name=ac-periodic-v5-smoke
#SBATCH --output=/network/scratch/l/lia/skae/ac-periodic-v5-smoke-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-periodic-v5-smoke-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

set -euo pipefail
umask 077

PROJECT_DIR="/home/mila/l/lia/skae"
PACKAGE_DIR="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_periodic_reencoding_v5"
CARD_PATH="${PACKAGE_DIR}/prediction_card.json"
SOURCE_MANIFEST="${PACKAGE_DIR}/source_manifest.sha256"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_periodic_reencoding_smoke_20260721_v5"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
AUTHORIZATION="${SKAE_PERIODIC_V5_SMOKE_AUTHORIZE:-}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this outcome-free smoke with sbatch." >&2
  exit 1
fi
if [[ "${AUTHORIZATION}" != "YES_I_AUTHORIZE_OUTCOME_FREE_PERIODIC_SMOKE_V5" ]]; then
  echo "Outcome-free smoke authorization is absent." >&2
  exit 1
fi
if [[ "$(sha256sum "${CARD_PATH}" | awk '{print $1}')" != "${EXPECTED_CARD_SHA256}" ]] || \
   [[ "$(sha256sum "${SOURCE_MANIFEST}" | awk '{print $1}')" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "Smoke card or source manifest differs from the frozen hashes." >&2
  exit 1
fi
if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing pre-existing smoke root: ${OUTPUT_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
sha256sum --check --strict --status "${SOURCE_MANIFEST}"
CUDA_VISIBLE_DEVICES="" uv run python -m \
  experiments.neurips_2026.allen_cahn_periodic_reencoding.source_lock \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}"

RAW_TELEMETRY="$(mktemp /network/scratch/l/lia/skae/ac-periodic-v5-smoke-${SLURM_JOB_ID}-XXXXXX.csv)"
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

uv run python -m experiments.neurips_2026.allen_cahn_periodic_reencoding.smoke \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}"

uv run python -m \
  experiments.neurips_2026.allen_cahn_periodic_reencoding_v3.uuid_probe \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}"

cleanup_monitoring
trap - EXIT

uv run python -m experiments.neurips_2026.allen_cahn_periodic_reencoding_v5.smoke_audit \
  --root "${OUTPUT_ROOT}" \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --raw-telemetry "${RAW_TELEMETRY}"

SMOKE_RECEIPT_SHA256="$(sha256sum "${OUTPUT_ROOT}/smoke_receipt.json" | awk '{print $1}')"
UUID_PROBE_SHA256="$(sha256sum "${OUTPUT_ROOT}/lineage_uuid_probe.json" | awk '{print $1}')"
uv run python -m \
  experiments.neurips_2026.allen_cahn_periodic_reencoding_v5.guard smoke \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --output-root "${OUTPUT_ROOT}" \
  --expected-smoke-receipt-sha256 "${SMOKE_RECEIPT_SHA256}" \
  --expected-uuid-probe-sha256 "${UUID_PROBE_SHA256}"
printf 'smoke_receipt_sha256=%s\nuuid_probe_sha256=%s\n' \
  "${SMOKE_RECEIPT_SHA256}" "${UUID_PROBE_SHA256}"
