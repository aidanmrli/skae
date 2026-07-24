#!/usr/bin/env bash
#SBATCH --job-name=ac-periodic-v3-gpu
#SBATCH --output=/network/scratch/l/lia/skae/ac-periodic-v3-gpu-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/ac-periodic-v3-gpu-%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G

set -euo pipefail
umask 077

PROJECT_DIR="/home/mila/l/lia/skae"
PACKAGE_DIR="${PROJECT_DIR}/experiments/neurips_2026/allen_cahn_periodic_reencoding_v3"
CARD_PATH="${PACKAGE_DIR}/prediction_card.json"
SOURCE_MANIFEST="${PACKAGE_DIR}/source_manifest.sha256"
OUTPUT_ROOT="/network/scratch/l/lia/skae/allen_cahn_periodic_reencoding_confirmation_20260721_v3"
SMOKE_ROOT="/network/scratch/l/lia/skae/allen_cahn_periodic_reencoding_smoke_20260721_v3"
SMOKE_RECEIPT="${SMOKE_ROOT}/smoke_receipt.json"
UUID_PROBE="${SMOKE_ROOT}/lineage_uuid_probe.json"
EXPECTED_CARD_SHA256="${EXPECTED_CARD_SHA256:?required}"
EXPECTED_SOURCE_MANIFEST_SHA256="${EXPECTED_SOURCE_MANIFEST_SHA256:?required}"
EXPECTED_SMOKE_RECEIPT_SHA256="${EXPECTED_SMOKE_RECEIPT_SHA256:?required}"
EXPECTED_UUID_PROBE_SHA256="${EXPECTED_UUID_PROBE_SHA256:?required}"
AUTHORIZATION="${SKAE_PERIODIC_REENCODING_V3_AUTHORIZE:-}"
EXPECTED_AUTHORIZATION="YES_I_AUTHORIZE_FROZEN_PERIODIC_REENCODING_V3"

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
if [[ ! -f "${SMOKE_RECEIPT}" ]] || \
   [[ "$(sha256sum "${SMOKE_RECEIPT}" | awk '{print $1}')" != "${EXPECTED_SMOKE_RECEIPT_SHA256}" ]] || \
   [[ "$(jq -r '.status' "${SMOKE_RECEIPT}")" != "passed_outcome_free_gpu_smoke" ]] || \
   [[ "$(jq -r '.card_sha256' "${SMOKE_RECEIPT}")" != "${EXPECTED_CARD_SHA256}" ]] || \
   [[ "$(jq -r '.source_manifest_sha256' "${SMOKE_RECEIPT}")" != "${EXPECTED_SOURCE_MANIFEST_SHA256}" ]]; then
  echo "The scientific job requires the current passing smoke receipt." >&2
  exit 1
fi
if [[ ! -f "${UUID_PROBE}" ]] || \
   [[ "$(sha256sum "${UUID_PROBE}" | awk '{print $1}')" != "${EXPECTED_UUID_PROBE_SHA256}" ]] || \
   ! jq -e \
     --arg card "${EXPECTED_CARD_SHA256}" \
     --arg source "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
     --arg smoke_job "$(jq -r '.slurm_job_id' "${SMOKE_RECEIPT}")" \
     '(.status == "passed_real_cuda_uuid_crosscheck_strict_json") and
      (.card_sha256 == $card) and
      (.source_manifest_sha256 == $source) and
      (.slurm_job_id == $smoke_job) and
      (.raw_uuid_type == "_CUuuid") and
      (.pytorch_uuid_canonical | test("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")) and
      (.gpu_uuid == ("GPU-" + .pytorch_uuid_canonical)) and
      (.nvidia_smi_uuid_canonical == .gpu_uuid) and
      (.nvidia_smi_visible_gpu_count == 1) and
      (.uuid_sources_match == true) and
      (.scientific_outcomes_accessed == false)' \
     "${UUID_PROBE}" >/dev/null; then
  echo "The scientific job requires the current passing UUID probe." >&2
  exit 1
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" == *,* ]]; then
  echo "Exactly one visible GPU is required." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
sha256sum --check --strict --status "${SOURCE_MANIFEST}"
VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l)"
if [[ "${VISIBLE_GPU_COUNT}" -ne 1 ]]; then
  echo "The scientific job must see exactly one allocated GPU." >&2
  exit 1
fi
export SKAE_GIT_COMMIT
SKAE_GIT_COMMIT="$(git rev-parse HEAD)"

RAW_TELEMETRY="$(mktemp /network/scratch/l/lia/skae/ac-periodic-v3-gpu-${SLURM_JOB_ID}-XXXXXX.csv)"
nvidia-smi \
  --query-gpu=timestamp,uuid,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits \
  --loop=1 > "${RAW_TELEMETRY}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

uv run python -m experiments.neurips_2026.allen_cahn_periodic_reencoding_v3.run \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --smoke-receipt "${SMOKE_RECEIPT}" \
  --expected-smoke-receipt-sha256 "${EXPECTED_SMOKE_RECEIPT_SHA256}" \
  --output-root "${OUTPUT_ROOT}"

sleep 2
kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT

uv run python -m experiments.neurips_2026.allen_cahn_periodic_reencoding.telemetry \
  --root "${OUTPUT_ROOT}" \
  --card "${CARD_PATH}" \
  --expected-card-sha256 "${EXPECTED_CARD_SHA256}" \
  --source-manifest "${SOURCE_MANIFEST}" \
  --expected-source-manifest-sha256 "${EXPECTED_SOURCE_MANIFEST_SHA256}" \
  --raw-telemetry "${RAW_TELEMETRY}"
