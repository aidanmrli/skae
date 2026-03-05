#!/bin/bash
set -euo pipefail

SYSTEMS_FILE="${SYSTEMS_FILE:-scripts/dysts_cache_systems.txt}"
CACHE_DIR="${CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS:-2}"
PROFILES="${PROFILES:-smoke full}"
SPLITS="${SPLITS:-train val test}"

if [[ ! -f "${SYSTEMS_FILE}" ]]; then
  echo "Missing SYSTEMS_FILE=${SYSTEMS_FILE}"
  exit 1
fi

NUM_SYSTEMS=$(sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' "${SYSTEMS_FILE}" | awk 'NF' | wc -l)
NUM_PROFILES=$(wc -w <<< "${PROFILES}")
NUM_SPLITS=$(wc -w <<< "${SPLITS}")
TOTAL=$((NUM_SYSTEMS * NUM_PROFILES * NUM_SPLITS))

if (( TOTAL <= 0 )); then
  echo "Nothing to submit (TOTAL=${TOTAL})."
  exit 1
fi

ARRAY_RANGE="0-$((TOTAL - 1))"
echo "Submitting prebuild matrix: systems=${NUM_SYSTEMS}, profiles=${NUM_PROFILES}, splits=${NUM_SPLITS}, total=${TOTAL}"
echo "Array range: ${ARRAY_RANGE}"

sbatch \
  --array="${ARRAY_RANGE}" \
  --export=ALL,SYSTEMS_FILE="${SYSTEMS_FILE}",CACHE_DIR="${CACHE_DIR}",CACHE_NUM_WORKERS="${CACHE_NUM_WORKERS}",PROFILES="${PROFILES}",SPLITS="${SPLITS}" \
  scripts/prebuild_dysts_cache_matrix.sh

