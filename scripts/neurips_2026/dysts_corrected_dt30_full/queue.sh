#!/usr/bin/env bash
#SBATCH --job-name=queue-dysts-dt30-v3
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
ROOT_DIR="${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/common/cluster_env.sh"
cd "${PROJECT_DIR}"

PACKAGE="experiments/neurips_2026/dysts_corrected_dt30_full"
SOURCE_MANIFEST="${PROJECT_DIR}/${PACKAGE}/source_manifest.sha256"
RESULTS_DIR="${SKAE_SCRATCH_ROOT}/results/dysts_corrected_dt30_full_20260722_v2"
BASE_OUT="${SKAE_SCRATCH_ROOT}/dysts_corrected_dt30_full_20260722_v2"
SMOKE_BASE_OUT="${SKAE_SCRATCH_ROOT}/dysts_corrected_dt30_smoke_20260722_v2"
CACHE_DIR="${SKAE_SCRATCH_ROOT}/dysts_native_cache_schema3_dt30_20260722_v2"
PHASE="dysts_corrected_dt30_p256_seq10_100k"
SMOKE_PHASE="dysts_corrected_dt30_smoke_p256_seq10_10k"
TASK_DIR="${RESULTS_DIR}/task_tables"
PREFLIGHT_DIR="${RESULTS_DIR}/preflight"
QUEUE_DIR="${RESULTS_DIR}/queue"
EVAL_DIR="${RESULTS_DIR}/long_horizon_eval"
SMOKE_TSV="${TASK_DIR}/smoke_tasks.tsv"
FULL_TSV="${TASK_DIR}/full_tasks.tsv"
SMOKE_MANIFEST="${TASK_DIR}/smoke_manifest.json"
FULL_MANIFEST="${TASK_DIR}/full_manifest.json"
ROOT_SPECS="${TASK_DIR}/root_specs.tsv"
SYSTEMS_FILE="${TASK_DIR}/systems.txt"
SMOKE_SYSTEMS_FILE="${TASK_DIR}/smoke_systems.txt"
SMOKE_GATE="${PREFLIGHT_DIR}/gpu_smoke_gate.json"

for path in "${RESULTS_DIR}" "${BASE_OUT}" "${SMOKE_BASE_OUT}" "${CACHE_DIR}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing to reuse ${path}" >&2; exit 1; }
done
mkdir -p "${TASK_DIR}" "${PREFLIGHT_DIR}" "${QUEUE_DIR}" "${EVAL_DIR}"
sha256sum -c "${SOURCE_MANIFEST}"

uv run pytest \
  tests/test_data.py::TestDystsAdapter \
  tests/test_dysts_cache_strict.py \
  tests/test_dysts_dt30_basinblock_tasks.py \
  tests/test_dysts_eval_protocol.py \
  tests/test_evaluation_per_ic_survival.py \
  tests/test_model.py -q
uv run python -m experiments.neurips_2026.workflows.dysts_time_grid_validation \
  --output "${PREFLIGHT_DIR}/time_grid_validation.json"

uv run skae-paper tasks dysts \
  --phase_label "${SMOKE_PHASE}" --output_tsv "${SMOKE_TSV}" \
  --output_manifest_json "${SMOKE_MANIFEST}" \
  --systems_csv dysts:Chua \
  --model_variants_csv lista,lista_bd,lista_sb,sparse_mlp_bd,sparse_mlp,dense_mlp_tanh \
  --seeds_csv 0,1 --num_steps 10000 --dt_multiplier 30 \
  --lista_sb_num_loops 1 --dysts_cache_profile smoke
uv run skae-paper tasks dysts \
  --phase_label "${PHASE}" --output_tsv "${FULL_TSV}" \
  --output_manifest_json "${FULL_MANIFEST}" \
  --num_steps 100000 --dt_multiplier 30 --lista_sb_num_loops 1 \
  --dysts_cache_profile full
uv run python -m experiments.neurips_2026.dysts_corrected_dt30_full.prepare \
  --smoke_tsv "${SMOKE_TSV}" --full_tsv "${FULL_TSV}" \
  --base_out "${BASE_OUT}" --phase "${PHASE}" \
  --root_specs "${ROOT_SPECS}" --systems_file "${SYSTEMS_FILE}" \
  --smoke_systems_file "${SMOKE_SYSTEMS_FILE}" \
  --receipt "${PREFLIGHT_DIR}/task_validation.json"

SMOKE_HASH="$(sha256sum "${SMOKE_TSV}" | awk '{print $1}')"
FULL_HASH="$(sha256sum "${FULL_TSV}" | awk '{print $1}')"

SMOKE_CACHE_JOB=$(
  SYSTEMS_FILE="${SMOKE_SYSTEMS_FILE}" CACHE_DIR="${CACHE_DIR}" \
  CACHE_NUM_WORKERS=2 PROFILES=smoke SPLITS="train val" \
  DYSTS_DT_MULTIPLIER=30 SOURCE_MANIFEST="${SOURCE_MANIFEST}" \
  sbatch --parsable --array=0-1 scripts/neurips_2026/dysts/prebuild_cache.sh
)
SMOKE_TRAIN_JOB=$(
  TASK_TSV="${SMOKE_TSV}" TASK_TSV_SHA256="${SMOKE_HASH}" \
  BASE_OUT="${SMOKE_BASE_OUT}" PACK_SIZE=12 PACK_CONCURRENCY=12 \
  TRAIN_SKIP_EVAL=1 SOURCE_MANIFEST="${SOURCE_MANIFEST}" \
  GPU_TELEMETRY_INTERVAL=30 \
  sbatch --parsable --dependency="afterok:${SMOKE_CACHE_JOB}" \
    --array=0-0 --cpus-per-task=12 --mem=32G --time=02:00:00 \
    scripts/common/run_benchmark_packed_array.sh
)
SMOKE_GATE_JOB=$(
  SOURCE_MANIFEST="${SOURCE_MANIFEST}" SMOKE_BASE_OUT="${SMOKE_BASE_OUT}" \
  SMOKE_GATE="${SMOKE_GATE}" \
  sbatch --parsable --dependency="afterok:${SMOKE_TRAIN_JOB}" \
    scripts/neurips_2026/dysts_corrected_dt30_full/adjudicate_smoke.sh
)
FULL_CACHE_JOB=$(
  SYSTEMS_FILE="${SYSTEMS_FILE}" CACHE_DIR="${CACHE_DIR}" CACHE_NUM_WORKERS=2 \
  PROFILES=full SPLITS="train val policy test" DYSTS_DT_MULTIPLIER=30 \
  SOURCE_MANIFEST="${SOURCE_MANIFEST}" \
  sbatch --parsable --dependency="afterok:${SMOKE_GATE_JOB}" --array=0-39 \
    scripts/neurips_2026/dysts/prebuild_cache.sh
)
FULL_TRAIN_JOB=$(
  TASK_TSV="${FULL_TSV}" TASK_TSV_SHA256="${FULL_HASH}" BASE_OUT="${BASE_OUT}" \
  PACK_SIZE=12 PACK_CONCURRENCY=12 TRAIN_SKIP_EVAL=1 \
  SOURCE_MANIFEST="${SOURCE_MANIFEST}" GPU_TELEMETRY_INTERVAL=60 \
  sbatch --parsable --dependency="afterok:${FULL_CACHE_JOB}" \
    --array=0-74%24 --cpus-per-task=12 --mem=32G --time=3-00:00:00 \
    scripts/common/run_benchmark_packed_array.sh
)
EVAL_QUEUE_JOB=$(
  RESULTS_DIR="${EVAL_DIR}" INPUT_ROOT_SPECS_TSV="${ROOT_SPECS}" \
  DYSTS_CACHE_DIR="${CACHE_DIR}" DYSTS_CACHE_PROFILE=full \
  DYSTS_CACHE_SPLIT=test DYSTS_DT_MULTIPLIER=30 \
  DYSTS_PERIODIC_REENCODE_PERIODS="25 100 200" \
  HORIZONS="100 500 1000 1500 2000 3000 4000 5000" \
  OUTPUT_TAG=dysts_corrected_dt30_h100_to_h5000 \
  SOURCE_MANIFEST="${SOURCE_MANIFEST}" REQUIRE_TRAINING_RECEIPT=1 \
  EVAL_PACK_SIZE=12 ARRAY_PARALLEL=48 \
  sbatch --parsable --dependency="afterok:${FULL_TRAIN_JOB}" \
    scripts/neurips_2026/dysts/queue_evaluation.sh
)

uv run python - "${QUEUE_DIR}/queue_record.json" <<PY
import json
from pathlib import Path
payload = {
  "smoke_cache_job": "${SMOKE_CACHE_JOB}",
  "smoke_training_job": "${SMOKE_TRAIN_JOB}",
  "smoke_gate_job": "${SMOKE_GATE_JOB}",
  "full_cache_job": "${FULL_CACHE_JOB}",
  "full_training_job": "${FULL_TRAIN_JOB}",
  "evaluation_queue_job": "${EVAL_QUEUE_JOB}",
  "smoke_task_sha256": "${SMOKE_HASH}",
  "full_task_sha256": "${FULL_HASH}",
  "expected_full_fits": 900,
  "cache_schema": 3,
}
Path("${QUEUE_DIR}/queue_record.json").write_text(json.dumps(payload, indent=2) + "\n")
PY
cat "${QUEUE_DIR}/queue_record.json"
