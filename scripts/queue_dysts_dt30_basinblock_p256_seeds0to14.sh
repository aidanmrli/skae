#!/bin/bash
#
# Queue the Dysts dt-x30 basin-block rerun and its long-horizon reevaluation.
#
# Submit:
#   sbatch scripts/queue_dysts_dt30_basinblock_p256_seeds0to14.sh
#
# Optional env vars:
#   DATE_TAG=20260430
#   BASE_OUT=/network/scratch/l/lia/skae/dysts_dt30_basinblock_p256_seq10_100k_${DATE_TAG}
#   RESULTS_DIR=results/dysts_dt30_basinblock_p256_seq10_100k_${DATE_TAG}
#   TRAIN_ARRAY_PARALLEL=90
#   TRAIN_PACK_SIZE=12
#   TRAIN_TIME_LIMIT=3-00:00:00
#   SYSTEMS_CSV=dysts:Chua,dysts:Dadras
#   MODEL_VARIANTS_CSV=lista_bd,lista_sb
#   SEEDS_CSV=0,1,2
#   MAX_EXISTING_JOBS_BEFORE_SUBMIT=10000
#   ARRAY_PARALLEL=64
#   EVAL_TIME_LIMIT=06:00:00
#
#SBATCH --job-name=queue_dysts_dt30
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3-00:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-dysts-dt30-%j.out
#SBATCH -e /network/scratch/l/lia/skae/queue-dysts-dt30-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
NUM_STEPS="${NUM_STEPS:-100000}"
PHASE_LABEL="${PHASE_LABEL:-dysts_dt30_basinblock_p256_seq10_100k}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/dysts_dt30_basinblock_p256_seq10_100k_${DATE_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/dysts_dt30_basinblock_p256_seq10_100k_${DATE_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_DIR}/queue}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${RESULTS_DIR}/long_horizon_eval}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-full}"
DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER:-30}"
DYSTS_PERIODIC_REENCODE_PERIODS="${DYSTS_PERIODIC_REENCODE_PERIODS:-10 25 50 100 150 200}"
HORIZONS="${HORIZONS:-100 500 1000 1500 2000 3000 4000 5000}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
TRAIN_ARRAY_PARALLEL="${TRAIN_ARRAY_PARALLEL:-90}"
TRAIN_PACK_SIZE="${TRAIN_PACK_SIZE:-12}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-3-00:00:00}"
ARRAY_PARALLEL="${ARRAY_PARALLEL:-64}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-06:00:00}"
VALIDATION_INDEX="${VALIDATION_INDEX:-0}"
OUTPUT_TAG="${OUTPUT_TAG:-dysts_dt30_h100_to_h5000_p17periods}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-10000}"
SUBMIT_WAIT_SECONDS="${SUBMIT_WAIT_SECONDS:-300}"

wait_for_submit_capacity() {
  local label="$1"
  while true; do
    local active_jobs
    active_jobs=$(squeue -u "${USER}" -h -r | wc -l)
    if (( active_jobs <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
      echo "Submit capacity available for ${label}: active_jobs=${active_jobs}, threshold=${MAX_EXISTING_JOBS_BEFORE_SUBMIT}"
      return 0
    fi
    echo "Waiting to submit ${label}: active_jobs=${active_jobs}, threshold=${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; sleeping ${SUBMIT_WAIT_SECONDS}s"
    sleep "${SUBMIT_WAIT_SECONDS}"
  done
}

mkdir -p "${TASK_DIR}" "${QUEUE_DIR}" "${EVAL_RESULTS_DIR}"

TASK_TSV="${TASK_DIR}/dysts_dt30_basinblock_tasks.tsv"
MANIFEST_JSON="${TASK_DIR}/dysts_dt30_basinblock_manifest.json"
ROOT_SPECS_TSV="${TASK_DIR}/dysts_dt30_root_specs.tsv"
SYSTEMS_FILE="${TASK_DIR}/dysts_dt30_systems.txt"
QUEUE_RECORD_JSON="${QUEUE_DIR}/queue_record.json"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "PHASE_LABEL: ${PHASE_LABEL}"
echo "NUM_STEPS: ${NUM_STEPS}"
echo "HORIZONS: ${HORIZONS}"
echo "DYSTS_PERIODIC_REENCODE_PERIODS: ${DYSTS_PERIODIC_REENCODE_PERIODS}"
echo "DYSTS_DT_MULTIPLIER: ${DYSTS_DT_MULTIPLIER}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV:-<default>}"
echo "MODEL_VARIANTS_CSV: ${MODEL_VARIANTS_CSV:-<default>}"
echo "SEEDS_CSV: ${SEEDS_CSV:-<default>}"
echo "TRAIN_PACK_SIZE: ${TRAIN_PACK_SIZE}"
echo "MAX_EXISTING_JOBS_BEFORE_SUBMIT: ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}"

BUILD_ARGS=(
  tools/build_dysts_dt30_basinblock_tasks.py
  --phase_label "${PHASE_LABEL}"
  --output_tsv "${TASK_TSV}"
  --output_manifest_json "${MANIFEST_JSON}"
  --num_steps "${NUM_STEPS}"
  --dt_multiplier "${DYSTS_DT_MULTIPLIER}"
)
if [[ -n "${SYSTEMS_CSV}" ]]; then
  BUILD_ARGS+=(--systems_csv "${SYSTEMS_CSV}")
fi
if [[ -n "${MODEL_VARIANTS_CSV}" ]]; then
  BUILD_ARGS+=(--model_variants_csv "${MODEL_VARIANTS_CSV}")
fi
if [[ -n "${SEEDS_CSV}" ]]; then
  BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
fi

uv run python "${BUILD_ARGS[@]}"

uv run python - "${TASK_TSV}" "${ROOT_SPECS_TSV}" "${SYSTEMS_FILE}" "${BASE_OUT}" "${PHASE_LABEL}" <<'PY'
import csv
import sys
from pathlib import Path

task_tsv = Path(sys.argv[1])
root_specs_tsv = Path(sys.argv[2])
systems_file = Path(sys.argv[3])
base_out = Path(sys.argv[4])
phase_label = sys.argv[5]

display_names = {
    "lista": ("LISTA", "lista"),
    "lista_bd": ("LISTA-BD", "lista"),
    "lista_sb": ("LISTA-SB", "lista"),
    "sparse_mlp": ("Sparse MLP", "mlp"),
    "sparse_mlp_bd": ("Sparse MLP-BD", "mlp"),
    "dense_mlp_tanh": ("Dense MLP tanh", "mlp"),
}
variants = []
systems = []
with task_tsv.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for row in reader:
        variant = row["model_variant"]
        system = row["system_key"]
        if variant not in variants:
            variants.append(variant)
        if system not in systems:
            systems.append(system)

root_specs_tsv.parent.mkdir(parents=True, exist_ok=True)
with root_specs_tsv.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["label", "display_name", "model_family", "root_dir"],
        delimiter="\t",
    )
    writer.writeheader()
    for variant in variants:
        display_name, family = display_names[variant]
        writer.writerow(
            {
                "label": variant,
                "display_name": display_name,
                "model_family": family,
                "root_dir": str(base_out / phase_label / variant),
            }
        )

systems_file.write_text(
    "".join(system.split(":", 1)[1] + "\n" for system in systems)
)
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
SYSTEM_COUNT=$(wc -l < "${SYSTEMS_FILE}")
MODEL_COUNT=$(tail -n +2 "${TASK_TSV}" | cut -f3 | sort -u | wc -l)
SEED_COUNT=$(tail -n +2 "${TASK_TSV}" | cut -f9 | sort -u | wc -l)
if (( TASK_COUNT <= 0 )); then
  echo "No training tasks were built."
  exit 1
fi
if (( VALIDATION_INDEX < 0 || VALIDATION_INDEX >= TASK_COUNT )); then
  echo "VALIDATION_INDEX=${VALIDATION_INDEX} is out of range for TASK_COUNT=${TASK_COUNT}"
  exit 1
fi

wait_for_submit_capacity "pretrain cache"
PRETRAIN_CACHE_JOB_ID=$(
  SYSTEMS_FILE="${SYSTEMS_FILE}" \
  CACHE_DIR="${DYSTS_CACHE_DIR}" \
  CACHE_NUM_WORKERS=2 \
  PROFILES="${DYSTS_CACHE_PROFILE}" \
  SPLITS="train val" \
  DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER}" \
  sbatch --parsable -p long --array=0-$((SYSTEM_COUNT * 2 - 1)) scripts/prebuild_dysts_cache_matrix.sh
)

TRAIN_ARRAY_COUNT=$(( (TASK_COUNT + TRAIN_PACK_SIZE - 1) / TRAIN_PACK_SIZE ))
wait_for_submit_capacity "packed training array"
TRAIN_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
  BASE_OUT="${BASE_OUT}" \
  ARRAY_OFFSET=0 \
  PACK_SIZE="${TRAIN_PACK_SIZE}" \
  sbatch --parsable -p long --time="${TRAIN_TIME_LIMIT}" \
    --dependency=afterok:${PRETRAIN_CACHE_JOB_ID} \
    --array=0-$((TRAIN_ARRAY_COUNT - 1))%${TRAIN_ARRAY_PARALLEL} \
    scripts/run_paper_benchmark_packed_array.sh
)

TRAIN_DEP="${TRAIN_JOB_ID}"
EVAL_QUEUE_JOB_ID=$(
  RESULTS_DIR="${EVAL_RESULTS_DIR}" \
  INPUT_ROOT_SPECS_TSV="${ROOT_SPECS_TSV}" \
  DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR}" \
  DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE}" \
  DYSTS_CACHE_SPLIT=test \
  DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER}" \
  DYSTS_PERIODIC_REENCODE_PERIODS="${DYSTS_PERIODIC_REENCODE_PERIODS}" \
  HORIZONS="${HORIZONS}" \
  OUTPUT_TAG="${OUTPUT_TAG}" \
  ARRAY_PARALLEL="${ARRAY_PARALLEL}" \
  EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT}" \
  VALIDATION_INDEX="${VALIDATION_INDEX}" \
  EVAL_PACK_SIZE="${EVAL_PACK_SIZE:-12}" \
  sbatch --parsable -p long --dependency=afterany:${TRAIN_DEP} scripts/queue_dysts_long_horizon_eval.sh
)

TRAIN_JOB_IDS_STR="${TRAIN_JOB_ID}"
uv run python - "${QUEUE_RECORD_JSON}" "${TRAIN_JOB_IDS_STR}" <<PY
import json
import sys
from pathlib import Path

payload = {
    "date_tag": "${DATE_TAG}",
    "phase_label": "${PHASE_LABEL}",
    "base_out": "${BASE_OUT}",
    "results_dir": "${RESULTS_DIR}",
    "eval_results_dir": "${EVAL_RESULTS_DIR}",
    "task_tsv": "${TASK_TSV}",
    "manifest_json": "${MANIFEST_JSON}",
    "root_specs_tsv": "${ROOT_SPECS_TSV}",
    "systems_file": "${SYSTEMS_FILE}",
    "task_count": ${TASK_COUNT},
    "system_count": ${SYSTEM_COUNT},
    "model_count": ${MODEL_COUNT},
    "seed_count": ${SEED_COUNT},
    "num_steps": ${NUM_STEPS},
    "target_size": 256,
    "sequence_length": 10,
    "systems_csv": "${SYSTEMS_CSV}",
    "model_variants_csv": "${MODEL_VARIANTS_CSV}",
    "seeds_csv": "${SEEDS_CSV}",
    "train_pack_size": ${TRAIN_PACK_SIZE},
    "train_array_count": ${TRAIN_ARRAY_COUNT},
    "dysts_dt_multiplier": "${DYSTS_DT_MULTIPLIER}",
    "horizons": "${HORIZONS}",
    "dysts_periodic_reencode_periods": "${DYSTS_PERIODIC_REENCODE_PERIODS}",
    "pretrain_cache_job_id": "${PRETRAIN_CACHE_JOB_ID}",
    "train_job_ids": [item for item in sys.argv[2].split(",") if item],
    "eval_queue_job_id": "${EVAL_QUEUE_JOB_ID}",
    "train_dependency": "${TRAIN_DEP}",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "Queued Dysts dt-x30 basin-block rerun."
echo "Pretrain cache array: ${PRETRAIN_CACHE_JOB_ID}"
echo "Training arrays: ${TRAIN_DEP}"
echo "Evaluation queue launcher: ${EVAL_QUEUE_JOB_ID}"
echo "Queue record: ${QUEUE_RECORD_JSON}"
