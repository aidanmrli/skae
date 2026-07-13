#!/bin/bash
#
# Queue a one-seed Dysts dt-x30 pilot for GenericKM sparse MLP rows with
# LISTAKM-style decoder-atom normalization enabled.
#
# Submit:
#   sbatch scripts/queue_dysts_sparse_mlp_decoder_norm_pilot.sh
#
# Optional env vars:
#   DATE_TAG=20260512
#   SEEDS_CSV=0
#   SYSTEMS_CSV=dysts:Chua,dysts:Dadras
#   NUM_STEPS=100000
#   TRAIN_TIME_LIMIT=03:00:00
#
#SBATCH --job-name=queue_normdec
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-normdec-%j.out
#SBATCH -e /network/scratch/l/lia/skae/queue-normdec-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
PHASE_LABEL="${PHASE_LABEL:-dysts_dt30_sparse_mlp_normdec_pilot}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/dysts_dt30_sparse_mlp_normdec_pilot_${DATE_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/dysts_dt30_sparse_mlp_normdec_pilot_${DATE_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_DIR}/queue}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${RESULTS_DIR}/long_horizon_eval}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-full}"
DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER:-30}"
DYSTS_PERIODIC_REENCODE_PERIODS="${DYSTS_PERIODIC_REENCODE_PERIODS:-10 25 50 100 150 200}"
HORIZONS="${HORIZONS:-100 500 1000 1500 2000 3000 4000 5000}"
SYSTEMS_CSV="${SYSTEMS_CSV:-dysts:Chua,dysts:Dadras,dysts:DequanLi,dysts:Hadley,dysts:LuChenCheng,dysts:QiChen,dysts:Sakarya,dysts:SanUmSrisuchinwong,dysts:ShimizuMorioka,dysts:WangSun}"
SEEDS_CSV="${SEEDS_CSV:-0}"
NUM_STEPS="${NUM_STEPS:-100000}"
TRAIN_PACK_SIZE="${TRAIN_PACK_SIZE:-1}"
TRAIN_ARRAY_PARALLEL="${TRAIN_ARRAY_PARALLEL:-20}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-03:00:00}"
ARRAY_PARALLEL="${ARRAY_PARALLEL:-20}"
EVAL_PACK_SIZE="${EVAL_PACK_SIZE:-4}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-03:00:00}"
VALIDATION_INDEX="${VALIDATION_INDEX:-0}"
OUTPUT_TAG="${OUTPUT_TAG:-dysts_dt30_h100_to_h5000_normdec_pilot}"
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

TASK_TSV="${TASK_DIR}/dysts_sparse_mlp_normdec_tasks.tsv"
MANIFEST_JSON="${TASK_DIR}/dysts_sparse_mlp_normdec_manifest.json"
ROOT_SPECS_TSV="${TASK_DIR}/dysts_sparse_mlp_normdec_root_specs.tsv"
SYSTEMS_FILE="${TASK_DIR}/dysts_sparse_mlp_normdec_systems.txt"
QUEUE_RECORD_JSON="${QUEUE_DIR}/queue_record.json"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "PHASE_LABEL: ${PHASE_LABEL}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV}"
echo "SEEDS_CSV: ${SEEDS_CSV}"
echo "NUM_STEPS: ${NUM_STEPS}"

uv run python tools/build_dysts_dt30_basinblock_tasks.py \
  --phase_label "${PHASE_LABEL}" \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --model_variants_csv "sparse_mlp,sparse_mlp_bd" \
  --systems_csv "${SYSTEMS_CSV}" \
  --seeds_csv "${SEEDS_CSV}" \
  --num_steps "${NUM_STEPS}" \
  --dt_multiplier "${DYSTS_DT_MULTIPLIER}"

uv run python - "${TASK_TSV}" "${ROOT_SPECS_TSV}" "${SYSTEMS_FILE}" "${BASE_OUT}" "${PHASE_LABEL}" <<'PY'
import csv
import sys
from pathlib import Path

task_tsv = Path(sys.argv[1])
root_specs_tsv = Path(sys.argv[2])
systems_file = Path(sys.argv[3])
base_out = Path(sys.argv[4])
phase_label = sys.argv[5]

renames = {
    "sparse_mlp": ("sparse_mlp_normdec", "Sparse MLP norm-dec", "mlp"),
    "sparse_mlp_bd": ("sparse_mlp_bd_normdec", "Sparse MLP-BD norm-dec", "mlp"),
}

with task_tsv.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

if "normalize_decoder_atoms" not in fieldnames:
    fieldnames.append("normalize_decoder_atoms")

variants = []
systems = []
for row in rows:
    old_variant = row["model_variant"]
    new_variant, _, _ = renames[old_variant]
    row["model_variant"] = new_variant
    row["normalize_decoder_atoms"] = "true"
    system = row["system_key"]
    if new_variant not in variants:
        variants.append(new_variant)
    if system not in systems:
        systems.append(system)

with task_tsv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

root_specs_tsv.parent.mkdir(parents=True, exist_ok=True)
with root_specs_tsv.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["label", "display_name", "model_family", "root_dir"],
        delimiter="\t",
    )
    writer.writeheader()
    for old_variant, (new_variant, display_name, family) in renames.items():
        if new_variant in variants:
            writer.writerow(
                {
                    "label": new_variant,
                    "display_name": display_name,
                    "model_family": family,
                    "root_dir": str(base_out / phase_label / new_variant),
                }
            )

systems_file.write_text("".join(system.split(":", 1)[1] + "\n" for system in systems))
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
SYSTEM_COUNT=$(wc -l < "${SYSTEMS_FILE}")
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
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
wait_for_submit_capacity "normalized-decoder packed training array"
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
  EVAL_PACK_SIZE="${EVAL_PACK_SIZE}" \
  sbatch --parsable -p long --dependency=afterany:${TRAIN_JOB_ID} scripts/queue_dysts_long_horizon_eval.sh
)

cat > "${QUEUE_RECORD_JSON}" <<EOF
{
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
  "seeds_csv": "${SEEDS_CSV}",
  "num_steps": ${NUM_STEPS},
  "normalize_decoder_atoms": true,
  "pretrain_cache_job_id": "${PRETRAIN_CACHE_JOB_ID}",
  "train_job_id": "${TRAIN_JOB_ID}",
  "eval_queue_job_id": "${EVAL_QUEUE_JOB_ID}"
}
EOF

echo "Queued Dysts normalized-decoder Sparse MLP pilot."
echo "Cache array: ${PRETRAIN_CACHE_JOB_ID}"
echo "Training array: ${TRAIN_JOB_ID}"
echo "Long-horizon queue: ${EVAL_QUEUE_JOB_ID}"
echo "Queue record: ${QUEUE_RECORD_JSON}"
