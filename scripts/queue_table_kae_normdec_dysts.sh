#!/bin/bash
#
# Queue retained-10 Dysts dt-x30 Table-1 KAE replacements/ablations with
# normalized linear decoder flag and configurable latent sparsity target.
#
# Submit examples:
#   SPARSITY_TARGET=rollout sbatch scripts/queue_table_kae_normdec_dysts.sh
#   SPARSITY_TARGET=encoded sbatch scripts/queue_table_kae_normdec_dysts.sh

#SBATCH --job-name=queue_kae_normdysts
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-kae-normdysts-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-kae-normdysts-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run under SLURM."
  echo "Submit it with: sbatch scripts/queue_table_kae_normdec_dysts.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
SPARSITY_TARGET="${SPARSITY_TARGET:-rollout}"
if [[ "${SPARSITY_TARGET}" != "rollout" && "${SPARSITY_TARGET}" != "encoded" && "${SPARSITY_TARGET}" != "encoded_rollout" ]]; then
  echo "SPARSITY_TARGET must be rollout, encoded, or encoded_rollout; got ${SPARSITY_TARGET}"
  exit 2
fi

VARIANT_SUFFIX="${VARIANT_SUFFIX:-normdec_${SPARSITY_TARGET}}"
PHASE_LABEL="${PHASE_LABEL:-dysts_dt30_table_kae_${VARIANT_SUFFIX}}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/dysts_dt30_table_kae_${VARIANT_SUFFIX}_${DATE_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/dysts_dt30_table_kae_${VARIANT_SUFFIX}_${DATE_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_DIR}/queue}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${RESULTS_DIR}/long_horizon_eval}"
DYSTS_CACHE_DIR="${DYSTS_CACHE_DIR:-/network/scratch/l/lia/skae/dysts_native_cache}"
DYSTS_CACHE_PROFILE="${DYSTS_CACHE_PROFILE:-full}"
DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER:-30}"
DYSTS_PERIODIC_REENCODE_PERIODS="${DYSTS_PERIODIC_REENCODE_PERIODS:-10 25 50 100 150 200}"
HORIZONS="${HORIZONS:-100 500 1000 1500 2000 3000 4000 5000}"
SYSTEMS_CSV="${SYSTEMS_CSV:-dysts:Chua,dysts:Dadras,dysts:DequanLi,dysts:Hadley,dysts:LuChenCheng,dysts:QiChen,dysts:Sakarya,dysts:SanUmSrisuchinwong,dysts:ShimizuMorioka,dysts:WangSun}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-lista,lista_bd,lista_sb,sparse_mlp,sparse_mlp_bd,dense_mlp_tanh}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
NUM_STEPS="${NUM_STEPS:-100000}"
TRAIN_PACK_SIZE="${TRAIN_PACK_SIZE:-4}"
TRAIN_ARRAY_PARALLEL="${TRAIN_ARRAY_PARALLEL:-60}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-03:00:00}"
ARRAY_PARALLEL="${ARRAY_PARALLEL:-48}"
EVAL_PACK_SIZE="${EVAL_PACK_SIZE:-8}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-03:00:00}"
VALIDATION_INDEX="${VALIDATION_INDEX:-0}"
OUTPUT_TAG="${OUTPUT_TAG:-dysts_dt30_h100_to_h5000_${VARIANT_SUFFIX}}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-10000}"
SUBMIT_WAIT_SECONDS="${SUBMIT_WAIT_SECONDS:-120}"

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

TASK_TSV="${TASK_DIR}/dysts_dt30_table_kae_${VARIANT_SUFFIX}_tasks.tsv"
MANIFEST_JSON="${TASK_DIR}/dysts_dt30_table_kae_${VARIANT_SUFFIX}_manifest.json"
ROOT_SPECS_TSV="${TASK_DIR}/dysts_dt30_table_kae_${VARIANT_SUFFIX}_root_specs.tsv"
SYSTEMS_FILE="${TASK_DIR}/dysts_dt30_table_kae_${VARIANT_SUFFIX}_systems.txt"
QUEUE_RECORD_JSON="${QUEUE_DIR}/queue_record.json"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "SPARSITY_TARGET: ${SPARSITY_TARGET}"
echo "VARIANT_SUFFIX: ${VARIANT_SUFFIX}"
echo "BASE_OUT: ${BASE_OUT}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "PHASE_LABEL: ${PHASE_LABEL}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV}"
echo "MODEL_VARIANTS_CSV: ${MODEL_VARIANTS_CSV}"
echo "SEEDS_CSV: ${SEEDS_CSV}"

uv run python tools/build_dysts_dt30_basinblock_tasks.py \
  --phase_label "${PHASE_LABEL}" \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --model_variants_csv "${MODEL_VARIANTS_CSV}" \
  --systems_csv "${SYSTEMS_CSV}" \
  --seeds_csv "${SEEDS_CSV}" \
  --num_steps "${NUM_STEPS}" \
  --dt_multiplier "${DYSTS_DT_MULTIPLIER}"

SPARSITY_TARGET="${SPARSITY_TARGET}" \
VARIANT_SUFFIX="${VARIANT_SUFFIX}" \
BASE_OUT="${BASE_OUT}" \
PHASE_LABEL="${PHASE_LABEL}" \
  uv run python - "${TASK_TSV}" "${ROOT_SPECS_TSV}" "${SYSTEMS_FILE}" <<'PY'
import csv
import os
import sys
from pathlib import Path

task_tsv = Path(sys.argv[1])
root_specs_tsv = Path(sys.argv[2])
systems_file = Path(sys.argv[3])

sparsity_target = os.environ["SPARSITY_TARGET"]
variant_suffix = os.environ["VARIANT_SUFFIX"]
base_out = Path(os.environ["BASE_OUT"])
phase_label = os.environ["PHASE_LABEL"]

display_names = {
    "lista": ("LISTA", "lista"),
    "lista_bd": ("LISTA-BD", "lista"),
    "lista_sb": ("LISTA-SB", "lista"),
    "sparse_mlp": ("Sparse MLP", "mlp"),
    "sparse_mlp_bd": ("Sparse MLP-BD", "mlp"),
    "dense_mlp_tanh": ("Dense MLP", "mlp"),
}

with task_tsv.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

for extra in ("normalize_decoder_atoms", "sparsity_target"):
    if extra not in fieldnames:
        fieldnames.append(extra)

variants = []
systems = []
for row in rows:
    old_variant = row["model_variant"]
    new_variant = f"{old_variant}_{variant_suffix}"
    row["model_variant"] = new_variant
    row["normalize_decoder_atoms"] = "true"
    row["sparsity_target"] = sparsity_target
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
    for new_variant in variants:
        old_variant = new_variant[: -len(f"_{variant_suffix}")]
        display_name, family = display_names[old_variant]
        writer.writerow(
            {
                "label": new_variant,
                "display_name": f"{display_name} ({variant_suffix})",
                "model_family": family,
                "root_dir": str(base_out / phase_label / new_variant),
            }
        )

systems_file.write_text("".join(system.split(":", 1)[1] + "\n" for system in systems))
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
SYSTEM_COUNT=$(wc -l < "${SYSTEMS_FILE}")
MODEL_COUNT=$(tail -n +2 "${TASK_TSV}" | cut -f3 | sort -u | wc -l)
SEED_COUNT=$(tail -n +2 "${TASK_TSV}" | cut -f9 | sort -u | wc -l)
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
  "variant_suffix": "${VARIANT_SUFFIX}",
  "sparsity_target": "${SPARSITY_TARGET}",
  "normalize_decoder_atoms": true,
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
  "train_pack_size": ${TRAIN_PACK_SIZE},
  "train_array_count": ${TRAIN_ARRAY_COUNT},
  "pretrain_cache_job_id": "${PRETRAIN_CACHE_JOB_ID}",
  "train_job_id": "${TRAIN_JOB_ID}",
  "eval_queue_job_id": "${EVAL_QUEUE_JOB_ID}"
}
EOF

echo "Queued Dysts KAE ${VARIANT_SUFFIX} table packet."
echo "Cache array: ${PRETRAIN_CACHE_JOB_ID}"
echo "Training array: ${TRAIN_JOB_ID}"
echo "Long-horizon queue: ${EVAL_QUEUE_JOB_ID}"
echo "Queue record: ${QUEUE_RECORD_JSON}"
