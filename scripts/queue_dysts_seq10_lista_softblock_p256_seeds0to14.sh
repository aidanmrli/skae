#!/bin/bash
#
# Queue the Dysts seq=10, d_z=256 LISTA soft-block add-on.
#
# This is a Table 4 sensitivity/add-on row matching the current Dysts rerun:
#   - 12 Dysts systems
#   - seeds 0-14
#   - 200k training steps
#   - sequence_length=10
#   - target_size=256
#   - sparsity_coeff=0.006
#   - LISTA-BD-compatible encoder (1 loop, ReLU final op)
#   - dense LISTA transition with soft block penalty over 16 fixed blocks
#
# Submit with:
#   sbatch scripts/queue_dysts_seq10_lista_softblock_p256_seeds0to14.sh

#SBATCH --job-name=queue_dysts_lsb_p256
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-dysts-lsb-p256-%j.out
#SBATCH -e /network/scratch/l/lia/skae/queue-dysts-lsb-p256-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-20260428}"
MODEL_LABEL="${MODEL_LABEL:-lista_softblock_p256_seq10_sc6em3}"
DISPLAY_NAME="${DISPLAY_NAME:-LISTA soft-block p256 (sc=6e-3)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-dysts_seq10_lista_softblock_p256_sc6em3_seeds0to14_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-paper_followup_recipes}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_DIR}/queue}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-results/dysts_long_horizon_eval_${EXPERIMENT_TAG}}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
SYSTEMS_CSV="${SYSTEMS_CSV:-dysts:Chua,dysts:Dadras,dysts:DequanLi,dysts:Hadley,dysts:LorenzCoupled,dysts:LuChenCheng,dysts:MultiChua,dysts:QiChen,dysts:Sakarya,dysts:SanUmSrisuchinwong,dysts:ShimizuMorioka,dysts:WangSun}"
RECIPE_SPEC="${RECIPE_SPEC:-${MODEL_LABEL}:lista_dense:200000:5e-5:5e-6:1e-4:0.03:1.0:0.006}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-64}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-800}"
SOFT_BLOCK_NUM_BLOCKS="${SOFT_BLOCK_NUM_BLOCKS:-16}"
SOFT_BLOCK_WEIGHT="${SOFT_BLOCK_WEIGHT:-0.0001}"
SOFT_BLOCK_NORM="${SOFT_BLOCK_NORM:-l1}"
EVAL_HORIZONS="${EVAL_HORIZONS:-5000 10000 20000 30000 40000 50000 60000}"
EVAL_ARRAY_PARALLEL="${EVAL_ARRAY_PARALLEL:-48}"
EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT:-06:00:00}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${QUEUE_DIR}" "${EVAL_RESULTS_DIR}"

TASK_TSV="${TASK_DIR}/dysts_lista_softblock_p256_tasks.tsv"
MANIFEST_JSON="${TASK_DIR}/dysts_lista_softblock_p256_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/dysts_lista_softblock_p256_roots.txt"
DYSTS_ROOT_SPECS_TSV="${RESULTS_DIR}/dysts_long_horizon_root_specs.tsv"
QUEUE_RECORD_JSON="${QUEUE_DIR}/queue_record.json"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "Model label: ${MODEL_LABEL}"
echo "Results dir: ${RESULTS_DIR}"
echo "Base out: ${BASE_OUT}"

uv run python tools/build_paper_followup_recipe_tasks.py \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --phase_label "${PHASE_LABEL}" \
  --systems_csv "${SYSTEMS_CSV}" \
  --seeds_csv "${SEEDS_CSV}" \
  --recipe_specs_csv "${RECIPE_SPEC}" \
  --eval_profile full

SOFT_BLOCK_NUM_BLOCKS="${SOFT_BLOCK_NUM_BLOCKS}" \
SOFT_BLOCK_WEIGHT="${SOFT_BLOCK_WEIGHT}" \
SOFT_BLOCK_NORM="${SOFT_BLOCK_NORM}" \
MODEL_LABEL="${MODEL_LABEL}" \
MANIFEST_JSON="${MANIFEST_JSON}" \
  uv run python - "${TASK_TSV}" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

task_tsv = Path(sys.argv[1])
manifest_json = Path(os.environ["MANIFEST_JSON"])
soft_block_num_blocks = os.environ["SOFT_BLOCK_NUM_BLOCKS"]
soft_block_weight = os.environ["SOFT_BLOCK_WEIGHT"]
soft_block_norm = os.environ["SOFT_BLOCK_NORM"]
model_label = os.environ["MODEL_LABEL"]

with task_tsv.open(newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    fieldnames = list(reader.fieldnames or [])
    rows = list(reader)

extra_fields = [
    "soft_block",
    "soft_block_num_blocks",
    "soft_block_weight",
    "soft_block_norm",
]
for field in extra_fields:
    if field not in fieldnames:
        fieldnames.append(field)

for row in rows:
    row["model_variant"] = model_label
    row["target_size"] = "256"
    row["sequence_length"] = "10"
    row["sparsity_coeff"] = "0.006"
    row["k_structure"] = "dense"
    row["lista_alpha"] = "0.15"
    row["lista_num_loops"] = "1"
    row["lista_final_op"] = "relu"
    row["soft_block"] = "1"
    row["soft_block_num_blocks"] = soft_block_num_blocks
    row["soft_block_weight"] = soft_block_weight
    row["soft_block_norm"] = soft_block_norm

with task_tsv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

payload = json.loads(manifest_json.read_text())
payload["soft_block"] = {
    "enabled": True,
    "num_blocks": int(soft_block_num_blocks),
    "weight": float(soft_block_weight),
    "norm": soft_block_norm,
}
payload["lista"] = {
    "alpha": 0.15,
    "num_loops": 1,
    "final_op": "relu",
}
payload["target_size"] = 256
payload["sequence_length"] = 10
manifest_json.write_text(json.dumps(payload, indent=2))
PY

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

printf '%s=%s/%s/%s\n' "${MODEL_LABEL}" "${BASE_OUT}" "${PHASE_LABEL}" "${MODEL_LABEL}" > "${ROOT_SPECS_FILE}"
{
  printf 'label\tdisplay_name\tmodel_family\troot_dir\n'
  printf '%s\t%s\tlista\t%s/%s/%s\n' "${MODEL_LABEL}" "${DISPLAY_NAME}" "${BASE_OUT}" "${PHASE_LABEL}" "${MODEL_LABEL}"
} > "${DYSTS_ROOT_SPECS_TSV}"

echo "Generated ${TASK_COUNT} training tasks."
echo "Task TSV: ${TASK_TSV}"
echo "Manifest: ${MANIFEST_JSON}"
echo "Dysts eval root specs: ${DYSTS_ROOT_SPECS_TSV}"

while true; do
  CURRENT_JOBS=$(squeue -u "${USER}" -h -r | wc -l)
  if (( CURRENT_JOBS <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
    break
  fi
  echo "Current expanded job count ${CURRENT_JOBS} exceeds ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; sleeping before array submit."
  sleep 60
done

ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --parsable --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}" scripts/run_paper_benchmark_array.sh
)

EVAL_QUEUE_JOB_ID=$(
  DATE_TAG="${DATE_TAG}" \
  RESULTS_DIR="${EVAL_RESULTS_DIR}" \
  OUTPUT_TAG="dysts_long_horizon_h5k_to_h60k_seq10" \
  HORIZONS="${EVAL_HORIZONS}" \
  DYSTS_CACHE_PROFILE="long60" \
  INPUT_ROOT_SPECS_TSV="${DYSTS_ROOT_SPECS_TSV}" \
  EVAL_TIME_LIMIT="${EVAL_TIME_LIMIT}" \
  ARRAY_PARALLEL="${EVAL_ARRAY_PARALLEL}" \
    sbatch --parsable --dependency=afterany:"${ARRAY_JOB_ID}" scripts/queue_dysts_long_horizon_eval.sh
)

cat > "${QUEUE_RECORD_JSON}" <<EOF
{
  "date_tag": "${DATE_TAG}",
  "model_label": "${MODEL_LABEL}",
  "display_name": "${DISPLAY_NAME}",
  "results_dir": "${RESULTS_DIR}",
  "base_out": "${BASE_OUT}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "root_specs_file": "${ROOT_SPECS_FILE}",
  "dysts_root_specs_tsv": "${DYSTS_ROOT_SPECS_TSV}",
  "task_count": ${TASK_COUNT},
  "systems_csv": "${SYSTEMS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "target_size": 256,
  "sequence_length": 10,
  "sparsity_coeff": 0.006,
  "soft_block_num_blocks": ${SOFT_BLOCK_NUM_BLOCKS},
  "soft_block_weight": ${SOFT_BLOCK_WEIGHT},
  "soft_block_norm": "${SOFT_BLOCK_NORM}",
  "array_throttle": ${ARRAY_THROTTLE},
  "array_job_id": "${ARRAY_JOB_ID}",
  "eval_results_dir": "${EVAL_RESULTS_DIR}",
  "eval_horizons": "${EVAL_HORIZONS}",
  "eval_queue_job_id": "${EVAL_QUEUE_JOB_ID}"
}
EOF

echo "Queued Dysts LISTA soft-block p256 add-on."
echo "Training array: ${ARRAY_JOB_ID}"
echo "Eval queue launcher: ${EVAL_QUEUE_JOB_ID}"
echo "Queue record: ${QUEUE_RECORD_JSON}"
