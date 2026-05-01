#!/bin/bash
#
# Queue the hard-system sparse-KAE forecasting redo requested for the paper.
#
# Systems:
#   - competitive_lv fixed 8-basin, dt=0.005
#   - Hopfield N=16, P=16, dt=0.00625
#   - Kuramoto identical ring N=16, dt=0.00625
#
# Models:
#   dense_mlp_tanh, sparse_mlp, sparse_mlp_bd, lista, lista_bd, lista_sb
#
# Protocol:
#   100k steps, sequence_length=8, target_size=1024, sparsity_coeff=0.006
#   for all sparse models, dense_mlp_tanh sparsity_coeff=0, and all learning
#   rates are half the previous 200k/seq10 setup.
#
# Submit with:
#   sbatch scripts/queue_hard_system_sparse_kae_redo.sh

#SBATCH --job-name=queue_hard_skae_redo
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-hard-skae-redo-%j.out
#SBATCH -e /network/scratch/l/lia/skae/queue-hard-skae-redo-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-hard_system_sparse_kae_redo_p1024_seq8_100k_halflr_sc6em3_tanh_dense_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-hard_system_sparse_kae_redo}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
QUEUE_DIR="${QUEUE_DIR:-${RESULTS_DIR}/queue}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect}"
COMPARE_DIR="${COMPARE_DIR:-${RESULTS_DIR}/compare}"

TASK_TSV="${TASK_TSV:-${TASK_DIR}/hard_system_sparse_kae_redo.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${TASK_DIR}/hard_system_sparse_kae_redo_manifest.json}"
ROOT_SPECS_FILE="${ROOT_SPECS_FILE:-${ROOT_SPEC_DIR}/hard_system_sparse_kae_redo_roots.txt}"
QUEUE_RECORD_JSON="${QUEUE_RECORD_JSON:-${QUEUE_DIR}/queue_record.json}"

SYSTEMS_CSV="${SYSTEMS_CSV:-competitive_lv_fixed_8basin_dt0p005,hopfield_n16_p16_dt0p00625,kuramoto_n16_identical_dt0p00625}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-dense_mlp_tanh,sparse_mlp,sparse_mlp_bd,lista,lista_bd,lista_sb}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
NUM_STEPS="${NUM_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-1024}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.006}"
DENSE_SPARSITY_COEFF="${DENSE_SPARSITY_COEFF:-0.0}"
GENERIC_LR="${GENERIC_LR:-5e-5}"
GENERIC_K_MATRIX_LR="${GENERIC_K_MATRIX_LR:-5e-6}"
LISTA_ALPHA="${LISTA_ALPHA:-0.15}"
LISTA_LR="${LISTA_LR:-2.5e-5}"
LISTA_K_MATRIX_LR="${LISTA_K_MATRIX_LR:-2.5e-6}"
SOFT_BLOCK_WEIGHT="${SOFT_BLOCK_WEIGHT:-0.0001}"
SOFT_BLOCK_NORM="${SOFT_BLOCK_NORM:-l1}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
HORIZONS_CSV="${HORIZONS_CSV:-100,500,1000}"
COMPARE_HORIZONS_CSV="${COMPARE_HORIZONS_CSV:-${HORIZONS_CSV}}"
ANCHOR_ROOT="${ANCHOR_ROOT:-dense_mlp_tanh}"
CANDIDATE_ROOTS_CSV="${CANDIDATE_ROOTS_CSV:-sparse_mlp,sparse_mlp_bd,lista,lista_bd,lista_sb}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-64}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-800}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${QUEUE_DIR}" "${COLLECT_DIR}" "${COMPARE_DIR}"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "Experiment tag: ${EXPERIMENT_TAG}"
echo "Results dir: ${RESULTS_DIR}"
echo "Base out: ${BASE_OUT}"

uv run python tools/build_hard_system_sparse_kae_redo_tasks.py \
  --phase_label "${PHASE_LABEL}" \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --systems_csv "${SYSTEMS_CSV}" \
  --model_variants_csv "${MODEL_VARIANTS_CSV}" \
  --seeds_csv "${SEEDS_CSV}" \
  --num_steps "${NUM_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --target_size "${TARGET_SIZE}" \
  --sequence_length "${SEQUENCE_LENGTH}" \
  --sparsity_coeff "${SPARSITY_COEFF}" \
  --dense_sparsity_coeff "${DENSE_SPARSITY_COEFF}" \
  --generic_lr "${GENERIC_LR}" \
  --generic_k_matrix_lr "${GENERIC_K_MATRIX_LR}" \
  --lista_alpha "${LISTA_ALPHA}" \
  --lista_lr "${LISTA_LR}" \
  --lista_k_matrix_lr "${LISTA_K_MATRIX_LR}" \
  --soft_block_weight "${SOFT_BLOCK_WEIGHT}" \
  --soft_block_norm "${SOFT_BLOCK_NORM}" \
  --eval_profile "${EVAL_PROFILE}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

: > "${ROOT_SPECS_FILE}"
IFS=',' read -r -a MODEL_VARIANTS <<< "${MODEL_VARIANTS_CSV}"
for model_variant in "${MODEL_VARIANTS[@]}"; do
  model_variant="$(echo "${model_variant}" | xargs)"
  [[ -z "${model_variant}" ]] && continue
  printf '%s=%s/%s/%s\n' "${model_variant}" "${BASE_OUT}" "${PHASE_LABEL}" "${model_variant}" >> "${ROOT_SPECS_FILE}"
done

echo "Generated ${TASK_COUNT} training tasks."
echo "Task TSV: ${TASK_TSV}"
echo "Manifest: ${MANIFEST_JSON}"
echo "Root specs: ${ROOT_SPECS_FILE}"

if command -v squeue >/dev/null 2>&1; then
  while true; do
    CURRENT_JOBS=$(squeue -u "${USER}" -h -r | wc -l)
    if (( CURRENT_JOBS <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
      break
    fi
    echo "Current expanded job count ${CURRENT_JOBS} exceeds ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; sleeping before array submit."
    sleep 60
  done
fi

ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --parsable --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}" scripts/run_paper_benchmark_array.sh
)

COLLECT_JOB_ID=$(
  ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
  OUT_DIR="${COLLECT_DIR}" \
  HORIZONS_CSV="${HORIZONS_CSV}" \
  PAPER_SUMMARY=1 \
    sbatch --parsable --dependency=afterany:"${ARRAY_JOB_ID}" scripts/collect_paper_benchmark.sh
)

IFS=',' read -r -a COMPARE_HORIZONS <<< "${COMPARE_HORIZONS_CSV}"
COMPARE_JOB_IDS=()
for horizon in "${COMPARE_HORIZONS[@]}"; do
  horizon="$(echo "${horizon}" | xargs)"
  [[ -z "${horizon}" ]] && continue
  compare_job_id=$(
    ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
    OUT_DIR="${COMPARE_DIR}/h${horizon}" \
    CANDIDATE_ROOTS_CSV="${CANDIDATE_ROOTS_CSV}" \
    ANCHOR_ROOT="${ANCHOR_ROOT}" \
    HORIZON="${horizon}" \
      sbatch --parsable --dependency=afterany:"${COLLECT_JOB_ID}" scripts/compare_paper_benchmark.sh
  )
  COMPARE_JOB_IDS+=("${compare_job_id}")
done

COMPARE_JOB_IDS_CSV="$(IFS=','; echo "${COMPARE_JOB_IDS[*]}")"

cat > "${QUEUE_RECORD_JSON}" <<EOF
{
  "date_tag": "${DATE_TAG}",
  "experiment_tag": "${EXPERIMENT_TAG}",
  "phase_label": "${PHASE_LABEL}",
  "results_dir": "${RESULTS_DIR}",
  "base_out": "${BASE_OUT}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "root_specs_file": "${ROOT_SPECS_FILE}",
  "task_count": ${TASK_COUNT},
  "systems_csv": "${SYSTEMS_CSV}",
  "model_variants_csv": "${MODEL_VARIANTS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "num_steps": ${NUM_STEPS},
  "batch_size": ${BATCH_SIZE},
  "target_size": ${TARGET_SIZE},
  "sequence_length": ${SEQUENCE_LENGTH},
  "sparsity_coeff": ${SPARSITY_COEFF},
  "dense_sparsity_coeff": ${DENSE_SPARSITY_COEFF},
  "generic_lr": ${GENERIC_LR},
  "generic_k_matrix_lr": ${GENERIC_K_MATRIX_LR},
  "lista_alpha": ${LISTA_ALPHA},
  "lista_lr": ${LISTA_LR},
  "lista_k_matrix_lr": ${LISTA_K_MATRIX_LR},
  "soft_block_weight": ${SOFT_BLOCK_WEIGHT},
  "soft_block_norm": "${SOFT_BLOCK_NORM}",
  "horizons_csv": "${HORIZONS_CSV}",
  "anchor_root": "${ANCHOR_ROOT}",
  "candidate_roots_csv": "${CANDIDATE_ROOTS_CSV}",
  "array_throttle": ${ARRAY_THROTTLE},
  "array_job_id": "${ARRAY_JOB_ID}",
  "collect_job_id": "${COLLECT_JOB_ID}",
  "compare_job_ids": "${COMPARE_JOB_IDS_CSV}"
}
EOF

echo "Queued hard-system sparse-KAE redo."
echo "Training array: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Compare jobs: ${COMPARE_JOB_IDS_CSV}"
echo "Queue record: ${QUEUE_RECORD_JSON}"
