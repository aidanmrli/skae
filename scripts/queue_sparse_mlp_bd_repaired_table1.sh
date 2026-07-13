#!/bin/bash
#
# Queue the repaired Sparse MLP-BD control for the paper-facing Table 1
# controlled multibasin forecasting and per-basin deep support diagnostics.
#
# This reruns only:
#   mlp_sparse_blockdiag_hardinit_basin_partition_control
#
# Training runs on GPU via scripts/run_paper_benchmark_array.sh.
#
# Submit:
#   sbatch scripts/queue_sparse_mlp_bd_repaired_table1.sh
#
# Optional env vars:
#   DATE_TAG=20260506
#   EXPERIMENT_TAG=transition_rich_sparse_mlp_bd_repaired_table1_${DATE_TAG}
#   ARRAY_THROTTLE=64
#   MAX_EXISTING_JOBS_BEFORE_SUBMIT=650

#SBATCH --job-name=queue_smlpbd_fix
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-sparse-mlp-bd-repaired-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-sparse-mlp-bd-repaired-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run under SLURM."
  echo "Submit it with: sbatch scripts/queue_sparse_mlp_bd_repaired_table1.sh"
  exit 2
fi

source .venv/bin/activate

DATE_TAG="${DATE_TAG:-20260506}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-transition_rich_sparse_mlp_bd_repaired_table1_${DATE_TAG}}"
PHASE_LABEL="${PHASE_LABEL:-transition_rich_basin_partition}"
MODEL_VARIANT="${MODEL_VARIANT:-mlp_sparse_blockdiag_hardinit_basin_partition_control}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
COLLECT_DIR="${COLLECT_DIR:-${RESULTS_DIR}/collect_pass0}"
INTERP_DIR="${INTERP_DIR:-${RESULTS_DIR}/interpretability_per_basin_deep_pass0}"
QUEUE_LOG_DIR="${QUEUE_LOG_DIR:-${RESULTS_DIR}/queue_logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-64}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-650}"
SUBMIT_WAIT_SECONDS="${SUBMIT_WAIT_SECONDS:-60}"

SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
SYSTEMS_CSV="${SYSTEMS_CSV:-gated_local_linear,gated_transfer_linear,claude:arrested_spiral,claude:cal_asymmetric_3,claude:cal_high_cross_3,claude:cal_hexagon_6,claude:cal_octagon_8,claude:cal_pentagon_5,claude:cal_square_4,claude:duffing_triple_well,claude:snic_multi,claude:transition_routes_4,claude:var_depth_gradient_4,claude:var_diamond_4,claude:var_l_shape_5}"

mkdir -p \
  "${TASK_DIR}" \
  "${ROOT_SPEC_DIR}" \
  "${COLLECT_DIR}" \
  "${INTERP_DIR}/shards" \
  "${INTERP_DIR}/logs" \
  "${QUEUE_LOG_DIR}" \
  "${AUTOMATION_DIR}"

TASK_TSV="${TASK_DIR}/sparse_mlp_bd_repaired_table1_tasks.tsv"
MANIFEST_JSON="${TASK_DIR}/sparse_mlp_bd_repaired_table1_manifest.json"
ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/sparse_mlp_bd_repaired_table1_roots.txt"

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Date: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "MODEL_VARIANT: ${MODEL_VARIANT}"
echo "SYSTEMS_CSV: ${SYSTEMS_CSV}"
echo "SEEDS_CSV: ${SEEDS_CSV}"

uv run python tools/build_transition_rich_basin_partition_tasks.py \
  --phase_label "${PHASE_LABEL}" \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --model_variants_csv "${MODEL_VARIANT}" \
  --systems_csv "${SYSTEMS_CSV}" \
  --seeds_csv "${SEEDS_CSV}" \
  --num_steps_override 200000

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
  echo "No tasks generated in ${TASK_TSV}"
  exit 1
fi

printf '%s=%s/%s/%s\n' "${MODEL_VARIANT}" "${BASE_OUT}" "${PHASE_LABEL}" "${MODEL_VARIANT}" > "${ROOT_SPECS_FILE}"

while true; do
  CURRENT_JOBS=$(squeue -u "${USER}" -h -r | wc -l)
  if (( CURRENT_JOBS <= MAX_EXISTING_JOBS_BEFORE_SUBMIT )); then
    break
  fi
  echo "Current expanded job count ${CURRENT_JOBS} exceeds ${MAX_EXISTING_JOBS_BEFORE_SUBMIT}; sleeping."
  sleep "${SUBMIT_WAIT_SECONDS}"
done

ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --array=0-$((TASK_COUNT - 1))%"${ARRAY_THROTTLE}" scripts/run_paper_benchmark_array.sh | awk '{print $4}'
)

COLLECT_JOB_ID=$(
  ROOT_SPECS_FILE="${ROOT_SPECS_FILE}" \
  OUT_DIR="${COLLECT_DIR}" \
  HORIZONS_CSV="100,500,1000" \
  GOOD_THRESHOLD="50" \
    sbatch --dependency=afterany:"${ARRAY_JOB_ID}" scripts/collect_transition_rich_basin_partition.sh | awk '{print $4}'
)

INTERP_SHARD_DIR="${INTERP_DIR}/shards/${MODEL_VARIANT}"
mkdir -p "${INTERP_SHARD_DIR}"
INTERP_SHARD_JOB_ID=$(
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  OUT_DIR="${INTERP_SHARD_DIR}" \
  ROOT_LABELS_CSV="${MODEL_VARIANT}" \
  PROGRESS_EVERY_RUNS="1" \
  FLUSH_EVERY_RUNS="5" \
  DEPTH_SLICE_MODE="per_basin" \
    sbatch \
      --dependency=afterok:"${COLLECT_JOB_ID}" \
      --job-name="tr_interp_smlpbd_fix" \
      --time="12:00:00" \
      --cpus-per-task="4" \
      --mem="16G" \
      --output="${INTERP_DIR}/logs/${MODEL_VARIANT}-%A.out" \
      --error="${INTERP_DIR}/logs/${MODEL_VARIANT}-%A.err" \
      scripts/reduce_transition_rich_interpretability_metrics.sh | awk '{print $4}'
)

INTERP_MERGE_JOB_ID=$(
  SHARDS_DIR="${INTERP_DIR}/shards" \
  OUT_DIR="${INTERP_DIR}" \
  ROWS_CSV="${COLLECT_DIR}/forecasting_rows.csv" \
  ROOT_LABELS_CSV="${MODEL_VARIANT}" \
    sbatch \
      --dependency=afterok:"${INTERP_SHARD_JOB_ID}" \
      --time="00:30:00" \
      --mem="4G" \
      --output="${INTERP_DIR}/logs/merge-%A.out" \
      --error="${INTERP_DIR}/logs/merge-%A.err" \
      scripts/merge_transition_rich_interpretability_shards.sh | awk '{print $4}'
)

cat > "${AUTOMATION_DIR}/sparse_mlp_bd_repaired_table1_queue.json" <<EOF
{
  "model_variant": "${MODEL_VARIANT}",
  "results_dir": "${RESULTS_DIR}",
  "base_out": "${BASE_OUT}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "root_specs_file": "${ROOT_SPECS_FILE}",
  "task_count": ${TASK_COUNT},
  "array_job_id": "${ARRAY_JOB_ID}",
  "collect_job_id": "${COLLECT_JOB_ID}",
  "interpretability_shard_job_id": "${INTERP_SHARD_JOB_ID}",
  "interpretability_merge_job_id": "${INTERP_MERGE_JOB_ID}",
  "systems_csv": "${SYSTEMS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "training_device": "cuda via scripts/run_paper_benchmark_array.sh"
}
EOF

{
  printf 'RESULTS_DIR=%q\n' "${RESULTS_DIR}"
  printf 'BASE_OUT=%q\n' "${BASE_OUT}"
  printf 'TASK_TSV=%q\n' "${TASK_TSV}"
  printf 'MANIFEST_JSON=%q\n' "${MANIFEST_JSON}"
  printf 'ROOT_SPECS_FILE=%q\n' "${ROOT_SPECS_FILE}"
  printf 'TASK_COUNT=%q\n' "${TASK_COUNT}"
  printf 'ARRAY_JOB_ID=%q\n' "${ARRAY_JOB_ID}"
  printf 'COLLECT_JOB_ID=%q\n' "${COLLECT_JOB_ID}"
  printf 'INTERP_SHARD_JOB_ID=%q\n' "${INTERP_SHARD_JOB_ID}"
  printf 'INTERP_MERGE_JOB_ID=%q\n' "${INTERP_MERGE_JOB_ID}"
} > "${QUEUE_LOG_DIR}/launch_record.env"

echo "Queued repaired Sparse MLP-BD Table 1 rerun."
echo "Training array: ${ARRAY_JOB_ID}"
echo "Collect job: ${COLLECT_JOB_ID}"
echo "Support diagnostics: ${INTERP_SHARD_JOB_ID} -> ${INTERP_MERGE_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
