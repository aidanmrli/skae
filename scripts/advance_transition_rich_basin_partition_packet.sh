#!/bin/bash
#
# Advance the transition-rich basin-partition confirmatory packet by one step.
# After a resolve pass completes, this launcher either:
# 1. queues the actual next dt-rescue pass plus its collect / resolve jobs, and
#    requeues itself on that next resolve; or
# 2. if no rescue is requested, queues the final interpretability reducer plus
#    the combined paper-facing LISTA-vs-control comparison summary.
#
# Required env vars:
#   CURRENT_PASS=<0,1,...>
#   RESULTS_DIR=results/<experiment_tag>
#   BASE_OUT=/network/scratch/l/lia/skae/<experiment_tag>
#
# Optional env vars:
#   TASK_DIR=<defaults to RESULTS_DIR/task_tables>
#   ROOT_SPEC_DIR=<defaults to RESULTS_DIR/root_specs>
#   RESOLVE_DIR=<defaults to RESULTS_DIR/dt_resolution>
#   MAX_HALVINGS=6
#   THRESHOLD=50
#   MIN_SEEDS=1
#   NUM_STEPS_OVERRIDE=<inferred from task table if unset>
#   SEEDS_CSV=<inferred from task table if unset>
#   MODEL_VARIANTS_CSV=<inferred from task table if unset>
#   EVAL_PROFILE=<inferred from task table if unset>
#   CANDIDATE_ROOTS_CSV=lista_blockdiag_signsplit_hardinit_basin_partition,lista_dense_softblock_signsplit_p64_hardinit_basin_partition
#   CONTROL_ROOTS_CSV=mlp_sparse_basin_partition_control,mlp_zero_sparse_basin_partition_control
#   SUPPORT_SCHEME=absolute:0.001
#   SUBSET=deep
#   GOOD_THRESHOLD=50
#
# Submit with:
#   CURRENT_PASS=0 RESULTS_DIR=results/<tag> BASE_OUT=/network/scratch/... \
#     sbatch --dependency=afterany:<resolve_jobid> \
#       scripts/advance_transition_rich_basin_partition_packet.sh
#
#SBATCH --job-name=advance_tr_bp
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:45:00
#SBATCH -o /network/scratch/l/lia/skae/advance-transition-rich-%A.out
#SBATCH -e /network/scratch/l/lia/skae/advance-transition-rich-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"

CURRENT_PASS="${CURRENT_PASS:?CURRENT_PASS is required}"
RESULTS_DIR="${RESULTS_DIR:?RESULTS_DIR is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"

TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
RESOLVE_DIR="${RESOLVE_DIR:-${RESULTS_DIR}/dt_resolution}"
MAX_HALVINGS="${MAX_HALVINGS:-6}"
THRESHOLD="${THRESHOLD:-50}"
MIN_SEEDS="${MIN_SEEDS:-1}"
NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-}"
EVAL_PROFILE="${EVAL_PROFILE:-}"
CANDIDATE_ROOTS_CSV="${CANDIDATE_ROOTS_CSV:-lista_blockdiag_signsplit_hardinit_basin_partition,lista_dense_softblock_signsplit_p64_hardinit_basin_partition}"
CONTROL_ROOTS_CSV="${CONTROL_ROOTS_CSV:-mlp_sparse_basin_partition_control,mlp_zero_sparse_basin_partition_control}"
SUPPORT_SCHEME="${SUPPORT_SCHEME:-absolute:0.001}"
SUBSET="${SUBSET:-deep}"
GOOD_THRESHOLD="${GOOD_THRESHOLD:-50}"

DEFAULT_TSV="${TASK_DIR}/transition_rich_basin_partition.tsv"
mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${RESOLVE_DIR}" "${RESULTS_DIR}/automation"

infer_from_default_tsv() {
  local tsv_path="$1"
  uv run python - "${tsv_path}" <<'PY'
import csv
import shlex
import sys

path = sys.argv[1]
with open(path, newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))

num_steps = sorted({row["num_steps"] for row in rows if row.get("num_steps")})
seeds = sorted({int(row["seed"]) for row in rows if row.get("seed") not in (None, "")})
models = sorted({row["model_variant"] for row in rows if row.get("model_variant")})
eval_profiles = sorted({row["eval_profile"] for row in rows if row.get("eval_profile")})

print(f"INFERRED_NUM_STEPS_OVERRIDE={shlex.quote(num_steps[0] if len(num_steps) == 1 else '')}")
print(f"INFERRED_SEEDS_CSV={shlex.quote(','.join(str(seed) for seed in seeds))}")
print(f"INFERRED_MODEL_VARIANTS_CSV={shlex.quote(','.join(models))}")
print(f"INFERRED_EVAL_PROFILE={shlex.quote(eval_profiles[0] if len(eval_profiles) == 1 else '')}")
PY
}

if [[ -z "${NUM_STEPS_OVERRIDE}" || -z "${SEEDS_CSV}" || -z "${MODEL_VARIANTS_CSV}" || -z "${EVAL_PROFILE}" ]]; then
  if [[ ! -f "${DEFAULT_TSV}" ]]; then
    echo "Cannot infer defaults because ${DEFAULT_TSV} is missing."
    exit 1
  fi
  eval "$(infer_from_default_tsv "${DEFAULT_TSV}")"
  NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE:-${INFERRED_NUM_STEPS_OVERRIDE:-}}"
  SEEDS_CSV="${SEEDS_CSV:-${INFERRED_SEEDS_CSV:-}}"
  MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-${INFERRED_MODEL_VARIANTS_CSV:-}}"
  EVAL_PROFILE="${EVAL_PROFILE:-${INFERRED_EVAL_PROFILE:-full}}"
fi

NEXT_PASS=$((CURRENT_PASS + 1))
REQUEST_TSV="${RESOLVE_DIR}/pass${CURRENT_PASS}/dt_rescue_request_pass${NEXT_PASS}.tsv"
NEXT_TASK_TSV="${TASK_DIR}/transition_rich_rescue_pass${NEXT_PASS}.tsv"
NEXT_MANIFEST_JSON="${TASK_DIR}/transition_rich_rescue_pass${NEXT_PASS}_manifest.json"
FINAL_ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/transition_rich_collect_pass${CURRENT_PASS}_roots.txt"
if (( CURRENT_PASS == 0 )); then
  FINAL_ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/transition_rich_collect_pass0_roots.txt"
fi
AUTOMATION_DIR="${RESULTS_DIR}/automation"

write_root_specs() {
  local output_file="$1"
  local max_rescue_pass="$2"
  local labels_csv="$3"
  IFS=',' read -r -a labels <<< "${labels_csv}"
  : > "${output_file}"
  for model_variant in "${labels[@]}"; do
    [[ -z "${model_variant}" ]] && continue
    echo "${model_variant}=${BASE_OUT}/transition_rich_basin_partition/${model_variant}" >> "${output_file}"
    local pass_index
    for ((pass_index=1; pass_index<=max_rescue_pass; pass_index++)); do
      echo "${model_variant}=${BASE_OUT}/rescue_pass${pass_index}/${model_variant}" >> "${output_file}"
    done
  done
}

count_tsv_rows() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo 0
    return
  fi
  local lines
  lines=$(wc -l < "${path}")
  if (( lines <= 1 )); then
    echo 0
  else
    echo $((lines - 1))
  fi
}

echo "============================================="
echo "Advance Transition-Rich Basin Partition Packet"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "CURRENT_PASS: ${CURRENT_PASS}"
echo "RESULTS_DIR: ${RESULTS_DIR}"
echo "BASE_OUT: ${BASE_OUT}"
echo "REQUEST_TSV: ${REQUEST_TSV}"
echo "NUM_STEPS_OVERRIDE: ${NUM_STEPS_OVERRIDE:-<default>}"
echo "SEEDS_CSV: ${SEEDS_CSV:-<none>}"
echo "MODEL_VARIANTS_CSV: ${MODEL_VARIANTS_CSV:-<none>}"
echo "EVAL_PROFILE: ${EVAL_PROFILE}"
echo "============================================="

if (( NEXT_PASS <= MAX_HALVINGS )); then
  if [[ ! -f "${REQUEST_TSV}" ]]; then
    echo "Missing resolve output: ${REQUEST_TSV}"
    exit 1
  fi
  REQUEST_ROWS=$(count_tsv_rows "${REQUEST_TSV}")
else
  REQUEST_ROWS=0
fi

if (( NEXT_PASS <= MAX_HALVINGS && REQUEST_ROWS > 0 )); then
  if [[ ! -f "${NEXT_TASK_TSV}" ]]; then
    BUILD_ARGS=(
      --output_tsv "${NEXT_TASK_TSV}"
      --output_manifest_json "${NEXT_MANIFEST_JSON}"
      --phase_label "rescue_pass${NEXT_PASS}"
      --eval_profile "${EVAL_PROFILE}"
      --dt_table "${REQUEST_TSV}"
      --dt_column requested_dt
    )
    if [[ -n "${NUM_STEPS_OVERRIDE}" ]]; then
      BUILD_ARGS+=(--num_steps_override "${NUM_STEPS_OVERRIDE}")
    fi
    if [[ -n "${SEEDS_CSV}" ]]; then
      BUILD_ARGS+=(--seeds_csv "${SEEDS_CSV}")
    fi
    if [[ -n "${MODEL_VARIANTS_CSV}" ]]; then
      BUILD_ARGS+=(--model_variants_csv "${MODEL_VARIANTS_CSV}")
    fi
    uv run python tools/build_transition_rich_basin_partition_tasks.py "${BUILD_ARGS[@]}"
  fi

  TASK_ROWS=$(count_tsv_rows "${NEXT_TASK_TSV}")
  if (( TASK_ROWS == 0 )); then
    echo "Rescue requested but ${NEXT_TASK_TSV} has no rows."
    exit 1
  fi

  NEXT_ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/transition_rich_collect_pass${NEXT_PASS}_roots.txt"
  write_root_specs "${NEXT_ROOT_SPECS_FILE}" "${NEXT_PASS}" "${MODEL_VARIANTS_CSV}"

  RESCUE_JOB_ID=$(
    TASK_TSV="${NEXT_TASK_TSV}" BASE_OUT="${BASE_OUT}" \
      sbatch --array=0-$((TASK_ROWS - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}'
  )

  COLLECT_JOB_ID=$(
    ROOT_SPECS_FILE="${NEXT_ROOT_SPECS_FILE}" \
    OUT_DIR="${RESULTS_DIR}/collect_pass${NEXT_PASS}" \
    GOOD_THRESHOLD="${THRESHOLD}" \
      sbatch --dependency=afterany:"${RESCUE_JOB_ID}" scripts/collect_transition_rich_basin_partition.sh | awk '{print $4}'
  )

  RESOLVE_JOB_ID=$(
    ROWS_CSV="${RESULTS_DIR}/collect_pass${NEXT_PASS}/forecasting_rows.csv" \
    OUT_DIR="${RESOLVE_DIR}/pass${NEXT_PASS}" \
    CURRENT_PASS="${NEXT_PASS}" \
    MAX_HALVINGS="${MAX_HALVINGS}" \
    THRESHOLD="${THRESHOLD}" \
    MIN_SEEDS="${MIN_SEEDS}" \
    NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE}" \
    MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV}" \
    NEXT_TASK_TSV="${TASK_DIR}/transition_rich_rescue_pass$((NEXT_PASS + 1)).tsv" \
    MANIFEST_JSON="${TASK_DIR}/transition_rich_basin_partition_manifest.json" \
    SEEDS_CSV="${SEEDS_CSV}" \
    EVAL_PROFILE="${EVAL_PROFILE}" \
      sbatch --dependency=afterany:"${COLLECT_JOB_ID}" scripts/resolve_transition_rich_basin_partition_dt.sh | awk '{print $4}'
  )

  ADVANCE_JOB_ID=$(
    CURRENT_PASS="${NEXT_PASS}" \
    RESULTS_DIR="${RESULTS_DIR}" \
    BASE_OUT="${BASE_OUT}" \
    TASK_DIR="${TASK_DIR}" \
    ROOT_SPEC_DIR="${ROOT_SPEC_DIR}" \
    RESOLVE_DIR="${RESOLVE_DIR}" \
    MAX_HALVINGS="${MAX_HALVINGS}" \
    THRESHOLD="${THRESHOLD}" \
    MIN_SEEDS="${MIN_SEEDS}" \
    NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE}" \
    SEEDS_CSV="${SEEDS_CSV}" \
    MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV}" \
    EVAL_PROFILE="${EVAL_PROFILE}" \
    CANDIDATE_ROOTS_CSV="${CANDIDATE_ROOTS_CSV}" \
    CONTROL_ROOTS_CSV="${CONTROL_ROOTS_CSV}" \
    SUPPORT_SCHEME="${SUPPORT_SCHEME}" \
    SUBSET="${SUBSET}" \
    GOOD_THRESHOLD="${GOOD_THRESHOLD}" \
      sbatch --dependency=afterany:"${RESOLVE_JOB_ID}" scripts/advance_transition_rich_basin_partition_packet.sh | awk '{print $4}'
  )

  cat > "${AUTOMATION_DIR}/advance_pass${CURRENT_PASS}.json" <<EOF
{
  "current_pass": ${CURRENT_PASS},
  "next_pass": ${NEXT_PASS},
  "request_rows": ${REQUEST_ROWS},
  "rescue_job_id": "${RESCUE_JOB_ID}",
  "collect_job_id": "${COLLECT_JOB_ID}",
  "resolve_job_id": "${RESOLVE_JOB_ID}",
  "advance_job_id": "${ADVANCE_JOB_ID}"
}
EOF

  echo "Queued rescue continuation for pass ${NEXT_PASS}."
  echo "Request rows: ${REQUEST_ROWS}"
  echo "Rescue array: ${RESCUE_JOB_ID}"
  echo "Collect pass${NEXT_PASS}: ${COLLECT_JOB_ID}"
  echo "Resolve pass${NEXT_PASS}: ${RESOLVE_JOB_ID}"
  echo "Advance watcher: ${ADVANCE_JOB_ID}"
  exit 0
fi

FINAL_PASS="${CURRENT_PASS}"
FINAL_COLLECT_DIR="${RESULTS_DIR}/collect_pass${FINAL_PASS}"
FINAL_ROWS_CSV="${FINAL_COLLECT_DIR}/forecasting_rows.csv"
if [[ ! -f "${FINAL_ROWS_CSV}" ]]; then
  echo "Missing final forecasting rows: ${FINAL_ROWS_CSV}"
  exit 1
fi

write_root_specs "${FINAL_ROOT_SPECS_FILE}" "${FINAL_PASS}" "${MODEL_VARIANTS_CSV}"

FINAL_INTERP_DIR="${RESULTS_DIR}/interpretability_final_pass${FINAL_PASS}"
FINAL_COMPARE_DIR="${RESULTS_DIR}/final_comparison_pass${FINAL_PASS}"

REDUCE_JOB_ID=$(
  ROWS_CSV="${FINAL_ROWS_CSV}" \
  OUT_DIR="${FINAL_INTERP_DIR}" \
  ROOT_LABELS_FILE="${FINAL_ROOT_SPECS_FILE}" \
  SEEDS_CSV="${SEEDS_CSV}" \
    sbatch scripts/reduce_transition_rich_interpretability_metrics.sh | awk '{print $4}'
)

SUMMARY_JOB_ID=$(
  FORECAST_ROWS_CSV="${FINAL_ROWS_CSV}" \
  INTERPRETABILITY_ROWS_CSV="${FINAL_INTERP_DIR}/interpretability_rows.csv" \
  OUT_DIR="${FINAL_COMPARE_DIR}" \
  CANDIDATE_ROOTS_CSV="${CANDIDATE_ROOTS_CSV}" \
  CONTROL_ROOTS_CSV="${CONTROL_ROOTS_CSV}" \
  SUPPORT_SCHEME="${SUPPORT_SCHEME}" \
  SUBSET="${SUBSET}" \
  GOOD_THRESHOLD="${GOOD_THRESHOLD}" \
    sbatch --dependency=afterok:"${REDUCE_JOB_ID}" scripts/summarize_transition_rich_final_comparison.sh | awk '{print $4}'
)

cat > "${AUTOMATION_DIR}/advance_pass${CURRENT_PASS}.json" <<EOF
{
  "current_pass": ${CURRENT_PASS},
  "finalized": true,
  "request_rows": ${REQUEST_ROWS},
  "final_pass": ${FINAL_PASS},
  "reduce_job_id": "${REDUCE_JOB_ID}",
  "summary_job_id": "${SUMMARY_JOB_ID}",
  "final_rows_csv": "${FINAL_ROWS_CSV}",
  "final_interpretability_dir": "${FINAL_INTERP_DIR}",
  "final_comparison_dir": "${FINAL_COMPARE_DIR}"
}
EOF

echo "No further rescue requested. Finalized packet at pass ${FINAL_PASS}."
echo "Interpretability reducer: ${REDUCE_JOB_ID}"
echo "Final comparison summary: ${SUMMARY_JOB_ID}"
