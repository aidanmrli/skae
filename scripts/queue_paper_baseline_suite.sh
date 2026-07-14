#!/bin/bash
#
# Queue standalone paper baselines for three seeds on the retained
# 15-system multibasin benchmark used by docs/neurips_sparse_koopman_multibasin.tex.
#
# Submit with:
#   sbatch scripts/queue_paper_baseline_suite.sh
#
# Optional env vars:
#   EXPERIMENT_TAG=paper_baseline_retained15_20260512
#   SYSTEMS_CSV=gated_local_linear,gated_transfer_linear,claude:arrested_spiral,...
#   SEEDS_CSV=0,1,2
#   BASELINE_FAMILIES=classical_koopman,mixture_local_linear
#   DYSTS_DT_MULTIPLIER=30
#   DYSTS_STANDARDIZE=1
#   ARRAY_THROTTLE=32
#
#SBATCH --job-name=queue_paper_base
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o /network/scratch/l/lia/skae/queue-paper-baseline-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-paper-baseline-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_paper_baseline_suite.sh"
  exit 2
fi

source .venv/bin/activate

EXPERIMENT_TAG="${EXPERIMENT_TAG:-paper_baseline_retained15_20260512}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_TSV="${TASK_TSV:-${RESULTS_DIR}/paper_baseline_tasks.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${RESULTS_DIR}/paper_baseline_manifest.json}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
BASE_OUT="${BASE_OUT:-${RESULTS_DIR}/runs}"
SYSTEMS_CSV="${SYSTEMS_CSV:-gated_local_linear,gated_transfer_linear,claude:arrested_spiral,claude:cal_asymmetric_3,claude:cal_high_cross_3,claude:cal_hexagon_6,claude:cal_octagon_8,claude:cal_pentagon_5,claude:cal_square_4,claude:duffing_triple_well,claude:snic_multi,claude:transition_routes_4,claude:var_depth_gradient_4,claude:var_diamond_4,claude:var_l_shape_5}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
BASELINE_FAMILIES="${BASELINE_FAMILIES:-classical_koopman,mixture_local_linear}"
HORIZONS="${HORIZONS:-100,500,1000}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-256}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-1000}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.6}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-1e-6}"
EDMD_DEGREE="${EDMD_DEGREE:-3}"
KERNEL_CENTERS="${KERNEL_CENTERS:-128}"
KERNEL_GAMMA="${KERNEL_GAMMA:-0.0}"
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-0}"
NUM_COMPONENTS="${NUM_COMPONENTS:-4}"
COMPONENT_MODE="${COMPONENT_MODE:-fixed}"
ENV_DT="${ENV_DT:-0.0}"
DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER:-0.0}"
DYSTS_STANDARDIZE="${DYSTS_STANDARDIZE:-0}"
CONFIG_NAME="${CONFIG_NAME:-default}"
TORCH_THREADS="${TORCH_THREADS:-1}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-32}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${BASE_OUT}"

uv run python tools/build_paper_baseline_tasks.py \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --systems "${SYSTEMS_CSV}" \
  --seeds "${SEEDS_CSV}" \
  --baseline_families "${BASELINE_FAMILIES}" \
  --horizons "${HORIZONS}" \
  --num_trajectories "${NUM_TRAJECTORIES}" \
  --trajectory_length "${TRAJECTORY_LENGTH}" \
  --train_fraction "${TRAIN_FRACTION}" \
  --ridge_lambda "${RIDGE_LAMBDA}" \
  --edmd_degree "${EDMD_DEGREE}" \
  --kernel_centers "${KERNEL_CENTERS}" \
  --kernel_gamma "${KERNEL_GAMMA}" \
  --max_train_pairs "${MAX_TRAIN_PAIRS}" \
  --num_components "${NUM_COMPONENTS}" \
  --component_mode "${COMPONENT_MODE}" \
  --env_dt "${ENV_DT}" \
  --dysts_dt_multiplier "${DYSTS_DT_MULTIPLIER}" \
  --dysts_standardize "${DYSTS_STANDARDIZE}" \
  --config_name "${CONFIG_NAME}" \
  --torch_threads "${TORCH_THREADS}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if [[ "${TASK_COUNT}" -le 0 ]]; then
  echo "No tasks generated in ${TASK_TSV}."
  exit 1
fi

ARRAY_SPEC="0-$((TASK_COUNT - 1))%${ARRAY_THROTTLE}"
BASELINE_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
  BASE_OUT="${BASE_OUT}" \
    sbatch \
      --parsable \
      --array="${ARRAY_SPEC}" \
      --output="${LOG_DIR}/baseline-%A_%a.out" \
      --error="${LOG_DIR}/baseline-%A_%a.err" \
      scripts/run_paper_baseline_suite.sh
)
BASELINE_JOB_ID="${BASELINE_JOB_ID%%;*}"

cat > "${RESULTS_DIR}/queue.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "results_dir": "${RESULTS_DIR}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "base_out": "${BASE_OUT}",
  "systems_csv": "${SYSTEMS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "baseline_families": "${BASELINE_FAMILIES}",
  "dysts_dt_multiplier": "${DYSTS_DT_MULTIPLIER}",
  "dysts_standardize": "${DYSTS_STANDARDIZE}",
  "task_count": ${TASK_COUNT},
  "array_spec": "${ARRAY_SPEC}",
  "baseline_job_id": "${BASELINE_JOB_ID}"
}
EOF

echo "Queued standalone paper baseline suite."
echo "Task count: ${TASK_COUNT}"
echo "Array job: ${BASELINE_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
