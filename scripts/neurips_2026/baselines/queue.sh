#!/bin/bash
#
# Queue the standalone-baseline task contract used by the paper. Scientific
# defaults are resolved by experiments.neurips_2026.baselines.tasks.
#
# Submit with:
#   sbatch scripts/neurips_2026/baselines/queue.sh
#
# Optional env vars:
#   EXPERIMENT_TAG=paper_baseline_retained15_20260512
#   SYSTEMS_CSV=<explicit system subset override>
#   SEEDS_CSV=<explicit seed subset override>
#   BASELINE_FAMILIES=<explicit family subset override>
#   HORIZONS, NUM_TRAJECTORIES, TRAJECTORY_LENGTH, TRAIN_FRACTION
#   RIDGE_LAMBDA, EDMD_DEGREE, KERNEL_CENTERS, KERNEL_GAMMA
#   MAX_TRAIN_PAIRS, NUM_COMPONENTS, COMPONENT_MODE, ENV_DT
#   DYSTS_DT_MULTIPLIER, DYSTS_STANDARDIZE, CONFIG_NAME, TORCH_THREADS
#   ARRAY_THROTTLE=32
#
#SBATCH --job-name=queue_paper_base
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH -o slurm-%x-%A.out
#SBATCH -e slurm-%x-%A.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/neurips_2026/baselines/queue.sh"
  exit 2
fi

source .venv/bin/activate

EXPERIMENT_TAG="${EXPERIMENT_TAG:-paper_baseline_retained15_20260512}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
TASK_TSV="${TASK_TSV:-${RESULTS_DIR}/paper_baseline_tasks.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${RESULTS_DIR}/paper_baseline_manifest.json}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
BASE_OUT="${BASE_OUT:-${RESULTS_DIR}/runs}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
SEEDS_CSV="${SEEDS_CSV:-}"
BASELINE_FAMILIES="${BASELINE_FAMILIES:-}"
HORIZONS="${HORIZONS:-}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-}"
TRAIN_FRACTION="${TRAIN_FRACTION:-}"
RIDGE_LAMBDA="${RIDGE_LAMBDA:-}"
EDMD_DEGREE="${EDMD_DEGREE:-}"
KERNEL_CENTERS="${KERNEL_CENTERS:-}"
KERNEL_GAMMA="${KERNEL_GAMMA:-}"
MAX_TRAIN_PAIRS="${MAX_TRAIN_PAIRS:-}"
NUM_COMPONENTS="${NUM_COMPONENTS:-}"
COMPONENT_MODE="${COMPONENT_MODE:-}"
ENV_DT="${ENV_DT:-}"
DYSTS_DT_MULTIPLIER="${DYSTS_DT_MULTIPLIER:-}"
DYSTS_STANDARDIZE="${DYSTS_STANDARDIZE:-}"
CONFIG_NAME="${CONFIG_NAME:-}"
TORCH_THREADS="${TORCH_THREADS:-}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-32}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${BASE_OUT}"

TASK_ARGS=(
  --output_tsv "${TASK_TSV}"
  --output_manifest_json "${MANIFEST_JSON}"
)
append_override() {
  local flag="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    TASK_ARGS+=("${flag}" "${value}")
  fi
}
append_override --systems "${SYSTEMS_CSV}"
append_override --seeds "${SEEDS_CSV}"
append_override --baseline_families "${BASELINE_FAMILIES}"
append_override --horizons "${HORIZONS}"
append_override --num_trajectories "${NUM_TRAJECTORIES}"
append_override --trajectory_length "${TRAJECTORY_LENGTH}"
append_override --train_fraction "${TRAIN_FRACTION}"
append_override --ridge_lambda "${RIDGE_LAMBDA}"
append_override --edmd_degree "${EDMD_DEGREE}"
append_override --kernel_centers "${KERNEL_CENTERS}"
append_override --kernel_gamma "${KERNEL_GAMMA}"
append_override --max_train_pairs "${MAX_TRAIN_PAIRS}"
append_override --num_components "${NUM_COMPONENTS}"
append_override --component_mode "${COMPONENT_MODE}"
append_override --env_dt "${ENV_DT}"
append_override --dysts_dt_multiplier "${DYSTS_DT_MULTIPLIER}"
append_override --dysts_standardize "${DYSTS_STANDARDIZE}"
append_override --config_name "${CONFIG_NAME}"
append_override --torch_threads "${TORCH_THREADS}"
uv run skae-paper tasks baselines "${TASK_ARGS[@]}"

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
      scripts/neurips_2026/baselines/run_array.sh
)
BASELINE_JOB_ID="${BASELINE_JOB_ID%%;*}"

cat > "${RESULTS_DIR}/queue.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "results_dir": "${RESULTS_DIR}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "base_out": "${BASE_OUT}",
  "systems_override": "${SYSTEMS_CSV}",
  "resolved_roster_source": "${MANIFEST_JSON}",
  "seeds_override": "${SEEDS_CSV}",
  "baseline_families_override": "${BASELINE_FAMILIES}",
  "task_count": ${TASK_COUNT},
  "array_spec": "${ARRAY_SPEC}",
  "baseline_job_id": "${BASELINE_JOB_ID}"
}
EOF

echo "Queued standalone paper baseline suite."
echo "Task count: ${TASK_COUNT}"
echo "Array job: ${BASELINE_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
