#!/bin/bash
#
# Run initial-latent support-coordinate interventions on one checkpoint.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<output directory>
#
# Optional env vars:
#   ROOT_LABEL, SYSTEM, SEED
#   NUM_INITIAL_POINTS, NUM_CANDIDATE_TRAJECTORIES, TRAJECTORY_LENGTH
#   EVAL_SEED, ENDPOINT_ROLLOUT_STEPS, SUPPORT_DEFINITION, HORIZONS
#   MAX_DROP, RANDOM_SUPPORT_REPEATS, RANDOM_SEED, DEPTH_SLICE_MODE
#   REQUIRE_STABLE_TRUE_BASIN
#   DEVICE=cuda
#
# Unset scientific values resolve from
# experiments.neurips_2026.interventions.protocol.
#
#SBATCH --job-name=support_coord_int
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH -o slurm-%x-%A.out
#SBATCH -e slurm-%x-%A.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit this launcher with: sbatch scripts/neurips_2026/interventions/run.sh" >&2
  exit 2
fi

source scripts/common/gpu_guard.sh
trap gpu_guard_stop_sampler EXIT
module load cuda/12.6.0
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"

ROWS_CSV="${ROWS_CSV:?ROWS_CSV is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
ROOT_LABEL="${ROOT_LABEL:-}"
SYSTEM="${SYSTEM:-}"
SEED="${SEED:-}"
NUM_INITIAL_POINTS="${NUM_INITIAL_POINTS:-}"
NUM_CANDIDATE_TRAJECTORIES="${NUM_CANDIDATE_TRAJECTORIES:-}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-}"
EVAL_SEED="${EVAL_SEED:-}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-}"
SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-}"
HORIZONS="${HORIZONS:-}"
MAX_DROP="${MAX_DROP:-}"
RANDOM_SUPPORT_REPEATS="${RANDOM_SUPPORT_REPEATS:-}"
RANDOM_SEED="${RANDOM_SEED:-}"
DEPTH_SLICE_MODE="${DEPTH_SLICE_MODE:-}"
REQUIRE_STABLE_TRUE_BASIN="${REQUIRE_STABLE_TRUE_BASIN:-}"
DEVICE="${DEVICE:-cuda}"

gpu_guard_assert_cuda_visible "support-coordinate intervention replay"
gpu_guard_print_context "Support Coordinate Interventions"

echo "============================================="
echo "Support Coordinate Interventions"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "OUT_DIR: ${OUT_DIR}"
echo "Protocol: experiments.neurips_2026.interventions.protocol"
echo "Scientific settings: canonical defaults plus explicit environment overrides"
echo "DEVICE: ${DEVICE}"
echo "============================================="

ARGS=(
  --rows_csv "${ROWS_CSV}"
  --output_dir "${OUT_DIR}"
  --device "${DEVICE}"
)
append_override() {
  local flag="$1"
  local value="$2"
  if [[ -n "${value}" ]]; then
    ARGS+=("${flag}" "${value}")
  fi
}
append_override --root_label "${ROOT_LABEL}"
append_override --system "${SYSTEM}"
append_override --seed "${SEED}"
append_override --num_initial_points "${NUM_INITIAL_POINTS}"
append_override --num_candidate_trajectories "${NUM_CANDIDATE_TRAJECTORIES}"
append_override --trajectory_length "${TRAJECTORY_LENGTH}"
append_override --eval_seed "${EVAL_SEED}"
append_override --endpoint_rollout_steps "${ENDPOINT_ROLLOUT_STEPS}"
append_override --support_definition "${SUPPORT_DEFINITION}"
append_override --horizons "${HORIZONS}"
append_override --max_drop "${MAX_DROP}"
append_override --random_support_repeats "${RANDOM_SUPPORT_REPEATS}"
append_override --random_seed "${RANDOM_SEED}"
append_override --depth_slice_mode "${DEPTH_SLICE_MODE}"
if [[ "${REQUIRE_STABLE_TRUE_BASIN}" == "0" ]]; then
  ARGS+=(--no-require_stable_true_basin)
elif [[ "${REQUIRE_STABLE_TRUE_BASIN}" == "1" ]]; then
  ARGS+=(--require_stable_true_basin)
elif [[ -n "${REQUIRE_STABLE_TRUE_BASIN}" ]]; then
  echo "REQUIRE_STABLE_TRUE_BASIN must be 0, 1, or unset." >&2
  exit 2
fi

gpu_guard_start_sampler \
  "${OUT_DIR}/gpu_utilization_${SLURM_JOB_ID}.csv" \
  "${GPU_TELEMETRY_INTERVAL:-30}"
gpu_guard_phase "support-coordinate intervention start"
set +e
uv run skae-paper evaluate interventions "${ARGS[@]}"
EXIT_CODE=$?
set -e
gpu_guard_phase "support-coordinate intervention end exit_code=${EXIT_CODE}"
gpu_guard_stop_sampler
exit "${EXIT_CODE}"
