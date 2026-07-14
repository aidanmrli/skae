#!/bin/bash
#
# Run initial-latent support-coordinate interventions on one checkpoint.
#
# Required env vars:
#   ROWS_CSV=<forecasting_rows.csv>
#   OUT_DIR=<output directory>
#
# Optional env vars:
#   ROOT_LABEL=<model root label>
#   SYSTEM=<system_key>
#   SEED=<training seed>
#   NUM_INITIAL_POINTS=100
#   NUM_CANDIDATE_TRAJECTORIES=512
#   TRAJECTORY_LENGTH=64
#   SUPPORT_DEFINITION=absolute:0.001
#   HORIZONS=1,3,5,7,9,11,13,15,17,19,21
#   MAX_DROP=10
#   RANDOM_SUPPORT_REPEATS=20
#   DEVICE=cuda
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
ROOT_LABEL="${ROOT_LABEL:-lista_dense_signsplit_p256_hardinit_basin_partition}"
SYSTEM="${SYSTEM:-gated_local_linear}"
SEED="${SEED:-0}"
NUM_INITIAL_POINTS="${NUM_INITIAL_POINTS:-100}"
NUM_CANDIDATE_TRAJECTORIES="${NUM_CANDIDATE_TRAJECTORIES:-512}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-64}"
EVAL_SEED="${EVAL_SEED:-42}"
ENDPOINT_ROLLOUT_STEPS="${ENDPOINT_ROLLOUT_STEPS:-5000}"
SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-absolute:0.001}"
HORIZONS="${HORIZONS:-1,3,5,7,9,11,13,15,17,19,21}"
MAX_DROP="${MAX_DROP:-10}"
RANDOM_SUPPORT_REPEATS="${RANDOM_SUPPORT_REPEATS:-20}"
RANDOM_SEED="${RANDOM_SEED:-123}"
DEPTH_SLICE_MODE="${DEPTH_SLICE_MODE:-per_basin}"
REQUIRE_STABLE_TRUE_BASIN="${REQUIRE_STABLE_TRUE_BASIN:-1}"
DEVICE="${DEVICE:-cuda}"

gpu_guard_assert_cuda_visible "support-coordinate intervention replay"
gpu_guard_print_context "Support Coordinate Interventions"

STABLE_FLAG="--require_stable_true_basin"
if [[ "${REQUIRE_STABLE_TRUE_BASIN}" == "0" ]]; then
  STABLE_FLAG="--no-require_stable_true_basin"
fi

echo "============================================="
echo "Support Coordinate Interventions"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "ROWS_CSV: ${ROWS_CSV}"
echo "OUT_DIR: ${OUT_DIR}"
echo "ROOT_LABEL: ${ROOT_LABEL}"
echo "SYSTEM: ${SYSTEM}"
echo "SEED: ${SEED}"
echo "NUM_INITIAL_POINTS: ${NUM_INITIAL_POINTS}"
echo "SUPPORT_DEFINITION: ${SUPPORT_DEFINITION}"
echo "HORIZONS: ${HORIZONS}"
echo "MAX_DROP: ${MAX_DROP}"
echo "RANDOM_SUPPORT_REPEATS: ${RANDOM_SUPPORT_REPEATS}"
echo "DEPTH_SLICE_MODE: ${DEPTH_SLICE_MODE}"
echo "DEVICE: ${DEVICE}"
echo "============================================="

gpu_guard_start_sampler \
  "${OUT_DIR}/gpu_utilization_${SLURM_JOB_ID}.csv" \
  "${GPU_TELEMETRY_INTERVAL:-30}"
gpu_guard_phase "support-coordinate intervention start"
set +e
uv run skae-paper evaluate interventions \
  --rows_csv "${ROWS_CSV}" \
  --output_dir "${OUT_DIR}" \
  --root_label "${ROOT_LABEL}" \
  --system "${SYSTEM}" \
  --seed "${SEED}" \
  --num_initial_points "${NUM_INITIAL_POINTS}" \
  --num_candidate_trajectories "${NUM_CANDIDATE_TRAJECTORIES}" \
  --trajectory_length "${TRAJECTORY_LENGTH}" \
  --eval_seed "${EVAL_SEED}" \
  --endpoint_rollout_steps "${ENDPOINT_ROLLOUT_STEPS}" \
  --support_definition "${SUPPORT_DEFINITION}" \
  --horizons "${HORIZONS}" \
  --max_drop "${MAX_DROP}" \
  --random_support_repeats "${RANDOM_SUPPORT_REPEATS}" \
  --random_seed "${RANDOM_SEED}" \
  --depth_slice_mode "${DEPTH_SLICE_MODE}" \
  "${STABLE_FLAG}" \
  --device "${DEVICE}"
EXIT_CODE=$?
set -e
gpu_guard_phase "support-coordinate intervention end exit_code=${EXIT_CODE}"
gpu_guard_stop_sampler
exit "${EXIT_CODE}"
