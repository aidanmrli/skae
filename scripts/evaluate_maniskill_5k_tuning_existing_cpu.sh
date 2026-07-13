#!/usr/bin/env bash
# CPU-only eval pass for existing ManiSkill 5k tuning checkpoints.
#SBATCH --job-name=mskill5k_eval
#SBATCH --partition=long
#SBATCH --output=logs/maniskill_5k_eval_%j.out
#SBATCH --error=logs/maniskill_5k_eval_%j.err
#SBATCH --time=00:40:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

set -euo pipefail

cd /home/mila/l/lia/skae
mkdir -p logs

RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
DATASET="${DATASET:-data/maniskill/perturbation_assessment_seed0_e20/all_setups.npz}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-checkpoint.pt}"
EVAL_DIR_NAME="${EVAL_DIR_NAME:-eval_test_periodic_wide}"
HORIZONS="${HORIZONS:-10,20,30,40,50,75,100,125}"
PERIODIC_REENCODE_PERIODS="${PERIODIC_REENCODE_PERIODS:-1,2,5,10,20,50,100}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-0.2}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.4}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-4}"

if (( EVAL_CONCURRENCY <= 0 )); then
  echo "EVAL_CONCURRENCY must be positive, got ${EVAL_CONCURRENCY}" >&2
  exit 2
fi

export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

echo "date=$(date)"
echo "host=$(hostname)"
echo "commit=$(git rev-parse --short HEAD || true)"
echo "run_root=${RUN_ROOT}"
echo "dataset=${DATASET}"
echo "checkpoint_name=${CHECKPOINT_NAME}"
echo "eval_dir_name=${EVAL_DIR_NAME}"
echo "horizons=${HORIZONS}"
echo "periodic_reencode_periods=${PERIODIC_REENCODE_PERIODS}"
echo "eval_concurrency=${EVAL_CONCURRENCY}"

mapfile -t CHECKPOINTS < <(find "${RUN_ROOT}" -mindepth 3 -maxdepth 3 -path "*/seed*/${CHECKPOINT_NAME}" | sort)
if (( ${#CHECKPOINTS[@]} == 0 )); then
  echo "No ${CHECKPOINT_NAME} checkpoints found under ${RUN_ROOT}" >&2
  exit 1
fi
echo "checkpoint_count=${#CHECKPOINTS[@]}"

RUNNING_TASKS=0
FAILED_TASKS=0

wait_for_slot() {
  local status=0
  set +e
  wait -n
  status=$?
  set -e
  if (( status != 0 )); then
    FAILED_TASKS=1
  fi
  RUNNING_TASKS=$((RUNNING_TASKS - 1))
}

for checkpoint in "${CHECKPOINTS[@]}"; do
  run_dir="$(dirname "${checkpoint}")"
  output_dir="${run_dir}/${EVAL_DIR_NAME}"
  if [[ "${FORCE:-0}" != "1" && -f "${output_dir}/metrics_summary.json" ]]; then
    echo "Existing summary found; skipping ${checkpoint}"
    continue
  fi
  mkdir -p "${output_dir}"
  (
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    echo "evaluating checkpoint=${checkpoint}"
    uv run python tools/evaluate_maniskill_controlled_lista.py \
      --dataset "${DATASET}" \
      --checkpoint "${checkpoint}" \
      --output_dir "${output_dir}" \
      --device cpu \
      --split test \
      --horizons "${HORIZONS}" \
      --periodic_reencode_periods "${PERIODIC_REENCODE_PERIODS}" \
      --support_threshold "${SUPPORT_THRESHOLD}" \
      --family_jaccard "${FAMILY_JACCARD}"
    echo "summary=${output_dir}/metrics_summary.json"
  ) &
  RUNNING_TASKS=$((RUNNING_TASKS + 1))
  if (( RUNNING_TASKS >= EVAL_CONCURRENCY )); then
    wait_for_slot
  fi
done

while (( RUNNING_TASKS > 0 )); do
  wait_for_slot
done

exit "${FAILED_TASKS}"
