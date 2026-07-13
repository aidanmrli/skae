#!/usr/bin/env bash
#
# Queue support-threshold/Jaccard sweeps for existing spatialized PDE checkpoints.
#
# Submit with:
#   sbatch scripts/queue_spatialized_reaction_diffusion_support_sweep_existing.sh
#
# The sweep is evaluation-only: it does not retrain checkpoints.
#
#SBATCH --job-name=queue-spatial-rd-support
#SBATCH --output=/network/scratch/l/lia/skae/queue-spatial-rd-support-%A.out
#SBATCH --error=/network/scratch/l/lia/skae/queue-spatial-rd-support-%A.err
#SBATCH --time=00:20:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-spatial_rd_existing_support_sweep_20260525}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
TASK_TSV="${TASK_TSV:-${RESULTS_DIR}/support_sweep_tasks.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${RESULTS_DIR}/support_sweep_manifest.json}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"

INPUT_ROOTS_CSV="${INPUT_ROOTS_CSV:-/network/scratch/l/lia/skae/spatial_rd_conv_matched_dense_control_clean1_20260521,/network/scratch/l/lia/skae/spatial_rd_conv_matched_sparse_controls_clean1_20260521,/network/scratch/l/lia/skae/spatial_rd_conv_matched_dense_control_clean1_seed1_20260521,/network/scratch/l/lia/skae/spatial_rd_conv_matched_sparse_controls_clean1_seed1_20260521}"
SUPPORT_THRESHOLDS_CSV="${SUPPORT_THRESHOLDS_CSV:-0.01,0.03,0.05,0.1,0.2,0.3}"
FAMILY_JACCARDS_CSV="${FAMILY_JACCARDS_CSV:-0.3,0.4,0.5,0.6,0.7,0.8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEEP_THRESHOLD="${DEEP_THRESHOLD:-0.7}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-6}"
RUNNER_PARTITION="${RUNNER_PARTITION:-long}"
RUNNER_GRES="${RUNNER_GRES-gpu:1}"
RUNNER_TIME="${RUNNER_TIME:-01:00:00}"
RUNNER_MEM="${RUNNER_MEM:-8G}"
RUNNER_CPUS="${RUNNER_CPUS:-2}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/run_spatialized_reaction_diffusion_support_sweep_array.sh}"

mkdir -p "${RESULTS_DIR}" "${OUTPUT_ROOT}" "${LOG_DIR}"

uv run python tools/build_spatialized_reaction_diffusion_support_sweep_tasks.py \
  --input_roots_csv "${INPUT_ROOTS_CSV}" \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --output_root "${OUTPUT_ROOT}" \
  --support_thresholds_csv "${SUPPORT_THRESHOLDS_CSV}" \
  --family_jaccards_csv "${FAMILY_JACCARDS_CSV}" \
  --batch_size "${BATCH_SIZE}" \
  --deep_threshold "${DEEP_THRESHOLD}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if [[ "${TASK_COUNT}" -le 0 ]]; then
  echo "No tasks generated in ${TASK_TSV}."
  exit 1
fi

ARRAY_SPEC="0-$((TASK_COUNT - 1))%${ARRAY_THROTTLE}"
SBATCH_RUNNER_ARGS=(
  --parsable
  --array="${ARRAY_SPEC}"
  --partition="${RUNNER_PARTITION}"
  --time="${RUNNER_TIME}"
  --mem="${RUNNER_MEM}"
  --cpus-per-task="${RUNNER_CPUS}"
  --output="${LOG_DIR}/spatial-rd-support-%A_%a.out"
  --error="${LOG_DIR}/spatial-rd-support-%A_%a.err"
)
if [[ -n "${RUNNER_GRES}" ]]; then
  SBATCH_RUNNER_ARGS+=(--gres="${RUNNER_GRES}")
fi
ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
    sbatch \
      "${SBATCH_RUNNER_ARGS[@]}" \
      "${RUNNER_SCRIPT}"
)
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"

cat > "${RESULTS_DIR}/queue.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "results_dir": "${RESULTS_DIR}",
  "output_root": "${OUTPUT_ROOT}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "log_dir": "${LOG_DIR}",
  "input_roots_csv": "${INPUT_ROOTS_CSV}",
  "support_thresholds_csv": "${SUPPORT_THRESHOLDS_CSV}",
  "family_jaccards_csv": "${FAMILY_JACCARDS_CSV}",
  "task_count": ${TASK_COUNT},
  "array_spec": "${ARRAY_SPEC}",
  "runner_partition": "${RUNNER_PARTITION}",
  "runner_gres": "${RUNNER_GRES}",
  "runner_time": "${RUNNER_TIME}",
  "runner_script": "${RUNNER_SCRIPT}",
  "array_job_id": "${ARRAY_JOB_ID}"
}
EOF

echo "Queued spatialized PDE support sweep."
echo "Task count: ${TASK_COUNT}"
echo "Array job: ${ARRAY_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
echo "Output root: ${OUTPUT_ROOT}"
