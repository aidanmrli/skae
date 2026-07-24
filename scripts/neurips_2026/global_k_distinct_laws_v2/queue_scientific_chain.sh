#!/bin/bash
# Submit full mixed training -> audit -> CPU H/G evaluation -> complete summary.

#SBATCH --job-name=gkv2_queue
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"

SMOKE_DECISION="${SMOKE_DECISION:?SMOKE_DECISION is required}"
CARD_PATH="${CARD_PATH:-experiments/neurips_2026/global_k_distinct_laws_v2_card.json}"
SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH:?SOURCE_LOCK_PATH is required}"
TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
TASK_MANIFEST="${TASK_MANIFEST:?TASK_MANIFEST is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
RESULT_ROOT="${RESULT_ROOT:?RESULT_ROOT is required}"
source scripts/common/cluster_env.sh
uv run python -m experiments.neurips_2026.global_k_distinct_laws_v2_preflight queue \
  --card "${CARD_PATH}" \
  --source_lock "${SOURCE_LOCK_PATH}" \
  --smoke_decision "${SMOKE_DECISION}" \
  --task_tsv "${TASK_TSV}" \
  --task_manifest "${TASK_MANIFEST}"
SOURCE_LOCK_SHA="$(sha256sum "${SOURCE_LOCK_PATH}" | awk '{print $1}')"
CARD_SHA="$(sha256sum "${CARD_PATH}" | awk '{print $1}')"

mkdir -p "${RESULT_ROOT}/slurm" "${RESULT_ROOT}/training_pack" "${RESULT_ROOT}/telemetry" "${RESULT_ROOT}/audit" "${RESULT_ROOT}/evaluation" "${RESULT_ROOT}/summary" "${RESULT_ROOT}/packet"
TRAIN_JOB_ID=$(
  sbatch --parsable --time=3-00:00:00 \
    --export=ALL,MODE=full,TASK_TSV="${TASK_TSV}",TASK_MANIFEST="${TASK_MANIFEST}",BASE_OUT="${BASE_OUT}",PACK_ROOT="${RESULT_ROOT}/training_pack",SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH}",EXPECTED_SOURCE_LOCK_SHA="${SOURCE_LOCK_SHA}",CARD_PATH="${CARD_PATH}",GPU_TELEMETRY_INTERVAL=15 \
    --output="${RESULT_ROOT}/slurm/train_%j.out" --error="${RESULT_ROOT}/slurm/train_%j.err" \
    scripts/neurips_2026/global_k_distinct_laws_v2/run_mixed_pack.sh
)
TELEMETRY_JOB_ID=$(
  sbatch --parsable --dependency="afterok:${TRAIN_JOB_ID}" \
    --export=ALL,TASK_TSV="${TASK_TSV}",TASK_MANIFEST="${TASK_MANIFEST}",PACK_ROOT="${RESULT_ROOT}/training_pack",OUTPUT_JSON="${RESULT_ROOT}/telemetry/scientific_gpu_assessment.json",SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH}",EXPECTED_SOURCE_LOCK_SHA="${SOURCE_LOCK_SHA}",CARD_PATH="${CARD_PATH}" \
    --output="${RESULT_ROOT}/slurm/telemetry_%j.out" --error="${RESULT_ROOT}/slurm/telemetry_%j.err" \
    scripts/neurips_2026/global_k_distinct_laws_v2/assess_scientific_gpu.sh
)
AUDIT_JOB_ID=$(
  sbatch --parsable --dependency="afterok:${TELEMETRY_JOB_ID}" --array=0-19%20 \
    --export=ALL,TASK_TSV="${TASK_TSV}",BASE_OUT="${BASE_OUT}",OUTPUT_DIR="${RESULT_ROOT}/audit",SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH}",EXPECTED_SOURCE_LOCK_SHA="${SOURCE_LOCK_SHA}",CARD_PATH="${CARD_PATH}" \
    --output="${RESULT_ROOT}/slurm/audit_%A_%a.out" --error="${RESULT_ROOT}/slurm/audit_%A_%a.err" \
    scripts/neurips_2026/global_k_distinct_laws_v2/run_checkpoint_audit.sh
)
AUDIT_SUMMARY_JOB_ID=$(
  sbatch --parsable --dependency="afterok:${AUDIT_JOB_ID}" \
    --export=ALL,AUDIT_MODE=summary,TASK_TSV="${TASK_TSV}",BASE_OUT="${BASE_OUT}",OUTPUT_DIR="${RESULT_ROOT}/audit",SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH}",EXPECTED_SOURCE_LOCK_SHA="${SOURCE_LOCK_SHA}",CARD_PATH="${CARD_PATH}" \
    --output="${RESULT_ROOT}/slurm/audit_summary_%j.out" --error="${RESULT_ROOT}/slurm/audit_summary_%j.err" \
    scripts/neurips_2026/global_k_distinct_laws_v2/run_checkpoint_audit.sh
)
EVAL_JOB_ID=$(
  sbatch --parsable --dependency="afterok:${AUDIT_SUMMARY_JOB_ID}" --array=0-19%20 \
    --export=ALL,TASK_TSV="${TASK_TSV}",BASE_OUT="${BASE_OUT}",AUDIT_DIR="${RESULT_ROOT}/audit",OUTPUT_DIR="${RESULT_ROOT}/evaluation",SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH}",EXPECTED_SOURCE_LOCK_SHA="${SOURCE_LOCK_SHA}",CARD_PATH="${CARD_PATH}" \
    --output="${RESULT_ROOT}/slurm/eval_%A_%a.out" --error="${RESULT_ROOT}/slurm/eval_%A_%a.err" \
    scripts/neurips_2026/global_k_distinct_laws_v2/run_evaluation.sh
)
SUMMARY_JOB_ID=$(
  sbatch --parsable --dependency="afterok:${EVAL_JOB_ID}" \
    --export=ALL,TASK_TSV="${TASK_TSV}",AUDIT_DIR="${RESULT_ROOT}/audit",INPUT_DIR="${RESULT_ROOT}/evaluation",OUTPUT_JSON="${RESULT_ROOT}/summary/decision.json",SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH}",EXPECTED_SOURCE_LOCK_SHA="${SOURCE_LOCK_SHA}",CARD_PATH="${CARD_PATH}" \
    --output="${RESULT_ROOT}/slurm/summary_%j.out" --error="${RESULT_ROOT}/slurm/summary_%j.err" \
    scripts/neurips_2026/global_k_distinct_laws_v2/run_summary.sh
)
PACKET_JOB_ID=$(
  sbatch --parsable --dependency="afterok:${SUMMARY_JOB_ID}" \
    --export=ALL,TASK_TSV="${TASK_TSV}",AUDIT_SUMMARY="${RESULT_ROOT}/audit/summary.json",EVALUATION_DIR="${RESULT_ROOT}/evaluation",DECISION_JSON="${RESULT_ROOT}/summary/decision.json",TELEMETRY_ASSESSMENT="${RESULT_ROOT}/telemetry/scientific_gpu_assessment.json",OUTPUT_JSON="${RESULT_ROOT}/packet/distinct_laws_v2_packet.json",SOURCE_LOCK_PATH="${SOURCE_LOCK_PATH}",EXPECTED_SOURCE_LOCK_SHA="${SOURCE_LOCK_SHA}",CARD_PATH="${CARD_PATH}" \
    --output="${RESULT_ROOT}/slurm/packet_%j.out" --error="${RESULT_ROOT}/slurm/packet_%j.err" \
    scripts/neurips_2026/global_k_distinct_laws_v2/run_packet.sh
)

echo "Source lock: ${SOURCE_LOCK_SHA}"
echo "Training: ${TRAIN_JOB_ID}"
echo "Outcome-blind scientific GPU assessment: ${TELEMETRY_JOB_ID}"
echo "Audit array: ${AUDIT_JOB_ID}; audit summary: ${AUDIT_SUMMARY_JOB_ID}"
echo "Evaluation array: ${EVAL_JOB_ID}; summary: ${SUMMARY_JOB_ID}"
echo "Authenticated compact packet: ${PACKET_JOB_ID}"
