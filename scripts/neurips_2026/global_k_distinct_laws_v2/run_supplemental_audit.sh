#!/bin/bash
# Independently reproduce V2 validation/adjudication and enforce wording guards.

#SBATCH --job-name=gkv2_suppaudit
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
source scripts/common/cluster_env.sh

SUPPLEMENTAL_LOCK="${SUPPLEMENTAL_LOCK:-experiments/neurips_2026/global_k_distinct_laws_v2_supplemental_audit_lock.json}"
SUPPLEMENTAL_CARD="${SUPPLEMENTAL_CARD:-experiments/neurips_2026/global_k_distinct_laws_v2_supplemental_audit_card.json}"
V2_CARD="${V2_CARD:-experiments/neurips_2026/global_k_distinct_laws_v2_card.json}"
EXPECTED_SUPPLEMENTAL_LOCK_SHA="${EXPECTED_SUPPLEMENTAL_LOCK_SHA:?EXPECTED_SUPPLEMENTAL_LOCK_SHA is required}"

if [[ "$(sha256sum "${SUPPLEMENTAL_LOCK}" | awk '{print $1}')" != "${EXPECTED_SUPPLEMENTAL_LOCK_SHA}" ]]; then
  echo "Supplemental-lock hash mismatch." >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
uv run python -m experiments.neurips_2026.global_k_distinct_laws_v2_supplemental_audit \
  --supplemental_lock "${SUPPLEMENTAL_LOCK}" \
  --supplemental_card "${SUPPLEMENTAL_CARD}" \
  --v2_card "${V2_CARD}" \
  --source_lock "${SOURCE_LOCK:?SOURCE_LOCK is required}" \
  --task_tsv "${TASK_TSV:?TASK_TSV is required}" \
  --base_out "${BASE_OUT:?BASE_OUT is required}" \
  --audit_dir "${AUDIT_DIR:?AUDIT_DIR is required}" \
  --evaluation_dir "${EVALUATION_DIR:?EVALUATION_DIR is required}" \
  --decision "${DECISION:?DECISION is required}" \
  --output "${OUTPUT:?OUTPUT is required}"
