#!/usr/bin/env bash
#SBATCH --job-name=queue-dysts-sb1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail

PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
ROOT_DIR="${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/common/cluster_env.sh"
RESULTS_DIR="${SKAE_SCRATCH_ROOT}/results/dysts_lista_sb_refinement1_20260722"
mkdir -p "${RESULTS_DIR}/logs"
cd "${PROJECT_DIR}"
sha256sum -c experiments/neurips_2026/dysts_lista_sb_refinement1/source_manifest.sha256

DATE_TAG=20260722 \
PHASE_LABEL=dysts_dt30_lista_sb_refinement1_p256_seq10_100k \
BASE_OUT="${SKAE_SCRATCH_ROOT}/dysts_lista_sb_refinement1_20260722" \
RESULTS_DIR="${RESULTS_DIR}" \
MODEL_VARIANTS_CSV=lista_sb \
LISTA_SB_NUM_LOOPS=1 \
ALLOW_ROOT_SUBSET=1 \
TRAIN_PACK_SIZE=12 \
TRAIN_PACK_CONCURRENCY="${TRAIN_PACK_CONCURRENCY:-12}" \
TRAIN_CPUS_PER_TASK="${TRAIN_CPUS_PER_TASK:-12}" \
TRAIN_ARRAY_PARALLEL="${TRAIN_ARRAY_PARALLEL:-24}" \
EVAL_PACK_SIZE=12 \
sbatch scripts/neurips_2026/dysts/queue_training.sh
