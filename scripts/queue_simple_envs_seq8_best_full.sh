#!/bin/bash
#
# Queue full (10k-step) seq-8 simple-env sweeps for tuned generic_sparse and LISTAKM.
#
# Submit:
#   sbatch scripts/queue_simple_envs_seq8_best_full.sh
#
# Optional env overrides are forwarded to both sweeps, e.g.:
#   NUM_STEPS=10000 SEED=0 EVAL_PROFILE=full
#
#SBATCH --job-name=queue_s8_best_full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/queue-s8-best-full-%j.out

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Queueing full 10k-step seq-8 sweeps from ${ROOT_DIR}"
echo "Timestamp: $(date)"

export NUM_STEPS="${NUM_STEPS:-10000}"
export SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-8}"
export EVAL_PROFILE="${EVAL_PROFILE:-full}"
export SEED="${SEED:-0}"

GENERIC_JOB=$(sbatch scripts/sweep_simple_envs_seq8_generic_sparse_best_full.sh | awk '{print $4}')
LISTA_JOB=$(sbatch scripts/sweep_simple_envs_seq8_lista_best_relu_blockdiag_full.sh | awk '{print $4}')

echo "Queued jobs:"
echo "  generic_sparse full sweep: ${GENERIC_JOB}"
echo "  LISTAKM ReLU block-diag full sweep: ${LISTA_JOB}"

