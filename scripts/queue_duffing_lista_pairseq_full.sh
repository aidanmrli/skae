#!/bin/bash
#
# Queue Duffing LISTA L1-vs-L8 full parity runs plus collection.
#
# Submit:
#   sbatch scripts/queue_duffing_lista_pairseq_full.sh
#
# Optional overrides:
#   BASE_OUT=/network/scratch/l/lia/skae/duffing_lista_pairseq_full_20260303
#   OUT_DIR=results/duffing_lista_pairseq_full_20260303
#   SUMMARY_PREFIX=duffing_lista_pairseq_full
#   NUM_STEPS=10000 EVAL_PROFILE=full
#
#SBATCH --job-name=queue_duf_ls_fl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/queue-duf-ls-fl-%j.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/duffing_lista_pairseq_full_20260303}"
OUT_DIR="${OUT_DIR:-results/duffing_lista_pairseq_full_20260303}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-duffing_lista_pairseq_full}"

NUM_STEPS="${NUM_STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TARGET_SIZE="${TARGET_SIZE:-256}"
RES_COEFF="${RES_COEFF:-1.0}"
RECONST_COEFF="${RECONST_COEFF:-0.03}"
PRED_COEFF="${PRED_COEFF:-1.0}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0.0025}"
LISTA_ALPHA="${LISTA_ALPHA:-0.1}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-5}"
K_STRUCTURE="${K_STRUCTURE:-dense}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"

echo "Queueing Duffing LISTA full parity"
echo "Timestamp: $(date)"
echo "BASE_OUT: ${BASE_OUT}"
echo "OUT_DIR: ${OUT_DIR}"

SWEEP_JOB_ID=$(sbatch \
  --export=ALL,BASE_OUT="${BASE_OUT}",NUM_STEPS="${NUM_STEPS}",BATCH_SIZE="${BATCH_SIZE}",TARGET_SIZE="${TARGET_SIZE}",RES_COEFF="${RES_COEFF}",RECONST_COEFF="${RECONST_COEFF}",PRED_COEFF="${PRED_COEFF}",SPARSITY_COEFF="${SPARSITY_COEFF}",LISTA_ALPHA="${LISTA_ALPHA}",LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS}",K_STRUCTURE="${K_STRUCTURE}",EVAL_PROFILE="${EVAL_PROFILE}" \
  "${ROOT_DIR}/scripts/sweep_duffing_lista_pairseq_full.sh" | awk '{print $4}')

COLLECT_JOB_ID=$(sbatch \
  --dependency="afterany:${SWEEP_JOB_ID}" \
  --export=ALL,BASE_OUT="${BASE_OUT}",OUT_DIR="${OUT_DIR}",SUMMARY_PREFIX="${SUMMARY_PREFIX}" \
  "${ROOT_DIR}/scripts/collect_duffing_lista_pairseq.sh" | awk '{print $4}')

echo "Submitted full sweep job: ${SWEEP_JOB_ID}"
echo "Submitted collection job: ${COLLECT_JOB_ID} (afterany:${SWEEP_JOB_ID})"
