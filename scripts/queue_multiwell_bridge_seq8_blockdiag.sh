#!/bin/bash
#
# Queue the focused seq8 block-diagonal LISTA sweep for the 2D multi-well bridge systems.
#
# Submit:
#   sbatch scripts/queue_multiwell_bridge_seq8_blockdiag.sh
#
# Optional env overrides are forwarded to both jobs, e.g.:
#   NUM_STEPS=20000 TARGET_SIZES_CSV=64,128,256 LISTA_NUM_LOOPS_CSV=1,3,5
#
#SBATCH --job-name=queue_mwb_s8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH -o /network/scratch/l/lia/skae/queue-mwb-s8-%j.out
#SBATCH -e /network/scratch/l/lia/skae/queue-mwb-s8-%j.err

set -euo pipefail

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${WORK_DIR}"

echo "Queueing multi-well bridge seq8 block-diagonal sweep from ${WORK_DIR}"
echo "Timestamp: $(date)"

SWEEP_JOB_ID=$(sbatch scripts/sweep_multiwell_bridge_seq8_blockdiag.sh | awk '{print $4}')
COLLECT_JOB_ID=$(sbatch --dependency=afterany:${SWEEP_JOB_ID} scripts/collect_multiwell_bridge_seq8_blockdiag.sh | awk '{print $4}')

echo "Queued jobs:"
echo "  sweep: ${SWEEP_JOB_ID}"
echo "  collect: ${COLLECT_JOB_ID}"
