#!/bin/bash
# Long-running orchestrator for the Dysts seq10/seeds0to14 pipeline.
#
# Stages:
#   1. Wait for room (user job count < 800), submit Dysts chunk 2 (200 tasks).
#   2. Wait for Dysts chunks 1 + 2 to complete, then submit the H<=60k
#      Dysts eval (queue_dysts_long_horizon_eval.sh) targeting only the new
#      training output via a custom root_specs_tsv.
#   3. Log progress to /network/scratch/l/lia/skae/dysts_orchestrator.log.
#
#SBATCH --job-name=dysts_orchestrator
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=2-00:00:00
#SBATCH -o /network/scratch/l/lia/skae/dysts-orchestrator-%j.out
#SBATCH -e /network/scratch/l/lia/skae/dysts-orchestrator-%j.err

set -euo pipefail

cd /home/mila/l/lia/skae
source .venv/bin/activate

LOG="/network/scratch/l/lia/skae/dysts_orchestrator.log"
TASK_TSV="/home/mila/l/lia/skae/results/paper_followup_recipes_200k_seq10_seeds0to14_20260428/task_tables/paper_followup_recipes.tsv"
BASE_OUT="/network/scratch/l/lia/skae/paper_followup_recipes_200k_seq10_seeds0to14_20260428"
ROOT_SPECS_TSV="/home/mila/l/lia/skae/results/paper_followup_recipes_200k_seq10_seeds0to14_20260428/dysts_long_horizon_root_specs.tsv"
EVAL_RESULTS_DIR="/home/mila/l/lia/skae/results/dysts_long_horizon_eval_seq10_h60k_seeds0to14_20260428"
EVAL_HORIZONS="5000 10000 20000 30000 40000 50000 60000"

# Job IDs of submissions we already know about.
CHUNK1_JOB_ID="9392814"
CHUNK2_JOB_ID=""
EVAL_JOB_ID=""

log() {
    echo "[$(date)] $*" | tee -a "${LOG}"
}

job_state() {
    # Returns the SLURM state of an array job (uses first element as proxy).
    local jid="$1"
    sacct -j "${jid}" --format=State -n -P 2>/dev/null | head -1 | tr -d ' '
}

is_array_complete() {
    # Returns 0 (true) if all elements of an array job have terminated
    # (any of COMPLETED/FAILED/CANCELLED/TIMEOUT).
    local jid="$1"
    local pending_or_running
    pending_or_running=$(sacct -j "${jid}" --format=State -n -P 2>/dev/null \
        | awk 'NF' \
        | grep -cE "PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED" || true)
    [[ "${pending_or_running}" == "0" ]]
}

log "orchestrator start (pid=$$, slurm_job=${SLURM_JOB_ID:-none})"
log "chunk1 job id: ${CHUNK1_JOB_ID}"

# -----------------------------------------------------------------------
# Stage 1: submit Dysts chunk 2 once user job count drops below 800.
# -----------------------------------------------------------------------
while [[ -z "${CHUNK2_JOB_ID}" ]]; do
    count=$(squeue -u lia -h -r 2>/dev/null | wc -l)
    log "stage 1: user job count=${count}, target<800"
    if (( count < 800 )); then
        CHUNK2_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" ARRAY_OFFSET=700 \
            sbatch --parsable --array=0-199%64 \
            /home/mila/l/lia/skae/scripts/run_paper_benchmark_array.sh 2>>"${LOG}" || true)
        if [[ -n "${CHUNK2_JOB_ID}" ]]; then
            log "stage 1: submitted Dysts chunk 2 as job ${CHUNK2_JOB_ID}"
        else
            log "stage 1: chunk-2 submit failed, retrying in 60s"
        fi
    fi
    sleep 60
done

# -----------------------------------------------------------------------
# Stage 2: wait for both Dysts training chunks to finish.
# -----------------------------------------------------------------------
log "stage 2: waiting for Dysts chunks 1 (${CHUNK1_JOB_ID}) and 2 (${CHUNK2_JOB_ID}) to complete"
while ! is_array_complete "${CHUNK1_JOB_ID}" || ! is_array_complete "${CHUNK2_JOB_ID}"; do
    log "stage 2: chunk1=$(job_state "${CHUNK1_JOB_ID}") chunk2=$(job_state "${CHUNK2_JOB_ID}")"
    sleep 300
done
log "stage 2: both Dysts chunks have terminated; queueing long-horizon eval"

# -----------------------------------------------------------------------
# Stage 3: submit long-horizon Dysts eval against the new checkpoints.
# Wait until user job count drops below 800 again before submitting the
# eval (the eval queue script submits its own chains).
# -----------------------------------------------------------------------
mkdir -p "${EVAL_RESULTS_DIR}"
while :; do
    count=$(squeue -u lia -h -r 2>/dev/null | wc -l)
    log "stage 3: user job count=${count}, target<800"
    if (( count < 800 )); then
        EVAL_JOB_ID=$(DATE_TAG=20260428 \
            RESULTS_DIR="${EVAL_RESULTS_DIR}" \
            OUTPUT_TAG="dysts_long_horizon_h5k_to_h60k_seq10" \
            HORIZONS="${EVAL_HORIZONS}" \
            DYSTS_CACHE_PROFILE="long60" \
            INPUT_ROOT_SPECS_TSV="${ROOT_SPECS_TSV}" \
            EVAL_TIME_LIMIT=06:00:00 \
            ARRAY_PARALLEL=48 \
            sbatch --parsable /home/mila/l/lia/skae/scripts/queue_dysts_long_horizon_eval.sh 2>>"${LOG}" || true)
        if [[ -n "${EVAL_JOB_ID}" ]]; then
            log "stage 3: submitted long-horizon eval queue launcher as job ${EVAL_JOB_ID}"
            break
        else
            log "stage 3: long-horizon submit failed, retrying in 60s"
        fi
    fi
    sleep 60
done

log "orchestrator done; last submitted job is queue launcher ${EVAL_JOB_ID}"
log "review /network/scratch/l/lia/skae/queue-dysts-long-eval-${EVAL_JOB_ID}.out for downstream chain (validate/eval/collect)"
