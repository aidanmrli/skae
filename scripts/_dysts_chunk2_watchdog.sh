#!/bin/bash
# Background watchdog: submit Dysts chunk-2 once enough jobs drain to fit
# under MaxSubmit=1000. Chunk 2 covers tasks 700-899 of the 900-task TSV.
set -euo pipefail
TASK_TSV="/home/mila/l/lia/skae/results/paper_followup_recipes_200k_seq10_seeds0to14_20260428/task_tables/paper_followup_recipes.tsv"
BASE_OUT="/network/scratch/l/lia/skae/paper_followup_recipes_200k_seq10_seeds0to14_20260428"
LOG=/network/scratch/l/lia/skae/dysts_chunk2_watchdog.log
echo "[$(date)] watchdog start" >> "${LOG}"
while :; do
    count=$(squeue -u lia -h -r | wc -l)
    echo "[$(date)] user job count: ${count}" >> "${LOG}"
    if (( count < 800 )); then
        JOB2=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" ARRAY_OFFSET=700 \
            sbatch --parsable --array=0-199%64 /home/mila/l/lia/skae/scripts/run_paper_benchmark_array.sh 2>>"${LOG}" || true)
        if [[ -n "${JOB2}" ]]; then
            echo "[$(date)] submitted chunk 2 as job ${JOB2}" >> "${LOG}"
            exit 0
        else
            echo "[$(date)] submit failed, retrying in 60s" >> "${LOG}"
        fi
    fi
    sleep 60
done
