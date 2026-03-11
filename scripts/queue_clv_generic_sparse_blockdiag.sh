#!/bin/bash
# Train generic_sparse + block_diagonal K on competitive_lv (multi-basin).
# This is the fairness control: same MLP encoder as generic_sparse, but with
# block-diagonal Koopman matrix (k_block_size=16) to match lista_blockdiag.
#
# Groups:
#   A) 50k steps, seeds 0-2 (canonical benchmark)
#   B) 200k steps, seeds 0-2 (longer training)
#
# Total: 6 training runs

set -euo pipefail

cd /home/mila/l/lia/skae

DATE_TAG="$(date +%Y%m%d)"
EXPERIMENT_TAG="clv_generic_sparse_blockdiag_${DATE_TAG}"
BASE_OUT="/network/scratch/l/lia/skae/${EXPERIMENT_TAG}"
RESULTS_DIR="results/${EXPERIMENT_TAG}"
TASK_TSV="${RESULTS_DIR}/task_specs.tsv"

mkdir -p "${RESULTS_DIR}" "${BASE_OUT}"

TASK_ID=0

printf "task_id\tgroup\troot_label\tpreset\tnum_steps\tseed\tlr\tk_matrix_lr\tweight_decay\tsparsity_coeff\treconst_coeff\tpred_coeff\tk_structure\tk_block_size\n" > "${TASK_TSV}"

add_task() {
    local group="$1" root_label="$2" preset="$3" num_steps="$4" seed="$5"
    local lr="$6" k_lr="$7" wd="$8" sc="$9" rc="${10}" pc="${11}"
    local k_struct="${12}" k_bs="${13}"
    printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${TASK_ID}" "${group}" "${root_label}" "${preset}" "${num_steps}" \
        "${seed}" "${lr}" "${k_lr}" "${wd}" "${sc}" "${rc}" "${pc}" \
        "${k_struct}" "${k_bs}" >> "${TASK_TSV}"
    TASK_ID=$((TASK_ID + 1))
}

# ---- Group A: 50k canonical benchmark (matches existing generic_sparse runs) ----
for SEED in 0 1 2; do
    add_task A generic_sparse_blockdiag generic_sparse 50000 "${SEED}" \
        1e-4 1e-5 1e-4 0.0025 0.03 1.0 \
        block_diagonal 16
done

# ---- Group B: 200k (matches existing generic_sparse_ns200k_best runs) ----
for SEED in 0 1 2; do
    add_task B generic_sparse_blockdiag_200k generic_sparse 200000 "${SEED}" \
        1e-4 1e-5 1e-4 0.0025 0.03 1.0 \
        block_diagonal 16
done

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
echo "Generated ${TASK_COUNT} tasks in ${TASK_TSV}"
head -5 "${TASK_TSV}"
echo "..."

# ---- Submit SLURM array ----
TASK_TSV="$(pwd)/${TASK_TSV}"
ARRAY_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --parsable --array=0-$((TASK_COUNT - 1)) \
    scripts/run_clv_generic_sparse_blockdiag_array.sh)

echo "Submitted array job: ${ARRAY_JOB_ID} (${TASK_COUNT} tasks)"
echo "Task TSV: ${TASK_TSV}"
echo "Results: ${BASE_OUT}"
