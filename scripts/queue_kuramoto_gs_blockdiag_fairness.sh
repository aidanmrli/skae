#!/bin/bash
# Kuramoto fairness control: generic_sparse + block_diagonal K (k_block_size=16).
# Matches kuramoto_dt00625_200k_compare parameters exactly, adding only
# k_structure=block_diagonal and k_block_size=16.
#
# N=16 oscillators, dt=0.00625, 200k steps, seeds 0-4 → 5 tasks

set -euo pipefail
cd /home/mila/l/lia/skae

DATE_TAG="$(date +%Y%m%d)"
EXPERIMENT_TAG="kuramoto_gs_blockdiag_fairness_${DATE_TAG}"
BASE_OUT="/network/scratch/l/lia/skae/${EXPERIMENT_TAG}"
RESULTS_DIR="results/${EXPERIMENT_TAG}"
TSV_DIR="${RESULTS_DIR}/task_tables"
TASK_TSV="${TSV_DIR}/kuramoto_gs_blockdiag.tsv"

mkdir -p "${TSV_DIR}" "${BASE_OUT}"

# Header matches run_paper_benchmark_array.sh expectations
printf "task_id\tphase\tmodel_variant\tconfig_name\tsystem_key\tsystem_slug\tsystem_group\tenv_name\tseed\tnum_steps\tbatch_size\ttarget_size\tsequence_length\tres_coeff\treconst_coeff\tpred_coeff\tsparsity_coeff\tlista_alpha\tlista_num_loops\tlista_final_op\tk_structure\tk_block_size\tenv_dt\teval_profile\tstandardize\tdysts_native_cache\tdysts_cache_profile\tdysts_cache_reuse\tdysts_ic_noise_scale\n" > "${TASK_TSV}"

TASK_ID=0
for SEED in 0 1 2 3 4; do
    printf "%d\tkuramoto_gs_blockdiag\tgeneric_sparse_blockdiag\tgeneric_sparse\tkuramoto\tkuramoto\tbuiltin_high_dim\tkuramoto\t%d\t200000\t256\t256\t8\t1.0\t0.03\t1.0\t0.0025\t\t\t\tblock_diagonal\t16\t0.00625\tfull\t0\t0\t\t0\t\n" \
        "${TASK_ID}" "${SEED}" >> "${TASK_TSV}"
    TASK_ID=$((TASK_ID + 1))
done

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
echo "Generated ${TASK_COUNT} tasks in ${TASK_TSV}"
cat "${TASK_TSV}" | head -3
echo "..."

TASK_TSV="$(pwd)/${TASK_TSV}"
ARRAY_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --parsable --array=0-$((TASK_COUNT - 1)) \
    scripts/run_paper_benchmark_array.sh)

echo "Submitted array job: ${ARRAY_JOB_ID} (${TASK_COUNT} tasks)"
echo "Task TSV: ${TASK_TSV}"
echo "Results: ${BASE_OUT}"
