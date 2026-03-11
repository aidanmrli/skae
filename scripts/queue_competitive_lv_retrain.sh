#!/bin/bash
# Retrain ALL paper-facing competitive_lv experiments with the corrected
# INTERACTION_SCALE=0.70 (multi-basin, 4 major basins).
#
# Groups:
#   A) v4 canonical benchmark (50k): generic_sparse, lista_dense, lista_blockdiag, lista_diagonal × seeds 0-2
#   B) 200k fairness followup: generic_sparse_ns200k_best, lista_blockdiag_ns200k_denseopt_sc3em3,
#      lista_blockdiag_ns200k_denseopt_sc6em3 × seeds 0-2
#   C) Promoted dense Stage 4 (200k): lista_dense promoted recipe × seeds 0-2
#   D) More seeds (Subagent D): generic_sparse + lista_dense promoted × seeds 3,4
#
# Total: 12 + 9 + 3 + 4 = 28 training runs
#
# The new default INTERACTION_SCALE=0.70 is picked up automatically from Config.

set -euo pipefail

cd /home/mila/l/lia/skae

DATE_TAG="$(date +%Y%m%d)"
EXPERIMENT_TAG="competitive_lv_multibas_retrain_${DATE_TAG}"
BASE_OUT="/network/scratch/l/lia/skae/${EXPERIMENT_TAG}"
RESULTS_DIR="results/${EXPERIMENT_TAG}"
TASK_TSV="${RESULTS_DIR}/task_specs.tsv"

mkdir -p "${RESULTS_DIR}" "${BASE_OUT}"

TASK_ID=0

# TSV columns: task_id group root_label preset num_steps seed lr k_lr wd sc rc pc
#              k_structure k_block_size lista_alpha lista_num_loops lista_final_op
printf "task_id\tgroup\troot_label\tpreset\tnum_steps\tseed\tlr\tk_matrix_lr\tweight_decay\tsparsity_coeff\treconst_coeff\tpred_coeff\tk_structure\tk_block_size\tlista_alpha\tlista_num_loops\tlista_final_op\n" > "${TASK_TSV}"

add_task() {
    local group="$1" root_label="$2" preset="$3" num_steps="$4" seed="$5"
    local lr="$6" k_lr="$7" wd="$8" sc="$9" rc="${10}" pc="${11}"
    local k_struct="${12:-}" k_bs="${13:-}" la="${14:-}" lnl="${15:-}" lfo="${16:-}"
    printf "%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${TASK_ID}" "${group}" "${root_label}" "${preset}" "${num_steps}" \
        "${seed}" "${lr}" "${k_lr}" "${wd}" "${sc}" "${rc}" "${pc}" \
        "${k_struct}" "${k_bs}" "${la}" "${lnl}" "${lfo}" >> "${TASK_TSV}"
    TASK_ID=$((TASK_ID + 1))
}

# ---- Group A: v4 canonical benchmark (50k) ---------------------------------
# These match paper_benchmark_manifest.py exactly
for SEED in 0 1 2; do
    # generic_sparse: preset=generic_sparse, no LISTA overrides
    add_task A generic_sparse generic_sparse 50000 "${SEED}" \
        1e-4 1e-5 1e-4 0.0025 0.03 1.0

    # lista_dense: preset=lista_parity_generic_sparse, k_structure=dense
    add_task A lista_dense lista_parity_generic_sparse 50000 "${SEED}" \
        1e-4 1e-5 1e-4 0.006 0.03 1.0 \
        dense "" 0.15 1 relu

    # lista_blockdiag: preset=lista_parity_generic_sparse, k_structure=block_diagonal, k_block_size=16
    add_task A lista_blockdiag lista_parity_generic_sparse 50000 "${SEED}" \
        1e-4 1e-5 1e-4 0.006 0.03 1.0 \
        block_diagonal 16 0.15 1 relu

    # lista_diagonal: preset=lista_parity_generic_sparse, k_structure=diagonal
    add_task A lista_diagonal lista_parity_generic_sparse 50000 "${SEED}" \
        1e-4 1e-5 1e-4 0.006 0.03 1.0 \
        diagonal "" 0.15 1 relu
done

# ---- Group B: 200k fairness followup ---------------------------------------
for SEED in 0 1 2; do
    # generic_sparse at 200k
    add_task B generic_sparse_ns200k_best generic_sparse 200000 "${SEED}" \
        1e-4 1e-5 1e-4 0.0025 0.03 1.0

    # lista_blockdiag with dense-opt recipe, sc=0.003
    add_task B lista_blockdiag_ns200k_denseopt_sc3em3 lista_parity_generic_sparse 200000 "${SEED}" \
        5e-5 5e-6 1e-4 0.003 0.03 1.0 \
        block_diagonal 16 0.15 1 relu

    # lista_blockdiag with dense-opt recipe, sc=0.006
    add_task B lista_blockdiag_ns200k_denseopt_sc6em3 lista_parity_generic_sparse 200000 "${SEED}" \
        5e-5 5e-6 1e-4 0.006 0.03 1.0 \
        block_diagonal 16 0.15 1 relu
done

# ---- Group C: Promoted dense Stage 4 (200k) --------------------------------
for SEED in 0 1 2; do
    add_task C lista_dense_promoted_stage4 lista_parity_generic_sparse 200000 "${SEED}" \
        5e-5 5e-6 1e-4 0.003 0.03 1.0 \
        dense "" 0.15 1 relu
done

# ---- Group D: More seeds on headline positives (200k) ----------------------
for SEED in 3 4; do
    add_task D generic_sparse_ns200k_best generic_sparse 200000 "${SEED}" \
        1e-4 1e-5 1e-4 0.0025 0.03 1.0

    add_task D lista_dense_promoted_stage4 lista_parity_generic_sparse 200000 "${SEED}" \
        5e-5 5e-6 1e-4 0.003 0.03 1.0 \
        dense "" 0.15 1 relu
done

# ---- Count tasks -----------------------------------------------------------
TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
echo "Generated ${TASK_COUNT} tasks in ${TASK_TSV}"
head -5 "${TASK_TSV}"
echo "..."

# ---- Submit SLURM array ----------------------------------------------------
TASK_TSV="$(pwd)/${TASK_TSV}"
ARRAY_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --parsable --array=0-$((TASK_COUNT - 1)) \
    scripts/run_competitive_lv_retrain_array.sh)

echo "Submitted array job: ${ARRAY_JOB_ID} (${TASK_COUNT} tasks)"
echo "Task TSV: ${TASK_TSV}"
echo "Results: ${BASE_OUT}"
