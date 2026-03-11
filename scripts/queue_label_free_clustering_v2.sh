#!/bin/bash
# Queue label-free clustering v2 evaluation across multiple systems,
# model families, and seeds using existing checkpoints.
#
# Systems: duffing (2 basins),
#          multiwell_energy/gradient/rotational (5 basins each),
#          multiwell_energy_hd/gradient_hd/rotational_hd (5 basins each),
#          kuramoto N=16 (winding-number basins)
#
# Families: generic_sparse, lista_dense, lista_blockdiag

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATE_TAG="$(date +%Y%m%d)"
EXPERIMENT_TAG="label_free_clustering_v2_${DATE_TAG}"
BASE_OUT="/network/scratch/l/lia/skae/${EXPERIMENT_TAG}"
RESULTS_DIR="${REPO_ROOT}/results/${EXPERIMENT_TAG}"
TASK_TSV="${RESULTS_DIR}/task_specs.tsv"
SUMMARY_DIR="${BASE_OUT}/summary"

mkdir -p "${RESULTS_DIR}" "${BASE_OUT}"

# ---- Checkpoint sources ----------------------------------------------------
BENCH50K="/network/scratch/l/lia/skae/paper_benchmark_20260307_paper_final_ts256_50k_v4/full"
KURAMOTO200K="/network/scratch/l/lia/skae/kuramoto_dt00625_200k_compare_20260308/kuramoto_dt00625_200k"

# ---- Build task TSV --------------------------------------------------------
TASK_ID=0

printf "task_id\tsystem\tfamily\troot_label\tseed\tcheckpoint\n" > "${TASK_TSV}"

add_task() {
    local system="$1" family="$2" root_label="$3" seed="$4" glob_pattern="$5"
    # Resolve glob to a single file
    local resolved
    resolved=$(compgen -G "${glob_pattern}" | head -1)
    if [[ -z "${resolved}" || ! -f "${resolved}" ]]; then
        echo "WARN: no checkpoint matching: ${glob_pattern}" >&2
        return
    fi
    printf "%d\t%s\t%s\t%s\t%s\t%s\n" \
        "${TASK_ID}" "${system}" "${family}" "${root_label}" "${seed}" "${resolved}" >> "${TASK_TSV}"
    TASK_ID=$((TASK_ID + 1))
}

# ---- Low-dim systems from 50k benchmark (3 seeds) -------------------------

# system_name:dt_dir
LOWDIM_SYSTEMS=(
    "duffing:dt_0p01"
    "multiwell_energy:dt_0p02"
    "multiwell_gradient:dt_0p02"
    "multiwell_rotational:dt_0p02"
    "multiwell_energy_hd:dt_0p02"
    "multiwell_gradient_hd:dt_0p005"
    "multiwell_rotational_hd:dt_0p005"
    "multiwell_strong_transition:dt_0p005"
    "multiwell_strong_transition_hd:dt_0p005"
)

for SEED in 0 1 2; do
    for ENTRY in "${LOWDIM_SYSTEMS[@]}"; do
        SYS="${ENTRY%%:*}"
        DT="${ENTRY##*:}"
        add_task "${SYS}" generic "generic_sparse" "${SEED}" \
            "${BENCH50K}/generic_sparse/${SYS}/${DT}/seed_${SEED}/*/checkpoint.pt"
        add_task "${SYS}" dense_lista "lista_dense" "${SEED}" \
            "${BENCH50K}/lista_dense/${SYS}/${DT}/seed_${SEED}/*/checkpoint.pt"
        add_task "${SYS}" blockdiag_lista "lista_blockdiag" "${SEED}" \
            "${BENCH50K}/lista_blockdiag/${SYS}/${DT}/seed_${SEED}/*/checkpoint.pt"
    done
done

# ---- Kuramoto from 200k fine-dt runs (5 seeds) ----------------------------
for SEED in 0 1 2 3 4; do
    add_task kuramoto generic "generic_sparse" "${SEED}" \
        "${KURAMOTO200K}/generic_sparse/kuramoto/dt_0p00625/seed_${SEED}/*/checkpoint.pt"
    add_task kuramoto dense_lista "lista_dense" "${SEED}" \
        "${KURAMOTO200K}/lista_dense/kuramoto/dt_0p00625/seed_${SEED}/*/checkpoint.pt"
    add_task kuramoto blockdiag_lista "lista_blockdiag" "${SEED}" \
        "${KURAMOTO200K}/lista_blockdiag/kuramoto/dt_0p00625/seed_${SEED}/*/checkpoint.pt"
done

# ---- Count tasks -----------------------------------------------------------
TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if (( TASK_COUNT <= 0 )); then
    echo "ERROR: No tasks generated. Check checkpoint paths."
    exit 1
fi
echo "Generated ${TASK_COUNT} tasks in ${TASK_TSV}"
cat "${TASK_TSV}" | head -5
echo "..."

# ---- Submit SLURM jobs -----------------------------------------------------
ARRAY_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" \
    sbatch --array=0-$((TASK_COUNT - 1)) \
    scripts/run_label_free_clustering_v2_array.sh | awk '{print $4}')

COLLECT_JOB_ID=$(TASK_TSV="${TASK_TSV}" BASE_OUT="${BASE_OUT}" SUMMARY_DIR="${SUMMARY_DIR}" \
    sbatch --dependency=afterany:${ARRAY_JOB_ID} \
    scripts/run_label_free_clustering_v2_collect.sh | awk '{print $4}')

echo "Submitted array job: ${ARRAY_JOB_ID} (${TASK_COUNT} tasks)"
echo "Submitted collect job: ${COLLECT_JOB_ID} (depends on ${ARRAY_JOB_ID})"
echo "Task TSV: ${TASK_TSV}"
echo "Results: ${BASE_OUT}"
echo "Summary: ${SUMMARY_DIR}"
