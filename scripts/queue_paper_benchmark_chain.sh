#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATE_TAG="${DATE_TAG:-$(date +%Y%m%d)}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/paper_benchmark_${DATE_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/paper_benchmark_${DATE_TAG}}"
TASK_DIR="${TASK_DIR:-${RESULTS_DIR}/task_tables}"
ROOT_SPEC_DIR="${ROOT_SPEC_DIR:-${RESULTS_DIR}/root_specs}"
RESOLVE_DIR="${RESOLVE_DIR:-${RESULTS_DIR}/dt_resolution}"

mkdir -p "${TASK_DIR}" "${ROOT_SPEC_DIR}" "${RESOLVE_DIR}"

SMOKE_TSV="${TASK_DIR}/paper_smoke.tsv"
ANCHOR_TSV="${TASK_DIR}/paper_anchor.tsv"
RESCUE1_TSV="${TASK_DIR}/paper_rescue_pass1.tsv"
RESCUE2_TSV="${TASK_DIR}/paper_rescue_pass2.tsv"
FULL_TSV="${TASK_DIR}/paper_full.tsv"
MANIFEST_JSON="${TASK_DIR}/paper_manifest.json"

uv run python tools/build_paper_benchmark_tasks.py \
  --phase smoke \
  --output_tsv "${SMOKE_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}"

uv run python tools/build_paper_benchmark_tasks.py \
  --phase anchor \
  --output_tsv "${ANCHOR_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}"

cat > "${ROOT_SPEC_DIR}/paper_smoke_roots.txt" <<EOF
generic_sparse=${BASE_OUT}/smoke/generic_sparse
lista_dense=${BASE_OUT}/smoke/lista_dense
lista_diagonal=${BASE_OUT}/smoke/lista_diagonal
lista_blockdiag=${BASE_OUT}/smoke/lista_blockdiag
EOF

cat > "${ROOT_SPEC_DIR}/paper_anchor_default_roots.txt" <<EOF
generic_sparse_default=${BASE_OUT}/anchor/generic_sparse
EOF

cat > "${ROOT_SPEC_DIR}/paper_anchor_pass1_roots.txt" <<EOF
generic_sparse_default=${BASE_OUT}/anchor/generic_sparse
generic_sparse_rescue_pass1=${BASE_OUT}/rescue_pass1/generic_sparse
EOF

cat > "${ROOT_SPEC_DIR}/paper_anchor_pass2_roots.txt" <<EOF
generic_sparse_default=${BASE_OUT}/anchor/generic_sparse
generic_sparse_rescue_pass1=${BASE_OUT}/rescue_pass1/generic_sparse
generic_sparse_rescue_pass2=${BASE_OUT}/rescue_pass2/generic_sparse
EOF

cat > "${ROOT_SPEC_DIR}/paper_full_roots.txt" <<EOF
generic_sparse=${BASE_OUT}/full/generic_sparse
lista_dense=${BASE_OUT}/full/lista_dense
lista_diagonal=${BASE_OUT}/full/lista_diagonal
lista_blockdiag=${BASE_OUT}/full/lista_blockdiag
EOF

SMOKE_COUNT=$(( $(wc -l < "${SMOKE_TSV}") - 1 ))
ANCHOR_COUNT=$(( $(wc -l < "${ANCHOR_TSV}") - 1 ))
FULL_COUNT=$((29 * 4 * 3))
RESCUE_MAX_COUNT=$((29 * 3))

SMOKE_JOB_ID=$(TASK_TSV="${SMOKE_TSV}" BASE_OUT="${BASE_OUT}" sbatch --array=0-$((SMOKE_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}')
SMOKE_COLLECT_ID=$(ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/paper_smoke_roots.txt" OUT_DIR="${RESULTS_DIR}/smoke_collect" PAPER_SUMMARY=0 sbatch --dependency=afterany:${SMOKE_JOB_ID} scripts/collect_paper_benchmark.sh | awk '{print $4}')

ANCHOR_JOB_ID=$(TASK_TSV="${ANCHOR_TSV}" BASE_OUT="${BASE_OUT}" sbatch --dependency=afterany:${SMOKE_COLLECT_ID} --array=0-$((ANCHOR_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}')
ANCHOR_COLLECT_ID=$(ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/paper_anchor_default_roots.txt" OUT_DIR="${RESULTS_DIR}/anchor_collect_default" PAPER_SUMMARY=0 sbatch --dependency=afterany:${ANCHOR_JOB_ID} scripts/collect_paper_benchmark.sh | awk '{print $4}')

RESOLVE0_ID=$(ROWS_CSV="${RESULTS_DIR}/anchor_collect_default/forecasting_rows.csv" OUT_DIR="${RESOLVE_DIR}/pass0" CURRENT_PASS=0 NEXT_TASK_TSV="${RESCUE1_TSV}" MANIFEST_JSON="${MANIFEST_JSON}" sbatch --dependency=afterany:${ANCHOR_COLLECT_ID} scripts/resolve_paper_benchmark_dt.sh | awk '{print $4}')
RESCUE1_JOB_ID=$(TASK_TSV="${RESCUE1_TSV}" BASE_OUT="${BASE_OUT}" sbatch --dependency=afterany:${RESOLVE0_ID} --array=0-$((RESCUE_MAX_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}')
RESCUE1_COLLECT_ID=$(ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/paper_anchor_pass1_roots.txt" OUT_DIR="${RESULTS_DIR}/anchor_collect_pass1" PAPER_SUMMARY=0 sbatch --dependency=afterany:${RESCUE1_JOB_ID} scripts/collect_paper_benchmark.sh | awk '{print $4}')

RESOLVE1_ID=$(ROWS_CSV="${RESULTS_DIR}/anchor_collect_pass1/forecasting_rows.csv" OUT_DIR="${RESOLVE_DIR}/pass1" CURRENT_PASS=1 NEXT_TASK_TSV="${RESCUE2_TSV}" MANIFEST_JSON="${MANIFEST_JSON}" sbatch --dependency=afterany:${RESCUE1_COLLECT_ID} scripts/resolve_paper_benchmark_dt.sh | awk '{print $4}')
RESCUE2_JOB_ID=$(TASK_TSV="${RESCUE2_TSV}" BASE_OUT="${BASE_OUT}" sbatch --dependency=afterany:${RESOLVE1_ID} --array=0-$((RESCUE_MAX_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}')
RESCUE2_COLLECT_ID=$(ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/paper_anchor_pass2_roots.txt" OUT_DIR="${RESULTS_DIR}/anchor_collect_pass2" PAPER_SUMMARY=0 sbatch --dependency=afterany:${RESCUE2_JOB_ID} scripts/collect_paper_benchmark.sh | awk '{print $4}')

RESOLVE2_ID=$(ROWS_CSV="${RESULTS_DIR}/anchor_collect_pass2/forecasting_rows.csv" OUT_DIR="${RESOLVE_DIR}/pass2" CURRENT_PASS=2 NEXT_TASK_TSV="${FULL_TSV}" MANIFEST_JSON="${MANIFEST_JSON}" sbatch --dependency=afterany:${RESCUE2_COLLECT_ID} scripts/resolve_paper_benchmark_dt.sh | awk '{print $4}')
FULL_JOB_ID=$(TASK_TSV="${FULL_TSV}" BASE_OUT="${BASE_OUT}" sbatch --dependency=afterany:${RESOLVE2_ID} --array=0-$((FULL_COUNT - 1)) scripts/run_paper_benchmark_array.sh | awk '{print $4}')
FINAL_COLLECT_ID=$(ROOT_SPECS_FILE="${ROOT_SPEC_DIR}/paper_full_roots.txt" OUT_DIR="${RESULTS_DIR}/final_collect" PAPER_SUMMARY=1 sbatch --dependency=afterany:${FULL_JOB_ID} scripts/collect_paper_benchmark.sh | awk '{print $4}')
COMPARE_ID=$(ROWS_CSV="${RESULTS_DIR}/final_collect/forecasting_rows.csv" OUT_DIR="${RESULTS_DIR}/final_compare" sbatch --dependency=afterany:${FINAL_COLLECT_ID} scripts/compare_paper_benchmark.sh | awk '{print $4}')

echo "Queued research-paper benchmark chain."
echo "Smoke array: ${SMOKE_JOB_ID}"
echo "Smoke collect: ${SMOKE_COLLECT_ID}"
echo "Anchor array: ${ANCHOR_JOB_ID}"
echo "Anchor collect: ${ANCHOR_COLLECT_ID}"
echo "Resolve pass0: ${RESOLVE0_ID}"
echo "Rescue pass1 array: ${RESCUE1_JOB_ID}"
echo "Rescue pass1 collect: ${RESCUE1_COLLECT_ID}"
echo "Resolve pass1: ${RESOLVE1_ID}"
echo "Rescue pass2 array: ${RESCUE2_JOB_ID}"
echo "Rescue pass2 collect: ${RESCUE2_COLLECT_ID}"
echo "Resolve pass2: ${RESOLVE2_ID}"
echo "Full array: ${FULL_JOB_ID}"
echo "Final collect: ${FINAL_COLLECT_ID}"
echo "Final compare: ${COMPARE_ID}"
