#!/bin/bash
#
# Requeue failed paper-parallel workstreams E, D-N32, and H.
# Run from the main repo root: bash scripts/requeue_failed_paper_parallel_20260310.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== Requeue failed paper-parallel workstreams ==="
echo "Repo: $(pwd)"
echo "Date: $(date)"
echo ""

# ─────────────────────────────────────────────────────────────────────
# E: Kuramoto robustness (uniform_spread) – 10 training jobs
# Root cause: corrupted .venv in subagent workspace. Now submitting from main repo.
# ─────────────────────────────────────────────────────────────────────
echo "--- Workstream E: Kuramoto robustness ---"

E_TASK_TSV="results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/task_tables/paper_parallel_20260309_e_kuramoto_robustness.tsv"
E_BASE_OUT="/network/scratch/l/lia/skae/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309"
E_ROOT_SPECS="results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/root_specs/paper_parallel_20260309_e_kuramoto_robustness_roots.txt"
E_COLLECT_OUT="results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/collect"
E_COMPARE_OUT="results/paper_parallel_20260309_e_kuramoto_uniform_spread_n16_dt0p00625_20260309/compare"

E_TRAIN=$(TASK_TSV="${E_TASK_TSV}" BASE_OUT="${E_BASE_OUT}" \
  sbatch --parsable --array=0-9 scripts/paper_parallel_20260309_e_run_kuramoto_robustness_array.sh)
echo "  E train array: ${E_TRAIN}"

E_COLLECT=$(ROOT_SPECS_FILE="${E_ROOT_SPECS}" OUT_DIR="${E_COLLECT_OUT}" PAPER_SUMMARY=1 \
  sbatch --parsable --dependency=afterany:${E_TRAIN} scripts/collect_paper_benchmark.sh)
echo "  E collector: ${E_COLLECT}"

E_COMPARE=$(ROWS_CSV="${E_COLLECT_OUT}/forecasting_rows.csv" OUT_DIR="${E_COMPARE_OUT}" \
  CANDIDATE_ROOTS_CSV=lista_blockdiag_uniform_spread ANCHOR_ROOT=generic_sparse_uniform_spread HORIZON=1000 \
  sbatch --parsable --dependency=afterany:${E_COLLECT} scripts/compare_paper_benchmark.sh)
echo "  E compare: ${E_COMPARE}"

echo ""

# ─────────────────────────────────────────────────────────────────────
# D-N32: Kuramoto N=32 extra seeds 3,4 – 4 training jobs
# Root cause: sbatch --wrap used /bin/sh which lacks `module`. Now using array runner.
# ─────────────────────────────────────────────────────────────────────
echo "--- Workstream D: Kuramoto N=32 extra seeds ---"

D_N32_TASK_TSV="results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/task_tables/kuramoto_n32_seeds_3_4.tsv"
D_N32_BASE_OUT="/network/scratch/l/lia/skae/kuramoto_n32_dt00625_200k_confirm_20260309"
D_N32_ROOT_SPECS="results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/root_specs/kuramoto_n32_roots.txt"
D_N32_COLLECT_OUT="results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/collect"
D_N32_COMPARE_OUT="results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/compare"

D_N32_TRAIN=$(TASK_TSV="${D_N32_TASK_TSV}" BASE_OUT="${D_N32_BASE_OUT}" \
  sbatch --parsable --array=0-3 scripts/run_paper_benchmark_array.sh)
echo "  D-N32 train array: ${D_N32_TRAIN}"

D_N32_COLLECT=$(ROOT_SPECS_FILE="${D_N32_ROOT_SPECS}" OUT_DIR="${D_N32_COLLECT_OUT}" PAPER_SUMMARY=1 \
  sbatch --parsable --dependency=afterany:${D_N32_TRAIN} scripts/collect_paper_benchmark.sh)
echo "  D-N32 collector: ${D_N32_COLLECT}"

D_N32_COMPARE=$(ROWS_CSV="${D_N32_COLLECT_OUT}/forecasting_rows.csv" OUT_DIR="${D_N32_COMPARE_OUT}" \
  CANDIDATE_ROOTS_CSV=lista_blockdiag ANCHOR_ROOT=generic_sparse HORIZON=1000 \
  sbatch --parsable --dependency=afterany:${D_N32_COLLECT} scripts/compare_paper_benchmark.sh)
echo "  D-N32 compare: ${D_N32_COMPARE}"

echo ""

# ─────────────────────────────────────────────────────────────────────
# H: Label-free clustering – 9 evaluation jobs
# Root cause: BASH_SOURCE resolved to /var/spool/slurmd/ under sbatch.
# Fixed: added SLURM_SUBMIT_DIR fallback in the array runner.
# ─────────────────────────────────────────────────────────────────────
echo "--- Workstream H: Label-free clustering ---"

H_TASK_TSV="results/paper_parallel_20260309_h_label_free_clustering/task_tables/paper_parallel_20260309_h_label_free_clustering.tsv"
H_SUMMARY_DIR="/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/summary"

H_EVAL=$(sbatch --parsable --array=0-8 \
  scripts/paper_parallel_20260309_h_run_label_free_clustering_array.sh "${H_TASK_TSV}")
echo "  H eval array: ${H_EVAL}"

H_COLLECT=$(sbatch --parsable --dependency=afterany:${H_EVAL} \
  --job-name=pp_h_lf_collect --partition=long-cpu --cpus-per-task=1 --mem=4G --time=00:15:00 \
  --output=/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/collect-%j.out \
  --wrap="cd /home/mila/l/lia/skae && export UV_LINK_MODE=copy && uv run python tools/paper_parallel_20260309_h_collect_label_free_clustering.py --task_tsv '${H_TASK_TSV}' --summary_dir '${H_SUMMARY_DIR}'")
echo "  H collector: ${H_COLLECT}"

echo ""
echo "=== All requeued ==="
echo "E: train=${E_TRAIN} collect=${E_COLLECT} compare=${E_COMPARE}"
echo "D-N32: train=${D_N32_TRAIN} collect=${D_N32_COLLECT} compare=${D_N32_COMPARE}"
echo "H: eval=${H_EVAL} collect=${H_COLLECT}"
