#!/bin/bash
#
# Requeue D-N32 (Kuramoto N=32 seeds 3,4) and H (label-free clustering).
# D-N32: The task table has log_dir format, not phase format, so we use direct sbatch --wrap.
# H: Fixed len(dataset) -> len(dataset.trajectories) bug.
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

REPO_ROOT="$(pwd)"
echo "=== Requeue D-N32 and H (attempt 2) ==="
echo "Repo: ${REPO_ROOT}"
echo ""

# ─────────────────────────────────────────────────────────────────────
# D-N32: Kuramoto N=32 extra seeds 3,4
# Use sbatch scripts (not --wrap) with explicit module loading.
# ─────────────────────────────────────────────────────────────────────
echo "--- D-N32: Kuramoto N=32 seeds 3,4 ---"

D_N32_ROOT_SPECS="results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/root_specs/kuramoto_n32_roots.txt"
D_N32_COLLECT_OUT="results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/collect"
D_N32_COMPARE_OUT="results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/compare"

COMMON_SBATCH="--partition=long --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=24:00:00"

# generic_sparse seed 3
D1=$(sbatch --parsable ${COMMON_SBATCH} --job-name=d_n32_gs3 \
  --output=/network/scratch/l/lia/skae/d-n32-gs3-%j.out \
  --error=/network/scratch/l/lia/skae/d-n32-gs3-%j.err \
  --wrap="#!/bin/bash
cd ${REPO_ROOT}
module load cuda/12.6.0
module load cuda/12.6.0/cudnn/9.3
UV_NO_SYNC=1 uv run python tools/train.py \
  --config generic_sparse --env kuramoto --env_dt 0.00625 \
  --num_steps 200000 --batch_size 256 --target_size 256 --sequence_length 8 \
  --res_coeff 1.0 --reconst_coeff 0.03 --pred_coeff 1.0 --sparsity_coeff 0.0025 \
  --eval_profile full --seed 3 --device cuda \
  --kuramoto_num_oscillators 32 \
  --log_dir /network/scratch/l/lia/skae/kuramoto_n32_dt00625_200k_confirm_20260309/generic_sparse")
echo "  gs seed3: ${D1}"

# generic_sparse seed 4
D2=$(sbatch --parsable ${COMMON_SBATCH} --job-name=d_n32_gs4 \
  --output=/network/scratch/l/lia/skae/d-n32-gs4-%j.out \
  --error=/network/scratch/l/lia/skae/d-n32-gs4-%j.err \
  --wrap="#!/bin/bash
cd ${REPO_ROOT}
module load cuda/12.6.0
module load cuda/12.6.0/cudnn/9.3
UV_NO_SYNC=1 uv run python tools/train.py \
  --config generic_sparse --env kuramoto --env_dt 0.00625 \
  --num_steps 200000 --batch_size 256 --target_size 256 --sequence_length 8 \
  --res_coeff 1.0 --reconst_coeff 0.03 --pred_coeff 1.0 --sparsity_coeff 0.0025 \
  --eval_profile full --seed 4 --device cuda \
  --kuramoto_num_oscillators 32 \
  --log_dir /network/scratch/l/lia/skae/kuramoto_n32_dt00625_200k_confirm_20260309/generic_sparse")
echo "  gs seed4: ${D2}"

# lista_blockdiag seed 3
D3=$(sbatch --parsable ${COMMON_SBATCH} --job-name=d_n32_bd3 \
  --output=/network/scratch/l/lia/skae/d-n32-bd3-%j.out \
  --error=/network/scratch/l/lia/skae/d-n32-bd3-%j.err \
  --wrap="#!/bin/bash
cd ${REPO_ROOT}
module load cuda/12.6.0
module load cuda/12.6.0/cudnn/9.3
UV_NO_SYNC=1 uv run python tools/train.py \
  --config lista_parity_generic_sparse --env kuramoto --env_dt 0.00625 \
  --num_steps 200000 --batch_size 256 --target_size 256 --sequence_length 8 \
  --res_coeff 1.0 --reconst_coeff 0.03 --pred_coeff 1.0 --sparsity_coeff 0.006 \
  --lista_alpha 0.15 --lista_num_loops 1 --lista_final_op relu \
  --k_structure block_diagonal --k_block_size 16 \
  --eval_profile full --seed 3 --device cuda \
  --kuramoto_num_oscillators 32 \
  --log_dir /network/scratch/l/lia/skae/kuramoto_n32_dt00625_200k_confirm_20260309/lista_blockdiag")
echo "  bd seed3: ${D3}"

# lista_blockdiag seed 4
D4=$(sbatch --parsable ${COMMON_SBATCH} --job-name=d_n32_bd4 \
  --output=/network/scratch/l/lia/skae/d-n32-bd4-%j.out \
  --error=/network/scratch/l/lia/skae/d-n32-bd4-%j.err \
  --wrap="#!/bin/bash
cd ${REPO_ROOT}
module load cuda/12.6.0
module load cuda/12.6.0/cudnn/9.3
UV_NO_SYNC=1 uv run python tools/train.py \
  --config lista_parity_generic_sparse --env kuramoto --env_dt 0.00625 \
  --num_steps 200000 --batch_size 256 --target_size 256 --sequence_length 8 \
  --res_coeff 1.0 --reconst_coeff 0.03 --pred_coeff 1.0 --sparsity_coeff 0.006 \
  --lista_alpha 0.15 --lista_num_loops 1 --lista_final_op relu \
  --k_structure block_diagonal --k_block_size 16 \
  --eval_profile full --seed 4 --device cuda \
  --kuramoto_num_oscillators 32 \
  --log_dir /network/scratch/l/lia/skae/kuramoto_n32_dt00625_200k_confirm_20260309/lista_blockdiag")
echo "  bd seed4: ${D4}"

# Collector and compare
D_N32_COLLECT=$(ROOT_SPECS_FILE="${D_N32_ROOT_SPECS}" OUT_DIR="${D_N32_COLLECT_OUT}" PAPER_SUMMARY=1 \
  sbatch --parsable --dependency=afterany:${D1}:${D2}:${D3}:${D4} scripts/collect_paper_benchmark.sh)
echo "  collect: ${D_N32_COLLECT}"

D_N32_COMPARE=$(ROWS_CSV="${D_N32_COLLECT_OUT}/forecasting_rows.csv" OUT_DIR="${D_N32_COMPARE_OUT}" \
  CANDIDATE_ROOTS_CSV=lista_blockdiag ANCHOR_ROOT=generic_sparse HORIZON=1000 \
  sbatch --parsable --dependency=afterany:${D_N32_COLLECT} scripts/compare_paper_benchmark.sh)
echo "  compare: ${D_N32_COMPARE}"

echo ""

# ─────────────────────────────────────────────────────────────────────
# H: Label-free clustering (fixed len() bug)
# ─────────────────────────────────────────────────────────────────────
echo "--- H: Label-free clustering (attempt 3) ---"

H_TASK_TSV="results/paper_parallel_20260309_h_label_free_clustering/task_tables/paper_parallel_20260309_h_label_free_clustering.tsv"
H_SUMMARY_DIR="/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/summary"

H_EVAL=$(sbatch --parsable --array=0-8 \
  scripts/paper_parallel_20260309_h_run_label_free_clustering_array.sh "${H_TASK_TSV}")
echo "  H eval array: ${H_EVAL}"

H_COLLECT=$(sbatch --parsable --dependency=afterany:${H_EVAL} \
  --job-name=pp_h_lf_collect --partition=long-cpu --cpus-per-task=1 --mem=4G --time=00:15:00 \
  --output=/network/scratch/l/lia/skae/paper_parallel_20260309_h_label_free_clustering/collect-%j.out \
  --wrap="cd ${REPO_ROOT} && export UV_LINK_MODE=copy && uv run python tools/paper_parallel_20260309_h_collect_label_free_clustering.py --task_tsv '${H_TASK_TSV}' --summary_dir '${H_SUMMARY_DIR}'")
echo "  H collector: ${H_COLLECT}"

echo ""
echo "=== Done ==="
echo "D-N32: train=${D1},${D2},${D3},${D4} collect=${D_N32_COLLECT} compare=${D_N32_COMPARE}"
echo "H: eval=${H_EVAL} collect=${H_COLLECT}"
