#!/bin/bash
#SBATCH --job-name=pp_a_clv_sup
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -o /network/scratch/l/lia/skae/paper-parallel-a-clv-support-%j.out
#SBATCH -e /network/scratch/l/lia/skae/paper-parallel-a-clv-support-%j.err

set -euo pipefail

cd /home/mila/l/lia/skae

OUTPUT_DIR="/home/mila/l/lia/skae/results/paper_parallel_20260309_a_competitive_lv_support_alignment"
mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1

uv run python tools/paper_parallel_20260309_a_competitive_lv_support_alignment.py \
  --output-dir "${OUTPUT_DIR}" \
  --system competitive_lv \
  --num-trajectories 100 \
  --trajectory-length 500 \
  --long-rollout-steps 5000 \
  --eval-seed 42 \
  --support-threshold 1e-3 \
  --support-mode mean \
  --cosine-aggregation mean \
  --entry "generic_sparse_ns200k_best::0::/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/generic_sparse_ns200k_best/competitive_lv/dt_0p01/seed_0/20260309-114103/checkpoint.pt" \
  --entry "generic_sparse_ns200k_best::1::/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/generic_sparse_ns200k_best/competitive_lv/dt_0p01/seed_1/20260309-114103/checkpoint.pt" \
  --entry "generic_sparse_ns200k_best::2::/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/generic_sparse_ns200k_best/competitive_lv/dt_0p01/seed_2/20260309-114103/checkpoint.pt" \
  --entry "lista_dense_promoted_stage4::0::/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309/paper_rerun/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3/competitive_lv/dt_0p01/seed_0/20260309-083735/checkpoint.pt" \
  --entry "lista_dense_promoted_stage4::1::/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309/paper_rerun/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3/competitive_lv/dt_0p01/seed_1/20260309-083735/checkpoint.pt" \
  --entry "lista_dense_promoted_stage4::2::/network/scratch/l/lia/skae/dense_lista_paper_rerun_stage4_20260309/paper_rerun/lista_dense_ns200k_lr5em5_klr5em6_wd1em4_rc3em2_pc1ep0_sc3em3/competitive_lv/dt_0p01/seed_2/20260309-083733/checkpoint.pt" \
  --entry "lista_blockdiag_ns200k_denseopt_sc3em3::0::/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/lista_blockdiag_ns200k_denseopt_sc3em3/competitive_lv/dt_0p01/seed_0/20260309-131149/checkpoint.pt" \
  --entry "lista_blockdiag_ns200k_denseopt_sc3em3::1::/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/lista_blockdiag_ns200k_denseopt_sc3em3/competitive_lv/dt_0p01/seed_1/20260309-131149/checkpoint.pt" \
  --entry "lista_blockdiag_ns200k_denseopt_sc3em3::2::/network/scratch/l/lia/skae/paper_followup_recipes_200k_20260309/paper_followup_recipes/lista_blockdiag_ns200k_denseopt_sc3em3/competitive_lv/dt_0p01/seed_2/20260309-131151/checkpoint.pt"
