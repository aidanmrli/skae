#!/bin/bash
# Disposable wrapper for the Dysts seq=10, 12-system, 15-seed re-train.
# Submit via:
#   sbatch --partition=long --mem=8G -c 2 -t 02:00:00 scripts/_queue_dysts_seq10_seeds0to14.sh
#
#SBATCH --job-name=queue_dysts_seq10
#SBATCH -o /network/scratch/l/lia/skae/queue-dysts-followup-%j.out
#SBATCH -e /network/scratch/l/lia/skae/queue-dysts-followup-%j.err

set -euo pipefail
cd /home/mila/l/lia/skae
source .venv/bin/activate

export SEEDS_CSV="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
export EXPERIMENT_TAG="paper_followup_recipes_200k_seq10_seeds0to14_20260428"
export SYSTEMS_CSV="dysts:Chua,dysts:Dadras,dysts:DequanLi,dysts:Hadley,dysts:LorenzCoupled,dysts:LuChenCheng,dysts:MultiChua,dysts:QiChen,dysts:Sakarya,dysts:SanUmSrisuchinwong,dysts:ShimizuMorioka,dysts:WangSun"
export RECIPE_SPECS_CSV="generic_sparse_ns200k_best:generic_sparse:200000:1e-4:1e-5:1e-4:0.03:1.0:0.006,generic_sparse_sc0_ns200k_best:generic_sparse:200000:1e-4:1e-5:1e-4:0.03:1.0:0.0,generic_sparse_blockdiag_ns200k_sc6em3:generic_sparse_blockdiag:200000:1e-4:1e-5:1e-4:0.03:1.0:0.006,lista_blockdiag_ns200k_denseopt_sc6em3:lista_blockdiag:200000:5e-5:5e-6:1e-4:0.03:1.0:0.006,lista_dense_promoted_stage4:lista_dense:200000:5e-5:5e-6:1e-4:0.03:1.0:0.006"
export ARRAY_PARALLEL=64

bash scripts/queue_paper_followup_recipes.sh
