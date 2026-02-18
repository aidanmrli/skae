#!/bin/bash

#SBATCH --job-name=lista_op_collect
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH -o /network/scratch/l/lia/skae/lista-op-collect-%j.out
#SBATCH --requeue

set -euo pipefail

source .venv/bin/activate

BASE_DIR="${BASE_DIR:-/network/scratch/l/lia/skae/lista_final_op_experiment}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/results}"

echo "Collecting LISTA final-op experiment results"
echo "Base: ${BASE_DIR}"
echo "Out:  ${OUTPUT_DIR}"

uv run python tools/collect_lista_final_op_experiment.py \
  --base_dir "${BASE_DIR}" \
  --output_dir "${OUTPUT_DIR}"
