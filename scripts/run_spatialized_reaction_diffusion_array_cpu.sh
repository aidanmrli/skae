#!/usr/bin/env bash
#SBATCH --job-name=spatial-rd-array-cpu
#SBATCH --output=/network/scratch/l/lia/skae/spatial-rd-array-cpu-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/spatial-rd-array-cpu-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G

set -euo pipefail

bash scripts/run_spatialized_reaction_diffusion_array.sh
