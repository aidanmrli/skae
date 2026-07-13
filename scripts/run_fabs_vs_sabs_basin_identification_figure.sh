#!/bin/bash
#SBATCH --job-name=fabs-sabs-fig
#SBATCH --output=/tmp/fabs_sabs_figure_%j.out
#SBATCH --error=/tmp/fabs_sabs_figure_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=long

set -euo pipefail

cd /home/mila/l/lia/skae
uv run python tools/make_fabs_vs_sabs_basin_identification.py
