#!/bin/bash
#SBATCH --job-name=per_sys_pvalues
#SBATCH --partition=long
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH --time=00:30:00
#SBATCH --output=/network/scratch/l/lia/skae/per_sys_pvalues_%j.out
#SBATCH --error=/network/scratch/l/lia/skae/per_sys_pvalues_%j.err

set -euo pipefail
cd /home/mila/l/lia/skae

uv run python tools/per_system_paired_tests.py
