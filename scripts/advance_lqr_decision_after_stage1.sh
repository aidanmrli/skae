#!/bin/bash

#SBATCH --job-name=lqr_dec_adv
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH -o /network/scratch/l/lia/skae/lqr_dec_adv-%j.out
#SBATCH --requeue

# Automatically advance to Stage 2/3 after Stage 1 is complete.
# 1) Read Stage 1 outputs and choose BD* (bd_c1 or bd_c2)
# 2) Submit Stage 2 with BD_STAR
# 3) Submit Stage 3 dependent on Stage 2 completion

set -euo pipefail

module load cuda/12.6.0
source .venv/bin/activate

BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/lqr_decision}"
DECISION_JSON="${BASE_OUT}/stage1_bd_star_decision.json"

echo "[advance] selecting BD* from Stage 1 results under ${BASE_OUT}"
uv run python tools/select_bd_star_from_stage1.py \
  --base_dir "${BASE_OUT}" \
  --expected_runs 24 \
  --output_json "${DECISION_JSON}"

BD_STAR=$(uv run python - <<'PY'
import json
from pathlib import Path
p = Path("/network/scratch/l/lia/skae/lqr_decision/stage1_bd_star_decision.json")
with open(p) as f:
    d = json.load(f)
print(d["bd_star"])
PY
)

echo "[advance] selected BD_STAR=${BD_STAR}"

SUBMIT2=$(BD_STAR="${BD_STAR}" sbatch scripts/sweep_lqr_decision_stage2.sh)
JOB2=$(echo "$SUBMIT2" | awk '{print $4}')
echo "[advance] stage2 submit: ${SUBMIT2}"

SUBMIT3=$(BD_STAR="${BD_STAR}" sbatch --dependency=afterany:${JOB2} scripts/sweep_lqr_decision_stage3.sh)
JOB3=$(echo "$SUBMIT3" | awk '{print $4}')
echo "[advance] stage3 submit: ${SUBMIT3}"

echo "[advance] done. stage2_job=${JOB2} stage3_job=${JOB3} bd_star=${BD_STAR}"
