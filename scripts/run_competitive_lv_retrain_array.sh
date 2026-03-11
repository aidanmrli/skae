#!/bin/bash
#SBATCH --job-name=clv_retrain
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/network/scratch/l/lia/skae/clv-retrain-%A_%a.out
#SBATCH --error=/network/scratch/l/lia/skae/clv-retrain-%A_%a.err

set -euo pipefail

TASK_TSV="${TASK_TSV:?TASK_TSV not set}"
BASE_OUT="${BASE_OUT:?BASE_OUT not set}"
TASK_IDX="${SLURM_ARRAY_TASK_ID:?Not running as array job}"

cd /home/mila/l/lia/skae

# Read task row. Using `read` with tab in IFS collapses empty fields, which
# shifts optional LISTA columns for dense/diagonal rows.
mapfile -t TASK_FIELDS < <(
    awk -F '\t' -v idx="${TASK_IDX}" '
        NR == idx + 2 {
            for (i = 1; i <= 17; ++i) {
                print $i
            }
            exit
        }
    ' "${TASK_TSV}"
)

if [[ ${#TASK_FIELDS[@]} -eq 0 ]]; then
    echo "ERROR: No task at index ${TASK_IDX}"
    exit 1
fi

TASK_ID="${TASK_FIELDS[0]-}"
GROUP="${TASK_FIELDS[1]-}"
ROOT_LABEL="${TASK_FIELDS[2]-}"
PRESET="${TASK_FIELDS[3]-}"
NUM_STEPS="${TASK_FIELDS[4]-}"
SEED="${TASK_FIELDS[5]-}"
LR="${TASK_FIELDS[6]-}"
K_LR="${TASK_FIELDS[7]-}"
WD="${TASK_FIELDS[8]-}"
SC="${TASK_FIELDS[9]-}"
RC="${TASK_FIELDS[10]-}"
PC="${TASK_FIELDS[11]-}"
K_STRUCT="${TASK_FIELDS[12]-}"
K_BS="${TASK_FIELDS[13]-}"
LA="${TASK_FIELDS[14]-}"
LNL="${TASK_FIELDS[15]-}"
LFO="${TASK_FIELDS[16]-}"

ENV="competitive_lv"
ENV_DT="0.01"

OUTPUT_DIR="${BASE_OUT}/${GROUP}/${ROOT_LABEL}/${ENV}/dt_0p01/seed_${SEED}"
echo "Task ${TASK_ID} (group=${GROUP}): root=${ROOT_LABEL} preset=${PRESET} steps=${NUM_STEPS} seed=${SEED}"
echo "Output: ${OUTPUT_DIR}"

# Build command
CMD=(uv run python tools/train.py
    --env "${ENV}"
    --env_dt "${ENV_DT}"
    --config "${PRESET}"
    --target_size 256
    --sequence_length 8
    --batch_size 256
    --num_steps "${NUM_STEPS}"
    --seed "${SEED}"
    --lr "${LR}"
    --k_matrix_lr "${K_LR}"
    --weight_decay "${WD}"
    --sparsity_coeff "${SC}"
    --reconst_coeff "${RC}"
    --pred_coeff "${PC}"
    --log_dir "${OUTPUT_DIR}"
)

# Add optional LISTA/K-structure overrides if non-empty
if [[ -n "${K_STRUCT}" ]]; then
    CMD+=(--k_structure "${K_STRUCT}")
fi
if [[ -n "${K_BS}" ]]; then
    CMD+=(--k_block_size "${K_BS}")
fi
if [[ -n "${LA}" ]]; then
    CMD+=(--lista_alpha "${LA}")
fi
if [[ -n "${LNL}" ]]; then
    CMD+=(--lista_num_loops "${LNL}")
fi
if [[ -n "${LFO}" ]]; then
    CMD+=(--lista_final_op "${LFO}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
