#!/usr/bin/env bash
#
# Queue the state-observation action-conditioned control world-model suite.
#
# Submit with:
#   sbatch scripts/queue_control_world_model_state_pilot.sh
#
# Useful smoke override:
#   EXPERIMENT_TAG=control_wm_smoke TASKS_CSV=cartpole_swingup \
#   VARIANTS_CSV=sparse_additive,dense_additive SEEDS_CSV=0 \
#   DATA_FRACTIONS_CSV=1.0 NUM_EPISODES=12 EPISODE_LENGTH=32 \
#   NUM_STEPS=20 BATCH_SIZE=8 SEQUENCE_LENGTH=4 EVAL_HORIZONS=1,4 \
#   ARRAY_THROTTLE=1 RUNNER_TIME=00:20:00 \
#   sbatch scripts/queue_control_world_model_state_pilot.sh
#
#SBATCH --job-name=queue-ctrl-wm
#SBATCH --output=/network/scratch/l/lia/skae/queue-control-wm-%A.out
#SBATCH --error=/network/scratch/l/lia/skae/queue-control-wm-%A.err
#SBATCH --time=00:30:00
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

EXPERIMENT_TAG="${EXPERIMENT_TAG:-control_world_model_state_pilot_20260623}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}/runs}"
DATASET_ROOT="${DATASET_ROOT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}/datasets}"
TASK_TSV="${TASK_TSV:-${RESULTS_DIR}/control_world_model_tasks.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${RESULTS_DIR}/control_world_model_manifest.json}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"

TASKS_CSV="${TASKS_CSV:-cartpole_swingup,finger_spin,cheetah_run,walker_walk}"
VARIANTS_CSV="${VARIANTS_CSV:-sparse_additive,dense_additive,sparse_bilinear,dense_bilinear,mlp}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
DATA_FRACTIONS_CSV="${DATA_FRACTIONS_CSV:-0.1,0.25,0.5,1.0}"
DATASET_SEED="${DATASET_SEED:-0}"
NUM_EPISODES="${NUM_EPISODES:-256}"
EPISODE_LENGTH="${EPISODE_LENGTH:-250}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.70}"
VAL_FRACTION="${VAL_FRACTION:-0.15}"
NUM_STEPS="${NUM_STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-10}"
EVAL_HORIZONS="${EVAL_HORIZONS:-1,5,10,20,50}"
Z_DIM="${Z_DIM:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
EVAL_EVERY="${EVAL_EVERY:-500}"
PLANNING_CANDIDATES="${PLANNING_CANDIDATES:-256}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-16}"
RUNNER_TIME="${RUNNER_TIME:-04:00:00}"
RUNNER_MEM="${RUNNER_MEM:-12G}"
RUNNER_CPUS="${RUNNER_CPUS:-4}"

mkdir -p "${RESULTS_DIR}" "${BASE_OUT}" "${DATASET_ROOT}" "${LOG_DIR}"

uv run python tools/build_control_world_model_tasks.py \
  --output_tsv "${TASK_TSV}" \
  --output_manifest_json "${MANIFEST_JSON}" \
  --base_out "${BASE_OUT}" \
  --dataset_root "${DATASET_ROOT}" \
  --tasks_csv "${TASKS_CSV}" \
  --variants_csv "${VARIANTS_CSV}" \
  --seeds_csv "${SEEDS_CSV}" \
  --data_fractions_csv "${DATA_FRACTIONS_CSV}" \
  --dataset_seed "${DATASET_SEED}" \
  --num_episodes "${NUM_EPISODES}" \
  --episode_length "${EPISODE_LENGTH}" \
  --train_fraction "${TRAIN_FRACTION}" \
  --val_fraction "${VAL_FRACTION}" \
  --num_steps "${NUM_STEPS}" \
  --batch_size "${BATCH_SIZE}" \
  --sequence_length "${SEQUENCE_LENGTH}" \
  --eval_horizons "${EVAL_HORIZONS}" \
  --z_dim "${Z_DIM}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --eval_every "${EVAL_EVERY}" \
  --planning_candidates "${PLANNING_CANDIDATES}"

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if [[ "${TASK_COUNT}" -le 0 ]]; then
  echo "No tasks generated in ${TASK_TSV}."
  exit 1
fi

ARRAY_SPEC="0-$((TASK_COUNT - 1))%${ARRAY_THROTTLE}"
ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
    sbatch \
      --parsable \
      --array="${ARRAY_SPEC}" \
      --partition=long \
      --time="${RUNNER_TIME}" \
      --mem="${RUNNER_MEM}" \
      --cpus-per-task="${RUNNER_CPUS}" \
      --output="${LOG_DIR}/control-wm-%A_%a.out" \
      --error="${LOG_DIR}/control-wm-%A_%a.err" \
      scripts/run_control_world_model_state_array.sh
)
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"

QUEUE_JSON_PATH="${RESULTS_DIR}/queue.json" \
QUEUE_EXPERIMENT_TAG="${EXPERIMENT_TAG}" \
QUEUE_RESULTS_DIR="${RESULTS_DIR}" \
QUEUE_BASE_OUT="${BASE_OUT}" \
QUEUE_DATASET_ROOT="${DATASET_ROOT}" \
QUEUE_TASK_TSV="${TASK_TSV}" \
QUEUE_MANIFEST_JSON="${MANIFEST_JSON}" \
QUEUE_LOG_DIR="${LOG_DIR}" \
QUEUE_TASKS_CSV="${TASKS_CSV}" \
QUEUE_VARIANTS_CSV="${VARIANTS_CSV}" \
QUEUE_SEEDS_CSV="${SEEDS_CSV}" \
QUEUE_DATA_FRACTIONS_CSV="${DATA_FRACTIONS_CSV}" \
QUEUE_TASK_COUNT="${TASK_COUNT}" \
QUEUE_ARRAY_SPEC="${ARRAY_SPEC}" \
QUEUE_ARRAY_JOB_ID="${ARRAY_JOB_ID}" \
QUEUE_NUM_EPISODES="${NUM_EPISODES}" \
QUEUE_EPISODE_LENGTH="${EPISODE_LENGTH}" \
QUEUE_NUM_STEPS="${NUM_STEPS}" \
uv run python - <<'PY'
import json
import os

payload = {
    "experiment_tag": os.environ["QUEUE_EXPERIMENT_TAG"],
    "results_dir": os.environ["QUEUE_RESULTS_DIR"],
    "base_out": os.environ["QUEUE_BASE_OUT"],
    "dataset_root": os.environ["QUEUE_DATASET_ROOT"],
    "task_tsv": os.environ["QUEUE_TASK_TSV"],
    "manifest_json": os.environ["QUEUE_MANIFEST_JSON"],
    "log_dir": os.environ["QUEUE_LOG_DIR"],
    "tasks_csv": os.environ["QUEUE_TASKS_CSV"],
    "variants_csv": os.environ["QUEUE_VARIANTS_CSV"],
    "seeds_csv": os.environ["QUEUE_SEEDS_CSV"],
    "data_fractions_csv": os.environ["QUEUE_DATA_FRACTIONS_CSV"],
    "task_count": int(os.environ["QUEUE_TASK_COUNT"]),
    "array_spec": os.environ["QUEUE_ARRAY_SPEC"],
    "array_job_id": os.environ["QUEUE_ARRAY_JOB_ID"],
    "num_episodes": int(os.environ["QUEUE_NUM_EPISODES"]),
    "episode_length": int(os.environ["QUEUE_EPISODE_LENGTH"]),
    "num_steps": int(os.environ["QUEUE_NUM_STEPS"]),
}
with open(os.environ["QUEUE_JSON_PATH"], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY

echo "Queued control world-model state suite."
echo "Task count: ${TASK_COUNT}"
echo "Array job: ${ARRAY_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
