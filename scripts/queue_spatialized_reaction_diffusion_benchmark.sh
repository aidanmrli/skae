#!/usr/bin/env bash
#
# Queue the spatialized multibasin reaction-diffusion benchmark.
#
# Submit with:
#   sbatch scripts/queue_spatialized_reaction_diffusion_benchmark.sh
#
# Useful smoke override:
#   EXPERIMENT_TAG=spatial_rd_smoke_20260521 SYSTEMS_CSV=cal_square_4 \
#   SEEDS_CSV=0 MODEL_VARIANTS_CSV=conv_lista GRID_SIZE=16 NUM_STEPS=100 \
#   TRAIN_TRAJECTORIES=16 VAL_TRAJECTORIES=4 TEST_TRAJECTORIES=4 \
#   TRAJECTORY_LENGTH=12 LABEL_EXTRA_OBSERVATIONS=12 ARRAY_THROTTLE=1 \
#   sbatch scripts/queue_spatialized_reaction_diffusion_benchmark.sh
#
#SBATCH --job-name=queue-spatial-rd
#SBATCH --output=/network/scratch/l/lia/skae/queue-spatial-rd-%A.out
#SBATCH --error=/network/scratch/l/lia/skae/queue-spatial-rd-%A.err
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

EXPERIMENT_TAG="${EXPERIMENT_TAG:-spatialized_rd_conv_pilot_20260521}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
BASE_OUT="${BASE_OUT:-/network/scratch/l/lia/skae/${EXPERIMENT_TAG}}"
DATASET_BASE_OUT="${DATASET_BASE_OUT:-}"
TASK_TSV="${TASK_TSV:-${RESULTS_DIR}/spatialized_rd_tasks.tsv}"
MANIFEST_JSON="${MANIFEST_JSON:-${RESULTS_DIR}/spatialized_rd_manifest.json}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
PREFLIGHT_VALIDATE_DATASETS="${PREFLIGHT_VALIDATE_DATASETS:-0}"
PREFLIGHT_MIN_LABELS_PER_SPLIT="${PREFLIGHT_MIN_LABELS_PER_SPLIT:-2}"
PREFLIGHT_JSON="${PREFLIGHT_JSON:-${RESULTS_DIR}/dataset_preflight.json}"

SYSTEMS_CSV="${SYSTEMS_CSV:-cal_square_4,cal_high_cross_3,transition_routes_4,var_l_shape_5,cal_pentagon_5}"
MODEL_VARIANTS_CSV="${MODEL_VARIANTS_CSV:-conv_lista,conv_dense,conv_sparse_mlp}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
GRID_SIZE="${GRID_SIZE:-32}"
DIFFUSION="${DIFFUSION:-0.01}"
RK4_DT="${RK4_DT:-0.005}"
SUBSTEPS_PER_OBSERVATION="${SUBSTEPS_PER_OBSERVATION:-10}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-24}"
LABEL_EXTRA_OBSERVATIONS="${LABEL_EXTRA_OBSERVATIONS:-24}"
TRAIN_TRAJECTORIES="${TRAIN_TRAJECTORIES:-96}"
VAL_TRAJECTORIES="${VAL_TRAJECTORIES:-24}"
TEST_TRAJECTORIES="${TEST_TRAJECTORIES:-24}"
LAPLACIAN_SCALING="${LAPLACIAN_SCALING:-continuum}"
TARGET_SIZE="${TARGET_SIZE:-0}"
MIN_LATENT_STATE_RATIO="${MIN_LATENT_STATE_RATIO:-4.0}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-64}"
NUM_BLOCKS="${NUM_BLOCKS:-3}"
CONV_ACTIVATION="${CONV_ACTIVATION:-}"
NUM_STEPS="${NUM_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-4}"
TRAIN_OBSERVATION_LIMIT="${TRAIN_OBSERVATION_LIMIT:-0}"
LISTA_NUM_LOOPS="${LISTA_NUM_LOOPS:-2}"
LISTA_NUM_LOOPS_CSV="${LISTA_NUM_LOOPS_CSV:-}"
LISTA_ALPHA="${LISTA_ALPHA:-1e-3}"
LISTA_ALPHA_CSV="${LISTA_ALPHA_CSV:-}"
SPARSITY_COEFF="${SPARSITY_COEFF:-0}"
SPARSITY_COEFF_CSV="${SPARSITY_COEFF_CSV:-}"
SUPPORT_THRESHOLD="${SUPPORT_THRESHOLD:-1e-4}"
SUPPORT_THRESHOLD_CSV="${SUPPORT_THRESHOLD_CSV:-}"
FAMILY_JACCARD="${FAMILY_JACCARD:-0.7}"
FAMILY_JACCARD_CSV="${FAMILY_JACCARD_CSV:-}"
MAX_VALIDATION_REPS="${MAX_VALIDATION_REPS:-256}"
DEEP_THRESHOLD="${DEEP_THRESHOLD:-0.7}"
EVAL_HORIZONS="${EVAL_HORIZONS:-1,4,8,12}"
EVAL_HORIZON="${EVAL_HORIZON:-8}"
PERIODIC_REENCODE_PERIODS="${PERIODIC_REENCODE_PERIODS:-}"
EVAL_EVERY="${EVAL_EVERY:-500}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
K_STABILITY_WEIGHT="${K_STABILITY_WEIGHT:-1e-4}"
SHARE_DATASET_BY_SEED="${SHARE_DATASET_BY_SEED:-0}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-8}"
RUNNER_PARTITION="${RUNNER_PARTITION:-long}"
RUNNER_GRES="${RUNNER_GRES-gpu:1}"
RUNNER_TIME="${RUNNER_TIME:-02:50:00}"
RUNNER_MEM="${RUNNER_MEM:-16G}"
RUNNER_CPUS="${RUNNER_CPUS:-4}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/run_spatialized_reaction_diffusion_array.sh}"
if [[ -z "${RUNNER_GRES}" && "${RUNNER_SCRIPT}" == "scripts/run_spatialized_reaction_diffusion_array.sh" ]]; then
  RUNNER_SCRIPT="scripts/run_spatialized_reaction_diffusion_array_cpu.sh"
fi

mkdir -p "${RESULTS_DIR}" "${BASE_OUT}" "${LOG_DIR}"

BUILD_TASK_ARGS=(
  uv run python tools/build_spatialized_reaction_diffusion_tasks.py
  --output_tsv "${TASK_TSV}"
  --output_manifest_json "${MANIFEST_JSON}"
  --base_out "${BASE_OUT}"
  --systems_csv "${SYSTEMS_CSV}"
  --model_variants_csv "${MODEL_VARIANTS_CSV}"
  --seeds_csv "${SEEDS_CSV}"
  --grid_size "${GRID_SIZE}"
  --diffusion "${DIFFUSION}"
  --rk4_dt "${RK4_DT}"
  --substeps_per_observation "${SUBSTEPS_PER_OBSERVATION}"
  --trajectory_length "${TRAJECTORY_LENGTH}"
  --label_extra_observations "${LABEL_EXTRA_OBSERVATIONS}"
  --train_trajectories "${TRAIN_TRAJECTORIES}"
  --val_trajectories "${VAL_TRAJECTORIES}"
  --test_trajectories "${TEST_TRAJECTORIES}"
  --laplacian_scaling "${LAPLACIAN_SCALING}"
  --target_size "${TARGET_SIZE}"
  --min_latent_state_ratio "${MIN_LATENT_STATE_RATIO}"
  --hidden_channels "${HIDDEN_CHANNELS}"
  --num_blocks "${NUM_BLOCKS}"
  --conv_activation "${CONV_ACTIVATION}"
  --num_steps "${NUM_STEPS}"
  --batch_size "${BATCH_SIZE}"
  --sequence_length "${SEQUENCE_LENGTH}"
  --train_observation_limit "${TRAIN_OBSERVATION_LIMIT}"
  --lista_num_loops "${LISTA_NUM_LOOPS}"
  --lista_alpha "${LISTA_ALPHA}"
  --sparsity_coeff "${SPARSITY_COEFF}"
  --support_threshold "${SUPPORT_THRESHOLD}"
  --family_jaccard "${FAMILY_JACCARD}"
  --max_validation_reps "${MAX_VALIDATION_REPS}"
  --deep_threshold "${DEEP_THRESHOLD}"
  --eval_horizons "${EVAL_HORIZONS}"
  --eval_horizon "${EVAL_HORIZON}"
)
if [[ -n "${DATASET_BASE_OUT}" ]]; then
  BUILD_TASK_ARGS+=(--dataset_base_out "${DATASET_BASE_OUT}")
fi
if [[ "${SHARE_DATASET_BY_SEED}" == "1" ]]; then
  BUILD_TASK_ARGS+=(--share_dataset_by_seed)
fi
if [[ -n "${LISTA_ALPHA_CSV}" ]]; then
  BUILD_TASK_ARGS+=(--lista_alpha_csv "${LISTA_ALPHA_CSV}")
fi
if [[ -n "${LISTA_NUM_LOOPS_CSV}" ]]; then
  BUILD_TASK_ARGS+=(--lista_num_loops_csv "${LISTA_NUM_LOOPS_CSV}")
fi
if [[ -n "${SPARSITY_COEFF_CSV}" ]]; then
  BUILD_TASK_ARGS+=(--sparsity_coeff_csv "${SPARSITY_COEFF_CSV}")
fi
if [[ -n "${SUPPORT_THRESHOLD_CSV}" ]]; then
  BUILD_TASK_ARGS+=(--support_threshold_csv "${SUPPORT_THRESHOLD_CSV}")
fi
if [[ -n "${FAMILY_JACCARD_CSV}" ]]; then
  BUILD_TASK_ARGS+=(--family_jaccard_csv "${FAMILY_JACCARD_CSV}")
fi
"${BUILD_TASK_ARGS[@]}"

if [[ "${PREFLIGHT_VALIDATE_DATASETS}" == "1" ]]; then
  uv run python tools/preflight_spatialized_multibasin_datasets.py \
    --task_tsv "${TASK_TSV}" \
    --min_labels_per_split "${PREFLIGHT_MIN_LABELS_PER_SPLIT}" \
    --output_json "${PREFLIGHT_JSON}"
fi

TASK_COUNT=$(( $(wc -l < "${TASK_TSV}") - 1 ))
if [[ "${TASK_COUNT}" -le 0 ]]; then
  echo "No tasks generated in ${TASK_TSV}."
  exit 1
fi

ARRAY_SPEC="0-$((TASK_COUNT - 1))%${ARRAY_THROTTLE}"
SBATCH_RUNNER_ARGS=(
  --parsable
  --array="${ARRAY_SPEC}"
  --partition="${RUNNER_PARTITION}"
  --time="${RUNNER_TIME}"
  --mem="${RUNNER_MEM}"
  --cpus-per-task="${RUNNER_CPUS}"
      --output="${LOG_DIR}/spatial-rd-%A_%a.out"
      --error="${LOG_DIR}/spatial-rd-%A_%a.err"
)
if [[ -n "${RUNNER_GRES}" ]]; then
  SBATCH_RUNNER_ARGS+=(--gres="${RUNNER_GRES}")
fi
ARRAY_JOB_ID=$(
  TASK_TSV="${TASK_TSV}" \
  EVAL_EVERY="${EVAL_EVERY}" \
  RESUME_FROM_LATEST="${RESUME_FROM_LATEST}" \
  K_STABILITY_WEIGHT="${K_STABILITY_WEIGHT}" \
  PERIODIC_REENCODE_PERIODS="${PERIODIC_REENCODE_PERIODS}" \
    sbatch \
      "${SBATCH_RUNNER_ARGS[@]}" \
      "${RUNNER_SCRIPT}"
)
ARRAY_JOB_ID="${ARRAY_JOB_ID%%;*}"

cat > "${RESULTS_DIR}/queue.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "results_dir": "${RESULTS_DIR}",
  "base_out": "${BASE_OUT}",
  "dataset_base_out": "${DATASET_BASE_OUT}",
  "task_tsv": "${TASK_TSV}",
  "manifest_json": "${MANIFEST_JSON}",
  "log_dir": "${LOG_DIR}",
  "systems_csv": "${SYSTEMS_CSV}",
  "model_variants_csv": "${MODEL_VARIANTS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "task_count": ${TASK_COUNT},
  "min_latent_state_ratio": "${MIN_LATENT_STATE_RATIO}",
  "conv_activation": "${CONV_ACTIVATION}",
  "train_observation_limit": "${TRAIN_OBSERVATION_LIMIT}",
  "lista_num_loops_csv": "${LISTA_NUM_LOOPS_CSV}",
  "lista_alpha_csv": "${LISTA_ALPHA_CSV}",
  "sparsity_coeff_csv": "${SPARSITY_COEFF_CSV}",
  "support_threshold_csv": "${SUPPORT_THRESHOLD_CSV}",
  "family_jaccard_csv": "${FAMILY_JACCARD_CSV}",
  "array_spec": "${ARRAY_SPEC}",
  "runner_partition": "${RUNNER_PARTITION}",
  "runner_gres": "${RUNNER_GRES}",
  "runner_time": "${RUNNER_TIME}",
  "eval_every": "${EVAL_EVERY}",
  "periodic_reencode_periods": "${PERIODIC_REENCODE_PERIODS}",
  "k_stability_weight": "${K_STABILITY_WEIGHT}",
  "share_dataset_by_seed": "${SHARE_DATASET_BY_SEED}",
  "preflight_validate_datasets": "${PREFLIGHT_VALIDATE_DATASETS}",
  "preflight_min_labels_per_split": "${PREFLIGHT_MIN_LABELS_PER_SPLIT}",
  "preflight_json": "${PREFLIGHT_JSON}",
  "resume_from_latest": "${RESUME_FROM_LATEST}",
  "runner_script": "${RUNNER_SCRIPT}",
  "array_job_id": "${ARRAY_JOB_ID}"
}
EOF

echo "Queued spatialized reaction-diffusion benchmark."
echo "Task count: ${TASK_COUNT}"
echo "Array job: ${ARRAY_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
