#!/bin/bash
#
# SLURM array runner for staged routed local-K LISTA training.
#
# Required env vars:
#   TASK_TSV=<path>
#   BASE_OUT=<output root>
#
# Optional:
#   ARRAY_OFFSET=0
#   SUPPORT_DEFINITION=absolute:0.001
#   FAMILY_JACCARD_THRESHOLD=0.4
#   SUPPORT_FIT_BATCHES=16
#   MIN_FAMILY_TRANSITIONS=1
#   SUPPORT_FAMILY_FIT_SOURCE=stage1_buffer
#   ROUTING_OBJECT=support_family
#   STABLE_BASE_OBJECT=family
#   STABLE_BASE_FAMILY_JACCARD=0.8
#   STABLE_TAIL_WINDOW=32
#   STABLE_MIN_EDGE_COUNT=2
#   STABLE_MIN_EDGE_PROBABILITY=0.02
#   STABLE_MAX_RECURRENT_OUT_PROBABILITY=0.05
#   STABLE_MIN_TAIL_COUNT=8
#   STABLE_MIN_ABSORPTION_OBSERVATIONS=8
#   STABLE_MIN_ABSORPTION_CONFIDENCE=0.80
#   STABLE_FIT_TRAJECTORIES=256
#   STABLE_FIT_TRAJECTORY_LENGTH=192
#   STABLE_FIT_SEED_OFFSET=271828
#   LOCAL_MAP_PARAMETERIZATION=source_target_affine_global_init
#   LOCAL_LR=
#   STAGE2_SELECTION_METRIC=
#   STAGE2_SELECTION_PERIODS=
#   STAGE2_SELECTION_HORIZONS=
#   STAGE2_SELECTION_BATCH_SIZE=
#   STAGE2_SELECTION_SEED_OFFSET=
#   BASELINE_ROUTE_SEED_OFFSET=
#   BASELINE_LATENT_CLUSTER_COUNT=
#   BASELINE_KMEANS_N_INIT=
#   LATENT_FATE_TAIL_WINDOW=16
#   LATENT_FATE_MAX_CLUSTERS=12
#   LATENT_FATE_MIN_SILHOUETTE=0.05
#   LATENT_FATE_PCA_COMPONENTS=16
#   EVAL_PERIODIC_PERIODS_OVERRIDE=
#   NUM_STEPS_OVERRIDE=
#   STAGE1_STEPS_OVERRIDE=
#   EVAL_EVERY_OVERRIDE=
#   EVAL_NUM_STEPS_OVERRIDE=
#   EVAL_PROFILE=full
#   SKIP_COMPLETED=1
#   RESUME_FROM_LATEST=1
#   SAVE_METRICS_HISTORY=0
#   SAVE_LAST_CHECKPOINT=0
#   SAVE_STAGE2_ARTIFACTS=0
#   SAVE_EVAL_ROLLOUT_ARTIFACTS=0
#   SAVE_EVAL_PLOTS=0
#   SAVE_EVAL_PER_IC_VALUES=0
#   SAVE_EVAL_ERROR_CURVES=0

#SBATCH --job-name=staged_fabs_k
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH -o /network/scratch/l/lia/skae/staged-fabs-local-k-%A_%a.out
#SBATCH -e /network/scratch/l/lia/skae/staged-fabs-local-k-%A_%a.err
#SBATCH --requeue

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

source scripts/slurm_gpu_guard.sh
trap gpu_guard_stop_sampler EXIT
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Repo: ${ROOT_DIR}"
echo "Git commit: $(git rev-parse HEAD)"
echo "Start Time: $(date)"

TASK_TSV="${TASK_TSV:?TASK_TSV is required}"
BASE_OUT="${BASE_OUT:?BASE_OUT is required}"
ARRAY_OFFSET="${ARRAY_OFFSET:-0}"
SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-absolute:0.001}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.4}"
SUPPORT_FIT_BATCHES="${SUPPORT_FIT_BATCHES:-16}"
MIN_FAMILY_TRANSITIONS="${MIN_FAMILY_TRANSITIONS:-1}"
SUPPORT_FAMILY_FIT_SOURCE="${SUPPORT_FAMILY_FIT_SOURCE:-stage1_buffer}"
ROUTING_OBJECT="${ROUTING_OBJECT:-support_family}"
STABLE_BASE_OBJECT="${STABLE_BASE_OBJECT:-family}"
STABLE_BASE_FAMILY_JACCARD="${STABLE_BASE_FAMILY_JACCARD:-0.8}"
STABLE_TAIL_WINDOW="${STABLE_TAIL_WINDOW:-32}"
STABLE_MIN_EDGE_COUNT="${STABLE_MIN_EDGE_COUNT:-2}"
STABLE_MIN_EDGE_PROBABILITY="${STABLE_MIN_EDGE_PROBABILITY:-0.02}"
STABLE_MAX_RECURRENT_OUT_PROBABILITY="${STABLE_MAX_RECURRENT_OUT_PROBABILITY:-0.05}"
STABLE_MIN_TAIL_COUNT="${STABLE_MIN_TAIL_COUNT:-8}"
STABLE_MIN_ABSORPTION_OBSERVATIONS="${STABLE_MIN_ABSORPTION_OBSERVATIONS:-8}"
STABLE_MIN_ABSORPTION_CONFIDENCE="${STABLE_MIN_ABSORPTION_CONFIDENCE:-0.80}"
STABLE_FIT_TRAJECTORIES="${STABLE_FIT_TRAJECTORIES:-256}"
STABLE_FIT_TRAJECTORY_LENGTH="${STABLE_FIT_TRAJECTORY_LENGTH:-192}"
STABLE_FIT_SEED_OFFSET="${STABLE_FIT_SEED_OFFSET:-271828}"
BASELINE_ROUTE_SEED_OFFSET="${BASELINE_ROUTE_SEED_OFFSET:-314159}"
BASELINE_LATENT_CLUSTER_COUNT="${BASELINE_LATENT_CLUSTER_COUNT:-0}"
BASELINE_KMEANS_N_INIT="${BASELINE_KMEANS_N_INIT:-10}"
LATENT_FATE_TAIL_WINDOW="${LATENT_FATE_TAIL_WINDOW:-16}"
LATENT_FATE_MAX_CLUSTERS="${LATENT_FATE_MAX_CLUSTERS:-12}"
LATENT_FATE_MIN_SILHOUETTE="${LATENT_FATE_MIN_SILHOUETTE:-0.05}"
LATENT_FATE_PCA_COMPONENTS="${LATENT_FATE_PCA_COMPONENTS:-16}"
LOCAL_MAP_PARAMETERIZATION="${LOCAL_MAP_PARAMETERIZATION:-source_target_affine_global_init}"
LOCAL_LR="${LOCAL_LR:-}"
STAGE2_SELECTION_METRIC="${STAGE2_SELECTION_METRIC:-}"
STAGE2_SELECTION_PERIODS="${STAGE2_SELECTION_PERIODS:-}"
STAGE2_SELECTION_HORIZONS="${STAGE2_SELECTION_HORIZONS:-}"
STAGE2_SELECTION_BATCH_SIZE="${STAGE2_SELECTION_BATCH_SIZE:-}"
STAGE2_SELECTION_SEED_OFFSET="${STAGE2_SELECTION_SEED_OFFSET:-}"
EVAL_PERIODIC_PERIODS_OVERRIDE="${EVAL_PERIODIC_PERIODS_OVERRIDE:-}"
NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE:-}"
STAGE1_STEPS_OVERRIDE="${STAGE1_STEPS_OVERRIDE:-}"
EVAL_EVERY_OVERRIDE="${EVAL_EVERY_OVERRIDE:-}"
EVAL_NUM_STEPS_OVERRIDE="${EVAL_NUM_STEPS_OVERRIDE:-}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
SAVE_METRICS_HISTORY="${SAVE_METRICS_HISTORY:-0}"
SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT:-0}"
SAVE_STAGE2_ARTIFACTS="${SAVE_STAGE2_ARTIFACTS:-0}"
SAVE_EVAL_ROLLOUT_ARTIFACTS="${SAVE_EVAL_ROLLOUT_ARTIFACTS:-0}"
SAVE_EVAL_PLOTS="${SAVE_EVAL_PLOTS:-0}"
SAVE_EVAL_PER_IC_VALUES="${SAVE_EVAL_PER_IC_VALUES:-0}"
SAVE_EVAL_ERROR_CURVES="${SAVE_EVAL_ERROR_CURVES:-0}"

tagify() {
  local raw="$1"
  raw="${raw//-/m}"
  raw="${raw//./p}"
  echo "${raw}"
}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
LINE_NO=$((TASK_ID + ARRAY_OFFSET + 2))
TASK_LINE="$(sed -n "${LINE_NO}p" "${TASK_TSV}" || true)"
if [[ -z "${TASK_LINE}" ]]; then
  echo "No task row for array index ${TASK_ID} (line ${LINE_NO}) in ${TASK_TSV}. Exiting."
  exit 0
fi

TASK_EXPORTS="$(
  uv run python - "${TASK_TSV}" "${LINE_NO}" <<'PY'
import csv
import shlex
import sys

path = sys.argv[1]
line_no = int(sys.argv[2])

with open(path, newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    for current_line_no, row in enumerate(reader, start=2):
        if current_line_no != line_no:
            continue
        for key, value in row.items():
            print(f"{key}={shlex.quote(value or '')}")
        break
    else:
        sys.exit(3)
PY
)"
eval "${TASK_EXPORTS}"

DT_TAG="$(tagify "${env_dt}")"
SYSTEM_SLUG="${system_slug:-${system_key//:/_}}"
SEED_DIR="${BASE_OUT}/${phase}/${model_variant}/${SYSTEM_SLUG}/dt_${DT_TAG}/seed_${seed}"
COMPLETED_RUN=""
if [[ "${SKIP_COMPLETED}" == "1" && -d "${SEED_DIR}" ]]; then
  COMPLETED_RUN="$(
    find "${SEED_DIR}" -mindepth 1 -maxdepth 1 -type d \
      -name '20*' -exec test -f '{}/evaluation_results_best.json' ';' -print \
      | sort | tail -n 1
  )"
fi

echo "============================================="
echo "Staged Routed Local-K Array Runner"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array Task: ${TASK_ID}"
echo "Task Row: ${task_id}"
echo "Phase: ${phase}"
echo "Model Variant: ${model_variant}"
echo "System: ${system_key}"
echo "Env: ${env_name}"
echo "Seed: ${seed}"
echo "DT: ${env_dt}"
echo "Total steps: ${num_steps}"
echo "Support: ${SUPPORT_DEFINITION}"
echo "Family Jaccard threshold: ${FAMILY_JACCARD_THRESHOLD}"
echo "Support fit batches: ${SUPPORT_FIT_BATCHES}"
echo "Min family transitions: ${MIN_FAMILY_TRANSITIONS}"
echo "Support-family fit source: ${SUPPORT_FAMILY_FIT_SOURCE}"
echo "Routing object: ${ROUTING_OBJECT}"
echo "Stable base object: ${STABLE_BASE_OBJECT}"
echo "Stable base family Jaccard: ${STABLE_BASE_FAMILY_JACCARD}"
echo "Stable tail window: ${STABLE_TAIL_WINDOW}"
echo "Stable min edge count: ${STABLE_MIN_EDGE_COUNT}"
echo "Stable min edge probability: ${STABLE_MIN_EDGE_PROBABILITY}"
echo "Stable max recurrent out probability: ${STABLE_MAX_RECURRENT_OUT_PROBABILITY}"
echo "Stable min tail count: ${STABLE_MIN_TAIL_COUNT}"
echo "Stable min absorption observations: ${STABLE_MIN_ABSORPTION_OBSERVATIONS}"
echo "Stable min absorption confidence: ${STABLE_MIN_ABSORPTION_CONFIDENCE}"
echo "Stable fit trajectories: ${STABLE_FIT_TRAJECTORIES}"
echo "Stable fit trajectory length: ${STABLE_FIT_TRAJECTORY_LENGTH}"
echo "Stable fit seed offset: ${STABLE_FIT_SEED_OFFSET}"
echo "Baseline route seed offset: ${BASELINE_ROUTE_SEED_OFFSET}"
echo "Baseline latent cluster count: ${BASELINE_LATENT_CLUSTER_COUNT}"
echo "Baseline k-means n_init: ${BASELINE_KMEANS_N_INIT}"
echo "Latent fate tail window: ${LATENT_FATE_TAIL_WINDOW}"
echo "Latent fate max clusters: ${LATENT_FATE_MAX_CLUSTERS}"
echo "Latent fate min silhouette: ${LATENT_FATE_MIN_SILHOUETTE}"
echo "Latent fate PCA components: ${LATENT_FATE_PCA_COMPONENTS}"
echo "Local map parameterization: ${LOCAL_MAP_PARAMETERIZATION}"
echo "Stage-2 selection metric: ${STAGE2_SELECTION_METRIC:-default}"
echo "Stage-2 selection periods: ${STAGE2_SELECTION_PERIODS:-default}"
echo "Stage-2 selection horizons: ${STAGE2_SELECTION_HORIZONS:-default}"
echo "Stage-2 selection batch size: ${STAGE2_SELECTION_BATCH_SIZE:-default}"
echo "Stage-2 selection seed offset: ${STAGE2_SELECTION_SEED_OFFSET:-default}"
echo "Eval periodic periods override: ${EVAL_PERIODIC_PERIODS_OVERRIDE:-default}"
echo "Eval profile: ${EVAL_PROFILE}"
echo "Skip completed: ${SKIP_COMPLETED}"
echo "Resume from latest: ${RESUME_FROM_LATEST}"
echo "Base out: ${BASE_OUT}"
echo "Seed dir: ${SEED_DIR}"
if [[ -n "${COMPLETED_RUN}" ]]; then
  echo "Completed staged run already exists: ${COMPLETED_RUN}"
  echo "Skipping completed task before CUDA setup."
  exit 0
fi
module load cuda/12.6.0
gpu_guard_assert_cuda_visible "staged local-K task ${task_id}"
gpu_guard_print_context "Staged Routed Local-K Array Runner"
echo "============================================="

TRAIN_ARGS=(
  --task_tsv "${TASK_TSV}"
  --array_index "${TASK_ID}"
  --array_offset "${ARRAY_OFFSET}"
  --base_out "${BASE_OUT}"
  --support_definition "${SUPPORT_DEFINITION}"
  --family_jaccard_threshold "${FAMILY_JACCARD_THRESHOLD}"
  --support_fit_batches "${SUPPORT_FIT_BATCHES}"
  --min_family_transitions "${MIN_FAMILY_TRANSITIONS}"
  --support_family_fit_source "${SUPPORT_FAMILY_FIT_SOURCE}"
  --routing_object "${ROUTING_OBJECT}"
  --stable_base_object "${STABLE_BASE_OBJECT}"
  --stable_base_family_jaccard "${STABLE_BASE_FAMILY_JACCARD}"
  --stable_tail_window "${STABLE_TAIL_WINDOW}"
  --stable_min_edge_count "${STABLE_MIN_EDGE_COUNT}"
  --stable_min_edge_probability "${STABLE_MIN_EDGE_PROBABILITY}"
  --stable_max_recurrent_out_probability "${STABLE_MAX_RECURRENT_OUT_PROBABILITY}"
  --stable_min_tail_count "${STABLE_MIN_TAIL_COUNT}"
  --stable_min_absorption_observations "${STABLE_MIN_ABSORPTION_OBSERVATIONS}"
  --stable_min_absorption_confidence "${STABLE_MIN_ABSORPTION_CONFIDENCE}"
  --stable_fit_trajectories "${STABLE_FIT_TRAJECTORIES}"
  --stable_fit_trajectory_length "${STABLE_FIT_TRAJECTORY_LENGTH}"
  --stable_fit_seed_offset "${STABLE_FIT_SEED_OFFSET}"
  --baseline_route_seed_offset "${BASELINE_ROUTE_SEED_OFFSET}"
  --baseline_latent_cluster_count "${BASELINE_LATENT_CLUSTER_COUNT}"
  --baseline_kmeans_n_init "${BASELINE_KMEANS_N_INIT}"
  --latent_fate_tail_window "${LATENT_FATE_TAIL_WINDOW}"
  --latent_fate_max_clusters "${LATENT_FATE_MAX_CLUSTERS}"
  --latent_fate_min_silhouette "${LATENT_FATE_MIN_SILHOUETTE}"
  --latent_fate_pca_components "${LATENT_FATE_PCA_COMPONENTS}"
  --local_map_parameterization "${LOCAL_MAP_PARAMETERIZATION}"
  --device cuda
  --eval_profile "${EVAL_PROFILE}"
)

if [[ -n "${LOCAL_LR}" ]]; then
  TRAIN_ARGS+=(--local_lr "${LOCAL_LR}")
fi
if [[ -n "${STAGE2_SELECTION_METRIC}" ]]; then
  TRAIN_ARGS+=(--stage2_selection_metric "${STAGE2_SELECTION_METRIC}")
fi
if [[ -n "${STAGE2_SELECTION_PERIODS}" ]]; then
  TRAIN_ARGS+=(--stage2_selection_periods "${STAGE2_SELECTION_PERIODS}")
fi
if [[ -n "${STAGE2_SELECTION_HORIZONS}" ]]; then
  TRAIN_ARGS+=(--stage2_selection_horizons "${STAGE2_SELECTION_HORIZONS}")
fi
if [[ -n "${STAGE2_SELECTION_BATCH_SIZE}" ]]; then
  TRAIN_ARGS+=(--stage2_selection_batch_size "${STAGE2_SELECTION_BATCH_SIZE}")
fi
if [[ -n "${STAGE2_SELECTION_SEED_OFFSET}" ]]; then
  TRAIN_ARGS+=(--stage2_selection_seed_offset "${STAGE2_SELECTION_SEED_OFFSET}")
fi
if [[ -n "${EVAL_PERIODIC_PERIODS_OVERRIDE}" ]]; then
  TRAIN_ARGS+=(--eval_periodic_periods_override "${EVAL_PERIODIC_PERIODS_OVERRIDE}")
fi
if [[ -n "${NUM_STEPS_OVERRIDE}" ]]; then
  TRAIN_ARGS+=(--num_steps_override "${NUM_STEPS_OVERRIDE}")
fi
if [[ -n "${STAGE1_STEPS_OVERRIDE}" ]]; then
  TRAIN_ARGS+=(--stage1_steps_override "${STAGE1_STEPS_OVERRIDE}")
fi
if [[ -n "${EVAL_EVERY_OVERRIDE}" ]]; then
  TRAIN_ARGS+=(--eval_every_override "${EVAL_EVERY_OVERRIDE}")
fi
if [[ -n "${EVAL_NUM_STEPS_OVERRIDE}" ]]; then
  TRAIN_ARGS+=(--eval_num_steps_override "${EVAL_NUM_STEPS_OVERRIDE}")
fi
if [[ "${SKIP_COMPLETED}" == "1" ]]; then
  TRAIN_ARGS+=(--skip_completed)
fi
if [[ "${RESUME_FROM_LATEST}" == "0" ]]; then
  TRAIN_ARGS+=(--no_resume_from_latest)
fi
if [[ "${SAVE_METRICS_HISTORY}" == "1" ]]; then
  TRAIN_ARGS+=(--save_metrics_history)
fi
if [[ "${SAVE_LAST_CHECKPOINT}" == "1" ]]; then
  TRAIN_ARGS+=(--save_last_checkpoint)
fi
if [[ "${SAVE_STAGE2_ARTIFACTS}" == "1" ]]; then
  TRAIN_ARGS+=(--save_stage2_artifacts)
fi
if [[ "${SAVE_EVAL_ROLLOUT_ARTIFACTS}" == "1" ]]; then
  TRAIN_ARGS+=(--save_eval_rollout_artifacts)
fi
if [[ "${SAVE_EVAL_PLOTS}" == "1" ]]; then
  TRAIN_ARGS+=(--save_eval_plots)
fi
if [[ "${SAVE_EVAL_PER_IC_VALUES}" == "1" ]]; then
  TRAIN_ARGS+=(--save_eval_per_ic_values)
fi
if [[ "${SAVE_EVAL_ERROR_CURVES}" == "1" ]]; then
  TRAIN_ARGS+=(--save_eval_error_curves)
fi

gpu_guard_start_sampler \
  "${SEED_DIR}/gpu_utilization_${SLURM_JOB_ID:-local}_${TASK_ID}.csv" \
  "${GPU_TELEMETRY_INTERVAL:-30}"
gpu_guard_phase "staged local-K training start task_id=${task_id}"
set +e
uv run python tools/train_staged_support_family_local_k.py "${TRAIN_ARGS[@]}"
EXIT_CODE=$?
set -e
gpu_guard_phase "staged local-K training end task_id=${task_id} exit_code=${EXIT_CODE}"
gpu_guard_stop_sampler

echo "End Time: $(date)"
exit "${EXIT_CODE}"
