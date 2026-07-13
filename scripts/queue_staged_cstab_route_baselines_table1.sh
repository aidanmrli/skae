#!/bin/bash
#
# Queue reviewer-facing matched route baselines for the retained Table 1
# controlled-multibasin C_stab staged local-K result.
#
# Each submitted child launcher reuses:
#   - the same dense-LISTA Table 1 source TSV and retained 15-system roster
#   - 200k total steps from the task table: 100k global stage + 100k local stage
#   - C_stab fit data size: 512 training-distribution trajectories of length 192
#   - learned-intercept local affine maps
#   - best-periodic-horizon checkpoint selection and the same final period grid
#
# Submit with:
#   sbatch scripts/queue_staged_cstab_route_baselines_table1.sh
#
# Optional env vars:
#   DATE_TAG=20260519
#   ROUTE_OBJECTS_CSV=support_family,oracle_basin,latent_kmeans,random_matched
#   ARRAY_THROTTLE=32
#   ARRAY_PARTITION=long
#   ARRAY_GPUS=
#   ARRAY_JOB_TIME=03:00:00
#   DOWNSTREAM_DEPENDENCY_TYPE=afterok
#   MAX_EXISTING_JOBS_BEFORE_SUBMIT=650
#   NUM_STEPS_OVERRIDE=
#   STAGE1_STEPS_OVERRIDE=
#   EVAL_EVERY_OVERRIDE=
#   EVAL_NUM_STEPS_OVERRIDE=
#   SKIP_COMPLETED=1
#   RESUME_FROM_LATEST=1
#   SAVE_LAST_CHECKPOINT=0

#SBATCH --job-name=queue_cstab_route_bases
#SBATCH --ntasks=1
#SBATCH --partition=main-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH -o /network/scratch/l/lia/skae/queue-cstab-route-baselines-%j.out
#SBATCH -e /network/scratch/l/lia/skae/queue-cstab-route-baselines-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_staged_cstab_route_baselines_table1.sh"
  exit 2
fi

DATE_TAG="${DATE_TAG:-20260519}"
ROUTE_OBJECTS_CSV="${ROUTE_OBJECTS_CSV:-support_family,oracle_basin,latent_kmeans,random_matched}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-32}"
ARRAY_PARTITION="${ARRAY_PARTITION:-long}"
ARRAY_GPUS="${ARRAY_GPUS:-}"
ARRAY_JOB_TIME="${ARRAY_JOB_TIME:-03:00:00}"
DOWNSTREAM_DEPENDENCY_TYPE="${DOWNSTREAM_DEPENDENCY_TYPE:-afterok}"
MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT:-650}"
CHILD_QUEUE_PARTITION="${CHILD_QUEUE_PARTITION:-main-cpu}"
CHILD_QUEUE_TIME="${CHILD_QUEUE_TIME:-00:30:00}"
CHILD_QUEUE_CPUS_PER_TASK="${CHILD_QUEUE_CPUS_PER_TASK:-1}"

SOURCE_TSV="${SOURCE_TSV:-results/transition_rich_lista_dense_p256_hardinit_table123_20260430/task_tables/transition_rich_lista_dense_p256_hardinit_table123.tsv}"
SOURCE_VARIANT="${SOURCE_VARIANT:-lista_dense_signsplit_p256_hardinit_basin_partition}"
BASELINE_ROOT_LABEL="${BASELINE_ROOT_LABEL:-lista_dense_signsplit_p256_hardinit_basin_partition}"
BASELINE_ROOT="${BASELINE_ROOT:-/network/scratch/l/lia/skae/transition_rich_lista_dense_p256_hardinit_table123_20260430/transition_rich_basin_partition/lista_dense_signsplit_p256_hardinit_basin_partition}"
PHASE_LABEL="${PHASE_LABEL:-transition_rich_basin_partition}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
SYSTEMS_CSV="${SYSTEMS_CSV:-}"
EXCLUDED_SYSTEMS_CSV="${EXCLUDED_SYSTEMS_CSV:-multiwell_strong_transition,claude:checkerboard_potential}"
EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT:-225}"

SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-absolute:0.001}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.4}"
SUPPORT_FIT_BATCHES="${SUPPORT_FIT_BATCHES:-16}"
MIN_FAMILY_TRANSITIONS="${MIN_FAMILY_TRANSITIONS:-1}"
SUPPORT_FAMILY_FIT_SOURCE="${SUPPORT_FAMILY_FIT_SOURCE:-stable_fit_trajectories}"
STABLE_BASE_OBJECT="${STABLE_BASE_OBJECT:-family}"
STABLE_BASE_FAMILY_JACCARD="${STABLE_BASE_FAMILY_JACCARD:-0.8}"
STABLE_TAIL_WINDOW="${STABLE_TAIL_WINDOW:-32}"
STABLE_MIN_EDGE_COUNT="${STABLE_MIN_EDGE_COUNT:-2}"
STABLE_MIN_EDGE_PROBABILITY="${STABLE_MIN_EDGE_PROBABILITY:-0.02}"
STABLE_MAX_RECURRENT_OUT_PROBABILITY="${STABLE_MAX_RECURRENT_OUT_PROBABILITY:-0.05}"
STABLE_MIN_TAIL_COUNT="${STABLE_MIN_TAIL_COUNT:-8}"
STABLE_MIN_ABSORPTION_OBSERVATIONS="${STABLE_MIN_ABSORPTION_OBSERVATIONS:-8}"
STABLE_MIN_ABSORPTION_CONFIDENCE="${STABLE_MIN_ABSORPTION_CONFIDENCE:-0.80}"
STABLE_FIT_TRAJECTORIES="${STABLE_FIT_TRAJECTORIES:-512}"
STABLE_FIT_TRAJECTORY_LENGTH="${STABLE_FIT_TRAJECTORY_LENGTH:-192}"
STABLE_FIT_SEED_OFFSET="${STABLE_FIT_SEED_OFFSET:-271828}"
BASELINE_ROUTE_SEED_OFFSET="${BASELINE_ROUTE_SEED_OFFSET:-314159}"
BASELINE_LATENT_CLUSTER_COUNT="${BASELINE_LATENT_CLUSTER_COUNT:-0}"
BASELINE_KMEANS_N_INIT="${BASELINE_KMEANS_N_INIT:-10}"
LOCAL_MAP_PARAMETERIZATION="${LOCAL_MAP_PARAMETERIZATION:-source_target_affine_learned_intercept}"
LOCAL_LR="${LOCAL_LR:-}"

STAGE2_SELECTION_METRIC="${STAGE2_SELECTION_METRIC:-best_periodic_horizon_mse}"
STAGE2_SELECTION_PERIODS="${STAGE2_SELECTION_PERIODS:-1,2,5,10,20,25,50,100}"
STAGE2_SELECTION_HORIZONS="${STAGE2_SELECTION_HORIZONS:-100,500,1000}"
STAGE2_SELECTION_BATCH_SIZE="${STAGE2_SELECTION_BATCH_SIZE:-32}"
STAGE2_SELECTION_SEED_OFFSET="${STAGE2_SELECTION_SEED_OFFSET:-12345}"
EVAL_PERIODIC_PERIODS_OVERRIDE="${EVAL_PERIODIC_PERIODS_OVERRIDE:-1,2,5,10,20,25,50,100}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE:-}"
STAGE1_STEPS_OVERRIDE="${STAGE1_STEPS_OVERRIDE:-}"
EVAL_EVERY_OVERRIDE="${EVAL_EVERY_OVERRIDE:-}"
EVAL_NUM_STEPS_OVERRIDE="${EVAL_NUM_STEPS_OVERRIDE:-}"
QUEUE_WIDE_PERIODIC_REEVAL="${QUEUE_WIDE_PERIODIC_REEVAL:-1}"
WIDE_REEVAL_HORIZONS_CSV="${WIDE_REEVAL_HORIZONS_CSV:-100,500,1000}"
WIDE_REEVAL_PERIODS_CSV="${WIDE_REEVAL_PERIODS_CSV:-1,2,5,10,20,25,50,100}"
WIDE_REEVAL_BATCH_SIZE="${WIDE_REEVAL_BATCH_SIZE:-100}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RESUME_FROM_LATEST="${RESUME_FROM_LATEST:-1}"
SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT:-0}"

AUTOMATION_DIR="${AUTOMATION_DIR:-results/staged_cstab_route_baselines_${DATE_TAG}/automation}"
mkdir -p "${AUTOMATION_DIR}"
SUBMISSIONS_FILE="${AUTOMATION_DIR}/route_baseline_queue_submissions.tsv"
printf 'routing_object\texperiment_tag\ttarget_variant\tqueue_job_id\n' > "${SUBMISSIONS_FILE}"

IFS=',' read -r -a ROUTE_OBJECTS <<< "${ROUTE_OBJECTS_CSV}"
for raw_route in "${ROUTE_OBJECTS[@]}"; do
  route="$(echo "${raw_route}" | xargs)"
  [[ -n "${route}" ]] || continue

  case "${route}" in
    stable_support_component)
      experiment_tag="staged_cstab_main_stable_support_component_lista_full_${DATE_TAG}"
      target_variant="lista_cstab_learned_intercept_k_staged_p256_hardinit_basin_partition"
      ;;
    support_family)
      experiment_tag="staged_cstab_baseline_support_family_lista_full_${DATE_TAG}"
      target_variant="lista_fabs_learned_intercept_k_staged_p256_hardinit_basin_partition"
      ;;
    oracle_basin)
      experiment_tag="staged_cstab_baseline_oracle_basin_lista_full_${DATE_TAG}"
      target_variant="lista_oracle_basin_learned_intercept_k_staged_p256_hardinit_basin_partition"
      ;;
    latent_kmeans)
      experiment_tag="staged_cstab_baseline_latent_kmeans_lista_full_${DATE_TAG}"
      target_variant="lista_latent_kmeans_cstab_count_learned_intercept_k_staged_p256_hardinit_basin_partition"
      ;;
    random_matched)
      experiment_tag="staged_cstab_baseline_random_matched_lista_full_${DATE_TAG}"
      target_variant="lista_random_cstab_matched_learned_intercept_k_staged_p256_hardinit_basin_partition"
      ;;
    *)
      echo "Unsupported route baseline '${route}'." >&2
      exit 1
      ;;
  esac

  echo "Submitting ${route} as ${experiment_tag}"
  queue_job_id=$(
    DATE_TAG="${DATE_TAG}" \
    EXPERIMENT_TAG="${experiment_tag}" \
    PHASE_LABEL="${PHASE_LABEL}" \
    BASE_OUT="/network/scratch/l/lia/skae/${experiment_tag}" \
    RESULTS_DIR="results/${experiment_tag}" \
    SOURCE_TSV="${SOURCE_TSV}" \
    SOURCE_VARIANT="${SOURCE_VARIANT}" \
    TARGET_VARIANT="${target_variant}" \
    BASELINE_ROOT_LABEL="${BASELINE_ROOT_LABEL}" \
    BASELINE_ROOT="${BASELINE_ROOT}" \
    SEEDS_CSV="${SEEDS_CSV}" \
    SYSTEMS_CSV="${SYSTEMS_CSV}" \
    EXCLUDED_SYSTEMS_CSV="${EXCLUDED_SYSTEMS_CSV}" \
    EXPECTED_TASK_COUNT="${EXPECTED_TASK_COUNT}" \
    SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
    FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}" \
    SUPPORT_FIT_BATCHES="${SUPPORT_FIT_BATCHES}" \
    MIN_FAMILY_TRANSITIONS="${MIN_FAMILY_TRANSITIONS}" \
    SUPPORT_FAMILY_FIT_SOURCE="${SUPPORT_FAMILY_FIT_SOURCE}" \
    ROUTING_OBJECT="${route}" \
    STABLE_BASE_OBJECT="${STABLE_BASE_OBJECT}" \
    STABLE_BASE_FAMILY_JACCARD="${STABLE_BASE_FAMILY_JACCARD}" \
    STABLE_TAIL_WINDOW="${STABLE_TAIL_WINDOW}" \
    STABLE_MIN_EDGE_COUNT="${STABLE_MIN_EDGE_COUNT}" \
    STABLE_MIN_EDGE_PROBABILITY="${STABLE_MIN_EDGE_PROBABILITY}" \
    STABLE_MAX_RECURRENT_OUT_PROBABILITY="${STABLE_MAX_RECURRENT_OUT_PROBABILITY}" \
    STABLE_MIN_TAIL_COUNT="${STABLE_MIN_TAIL_COUNT}" \
    STABLE_MIN_ABSORPTION_OBSERVATIONS="${STABLE_MIN_ABSORPTION_OBSERVATIONS}" \
    STABLE_MIN_ABSORPTION_CONFIDENCE="${STABLE_MIN_ABSORPTION_CONFIDENCE}" \
    STABLE_FIT_TRAJECTORIES="${STABLE_FIT_TRAJECTORIES}" \
    STABLE_FIT_TRAJECTORY_LENGTH="${STABLE_FIT_TRAJECTORY_LENGTH}" \
    STABLE_FIT_SEED_OFFSET="${STABLE_FIT_SEED_OFFSET}" \
    BASELINE_ROUTE_SEED_OFFSET="${BASELINE_ROUTE_SEED_OFFSET}" \
    BASELINE_LATENT_CLUSTER_COUNT="${BASELINE_LATENT_CLUSTER_COUNT}" \
    BASELINE_KMEANS_N_INIT="${BASELINE_KMEANS_N_INIT}" \
    LOCAL_MAP_PARAMETERIZATION="${LOCAL_MAP_PARAMETERIZATION}" \
    LOCAL_LR="${LOCAL_LR}" \
    STAGE2_SELECTION_METRIC="${STAGE2_SELECTION_METRIC}" \
    STAGE2_SELECTION_PERIODS="${STAGE2_SELECTION_PERIODS}" \
    STAGE2_SELECTION_HORIZONS="${STAGE2_SELECTION_HORIZONS}" \
    STAGE2_SELECTION_BATCH_SIZE="${STAGE2_SELECTION_BATCH_SIZE}" \
    STAGE2_SELECTION_SEED_OFFSET="${STAGE2_SELECTION_SEED_OFFSET}" \
    EVAL_PERIODIC_PERIODS_OVERRIDE="${EVAL_PERIODIC_PERIODS_OVERRIDE}" \
    NUM_STEPS_OVERRIDE="${NUM_STEPS_OVERRIDE}" \
    STAGE1_STEPS_OVERRIDE="${STAGE1_STEPS_OVERRIDE}" \
    EVAL_EVERY_OVERRIDE="${EVAL_EVERY_OVERRIDE}" \
    EVAL_NUM_STEPS_OVERRIDE="${EVAL_NUM_STEPS_OVERRIDE}" \
    EVAL_PROFILE="${EVAL_PROFILE}" \
    QUEUE_WIDE_PERIODIC_REEVAL="${QUEUE_WIDE_PERIODIC_REEVAL}" \
    WIDE_REEVAL_HORIZONS_CSV="${WIDE_REEVAL_HORIZONS_CSV}" \
    WIDE_REEVAL_PERIODS_CSV="${WIDE_REEVAL_PERIODS_CSV}" \
    WIDE_REEVAL_BATCH_SIZE="${WIDE_REEVAL_BATCH_SIZE}" \
    SKIP_COMPLETED="${SKIP_COMPLETED}" \
    RESUME_FROM_LATEST="${RESUME_FROM_LATEST}" \
    SAVE_LAST_CHECKPOINT="${SAVE_LAST_CHECKPOINT}" \
    ARRAY_THROTTLE="${ARRAY_THROTTLE}" \
    ARRAY_PARTITION="${ARRAY_PARTITION}" \
    ARRAY_GPUS="${ARRAY_GPUS}" \
    ARRAY_JOB_TIME="${ARRAY_JOB_TIME}" \
    DOWNSTREAM_DEPENDENCY_TYPE="${DOWNSTREAM_DEPENDENCY_TYPE}" \
    MAX_EXISTING_JOBS_BEFORE_SUBMIT="${MAX_EXISTING_JOBS_BEFORE_SUBMIT}" \
      sbatch --parsable \
        --partition="${CHILD_QUEUE_PARTITION}" \
        --time="${CHILD_QUEUE_TIME}" \
        --cpus-per-task="${CHILD_QUEUE_CPUS_PER_TASK}" \
        scripts/queue_staged_support_family_local_k_table1.sh
  )
  printf '%s\t%s\t%s\t%s\n' "${route}" "${experiment_tag}" "${target_variant}" "${queue_job_id}" \
    >> "${SUBMISSIONS_FILE}"
done

echo "Queued matched C_stab route baselines."
echo "Submission record: ${SUBMISSIONS_FILE}"
