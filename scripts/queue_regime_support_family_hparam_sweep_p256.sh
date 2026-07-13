#!/bin/bash
#
# Queue a support-family hyperparameter sweep for the p256 local-K baseline.
#
# This launcher intentionally disables explicit clustering baselines so each
# worker evaluates only global K, learned support-family local K, and oracle
# basin local K. The goal is to tune the label-free support-family route toward
# the oracle-basin upper-bound without changing the trained checkpoints.
#
# Submit with:
#   sbatch scripts/queue_regime_support_family_hparam_sweep_p256.sh
#
# Optional env vars:
#   EXPERIMENT_TAG=regime_support_family_hparam_p256_20260506_iter1
#   SUPPORT_DEFINITIONS_CSV=topk:4,topk:6,topk:8,topk:12,topk:16,absolute:0.001,absolute:0.003,absolute:0.01,relative:0.05,relative:0.1,relative:0.2
#   JACCARD_THRESHOLDS_CSV=0.20,0.32,0.40,0.50,0.60,0.70,0.80
#   MIN_OPERATOR_TRANSITIONS_CSV=128
#   RIDGE_LAMBDAS_CSV=1e-4
#
#SBATCH --job-name=queue_regime_hp
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-regime-hparam-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-regime-hparam-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_regime_support_family_hparam_sweep_p256.sh"
  exit 2
fi

slugify() {
  local raw="$1"
  raw="${raw//:/_}"
  raw="${raw//./p}"
  raw="${raw//-/m}"
  raw="${raw//+/p}"
  raw="${raw//,/_}"
  raw="${raw//\//_}"
  raw="${raw// /}"
  echo "${raw}"
}

EXPERIMENT_TAG="${EXPERIMENT_TAG:-regime_support_family_hparam_p256_20260506_iter1}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
RUNS_DIR="${RUNS_DIR:-${RESULTS_DIR}/runs}"
SUMMARY_DIR="${SUMMARY_DIR:-${RESULTS_DIR}/summary}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"

ROWS_CSVS="${ROWS_CSVS:-results/transition_rich_lista_dense_p256_hardinit_table123_20260430/collect_pass0/forecasting_rows.csv}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:-lista_dense_signsplit_p256_hardinit_basin_partition}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
SUPPORT_DEFINITIONS_CSV="${SUPPORT_DEFINITIONS_CSV:-topk:4,topk:6,topk:8,topk:12,topk:16,absolute:0.001,absolute:0.003,absolute:0.01,relative:0.05,relative:0.1,relative:0.2}"
JACCARD_THRESHOLDS_CSV="${JACCARD_THRESHOLDS_CSV:-0.20,0.32,0.40,0.50,0.60,0.70,0.80}"
MIN_OPERATOR_TRANSITIONS_CSV="${MIN_OPERATOR_TRANSITIONS_CSV:-128}"
RIDGE_LAMBDAS_CSV="${RIDGE_LAMBDAS_CSV:-1e-4}"
TRAIN_FRACTIONS_CSV="${TRAIN_FRACTIONS_CSV:-0.5}"
NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-256}"
TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH:-256}"
EVAL_SEED="${EVAL_SEED:-42}"
LABEL_MODE="${LABEL_MODE:-auto}"
JOB_TIME="${JOB_TIME:-06:00:00}"
JOB_CPUS="${JOB_CPUS:-2}"
JOB_MEM="${JOB_MEM:-12G}"
SUMMARY_TIME="${SUMMARY_TIME:-00:30:00}"
SUMMARY_MEM="${SUMMARY_MEM:-4G}"

mkdir -p "${RUNS_DIR}" "${SUMMARY_DIR}" "${LOG_DIR}" "${AUTOMATION_DIR}"

IFS=',' read -r -a SUPPORT_DEFINITIONS <<< "${SUPPORT_DEFINITIONS_CSV}"
IFS=',' read -r -a JACCARD_THRESHOLDS <<< "${JACCARD_THRESHOLDS_CSV}"
IFS=',' read -r -a MIN_OPERATOR_TRANSITIONS_VALUES <<< "${MIN_OPERATOR_TRANSITIONS_CSV}"
IFS=',' read -r -a RIDGE_LAMBDAS <<< "${RIDGE_LAMBDAS_CSV}"
IFS=',' read -r -a TRAIN_FRACTIONS <<< "${TRAIN_FRACTIONS_CSV}"

JOB_IDS=()
COMBO_COUNT=0
for support_definition in "${SUPPORT_DEFINITIONS[@]}"; do
  for jaccard_threshold in "${JACCARD_THRESHOLDS[@]}"; do
    for min_transitions in "${MIN_OPERATOR_TRANSITIONS_VALUES[@]}"; do
      for ridge_lambda in "${RIDGE_LAMBDAS[@]}"; do
        for train_fraction in "${TRAIN_FRACTIONS[@]}"; do
          support_slug="$(slugify "${support_definition}")"
          j_slug="$(slugify "${jaccard_threshold}")"
          ridge_slug="$(slugify "${ridge_lambda}")"
          train_slug="$(slugify "${train_fraction}")"
          combo_slug="${ROOT_LABELS_CSV}__${support_slug}__j_${j_slug}__min_${min_transitions}__ridge_${ridge_slug}__train_${train_slug}"
          out_dir="${RUNS_DIR}/${combo_slug}"
          mkdir -p "${out_dir}"
          submit_output=$(
            ROWS_CSVS="${ROWS_CSVS}" \
            OUT_DIR="${out_dir}" \
            ROOT_LABELS_CSV="${ROOT_LABELS_CSV}" \
            SEEDS_CSV="${SEEDS_CSV}" \
            SUPPORT_DEFINITION="${support_definition}" \
            FEATURE_VIEWS="none" \
            CLUSTER_METHODS="none" \
            CLUSTER_COUNT_MODES="none" \
            NUM_TRAJECTORIES="${NUM_TRAJECTORIES}" \
            TRAJECTORY_LENGTH="${TRAJECTORY_LENGTH}" \
            EVAL_SEED="${EVAL_SEED}" \
            LABEL_MODE="${LABEL_MODE}" \
            DEVICE="cpu" \
            RIDGE_LAMBDA="${ridge_lambda}" \
            MIN_OPERATOR_TRANSITIONS="${min_transitions}" \
            FAMILY_JACCARD_THRESHOLD="${jaccard_threshold}" \
            TRAIN_FRACTION="${train_fraction}" \
            SKIP_STATE_DECODE="1" \
            PROGRESS_EVERY_RUNS="5" \
            FLUSH_EVERY_RUNS="5" \
              sbatch \
                --parsable \
                --job-name="regime_hp_${support_slug}_j${j_slug}" \
                --time="${JOB_TIME}" \
                --cpus-per-task="${JOB_CPUS}" \
                --mem="${JOB_MEM}" \
                --output="${LOG_DIR}/${combo_slug}-%A.out" \
                --error="${LOG_DIR}/${combo_slug}-%A.err" \
                scripts/run_regime_discovery_local_koopman.sh
          )
          job_id="${submit_output%%;*}"
          JOB_IDS+=("${job_id}")
          COMBO_COUNT=$((COMBO_COUNT + 1))
        done
      done
    done
  done
done

DEPENDENCY="$(IFS=:; echo "${JOB_IDS[*]}")"
summary_submit_output=$(
  SWEEP_DIR="${RUNS_DIR}" \
  OUT_DIR="${SUMMARY_DIR}" \
  TOP_K="30" \
    sbatch \
      --parsable \
      --dependency=afterany:"${DEPENDENCY}" \
      --time="${SUMMARY_TIME}" \
      --mem="${SUMMARY_MEM}" \
      --output="${LOG_DIR}/summary-%A.out" \
      --error="${LOG_DIR}/summary-%A.err" \
      scripts/summarize_regime_support_family_hparam_sweep.sh
)
SUMMARY_JOB_ID="${summary_submit_output%%;*}"

cat > "${AUTOMATION_DIR}/queue.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "results_dir": "${RESULTS_DIR}",
  "runs_dir": "${RUNS_DIR}",
  "summary_dir": "${SUMMARY_DIR}",
  "rows_csvs": "${ROWS_CSVS}",
  "root_labels_csv": "${ROOT_LABELS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "support_definitions_csv": "${SUPPORT_DEFINITIONS_CSV}",
  "jaccard_thresholds_csv": "${JACCARD_THRESHOLDS_CSV}",
  "min_operator_transitions_csv": "${MIN_OPERATOR_TRANSITIONS_CSV}",
  "ridge_lambdas_csv": "${RIDGE_LAMBDAS_CSV}",
  "train_fractions_csv": "${TRAIN_FRACTIONS_CSV}",
  "num_trajectories": "${NUM_TRAJECTORIES}",
  "trajectory_length": "${TRAJECTORY_LENGTH}",
  "combo_count": "${COMBO_COUNT}",
  "job_ids": "$(IFS=,; echo "${JOB_IDS[*]}")",
  "summary_job_id": "${SUMMARY_JOB_ID}"
}
EOF

echo "Queued ${COMBO_COUNT} support-family local-K hyperparameter jobs."
echo "Worker jobs: $(IFS=,; echo "${JOB_IDS[*]}")"
echo "Summary job: ${SUMMARY_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
