#!/bin/bash
#
# Queue the p256 oracle-vs-learned local-Koopman comparison on existing runs.
#
# Submit with:
#   sbatch scripts/queue_oracle_vs_learned_local_koopman_p256.sh
#
# Optional env vars:
#   EXPERIMENT_TAG=oracle_vs_learned_local_koopman_20260506
#   ROWS_CSVS=results/transition_rich_lista_dense_p256_hardinit_table123_20260430/collect_pass0/forecasting_rows.csv
#   ROOT_LABELS_CSV=lista_dense_signsplit_p256_hardinit_basin_partition
#   SEED_SPLITS_SEMICOLON=0,1,2,3,4;5,6,7,8,9;10,11,12,13,14
#   SEEDS_CSV=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
#
#SBATCH --job-name=queue_oracle_lk
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH -o /network/scratch/l/lia/skae/queue-oracle-local-k-%A.out
#SBATCH -e /network/scratch/l/lia/skae/queue-oracle-local-k-%A.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This launcher must run on a compute node."
  echo "Submit it with: sbatch scripts/queue_oracle_vs_learned_local_koopman_p256.sh"
  exit 2
fi

EXPERIMENT_TAG="${EXPERIMENT_TAG:-oracle_vs_learned_local_koopman_20260506}"
RESULTS_DIR="${RESULTS_DIR:-results/${EXPERIMENT_TAG}}"
ROWS_CSVS="${ROWS_CSVS:-results/transition_rich_lista_dense_p256_hardinit_table123_20260430/collect_pass0/forecasting_rows.csv}"
ROOT_LABELS_CSV="${ROOT_LABELS_CSV:-lista_dense_signsplit_p256_hardinit_basin_partition}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14}"
SEED_SPLITS_SEMICOLON="${SEED_SPLITS_SEMICOLON:-0,1,2,3,4;5,6,7,8,9;10,11,12,13,14}"
SHARDS_DIR="${SHARDS_DIR:-${RESULTS_DIR}/shards}"
MERGED_DIR="${MERGED_DIR:-${RESULTS_DIR}/merged}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
AUTOMATION_DIR="${AUTOMATION_DIR:-${RESULTS_DIR}/automation}"

mkdir -p "${SHARDS_DIR}" "${MERGED_DIR}" "${LOG_DIR}" "${AUTOMATION_DIR}"

SELF_ROUTED_JOB_IDS=()
IFS=';' read -r -a SEED_SPLITS <<< "${SEED_SPLITS_SEMICOLON}"
for seed_split in "${SEED_SPLITS[@]}"; do
  split_slug="seeds_${seed_split//,/_}"
  shard_out_dir="${SHARDS_DIR}/${ROOT_LABELS_CSV}__${split_slug}"
  mkdir -p "${shard_out_dir}"
  shard_job_id=$(
    ROWS_CSVS="${ROWS_CSVS}" \
    OUT_DIR="${shard_out_dir}" \
    ROOT_LABELS_CSV="${ROOT_LABELS_CSV}" \
    SEEDS_CSV="${seed_split}" \
    SUPPORT_DEFINITIONS="topk:8" \
    DEPTH_STRATA="all,deep" \
    ROLLOUT_MODES="global_k,support_local_centered,family_local_centered,latent_kmeans_local_centered,oracle_basin_local_centered" \
    REENCODE_PERIODS="0" \
    HORIZONS="100,500,1000" \
    LABEL_MODE="auto" \
    PROGRESS_EVERY_RUNS="1" \
    FLUSH_EVERY_RUNS="1" \
      sbatch \
        --job-name="oracle_lk_${split_slug}" \
        --time="12:00:00" \
        --cpus-per-task="4" \
        --mem="24G" \
        --output="${LOG_DIR}/${ROOT_LABELS_CSV}__${split_slug}-%A.out" \
        --error="${LOG_DIR}/${ROOT_LABELS_CSV}__${split_slug}-%A.err" \
        scripts/run_transition_rich_self_routed_forecasting.sh | awk '{print $4}'
  )
  SELF_ROUTED_JOB_IDS+=("${shard_job_id}")
done

SELF_ROUTED_DEPENDENCY="$(IFS=:; echo "${SELF_ROUTED_JOB_IDS[*]}")"
MERGE_JOB_ID=$(
  SHARDS_DIR="${SHARDS_DIR}" \
  OUT_DIR="${MERGED_DIR}" \
  ROWS_CSVS="${ROWS_CSVS}" \
  ROOT_LABELS_CSV="${ROOT_LABELS_CSV}" \
  SEEDS_CSV="${SEEDS_CSV}" \
    sbatch \
      --dependency=afterany:"${SELF_ROUTED_DEPENDENCY}" \
      --time="00:30:00" \
      --mem="4G" \
      --output="${LOG_DIR}/merge-%A.out" \
      --error="${LOG_DIR}/merge-%A.err" \
      scripts/merge_transition_rich_self_routed_forecasting_shards.sh | awk '{print $4}'
)

cat > "${AUTOMATION_DIR}/queue.json" <<EOF
{
  "experiment_tag": "${EXPERIMENT_TAG}",
  "results_dir": "${RESULTS_DIR}",
  "rows_csvs": "${ROWS_CSVS}",
  "root_labels_csv": "${ROOT_LABELS_CSV}",
  "seeds_csv": "${SEEDS_CSV}",
  "shard_job_ids": "$(IFS=,; echo "${SELF_ROUTED_JOB_IDS[*]}")",
  "merge_job_id": "${MERGE_JOB_ID}"
}
EOF

echo "Queued oracle-vs-learned local Koopman comparison."
echo "Shard jobs: $(IFS=,; echo "${SELF_ROUTED_JOB_IDS[*]}")"
echo "Merge job: ${MERGE_JOB_ID}"
echo "Results dir: ${RESULTS_DIR}"
