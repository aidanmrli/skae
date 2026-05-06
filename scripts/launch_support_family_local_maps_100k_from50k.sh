#!/bin/bash
#
# Launch the 100k continuation batches after the corresponding 50k aggregate
# has completed. This script submits the seed 0--4 and 5--9 worker batches and
# then submits a combined 10-seed aggregation for the benchmark.
#
# Required env vars:
#   BENCHMARK=multibasin|dysts
#
# Optional env vars:
#   WALLTIME=03:00:00
#   PARTITION=long
#   CPUS=1
#   MEM=8G
#   DEVICE=cuda
#   GRES=gpu:1
#   TRAIN_STEPS=100000
#   SUPPORT_DEFINITION=topk:8
#   FAMILY_JACCARD_THRESHOLD=0.4
#   STAGE2_MAP_MODE=family_local_centered # or global_dense_calibrated
#   ROOT_BASE=results/routed_stage2_local_maps_20260506
#
#SBATCH --job-name=sf_lm_100k_launch
#SBATCH --ntasks=1
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="${SLURM_SUBMIT_DIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${ROOT_DIR}"

BENCHMARK="${BENCHMARK:?BENCHMARK is required}"
WALLTIME="${WALLTIME:-03:00:00}"
PARTITION="${PARTITION:-long}"
CPUS="${CPUS:-1}"
MEM="${MEM:-8G}"
DEVICE="${DEVICE:-cuda}"
GRES="${GRES:-gpu:1}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
SUPPORT_DEFINITION="${SUPPORT_DEFINITION:-topk:8}"
FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD:-0.4}"
STAGE2_MAP_MODE="${STAGE2_MAP_MODE:-family_local_centered}"
ROOT_BASE="${ROOT_BASE:-results/routed_stage2_local_maps_20260506}"

COMMON_ENV=(
  REENCODE_PERIODS=5
  ROUTE_FREEZE_MODES=reroute_each_step
  TRAIN_STEPS="${TRAIN_STEPS}"
  SUPPORT_DEFINITION="${SUPPORT_DEFINITION}"
  FAMILY_JACCARD_THRESHOLD="${FAMILY_JACCARD_THRESHOLD}"
  STAGE2_MAP_MODE="${STAGE2_MAP_MODE}"
  WALLTIME="${WALLTIME}"
  PARTITION="${PARTITION}"
  CPUS="${CPUS}"
  MEM="${MEM}"
  DEVICE="${DEVICE}"
  GRES="${GRES}"
  MERGE_PARTITION=long-cpu
  AGG_PARTITION=long-cpu
  SKIP_COMPLETED=1
)

case "${BENCHMARK}" in
  multibasin)
    ROWS_CSVS="results/transition_rich_lista_dense_p256_hardinit_table123_20260430/collect_pass0/forecasting_rows.csv"
    ROOT_LABELS_CSV="lista_dense_signsplit_p256_hardinit_basin_partition"
    SYSTEMS_CSV="claude:arrested_spiral,claude:cal_asymmetric_3,claude:cal_hexagon_6,claude:cal_high_cross_3,claude:cal_octagon_8,claude:cal_pentagon_5,claude:cal_square_4,claude:duffing_triple_well,claude:snic_multi,claude:transition_routes_4,claude:var_depth_gradient_4,claude:var_diamond_4,claude:var_l_shape_5,gated_local_linear,gated_transfer_linear"
    HORIZONS="100,500,1000"
    LABEL_MODE="auto"
    AGG_DATASETS="multibasin"
    OUT_0_4="${ROOT_BASE}/best_lista_multibasin_j040_100k_from50k_seed0_4"
    OUT_5_9="${ROOT_BASE}/best_lista_multibasin_j040_100k_from50k_seed5_9"
    RESUME_0_4="${ROOT_BASE}/best_lista_multibasin_j040_50k_seed0_4"
    RESUME_5_9="${ROOT_BASE}/best_lista_multibasin_j040_50k_seed5_9"
    COMBINED_OUT="${ROOT_BASE}/combined_best_lista_multibasin_j040_100k_from50k_seed0_9"
    COMBINE_JOB_NAME="sf_lm_100k_mb10"
    ;;
  dysts)
    ROWS_CSVS="results/dysts_dt30_basinblock_p256_seq10_100k_20260430/long_horizon_eval/collect/forecasting_rows.csv"
    ROOT_LABELS_CSV="lista"
    SYSTEMS_CSV="dysts:Chua,dysts:Dadras,dysts:DequanLi,dysts:Hadley,dysts:LuChenCheng,dysts:QiChen,dysts:Sakarya,dysts:SanUmSrisuchinwong,dysts:ShimizuMorioka,dysts:WangSun"
    HORIZONS="100,500,1000,1500,2000,3000,4000,5000"
    LABEL_MODE="none"
    AGG_DATASETS="dysts"
    OUT_0_4="${ROOT_BASE}/best_lista_dysts_j040_100k_from50k_labelnone_seed0_4"
    OUT_5_9="${ROOT_BASE}/best_lista_dysts_j040_100k_from50k_labelnone_seed5_9"
    RESUME_0_4="${ROOT_BASE}/best_lista_dysts_j040_50k_labelnone_seed0_4"
    RESUME_5_9="${ROOT_BASE}/best_lista_dysts_j040_50k_labelnone_seed5_9"
    COMBINED_OUT="${ROOT_BASE}/combined_best_lista_dysts_j040_100k_from50k_labelnone_seed0_9"
    COMBINE_JOB_NAME="sf_lm_100k_dy10"
    ;;
  *)
    echo "Unknown BENCHMARK=${BENCHMARK}; expected multibasin or dysts" >&2
    exit 2
    ;;
esac

case "${STAGE2_MAP_MODE}" in
  family_local_centered)
    ;;
  global_dense_calibrated)
    case "${BENCHMARK}" in
      multibasin)
        OUT_0_4="${ROOT_BASE}/globalK_calibrated_multibasin_j040_100k_from50k_seed0_4"
        OUT_5_9="${ROOT_BASE}/globalK_calibrated_multibasin_j040_100k_from50k_seed5_9"
        RESUME_0_4="${ROOT_BASE}/globalK_calibrated_multibasin_j040_50k_seed0_4"
        RESUME_5_9="${ROOT_BASE}/globalK_calibrated_multibasin_j040_50k_seed5_9"
        COMBINED_OUT="${ROOT_BASE}/combined_globalK_calibrated_multibasin_j040_100k_from50k_seed0_9"
        COMBINE_JOB_NAME="sf_gk_100k_mb10"
        ;;
      dysts)
        OUT_0_4="${ROOT_BASE}/globalK_calibrated_dysts_j040_100k_from50k_labelnone_seed0_4"
        OUT_5_9="${ROOT_BASE}/globalK_calibrated_dysts_j040_100k_from50k_labelnone_seed5_9"
        RESUME_0_4="${ROOT_BASE}/globalK_calibrated_dysts_j040_50k_labelnone_seed0_4"
        RESUME_5_9="${ROOT_BASE}/globalK_calibrated_dysts_j040_50k_labelnone_seed5_9"
        COMBINED_OUT="${ROOT_BASE}/combined_globalK_calibrated_dysts_j040_100k_from50k_labelnone_seed0_9"
        COMBINE_JOB_NAME="sf_gk_100k_dy10"
        ;;
    esac
    ;;
  *)
    echo "Unknown STAGE2_MAP_MODE=${STAGE2_MAP_MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${ROOT_BASE}/logs" "${COMBINED_OUT}"
LOG_ROOT="${ROOT_DIR}/${ROOT_BASE}/logs"

submit_batch() {
  local out_dir="$1"
  local resume_dir="$2"
  local seeds_csv="$3"
  local launch_log="${out_dir}/launcher_submission.txt"

  mkdir -p "${out_dir}"
  echo "Submitting ${BENCHMARK} 100k continuation seeds ${seeds_csv} -> ${out_dir}" >&2
  output=$(
    env \
      ROWS_CSVS="${ROWS_CSVS}" \
      OUT_DIR="${out_dir}" \
      ROOT_LABELS_CSV="${ROOT_LABELS_CSV}" \
      SYSTEMS_CSV="${SYSTEMS_CSV}" \
      SEEDS_CSV="${seeds_csv}" \
      HORIZONS="${HORIZONS}" \
      LABEL_MODE="${LABEL_MODE}" \
      SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
      RESUME_FROM_OUTPUT_DIRS="${resume_dir}" \
      AGG_DATASETS="${AGG_DATASETS}" \
      "${COMMON_ENV[@]}" \
      bash scripts/queue_support_family_local_maps_stage2_poc.sh
  )
  printf '%s\n' "${output}" | tee "${launch_log}" >&2
  merge_job_id=$(printf '%s\n' "${output}" | awk -F= '/^MERGE_JOB_ID=/{gsub(/\047/, "", $2); print $2}')
  agg_job_id=$(printf '%s\n' "${output}" | awk -F= '/^AGG_JOB_ID=/{gsub(/\047/, "", $2); print $2}')
  if [[ -z "${merge_job_id}" ]]; then
    echo "Could not parse MERGE_JOB_ID from ${launch_log}" >&2
    exit 3
  fi
  printf '%s,%s\n' "${merge_job_id}" "${agg_job_id}"
}

batch_0_4=$(submit_batch "${OUT_0_4}" "${RESUME_0_4}" "0,1,2,3,4")
batch_5_9=$(submit_batch "${OUT_5_9}" "${RESUME_5_9}" "5,6,7,8,9")
merge_0_4="${batch_0_4%%,*}"
merge_5_9="${batch_5_9%%,*}"
agg_0_4="${batch_0_4#*,}"
agg_5_9="${batch_5_9#*,}"

combined_job_id=$(
  INPUT_DIRS_CSV="${OUT_0_4},${OUT_5_9}" \
  OUT_DIR="${COMBINED_OUT}" \
  DATASETS="${AGG_DATASETS}" \
  SUPPORT_DEFINITION="${SUPPORT_DEFINITION}" \
    sbatch \
      --dependency="afterok:${merge_0_4}:${merge_5_9}" \
      --job-name="${COMBINE_JOB_NAME}" \
      --partition=long-cpu \
      --time=00:30:00 \
      --mem=8G \
      --output="${LOG_ROOT}/${COMBINE_JOB_NAME}-%A.out" \
      --error="${LOG_ROOT}/${COMBINE_JOB_NAME}-%A.err" \
      scripts/combine_and_analyze_stage2_batches.sh | awk '{print $4}'
)

manifest="${COMBINED_OUT}/launcher_manifest.json"
cat > "${manifest}" <<EOF
{
  "benchmark": "${BENCHMARK}",
  "stage2_map_mode": "${STAGE2_MAP_MODE}",
  "support_definition": "${SUPPORT_DEFINITION}",
  "train_steps": "${TRAIN_STEPS}",
  "family_jaccard_threshold": "${FAMILY_JACCARD_THRESHOLD}",
  "out_0_4": "${OUT_0_4}",
  "out_5_9": "${OUT_5_9}",
  "resume_0_4": "${RESUME_0_4}",
  "resume_5_9": "${RESUME_5_9}",
  "merge_0_4": "${merge_0_4}",
  "merge_5_9": "${merge_5_9}",
  "agg_0_4": "${agg_0_4}",
  "agg_5_9": "${agg_5_9}",
  "combined_out": "${COMBINED_OUT}",
  "combined_job_id": "${combined_job_id}"
}
EOF

printf 'BENCHMARK=%q\n' "${BENCHMARK}"
printf 'MERGE_0_4=%q\n' "${merge_0_4}"
printf 'MERGE_5_9=%q\n' "${merge_5_9}"
printf 'AGG_0_4=%q\n' "${agg_0_4}"
printf 'AGG_5_9=%q\n' "${agg_5_9}"
printf 'COMBINED_JOB_ID=%q\n' "${combined_job_id}"
printf 'LAUNCHER_MANIFEST=%q\n' "${manifest}"
