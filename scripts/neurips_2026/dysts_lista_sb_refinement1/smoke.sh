#!/usr/bin/env bash
#SBATCH --job-name=smoke-dysts-sb1
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

set -euo pipefail
PROJECT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
ROOT_DIR="${PROJECT_DIR}"
source "${PROJECT_DIR}/scripts/common/cluster_env.sh"
OUTPUT_ROOT="${SKAE_SCRATCH_ROOT}/dysts_lista_sb_refinement1_smoke_v3_20260722"
TASK_TSV="${OUTPUT_ROOT}/tasks.tsv"
MANIFEST="${OUTPUT_ROOT}/manifest.json"
TELEMETRY="${OUTPUT_ROOT}/gpu_telemetry.csv"
mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_DIR}"
source .venv/bin/activate

uv run skae-paper tasks dysts \
  --phase_label dysts_lista_sb_refinement1_smoke_v3 \
  --output_tsv "${TASK_TSV}" --output_manifest_json "${MANIFEST}" \
  --systems_csv dysts:Chua --model_variants_csv lista_sb \
  --seeds_csv 0,1,2,3,4,5,6,7,8,9,10,11 \
  --num_steps 2000 --lista_sb_num_loops 1
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit --format=csv --loop=2 > "${TELEMETRY}" &
MONITOR_PID=$!
trap 'kill "${MONITOR_PID}" 2>/dev/null || true' EXIT

FAILED=0
CHILD_PIDS=()
for TASK_ID in $(seq 0 11); do
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  GPU_TELEMETRY=0 SLURM_ARRAY_TASK_ID="${TASK_ID}" TASK_TSV="${TASK_TSV}" \
  BASE_OUT="${OUTPUT_ROOT}/runs" ARRAY_OFFSET=0 \
  bash scripts/common/run_benchmark_task.sh &
  CHILD_PIDS+=("$!")
done
for PID in "${CHILD_PIDS[@]}"; do
  wait "${PID}" || FAILED=1
done

kill "${MONITOR_PID}" 2>/dev/null || true
wait "${MONITOR_PID}" 2>/dev/null || true
trap - EXIT
awk -F, 'NR > 1 {gsub(/[^0-9.]/, "", $3); u=$3+0; sum+=u; n++; if(u>0){active+=u; an++}; if(u>peak)peak=u} END {printf "samples\t%d\nmean_all_gpu_utilization_percent\t%.3f\nmean_active_gpu_utilization_percent\t%.3f\npeak_gpu_utilization_percent\t%.1f\n", n, n?sum/n:0, an?active/an:0, peak}' "${TELEMETRY}" > "${OUTPUT_ROOT}/gpu_utilization_audit.tsv"
cat "${OUTPUT_ROOT}/gpu_utilization_audit.tsv"
exit "${FAILED}"
