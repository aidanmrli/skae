#!/bin/bash
#
# Requeue Kuramoto N=32 seeds 3,4 for Subagent D.
# Original jobs 8914751-8914754 failed with exit code 127 (sbatch --wrap used sh, not bash).
# Fix: use a proper bash script with sbatch.
#
set -euo pipefail

cd /home/mila/l/lia/skae

BASE_OUT="/network/scratch/l/lia/skae/kuramoto_n32_dt00625_200k_confirm_20260309"

# Common training args
COMMON_ARGS="--env kuramoto --env_dt 0.00625 --num_steps 200000 --batch_size 256 --target_size 256 --sequence_length 8 --res_coeff 1.0 --reconst_coeff 0.03 --pred_coeff 1.0 --eval_profile full --device cuda --kuramoto_num_oscillators 32"

GS_ARGS="--config generic_sparse --sparsity_coeff 0.0025"
BD_ARGS="--config lista_parity_generic_sparse --sparsity_coeff 0.006 --lista_alpha 0.15 --lista_num_loops 1 --lista_final_op relu --k_structure block_diagonal --k_block_size 16"

TRAIN_IDS=""

for SEED in 3 4; do
  for VARIANT in generic_sparse lista_blockdiag; do
    if [[ "${VARIANT}" == "generic_sparse" ]]; then
      MODEL_ARGS="${GS_ARGS}"
    else
      MODEL_ARGS="${BD_ARGS}"
    fi
    LOG_DIR="${BASE_OUT}/${VARIANT}/kuramoto/dt_0p00625/steps_200000/sp_0p0025/n_32/seed_${SEED}"

    JOB_ID=$(sbatch --parsable \
      --job-name="k32_${VARIANT:0:4}_s${SEED}" \
      --partition=long \
      --cpus-per-task=4 \
      --gres=gpu:1 \
      --mem=16G \
      --time=24:00:00 \
      -o "/network/scratch/l/lia/skae/k32-requeue-%j.out" \
      -e "/network/scratch/l/lia/skae/k32-requeue-%j.err" \
      --requeue \
      <<SCRIPT
#!/bin/bash
set -euo pipefail
cd /home/mila/l/lia/skae
module load cuda/12.6.0
source .venv/bin/activate
echo "Host: \$(hostname)"
echo "Variant: ${VARIANT}, Seed: ${SEED}"
nvidia-smi || true
uv run python tools/train.py ${COMMON_ARGS} ${MODEL_ARGS} --seed ${SEED} --log_dir "${LOG_DIR}"
SCRIPT
    )
    echo "Submitted ${VARIANT} seed ${SEED}: job ${JOB_ID}"
    TRAIN_IDS="${TRAIN_IDS}:${JOB_ID}"
  done
done

TRAIN_IDS="${TRAIN_IDS:1}"  # strip leading colon
echo "All train jobs: ${TRAIN_IDS}"

# Dependent collection
COLLECT_ID=$(ROOT_SPECS_FILE=results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/root_specs/kuramoto_n32_roots.txt \
  OUT_DIR=results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/collect \
  PAPER_SUMMARY=1 \
  sbatch --parsable --dependency=afterany:${TRAIN_IDS} scripts/collect_paper_benchmark.sh)
echo "Collect job: ${COLLECT_ID}"

# Dependent comparison
COMPARE_ID=$(ROWS_CSV=results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/collect/forecasting_rows.csv \
  OUT_DIR=results/paper_parallel_20260309_d_kuramoto_n32_more_seeds/compare \
  CANDIDATE_ROOTS_CSV=lista_blockdiag \
  ANCHOR_ROOT=generic_sparse \
  HORIZON=1000 \
  sbatch --parsable --dependency=afterany:${COLLECT_ID} scripts/compare_paper_benchmark.sh)
echo "Compare job: ${COMPARE_ID}"

echo "Done. Chain: train(${TRAIN_IDS}) -> collect(${COLLECT_ID}) -> compare(${COMPARE_ID})"
