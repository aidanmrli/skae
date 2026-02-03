#!/bin/bash

#SBATCH --job-name=kstruct_eval
#SBATCH --ntasks=1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH -o /network/scratch/l/lia/skae/kstruct_eval-%j.out
#SBATCH --requeue

# ============================================================================
# K-Structure Evaluation Sweep (inference only)
# ============================================================================
# Iterates over all (target_size, k_structure) checkpoints and runs:
#   1. evaluate_checkpoints.py  -- long-horizon prediction MSE
#   2. analyze_k_eigenvalues.py -- per-block eigenvalue analysis + basin corr
#
# Single GPU job, sequential loop (~5 min per checkpoint, ~2h total for 25).
# ============================================================================

module load cuda/12.6.0
source .venv/bin/activate

BASE_DIR="${BASE_DIR:-/network/scratch/l/lia/skae/lyapunov_k_structure_sweep}"

TARGET_SIZES=(64 128 256 512 1024)
K_STRUCTURES=(dense diagonal block_diagonal arrowhead arrowhead_no_excl)

SYSTEM="lyapunov"
DEVICE="cuda"
NUM_TRAJ=100
SEED=42

TOTAL=$(( ${#TARGET_SIZES[@]} * ${#K_STRUCTURES[@]} ))
COUNT=0
SUCCEEDED=0
FAILED=0

echo "============================================="
echo "K-Structure Evaluation Sweep"
echo "Base directory: $BASE_DIR"
echo "Target sizes: ${TARGET_SIZES[*]}"
echo "K structures: ${K_STRUCTURES[*]}"
echo "Total experiments: $TOTAL"
echo "============================================="

for TS in "${TARGET_SIZES[@]}"; do
    for KS in "${K_STRUCTURES[@]}"; do
        COUNT=$((COUNT + 1))
        EXP_NAME="dim8_nb13_ts${TS}_${KS}"
        EXP_DIR="${BASE_DIR}/${EXP_NAME}"

        echo ""
        echo "---------------------------------------------"
        echo "[$COUNT/$TOTAL] $EXP_NAME"
        echo "---------------------------------------------"

        if [ ! -d "$EXP_DIR" ]; then
            echo "  SKIP: experiment directory not found"
            FAILED=$((FAILED + 1))
            continue
        fi

        # Find latest timestamp directory
        RUN_DIR=$(ls -dt "${EXP_DIR}"/[0-9]* 2>/dev/null | head -1)
        if [ -z "$RUN_DIR" ]; then
            echo "  SKIP: no timestamp directory found"
            FAILED=$((FAILED + 1))
            continue
        fi

        CKPT="${RUN_DIR}/checkpoint.pt"
        if [ ! -f "$CKPT" ]; then
            echo "  SKIP: checkpoint.pt not found in $RUN_DIR"
            FAILED=$((FAILED + 1))
            continue
        fi

        echo "  Run dir: $RUN_DIR"
        echo "  Checkpoint: $CKPT"

        # --- 1. Long-horizon prediction eval ---
        echo "  [1/2] Running evaluate_checkpoints.py ..."
        uv run python tools/evaluate_checkpoints.py \
            --run_dir "$RUN_DIR" \
            --system "$SYSTEM" \
            --checkpoints checkpoint.pt \
            --device "$DEVICE"

        EVAL_EXIT=$?
        if [ $EVAL_EXIT -ne 0 ]; then
            echo "  WARNING: evaluate_checkpoints.py exited with code $EVAL_EXIT"
        fi

        # --- 2. Eigenvalue analysis + basin correlation ---
        echo "  [2/2] Running analyze_k_eigenvalues.py ..."
        uv run python tools/analyze_k_eigenvalues.py \
            --checkpoint "$CKPT" \
            --system "$SYSTEM" \
            --correlate_basins \
            --num_trajectories "$NUM_TRAJ" \
            --output_dir "${RUN_DIR}/eigenvalue_analysis" \
            --device "$DEVICE" \
            --seed "$SEED"

        EIGEN_EXIT=$?
        if [ $EIGEN_EXIT -ne 0 ]; then
            echo "  WARNING: analyze_k_eigenvalues.py exited with code $EIGEN_EXIT"
        fi

        if [ $EVAL_EXIT -eq 0 ] && [ $EIGEN_EXIT -eq 0 ]; then
            SUCCEEDED=$((SUCCEEDED + 1))
        else
            FAILED=$((FAILED + 1))
        fi

        echo "  Done with $EXP_NAME"
    done
done

echo ""
echo "============================================="
echo "Sweep evaluation complete"
echo "  Succeeded: $SUCCEEDED / $TOTAL"
echo "  Failed/skipped: $FAILED / $TOTAL"
echo "============================================="

# --- 3. Collect all results ---
echo ""
echo "Collecting results ..."
uv run python tools/collect_k_structure_eval.py \
    --base_dir "$BASE_DIR" \
    --output "${BASE_DIR}/k_structure_comparison.json"

echo "Done."
