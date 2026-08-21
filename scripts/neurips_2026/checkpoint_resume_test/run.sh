#!/usr/bin/env bash
# Submit with sbatch.  This allocation deliberately exercises a catchable
# TERM, a same-job requeue, and resume from the persistent checkpoint store.

#SBATCH --job-name=skae-checkpoint-resume
#SBATCH --partition=long
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=checkpoint-resume-%j.out
#SBATCH --error=checkpoint-resume-%j.err

set -euo pipefail
: "${SCRATCH:?SCRATCH must point to persistent active checkpoint storage}"

JOB_KEY="${SLURM_JOB_ID:-manual}"
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/skae-checkpoint-resume/${JOB_KEY}}"
PERMANENT_DIR="${PERMANENT_CHECKPOINT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/runs/checkpoint-resume/${JOB_KEY}}"
TOTAL_STEPS="${CHECKPOINT_TEST_STEPS:-10000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_TEST_INTERVAL:-32}"
FORCE_TERM_AFTER="${CHECKPOINT_FORCE_TERM_AFTER:-2}"
FORCE_MARKER="$CHECKPOINT_DIR/forced-term-once"

mkdir -p "$CHECKPOINT_DIR" "$PERMANENT_DIR"

common_args=(
  --config generic
  --env duffing
  --num_steps "$TOTAL_STEPS"
  --batch_size 4
  --target_size 8
  --sequence_length 1
  --eval_every 1000000
  --skip_eval
  --device cpu
  --checkpoint_dir "$CHECKPOINT_DIR"
  --checkpoint_interval "$CHECKPOINT_INTERVAL"
  --checkpoint_retention 3
  --permanent_checkpoint_dir "$PERMANENT_DIR"
  --resume_if_available
  --save_metrics_history
)

TRAIN_PID=""

requeue_after_term() {
  trap - TERM INT
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    # Prefer the srun process group so the signal reaches the actual task;
    # retain the direct fallback for launchers that do not create a group.
    kill -TERM -- "-$TRAIN_PID" 2>/dev/null || kill -TERM "$TRAIN_PID" 2>/dev/null || true
    set +e
    wait "$TRAIN_PID"
    train_rc=$?
    set -e
    if [[ "$train_rc" -ne 75 ]]; then
      echo "TERM worker returned $train_rc; expected checkpoint exit 75" >&2
      exit 1
    fi
  fi
  test -s "$CHECKPOINT_DIR/latest.json"
  test -s "$CHECKPOINT_DIR/checkpoint_receipt.json"
  if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "cannot requeue outside SLURM" >&2
    exit 1
  fi
  scontrol requeue "$SLURM_JOB_ID"
  echo "requeued job $SLURM_JOB_ID after checkpointed TERM"
  exit 0
}

trap 'requeue_after_term' TERM
trap 'requeue_after_term' INT

start_worker() {
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
    uv run skae-train "${common_args[@]}" &
  TRAIN_PID=$!
}

echo "checkpoint_dir=$CHECKPOINT_DIR"
echo "permanent_checkpoint_dir=$PERMANENT_DIR"
echo "SLURM_RESTART_COUNT=$RESTART_COUNT"

if [[ "$RESTART_COUNT" -eq 0 && ! -e "$FORCE_MARKER" ]]; then
  start_worker
  for _ in $(seq 1 300); do
    [[ -s "$CHECKPOINT_DIR/latest.json" ]] && break
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
      echo "worker exited before first valid checkpoint" >&2
      wait "$TRAIN_PID" || true
      exit 1
    fi
    sleep 0.1
  done
  test -s "$CHECKPOINT_DIR/latest.json"
  sleep "$FORCE_TERM_AFTER"
  test -d "$CHECKPOINT_DIR"
  mkdir "$FORCE_MARKER"
  echo "forcing the one test TERM through the batch trap"
  kill -TERM "$$"
  echo "unreachable after TERM trap" >&2
  exit 1
fi

# The requeued allocation must advertise a positive restart count and resume
# from the persistent checkpoint without another forced interruption.
if [[ "$RESTART_COUNT" -le 0 ]]; then
  echo "restart phase requires SLURM_RESTART_COUNT>0" >&2
  exit 1
fi
start_worker
set +e
wait "$TRAIN_PID"
TRAIN_RC=$?
set -e
if [[ "$TRAIN_RC" -ne 0 ]]; then
  echo "resumed worker failed with exit code $TRAIN_RC" >&2
  exit "$TRAIN_RC"
fi

test -s "$CHECKPOINT_DIR/latest.json"
test -s "$CHECKPOINT_DIR/last.pt"
test -s "$PERMANENT_DIR/latest.pt"
test -s "$PERMANENT_DIR/latest.manifest.json"
echo "checkpoint requeue/resume test passed"
