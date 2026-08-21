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
#SBATCH --output=/network/scratch/l/lia/skae/slurm_logs/%x-%j.out
#SBATCH --error=/network/scratch/l/lia/skae/slurm_logs/%x-%j.err

set -euo pipefail

ROOT_DIR="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
cd "${ROOT_DIR}"
if [[ -f scripts/common/cluster_env.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/common/cluster_env.sh
fi
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# Reuse the locked cluster environment; PYTHONPATH keeps this candidate tree
# authoritative without creating or synchronizing a worktree-local .venv.
export UV_PROJECT="${SKAE_UV_PROJECT:-/home/mila/l/lia/skae}"
export UV_NO_SYNC=1

: "${SCRATCH:?SCRATCH must point to persistent active checkpoint storage}"

JOB_KEY="${SLURM_JOB_ID:-manual}"
RESTART_COUNT="${SLURM_RESTART_COUNT:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH}/skae-checkpoint-resume/${JOB_KEY}}"
PERMANENT_DIR="${PERMANENT_CHECKPOINT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/runs/checkpoint-resume/${JOB_KEY}}"
TOTAL_STEPS="${CHECKPOINT_TEST_STEPS:-10000}"
CHECKPOINT_INTERVAL="${CHECKPOINT_TEST_INTERVAL:-32}"
FORCE_TERM_AFTER="${CHECKPOINT_FORCE_TERM_AFTER:-2}"
FORCE_MARKER="$CHECKPOINT_DIR/forced-term-once"
TASK_PID_FILE="$CHECKPOINT_DIR/task-${RESTART_COUNT}.pid"

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
TASK_PID=""

wait_for_task_pid() {
  local task_pid
  for _ in $(seq 1 300); do
    if [[ -s "$TASK_PID_FILE" ]]; then
      task_pid="$(<"$TASK_PID_FILE")"
      if [[ ! "$task_pid" =~ ^[0-9]+$ ]]; then
        echo "invalid task PID in $TASK_PID_FILE" >&2
        return 1
      fi
      if ! kill -0 "$task_pid" 2>/dev/null; then
        echo "task PID $task_pid exited before TERM forwarding" >&2
        return 1
      fi
      TASK_PID="$task_pid"
      return 0
    fi
    if [[ -n "$TRAIN_PID" ]] && ! kill -0 "$TRAIN_PID" 2>/dev/null; then
      return 1
    fi
    sleep 0.1
  done
  echo "timed out waiting for task PID file $TASK_PID_FILE" >&2
  return 1
}

requeue_after_term() {
  trap - TERM INT
  if [[ -n "$TRAIN_PID" ]]; then
    # Signal the task PID recorded by the wrapper, never the srun client or
    # its process group.  Signalling srun itself can make Slurm kill the task
    # with SIGKILL before the runner writes its final checkpoint (rc 137).
    if wait_for_task_pid; then
      echo "forwarding TERM to checkpoint task PID $TASK_PID"
      kill -TERM "$TASK_PID" 2>/dev/null || true
    fi
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
  # The wrapper PID is preserved by exec and is therefore also the PID of the
  # training task.  Replace the per-attempt PID file before launching so a
  # stale PID can never be mistaken for this task.
  rm -f -- "$TASK_PID_FILE"
  TASK_PID=""
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" \
    bash -c '
      set -euo pipefail
      pid_file="$1"
      shift
      temporary="${pid_file}.tmp.$$"
      printf "%s\\n" "$$" > "$temporary"
      mv -f "$temporary" "$pid_file"
      exec "$@"
    ' checkpoint-task "$TASK_PID_FILE" uv run skae-train "${common_args[@]}" &
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
