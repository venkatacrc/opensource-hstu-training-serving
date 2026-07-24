#!/usr/bin/env bash
# Phase 5: the main event -- full HSTU-8B (or 10B) training run.
#
# Launched as a DETACHED container so it survives SSH/session drops. This
# script then polls it, tails progress into $STATE_DIR, and enforces the
# wall-clock budget (TRAIN_BUDGET_HOURS) by stopping the container once the
# deadline passes -- training relies on periodic checkpointing
# (TrainerArgs.ckpt_save_interval) so a time-boxed stop always leaves a usable
# checkpoint for scripts/06_export_checkpoint.sh.
#
# Resumable: if a checkpoint already exists for this (model size, dataset)
# combination, training automatically resumes from it (ckpt_load_dir =
# ckpt_save_dir). Re-running this script after an interruption continues
# rather than restarting.
#
# Usage:
#   ./scripts/05_train.sh                  # launch + wait (up to TRAIN_BUDGET_HOURS), trains PRIMARY_DATASET
#   ./scripts/05_train.sh --detach-only     # launch and return immediately
#   ./scripts/05_train.sh --status          # just print current progress
#
# Env overrides (set before invoking to train the secondary/serving-benchmark
# checkpoint instead, e.g.):
#   TRAIN_DATASET=kuairand-1k TRAIN_HOURS=$SECONDARY_TRAIN_BUDGET_HOURS ./scripts/05_train.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/lib/common.sh"

: "${TRAIN_DATASET:=$PRIMARY_DATASET}"
: "${TRAIN_HOURS:=$TRAIN_BUDGET_HOURS}"
TRAIN_BUDGET_HOURS="$TRAIN_HOURS"   # local override used below

DATASET_SLUG="${TRAIN_DATASET//-/}"
PHASE="05_train_${DATASET_SLUG}"
MODE="wait"
[[ "${1:-}" == "--detach-only" ]] && MODE="detach-only"
[[ "${1:-}" == "--status" ]] && MODE="status"

RUN_NAME="hstu_${MODEL_SIZE}_${DATASET_SLUG}"
TRAIN_CONTAINER="${CONTAINER_NAME}-train-${RUN_NAME}"
RUN_CKPT_DIR="/workspace/checkpoints/${RUN_NAME}"
HOST_TRAIN_LOG="$STATE_DIR/${PHASE}_${RUN_NAME}.log"
HOST_GPU_LOG="$STATE_DIR/${PHASE}_${RUN_NAME}_gpu_mem.csv"
PROGRESS_MARKER="$STATE_DIR/${PHASE}_${RUN_NAME}.progress"

print_status() {
  if docker ps --format '{{.Names}}' | grep -qx "$TRAIN_CONTAINER"; then
    log "Training container '$TRAIN_CONTAINER' is RUNNING."
  else
    log "Training container '$TRAIN_CONTAINER' is NOT running."
  fi
  log "Last 20 log lines ($HOST_TRAIN_LOG):"
  tail -n 20 "$HOST_TRAIN_LOG" 2>/dev/null || echo "  (no log yet)"
  log "Latest MFU/throughput lines:"
  grep -iE "MFU|TFLOPS" "$HOST_TRAIN_LOG" 2>/dev/null | tail -n 5 || true
}

if [[ "$MODE" == "status" ]]; then
  print_status
  exit 0
fi

log "=== Phase 5: Train $RUN_NAME (dataset=$TRAIN_DATASET, budget=${TRAIN_BUDGET_HOURS}h) ==="
docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 || die "Image '$IMAGE_NAME' not found -- run scripts/01_build_env.sh first."

if phase_is_done "$PHASE"; then
  ok "Phase '$PHASE' already marked done. Re-run with 'rm state/${PHASE}.done' to force a fresh/continued run, or use --status to just check progress."
  exit 0
fi

BASE_GIN="$REPO_ROOT/configs/hstu_${MODEL_SIZE}_ranking_${DATASET_SLUG}.gin"
TUNED_GIN="$REPO_ROOT/configs/hstu_${MODEL_SIZE}_ranking_${DATASET_SLUG}.tuned.gin"
[[ -f "$BASE_GIN" ]] || die "Base config not found: $BASE_GIN"
[[ -f "$TUNED_GIN" ]] || die "Tuned config not found: $TUNED_GIN -- run scripts/04_size_model.sh first."

RUN_GIN="$REPO_ROOT/configs/.run_${RUN_NAME}.gin"

# Resume automatically if a checkpoint already exists on disk for this run.
RESUME_LINE=""
if [[ -d "$CKPT_DIR/$RUN_NAME" ]] && [[ -n "$(ls -A "$CKPT_DIR/$RUN_NAME" 2>/dev/null)" ]]; then
  log "Existing checkpoint found at $CKPT_DIR/$RUN_NAME -- will resume."
  RESUME_LINE="TrainerArgs.ckpt_load_dir = '${RUN_CKPT_DIR}'"
else
  log "No existing checkpoint -- starting fresh."
fi

render_effective_gin "$RUN_GIN" "$BASE_GIN" "$TUNED_GIN" -- \
  "TrainerArgs.ckpt_save_dir = '${RUN_CKPT_DIR}'" \
  "${RESUME_LINE:-# no resume}" \
  "DatasetArgs.dataset_path = '/workspace/data'"

log "Effective training config written to $RUN_GIN"

if docker ps -a --format '{{.Names}}' | grep -qx "$TRAIN_CONTAINER"; then
  if docker ps --format '{{.Names}}' | grep -qx "$TRAIN_CONTAINER"; then
    ok "Training container already running, attaching to monitor it."
  else
    log "Removing stale stopped container '$TRAIN_CONTAINER'..."
    docker rm "$TRAIN_CONTAINER" >/dev/null
    LAUNCH=1
  fi
else
  LAUNCH=1
fi

if [[ "${LAUNCH:-0}" -eq 1 ]]; then
  log "Launching detached training container '$TRAIN_CONTAINER'..."
  docker run -d --name "$TRAIN_CONTAINER" --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=64g \
    -v "$RECSYS_DIR:/workspace/recsys-examples" \
    -v "$DATA_DIR:/workspace/data" \
    -v "$CKPT_DIR:/workspace/checkpoints" \
    -v "$REPO_ROOT/configs:/workspace/configs" \
    -v "$REPO_ROOT/results:/workspace/results" \
    -w /workspace/recsys-examples/examples/hstu \
    "$IMAGE_NAME" bash -lc "
      export PYTHONPATH=\$PYTHONPATH:\$(realpath /workspace/recsys-examples/examples)
      mkdir -p ${RUN_CKPT_DIR}
      torchrun --nproc_per_node $NUM_GPUS --master_addr localhost --master_port $MASTER_PORT \
        ./training/pretrain_gr_ranking.py \
        --gin-config-file /workspace/configs/$(basename "$RUN_GIN") \
        2>&1 | tee /workspace/results/${PHASE}_${RUN_NAME}_container.log
    " >/dev/null
  ok "Launched. Container: $TRAIN_CONTAINER"
fi

# Stream container logs to a host-side file in the background for monitoring.
( docker logs -f "$TRAIN_CONTAINER" >> "$HOST_TRAIN_LOG" 2>&1 ) &
LOG_TAIL_PID=$!
( nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv -l 30 >> "$HOST_GPU_LOG" 2>/dev/null ) &
GPU_SAMPLER_PID=$!
trap 'kill "$LOG_TAIL_PID" "$GPU_SAMPLER_PID" 2>/dev/null || true' EXIT

if [[ "$MODE" == "detach-only" ]]; then
  ok "Training launched in the background (detach-only mode). Monitor with:"
  log "  docker logs -f $TRAIN_CONTAINER"
  log "  ./scripts/05_train.sh --status"
  exit 0
fi

DEADLINE=$(( $(date +%s) + TRAIN_BUDGET_HOURS * 3600 ))
log "Waiting for training to finish or hit the ${TRAIN_BUDGET_HOURS}h budget (deadline: $(date -d "@$DEADLINE" 2>/dev/null || date -r "$DEADLINE"))..."
log "(Safe to Ctrl-C this script -- the container keeps training in the background; re-run './scripts/05_train.sh --status' any time, or this script again to resume waiting.)"

while docker ps --format '{{.Names}}' | grep -qx "$TRAIN_CONTAINER"; do
  NOW=$(date +%s)
  if [[ "$NOW" -ge "$DEADLINE" ]]; then
    log "Wall-clock training budget (${TRAIN_BUDGET_HOURS}h) reached. Stopping container gracefully (allowing checkpoint flush)..."
    docker stop -t 180 "$TRAIN_CONTAINER" >/dev/null || true
    break
  fi
  sleep 60
  LAST_LINE=$(tail -n 1 "$HOST_TRAIN_LOG" 2>/dev/null)
  REMAINING=$(( (DEADLINE - NOW) / 60 ))
  log "training running, ${REMAINING}min budget remaining. last log line: ${LAST_LINE:0:160}"
done

kill "$LOG_TAIL_PID" "$GPU_SAMPLER_PID" 2>/dev/null || true
trap - EXIT

EXIT_CODE=$(docker inspect "$TRAIN_CONTAINER" --format='{{.State.ExitCode}}' 2>/dev/null || echo "unknown")
log "Training container final state: exit code $EXIT_CODE (0 and 137/143 [stopped by us] are both fine here)."

if [[ ! -d "$CKPT_DIR/$RUN_NAME" ]] || [[ -z "$(ls -A "$CKPT_DIR/$RUN_NAME" 2>/dev/null)" ]]; then
  die "No checkpoint was produced in $CKPT_DIR/$RUN_NAME. Check $HOST_TRAIN_LOG for errors before proceeding."
fi

log "Checkpoint(s) available:"
ls -la "$CKPT_DIR/$RUN_NAME" | tee "$STATE_DIR/${PHASE}_${RUN_NAME}_checkpoints.txt"

log "MFU/throughput summary (last 20 matching lines):"
grep -iE "MFU|TFLOPS" "$HOST_TRAIN_LOG" | tail -n 20 | tee "$STATE_DIR/${PHASE}_${RUN_NAME}_mfu_tail.txt" || warn "No MFU/TFLOPS lines found in log -- check that TrainerArgs.profile=True and the log format matches what scripts/08_generate_report.py expects."

phase_done_mark "$PHASE"
ok "Training phase complete for $RUN_NAME. Proceed to scripts/06_export_checkpoint.sh."
