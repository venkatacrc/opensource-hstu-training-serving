#!/usr/bin/env bash
# Phase 4: before committing to a multi-hour training run, (a) project
# per-GPU memory for a range of batch sizes using the upstream memory
# estimator, pick the largest batch size that fits comfortably, write a
# `*.tuned.gin` override, then (b) run a short real smoke-training run and
# check actual GPU memory / iteration time / MFU to confirm the projection.
#
# IMPORTANT caveat (see docs/ARCHITECTURE.md): the upstream
# training/benchmark/scripts/estimate_memory.py reports whole-model memory and
# does NOT itself divide by tensor_parallel_size. This script applies that
# division manually for the dense weight/optimizer/gradient terms (the parts
# Megatron-Core TP actually shards) and keeps activations undivided as a
# conservative upper bound. The smoke test in step (b) is the authoritative
# check; treat step (a) as a fast first-pass filter only.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/lib/common.sh"

# SIZE_DATASET lets this script be run for either the primary (ml-20m, TP=8)
# or secondary (kuairand-1k, TP=1 -- see configs/hstu_8b_ranking_kuairand1k.gin
# scope note) training run. Defaults to PRIMARY_DATASET.
: "${SIZE_DATASET:=$PRIMARY_DATASET}"
DATASET_SLUG="${SIZE_DATASET//-/}"
PHASE="04_size_model_${DATASET_SLUG}"
phase_skip_if_done "$PHASE"

BASE_GIN="$REPO_ROOT/configs/hstu_${MODEL_SIZE}_ranking_${DATASET_SLUG}.gin"
TUNED_GIN="$REPO_ROOT/configs/hstu_${MODEL_SIZE}_ranking_${DATASET_SLUG}.tuned.gin"
[[ -f "$BASE_GIN" ]] || die "Base config not found: $BASE_GIN"

# TP degree actually declared in the base config (ml-20m=8, kuairand-1k=1);
# torchrun always launches on all $NUM_GPUS ranks -- Megatron-Core deduces the
# data-parallel degree as NUM_GPUS / ACTUAL_TP automatically.
ACTUAL_TP=$(grep -E '^TensorModelParallelArgs\.tensor_model_parallel_size\s*=' "$BASE_GIN" | grep -oE '[0-9]+' | head -n1)
ACTUAL_TP="${ACTUAL_TP:-1}"

log "=== Phase 4: Size model ($MODEL_SIZE on $SIZE_DATASET, TP=$ACTUAL_TP, DP=$((NUM_GPUS / ACTUAL_TP))) ==="
docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 || die "Image '$IMAGE_NAME' not found -- run scripts/01_build_env.sh first."

# Pull the knobs we need out of the base gin file (single source of truth).
gin_int() { grep -E "^$1\s*=" "$BASE_GIN" | head -n1 | grep -oE '[0-9]+' | head -n1; }
HIDDEN_SIZE=$(gin_int 'NetworkArgs\.hidden_size')
NUM_LAYERS=$(gin_int 'NetworkArgs\.num_layers')
NUM_HEADS=$(gin_int 'NetworkArgs\.num_attention_heads')
KV_CHANNELS=$(gin_int 'NetworkArgs\.kv_channels')
ITEM_EMB_DIM=$(gin_int 'NetworkArgs\.item_embedding_dim')
CTX_EMB_DIM=$(gin_int 'NetworkArgs\.contextual_embedding_dim')
MAX_HIST=$(gin_int 'DatasetArgs\.max_history_seqlen')
MAX_CAND=$(gin_int 'DatasetArgs\.max_num_candidates')
MAX_SEQ_LEN=$(( (MAX_HIST + MAX_CAND) * 2 + 8 ))   # +8 slack for contextual tokens
AVG_SEQ_LEN=$(( MAX_SEQ_LEN / 2 ))

# Approximate vocab sizes (not present in the gin file; from the upstream
# dataset table in examples/hstu/training/README.md). Only used to size the
# (small, for these datasets) embedding term -- dense weights dominate.
case "$SIZE_DATASET" in
  ml-20m)       ITEM_ROWS=30000;    CTX_ROWS=2000 ;;
  ml-1m)        ITEM_ROWS=4000;     CTX_ROWS=2000 ;;
  kuairand-1k)  ITEM_ROWS=4400000;  CTX_ROWS=100000 ;;
  kuairand-27k) ITEM_ROWS=32100000; CTX_ROWS=2000000 ;;
  *)            ITEM_ROWS=1000000; CTX_ROWS=100000 ;;
esac

log "Model: hidden=$HIDDEN_SIZE layers=$NUM_LAYERS heads=$NUM_HEADS kv_channels=$KV_CHANNELS"
log "Sequence: max_history=$MAX_HIST candidates=$MAX_CAND -> max_seq_len~=$MAX_SEQ_LEN avg~=$AVG_SEQ_LEN"

CANDIDATE_BATCHES=(4 8 12 16 24 32 48 64)
GPU_MEM_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1 | awk '{print $1/1024}')
[[ -n "$GPU_MEM_GB" ]] || GPU_MEM_GB=180
SAFETY_FRACTION=0.75   # leave 25% headroom for NCCL buffers/fragmentation/CUDA context

log "Running memory projections for batch sizes: ${CANDIDATE_BATCHES[*]} (GPU capacity ${GPU_MEM_GB}GB, safety fraction $SAFETY_FRACTION)..."

RESULTS_DIR="/workspace/results/mem_estimates_${DATASET_SLUG}"
docker_gpu_run bash -lc "
  set -e
  mkdir -p $RESULTS_DIR
  cd /workspace/recsys-examples/examples/hstu/training/benchmark/scripts
  for B in ${CANDIDATE_BATCHES[*]}; do
    python3 estimate_memory.py \
      --hidden_size $HIDDEN_SIZE --num_layers $NUM_LAYERS --num_attention_heads $NUM_HEADS --kv_channels $KV_CHANNELS \
      --item_embedding_dim $ITEM_EMB_DIM --contextual_embedding_dim $CTX_EMB_DIM \
      --item_embedding_rows $ITEM_ROWS --contextual_embedding_rows $CTX_ROWS \
      --embedding_gpu_cache_ratio 1.0 --dtype bf16 \
      --recompute_layernorm --batch_size \$B --max_seq_len $MAX_SEQ_LEN --avg_seq_len $AVG_SEQ_LEN \
      --tensor_parallel_size $ACTUAL_TP --optimizer adam \
      --output_json ${RESULTS_DIR}/batch_\${B}.json > ${RESULTS_DIR}/batch_\${B}.log 2>&1 || true
  done
" || die "Memory estimation sweep failed to run."

log "Selecting largest batch size within the safety threshold (TP=$ACTUAL_TP applied manually to dense weights/optimizer/gradients)..."
docker_gpu_run python3 - "$RESULTS_DIR" "$ACTUAL_TP" "$GPU_MEM_GB" "$SAFETY_FRACTION" <<'PYEOF' | tee "$STATE_DIR/${PHASE}_selection.log"
import glob, json, os, sys

results_dir, tp, gpu_mem_gb, safety = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
budget_bytes = gpu_mem_gb * (1024**3) * safety

rows = []
for path in sorted(glob.glob(os.path.join(results_dir, "batch_*.json"))):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        print(f"  skip {path}: {e}")
        continue
    mb = d["memory_breakdown"]
    batch = d["training_config"]["batch_size"]
    # Dense weights/optimizer/gradients are sharded by Megatron-Core TP; the
    # upstream estimator does not apply this division itself, so we do it here.
    weights_pergpu = mb["weights"]["total_gpu"] / tp
    opt_pergpu = mb["total"]["optimizer_total"] / tp
    grads_pergpu = mb["total"]["gradients"] / tp
    # Activations are conservatively left un-divided (upper bound; real TP/SP
    # sharding of activations depends on runtime config validated by the smoke test).
    act = mb["total"]["activations"]
    total_pergpu = weights_pergpu + opt_pergpu + grads_pergpu + act
    rows.append((batch, total_pergpu, weights_pergpu, opt_pergpu, grads_pergpu, act))

rows.sort()
print(f"{'batch':>6} {'proj_per_gpu_GB':>16} {'weights':>10} {'optimizer':>10} {'grads':>10} {'activations':>12}")
best = None
for batch, total, w, o, g, a in rows:
    gb = total / 1024**3
    fits = gb <= (budget_bytes / 1024**3)
    marker = "OK" if fits else "OVER"
    print(f"{batch:>6} {gb:>16.1f} {w/1024**3:>10.1f} {o/1024**3:>10.1f} {g/1024**3:>10.1f} {a/1024**3:>12.1f}  [{marker}]")
    if fits:
        best = batch

if best is None:
    print("NO_FIT")
else:
    print(f"SELECTED_BATCH_SIZE={best}")
PYEOF

SELECTED_BATCH=$(grep -oE 'SELECTED_BATCH_SIZE=[0-9]+' "$STATE_DIR/${PHASE}_selection.log" | tail -n1 | cut -d= -f2)
if [[ -z "${SELECTED_BATCH:-}" ]]; then
  warn "No candidate batch size fit within the safety threshold on the projection; falling back to the smallest candidate (4) and relying on the smoke test / gradient checkpointing."
  SELECTED_BATCH=4
fi
ok "Selected train_batch_size=$SELECTED_BATCH (projection). Writing $TUNED_GIN"

cat > "$TUNED_GIN" <<EOF
# AUTO-GENERATED by scripts/04_size_model.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ)
# Empirically-tuned overrides on top of $(basename "$BASE_GIN").
# Load both files together, tuned file last so it wins:
#   --gin-config-file $(basename "$BASE_GIN") --gin-config-file $(basename "$TUNED_GIN")
TrainerArgs.train_batch_size = $SELECTED_BATCH
TrainerArgs.eval_batch_size = $SELECTED_BATCH
DatasetArgs.dataset_path = '/workspace/data'
EOF

log "Running a short smoke-training run (~50 iterations) to validate the projection empirically..."
SMOKE_LOG="$STATE_DIR/${PHASE}_smoke_train.log"
GPU_MEM_LOG="$STATE_DIR/${PHASE}_smoke_gpu_mem.csv"
SMOKE_GIN="$REPO_ROOT/configs/.smoke_${MODEL_SIZE}_${DATASET_SLUG}.gin"

# Single self-contained file: base + tuned + smoke-only overrides (last
# binding in a gin file wins), so we don't depend on the entrypoint accepting
# --gin-config-file more than once.
render_effective_gin "$SMOKE_GIN" "$BASE_GIN" "$TUNED_GIN" -- \
  "TrainerArgs.max_train_iters = 50" \
  "TrainerArgs.eval_interval = 1000" \
  "TrainerArgs.ckpt_save_interval = 1000" \
  "TrainerArgs.ckpt_save_dir = '/workspace/checkpoints/smoke_test_${DATASET_SLUG}'"

# Background GPU memory sampler for the duration of the smoke test.
( nvidia-smi --query-gpu=index,memory.used --format=csv -l 2 > "$GPU_MEM_LOG" 2>/dev/null ) &
SAMPLER_PID=$!

set +e
docker_gpu_run bash -lc "
  export PYTHONPATH=\$PYTHONPATH:\$(realpath /workspace/recsys-examples/examples)
  cd /workspace/recsys-examples/examples/hstu
  torchrun --nproc_per_node $NUM_GPUS --master_addr localhost --master_port $MASTER_PORT \
    ./training/pretrain_gr_ranking.py \
    --gin-config-file /workspace/configs/$(basename "$SMOKE_GIN")
" > "$SMOKE_LOG" 2>&1
SMOKE_EXIT=$?
set -e

kill "$SAMPLER_PID" 2>/dev/null || true
wait "$SAMPLER_PID" 2>/dev/null || true

PEAK_MEM_MIB=$(awk -F, 'NR>1{gsub(/[^0-9]/,"",$2); if($2+0>max) max=$2+0} END{print max+0}' "$GPU_MEM_LOG" 2>/dev/null)
log "Smoke test peak GPU memory observed: ${PEAK_MEM_MIB:-unknown} MiB per the busiest sampled GPU (log: $GPU_MEM_LOG)"

if [[ "$SMOKE_EXIT" -ne 0 ]] || grep -qi "out of memory\|CUDA error\|OOM" "$SMOKE_LOG"; then
  warn "Smoke test at batch_size=$SELECTED_BATCH failed or OOM'd. See $SMOKE_LOG."
  NEXT_BATCH=4
  for b in "${CANDIDATE_BATCHES[@]}"; do
    if [[ "$b" -lt "$SELECTED_BATCH" ]]; then NEXT_BATCH="$b"; fi
  done
  die "Reduce the batch size (try train_batch_size=$NEXT_BATCH in $TUNED_GIN) and re-run: rm state/${PHASE}.done && ./scripts/04_size_model.sh"
fi

ok "Smoke test passed at train_batch_size=$SELECTED_BATCH. Full log: $SMOKE_LOG"
grep -iE "MFU|TFLOPS|iteration|elapsed" "$SMOKE_LOG" | tail -n 20 || true

phase_done_mark "$PHASE"
ok "Model sizing complete. Final config: $BASE_GIN + $TUNED_GIN (train_batch_size=$SELECTED_BATCH, TP=$NUM_GPUS). Proceed to scripts/05_train.sh."
