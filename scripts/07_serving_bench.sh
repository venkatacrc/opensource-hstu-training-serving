#!/usr/bin/env bash
# Phase 7: sweep batch sizes across all 4 serving tiers and collect
# latency/throughput JSON for scripts/08_generate_report.py.
#
#   Tier 0: naive eager PyTorch          (inference_gr_ranking helpers, forward_nokvcache)
#   Tier 1: optimized (KV cache + CUDA graph replay via the same helpers, forward_with_kvcache)
#   Tier 2: Triton Inference Server, Python backend
#   Tier 3: Triton Inference Server, AOTInductor C++ backend (single batch size -- static shapes)
#
# Tiers 2/3 are skipped gracefully (not fatal) if scripts/06_export_checkpoint.sh
# recorded them as unavailable. Each tier's results land in
# results/serving_bench/<tier>_bs<N>.json; this script does not itself
# interpret them -- scripts/08_generate_report.py does.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/lib/common.sh"

PHASE=07_serving_bench
phase_skip_if_done "$PHASE"

DATASET_SLUG="${SECONDARY_DATASET//-/}"
RUN_NAME="hstu_${MODEL_SIZE}_${DATASET_SLUG}"
INFERENCE_GIN="/workspace/configs/hstu_${MODEL_SIZE}_ranking_${DATASET_SLUG}.gin"

log "=== Phase 7: Serving benchmark ($RUN_NAME, batch sizes: $BENCH_BATCH_SIZES) ==="
docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 || die "Image '$IMAGE_NAME' not found -- run scripts/01_build_env.sh first."

CONTAINER_CKPT_DIR=$(cat "$STATE_DIR/06_export_checkpoint_checkpoint_dir.txt" 2>/dev/null || true)
[[ -n "$CONTAINER_CKPT_DIR" ]] || die "Checkpoint path not recorded -- run scripts/06_export_checkpoint.sh first."
TIER2_AVAILABLE=$(cat "$STATE_DIR/06_export_checkpoint_tier2_available.txt" 2>/dev/null || echo 0)
TIER3_AVAILABLE=$(cat "$STATE_DIR/06_export_checkpoint_tier3_available.txt" 2>/dev/null || echo 0)

OUT_DIR_HOST="$REPO_ROOT/results/serving_bench"
mkdir -p "$OUT_DIR_HOST"
OUT_DIR="/workspace/results/serving_bench"

IFS=',' read -r -a BATCH_LIST <<< "$BENCH_BATCH_SIZES"

# ---------------------------------------------------------------------------
# Tiers 0 & 1: direct Python inference (no server), same GPU, sequential.
# ---------------------------------------------------------------------------
run_tier01() {
  local tier_label="$1" disable_kv_flag="$2"
  for BS in "${BATCH_LIST[@]}"; do
    log "[$tier_label] batch_size=$BS"
    local out_json="${OUT_DIR}/${tier_label}_bs${BS}.json"
    set +e
    docker_gpu_run bash -lc "
      export CUDA_VISIBLE_DEVICES=0
      export PYTHONPATH=\$PYTHONPATH:\$(realpath /workspace/recsys-examples/examples)
      mkdir -p $OUT_DIR
      python3 /workspace/host_repo/scripts/lib/bench_tier01.py \
        --gin_config_file $INFERENCE_GIN \
        --checkpoint_dir $CONTAINER_CKPT_DIR \
        --batch_size $BS $disable_kv_flag \
        --warmup $BENCH_WARMUP_REQUESTS --iters $BENCH_MEASURED_REQUESTS \
        --tier $tier_label --output_json $out_json
    " > "$STATE_DIR/${PHASE}_${tier_label}_bs${BS}.log" 2>&1
    RC=$?
    set -e
    if [[ "$RC" -ne 0 ]]; then
      warn "[$tier_label] batch_size=$BS failed (see $STATE_DIR/${PHASE}_${tier_label}_bs${BS}.log) -- continuing with remaining batch sizes."
    else
      ok "[$tier_label] batch_size=$BS done."
    fi
  done
}

log "--- Tier 0: naive eager (no KV cache, no CUDA graph) ---"
run_tier01 "tier0_naive_eager" "--disable_kvcache"

log "--- Tier 1: optimized (KV cache + CUDA graph) ---"
run_tier01 "tier1_optimized_kvcache" ""

# ---------------------------------------------------------------------------
# Tier 2: Triton Inference Server, Python backend
# ---------------------------------------------------------------------------
if [[ "$TIER2_AVAILABLE" == "1" ]]; then
  log "--- Tier 2: Triton Inference Server (Python backend) ---"
  TIER2_CONTAINER="${CONTAINER_NAME}-tier2"
  docker rm -f "$TIER2_CONTAINER" >/dev/null 2>&1 || true
  BS_MAX=$(echo "$BENCH_BATCH_SIZES" | tr ',' '\n' | sort -n | tail -n1)

  # No $RECSYS_DIR bind-mount over /workspace/recsys-examples -- see the
  # explanatory comment above docker_gpu_run() in lib/common.sh. All the
  # sed patching below happens against the image's own (fully-built) copy,
  # inside this single long-lived container, so tritonserver (started later
  # in the same bash -lc session) reads back exactly what was just patched.
  set +e
  docker run -d --name "$TIER2_CONTAINER" --gpus all --ipc=host --network host \
    --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=16g \
    -v "$DATA_DIR:/workspace/data" \
    -v "$CKPT_DIR:/workspace/checkpoints" \
    -v "$REPO_ROOT/configs:/workspace/configs" \
    -v "$REPO_ROOT/results:/workspace/results" \
    -w /workspace/recsys-examples/examples/hstu \
    "$IMAGE_NAME" bash -lc "
      set -e
      export CUDA_VISIBLE_DEVICES=0
      export HSTU_INFERENCE_ONLY=1
      export PYTHONPATH=\$PYTHONPATH:\$(realpath /workspace/recsys-examples/examples)
      MODEL_REPO=./inference/triton
      DENSE_DIR=\$MODEL_REPO/hstu_model
      SPARSE_DIR=\$MODEL_REPO/hstu_sparse
      GIN=$INFERENCE_GIN

      # Point the (reused training) gin at NVEmbedding for serving, and at
      # the preprocessed dataset, mirroring launch_and_test_triton_python_backend.sh step 3.
      sed -i '/^[[:space:]]*NetworkArgs\.embedding_backend[[:space:]]*=/d' \$GIN 2>/dev/null || true
      cp \$GIN /workspace/configs/.bench_inference_${DATASET_SLUG}.gin
      printf \"\nNetworkArgs.embedding_backend = 'NVEmb'\nDatasetArgs.dataset_path = '/workspace/data'\n\" >> /workspace/configs/.bench_inference_${DATASET_SLUG}.gin

      PS_MODULE_DIR=$CONTAINER_CKPT_DIR/ps_module
      rm -rf \"\$PS_MODULE_DIR\"; mkdir -p \"\$PS_MODULE_DIR\"
      find $CONTAINER_CKPT_DIR/dynamicemb_module -type f -regex '.*/.*_emb_.*' -print0 2>/dev/null | \
        while IFS= read -r -d '' f; do ln -s \"\$(realpath \"\$f\")\" \"\$PS_MODULE_DIR/\$(basename \"\$f\").dyn\"; done

      BENCH_GIN_CONTAINER=/workspace/configs/.bench_inference_${DATASET_SLUG}.gin
      mkdir -p \"\$DENSE_DIR/1\" \"\$SPARSE_DIR/1\"
      cp \"\$DENSE_DIR/model.py\" \"\$DENSE_DIR/1/model.py\"
      cp \"\$SPARSE_DIR/model.py\" \"\$SPARSE_DIR/1/model.py\"
      for CFG in \"\$DENSE_DIR/config.pbtxt\" \"\$SPARSE_DIR/config.pbtxt\"; do
        if ! grep -q 'key: \"HSTU_CHECKPOINT_DIR\"' \"\$CFG\"; then
          printf '\nparameters [\n {\n key: \"HSTU_CHECKPOINT_DIR\"\n value: {\n string_value: \"%s\"\n }\n }\n]\n' \"$CONTAINER_CKPT_DIR\" >> \"\$CFG\"
        else
          sed -i \"/key: \\\"HSTU_CHECKPOINT_DIR\\\"/,/}/ s#string_value: \\\"[^\\\"]*\\\"#string_value: \\\"$CONTAINER_CKPT_DIR\\\"#\" \"\$CFG\"
        fi
        # HSTU_GIN_CONFIG_FILE defaults (in both hstu_model/ and hstu_sparse/
        # config.pbtxt shipped upstream) to a tiny demo config
        # (inference/configs/kuairand_1k_inference_ranking.gin, hidden_size=512)
        # meant only for upstream's own smoke test -- NOT the HSTU-8B network
        # shape our checkpoint was trained with. Left unpatched, Triton loads
        # that 512-hidden-size model and then fails with a torch state_dict
        # size mismatch (checkpoint tensors are 8192-wide) as soon as it tries
        # to load our real weights. Point it at the real, NVEmb-patched gin
        # assembled above instead.
        if ! grep -q 'key: \"HSTU_GIN_CONFIG_FILE\"' \"\$CFG\"; then
          printf '\nparameters [\n {\n key: \"HSTU_GIN_CONFIG_FILE\"\n value: {\n string_value: \"%s\"\n }\n }\n]\n' \"\$BENCH_GIN_CONTAINER\" >> \"\$CFG\"
        else
          sed -i \"/key: \\\"HSTU_GIN_CONFIG_FILE\\\"/,/}/ s#string_value: \\\"[^\\\"]*\\\"#string_value: \\\"\$BENCH_GIN_CONTAINER\\\"#\" \"\$CFG\"
        fi
        if ! grep -q 'max_batch_size' \"\$CFG\"; then
          sed -i \"1i max_batch_size: ${BS_MAX}\" \"\$CFG\"
        fi
        echo \"--- effective \$CFG ---\"; cat \"\$CFG\"
      done

      export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:/opt/hpcx/ucc/lib:/opt/hpcx/ucx/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:\${LD_LIBRARY_PATH:-}
      tritonserver --model-repository \$MODEL_REPO > /workspace/results/${PHASE}_tier2_server.log 2>&1
    " > "$STATE_DIR/${PHASE}_tier2_launch.log" 2>&1
  LAUNCH_RC=$?
  set -e

  if [[ "$LAUNCH_RC" -ne 0 ]]; then
    warn "Failed to launch Tier 2 Triton container (see $STATE_DIR/${PHASE}_tier2_launch.log). Skipping Tier 2."
  else
    log "Waiting for Triton (Tier 2) to become ready..."
    READY=0
    for _ in $(seq 1 60); do
      if curl -sSf http://localhost:8000/v2/health/ready >/dev/null 2>&1; then READY=1; break; fi
      sleep 5
    done
    if [[ "$READY" -ne 1 ]]; then
      warn "Tier 2 Triton server did not become ready in time. See docker logs $TIER2_CONTAINER. Skipping Tier 2 sweep."
    else
      ok "Tier 2 Triton server ready."
      for BS in "${BATCH_LIST[@]}"; do
        log "[tier2_triton_python] batch_size=$BS"
        set +e
        docker_gpu_run_hostnet bash -lc "
          pip show tritonclient >/dev/null 2>&1 || pip install --quiet 'tritonclient[http]'
          mkdir -p $OUT_DIR
          python3 /workspace/host_repo/scripts/lib/bench_triton_tier2.py \
            --gin_config_file /workspace/configs/.bench_inference_${DATASET_SLUG}.gin --batch_size $BS \
            --warmup $BENCH_WARMUP_REQUESTS --iters $BENCH_MEASURED_REQUESTS \
            --output_json ${OUT_DIR}/tier2_triton_python_bs${BS}.json
        " > "$STATE_DIR/${PHASE}_tier2_bs${BS}.log" 2>&1
        RC=$?
        set -e
        [[ "$RC" -eq 0 ]] && ok "[tier2_triton_python] batch_size=$BS done." || warn "[tier2_triton_python] batch_size=$BS failed (see $STATE_DIR/${PHASE}_tier2_bs${BS}.log)."
      done
    fi
    docker logs "$TIER2_CONTAINER" > "$STATE_DIR/${PHASE}_tier2_server_full.log" 2>&1 || true
    docker rm -f "$TIER2_CONTAINER" >/dev/null 2>&1 || true
  fi
else
  warn "Tier 2 marked unavailable by scripts/06_export_checkpoint.sh -- skipping."
fi

# ---------------------------------------------------------------------------
# Tier 3: Triton Inference Server, AOTInductor C++ backend (single batch size)
# ---------------------------------------------------------------------------
if [[ "$TIER3_AVAILABLE" == "1" ]]; then
  log "--- Tier 3: Triton Inference Server (AOTInductor C++ backend) ---"
  TIER3_CONTAINER="${CONTAINER_NAME}-tier3"
  docker rm -f "$TIER3_CONTAINER" >/dev/null 2>&1 || true
  MAX_BS_FOR_EXPORT=$(echo "$BENCH_BATCH_SIZES" | tr ',' '\n' | sort -n | tail -n1)

  # No $RECSYS_DIR bind-mount here either -- see the comment above the Tier 2
  # docker run above / docker_gpu_run() in lib/common.sh for why: the
  # ${IMAGE_NAME}-tritonserver image already has the fully-built
  # /workspace/recsys-examples (triton_libs/, patched config.pbtxt) baked in.
  set +e
  docker run -d --name "$TIER3_CONTAINER" --gpus all --ipc=host --network host \
    -v "$TRITON_REPO_DIR:/triton_model_repo" \
    -v "$EXPORT_DIR:/workspace/export" \
    -v "$REPO_ROOT/results:/workspace/results" \
    "${IMAGE_NAME}-tritonserver" bash -lc "
      cd /workspace/recsys-examples/examples/hstu/inference_aoti
      export FLEXKV_LOG_LEVEL=WARNING
      export KVCACHE_MANAGER_CONFIG_FILE=\${PWD}/kvcache_cpp_runtime.yaml
      cp -apr /workspace/export/export_test_dump . 2>/dev/null || true
      python3 start_flexkv_server_for_kvcache_cpp.py --config_file \${KVCACHE_MANAGER_CONFIG_FILE} > /workspace/results/${PHASE}_tier3_flexkv.log 2>&1 &
      sleep 10
      tritonserver --model-repository=/triton_model_repo/ > /workspace/results/${PHASE}_tier3_server.log 2>&1
    " > "$STATE_DIR/${PHASE}_tier3_launch.log" 2>&1
  LAUNCH_RC=$?
  set -e

  if [[ "$LAUNCH_RC" -ne 0 ]]; then
    warn "Failed to launch Tier 3 Triton container (see $STATE_DIR/${PHASE}_tier3_launch.log). Skipping Tier 3."
  else
    log "Waiting for Triton (Tier 3) to become ready..."
    READY=0
    for _ in $(seq 1 60); do
      if curl -sSf http://localhost:8000/v2/health/ready >/dev/null 2>&1; then READY=1; break; fi
      sleep 5
    done
    if [[ "$READY" -ne 1 ]]; then
      warn "Tier 3 Triton server did not become ready in time. See docker logs $TIER3_CONTAINER. Skipping Tier 3 benchmark."
    else
      ok "Tier 3 Triton server ready. Benchmarking at the single exported batch size ($MAX_BS_FOR_EXPORT -- AOTI static shapes, see script header)."
      set +e
      docker_gpu_run_hostnet bash -lc "
        pip show tritonclient >/dev/null 2>&1 || pip install --quiet 'tritonclient[http]'
        mkdir -p $OUT_DIR
        python3 /workspace/host_repo/scripts/lib/bench_triton_tier3.py \
          --dump_dir /workspace/export/export_test_dump --batch_size $MAX_BS_FOR_EXPORT \
          --warmup $BENCH_WARMUP_REQUESTS --iters $BENCH_MEASURED_REQUESTS \
          --output_json ${OUT_DIR}/tier3_triton_aoti_bs${MAX_BS_FOR_EXPORT}.json
      " > "$STATE_DIR/${PHASE}_tier3_bs${MAX_BS_FOR_EXPORT}.log" 2>&1
      RC=$?
      set -e
      [[ "$RC" -eq 0 ]] && ok "[tier3_triton_aoti] batch_size=$MAX_BS_FOR_EXPORT done." || warn "[tier3_triton_aoti] failed (see $STATE_DIR/${PHASE}_tier3_bs${MAX_BS_FOR_EXPORT}.log)."
    fi
    docker logs "$TIER3_CONTAINER" > "$STATE_DIR/${PHASE}_tier3_server_full.log" 2>&1 || true
    docker rm -f "$TIER3_CONTAINER" >/dev/null 2>&1 || true
  fi
else
  warn "Tier 3 marked unavailable by scripts/06_export_checkpoint.sh -- skipping."
fi

N_RESULTS=$(find "$OUT_DIR_HOST" -name '*.json' | wc -l | tr -d ' ')
log "Serving benchmark produced $N_RESULTS result file(s) in $OUT_DIR_HOST"
[[ "$N_RESULTS" -gt 0 ]] || die "No serving benchmark results were produced at all (all tiers failed) -- check the logs referenced above."

phase_done_mark "$PHASE"
ok "Serving benchmark complete. Proceed to scripts/08_generate_report.py."
