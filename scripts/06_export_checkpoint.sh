#!/usr/bin/env bash
# Phase 6: prepare everything scripts/07_serving_bench.sh needs to compare all
# 4 serving tiers on the SECONDARY_DATASET (kuairand-1k, TP=1) checkpoint --
# see the scope note in configs/hstu_8b_ranking_kuairand1k.gin for why this
# checkpoint (not the primary ml-20m one) drives the serving benchmark.
#
# Tiers 0/1 (naive eager, optimized KV-cache+CUDA-graph) need nothing extra --
# scripts/07_serving_bench.sh calls inference/inference_gr_ranking.py directly
# against the raw checkpoint. This script prepares:
#   Tier 2: Triton Inference Server Python backend model repo
#   Tier 3: torch.export + AOTInductor C++ package, native replay validation,
#           and the (separate, lightweight) Triton AOTI runtime image
#
# Tier 3 is the highest-risk step in this whole pipeline (deepest, least
# generic part of the upstream reference: custom C++ build, FlexKV KV-cache
# server, AOTI packaging). Failures here are caught and recorded rather than
# aborting the run -- scripts/07_serving_bench.sh and scripts/08_generate_report.py
# both check state/06_export_checkpoint.tier3_available and simply omit Tier 3
# from the comparison if it's not "1", rather than failing the whole pipeline.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/lib/common.sh"

PHASE=06_export_checkpoint
phase_skip_if_done "$PHASE"

DATASET_SLUG="${SECONDARY_DATASET//-/}"
RUN_NAME="hstu_${MODEL_SIZE}_${DATASET_SLUG}"
HOST_RUN_CKPT_DIR="$CKPT_DIR/$RUN_NAME"

log "=== Phase 6: Export checkpoint for serving ($RUN_NAME) ==="
docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 || die "Image '$IMAGE_NAME' not found -- run scripts/01_build_env.sh first."
[[ -d "$HOST_RUN_CKPT_DIR" ]] || die "No checkpoint at $HOST_RUN_CKPT_DIR -- run: TRAIN_DATASET=$SECONDARY_DATASET TRAIN_HOURS=$SECONDARY_TRAIN_BUDGET_HOURS ./scripts/05_train.sh"

LATEST_ITER_DIR=$(find "$HOST_RUN_CKPT_DIR" -maxdepth 1 -type d -name 'iter*' | sort -V | tail -n1)
[[ -n "$LATEST_ITER_DIR" ]] || die "No iterNNNN checkpoint subdirectory found under $HOST_RUN_CKPT_DIR"
LATEST_ITER_NAME=$(basename "$LATEST_ITER_DIR")
CONTAINER_CKPT_DIR="/workspace/checkpoints/${RUN_NAME}/${LATEST_ITER_NAME}"
ok "Using checkpoint: $LATEST_ITER_DIR (container path: $CONTAINER_CKPT_DIR)"
echo "$CONTAINER_CKPT_DIR" > "$STATE_DIR/${PHASE}_checkpoint_dir.txt"

INFERENCE_GIN="/workspace/configs/hstu_${MODEL_SIZE}_ranking_${DATASET_SLUG}.gin"
[[ -f "$REPO_ROOT/configs/hstu_${MODEL_SIZE}_ranking_${DATASET_SLUG}.gin" ]] || die "Missing $REPO_ROOT/configs/hstu_${MODEL_SIZE}_ranking_${DATASET_SLUG}.gin"
log "Reusing training gin ($INFERENCE_GIN) for inference too -- it defines the same DatasetArgs/NetworkArgs/RankingArgs schema the inference entrypoints read; the unused TrainerArgs/OptimizerArgs bindings are simply never instantiated at inference time."

# ---------------------------------------------------------------------------
# Tier 2: Triton Inference Server Python backend model repo
# ---------------------------------------------------------------------------
# Actual staging + server launch happens in scripts/07_serving_bench.sh (it
# needs a long-lived server process); here we just validate the prerequisite
# the upstream launcher documents -- the checkpoint must contain a
# `dynamicemb_module` dir with the user_id/video_id embedding files it wires
# into NVEmbedding's `ps_module` layout.
log "--- Validating Tier 2 (Triton Python backend) prerequisites ---"
TIER2_OK=0
if [[ -d "$LATEST_ITER_DIR/dynamicemb_module" ]]; then
  ok "Checkpoint has dynamicemb_module/ -- Tier 2 (Triton Python backend) prerequisites satisfied."
  TIER2_OK=1
else
  warn "$LATEST_ITER_DIR has no dynamicemb_module/ dir -- Tier 2 (Triton Python backend) staging in scripts/07_serving_bench.sh may fail; check $LATEST_ITER_DIR's contents against docs/RUNBOOK.md#tier-2-troubleshooting."
fi
echo "$TIER2_OK" > "$STATE_DIR/${PHASE}_tier2_available.txt"

# ---------------------------------------------------------------------------
# Tier 3: torch.export + AOTInductor C++ package + Triton AOTI runtime image
# ---------------------------------------------------------------------------
log "--- Attempting Tier 3 (AOTInductor C++ + Triton) export (best-effort) ---"
TIER3_OK=0
KVCACHE_CFG="/workspace/recsys-examples/examples/hstu/inference_aoti/kvcache_cpp_runtime.yaml"
MAX_BS_FOR_EXPORT=$(echo "$BENCH_BATCH_SIZES" | tr ',' '\n' | sort -n | tail -n1)

set +e
docker_gpu_run bash -lc "
  set -e
  export FLEXKV_LOG_LEVEL=WARNING
  export DYNAMICEMB_OPS_LIB_DIR=/workspace/recsys-examples/corelib/dynamicemb/torch_binding_build/
  export PYTHONPATH=\$PYTHONPATH:\$(realpath /workspace/recsys-examples/examples)
  cd /workspace/recsys-examples/examples/hstu
  export KVCACHE_MANAGER_CONFIG_FILE=$KVCACHE_CFG

  python3 ./inference_aoti/export_inference_gr_ranking_kvcache.py \
    --gin_config_file $INFERENCE_GIN \
    --checkpoint_dir $CONTAINER_CKPT_DIR \
    --max_bs $MAX_BS_FOR_EXPORT --kvcache_config_file \${KVCACHE_MANAGER_CONFIG_FILE}

  python3 ./inference_aoti/start_flexkv_server_for_kvcache_cpp.py \
    --config_file \${KVCACHE_MANAGER_CONFIG_FILE} > /workspace/results/${PHASE}_flexkv_server.log 2>&1 &
  kvserver_pid=\$!
  sleep 10
  kill -0 \${kvserver_pid}

  ./inference_aoti/cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model \
    ./inference_aoti/hstu_gr_ranking_kvcache_model \
    ./inference_aoti/export_test_dump

  kill \${kvserver_pid} || true

  mkdir -p /workspace/export/hstu_gr_ranking_kvcache_model /workspace/export/export_test_dump
  cp -apr ./inference_aoti/hstu_gr_ranking_kvcache_model/. /workspace/export/hstu_gr_ranking_kvcache_model/
  cp -apr ./inference_aoti/export_test_dump/. /workspace/export/export_test_dump/
" > "$STATE_DIR/${PHASE}_tier3_export.log" 2>&1
TIER3_EXPORT_EXIT=$?
set -e

if [[ "$TIER3_EXPORT_EXIT" -eq 0 ]]; then
  ok "AOTI export + native C++ replay validation succeeded. Building the (separate, lightweight) Triton AOTI runtime image..."
  set +e
  docker build --progress=plain \
    --build-arg "PYTORCH_AOTI_IMAGE=$IMAGE_NAME" \
    -f "$RECSYS_DIR/docker/Dockerfile.tritonserver" \
    -t "${IMAGE_NAME}-tritonserver" \
    "$RECSYS_DIR" > "$STATE_DIR/${PHASE}_tritonserver_image_build.log" 2>&1
  TRITON_IMG_EXIT=$?
  set -e
  if [[ "$TRITON_IMG_EXIT" -eq 0 ]]; then
    log "Staging Tier 3 Triton model repo at $TRITON_REPO_DIR/hstu_gr_ranking_kvcache ..."
    mkdir -p "$TRITON_REPO_DIR/hstu_gr_ranking_kvcache/1"
    docker run --rm \
      -v "$RECSYS_DIR:/workspace/recsys-examples" \
      -v "$TRITON_REPO_DIR:/workspace/triton_model_repo" \
      -v "$EXPORT_DIR:/workspace/export" \
      "${IMAGE_NAME}-tritonserver" bash -lc "
        cp -apr /workspace/recsys-examples/examples/hstu/inference_aoti/triton_aoti/hstu_gr_ranking_kvcache/. /workspace/triton_model_repo/hstu_gr_ranking_kvcache/
        cp -apr /workspace/export/hstu_gr_ranking_kvcache_model/. /workspace/triton_model_repo/hstu_gr_ranking_kvcache/1/
      " >> "$STATE_DIR/${PHASE}_tritonserver_image_build.log" 2>&1
    TIER3_OK=1
    ok "Tier 3 (AOTInductor C++ + Triton) fully staged."
  else
    warn "Tier 3 Triton runtime image build failed (see $STATE_DIR/${PHASE}_tritonserver_image_build.log). Tier 3 will be excluded from the serving comparison."
  fi
else
  warn "Tier 3 AOTI export/replay failed (see $STATE_DIR/${PHASE}_tier3_export.log). This is the most novel/fragile step in the pipeline (torch.export + AOTInductor + FlexKV C++ replay for a checkpoint this large); Tier 3 will be excluded from the serving comparison and docs/RESULTS.md will note it as not available rather than blocking the rest of the pipeline. See docs/RUNBOOK.md#tier-3-troubleshooting."
fi
echo "$TIER3_OK" > "$STATE_DIR/${PHASE}_tier3_available.txt"

phase_done_mark "$PHASE"
ok "Export phase complete. Tier2 available=$TIER2_OK, Tier3 available=$TIER3_OK. Proceed to scripts/07_serving_bench.sh."
