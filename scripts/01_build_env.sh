#!/usr/bin/env bash
# Phase 1: clone NVIDIA/recsys-examples (pinned tag) and build the training +
# serving Docker image from its Dockerfile. This is the single longest fixed
# cost in the whole pipeline (~1.5-3h: FBGEMM HSTU kernel nvcc compile alone is
# documented upstream as ~55min, plus a from-source Triton Server build).
#
# Designed to be launched in the background (nohup) so scripts/02_fetch_datasets.sh
# and scripts/03_preprocess_data.sh can run concurrently on the CPU side.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/lib/common.sh"

PHASE=01_build_env
phase_skip_if_done "$PHASE"

log "=== Phase 1: Build training/serving environment ==="
require_cmd git docker

START_TS=$(date +%s)

ensure_recsys_checkout
log "Fetching submodules (FBGEMM, etc.) -- can take a few minutes..."
git -C "$RECSYS_DIR" submodule update --init --recursive
ok "recsys-examples checked out at $(git -C "$RECSYS_DIR" rev-parse --short HEAD) (tag $RECSYS_TAG)"

log "Building Docker image '$IMAGE_NAME' from $RECSYS_DIR/docker/Dockerfile (base=$BASE_IMAGE)."
log "This is a multi-stage build (FBGEMM -> TorchRec -> Triton Server -> Megatron/FlashAttention/FlexKV -> recsys-examples). Expect 1.5-3 hours on first run."
BUILD_LOG="$STATE_DIR/${PHASE}_docker_build.log"
log "Full build output -> $BUILD_LOG"

if ! docker build \
    --platform linux/amd64 \
    --build-arg "BASE_IMAGE=$BASE_IMAGE" \
    -f docker/Dockerfile \
    -t "$IMAGE_NAME" \
    "$RECSYS_DIR" 2>&1 | tee "$BUILD_LOG"; then
  die "Docker build failed. Check $BUILD_LOG for the failing layer. Common causes: disk space, network flakiness pulling apt/pip packages, or a submodule that didn't fetch (re-run 'git submodule update --init --recursive' in $RECSYS_DIR)."
fi

END_TS=$(date +%s)
BUILD_SECS=$((END_TS - START_TS))
log "Docker image build finished in $(elapsed_human "$BUILD_SECS")."
echo "$BUILD_SECS" > "$STATE_DIR/${PHASE}_duration_seconds.txt"

log "Verifying image can see all $NUM_GPUS GPUs..."
GPU_SEEN=$(docker run --rm --gpus all "$IMAGE_NAME" python3 -c "import torch; print(torch.cuda.device_count())")
[[ "$GPU_SEEN" -eq "$NUM_GPUS" ]] || die "Image reports $GPU_SEEN GPUs, expected $NUM_GPUS."
ok "Image sees $GPU_SEEN GPUs."

log "Verifying HSTU attention kernel + TorchRec + Megatron-Core import cleanly inside the image..."
docker run --rm --gpus all "$IMAGE_NAME" python3 -c "
import torch, torchrec, megatron.core, hstu
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('torchrec', torchrec.__version__ if hasattr(torchrec, '__version__') else 'ok')
print('megatron.core ok')
print('hstu (fbgemm_gpu_hstu) ok')
" || die "Core library import check failed inside the built image."

phase_done_mark "$PHASE"
ok "Environment build complete. Image: $IMAGE_NAME. Proceed to scripts/03_preprocess_data.sh (after datasets are fetched) then scripts/04_size_model.sh."
