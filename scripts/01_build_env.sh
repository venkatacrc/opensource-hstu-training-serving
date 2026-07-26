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

patch_dockerfile_known_upstream_issues() {
  # Known bug in recsys-examples@v26.06.01's docker/Dockerfile (base_triton
  # stage): it pins triton-inference-server/pytorch_backend to commit ceeecb7
  # via `git fetch --depth 1 origin r26.07 && git checkout --detach ceeecb7`.
  # That only works while ceeecb7 happens to be the exact tip of r26.07; the
  # instant upstream pushes any further commit to that branch, ceeecb7 falls
  # out of the depth-1 shallow window and `git checkout --detach ceeecb7`
  # fails with "fatal: git checkout: --detach does not take a path argument
  # 'ceeecb7'" (git can't resolve the unfetched SHA, so it tries -- and fails
  # -- to treat it as a pathspec). Reproduced independently against the live
  # repo: r26.07's tip has already moved to a commit after ceeecb7.
  #
  # Fix: fetch the exact pinned commit by full SHA instead of by branch name.
  # GitHub allows shallow-fetching an arbitrary reachable SHA directly
  # (verified against this exact repo/commit), so this is immune to the
  # branch moving further in the future -- no depth-guessing required.
  local dockerfile="$RECSYS_DIR/docker/Dockerfile"
  [[ -f "$dockerfile" ]] || return 0
  if grep -q 'git fetch --depth 1 origin r26.07' "$dockerfile" 2>/dev/null; then
    log "Patching known-broken pytorch_backend@ceeecb7 pin in $dockerfile (upstream shallow-fetch-by-branch bug -- see comment in $0 for details)..."
    sed -i.bak \
      -e 's/git fetch --depth 1 origin r26\.07;/git fetch --depth 1 origin ceeecb748caa785b359095a74c05d55dace2591a;/' \
      -e 's/git checkout --detach ceeecb7;/git checkout --detach FETCH_HEAD;/' \
      "$dockerfile"
    rm -f "$dockerfile.bak"
    grep -q 'git fetch --depth 1 origin ceeecb748caa785b359095a74c05d55dace2591a' "$dockerfile" \
      || die "Failed to patch $dockerfile (expected pattern not found after sed -- upstream file layout may have changed; inspect manually)."
    ok "Dockerfile patched."
  else
    log "Dockerfile does not contain the known r26.07-shallow-fetch pattern (already patched, or upstream fixed it) -- no patch needed."
  fi
}
patch_dockerfile_known_upstream_issues

patch_trainer_utils_known_upstream_issues() {
  # Known bug in recsys-examples@$RECSYS_TAG: training/trainer/utils.py:235
  # calls get_common_preprocessors() with no arguments, but the function
  # (examples/commons/hstu_data_preprocessor.py:721) requires a dataset_path
  # positional arg. Every other call site in the repo (inference_gr_ranking.py,
  # export_inference_gr_ranking*.py) already passes "" since they only use
  # the returned dict's dataset_name key, not the dataset_path itself --
  # apply the same fix here.
  local f="$RECSYS_DIR/examples/hstu/training/trainer/utils.py"
  [[ -f "$f" ]] || return 0
  if grep -q 'get_common_preprocessors()' "$f" 2>/dev/null; then
    log "Patching known-broken get_common_preprocessors() call (missing dataset_path arg) in $f..."
    sed -i.bak 's/get_common_preprocessors()/get_common_preprocessors("")/' "$f"
    rm -f "$f.bak"
    grep -q 'get_common_preprocessors()' "$f" && die "Failed to patch $f -- pattern still present after sed."
    ok "trainer/utils.py patched."
  else
    log "No bare get_common_preprocessors() call found in $f -- already patched or upstream fixed it."
  fi
}
patch_trainer_utils_known_upstream_issues

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
GPU_SEEN=$(docker run --rm --gpus all --entrypoint python3 "$IMAGE_NAME" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -n1)

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
