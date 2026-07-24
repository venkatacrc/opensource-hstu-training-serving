#!/usr/bin/env bash
# Phase 2: download the public datasets. Pure CPU/network work -- intended to
# run concurrently with scripts/01_build_env.sh's long Docker build (launch
# both in the background from run_all.sh).
#
# Sources (both fetched directly by the upstream preprocessor, no auth/manual
# click-through needed):
#   ml-20m       -> http://files.grouplens.org/datasets/movielens/ml-20m.zip
#   kuairand-1k  -> https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/lib/common.sh"

PHASE=02_fetch_datasets
phase_skip_if_done "$PHASE"

log "=== Phase 2: Fetch public datasets ($PRIMARY_DATASET, $SECONDARY_DATASET) ==="

ensure_recsys_checkout

# The preprocessor (examples/commons/hstu_data_preprocessor.py) is pure
# numpy/pandas -- no torch/CUDA needed -- so we run it in a small host-side
# venv to avoid waiting on the Docker build. Falls back to running it inside
# the (already-present) container image if a host venv can't be created.
HOST_VENV="$WORK_DIR/.preprocess_venv"

# NOTE: the preprocessor's --dataset_path must end up matching the path used
# at training time; scripts/04_size_model.sh points DatasetArgs.dataset_path at
# the same directory (mounted into the container as /workspace/data).
run_preprocessor() {
  local dataset="$1"
  python3 "$RECSYS_DIR/examples/commons/hstu_data_preprocessor.py" \
    --dataset_name "$dataset" --dataset_path "$DATA_DIR" --training --inference
}

if command -v python3 >/dev/null 2>&1 && python3 -m venv "$HOST_VENV" 2>/tmp/venv_err.log; then
  ok "Using host venv at $HOST_VENV for dataset download + preprocessing (runs concurrently with the Docker build -- no GPU/torch needed for this step)."
  # shellcheck disable=SC1091
  source "$HOST_VENV/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet numpy pandas
  for ds in "$PRIMARY_DATASET" "$SECONDARY_DATASET"; do
    log "Downloading + preprocessing (train + inference splits): $ds"
    run_preprocessor "$ds"
  done
  deactivate
else
  warn "Could not create a host venv ($(cat /tmp/venv_err.log 2>/dev/null)); falling back to running the preprocessor inside the container. This will block until scripts/01_build_env.sh's image is built."
  [[ -n "${IMAGE_NAME:-}" ]] || die "IMAGE_NAME not set"
  until docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; do
    log "Waiting for image '$IMAGE_NAME' to be built by 01_build_env.sh..."
    sleep 60
  done
  for ds in "$PRIMARY_DATASET" "$SECONDARY_DATASET"; do
    docker_gpu_run bash -lc "cd /workspace/recsys-examples/examples/commons && python3 hstu_data_preprocessor.py --dataset_name $ds --dataset_path /workspace/data --training --inference"
  done
fi

log "Dataset directory contents:"
du -sh "$DATA_DIR"/* 2>/dev/null | tee "$STATE_DIR/${PHASE}_sizes.txt"

phase_done_mark "$PHASE"
ok "Datasets ready in $DATA_DIR. Proceed to scripts/04_size_model.sh once scripts/01_build_env.sh finishes."
