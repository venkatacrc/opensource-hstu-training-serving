#!/usr/bin/env bash
# Shared helpers for scripts/*.sh. Source this after determining REPO_ROOT:
#   REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
#   source "$REPO_ROOT/lib/common.sh"

set -uo pipefail

if [[ -z "${REPO_ROOT:-}" ]]; then
  echo "common.sh: REPO_ROOT must be set before sourcing" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$REPO_ROOT/config.env"

mkdir -p "$WORK_DIR" "$DATA_DIR" "$CKPT_DIR" "$EXPORT_DIR" "$TRITON_REPO_DIR" \
         "$REPO_ROOT/state" "$REPO_ROOT/results"

# --- logging ----------------------------------------------------------------
_c_reset='\033[0m'; _c_blue='\033[1;34m'; _c_green='\033[1;32m'; _c_yellow='\033[1;33m'; _c_red='\033[1;31m'

log()  { printf "${_c_blue}[%s]${_c_reset} %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
ok()   { printf "${_c_green}[%s] OK:${_c_reset} %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
warn() { printf "${_c_yellow}[%s] WARN:${_c_reset} %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; }
die()  { printf "${_c_red}[%s] ERROR:${_c_reset} %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >&2; exit 1; }

# --- idempotency markers ------------------------------------------------------
# Each phase script calls `phase_done_mark <name>` on success and callers
# (run_all.sh) can check `phase_is_done <name>` to skip already-completed work.
STATE_DIR="$REPO_ROOT/state"

phase_is_done() {
  local name="$1"
  [[ -f "$STATE_DIR/${name}.done" ]]
}

phase_done_mark() {
  local name="$1"
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$STATE_DIR/${name}.done"
}

phase_skip_if_done() {
  local name="$1"
  if phase_is_done "$name"; then
    ok "Phase '$name' already completed at $(cat "$STATE_DIR/${name}.done") — skipping (rm state/${name}.done to force rerun)."
    exit 0
  fi
}

ensure_recsys_checkout() {
  # Idempotent clone/checkout of the pinned NVIDIA/recsys-examples tag. Shared
  # by 01_build_env.sh (needs it for the Docker build context) and
  # 02_fetch_datasets.sh (needs only the pure-Python preprocessor script, so it
  # can run this concurrently with the Docker build).
  require_cmd git
  if [[ -d "$RECSYS_DIR/.git" ]]; then
    log "recsys-examples already present at $RECSYS_DIR"
  else
    log "Cloning $RECSYS_REMOTE @ $RECSYS_TAG into $RECSYS_DIR ..."
    mkdir -p "$(dirname "$RECSYS_DIR")"
    git clone --recursive --branch "$RECSYS_TAG" "$RECSYS_REMOTE" "$RECSYS_DIR"
  fi
  git -C "$RECSYS_DIR" checkout "$RECSYS_TAG" 2>/dev/null || true
}

require_cmd() {
  for c in "$@"; do
    command -v "$c" >/dev/null 2>&1 || die "Required command not found: $c"
  done
}

# --- docker helpers -----------------------------------------------------------
docker_gpu_run() {
  # Run a one-shot command in the benchmark image with all GPUs, repo mounted.
  # -i so callers can pipe a heredoc into e.g. `docker_gpu_run python3 -`.
  docker run --rm -i --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size=32g \
    -v "$RECSYS_DIR:/workspace/recsys-examples" \
    -v "$DATA_DIR:/workspace/data" \
    -v "$CKPT_DIR:/workspace/checkpoints" \
    -v "$EXPORT_DIR:/workspace/export" \
    -v "$TRITON_REPO_DIR:/workspace/triton_model_repo" \
    -v "$REPO_ROOT/configs:/workspace/configs" \
    -v "$REPO_ROOT/results:/workspace/results" \
    -v "$REPO_ROOT:/workspace/host_repo" \
    -w /workspace/recsys-examples/examples/hstu \
    "$IMAGE_NAME" "$@"
}

docker_gpu_run_hostnet() {
  # Same as docker_gpu_run but with --network host, for clients that need to
  # reach a Triton server bound to localhost in another --network host container.
  docker run --rm -i --gpus all --network host --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    --shm-size=32g \
    -v "$RECSYS_DIR:/workspace/recsys-examples" \
    -v "$DATA_DIR:/workspace/data" \
    -v "$CKPT_DIR:/workspace/checkpoints" \
    -v "$EXPORT_DIR:/workspace/export" \
    -v "$TRITON_REPO_DIR:/workspace/triton_model_repo" \
    -v "$REPO_ROOT/configs:/workspace/configs" \
    -v "$REPO_ROOT/results:/workspace/results" \
    -v "$REPO_ROOT:/workspace/host_repo" \
    -w /workspace/recsys-examples/examples/hstu \
    "$IMAGE_NAME" "$@"
}

docker_gpu_daemon() {
  # Start (or reuse) a persistent named container for interactive / long steps.
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker run -d --name "$CONTAINER_NAME" --gpus all --ipc=host \
      --ulimit memlock=-1 --ulimit stack=67108864 --shm-size=32g \
      --network host \
      -v "$RECSYS_DIR:/workspace/recsys-examples" \
      -v "$DATA_DIR:/workspace/data" \
      -v "$CKPT_DIR:/workspace/checkpoints" \
      -v "$EXPORT_DIR:/workspace/export" \
      -v "$TRITON_REPO_DIR:/workspace/triton_model_repo" \
      -v "$REPO_ROOT/configs:/workspace/configs" \
      -v "$REPO_ROOT/results:/workspace/results" \
      -v "$REPO_ROOT:/workspace/host_repo" \
      -w /workspace/recsys-examples/examples/hstu \
      "$IMAGE_NAME" sleep infinity >/dev/null
    ok "Started persistent container '$CONTAINER_NAME'"
  fi
}

docker_exec() {
  docker exec "$CONTAINER_NAME" bash -lc "$*"
}

render_effective_gin() {
  # Concatenate one or more gin files (+ optional raw override lines) into a
  # single self-contained file. Within a single gin file, a binding set later
  # in the file wins over one set earlier, so this sidesteps any uncertainty
  # about whether the upstream entrypoints accept `--gin-config-file` more
  # than once on the CLI. Usage:
  #   render_effective_gin OUT_FILE base.gin tuned.gin -- "TrainerArgs.foo = 1"
  local out="$1"; shift
  : > "$out"
  local extra_mode=0
  for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then extra_mode=1; continue; fi
    if [[ "$extra_mode" -eq 1 ]]; then
      echo "$arg" >> "$out"
    else
      { echo "# ---- included from $(basename "$arg") ----"; cat "$arg"; echo; } >> "$out"
    fi
  done
}

gin_config_path() {
  # Map (model size, dataset) -> gin config file shipped in configs/.
  local size="$1" dataset="$2"
  echo "/workspace/configs/hstu_${size}_ranking_${dataset//-/}.gin"
}

elapsed_human() {
  local secs="$1"
  printf '%dh%02dm%02ds' $((secs/3600)) $(((secs%3600)/60)) $((secs%60))
}
