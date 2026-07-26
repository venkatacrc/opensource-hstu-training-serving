#!/usr/bin/env bash
# Phase 0: verify the box is fit for purpose and install the light host-level
# tooling (Docker, NVIDIA Container Toolkit, git) needed by later phases.
# Does NOT touch the GPU workload itself. Safe to re-run.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/lib/common.sh"

PHASE=00_preflight
log "=== Phase 0: Preflight checks ==="

FAIL=0
note() { log "  - $*"; }
fail() { printf '\033[1;31m  ! %s\033[0m\n' "$*" >&2; FAIL=1; }

# --- 1. GPUs ------------------------------------------------------------------
log "Checking GPUs..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi not found. NVIDIA driver must already be installed (out of scope for this pipeline)."
else
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
  DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n1)
  note "Found $GPU_COUNT GPU(s): $GPU_NAME, driver $DRIVER_VER, ${GPU_MEM} per GPU"
  if [[ "$GPU_COUNT" -lt "$NUM_GPUS" ]]; then
    fail "Expected NUM_GPUS=$NUM_GPUS GPUs but found $GPU_COUNT. Adjust NUM_GPUS in config.env if intentional."
  fi
  if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "B200\|GB200"; then
    warn "GPU is not a B200/GB200 — MFU comparisons against the documented 2500 TFLOPS/GPU BF16 reference will not apply."
  fi
  log "GPU topology (NVLink expected across all $NUM_GPUS for tensor-parallel training):"
  nvidia-smi topo -m 2>/dev/null | tee "$STATE_DIR/gpu_topology.txt" || warn "nvidia-smi topo -m failed"
fi

# --- 2. git ---------------------------------------------------------------------
log "Checking git..."
if ! command -v git >/dev/null 2>&1; then
  note "git not found, installing..."
  sudo apt-get update -y && sudo apt-get install -y git || fail "Could not install git"
else
  note "git present: $(git --version)"
fi

# --- 3. Docker -------------------------------------------------------------------
log "Checking Docker..."
if ! command -v docker >/dev/null 2>&1; then
  note "Docker not found, installing via get.docker.com convenience script..."
  curl -fsSL https://get.docker.com | sudo sh || fail "Docker install failed"
  sudo usermod -aG docker "$USER" || true
  warn "Added $USER to the docker group — you may need to log out/in (or run 'newgrp docker') for group membership to take effect in this shell."
else
  note "docker present: $(docker --version)"
fi

if ! docker info >/dev/null 2>&1; then
  fail "docker daemon not reachable by current user ($USER). Fix group membership instead of reaching for sudo: run 'newgrp docker' (or log out/in) then re-run this script. Do NOT run the pipeline scripts themselves with sudo even as a one-off -- doing so leaves root-owned files under state/ and results/ that silently break later non-sudo runs with a confusing 'Permission denied' (see lib/common.sh's ownership check, and docs/RUNBOOK.md, if you hit that)."
fi

# --- 4. NVIDIA Container Toolkit --------------------------------------------------
log "Checking NVIDIA Container Toolkit (docker --gpus all support)..."
if docker info 2>/dev/null | grep -qi nvidia; then
  note "nvidia runtime already registered with Docker."
else
  note "Installing NVIDIA Container Toolkit..."
  distribution=$(. /etc/os-release; echo "$ID$VERSION_ID")
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y nvidia-container-toolkit || fail "nvidia-container-toolkit install failed"
  sudo nvidia-ctk runtime configure --runtime=docker || fail "nvidia-ctk runtime configure failed"
  sudo systemctl restart docker || fail "docker restart failed"
fi

log "Smoke-testing 'docker run --gpus all' (using a tiny ubuntu image -- the toolkit hook injects nvidia-smi + driver libs regardless of base image, so we don't need to pull the ~10GB PyTorch NGC image just for this check)..."
if docker run --rm --gpus all ubuntu:24.04 nvidia-smi >/tmp/preflight_gpu_smoke.log 2>&1; then
  ok "GPU-enabled container smoke test passed."
else
  fail "GPU-enabled container smoke test failed. See /tmp/preflight_gpu_smoke.log ($(head -n1 /tmp/preflight_gpu_smoke.log 2>/dev/null))."
fi

# --- 5. Disk space ------------------------------------------------------------------
log "Checking disk space at $WORK_DIR..."
AVAIL_GB=$(df -BG --output=avail "$(dirname "$WORK_DIR")" 2>/dev/null | tail -n1 | tr -dc '0-9')
note "Available: ${AVAIL_GB:-unknown}GB (recommend >= 500GB for docker layers + datasets + checkpoints + export artifacts)"
if [[ -n "${AVAIL_GB:-}" ]] && [[ "$AVAIL_GB" -lt 500 ]]; then
  warn "Less than 500GB free — checkpoint retention in 05_train.sh is capped, but keep an eye on disk during training."
fi

# --- 6. Internet reachability -------------------------------------------------------
log "Checking internet reachability for required hosts..."
for host in github.com raw.githubusercontent.com files.grouplens.org zenodo.org pypi.org nvcr.io registry-1.docker.io; do
  if curl -sSf --max-time 8 -o /dev/null "https://$host"; then
    note "$host reachable"
  else
    warn "$host NOT reachable — later phases that depend on it will fail. Check firewall/proxy settings."
  fi
done

# --- 7. Free RAM (host, for DynamicEmb CPU-side tables + dataloaders) --------------
FREE_RAM_GB=$(free -g | awk '/^Mem:/{print $7}')
note "Available host RAM: ${FREE_RAM_GB:-unknown}GB"

if [[ "$FAIL" -ne 0 ]]; then
  die "Preflight failed — fix the items marked with '!' above and re-run scripts/00_preflight.sh."
fi

phase_done_mark "$PHASE"
ok "Preflight passed. Proceed to scripts/01_build_env.sh (or scripts/run_all.sh)."
