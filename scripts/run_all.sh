#!/usr/bin/env bash
# Top-level orchestrator: runs the entire HSTU train+serve pipeline end to
# end with a single command. Idempotent -- every phase script marks itself
# done in state/*.done, so re-running this after an interruption (Ctrl-C,
# SSH drop, reboot) resumes rather than restarting from scratch.
#
# Usage:
#   ./scripts/run_all.sh                 # run everything (resumes automatically -- each
#                                         # phase script skips itself if already marked done
#                                         # in state/*.done; rm the relevant marker to force a redo)
#   ./scripts/run_all.sh --status        # print phase completion status and exit
#
# Override any config.env value by exporting it first, e.g.:
#   TRAIN_BUDGET_HOURS=6 MODEL_SIZE=10b ./scripts/run_all.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/lib/common.sh"

SCRIPTS_DIR="$REPO_ROOT/scripts"

print_status() {
  log "=== Pipeline status ==="
  local expected=(
    "00_preflight" "01_build_env"
    "02_fetch_datasets" "03_preprocess_data"
    "04_size_model_${PRIMARY_DATASET//-/}" "05_train_${PRIMARY_DATASET//-/}"
    "04_size_model_${SECONDARY_DATASET//-/}" "05_train_${SECONDARY_DATASET//-/}"
    "06_export_checkpoint" "07_serving_bench"
  )
  for base in "${expected[@]}"; do
    local marker="$STATE_DIR/${base}.done"
    if [[ -f "$marker" ]]; then
      echo "  [DONE]    $base  ($(cat "$marker"))"
    else
      echo "  [pending] $base"
    fi
  done
  if [[ -f "$REPO_ROOT/docs/RESULTS.md" ]]; then
    echo "  [DONE]    08_generate_report  (docs/RESULTS.md exists)"
  else
    echo "  [pending] 08_generate_report"
  fi
}

if [[ "${1:-}" == "--status" ]]; then
  print_status
  exit 0
fi

START_TS=$(date +%s)
log "############################################################"
log "# HSTU ${MODEL_SIZE^^} train+serve pipeline -- starting run_all.sh"
log "# Primary dataset: $PRIMARY_DATASET (train budget ${TRAIN_BUDGET_HOURS}h)"
log "# Secondary dataset: $SECONDARY_DATASET (train budget ${SECONDARY_TRAIN_BUDGET_HOURS}h, serving benchmark)"
log "############################################################"

run_phase() {
  local script="$1"; shift
  log ">>> Running $script $*"
  bash "$SCRIPTS_DIR/$script" "$@" || die "$script failed -- see logs above / state/ for details."
}

# --- Phase 0: preflight -------------------------------------------------------
run_phase 00_preflight.sh

# --- Phases 1 & 2/3: env build (slow) and dataset fetch run concurrently -----
BUILD_LOG="$STATE_DIR/run_all_01_build_env.log"
if phase_is_done 01_build_env; then
  ok "01_build_env already done, skipping background launch."
  BUILD_PID=""
else
  log "Launching scripts/01_build_env.sh in the background (this is the long pole, ~1.5-3h)..."
  bash "$SCRIPTS_DIR/01_build_env.sh" > "$BUILD_LOG" 2>&1 &
  BUILD_PID=$!
fi

run_phase 02_fetch_datasets.sh

if [[ -n "$BUILD_PID" ]]; then
  log "Waiting for background Docker build (PID $BUILD_PID) to finish -- tail $BUILD_LOG to watch progress..."
  if ! wait "$BUILD_PID"; then
    die "scripts/01_build_env.sh failed -- see $BUILD_LOG"
  fi
  ok "Docker build finished."
fi

run_phase 03_preprocess_data.sh

# --- Phase 4/5: primary (ml-20m, TP=8) training ------------------------------
SIZE_DATASET="$PRIMARY_DATASET" bash "$SCRIPTS_DIR/04_size_model.sh" \
  || die "04_size_model.sh ($PRIMARY_DATASET) failed."
TRAIN_DATASET="$PRIMARY_DATASET" TRAIN_HOURS="$TRAIN_BUDGET_HOURS" bash "$SCRIPTS_DIR/05_train.sh" \
  || die "05_train.sh ($PRIMARY_DATASET) failed."

# --- Phase 4/5: secondary (kuairand-1k, TP=1) training for the serving bench -
SIZE_DATASET="$SECONDARY_DATASET" bash "$SCRIPTS_DIR/04_size_model.sh" \
  || die "04_size_model.sh ($SECONDARY_DATASET) failed."
TRAIN_DATASET="$SECONDARY_DATASET" TRAIN_HOURS="$SECONDARY_TRAIN_BUDGET_HOURS" bash "$SCRIPTS_DIR/05_train.sh" \
  || die "05_train.sh ($SECONDARY_DATASET) failed."

# --- Phase 6/7: export + serving benchmark -----------------------------------
run_phase 06_export_checkpoint.sh
run_phase 07_serving_bench.sh

# --- Phase 8: report ----------------------------------------------------------
log ">>> Running scripts/08_generate_report.py"
REPORT_VENV="$WORK_DIR/.report_venv"
if command -v python3 >/dev/null 2>&1 && python3 -m venv "$REPORT_VENV" 2>/dev/null; then
  # shellcheck disable=SC1091
  source "$REPORT_VENV/bin/activate"
  pip install --quiet --upgrade pip
  pip install --quiet matplotlib pandas
  python3 "$SCRIPTS_DIR/08_generate_report.py" || die "Report generation failed."
  deactivate
else
  warn "Could not create a host venv for the report generator; trying system python3 directly (charts may be skipped if matplotlib isn't installed)."
  python3 "$SCRIPTS_DIR/08_generate_report.py" || die "Report generation failed."
fi

END_TS=$(date +%s)
TOTAL_SECS=$((END_TS - START_TS))
log "############################################################"
ok "Pipeline complete in $(elapsed_human "$TOTAL_SECS")."
log "See docs/RESULTS.md for the architect-facing summary."
log "############################################################"
