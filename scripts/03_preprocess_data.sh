#!/usr/bin/env bash
# Phase 3: (re-)run preprocessing INSIDE the built container so the on-disk
# CSVs are produced by the exact numpy/pandas versions training will later
# read with, then sanity-check the result and record dataset stats for the
# final report. Cheap to re-run (the raw-archive download is skipped once the
# zip/tar.gz already exists on disk from scripts/02_fetch_datasets.sh).
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/lib/common.sh"

PHASE=03_preprocess_data
phase_skip_if_done "$PHASE"

log "=== Phase 3: Preprocess datasets inside container ($PRIMARY_DATASET, $SECONDARY_DATASET) ==="

docker image inspect "$IMAGE_NAME" >/dev/null 2>&1 || die "Image '$IMAGE_NAME' not found -- run scripts/01_build_env.sh first."

patch_preprocessor_known_upstream_issues() {
  # Known bug: examples/commons/hstu_data_preprocessor.py's
  # preprocess_inference() uses the pandas fillna(method="ffill")/"bfill"
  # calling convention, which was removed in the pandas version installed
  # in this NGC image (pandas 2.x+). Replace with the modern .ffill()/.bfill()
  # equivalents, which are functionally identical.
  local f="$RECSYS_DIR/examples/commons/hstu_data_preprocessor.py"
  [[ -f "$f" ]] || return 0
  if grep -qE 'fillna\(method=' "$f" 2>/dev/null; then
    log "Patching known-broken fillna(method=...) calls in $f (removed pandas API)..."
    sed -i.bak \
      -e 's/\.fillna(method="ffill")/\.ffill()/g' \
      -e "s/\.fillna(method='ffill')/\.ffill()/g" \
      -e 's/\.fillna(method="bfill")/\.bfill()/g' \
      -e "s/\.fillna(method='bfill')/\.bfill()/g" \
      "$f"
    rm -f "$f.bak"
    grep -qE 'fillna\(method=' "$f" && die "Failed to patch $f -- pattern still present after sed."
    ok "Preprocessor patched."
  else
    log "No fillna(method=...) pattern found in $f -- already patched or upstream fixed it."
  fi
}
patch_preprocessor_known_upstream_issues

for ds in "$PRIMARY_DATASET" "$SECONDARY_DATASET"; do
  log "Preprocessing $ds inside container..."
  docker_gpu_run bash -lc "
    cd /workspace/recsys-examples/examples/commons && \
    python3 hstu_data_preprocessor.py --dataset_name $ds --dataset_path /workspace/data --training --inference
  " || die "Preprocessing failed for $ds -- see output above."
done

log "Validating processed outputs and collecting dataset stats..."
STATS_JSON="$REPO_ROOT/results/dataset_stats.json"
docker_gpu_run python3 - "$PRIMARY_DATASET" "$SECONDARY_DATASET" <<'PYEOF' > "$STATS_JSON"
import json, os, sys
import pandas as pd

name_to_prefix = {
    "ml-1m": "ml-1m", "ml-20m": "ml-20m",
    "kuairand-pure": "KuaiRand-Pure", "kuairand-1k": "KuaiRand-1K", "kuairand-27k": "KuaiRand-27K",
}

out = {}
for ds in sys.argv[1:]:
    prefix = name_to_prefix.get(ds, ds)
    base = f"/workspace/data/{prefix}"
    train_csv = os.path.join(base, "processed_seqs.csv")
    entry = {"dataset": ds, "processed_dir": base}
    if os.path.isfile(train_csv):
        df = pd.read_csv(train_csv)
        entry["train_rows"] = int(len(df))
        entry["train_users"] = int(df["user_id"].nunique()) if "user_id" in df.columns else None
        entry["train_csv_bytes"] = os.path.getsize(train_csv)
    else:
        entry["error"] = f"missing {train_csv}"
    out[ds] = entry

print(json.dumps(out, indent=2))
PYEOF

if [[ -s "$STATS_JSON" ]]; then
  ok "Dataset stats written to $STATS_JSON"
  cat "$STATS_JSON"
else
  warn "Dataset stats collection produced no output -- check the docker_gpu_run invocation above."
fi

phase_done_mark "$PHASE"
ok "Preprocessing complete. Proceed to scripts/04_size_model.sh."
