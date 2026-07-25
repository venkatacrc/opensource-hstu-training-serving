# Runbook

Step-by-step operator guide: what to run, how long each phase takes, what
"normal" output looks like, and how to unstick each phase if something goes
wrong. See `docs/ARCHITECTURE.md` for the *why*; this doc is the *how*.

## Getting this repo onto the GPU box

This repo was authored on a dev machine; everything below assumes it's
checked out **on the 8x GPU box itself** (all phases need local GPU/Docker
access, so there's no remote-orchestration mode). Two ways to get it there:

**Option A -- push to GitHub, clone on the box (preferred, keeps history):**

```bash
# On the dev machine (this repo already has 'origin' configured):
git push origin main

# On the GPU box:
git clone https://github.com/venkatacrc/opensource-hstu-training-serving.git hstu-bench
cd hstu-bench
```

**Option B -- direct copy via rsync (no push required, e.g. private repo /
box has no GitHub credentials):**

```bash
# From the dev machine:
rsync -avz --exclude='.git' --exclude='workdir' --exclude='state' --exclude='results' \
  /path/to/opensource-hstu-training-serving/ \
  <user>@<gpu-box-host>:~/hstu-bench/

ssh <user>@<gpu-box-host>
cd ~/hstu-bench
```

Either way, once the repo is on the box, run it inside `tmux`/`screen` (or
`nohup`) since the full pipeline spans hours and an SSH drop would otherwise
kill the orchestrator shell (the training/build containers themselves are
already launched detached and survive on their own -- this is just so
`run_all.sh`'s own foreground polling/sequencing survives too):

```bash
cd hstu-bench
chmod +x scripts/*.sh          # in case file modes didn't survive the transfer

tmux new -s hstu-bench                 # or: screen -S hstu-bench
./scripts/run_all.sh 2>&1 | tee -a state/run_all.log
# detach with Ctrl-b d (tmux) / Ctrl-a d (screen); re-attach any time with
# `tmux attach -t hstu-bench` / `screen -r hstu-bench`.

# --- or, without tmux/screen: ---
nohup ./scripts/run_all.sh > state/run_all.log 2>&1 &
disown
```

If the shell running `run_all.sh` does get killed anyway (e.g. box reboot),
nothing is lost -- just re-run `./scripts/run_all.sh` (in a new tmux/nohup
session) and it resumes: already-`.done` phases are skipped instantly, and
`05_train.sh` re-attaches to its still-running detached container if one
exists.

Everything below (prerequisites, phase-by-phase detail, troubleshooting)
applies from this point regardless of which option you used to get here.

## Prerequisites

- Linux box with 8x NVIDIA B200 (or similar Blackwell/Hopper) GPUs, ~180GB+
  HBM/GPU, ≥2TB system RAM, ≥500GB free disk.
- Outbound internet access to: `github.com`, `raw.githubusercontent.com`,
  `files.grouplens.org`, `zenodo.org`, `pypi.org`, `nvcr.io`,
  `registry-1.docker.io`.
- sudo access (to install Docker + the NVIDIA Container Toolkit if not already
  present).

## One command to run everything

Once the repo is on the box (see above), the fully unattended path is just:

```bash
./scripts/run_all.sh
```

Everything below is for monitoring progress, running phases individually,
tuning the config, and troubleshooting.

Check progress at any time, from another terminal (or a second tmux window),
without disturbing the run:

```bash
./scripts/run_all.sh --status
./scripts/05_train.sh --status          # primary (ml-20m) training container
TRAIN_DATASET=kuairand-1k ./scripts/05_train.sh --status   # secondary training container
tail -f state/05_train_ml20m_*.log
```

## Configuration

All tunables live in `config.env` with sensible defaults, and can be
overridden by exporting them before running any script, e.g.:

```bash
TRAIN_BUDGET_HOURS=6 MODEL_SIZE=10b ./scripts/run_all.sh
```

Key ones:

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_SIZE` | `8b` | `8b` or `10b` (see `configs/hstu_10b_ranking_ml20m.gin`) |
| `PRIMARY_DATASET` | `ml-20m` | Main training corpus |
| `SECONDARY_DATASET` | `kuairand-1k` | Serving-benchmark corpus (TP=1) |
| `TRAIN_BUDGET_HOURS` | `12` | Wall-clock cap for the primary training run |
| `SECONDARY_TRAIN_BUDGET_HOURS` | `2` | Wall-clock cap for the secondary training run |
| `BENCH_BATCH_SIZES` | `1,2,4,8,16,32,64` | Serving benchmark sweep |
| `NUM_GPUS` | `8` | Adjust if running on a different node shape |

## Phase-by-phase

### Phase 0 -- `scripts/00_preflight.sh` (~1-5 min)

Checks GPUs/driver/topology, installs Docker + NVIDIA Container Toolkit if
missing, smoke-tests `docker run --gpus all`, checks disk space and internet
reachability to every host later phases need.

**Troubleshooting:**
- *"nvidia-smi not found"*: driver install is out of scope for this pipeline;
  install the NVIDIA driver first.
- *Docker group membership*: if Docker was just installed, run `newgrp
  docker` (or log out/in) before continuing, or re-run with `sudo`.
- *GPU smoke test fails*: check `/tmp/preflight_gpu_smoke.log`; usually means
  the NVIDIA Container Toolkit runtime isn't registered with Docker yet --
  the script attempts `nvidia-ctk runtime configure --runtime=docker &&
  systemctl restart docker` automatically, but a re-login may still be needed.

### Phase 1 -- `scripts/01_build_env.sh` (~1.5-3h, background)

Clones `NVIDIA/recsys-examples` @ the pinned tag and builds the Docker image
from `docker/Dockerfile`. This is the single longest fixed cost in the whole
pipeline (FBGEMM HSTU kernel compile alone is ~55min upstream, plus a
from-source Triton Server build). `run_all.sh` launches this in the
background and overlaps it with Phase 2.

**Troubleshooting:**
- Full build log: `state/run_all_01_build_env.log` (via `run_all.sh`) or
  `state/01_build_env_docker_build.log` (direct invocation).
- Transient network failures mid-build (apt/pip): just re-run
  `./scripts/01_build_env.sh` -- Docker layer caching means it resumes near
  where it left off.
- Submodule issues: `cd workdir/recsys-examples && git submodule update
  --init --recursive`.
- *Known upstream bug, patched automatically*: recsys-examples@v26.06.01's
  `docker/Dockerfile` (`base_triton` stage) pins
  `triton-inference-server/pytorch_backend` via `git fetch --depth 1 origin
  r26.07 && git checkout --detach ceeecb7`. That only works while `ceeecb7`
  happens to be the exact tip of the `r26.07` branch; as soon as upstream
  pushes any further commit there, `ceeecb7` falls outside the depth-1
  shallow window and the build fails with `fatal: git checkout: --detach
  does not take a path argument 'ceeecb7'`. `scripts/01_build_env.sh`
  detects and patches this automatically (fetches the exact pinned SHA
  directly instead of by branch name, which GitHub supports and which is
  immune to the branch moving further) before invoking `docker build` --
  no action needed. If you still hit this exact error, check that the patch
  step's log line ("Patching known-broken pytorch_backend@ceeecb7 pin...")
  actually ran and succeeded; if the upstream Dockerfile's layout has
  changed since v26.06.01, the `sed` pattern may need updating in
  `scripts/01_build_env.sh`'s `patch_dockerfile_known_upstream_issues`.

### Phase 2/3 -- `scripts/02_fetch_datasets.sh`, `scripts/03_preprocess_data.sh` (~5-20 min)

Downloads ml-20m (files.grouplens.org) and KuaiRand-1K (zenodo.org) --  both
direct, unauthenticated downloads, no manual click-through -- then
preprocesses them. Phase 2 tries a lightweight host-side Python venv first
(the preprocessor is pure numpy/pandas, no GPU/torch needed) so it can
overlap with Phase 1's Docker build; Phase 3 re-runs preprocessing inside the
finished container to guarantee the on-disk CSVs match the exact
numpy/pandas versions training will read with.

**Troubleshooting:**
- If `python3 -m venv` isn't available on the host, Phase 2 automatically
  falls back to waiting for the Docker image and running inside the
  container -- no action needed, just slower (loses the overlap with Phase 1).
- Dataset stats (row/user counts) land in `results/dataset_stats.json`.

### Phase 4 -- `scripts/04_size_model.sh` (~10-20 min, run twice)

Projects per-GPU memory across a batch-size sweep, picks the largest batch
size under a safety threshold, writes `configs/*.tuned.gin`, then validates
with a real ~50-iteration smoke training run. Run once per dataset:

```bash
SIZE_DATASET=ml-20m ./scripts/04_size_model.sh          # primary
SIZE_DATASET=kuairand-1k ./scripts/04_size_model.sh      # secondary
```

**Troubleshooting:**
- *Smoke test OOMs*: the script prints the next-smaller candidate batch size
  and how to retry. Edit `configs/hstu_*_ranking_*.tuned.gin` by hand if you
  want a specific value instead of the auto-selected one, then `rm
  state/04_size_model_<slug>.done` and re-run.
- *Memory projection table looks off*: remember the upstream estimator
  reports **whole-model** (non-TP-sharded) memory; this script manually
  divides the dense weights/optimizer/gradients by the TP degree declared in
  the base gin file. See the script's header comment and
  `docs/ARCHITECTURE.md` section 5.
- Full smoke-test log: `state/04_size_model_<slug>_smoke_train.log`.

### Phase 5 -- `scripts/05_train.sh` (up to `TRAIN_BUDGET_HOURS`, run twice)

The main event. Launches a **detached** container (survives SSH drops),
polls it, enforces the wall-clock budget by stopping the container once the
deadline passes (checkpointing means a time-boxed stop always leaves a usable
checkpoint). Resumable automatically -- if a checkpoint already exists for a
given (model size, dataset), training resumes from it.

```bash
TRAIN_DATASET=ml-20m TRAIN_HOURS=$TRAIN_BUDGET_HOURS ./scripts/05_train.sh
TRAIN_DATASET=kuairand-1k TRAIN_HOURS=$SECONDARY_TRAIN_BUDGET_HOURS ./scripts/05_train.sh
```

Safe to Ctrl-C -- the container keeps training in the background;
`./scripts/05_train.sh --status` any time, or re-run the same command to
resume waiting.

**Troubleshooting:**
- *No MFU/TFLOPS lines found in the log*: the exact log line format can
  differ by upstream version; check `TrainerArgs.profile = True` is set (it
  is, by default, in both base gin configs) and inspect
  `state/05_train_<slug>_*.log` directly. `scripts/08_generate_report.py`
  degrades gracefully (reports "not available") if it can't parse this.
- *Container crashed early*: `docker logs <container-name>` (name printed at
  launch, pattern `hstu-bench-train-hstu_<size>_<dataset>`); most likely an
  OOM that the Phase 4 smoke test didn't catch (data-dependent sequence
  length outliers) -- lower `train_batch_size` in the `.tuned.gin` file and
  re-run.
- *Want to stop early on purpose*: `docker stop <container-name>` -- the
  script treats this the same as a budget timeout (checks for a checkpoint,
  marks the phase done if one exists).

### Phase 6 -- `scripts/06_export_checkpoint.sh` (~15-45 min, best-effort for Tier 3)

Prepares Tier 2 (Triton Python backend prerequisites) and attempts Tier 3
(torch.export + AOTInductor C++ + a second Triton runtime image). Tier 3
failures are recorded, not fatal -- check
`state/06_export_checkpoint_tier3_available.txt` (`0` or `1`).

#### Tier 2 troubleshooting

Requires the secondary checkpoint's latest
`iterNNNN/` directory to contain `dynamicemb_module/` with
`user_id_emb_*`/`video_id_emb_*` files; if missing, something about the
DynamicEmb embedding path during the secondary training run didn't produce
the expected checkpoint layout -- inspect
`workdir/checkpoints/hstu_<size>_kuairand1k/iterNNNN/` directly.

#### Tier 3 troubleshooting

This is the deepest, most novel part of the whole
pipeline (custom C++ replay binary, FlexKV KV-cache server, AOTI static-shape
compilation). Check, in order:
1. `state/06_export_checkpoint_tier3_export.log` -- the `torch.export`/AOTI
   compile + native C++ replay validation step.
2. `state/06_export_checkpoint_tritonserver_image_build.log` -- the second,
   lightweight Docker image build (`docker/Dockerfile.tritonserver`).
3. If export succeeds but the C++ replay binary or the built-in libraries
   (`triton_libs/...`) are missing, that likely means Phase 1's Docker build
   didn't complete the AOTI/C++ artifact build stage -- check
   `state/01_build_env_docker_build.log` for that stage's output.

If Tier 3 can't be made to work in your environment, that's fine --
`docs/RESULTS.md` will simply report it as unavailable and the report and
recommendation are generated from Tiers 0-2.

### Phase 7 -- `scripts/07_serving_bench.sh` (~20-60 min)

Sweeps `BENCH_BATCH_SIZES` across every available tier, writing
`results/serving_bench/<tier>_bs<N>.json`. Tiers 0/1 run directly (no
server); Tier 2/3 launch a Triton server container, wait for
`/v2/health/ready`, run the sweep, then tear the server down.

#### Serving benchmark troubleshooting
- *A tier's Triton server never becomes ready*: check
  `state/07_serving_bench_tier{2,3}_server_full.log`. Common causes: the
  checkpoint's `HSTU_CHECKPOINT_DIR` parameter in `config.pbtxt` didn't get
  patched correctly (inspect the staged file directly in
  `workdir/recsys-examples/examples/hstu/inference/triton/hstu_model/config.pbtxt`),
  or `LD_PRELOAD`/`LD_LIBRARY_PATH` mismatches for the custom ops libraries.
- *One batch size fails but others succeed*: expected and non-fatal -- the
  script continues the sweep and logs a per-batch-size warning; check
  `state/07_serving_bench_<tier>_bs<N>.log` for that specific failure.
- Re-running only the benchmark (e.g. after fixing a client bug) without
  redoing Phases 1-6: `rm state/07_serving_bench.done && ./scripts/07_serving_bench.sh`.

### Phase 8 -- `scripts/08_generate_report.py` (~1-2 min)

Aggregates every JSON/log artifact into `docs/RESULTS.md` plus latency/
throughput charts under `results/charts/`. Pure Python (pandas + matplotlib),
no GPU/torch needed -- `run_all.sh` sets up a tiny venv for it automatically.
Re-run any time after any phase to refresh the report with partial results:

```bash
python3 scripts/08_generate_report.py
```

## Full reset

To start completely over (re-downloads, re-builds, re-trains everything):

```bash
rm -rf state/*.done workdir/
```

(`workdir/` holds the recsys-examples checkout, datasets, and checkpoints --
everything gitignored and regenerable. `results/` and `docs/RESULTS.md` are
left alone; delete them too if you want a fully clean slate.)

To re-run a single phase, delete just its marker: `rm
state/<phase>.done && ./scripts/<phase>.sh`.
