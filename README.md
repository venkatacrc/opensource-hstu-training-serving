# HSTU 7-10B Training + Serving Benchmark

A self-executing pipeline that trains an ~8B-parameter
[HSTU](https://arxiv.org/abs/2402.17152) (Hierarchical Sequential Transduction
Unit) generative-recommender model from scratch on public data, then
benchmarks four different serving stacks for latency/throughput -- all on a
single 8x GPU node (built and tested for 8x NVIDIA B200), driven by one
top-level command.

Built on [NVIDIA/recsys-examples](https://github.com/NVIDIA/recsys-examples).
See `docs/ARCHITECTURE.md` for the design rationale (model sizing math,
parallelism strategy, why two datasets) and `docs/RUNBOOK.md` for detailed
operator instructions and troubleshooting.

## Quickstart

```bash
git clone https://github.com/venkatacrc/opensource-hstu-training-serving.git hstu-bench
cd hstu-bench
./scripts/run_all.sh
```

(See `docs/RUNBOOK.md` for the recommended `tmux`/`nohup` wrapping, since the
full run spans hours, and for a copy/rsync alternative if the box can't reach
GitHub directly.)

That single command:

1. Checks/installs prerequisites (Docker, NVIDIA Container Toolkit).
2. Builds the training/serving environment from NVIDIA's Dockerfile
   (~1.5-3h, backgrounded).
3. Downloads and preprocesses MovieLens-20M and KuaiRand-1K (public, no auth).
4. Projects memory usage and picks a batch size, validated with a smoke test.
5. Trains an HSTU-8B model on ml-20m (TP=8, time-boxed, checkpointed,
   resumable) and a second one on kuairand-1k (TP=1, for the serving
   benchmark -- see `docs/ARCHITECTURE.md` §4 for why two runs).
6. Exports the second checkpoint for serving (Triton Python backend +
   best-effort AOTInductor C++/Triton).
7. Benchmarks all 4 serving tiers across a batch-size sweep.
8. Generates `docs/RESULTS.md` with tables, charts, and a recommendation.

Every phase is idempotent (state markers in `state/`), so interruptions
(Ctrl-C, SSH drop, reboot) are safe -- just re-run `./scripts/run_all.sh` and
it resumes. Check progress any time with `./scripts/run_all.sh --status`.

## Repo layout

```
config.env              Central tunables (model size, datasets, time budgets, batch sweep)
lib/common.sh            Shared bash helpers (logging, idempotency, docker wrappers)
configs/*.gin            Gin-config model/training definitions (HSTU-8B/10B x ml-20m/kuairand-1k)
scripts/00-08_*.sh|py     One script per pipeline phase (see docs/RUNBOOK.md)
scripts/run_all.sh        Top-level orchestrator
scripts/lib/*.py          Custom latency/throughput benchmark clients for the 4 serving tiers
docs/ARCHITECTURE.md      Design rationale, model sizing math, parallelism strategy
docs/RUNBOOK.md           Step-by-step operator guide + troubleshooting
docs/RESULTS.md           Auto-generated benchmark report (after running the pipeline)
state/                    Phase completion markers + logs (gitignored except markers)
results/                  Benchmark JSON, dataset stats, charts (gitignored by default)
workdir/                  recsys-examples checkout, datasets, checkpoints (gitignored, regenerable)
```

## Scope note

This is an **infrastructure/performance benchmark** -- it demonstrates
training and serving a large-scale HSTU model within a bounded time/hardware
budget, not a claim of state-of-the-art recommendation quality (though AUC is
tracked as a sanity signal). See `docs/ARCHITECTURE.md` for the full
reasoning behind every major design decision.
