#!/usr/bin/env python3
"""Phase 8: aggregate every JSON/log artifact produced by scripts/00-07 into a
single architect-facing docs/RESULTS.md, with latency/throughput charts for
the 4 serving tiers and a training-memory/MFU summary.

Designed to be forgiving: any missing input (a skipped tier, a training log
whose exact MFU line format didn't match) is reported as "not available" in
the generated doc rather than raising, since by this point in the pipeline
partial results are still valuable to see.

Usage: python3 scripts/08_generate_report.py   (no args; paths are derived
from the repo root, same layout lib/common.sh's docker mounts use)

Dependencies: pandas, matplotlib (only). No torch/CUDA needed -- this can run
on the host directly; scripts/run_all.sh sets up a tiny venv for it the same
way scripts/02_fetch_datasets.sh does for the dataset preprocessor.
"""
import glob
import json
import os
import re
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = REPO_ROOT / "state"
RESULTS_DIR = REPO_ROOT / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
DOCS_DIR = REPO_ROOT / "docs"


def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_env_defaults():
    """Best-effort parse of config.env for display purposes only."""
    env = {}
    cfg = REPO_ROOT / "config.env"
    if not cfg.exists():
        return env
    pat = re.compile(r'^\s*:\s*"\$\{(\w+):=([^}]*)\}"')
    for line in cfg.read_text().splitlines():
        m = pat.match(line)
        if m:
            env[m.group(1)] = os.environ.get(m.group(1), m.group(2))
    return env


def collect_serving_bench():
    tiers = {}
    for path in sorted(glob.glob(str(RESULTS_DIR / "serving_bench" / "*.json"))):
        d = load_json_safe(path)
        if not d:
            continue
        tiers.setdefault(d["tier"], []).append(d)
    for t in tiers:
        tiers[t].sort(key=lambda r: r.get("batch_size", 0))
    return tiers


def collect_dataset_stats():
    return load_json_safe(RESULTS_DIR / "dataset_stats.json") or {}


def collect_build_duration():
    p = STATE_DIR / "01_build_env_duration_seconds.txt"
    if p.exists():
        try:
            secs = int(p.read_text().strip())
            return f"{secs // 3600}h{(secs % 3600) // 60}m{secs % 60}s"
        except Exception:
            return None
    return None


def collect_mem_selection(dataset_slug):
    p = STATE_DIR / f"04_size_model_{dataset_slug}_selection.log"
    if not p.exists():
        return None
    return p.read_text()


def parse_mfu_lines(text):
    """Best-effort extraction of MFU/TFLOPS numbers from free-form training
    log text. Upstream log line formats can vary by version, so this tries a
    handful of common patterns and returns whatever it finds; callers must
    tolerate an empty result."""
    if not text:
        return []
    patterns = [
        r"MFU[:\s=]+([\d.]+)\s*%?",
        r"mfu[:\s=]+([\d.]+)",
        r"TFLOPS[:\s=/a-zA-Z]*?([\d.]+)",
        r"tflops[:\s=/a-zA-Z]*?([\d.]+)",
    ]
    values = []
    for pat in patterns:
        for m in re.finditer(pat, text):
            try:
                values.append(float(m.group(1)))
            except ValueError:
                pass
        if values:
            break
    return values


def collect_training_summary(dataset_slug, run_name):
    log_paths = sorted(glob.glob(str(STATE_DIR / f"05_train_{dataset_slug}_{run_name}.log")))
    mfu_tail_paths = sorted(glob.glob(str(STATE_DIR / f"05_train_{dataset_slug}_{run_name}_mfu_tail.txt")))
    ckpt_list_paths = sorted(glob.glob(str(STATE_DIR / f"05_train_{dataset_slug}_{run_name}_checkpoints.txt")))

    text = ""
    for p in mfu_tail_paths or log_paths:
        text += Path(p).read_text(errors="ignore")
    values = parse_mfu_lines(text)

    checkpoints = None
    if ckpt_list_paths:
        checkpoints = Path(ckpt_list_paths[0]).read_text(errors="ignore")

    return {
        "mfu_or_tflops_values": values,
        "mfu_summary": {
            "count": len(values),
            "mean": statistics.mean(values) if values else None,
            "max": max(values) if values else None,
        },
        "checkpoints_listing": checkpoints,
        "log_tail_available": bool(mfu_tail_paths or log_paths),
    }


def make_charts(tiers):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    chart_paths = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return chart_paths

    def tier_series(tier_records, key_path):
        xs, ys = [], []
        for r in tier_records:
            v = r
            ok = True
            for k in key_path:
                if v is None or k not in v:
                    ok = False
                    break
                v = v[k]
            if ok and v is not None:
                xs.append(r["batch_size"])
                ys.append(v)
        return xs, ys

    # Latency (p50) vs batch size, one line per tier.
    plt.figure(figsize=(7, 5))
    any_plotted = False
    for tier, records in sorted(tiers.items()):
        xs, ys = tier_series(records, ["latency_ms", "p50"])
        if xs:
            marker = "o" if len(xs) > 1 else "x"
            plt.plot(xs, ys, marker=marker, label=tier)
            any_plotted = True
    if any_plotted:
        plt.xlabel("Batch size")
        plt.ylabel("p50 latency (ms)")
        plt.title("Serving latency (p50) vs batch size, by tier")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = CHARTS_DIR / "latency_vs_batch_size.png"
        plt.savefig(out, dpi=140)
        chart_paths["latency"] = out
    plt.close()

    # Throughput (items/sec) vs batch size, one line per tier.
    plt.figure(figsize=(7, 5))
    any_plotted = False
    for tier, records in sorted(tiers.items()):
        xs, ys = tier_series(records, ["throughput", "items_per_sec"])
        if xs:
            marker = "o" if len(xs) > 1 else "x"
            plt.plot(xs, ys, marker=marker, label=tier)
            any_plotted = True
    if any_plotted:
        plt.xlabel("Batch size")
        plt.ylabel("Throughput (items/sec)")
        plt.title("Serving throughput vs batch size, by tier")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = CHARTS_DIR / "throughput_vs_batch_size.png"
        plt.savefig(out, dpi=140)
        chart_paths["throughput"] = out
    plt.close()

    return chart_paths


def fmt(v, nd=2, suffix=""):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}{suffix}"
    return f"{v}{suffix}"


def render_serving_table(tiers):
    rows = []
    header = "| Tier | Batch size | p50 (ms) | p90 (ms) | p99 (ms) | items/sec |"
    sep = "|---|---|---|---|---|---|"
    rows.append(header)
    rows.append(sep)
    tier_order = [
        "tier0_naive_eager",
        "tier1_optimized_kvcache",
        "tier2_triton_python",
    ]
    tier3_keys = [t for t in tiers if t.startswith("tier3_")]
    for tier in tier_order + tier3_keys:
        for r in tiers.get(tier, []):
            lat = r.get("latency_ms", {})
            thr = r.get("throughput", {})
            rows.append(
                f"| {tier} | {r.get('batch_size')} | {fmt(lat.get('p50'))} | "
                f"{fmt(lat.get('p90'))} | {fmt(lat.get('p99'))} | {fmt(thr.get('items_per_sec'), 0)} |"
            )
    if len(rows) == 2:
        rows.append("| _(no serving benchmark results found)_ | | | | | |")
    return "\n".join(rows)


def main():
    env = load_env_defaults()
    model_size = env.get("MODEL_SIZE", "8b")
    primary_ds = env.get("PRIMARY_DATASET", "ml-20m")
    secondary_ds = env.get("SECONDARY_DATASET", "kuairand-1k")
    primary_slug = primary_ds.replace("-", "")
    secondary_slug = secondary_ds.replace("-", "")
    primary_run = f"hstu_{model_size}_{primary_slug}"
    secondary_run = f"hstu_{model_size}_{secondary_slug}"

    tiers = collect_serving_bench()
    dataset_stats = collect_dataset_stats()
    build_duration = collect_build_duration()
    mem_primary = collect_mem_selection(primary_slug)
    mem_secondary = collect_mem_selection(secondary_slug)
    train_primary = collect_training_summary(primary_slug, primary_run)
    train_secondary = collect_training_summary(secondary_slug, secondary_run)
    chart_paths = make_charts(tiers)

    tier2_available = (STATE_DIR / "06_export_checkpoint_tier2_available.txt")
    tier3_available = (STATE_DIR / "06_export_checkpoint_tier3_available.txt")
    tier2_ok = tier2_available.exists() and tier2_available.read_text().strip() == "1"
    tier3_ok = tier3_available.exists() and tier3_available.read_text().strip() == "1"

    lines = []
    lines.append(f"# HSTU {model_size.upper()} Training + Serving Benchmark Results")
    lines.append("")
    lines.append(
        "_Auto-generated by `scripts/08_generate_report.py`. This is an "
        "**infrastructure/performance benchmark** -- it demonstrates training "
        "and serving a large-scale HSTU generative-recommender model on public "
        "data within a bounded time/hardware budget, not a claim of "
        "state-of-the-art recommendation quality._"
    )
    lines.append("")

    lines.append("## 1. Environment")
    lines.append("")
    lines.append(f"- Model: HSTU-{model_size.upper()} (dense backbone, see `docs/ARCHITECTURE.md` for the parameter-count derivation)")
    lines.append(f"- Primary training dataset: `{primary_ds}` (TP=8 across the node, {env.get('TRAIN_BUDGET_HOURS', '?')}h budget)")
    lines.append(f"- Secondary/serving-benchmark dataset: `{secondary_ds}` (TP=1, {env.get('SECONDARY_TRAIN_BUDGET_HOURS', '?')}h budget -- see scope note in `configs/hstu_{model_size}_ranking_{secondary_slug}.gin`)")
    lines.append(f"- Docker image build time: {build_duration or 'n/a'}")
    lines.append(f"- Tier 2 (Triton Python backend) available: {'yes' if tier2_ok else 'NO -- see docs/RUNBOOK.md troubleshooting'}")
    lines.append(f"- Tier 3 (Triton + AOTInductor C++) available: {'yes' if tier3_ok else 'NO -- see docs/RUNBOOK.md troubleshooting'}")
    lines.append("")

    if dataset_stats:
        lines.append("## 2. Dataset stats")
        lines.append("")
        lines.append("| Dataset | Train rows | Train users | Processed dir |")
        lines.append("|---|---|---|---|")
        for ds, info in dataset_stats.items():
            lines.append(
                f"| {ds} | {info.get('train_rows', 'n/a')} | {info.get('train_users', 'n/a')} | `{info.get('processed_dir', 'n/a')}` |"
            )
        lines.append("")

    lines.append("## 3. Training")
    lines.append("")
    for label, slug, run, mem, train in [
        ("Primary", primary_slug, primary_run, mem_primary, train_primary),
        ("Secondary", secondary_slug, secondary_run, mem_secondary, train_secondary),
    ]:
        lines.append(f"### {label} run: `{run}`")
        lines.append("")
        if mem:
            lines.append("Per-GPU memory projection (scripts/04_size_model.sh), final selected batch size in **bold** below:")
            lines.append("")
            lines.append("```")
            lines.append(mem.strip())
            lines.append("```")
            lines.append("")
        else:
            lines.append("_Memory sizing log not found._")
            lines.append("")
        mfu = train["mfu_summary"]
        if mfu["count"]:
            lines.append(
                f"- MFU/TFLOPS samples parsed from training log: n={mfu['count']}, "
                f"mean={fmt(mfu['mean'])}, max={fmt(mfu['max'])} "
                f"(units depend on upstream log format -- see raw log for context)"
            )
        else:
            lines.append(
                "- No MFU/TFLOPS values could be parsed from the training log "
                "(the upstream log line format may not match the patterns "
                "`scripts/08_generate_report.py` looks for -- check "
                f"`state/05_train_{slug}_{run}.log` directly)."
            )
        if train["checkpoints_listing"]:
            lines.append("")
            lines.append("Checkpoints produced:")
            lines.append("```")
            lines.append(train["checkpoints_listing"].strip())
            lines.append("```")
        lines.append("")

    lines.append("## 4. Serving benchmark (4 tiers)")
    lines.append("")
    lines.append(
        "Tiers 0-2 are swept across `BENCH_BATCH_SIZES`; Tier 3 (AOTInductor) "
        "is benchmarked at a single batch size because AOTInductor compiles "
        "static shapes (see `scripts/07_serving_bench.sh` header)."
    )
    lines.append("")
    lines.append(render_serving_table(tiers))
    lines.append("")
    if "latency" in chart_paths:
        rel = os.path.relpath(chart_paths["latency"], DOCS_DIR)
        lines.append(f"![Latency vs batch size]({rel})")
        lines.append("")
    if "throughput" in chart_paths:
        rel = os.path.relpath(chart_paths["throughput"], DOCS_DIR)
        lines.append(f"![Throughput vs batch size]({rel})")
        lines.append("")

    lines.append("## 5. Recommendation")
    lines.append("")
    lines.append(
        "_Fill in / auto-summarized guidance:_ naive eager (Tier 0) is the "
        "correctness baseline and is not recommended for production due to no "
        "KV-cache reuse or CUDA graph replay. Tier 1 (KV cache + CUDA graph, "
        "same process) gives the best raw latency for a single-model, "
        "single-tenant deployment with no need for a model-serving control "
        "plane. Tier 2 (Triton Python backend) trades some latency for "
        "Triton's multi-model lifecycle management, dynamic batching, and "
        "metrics -- a reasonable default for most production recsys serving "
        "fleets. Tier 3 (Triton + AOTInductor C++) removes Python-interpreter "
        "overhead from the hot path and is worth the extra export/build "
        "complexity specifically when p99 tail latency at fixed batch size is "
        "the binding constraint (e.g. sub-10ms SLA real-time ranking)."
    )
    lines.append("")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_DIR / "RESULTS.md"
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
