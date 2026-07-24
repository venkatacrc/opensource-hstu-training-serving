#!/usr/bin/env python3
"""Latency/throughput micro-benchmark for serving Tiers 0 and 1.

Tier 0 = naive eager (--disable_kvcache): forward_nokvcache, no CUDA graph.
Tier 1 = optimized (KV cache + CUDA graph, the default in the upstream model
         builder): forward_with_kvcache.

This intentionally reuses the exact model-building and batch-construction
helpers from the upstream `inference/inference_gr_ranking.py`
(`get_inference_dataset_and_embedding_configs`, `get_inference_hstu_model`,
`InferenceDataset`) rather than reimplementing HSTU's jagged-tensor batch
format from scratch -- that machinery is intricate and dataset-specific, and
the upstream simulate/eval entrypoints only report an aggregate
total-time-over-whole-dataset number, not the per-request latency
percentiles an architect-facing benchmark needs. This script adds that
percentile instrumentation around the SAME forward calls the upstream
`run_ranking_gr_simulate` function makes, run for a fixed number of iterations
at a fixed batch size instead of once over the whole dataset.

If the installed recsys-examples version has diverged from what this was
written against, the imports/function signatures below are the first thing to
check (see docs/RUNBOOK.md#serving-benchmark-troubleshooting).
"""
import argparse
import json
import math
import statistics
import sys
import time

import gin
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gin_config_file", required=True)
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--disable_kvcache", action="store_true")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tier", required=True, help="label written into the output JSON")
    ap.add_argument("--output_json", required=True)
    ap.add_argument(
        "--hstu_dir",
        default="/workspace/recsys-examples/examples/hstu",
        help="Path containing inference/inference_gr_ranking.py",
    )
    args = ap.parse_args()

    sys.path.insert(0, args.hstu_dir)
    sys.path.insert(0, f"{args.hstu_dir}/inference")
    sys.path.insert(0, f"{args.hstu_dir}/inference/model")

    from commons.datasets import get_data_loader
    from commons.datasets.inference_dataset import InferenceDataset
    from commons.hstu_data_preprocessor import get_common_preprocessors
    from inference_gr_ranking import (
        get_inference_dataset_and_embedding_configs,
        get_inference_hstu_model,
    )

    gin.parse_config_file(args.gin_config_file)

    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(False)
    dataproc = get_common_preprocessors(dataset_args.dataset_path or "")[
        dataset_args.dataset_name
    ]
    num_contextual_features = len(dataproc._contextual_feature_names)
    total_max_seqlen = (
        dataset_args.max_history_seqlen + dataset_args.max_num_candidates
    ) * 2 + num_contextual_features

    use_kvcache = not args.disable_kvcache
    with torch.inference_mode():
        model = get_inference_hstu_model(
            emb_configs,
            args.batch_size,
            num_contextual_features,
            total_max_seqlen,
            args.checkpoint_dir,
            use_kvcache,
        )

        dataset = InferenceDataset(
            seq_logs_file=dataproc._inference_sequence_file,
            batch_logs_file=dataproc._inference_batch_file,
            batch_size=args.batch_size,
            max_seqlen=total_max_seqlen,
            item_feature_name=dataproc._item_feature_name,
            contextual_feature_names=dataproc._contextual_feature_names,
            action_feature_name=dataproc._action_feature_name,
            max_num_candidates=dataset_args.max_num_candidates,
            item_vocab_size=10_000_000,
            userid_name="user_id",
            date_name="date",
            sequence_endptr_name="interval_indptr",
            timestamp_names=["date", "interval_end_ts"],
        )
        dataloader = get_data_loader(dataset=dataset)

        def batch_iter():
            # Cycle the (small, for kuairand-1k) inference dataset indefinitely
            # so we can run a fixed iteration count regardless of dataset size.
            while True:
                it = iter(dataloader)
                yielded_any = False
                for uids, dates, seq_endptrs in it:
                    batch = dataset.get_input_batch(
                        uids,
                        dates,
                        seq_endptrs,
                        torch.zeros_like(seq_endptrs),
                        with_contextual_features=True,
                        with_ranking_labels=False,
                    )
                    if batch is None:
                        continue
                    total_history_lengths = seq_endptrs * 2 + num_contextual_features
                    yielded_any = True
                    yield uids, batch, total_history_lengths
                if not yielded_any:
                    raise RuntimeError(
                        "InferenceDataset produced no batches -- check "
                        "scripts/03_preprocess_data.sh ran --inference for this dataset."
                    )

        gen = batch_iter()
        latencies_ms = []
        n_total = args.warmup + args.iters
        for i in range(n_total):
            uids, batch, total_history_lengths = next(gen)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_kvcache:
                model.forward_with_kvcache(batch, uids, total_history_lengths)
            else:
                model.forward_nokvcache(batch)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if i >= args.warmup:
                latencies_ms.append((t1 - t0) * 1000.0)

    latencies_ms.sort()

    def pct(p):
        if not latencies_ms:
            return None
        idx = min(len(latencies_ms) - 1, math.ceil(p / 100.0 * len(latencies_ms)) - 1)
        return latencies_ms[max(idx, 0)]

    mean_ms = statistics.mean(latencies_ms) if latencies_ms else None
    result = {
        "tier": args.tier,
        "batch_size": args.batch_size,
        "use_kvcache": use_kvcache,
        "warmup": args.warmup,
        "measured_iters": len(latencies_ms),
        "latency_ms": {
            "mean": mean_ms,
            "p50": pct(50),
            "p90": pct(90),
            "p99": pct(99),
            "min": latencies_ms[0] if latencies_ms else None,
            "max": latencies_ms[-1] if latencies_ms else None,
        },
        "throughput": {
            "requests_per_sec": (1000.0 / mean_ms) if mean_ms else None,
            "items_per_sec": (1000.0 / mean_ms * args.batch_size) if mean_ms else None,
        },
    }
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
