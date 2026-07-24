#!/usr/bin/env python3
"""Latency/throughput benchmark client for Tier 3 (Triton + AOTInductor C++).

Adapted from `inference_aoti/test_tritonserver_aoti_hstu_model.py`: replays
the dumped batch_*.pt input tensors produced by
`inference_aoti/export_inference_gr_ranking_kvcache.py` against the Triton
AOTI backend, cycling through the available dumped cases to hit the requested
iteration count, and recording p50/p90/p99 latency instead of only an
aggregate elapsed time.

SCOPE NOTE: AOTInductor compiles static shapes, so this tier is benchmarked
at the single batch size scripts/06_export_checkpoint.sh exported for
(the max of BENCH_BATCH_SIZES), not swept across all sizes like tiers 0-2 --
see docs/RESULTS.md for the caveat this implies when comparing tiers.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

INPUT_TENSORS = [
    ("INPUT__0", "values"),
    ("INPUT__1", "lengths"),
    ("INPUT__2", "num_candidates"),
    ("INPUT__3", "user_ids"),
    ("INPUT__4", "total_history_lengths"),
]


def _load_dumped_tensor(path: Path) -> np.ndarray:
    module = torch.jit.load(str(path), map_location="cpu")
    return module.tensor.detach().cpu().contiguous().numpy()


def _find_batch_indices(dump_dir: Path):
    indices = []
    for values_path in dump_dir.glob("batch_*_values.pt"):
        batch_id = values_path.name.removeprefix("batch_").removesuffix("_values.pt")
        indices.append(int(batch_id))
    return sorted(indices)


def _load_input_cases(dump_dir: Path):
    cases = []
    for batch_index in _find_batch_indices(dump_dir):
        prefix = dump_dir / f"batch_{batch_index:06d}"
        cases.append(
            [_load_dumped_tensor(Path(f"{prefix}_{suffix}.pt")) for _, suffix in INPUT_TENSORS]
        )
    if not cases:
        raise FileNotFoundError(f"No dumped input cases found in {dump_dir}")
    return cases


def _make_inputs(httpclient, case):
    inputs = []
    for (name, _), arr in zip(INPUT_TENSORS, case):
        if arr.dtype != np.int64:
            arr = arr.astype(np.int64, copy=False)
        ii = httpclient.InferInput(name, arr.shape, "INT64")
        ii.set_data_from_numpy(arr)
        inputs.append(ii)
    return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", type=Path, required=True)
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--model_name", default="hstu_gr_ranking_kvcache")
    ap.add_argument("--batch_size", type=int, required=True, help="informational only (fixed by AOTI export)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tier", default="tier3_triton_aoti")
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    import tritonclient.http as httpclient

    cases = _load_input_cases(args.dump_dir)
    outputs = [httpclient.InferRequestedOutput("OUTPUT__0"), httpclient.InferRequestedOutput("OUTPUT__1")]

    latencies_ms = []
    with httpclient.InferenceServerClient(url=args.url) as client:
        n_total = args.warmup + args.iters
        for i in range(n_total):
            case = cases[i % len(cases)]
            t0 = time.perf_counter()
            client.infer(args.model_name, inputs=_make_inputs(httpclient, case), outputs=outputs)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if i >= args.warmup:
                latencies_ms.append(elapsed_ms)

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
        "measured_iters": len(latencies_ms),
        "latency_ms": {
            "mean": mean_ms, "p50": pct(50), "p90": pct(90), "p99": pct(99),
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
