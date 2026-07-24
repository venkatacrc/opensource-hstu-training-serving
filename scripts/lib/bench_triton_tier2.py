#!/usr/bin/env python3
"""Latency/throughput benchmark client for Tier 2 (Triton Python backend).

Adapted from the upstream
`inference/triton/hstu_model/client.py::infer_batch` (same input tensor
names/dtypes, same KJT batch construction via `commons.datasets.get_dataset`),
parameterized by --batch_size so scripts/07_serving_bench.sh can sweep it
(the upstream client hard-codes MAX_BATCH_SIZE=2), and instrumented for
p50/p90/p99 latency instead of only an aggregate total.
"""
import argparse
import json
import math
import statistics
import sys
import time

import gin
import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype


def strip_padding_batch(batch, unpadded_batch_size):
    from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

    batch.batch_size = unpadded_batch_size
    kjt_dict = batch.features.to_dict()
    for k in kjt_dict:
        kjt_dict[k] = JaggedTensor.from_dense_lengths(
            kjt_dict[k].to_padded_dense()[: batch.batch_size],
            kjt_dict[k].lengths()[: batch.batch_size].long(),
        )
    batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
    batch.num_candidates = batch.num_candidates[: batch.batch_size]
    return batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gin_config_file", required=True)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--model_name", default="hstu_model")
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tier", default="tier2_triton_python")
    ap.add_argument("--output_json", required=True)
    ap.add_argument(
        "--hstu_dir", default="/workspace/recsys-examples/examples/hstu"
    )
    args = ap.parse_args()

    sys.path.insert(0, args.hstu_dir)
    from commons.datasets import get_data_loader
    from commons.datasets.hstu_sequence_dataset import get_dataset
    from utils import DatasetArgs

    gin.parse_config_file(args.gin_config_file)
    dataset_args = DatasetArgs()

    _, eval_dataset = get_dataset(
        dataset_name=dataset_args.dataset_name,
        dataset_path=dataset_args.dataset_path,
        max_history_seqlen=dataset_args.max_history_seqlen,
        max_num_candidates=dataset_args.max_num_candidates,
        num_tasks=1,
        batch_size=args.batch_size,
        rank=0,
        world_size=1,
        shuffle=False,
        random_seed=0,
        eval_batch_size=args.batch_size,
    )
    dataloader = get_data_loader(dataset=eval_dataset)

    def batch_iter():
        while True:
            it = iter(dataloader)
            got_any = False
            for b in it:
                got_any = True
                yield b
            if not got_any:
                raise RuntimeError("Eval dataloader produced no batches.")

    gen = batch_iter()

    def infer_once(client, batch):
        uids = batch.features.to_dict()["user_id"].values()
        nonlocal_batch = batch
        if uids.shape[0] != batch.batch_size:
            nonlocal_batch = strip_padding_batch(batch, uids.shape[0])
            uids = nonlocal_batch.features.to_dict()["user_id"].values()

        uids_np = uids.detach().numpy()
        tokens = nonlocal_batch.features.values().detach().numpy()
        token_lens = nonlocal_batch.features.lengths().detach().numpy()
        num_candidates = nonlocal_batch.num_candidates.detach().numpy()

        inputs = [
            httpclient.InferInput("USER_IDS", uids_np.shape, np_to_triton_dtype(uids_np.dtype)),
            httpclient.InferInput("TOKEN_LENGTHS", token_lens.shape, np_to_triton_dtype(token_lens.dtype)),
            httpclient.InferInput("TOKENS", tokens.shape, np_to_triton_dtype(tokens.dtype)),
            httpclient.InferInput("NUM_CANDIDATES", num_candidates.shape, np_to_triton_dtype(num_candidates.dtype)),
        ]
        inputs[0].set_data_from_numpy(uids_np)
        inputs[1].set_data_from_numpy(token_lens)
        inputs[2].set_data_from_numpy(tokens)
        inputs[3].set_data_from_numpy(num_candidates)
        outputs = [httpclient.InferRequestedOutput("OUTPUT")]

        t0 = time.perf_counter()
        client.infer(args.model_name, inputs, request_id="0", outputs=outputs)
        return (time.perf_counter() - t0) * 1000.0

    latencies_ms = []
    with httpclient.InferenceServerClient(args.url) as client:
        for i in range(args.warmup + args.iters):
            batch = next(gen)
            lat = infer_once(client, batch)
            if i >= args.warmup:
                latencies_ms.append(lat)

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
