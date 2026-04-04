#!/usr/bin/env python3
"""Benchmark: RigL Drop-and-Grow step timing at 90% sparsity (T037).

Run:
    python benchmarks/bench_rigl_cycle.py            # compare vs baseline
    python benchmarks/bench_rigl_cycle.py --baseline  # capture baseline
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def run_benchmark(num_blocks: int = 64, block_size: int = 64, n_calls: int = 100) -> dict:
    from pcg_llm.arch.adjacency import BlockSparseAdjacency

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = BlockSparseAdjacency(num_blocks=num_blocks, block_size=block_size)
    adj.initialize_erdos_renyi(sparsity=0.90)
    adj.W_structure = adj.W_structure.to(device)
    adj.mask = adj.mask.to(device)

    grads = torch.randn(num_blocks, num_blocks, device=device).abs()

    # Warm up
    for _ in range(5):
        adj.drop_and_grow(grads)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        adj.drop_and_grow(grads)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "benchmark": "bench_rigl_cycle",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {"num_blocks": num_blocks, "block_size": block_size,
                   "sparsity": 0.90, "n_calls": n_calls, "device": str(device)},
        "results": {
            "wall_clock_ms_total": round(elapsed_ms, 2),
            "wall_clock_ms_per_call": round(elapsed_ms / n_calls, 3),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RigL Drop-and-Grow cycle")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--n-calls", type=int, default=100)
    args = parser.parse_args()

    results = run_benchmark(num_blocks=args.num_blocks, n_calls=args.n_calls)
    r = results["results"]

    print("\n=== RigL Cycle Benchmark ===")
    print(f"Device: {results['config']['device']}")
    print(f"Wall-clock (total): {r['wall_clock_ms_total']:.1f} ms for {args.n_calls} calls")
    print(f"Wall-clock (per call): {r['wall_clock_ms_per_call']:.3f} ms")

    if args.baseline:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        path = results_dir / "bench_rigl_cycle_baseline.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline saved to {path}")


if __name__ == "__main__":
    main()
