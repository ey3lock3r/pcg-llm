#!/usr/bin/env python3
"""Benchmark: EAGLE draft + verification throughput (T037)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import torch


def run_benchmark(
    hidden_dim: int = 512,
    num_blocks: int = 16,
    eagle_k: int = 8,
    draft_len: int = 4,
    n_calls: int = 50,
) -> dict:
    try:
        from pcg_llm.arch.eagle_head import EAGLEExtrapolationHead
    except ImportError:
        return {
            "benchmark": "bench_eagle_throughput",
            "error": "EAGLEExtrapolationHead not available",
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = EAGLEExtrapolationHead(hidden_dim=hidden_dim, eagle_k=eagle_k, draft_len=draft_len)
    head = head.to(device)

    B = 2
    Z = torch.randn(B, num_blocks, hidden_dim, device=device)

    # Warm up
    for _ in range(3):
        head.generate_draft_tree(Z)

    t0 = time.perf_counter()
    for _ in range(n_calls):
        head.generate_draft_tree(Z)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    tokens_per_call = B * eagle_k * draft_len
    tokens_per_sec = (tokens_per_call * n_calls) / (elapsed_ms / 1000)

    return {
        "benchmark": "bench_eagle_throughput",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": {
            "hidden_dim": hidden_dim,
            "eagle_k": eagle_k,
            "draft_len": draft_len,
            "n_calls": n_calls,
            "device": str(device),
        },
        "results": {
            "wall_clock_ms_total": round(elapsed_ms, 2),
            "wall_clock_ms_per_call": round(elapsed_ms / n_calls, 3),
            "draft_tokens_per_second": round(tokens_per_sec, 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    results = run_benchmark()
    r = results.get("results", {})

    print("\n=== EAGLE Throughput Benchmark ===")
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    print(f"Draft tokens/sec: {r.get('draft_tokens_per_second', 0):.1f}")
    print(f"Wall-clock (per call): {r.get('wall_clock_ms_per_call', 0):.3f} ms")

    if args.baseline:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        path = results_dir / "bench_eagle_throughput_baseline.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline saved to {path}")


if __name__ == "__main__":
    main()
