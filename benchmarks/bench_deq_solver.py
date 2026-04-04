#!/usr/bin/env python3
"""Benchmark: DEQ solver wall-clock time + peak VRAM (T037).

Run:
    python benchmarks/bench_deq_solver.py            # compare vs baseline
    python benchmarks/bench_deq_solver.py --baseline  # capture baseline

Writes results to benchmarks/results/bench_deq_solver_baseline.json on --baseline.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def run_benchmark(
    hidden_dim: int = 512,
    num_blocks: int = 64,
    batch_size: int = 4,
    n_calls: int = 100,
    anderson_window: int = 3,
    max_iter: int = 12,
) -> dict:
    """Run N DEQ solve calls and measure wall-clock + peak VRAM."""
    from pcg_llm.arch.deq_solver import ConstrainedDEQSolver

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solver = ConstrainedDEQSolver(
        anderson_window=anderson_window,
        max_iter=max_iter,
        solver_tolerance=1e-2,
    )

    block_size = hidden_dim // num_blocks if num_blocks <= hidden_dim else hidden_dim
    z_shape = (batch_size, num_blocks, block_size)

    def f_theta(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(z * 0.9 + x * 0.1)

    # Warm up
    z0 = torch.zeros(*z_shape, device=device)
    x = torch.randn(*z_shape, device=device)
    for _ in range(5):
        solver.solve(f_theta, z0, x)

    # Measure
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    solver_steps_list = []
    for _ in range(n_calls):
        z0 = torch.zeros(*z_shape, device=device)
        x = torch.randn(*z_shape, device=device)
        _, info = solver.solve(f_theta, z0, x)
        solver_steps_list.append(info.get("solver_steps", 0))
    elapsed_ms = (time.perf_counter() - t0) * 1000

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2

    mean_steps = sum(solver_steps_list) / len(solver_steps_list)
    wall_clock_per_call_ms = elapsed_ms / n_calls

    return {
        "benchmark": "bench_deq_solver",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "hidden_dim": hidden_dim, "num_blocks": num_blocks,
            "batch_size": batch_size, "n_calls": n_calls,
            "anderson_window": anderson_window, "max_iter": max_iter,
            "device": str(device),
        },
        "results": {
            "wall_clock_ms_total": round(elapsed_ms, 2),
            "wall_clock_ms_per_call": round(wall_clock_per_call_ms, 3),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "solver_steps_mean": round(mean_steps, 2),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DEQ solver performance")
    parser.add_argument("--baseline", action="store_true", help="Save results as baseline")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-calls", type=int, default=100)
    args = parser.parse_args()

    results = run_benchmark(
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        batch_size=args.batch_size,
        n_calls=args.n_calls,
    )

    # Print summary table
    r = results["results"]
    print("\n=== DEQ Solver Benchmark ===")
    print(f"Device:            {results['config']['device']}")
    print(f"Wall-clock (total): {r['wall_clock_ms_total']:.1f} ms for {args.n_calls} calls")
    print(f"Wall-clock (per call): {r['wall_clock_ms_per_call']:.3f} ms")
    print(f"Peak VRAM:          {r['peak_vram_mb']:.1f} MB")
    print(f"Mean solver steps:  {r['solver_steps_mean']:.2f}")

    if args.baseline:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        baseline_path = results_dir / "bench_deq_solver_baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nBaseline saved to {baseline_path}")
    else:
        baseline_path = Path(__file__).parent / "results" / "bench_deq_solver_baseline.json"
        if baseline_path.exists():
            with open(baseline_path) as f:
                baseline = json.load(f)
            b_ms = baseline["results"]["wall_clock_ms_per_call"]
            c_ms = r["wall_clock_ms_per_call"]
            improvement = (b_ms - c_ms) / b_ms * 100 if b_ms > 0 else 0
            print(f"\nBaseline:  {b_ms:.3f} ms/call")
            print(f"Current:   {c_ms:.3f} ms/call")
            print(f"Change:    {improvement:+.1f}%  {'✓ PASS' if improvement >= 20 else '✗ FAIL (need >=20%)'}")
        else:
            print("\nNo baseline found. Run with --baseline to capture one.")


if __name__ == "__main__":
    main()
