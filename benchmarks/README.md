# PCG-LLM Benchmarks

This directory contains performance benchmarks for the PCG-LLM training system.
Benchmarks are required per **Constitution Principle V**: all DEQ solver, routing graph,
and attention mechanism changes must capture a baseline before optimization PRs.

## Available Benchmarks

| Script | Measures | Baseline File |
|---|---|---|
| `bench_deq_solver.py` | Wall-clock time + peak VRAM for 100 DEQ solve calls (Tiny PCG dims) | `results/bench_deq_solver_baseline.json` |
| `bench_rigl_cycle.py` | Drop-and-Grow step timing at 90% sparsity | `results/bench_rigl_cycle_baseline.json` |
| `bench_eagle_throughput.py` | EAGLE draft + verification tokens/sec vs. greedy decoding | `results/bench_eagle_throughput_baseline.json` |

## How to Run

### Prerequisites

```bash
pip install -e ".[triton,training]"
# GPU recommended for meaningful results
```

### Capture a baseline (first run, all optimizations OFF)

```bash
python benchmarks/bench_deq_solver.py --baseline
python benchmarks/bench_rigl_cycle.py --baseline
python benchmarks/bench_eagle_throughput.py --baseline
```

Each script writes `benchmarks/results/{name}_baseline.json` on first run with `--baseline`.

### Run with optimizations enabled

```bash
python benchmarks/bench_deq_solver.py
python benchmarks/bench_rigl_cycle.py
python benchmarks/bench_eagle_throughput.py
```

The scripts print a comparison table showing improvement vs. the stored baseline.

### Run all benchmarks

```bash
python -m pytest benchmarks/ -v --no-cov
# or directly:
for f in benchmarks/bench_*.py; do python "$f"; done
```

## Constitutional Requirement

Per **Principle V (Performance)**: the ≥20% per-step speedup gate must be verified by
running all three benchmarks with all optimization flags ON vs. the stored baseline
(all flags OFF = v3.0 behavior). Results must be committed to `benchmarks/results/`
before merging optimization PRs.

## Results Directory

`benchmarks/results/` stores JSON files with structured benchmark output:

```json
{
  "benchmark": "bench_deq_solver",
  "timestamp_utc": "2026-04-04T12:00:00Z",
  "config": {"hidden_dim": 512, "solver_iters": 12, "anderson_window": 3},
  "results": {
    "wall_clock_ms_mean": 42.3,
    "wall_clock_ms_p95": 51.1,
    "peak_vram_mb": 312.0,
    "iterations_per_second": 23.6
  }
}
```

The baseline JSON is committed to the repository; comparison runs print the delta
table but do NOT overwrite the baseline file.
