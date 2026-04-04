#!/usr/bin/env python3
"""Benchmark: DDP 2-GPU vs single-GPU throughput for Tiny PCG (T062).

Measures tokens/second and peak VRAM for:
  - Single-GPU training (baseline)
  - DDP 2-GPU training (via subprocess torchrun)

Run:
    # Single-process mode (single-GPU only; no torchrun needed)
    python benchmarks/bench_ddp_vs_single.py

    # Write results baseline JSON
    python benchmarks/bench_ddp_vs_single.py --baseline

Requires:
    - CUDA GPU(s) available (skips gracefully on CPU-only machines)
    - PyTorch >= 1.9 (for torchrun)

Satisfies constitution Principle V: benchmarks required for any multi-GPU
throughput claims made in documentation or PRs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "ddp_vs_single_baseline.json"

# Tiny PCG dimensions for benchmark
BENCH_CONFIG = {
    "hidden_dim": 64,
    "max_seq_len": 64,
    "vocab_size": 256,
    "block_size": 32,
    "initial_sparsity": 0.70,
    "max_solver_iters": 3,
    "solver_tolerance": 0.5,
    "checkpoint_interval": 9999,
    "checkpoint_backend": "local",
    "optimizer": "adamw",
    "optimizer_bits": 32,
    "normalize": "standard",
    "projection": "dense",
    "grad_checkpoint": False,
    "wandb_project": None,
    "warmup_steps": 0,
    "total_tokens": 100_000,
    "grad_accum_steps": 1,
    "batch_size": 4,
    "rigl_interval": 9999,
    "anderson_window": 3,
    "eagle_draft_len": 2,
    "eagle_k": 2,
}
N_STEPS = 20  # steps per measurement
BATCH_SIZE = 4
SEQ_LEN = 64


def _make_config(checkpoint_dir: str) -> Any:
    from pcg_llm.config import TrainingConfig

    return TrainingConfig(**{**BENCH_CONFIG, "checkpoint_dir": checkpoint_dir})


def bench_single_gpu(n_steps: int = N_STEPS) -> dict:
    """Measure single-GPU training throughput."""
    import tempfile

    from pcg_llm.training.trainer import PCGTrainer

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    vram_before = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0.0

    with tempfile.TemporaryDirectory() as tmp:
        config = _make_config(tmp)
        trainer = PCGTrainer(config)

        # Warm-up
        for _ in range(2):
            batch = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN))
            trainer.train_step(batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for _ in range(n_steps):
            batch = torch.randint(0, config.vocab_size, (BATCH_SIZE, SEQ_LEN))
            trainer.train_step(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    tokens_total = n_steps * BATCH_SIZE * SEQ_LEN
    tps = tokens_total / elapsed
    peak_vram = torch.cuda.max_memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0.0

    return {
        "mode": "single_gpu",
        "device": device_name,
        "n_steps": n_steps,
        "tokens_total": tokens_total,
        "elapsed_s": round(elapsed, 4),
        "tokens_per_sec": round(tps, 1),
        "peak_vram_gb": round(peak_vram, 3),
        "vram_delta_gb": round(peak_vram - vram_before, 3),
    }


def bench_ddp_2gpu() -> dict:
    """Measure DDP 2-GPU throughput via a torchrun subprocess.

    Returns a result dict, or a 'skipped' dict if torchrun or 2nd GPU
    are not available.
    """
    import shutil
    import tempfile

    if not torch.cuda.is_available():
        return {"mode": "ddp_2gpu", "skipped": True, "reason": "No CUDA GPU available"}
    if torch.cuda.device_count() < 2:
        return {
            "mode": "ddp_2gpu",
            "skipped": True,
            "reason": f"Only {torch.cuda.device_count()} GPU(s) found; need 2",
        }

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        return {
            "mode": "ddp_2gpu",
            "skipped": True,
            "reason": "torchrun not found in PATH (requires PyTorch >= 1.9)",
        }

    # Write a self-contained DDP worker script
    with tempfile.TemporaryDirectory() as tmp:
        worker_path = Path(tmp) / "ddp_worker.py"
        result_path = Path(tmp) / "result.json"

        bench_cfg = {**BENCH_CONFIG, "checkpoint_dir": tmp}
        worker_script = f"""
import sys, os, json, time, torch, torch.distributed as dist
sys.path.insert(0, {str(Path(__file__).parent.parent / "src")!r})

from pcg_llm.config import TrainingConfig
from pcg_llm.training.trainer import PCGTrainer

cfg = TrainingConfig(**{bench_cfg!r})
trainer = PCGTrainer(cfg)
rank = int(os.environ.get("RANK", 0))
n_steps = {N_STEPS}
batch_size = {BATCH_SIZE}
seq_len = {SEQ_LEN}

# Warm-up
for _ in range(2):
    batch = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    trainer.train_step(batch)

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

t0 = time.perf_counter()
for _ in range(n_steps):
    batch = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))
    trainer.train_step(batch)
if torch.cuda.is_available():
    torch.cuda.synchronize()
elapsed = time.perf_counter() - t0

if rank == 0:
    tokens_total = n_steps * batch_size * seq_len
    tps = tokens_total / elapsed
    peak_vram = torch.cuda.max_memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0.0
    result = {{
        "mode": "ddp_2gpu",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "n_steps": n_steps,
        "tokens_total": tokens_total,
        "elapsed_s": round(elapsed, 4),
        "tokens_per_sec": round(tps, 1),
        "peak_vram_gb_per_rank": round(peak_vram, 3),
    }}
    with open({str(result_path)!r}, "w") as f:
        json.dump(result, f)

if dist.is_initialized():
    dist.destroy_process_group()
"""
        worker_path.write_text(worker_script)

        proc = subprocess.run(
            [
                torchrun,
                "--nproc_per_node=2",
                "--master_port=29501",
                str(worker_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if proc.returncode != 0:
            return {
                "mode": "ddp_2gpu",
                "skipped": True,
                "reason": f"torchrun failed (exit {proc.returncode}): {proc.stderr[-500:]}",
            }

        if result_path.exists():
            return json.loads(result_path.read_text())

        return {
            "mode": "ddp_2gpu",
            "skipped": True,
            "reason": "Result file not written by DDP worker",
        }


def print_table(single: dict, ddp: dict) -> None:
    """Print a structured comparison table."""
    print("\n" + "=" * 60)
    print("  DDP vs Single-GPU Throughput Benchmark (Tiny PCG)")
    print("=" * 60)

    def row(label: str, s_val: str, d_val: str) -> None:
        print(f"  {label:<28} {s_val:>12}  {d_val:>12}")

    row("", "single-GPU", "DDP 2-GPU")
    print("  " + "-" * 56)
    row("Device", single.get("device", "?"), ddp.get("device", "N/A"))
    row("Steps", str(single.get("n_steps", "?")), str(ddp.get("n_steps", "N/A")))
    row(
        "Tokens/sec",
        f"{single.get('tokens_per_sec', 0):,.0f}",
        f"{ddp.get('tokens_per_sec', 0):,.0f}" if not ddp.get("skipped") else "skipped",
    )
    row(
        "Peak VRAM (GB)",
        f"{single.get('peak_vram_gb', 0):.2f}",
        f"{ddp.get('peak_vram_gb_per_rank', 0):.2f}/rank" if not ddp.get("skipped") else "skipped",
    )

    if not ddp.get("skipped"):
        speedup = ddp["tokens_per_sec"] / max(single["tokens_per_sec"], 1)
        print(f"\n  Speedup (DDP / single): {speedup:.2f}×")
        if speedup >= 1.7:
            print("  ✓ DDP achieves near-linear 2× scaling")
        else:
            print("  ⚠ DDP scaling below expected 1.7× threshold")
    else:
        print(f"\n  DDP skipped: {ddp.get('reason', 'unknown')}")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Write results to benchmarks/results/ddp_vs_single_baseline.json",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=N_STEPS,
        help=f"Number of training steps per measurement (default: {N_STEPS})",
    )
    args = parser.parse_args()

    global N_STEPS
    N_STEPS = args.steps

    print(f"Running single-GPU benchmark ({N_STEPS} steps)...")
    single_result = bench_single_gpu(n_steps=N_STEPS)
    print(
        f"  single-GPU: {single_result['tokens_per_sec']:,.0f} tokens/s  "
        f"({single_result.get('peak_vram_gb', 0):.2f} GB VRAM)"
    )

    print("Running DDP 2-GPU benchmark (requires torchrun + 2× GPU)...")
    ddp_result = bench_ddp_2gpu()
    if ddp_result.get("skipped"):
        print(f"  DDP skipped: {ddp_result.get('reason')}")
    else:
        print(
            f"  DDP 2-GPU: {ddp_result['tokens_per_sec']:,.0f} tokens/s  "
            f"({ddp_result.get('peak_vram_gb_per_rank', 0):.2f} GB/rank)"
        )

    print_table(single_result, ddp_result)

    if args.baseline:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        results = {
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "single_gpu": single_result,
            "ddp_2gpu": ddp_result,
        }
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        print(f"\nBaseline written to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
