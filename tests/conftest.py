"""Shared pytest fixtures for PCG-LLM tests."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Hardware / environment skip markers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Seed torch and Python RNG before each test for determinism."""
    import random
    random.seed(0)
    try:
        import torch
        torch.manual_seed(0)
    except ImportError:
        pass


def pytest_configure(config: Any) -> None:
    config.addinivalue_line("markers", "convergence: marks tests verifying DEQ convergence")
    config.addinivalue_line("markers", "gpu: marks tests requiring a CUDA GPU")
    config.addinivalue_line("markers", "slow: marks tests that are slow (integration)")


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(skip_gpu)


# ---------------------------------------------------------------------------
# Tiny config fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_config() -> dict[str, Any]:
    """Minimal TrainingConfig kwargs for fast unit tests (no GPU required)."""
    return {
        "hidden_dim": 64,
        "max_seq_len": 128,
        "vocab_size": 128256,
        "block_size": 32,
        "initial_sparsity": 0.70,
        "max_solver_iters": 5,
        "solver_tolerance": 1e-2,
        "anderson_window": 3,
        "anderson_beta": 1e-4,
        "lipschitz_target": 1.0,
        "lambda_sparse": 0.01,
        "gamma_variance": 0.01,
        "variance_floor": 0.1,
        "rigl_interval": 10,
        "rigl_drop_fraction": 0.10,
        "rigl_reroute_start": 0.30,
        "rigl_freeze_step_frac": 0.80,
        "eagle_k": 2,
        "eagle_draft_len": 2,
        "eagle_accept_threshold": 0.65,
        "base_lr": 1e-3,
        "warmup_steps": 5,
        "optimizer": "adamw",
        "optimizer_bits": 32,
        "normalize": "standard",
        "projection": "dense",
        "grad_checkpoint": False,
        "cpu_offload_mask": False,
        "grad_accum_steps": 1,
        "batch_size": 2,
        "checkpoint_interval": 10,
        "checkpoint_backend": "local",
        "checkpoint_dir": "/tmp/pcg_llm_test_checkpoints",
        "dataset_fineweb_frac": [0.92, 0.85, 0.70],
        "dataset_stack_frac": [0.08, 0.15, 0.30],
        "total_tokens": 1_000,
        "wandb_project": None,
    }


# ---------------------------------------------------------------------------
# Temporary checkpoint directory fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_checkpoint_dir(tmp_path: Path) -> Path:
    """A temporary directory for checkpoint tests; cleaned up after each test."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


# ---------------------------------------------------------------------------
# Mock dataset fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dataset_batch() -> dict[str, Any]:
    """A minimal mock batch simulating tokenized sequences.

    Returns a dict with 'input_ids' as a list of integer token sequences.
    """
    import random

    rng = random.Random(42)
    vocab_size = 128256
    seq_len = 32
    batch_size = 2
    return {
        "input_ids": [
            [rng.randint(0, vocab_size - 1) for _ in range(seq_len)] for _ in range(batch_size)
        ],
    }
