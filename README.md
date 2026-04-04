# PCG-LLM

**Predictive Coding Graph Language Model** — a non-hierarchical, dynamically routed neural architecture using Deep Equilibrium (DEQ) solving and Inference Learning, optimised for free-tier GPU training (Kaggle T4, Google Colab).

[![CI](https://github.com/ey3lock3r/pcg-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/ey3lock3r/pcg-llm/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

PCG-LLM extends a standard transformer with three core mechanisms:

| Component | Description |
|-----------|-------------|
| **DEQ Solver** | Fixed-point iteration `Z* = f_θ(Z*, X)` via Anderson Acceleration or Broyden. Spectral Normalization enforces Lipschitz constant < 1 for guaranteed convergence. |
| **Block-Structured RigL** | Dynamic sparse adjacency graph using 64×64 blocks (3B model) or 32×32 (Tiny). Never instantiates a dense adjacency matrix. |
| **EAGLE Head** | Single transformer block on penultimate PCG states. Generates K=8 speculative draft branches for fast decoding. |

Additional optimisations (all toggleable via config):

- **Muon optimizer** — orthogonal gradient updates via Newton-Schulz iteration for all `nn.Linear` weights
- **nGPT normalisation** — unit-norm node states inside the DEQ function, enabling higher learning rates
- **Monarch matrices** — butterfly-structured 64×64 block decomposition replacing dense 3072×3072 projections
- **FlexAttention** — block-local 64-token + 1 global summary token mask for the EAGLE head
- **8-bit optimizer states** — bitsandbytes quantised momentum/variance (~50% VRAM reduction)

---

## Installation

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
# Core only (inference)
uv sync

# Training on Kaggle/Colab (T4)
uv sync --extra training

# GCP preemptible training with GCS checkpointing
uv sync --extra training --extra gcp

# Benchmark evaluation
uv sync --extra eval

# Everything
uv sync --extra all
```

---

## Quick Start

### Training (Kaggle / Colab)

```python
from pcg_llm.config import TrainingConfig
from pcg_llm.training.trainer import Trainer

config = TrainingConfig(
    checkpoint_dir="/kaggle/working/checkpoints",
    checkpoint_interval=500,
    optimizer="muon",           # or "adamw"
    normalize="ngpt",           # or "layer_norm"
    projection="monarch",       # or "dense"
    optimizer_bits=8,
)

trainer = Trainer(config)
trainer.train(dataloader)
```

### Training (GCP Preemptible)

```python
config = TrainingConfig(
    checkpoint_dir="gs://my-bucket/pcg-llm/checkpoints",
    checkpoint_interval=250,
)
```

Training resumes automatically from the latest valid GCS checkpoint on restart.

### Evaluation

```bash
uv run pcg-llm eval \
  --checkpoint /path/to/step-050000.pt \
  --tasks arc_easy,hellaswag,mmlu,gsm8k,humaneval \
  --quantize 4bit
```

---

## Project Structure

```
src/pcg_llm/
├── arch/
│   ├── adjacency.py      # Block-structured sparse adjacency (RigL)
│   ├── deq_solver.py     # DEQ fixed-point solver + spectral norm
│   ├── eagle_head.py     # Speculative decoding head
│   ├── monarch.py        # Butterfly/Monarch matrix projections
│   └── node.py           # PCG node state
├── checkpointing/
│   ├── checkpoint.py     # CheckpointManager (SHA-256 + manifest.json)
│   ├── local.py          # Local backend (Kaggle /kaggle/working/)
│   ├── gcs.py            # GCS backend with exponential-backoff retry
│   └── signals.py        # SIGTERM / SIGUSR1 graceful shutdown
├── data/
│   ├── streaming.py      # FineWeb-Edu + The Stack v2 interleaved stream
│   └── tokenizer.py      # Llama-3 Tiktoken wrapper (vocab=128256)
├── training/
│   ├── trainer.py        # Main training loop
│   ├── optimizer.py      # Muon + HybridOptimizer
│   ├── normalization.py  # nGPT normalisation
│   ├── loss.py           # CrossEntropy + λ·L1(A) − γ·Var(Z*)
│   ├── scheduler.py      # LR scheduler
│   └── curriculum.py     # Curriculum learning
├── evaluation/
│   └── harness.py        # lm-eval wrapper (4-bit T4-compatible)
├── monitoring/
│   └── metrics.py        # W&B metrics (solver steps, sparsity, variance)
└── config.py             # TrainingConfig dataclass
```

---

## Key Invariants

- **DEQ convergence**: Spectral Normalization keeps Lipschitz constant < 1 on all weight matrices
- **Loss**: `CrossEntropy + λ·L1(A) − γ·Var(Z*)` with variance floor τ=0.1, γ=0.01
- **Checkpoints**: atomic write (`.tmp` → rename), SHA-256 verified, `manifest.json` indexed
- **Anderson window**: m=3 (not m=5), β=1e-4
- **Checkpoint interval**: 500 steps (Kaggle) / 250 steps (GCP)

---

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests (CPU only)
uv run pytest tests/ --ignore=tests/gpu --ignore=tests/integration -v

# Lint + format
uvx ruff check src/ tests/
uvx ruff format src/ tests/

# Type check
uv run mypy src/pcg_llm

# Security scan
uvx bandit -r src/ -c pyproject.toml
```

---

## Hardware Targets

| Platform | GPU | VRAM | Model Size | Checkpoint |
|----------|-----|------|------------|------------|
| Kaggle Notebook | 2× T4 | 16 GB | Tiny (50-100M) | Local, every 500 steps |
| Google Colab | T4 | 15 GB | Tiny (50-100M) | Local / GCS |
| GCP Preemptible | V100 | 16 GB | 3B | GCS, every 250 steps |
| GCP / Cloud | H100/A100 | 80 GB | 3B | GCS, every 250 steps |

---

## License

MIT — see [LICENSE](LICENSE).
