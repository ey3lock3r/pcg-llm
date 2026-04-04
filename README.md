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

Additional optimisations (all toggleable via config flags):

| Flag | Value | Effect |
|------|-------|--------|
| `optimizer` | `muon_adamw` | Muon (orthogonal Newton-Schulz updates) for `nn.Linear` weights + AdamW for embeddings/biases |
| `normalize` | `ngpt` | Unit-norm node states inside DEQ — eliminates LR warmup, enables 10× higher base LR |
| `projection` | `monarch` | Butterfly-structured 64×64 block decomposition replacing dense 3072×3072 projections |
| `optimizer_bits` | `8` | bitsandbytes 8-bit quantised momentum/variance (~50% VRAM reduction) |
| `grad_checkpoint` | `True` | Gradient checkpointing to trade compute for VRAM |
| `cpu_offload_mask` | `True` | Offload RigL sparse mask to CPU RAM (3B only) |

---

## Project Structure

```
src/pcg_llm/
├── arch/
│   ├── adjacency.py      # Block-structured sparse adjacency (RigL)
│   ├── deq_solver.py     # DEQ fixed-point solver + spectral norm
│   ├── eagle_head.py     # Speculative decoding head (K=8 draft branches)
│   ├── monarch.py        # Butterfly/Monarch matrix projections
│   └── node.py           # PCG node state + message passing
├── checkpointing/
│   ├── checkpoint.py     # CheckpointManager: SHA-256 verified, manifest.json indexed
│   ├── local.py          # Local backend with disk-quota guard
│   ├── gcs.py            # GCS backend with exponential-backoff retry (5 attempts)
│   └── signals.py        # SIGTERM / SIGUSR1 graceful shutdown handler
├── data/
│   ├── streaming.py      # FineWeb-Edu + The Stack v2 interleaved stream
│   └── tokenizer.py      # Llama-3 Tiktoken wrapper (vocab=128256)
├── training/
│   ├── trainer.py        # PCGTrainer: full training loop + resume logic
│   ├── optimizer.py      # Muon + HybridOptimizer (Muon + AdamW)
│   ├── normalization.py  # nGPT normalisation
│   ├── loss.py           # CrossEntropy + λ·L1(A) − γ·Var(Z*)
│   ├── scheduler.py      # Cosine LR scheduler
│   └── curriculum.py     # 3-epoch dataset curriculum (FineWeb → Stack ramp-up)
├── evaluation/
│   └── harness.py        # lm-eval wrapper (4-bit GPTQ for T4)
├── monitoring/
│   └── metrics.py        # W&B: solver steps, sparsity, node variance, EAGLE accept rate
└── config.py             # TrainingConfig frozen dataclass with full validation
```

---

## Key Invariants

- **DEQ convergence**: Spectral Normalization keeps Lipschitz constant < 1 on all weight matrices
- **Loss**: `CrossEntropy + λ·L1(A) − γ·Var(Z*)` with variance floor τ=0.1, γ=0.01
- **Checkpoints**: atomic write (`.tmp` → rename), SHA-256 verified, `manifest.json` indexed
- **Anderson window**: m=3, β=1e-4
- **Checkpoint interval**: 500 steps (Kaggle) / 250 steps (GCP)

---

## Hardware Targets

| Platform | GPU | VRAM | Preset | Checkpoint |
|----------|-----|------|--------|------------|
| Kaggle Notebook | 2× T4 | 16 GB | `tiny` | Local, every 500 steps |
| Google Colab | T4 | 15 GB | `tiny` | Local or GCS |
| GCP Preemptible | V100 | 16 GB | `3b` | GCS, every 250 steps |
| GCP / Cloud | H100/A100 | 80 GB | `3b` | GCS, every 250 steps |

---

## Roadmap & Next Steps

The implementation is complete. The following steps take it from code to a trained, evaluated model.

---

### Phase 1 — Tiny PCG on Kaggle T4 (P1, start here)

**Goal**: validate that the DEQ solver converges, RigL sparsity runs correctly, and checkpoint-and-resume works — before spending any GCP credits on the 3B model.

> **Notebook**: [`notebooks/kaggle_tiny_train.ipynb`](notebooks/kaggle_tiny_train.ipynb)
> Import this into a Kaggle Notebook with **2× T4 GPU** enabled. It handles all setup, training, checkpoint inspection, and health checks in one place. The steps below describe what the notebook does.

#### Step 1.1 — Set up the Kaggle Notebook

1. Create a new Kaggle Notebook with **2× T4 GPU** accelerator enabled.
2. In the notebook, clone the repo and install dependencies:

```bash
!git clone https://github.com/ey3lock3r/pcg-llm.git
%cd pcg-llm
!pip install uv --quiet
!uv sync --extra training
```

3. Accept the Llama-3 model license at [huggingface.co/meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and log in so the tokenizer can download:

```bash
!uv run huggingface-cli login --token YOUR_HF_TOKEN
```

#### Step 1.2 — Export and inspect the config

Before training, export the resolved config to verify all flags are set correctly:

```bash
!uv run pcg-llm export-config --preset tiny --output /kaggle/working/config.json
!cat /kaggle/working/config.json
```

Key fields to verify:

| Field | Expected value |
|-------|---------------|
| `hidden_dim` | 512 |
| `checkpoint_dir` | `/kaggle/working/checkpoints` |
| `checkpoint_interval` | 500 |
| `optimizer` | `muon_adamw` |
| `normalize` | `ngpt` |
| `projection` | `monarch` |
| `optimizer_bits` | 32 |

To override individual flags without editing the config:

```bash
# Example: disable Monarch projections for a baseline run
!uv run pcg-llm export-config --preset tiny --projection dense --output /kaggle/working/config_baseline.json
```

#### Step 1.3 — Run the first training session

```bash
!uv run pcg-llm train \
  --preset tiny \
  --total-tokens 1B \
  --wandb-project pcg-llm-tiny-kaggle
```

This trains for ~2,000 steps on a 1B-token slice of FineWeb-Edu. Expected behaviour:
- Checkpoint files written every 500 steps to `/kaggle/working/checkpoints/`
- W&B dashboard shows: loss decreasing, solver steps ≤ 8/token, node variance > 0.1
- No `CRITICAL` disk-quota warnings (T4 Kaggle has ~20 GB working space)

#### Step 1.4 — Validate checkpoint-and-resume

1. Manually stop the kernel (simulates a session timeout) mid-run, e.g. at step 1,200.
2. Restart the kernel and rerun the exact same command — training must resume from the latest valid checkpoint (step 1,000 or 1,500):

```bash
!uv run pcg-llm train --preset tiny --total-tokens 1B
# Training should print: "Resuming from step-001000.pt" (or similar)
```

3. Confirm the loss trajectory continues smoothly from the checkpoint — no loss spike.

#### Step 1.5 — Check acceptance criteria

After a full ~4-hour run (~5,000 steps on T4):

| Metric | Target |
|--------|--------|
| Perplexity on held-out 50M tokens | ≤ 50 |
| Mean solver steps per token | ≤ 8 |
| Node variance (W&B `node_variance`) | > 0.1 throughout |
| EAGLE draft acceptance rate | ≥ 40% |

If perplexity is stuck above 100, check:
- `node_variance` in W&B — if near zero, the variance loss term (γ) may need tuning
- `solver_steps_mean` — if consistently hitting `max_solver_iters=12`, reduce `base_lr` by 3×
- Loss curve — if diverging in the first 500 steps, warmup may be too short (increase `--warmup-steps 1000`)

---

### Phase 2 — Optimisation A/B Testing (P2)

**Goal**: confirm each optimisation flag improves throughput without regressing loss.

Run four variants of the Tiny PCG for 1,000 steps each and compare wall-clock time and loss:

```bash
# Baseline (all optimisations off)
!uv run pcg-llm train --preset tiny --total-tokens 100M \
  --optimizer adamw --normalize standard --projection dense --optimizer-bits 32 \
  --wandb-project pcg-llm-ablation --checkpoint-dir /kaggle/working/ckpt_baseline

# Muon optimizer only
!uv run pcg-llm train --preset tiny --total-tokens 100M \
  --optimizer muon_adamw --normalize standard --projection dense \
  --wandb-project pcg-llm-ablation --checkpoint-dir /kaggle/working/ckpt_muon

# nGPT normalisation only
!uv run pcg-llm train --preset tiny --total-tokens 100M \
  --optimizer adamw --normalize ngpt --projection dense \
  --wandb-project pcg-llm-ablation --checkpoint-dir /kaggle/working/ckpt_ngpt

# All optimisations on (full config)
!uv run pcg-llm train --preset tiny --total-tokens 100M \
  --wandb-project pcg-llm-ablation --checkpoint-dir /kaggle/working/ckpt_full
```

**Target**: the all-optimisations run finishes 1,000 steps at least 20% faster than the baseline.

---

### Phase 3 — GCS Checkpoint Setup for GCP (P3)

**Goal**: configure durable GCS checkpointing before starting any GCP runs.

#### Step 3.1 — Create a GCS bucket

```bash
gcloud storage buckets create gs://pcg-llm-checkpoints \
  --location=us-central1 \
  --uniform-bucket-level-access
```

#### Step 3.2 — Authenticate on the GCP instance

```bash
gcloud auth application-default login
pip install uv --quiet
git clone https://github.com/ey3lock3r/pcg-llm.git && cd pcg-llm
uv sync --extra training --extra gcp
```

#### Step 3.3 — Start the 3B training run

```bash
uv run pcg-llm train \
  --preset 3b \
  --checkpoint-dir gs://pcg-llm-checkpoints/3b \
  --wandb-project pcg-llm-3b
```

The GCS backend writes checkpoints atomically (temp blob → server-side rename) with exponential-backoff retry. If the preemptible instance is reclaimed mid-write, the previous checkpoint remains intact.

#### Step 3.4 — Simulate preemption and resume

1. Terminate the VM manually at step ~750.
2. Spin up a fresh preemptible instance, re-run the exact same command.
3. Confirm training resumes from the latest valid GCS checkpoint with correct model state.

---

### Phase 4 — Benchmark Evaluation (P3)

**Goal**: verify the trained 3B model scores within 5 percentage points of Phi-3-mini (3.8B) and Gemma-2 2B on standard benchmarks.

#### Step 4.1 — Install eval dependencies

```bash
uv sync --extra eval
```

#### Step 4.2 — Run benchmarks

```bash
uv run pcg-llm evaluate \
  --checkpoint gs://pcg-llm-checkpoints/3b/step-050000.pt \
  --tasks arc_challenge,hellaswag,mmlu,gsm8k,humaneval \
  --quantize 4bit \
  --output results.json

cat results.json
```

#### Step 4.3 — Compare against targets

| Benchmark | Phi-3-mini 3.8B | Gemma-2 2B | PCG-LLM 3B target |
|-----------|----------------|------------|-------------------|
| ARC-Challenge | ~55% | ~52% | ≥ 47% |
| HellaSwag | ~78% | ~74% | ≥ 69% |
| MMLU | ~70% | ~51% | ≥ 46% |
| GSM8K | ~55% | ~24% | ≥ 19% |
| HumanEval | ~58% | ~36% | ≥ 31% |

---

### Phase 5 — EAGLE Head Fine-Tuning (P3, after 3B baseline)

Once the base 3B model is trained, fine-tune the EAGLE speculative decoding head separately:

```python
from pcg_llm.config import TrainingConfig
from pcg_llm.training.trainer import PCGTrainer

config = TrainingConfig.from_preset("3b")
trainer = PCGTrainer(config=config)
trainer.resume_if_available()
trainer.fine_tune_eagle(steps=5000)
```

**Target**: EAGLE draft acceptance rate ≥ 60%. Below 60% the speculative head is slower than greedy decoding.

---

## Installation

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
# Core only
uv sync

# Training (Kaggle / Colab)
uv sync --extra training

# GCP preemptible with GCS checkpointing
uv sync --extra training --extra gcp

# Benchmark evaluation
uv sync --extra eval

# Everything
uv sync --extra all
```

---

## CLI Reference

```
uv run pcg-llm <command> [options]

Commands:
  train           Start or resume training
  evaluate        Run lm-eval benchmarks on a checkpoint
  export-config   Dump the resolved TrainingConfig to JSON

train options:
  --preset tiny|3b          Use a named preset (tiny=512d Kaggle, 3b=3072d GCP)
  --config PATH             Load config from a JSON file
  --checkpoint-dir PATH     Local path or gs:// URI
  --checkpoint-backend      local | gcs
  --checkpoint-interval N   Steps between saves (default: 500)
  --optimizer               adamw | muon_adamw (default: muon_adamw)
  --optimizer-bits          32 | 8 (default: 32)
  --normalize               standard | ngpt (default: ngpt)
  --projection              dense | monarch (default: monarch)
  --total-tokens            e.g. 10B, 100B, 1T
  --base-lr FLOAT           Base learning rate (default: 1e-3)
  --batch-size N            Per-GPU micro-batch size (default: 8)
  --grad-accum N            Gradient accumulation steps (default: 8)
  --wandb-project NAME      W&B project (omit to disable W&B)
  --no-resume               Start from scratch (ignore existing checkpoints)
  --seed N                  Global random seed (default: 42)

evaluate options:
  --checkpoint PATH         Path to .pt checkpoint (required)
  --tasks TASKS             Comma-separated lm-eval task names
  --quantize none|4bit|8bit Inference quantization (default: 4bit)
  --output PATH             Write results JSON to file (default: stdout)
```

---

## Development

```bash
# Install dev dependencies and pre-commit hooks
uv sync --extra dev
uv run pre-commit install

# Run tests (CPU; GPU tests require CUDA)
uv run pytest tests/ --ignore=tests/gpu --ignore=tests/integration -v \
  --cov=src/pcg_llm --cov-report=term-missing

# Lint + format
uvx ruff check src/ tests/
uvx ruff format src/ tests/

# Type check
uv run mypy src/pcg_llm

# Security scan
uvx bandit -r src/ -c pyproject.toml
```

---

## License

MIT — see [LICENSE](LICENSE).
