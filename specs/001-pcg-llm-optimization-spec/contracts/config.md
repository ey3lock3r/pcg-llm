# Contract: TrainingConfig Schema

**Feature**: `001-pcg-llm-optimization-spec`
**Date**: 2026-04-04

---

## Overview

`TrainingConfig` is a Python dataclass (frozen, with `__post_init__` validation) that serves as the single source of truth for all training parameters. It is:
- Constructed from CLI arguments or JSON config file
- Embedded in every checkpoint for exact reproducibility
- Passed to all modules (`PCGTrainer`, `CheckpointManager`, `DataCurriculum`, etc.) as a read-only object

---

## Validation Rules

All rules are enforced in `__post_init__`. Violations raise `ValueError` with a descriptive message.

| Rule | Condition |
|---|---|
| Data mixing sums to 1 | `sum(fineweb_fracs) + sum(stack_fracs) per epoch == 1.0` |
| Sparsity in range | `0.0 < initial_sparsity < 1.0` |
| Anderson window positive | `1 <= anderson_window <= 10` |
| Optimizer bits valid | `optimizer_bits in {32, 8}` |
| Normalize valid | `normalize in {"standard", "ngpt"}` |
| Projection valid | `projection in {"dense", "monarch"}` |
| Monarch min dim | If `projection == "monarch"`: `hidden_dim >= 64` |
| GCS path | If `checkpoint_backend == "gcs"`: `checkpoint_dir.startswith("gs://")` |
| Solver tolerance positive | `solver_tolerance > 0` |
| Block size divides seq_len | `max_seq_len % block_size == 0` |
| EAGLE draft len | `1 <= eagle_draft_len <= 16` |
| EAGLE K | `1 <= eagle_k <= 32` |
| LR positive | `base_lr > 0` |
| Grad accum positive | `grad_accum_steps >= 1` |

---

## Named Presets

Two built-in presets are available. All fields not listed use the dataclass defaults shown in data-model.md.

### `"tiny"` preset

```python
hidden_dim = 512
max_seq_len = 2048
block_size = 32
initial_sparsity = 0.70
max_solver_iters = 12
solver_tolerance = 1e-2
base_lr = 1e-3
warmup_steps = 500
total_tokens = 10_000_000_000  # 10B
batch_size = 8
grad_accum_steps = 8
checkpoint_interval = 500
checkpoint_backend = "local"
checkpoint_dir = "/kaggle/working/checkpoints"
optimizer = "muon_adamw"
optimizer_bits = 32
normalize = "ngpt"
projection = "monarch"
grad_checkpoint = True
cpu_offload_mask = False
wandb_project = "pcg-llm-tiny"
```

### `"3b"` preset

```python
hidden_dim = 3072
max_seq_len = 4096
block_size = 64
initial_sparsity = 0.90
max_solver_iters = 25
solver_tolerance = 1e-5
base_lr = 3e-4
warmup_steps = 2000
total_tokens = 100_000_000_000  # 100B
batch_size = 4
grad_accum_steps = 8
checkpoint_interval = 250
checkpoint_backend = "gcs"
checkpoint_dir = "gs://pcg-llm-checkpoints/3b"
optimizer = "muon_adamw"
optimizer_bits = 8
normalize = "ngpt"
projection = "monarch"
grad_checkpoint = True
cpu_offload_mask = True
wandb_project = "pcg-llm-3b"
```

---

## Stability Contract

- Architectural fields (`hidden_dim`, `max_seq_len`, `block_size`, `vocab_size`) are **immutable once a training run starts**. Changing them between a checkpoint and a resume causes an abort.
- Optimization flags (`optimizer`, `optimizer_bits`, `normalize`, `projection`, `grad_checkpoint`, `cpu_offload_mask`) CAN be changed between a checkpoint and a resume, but will log a prominent warning.
- Training hyperparameters (`base_lr`, `warmup_steps`, `lambda_sparse`, `gamma_variance`) CAN be changed between checkpoint and resume for mid-run tuning.
