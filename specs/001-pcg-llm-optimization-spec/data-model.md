# Data Model: PCG-LLM Optimization & Free-Tier Training

**Feature**: `001-pcg-llm-optimization-spec`
**Date**: 2026-04-04

---

## Core Entities

### 1. TrainingConfig

The central configuration dataclass. All size-specific, optimization-specific, and platform-specific settings live here. Serialized to JSON for checkpoint inclusion.

| Field | Type | Default (Tiny) | Default (3B) | Description |
|---|---|---|---|---|
| `hidden_dim` | `int` | 512 | 3072 | Node state dimension d |
| `max_seq_len` | `int` | 2048 | 4096 | Maximum context length N |
| `vocab_size` | `int` | 128256 | 128256 | Llama-3 Tiktoken vocab |
| `block_size` | `int` | 32 | 64 | RigL block side length (square) |
| `initial_sparsity` | `float` | 0.70 | 0.90 | Erdős–Rényi init sparsity |
| `max_solver_iters` | `int` | 12 | 25 | DEQ solver iteration cap |
| `solver_tolerance` | `float` | 1e-2 | 1e-5 | DEQ convergence threshold ε |
| `anderson_window` | `int` | 3 | 3 | Anderson history buffer size m |
| `anderson_beta` | `float` | 1e-4 | 1e-4 | Anderson regularization |
| `lipschitz_target` | `float` | 1.0 | 1.0 | Spectral norm upper bound |
| `lambda_sparse` | `float` | 0.01 | 0.01 | L1 sparsity penalty coefficient |
| `gamma_variance` | `float` | 0.01 | 0.01 | Variance hinge coefficient |
| `variance_floor` | `float` | 0.1 | 0.1 | Node state variance threshold τ |
| `rigl_interval` | `int` | 100 | 100 | Drop-and-Grow evaluation interval |
| `rigl_drop_fraction` | `float` | 0.10 | 0.10 | Fraction of blocks to drop/grow |
| `rigl_reroute_start` | `float` | 0.30 | 0.30 | Initial re-routing fraction |
| `rigl_freeze_step_frac` | `float` | 0.80 | 0.80 | Fraction of total steps at freeze |
| `eagle_k` | `int` | 8 | 8 | Draft tree branching factor K |
| `eagle_draft_len` | `int` | 4 | 4 | Initial draft sequence length |
| `eagle_accept_threshold` | `float` | 0.65 | 0.65 | Acceptance rate gate for L expansion |
| `base_lr` | `float` | 1e-3 | 3e-4 | Base learning rate |
| `warmup_steps` | `int` | 500 | 2000 | LR warmup steps |
| `optimizer` | `str` | `"muon_adamw"` | `"muon_adamw"` | `"adamw"` or `"muon_adamw"` |
| `optimizer_bits` | `int` | 32 | 8 | Optimizer state precision (32 or 8) |
| `normalize` | `str` | `"ngpt"` | `"ngpt"` | `"standard"` or `"ngpt"` |
| `projection` | `str` | `"monarch"` | `"monarch"` | `"dense"` or `"monarch"` |
| `grad_checkpoint` | `bool` | `True` | `True` | Enable gradient checkpointing |
| `cpu_offload_mask` | `bool` | `False` | `True` | Offload RigL mask to CPU |
| `grad_accum_steps` | `int` | 8 | 8 | Gradient accumulation steps |
| `batch_size` | `int` | 8 | 4 | Per-GPU micro-batch size |
| `checkpoint_interval` | `int` | 500 | 250 | Steps between checkpoint saves |
| `checkpoint_backend` | `str` | `"local"` | `"gcs"` | `"local"` or `"gcs"` |
| `checkpoint_dir` | `str` | `"/kaggle/working/checkpoints"` | `"gs://bucket/checkpoints"` | Checkpoint storage path |
| `dataset_fineweb_frac` | `list[float]` | `[0.92, 0.85, 0.70]` | same | Per-epoch FineWeb-Edu fraction |
| `dataset_stack_frac` | `list[float]` | `[0.08, 0.15, 0.30]` | same | Per-epoch Stack v2 fraction |
| `total_tokens` | `int` | 10B | 100B | Total training tokens |
| `wandb_project` | `str` | `"pcg-llm-tiny"` | `"pcg-llm-3b"` | W&B project name (or `None`) |

**Validation rules**:
- `dataset_fineweb_frac[i] + dataset_stack_frac[i]` MUST equal 1.0 for all i
- `initial_sparsity` MUST be in (0.0, 1.0)
- `anderson_window` MUST be in [1, 10]
- `eagle_draft_len` MUST be in [1, 16]
- `optimizer_bits` MUST be 32 or 8
- `checkpoint_backend == "gcs"` requires `checkpoint_dir` to start with `"gs://"`

---

### 2. PCGNodeState (Z*)

The DEQ fixed-point tensor. Not persisted directly (recomputed from model weights), but its shape and dtype are part of the model's invariants.

| Attribute | Value |
|---|---|
| Shape | `[batch_size, seq_len, hidden_dim]` |
| Dtype | `bfloat16` |
| Representation | Unit-norm vectors when nGPT normalization is active; unconstrained otherwise |
| Lifecycle | Computed by DEQ solver during forward pass; not stored between batches |
| Collapse guard | `Var(Z*)` must stay ≥ `variance_floor` at all times after warmup |

---

### 3. BlockSparseAdjacency (A_evolved)

The learned graph topology. The primary data structure modified by RigL.

| Attribute | Value |
|---|---|
| Shape | `[num_blocks, num_blocks]` where `num_blocks = seq_len / block_size` |
| Dtype | `bool` (mask) + `float32` (structural weights W_structure) |
| Initial state | Erdős–Rényi random graph at configured sparsity level |
| Block size | 32×32 (Tiny PCG) or 64×64 (3B PCG) |
| Memory layout | Contiguous row-major; GPU SRAM-aligned for Triton kernel access |
| Sparsity constraint | Maintained in [0.80, 0.98] throughout non-freeze training |
| Freeze condition | `current_step >= rigl_freeze_step_frac × total_steps` |

**State transitions**:
```
INIT (Erdős–Rényi, 70% or 90% sparse)
  → TRAINING (Drop-and-Grow every 100 steps, re-routing fraction cosine-decays)
  → FROZEN (re-routing fraction reaches 0%, no more structural changes)
```

---

### 4. CheckpointManifest

The authoritative index of all available checkpoints in a checkpoint directory.

| Field | Type | Description |
|---|---|---|
| `version` | `str` | Manifest schema version (e.g., `"1.0"`) |
| `checkpoints` | `list[CheckpointEntry]` | Ordered list, most recent last |
| `latest_valid_step` | `int` | Step number of the most recent SHA-256 verified checkpoint |

**CheckpointEntry**:

| Field | Type | Description |
|---|---|---|
| `step` | `int` | Global training step at checkpoint time |
| `path` | `str` | Absolute or GCS path to the `.pt` file |
| `sha256` | `str` | SHA-256 hex digest of the `.pt` file |
| `timestamp` | `str` | ISO 8601 UTC timestamp of checkpoint write |
| `loss_total` | `float` | Total loss value at checkpoint step |
| `sparsity` | `float` | Measured adjacency sparsity at checkpoint step |
| `solver_steps_mean` | `float` | Mean DEQ solver steps over last 100 steps |
| `valid` | `bool` | False if checksum verification failed on last load attempt |

**State transitions**:
```
WRITTEN (.pt.tmp) → RENAMED (.pt) → MANIFEST_UPDATED → VALID
                                                      ↘ INVALID (checksum mismatch on load)
```

---

### 5. EAGLEDraftTree

The speculative decoding draft produced by the EAGLE Extrapolation Head.

| Attribute | Value |
|---|---|
| Shape | `[batch_size, K, draft_len, hidden_dim]` — draft in latent space |
| Token shape | `[batch_size, K, draft_len]` — token IDs after projection |
| K | 8 (branching factor; configurable) |
| draft_len | 4 (initial) → 8 (after fine-tuning gate passes) |
| Acceptance criteria | Longest contiguous path where PCG energy constraints are met |
| Fallback | If all K branches rejected → 3-step DEQ iteration for next token |

**Acceptance state per branch**:
```
DRAFT → VERIFIED (all tokens in path accepted) → OUTPUT
      → PARTIAL_REJECT (tokens accepted up to position j, rejected from j+1) → PRUNE + REDRAFT
      → FULL_REJECT (token 0 rejected) → FULL_FALLBACK (3-step DEQ)
```

---

### 6. TrainingMetrics

The set of custom metrics logged per training step for health monitoring.

| Metric | Type | Alert Threshold | Description |
|---|---|---|---|
| `solver_steps_mean` | `float` | > 80% of max_solver_iters | Average DEQ solver steps per token |
| `solver_steps_p95` | `float` | > max_solver_iters | 95th percentile solver steps |
| `sparsity` | `float` | < 0.80 or > 0.98 | Measured adjacency mask sparsity |
| `node_variance` | `float` | < variance_floor (0.1) | Mean node state variance Var(Z*) |
| `eagle_accept_rate` | `float` | < 0.60 (warning) | EAGLE draft acceptance rate over last 1,000 steps |
| `gamma_current` | `float` | — | Current γ coefficient (auto-adjusted on collapse) |
| `loss_crossentropy` | `float` | — | CrossEntropy component |
| `loss_sparsity` | `float` | — | L1 sparsity penalty component |
| `loss_variance` | `float` | — | Variance Hinge component (negative term) |
| `loss_total` | `float` | — | Combined loss |
| `rigl_reroute_fraction` | `float` | — | Current RigL re-routing fraction |
| `rigl_frozen` | `bool` | — | Whether graph topology is frozen |
| `step` | `int` | — | Global training step |
| `tokens_seen` | `int` | — | Total tokens processed |
| `throughput_tps` | `float` | — | Training throughput in tokens/second |
| `vram_gb` | `float` | — | Peak VRAM usage (GPU 0) in GB |

---

## Entity Relationships

```
TrainingConfig
    ├── controls → BlockSparseAdjacency (initial_sparsity, block_size, rigl_*)
    ├── controls → DEQ Solver (anderson_window, solver_tolerance, max_solver_iters)
    ├── controls → EAGLEDraftTree (eagle_k, eagle_draft_len)
    └── controls → CheckpointManifest (checkpoint_interval, checkpoint_backend, checkpoint_dir)

PCGTrainer
    ├── reads → TrainingConfig
    ├── writes → CheckpointManifest (via CheckpointManager)
    ├── produces → PCGNodeState (Z*) per forward pass
    ├── updates → BlockSparseAdjacency (via RigL Drop-and-Grow every 100 steps)
    └── logs → TrainingMetrics (every step)

CheckpointManager
    ├── reads → TrainingConfig (checkpoint settings)
    ├── manages → CheckpointManifest
    └── delegates → LocalCheckpointBackend | GCSCheckpointBackend

EAGLEHead
    ├── reads → PCGNodeState (penultimate layer)
    ├── produces → EAGLEDraftTree
    └── reports → TrainingMetrics.eagle_accept_rate
```
