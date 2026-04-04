# Contract: Checkpoint Format Schema

**Feature**: `001-pcg-llm-optimization-spec`
**Date**: 2026-04-04

---

## Checkpoint File (`.pt`)

Each checkpoint is a `torch.save`-serialized Python dict. The file is written atomically (`.pt.tmp` → `.pt` rename). The dict MUST contain all keys listed below; missing keys are treated as a corrupt checkpoint and the file is marked invalid in the manifest.

### Top-Level Schema

```python
{
    # Identity
    "schema_version":        str,    # e.g. "1.0" — bump on breaking changes
    "step":                  int,    # Global training step (0-indexed)
    "config":                dict,   # Full TrainingConfig serialized to dict

    # Model
    "model_state_dict":      OrderedDict,   # torch state_dict, BF16 weights

    # Optimizer
    "optimizer_state_dict":  dict,          # torch optimizer state_dict

    # RigL topology
    "rigl_mask":             torch.Tensor,  # bool, shape [num_blocks, num_blocks]
    "rigl_weights":          torch.Tensor,  # float32 structural weights W_structure
    "rigl_schedule_step":    int,           # steps taken within current 100-step interval
    "rigl_rerouting_fraction": float,       # current cosine-decayed re-routing fraction
    "rigl_frozen":           bool,          # True when topology is permanently frozen

    # DEQ Solver (Anderson Acceleration)
    "anderson_iterates":     list,          # list of m tensors, each shape [B, N, d] BF16
    "anderson_residuals":    list,          # list of m tensors, matching iterates

    # Training curriculum
    "curriculum_epoch":      int,           # current epoch index (0-based)
    "curriculum_shard_index": int,          # dataset shard index within current epoch
    "curriculum_gating_temp": float,        # current gating temperature

    # Loss tracking
    "loss_crossentropy":     float,
    "loss_sparsity":         float,
    "loss_variance_hinge":   float,
    "loss_total":            float,
    "gamma_current":         float,         # current γ (may differ from config if auto-nudged)

    # EAGLE head state
    "eagle_draft_len":       int,           # current draft length (4 or 8)
    "eagle_accept_rate_ema": float,         # exponential moving average of acceptance rate

    # RNG state (for exact reproducibility)
    "rng_torch":             dict,          # torch.get_rng_state() per device
    "rng_numpy":             bytes,         # numpy.random.get_state() pickled
    "rng_python":            tuple,         # random.getstate()

    # Metadata
    "timestamp_utc":         str,           # ISO 8601 UTC, e.g. "2026-04-04T12:00:00Z"
    "hostname":              str,           # for debugging multi-instance runs
    "sha256":                str,           # SHA-256 of the file itself (computed before rename)
}
```

---

## Manifest File (`manifest.json`)

The manifest is written atomically (temp file + rename) after every successful checkpoint write. It is the authoritative discovery mechanism for the resume logic.

```json
{
  "schema_version": "1.0",
  "latest_valid_step": 1000,
  "checkpoints": [
    {
      "step": 500,
      "path": "/kaggle/working/checkpoints/step-000500.pt",
      "sha256": "a3f2...",
      "timestamp_utc": "2026-04-04T10:00:00Z",
      "loss_total": 4.21,
      "sparsity": 0.891,
      "solver_steps_mean": 6.3,
      "valid": true
    },
    {
      "step": 1000,
      "path": "/kaggle/working/checkpoints/step-001000.pt",
      "sha256": "b7c1...",
      "timestamp_utc": "2026-04-04T11:00:00Z",
      "loss_total": 3.87,
      "sparsity": 0.894,
      "solver_steps_mean": 5.9,
      "valid": true
    }
  ]
}
```

---

## Invariants

1. `sha256` in the manifest entry MUST match the SHA-256 of the `.pt` file at the time of the manifest write.
2. A `.pt` file with no corresponding manifest entry MUST be ignored by the resume logic (may be an orphaned `.tmp` file that was renamed but whose manifest update was interrupted).
3. `latest_valid_step` MUST equal the `step` field of the most recent entry where `valid == true`.
4. Manifest entries are ordered by ascending `step`. The resume logic selects the last entry with `valid == true`.
5. When a checkpoint is found corrupt (`valid` set to `false`), the manifest entry MUST NOT be removed — it must be kept as a record of the corruption event for debugging.
6. The `config` field in the checkpoint dict MUST be compared against the current `TrainingConfig` on resume. If any architectural field (`hidden_dim`, `block_size`, `vocab_size`) differs, training MUST abort with a clear error; these cannot be changed mid-run.
