# Contract: Training Script CLI Interface

**Feature**: `001-pcg-llm-optimization-spec`
**Date**: 2026-04-04

The PCG-LLM training script (`python -m pcg_llm.training.trainer` or `python main.py train`) exposes the following command-line interface. All arguments correspond directly to fields in `TrainingConfig`.

---

## Primary Commands

### `train` — Start or Resume Training

```
python -m pcg_llm.training.trainer train [OPTIONS]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | `str` | `None` | Path to a JSON config file (all other flags override it) |
| `--preset` | `str` | `None` | Named preset: `"tiny"` or `"3b"` (loads built-in defaults) |
| `--hidden-dim` | `int` | 512 | Node state dimension d |
| `--seq-len` | `int` | 2048 | Max training context length N |
| `--block-size` | `int` | 32 | RigL block side length |
| `--initial-sparsity` | `float` | 0.70 | Adjacency initialization sparsity |
| `--max-solver-iters` | `int` | 12 | DEQ solver iteration cap |
| `--solver-tol` | `float` | 1e-2 | DEQ convergence threshold |
| `--optimizer` | `str` | `"muon_adamw"` | Optimizer: `"adamw"` or `"muon_adamw"` |
| `--optimizer-bits` | `int` | 32 | Optimizer state bits: `32` or `8` |
| `--normalize` | `str` | `"ngpt"` | Normalization: `"standard"` or `"ngpt"` |
| `--projection` | `str` | `"monarch"` | Projection: `"dense"` or `"monarch"` |
| `--base-lr` | `float` | 1e-3 | Base learning rate |
| `--warmup-steps` | `int` | 500 | LR warmup steps |
| `--total-tokens` | `str` | `"10B"` | Total training tokens (supports `"10B"`, `"100B"`, etc.) |
| `--batch-size` | `int` | 8 | Per-GPU micro-batch size |
| `--grad-accum` | `int` | 8 | Gradient accumulation steps |
| `--grad-checkpoint` / `--no-grad-checkpoint` | `bool` | `True` | Enable gradient checkpointing |
| `--cpu-offload-mask` / `--no-cpu-offload-mask` | `bool` | `False` | Offload RigL mask to CPU |
| `--checkpoint-interval` | `int` | 500 | Steps between checkpoint saves |
| `--checkpoint-backend` | `str` | `"local"` | Checkpoint backend: `"local"` or `"gcs"` |
| `--checkpoint-dir` | `str` | `"/kaggle/working/checkpoints"` | Checkpoint storage path |
| `--resume` / `--no-resume` | `bool` | `True` | Auto-detect and resume from latest checkpoint |
| `--wandb-project` | `str` | `None` | W&B project name (`None` = disable W&B) |
| `--seed` | `int` | 42 | Global random seed |

**Exit codes**:
- `0`: Training completed successfully (total_tokens reached)
- `1`: Fatal error (config validation failure, checkpoint corruption, unrecoverable DEQ divergence)
- `2`: Graceful shutdown (SIGTERM received; checkpoint saved before exit)

---

### `evaluate` — Run Benchmark Evaluation

```
python -m pcg_llm.training.trainer evaluate [OPTIONS]
```

| Argument | Type | Required | Description |
|---|---|---|---|
| `--checkpoint` | `str` | Yes | Path to a `.pt` checkpoint file |
| `--tasks` | `str` | No | Comma-separated benchmark tasks (default: `"arc_challenge,gsm8k,humaneval,hellaswag,mmlu"`) |
| `--quantize` | `str` | No | Quantization: `"none"`, `"4bit"`, `"8bit"` (default: `"4bit"`) |
| `--output` | `str` | No | Path to write results JSON (default: stdout) |

---

### `export-config` — Export Active Config to JSON

```
python -m pcg_llm.training.trainer export-config [OPTIONS]
```

Writes the fully-resolved `TrainingConfig` (after applying preset + overrides) to a JSON file. Useful for reproducibility — the exported config can be passed back via `--config` on a future run to exactly reproduce the configuration.

| Argument | Type | Description |
|---|---|---|
| `--preset` | `str` | Named preset to export |
| `--output` | `str` | Output path (default: `config.json`) |
| (all `train` options) | — | Config overrides applied before export |

---

## Config File Format (JSON)

When `--config path/to/config.json` is used, the file MUST be valid JSON matching the `TrainingConfig` schema. All fields are optional; unspecified fields use the preset or hardcoded defaults.

```json
{
  "hidden_dim": 512,
  "max_seq_len": 2048,
  "block_size": 32,
  "initial_sparsity": 0.70,
  "optimizer": "muon_adamw",
  "optimizer_bits": 8,
  "normalize": "ngpt",
  "projection": "monarch",
  "checkpoint_backend": "local",
  "checkpoint_dir": "/kaggle/working/checkpoints",
  "checkpoint_interval": 500,
  "grad_checkpoint": true,
  "grad_accum_steps": 8,
  "wandb_project": "pcg-llm-tiny"
}
```

---

## Invariants (must hold for all valid invocations)

1. `--checkpoint-backend gcs` requires `--checkpoint-dir` to start with `"gs://"`.
2. `--optimizer-bits 8` requires `bitsandbytes` to be installed; the script checks at startup and exits with code 1 and a clear error message if not.
3. `--normalize ngpt` is incompatible with `--projection dense` when `hidden_dim < 64` (Monarch butterfly requires at least 64-dimensional blocks). The script validates this and exits with code 1.
4. `--resume` (default) is a no-op when no checkpoint exists in `--checkpoint-dir`; it does NOT error.
5. The `--preset` flag sets a base configuration; all other explicit flags override preset values.
