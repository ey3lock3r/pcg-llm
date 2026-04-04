# Changelog

All notable changes to pcg-llm will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added — Feature 001: PCG-LLM Optimization & Free-Tier Training

#### New Source Modules
- `src/pcg_llm/config.py` — `TrainingConfig` frozen dataclass; `"tiny"` and `"3b"` named presets; full validation per `contracts/config.md`
- `src/pcg_llm/arch/node.py` — `PCGNode`: latent state estimator with Spectral Normalization; BF16-compatible
- `src/pcg_llm/arch/deq_solver.py` — `ConstrainedDEQSolver`: Anderson Acceleration (m=3, β=1e-4) with Broyden fallback; `apply_spectral_norm_constraint()`
- `src/pcg_llm/arch/adjacency.py` — `BlockSparseAdjacency`: Erdős–Rényi init, RigL Drop-and-Grow, `message_pass()`, `freeze()`
- `src/pcg_llm/arch/monarch.py` — `MonarchProjection`: two-factor butterfly decomposition via `torch.bmm`; `torch.compile`-compatible; optional per-factor spectral norm
- `src/pcg_llm/arch/eagle_head.py` — `EAGLEExtrapolationHead`: single transformer block on PCG penultimate states; generates draft tree `[B, K, draft_len, d]`; auto-expands draft_len 4→8 on acceptance EMA > 0.65
- `src/pcg_llm/training/loss.py` — `FreeEnergyLoss`: CrossEntropy + λ·L1 + γ·max(0,τ−Var(Z*)); auto-γ nudge (+10%) after 10 consecutive low-variance steps
- `src/pcg_llm/training/optimizer.py` — `MuonOptimizer` (Newton-Schulz orthogonalization, 5 iterations); `HybridOptimizer` (Muon for `nn.Linear.weight`, AdamW for embeddings/biases)
- `src/pcg_llm/training/normalization.py` — `nGPTNorm`: hyperspherical unit-norm; `LayerNorm` fallback; `build_norm()` factory
- `src/pcg_llm/training/scheduler.py` — `RigLSparsitySchedule`: cosine decay 0.30→0 over `freeze_step_frac × total_steps` steps
- `src/pcg_llm/training/curriculum.py` — `DataCurriculum`: per-epoch FineWeb-Edu/Stack v2 mixing (92%→85%→70% / 8%→15%→30%)
- `src/pcg_llm/training/trainer.py` — `PCGTrainer`: full training loop; `resume_if_available()`; `train()`; `generate()`; `fine_tune_eagle()`; W&B integration; SIGTERM-aware checkpointing
- `src/pcg_llm/checkpointing/local.py` — `LocalCheckpointBackend`: atomic `.pt.tmp`→`.pt` write; disk quota monitoring (warn <1 GB, critical <500 MB)
- `src/pcg_llm/checkpointing/gcs.py` — `GCSCheckpointBackend`: atomic GCS writes via temp blob + `rewrite()` rename; exponential backoff retry (max 5, base 1s, max 32s)
- `src/pcg_llm/checkpointing/checkpoint.py` — `CheckpointManager`: SHA-256 verification; atomic manifest updates; fall-back-to-previous on corruption
- `src/pcg_llm/checkpointing/signals.py` — `SignalHandler`: SIGTERM handler with GCP preemption deadline tracking (20s soft warn, 25s hard limit); exit code 2
- `src/pcg_llm/data/tokenizer.py` — `Llama3TokenizerWrapper`: HuggingFace Llama-3 tokenizer; vocab size 128256
- `src/pcg_llm/data/streaming.py` — `HuggingFaceStreamingDataset`: interleaved FineWeb-Edu + Stack v2 streaming with configurable mixing ratios
- `src/pcg_llm/evaluation/harness.py` — `BenchmarkHarness`: wraps `lm_eval`; 4-bit quantization via bitsandbytes; graceful ImportError fallback
- `src/pcg_llm/monitoring/metrics.py` — `TrainingMetrics`: per-step metrics, alert thresholds, `to_dict()` for W&B

#### CLI (`src/pcg_llm/__main__.py`, `src/pcg_llm/_cli.py`, `main.py`)
- `train` subcommand — full flag set per `contracts/cli.md`; `--preset tiny|3b`; `--resume/--no-resume`
- `evaluate` subcommand — run lm-eval benchmarks; `--quantize 4bit|8bit|none`
- `export-config` subcommand — serialize resolved config to JSON
- Exit codes: 0 (success), 1 (fatal error), 2 (graceful SIGTERM shutdown)

#### Checkpoint Format (schema version 1.0)
- 28 required keys including SHA-256, RigL state, Anderson iterates, curriculum, RNG state
- Atomic write: `.pt.tmp` → `.pt` → `manifest.json`
- Architectural field immutability check on resume

#### Test Suite
- `tests/conftest.py` — shared fixtures: `tiny_config`, `temp_checkpoint_dir`, `mock_dataset_batch`
- `tests/unit/` — 9 unit test modules covering config, tokenizer, streaming, DEQ solver, adjacency, loss, checkpoint, Muon optimizer, Monarch projection, metrics
- `tests/integration/` — 5 integration test modules: training loop, resume, EAGLE head, GCS resume, 3B smoke
- `tests/gpu/` — 2 GPU test modules: block-sparse matmul, FlexAttention mask

#### Benchmarks (`benchmarks/`)
- `bench_deq_solver.py` — wall-clock + peak VRAM for DEQ solver calls
- `bench_rigl_cycle.py` — Drop-and-Grow step timing at 90% sparsity
- `bench_eagle_throughput.py` — EAGLE draft tree throughput
- `benchmarks/README.md` — run instructions and constitutional Principle V requirements

#### Dependency Upgrades (`pyproject.toml`)
- `torch`: `>=2.4.0` → `>=2.5.0` (FlexAttention hard dependency)
- New optional extras: `training` (bitsandbytes, datasets, transformers, wandb), `eval` (lm-eval), `gcp` (google-cloud-storage, tenacity), `all`

### Added — Bootstrap (pre-existing)
- Initial project bootstrap with spec-driven development workflow
- Project constitution defining non-negotiable principles
- Python project structure with `src/pcg_llm/` layout
- Quality tooling: Ruff, Mypy, Pytest, Bandit
- Pre-commit hooks for automated quality enforcement
- GitHub Actions CI/CD pipeline
- Spec templates: features, RFCs, ADRs

---

## [0.1.0] - 2026-04-04

### Added
- Initial project scaffold
- Core project structure

---

[Unreleased]: https://github.com/your-username/pcg-llm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/pcg-llm/releases/tag/v0.1.0
