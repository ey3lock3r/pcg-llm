# Tasks: PCG-LLM Optimization Research & Free-Tier Training

**Input**: Design documents from `specs/001-pcg-llm-optimization-spec/`
**Prerequisites**: plan.md ✅ | spec.md ✅ | research.md ✅ | data-model.md ✅ | contracts/ ✅ | quickstart.md ✅

**Tests**: Constitution Principle II mandates TDD (Red-Green-Refactor). Test tasks are included and MUST be written before their corresponding implementation tasks.

**Organization**: Tasks are grouped by user story for independent delivery.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on concurrently executing tasks)
- **[Story]**: User story this task belongs to (US1–US4)
- All paths are relative to repository root

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Upgrade dependencies, create directory skeleton, configure test infrastructure.

- [X] T001 Update `pyproject.toml`: bump `torch>=2.5.0`, add `bitsandbytes>=0.43.0`, `datasets>=2.19.0`, `transformers>=4.40.0`, `wandb>=0.17.0`, `lm-eval>=0.4.0`, `google-cloud-storage>=2.16.0` as optional extras; add pytest markers (`convergence`, `gpu`, `slow`) to `[tool.pytest.ini_options]`
- [X] T002 Create package directory skeleton: `src/pcg_llm/arch/`, `src/pcg_llm/training/`, `src/pcg_llm/checkpointing/`, `src/pcg_llm/data/`, `src/pcg_llm/evaluation/`, `src/pcg_llm/monitoring/`, `tests/unit/`, `tests/integration/`, `tests/gpu/`, `benchmarks/` — each with an `__init__.py`
- [X] T003 [P] Create `tests/conftest.py`: shared pytest fixtures (tiny config fixture, temp checkpoint dir fixture, mock dataset fixture, CUDA availability skip marker)
- [X] T004 [P] Create `benchmarks/README.md`: document how to run benchmarks and capture baseline, referencing constitution Principle V requirement

**Checkpoint**: Directory structure in place, test infrastructure ready, dependencies declared.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core configuration dataclass and data infrastructure that every user story depends on. MUST be complete before any user story phase begins.

**⚠️ CRITICAL**: No user story work can start until this phase is complete.

- [X] T005 Write failing tests for `TrainingConfig` in `tests/unit/test_config.py`: validation rules from `contracts/config.md` (mixing fractions sum to 1.0, `block_size` divides `max_seq_len`, `optimizer_bits` in {32,8}, `checkpoint_backend=="gcs"` requires `gs://` prefix, Monarch min-dim guard, all invalid inputs raise `ValueError`)
- [X] T006 Implement `src/pcg_llm/config.py`: `TrainingConfig` frozen dataclass with all fields from `data-model.md` entity 1, `__post_init__` validation matching `contracts/config.md`, and `"tiny"` / `"3b"` named presets; pass T005
- [X] T007 [P] Write failing tests for the tokenizer in `tests/unit/test_tokenizer.py`: encode/decode round-trip, vocab size assertion (128256), special token handling
- [X] T008 [P] Implement `src/pcg_llm/data/tokenizer.py`: `Llama3TokenizerWrapper` wrapping HuggingFace `transformers` Llama-3 tokenizer; pass T007
- [X] T009 Write failing tests for dataset streaming in `tests/unit/test_streaming.py`: stream yields token tensors of correct shape, epoch boundary advances shard index, mixing ratio produces correct FineWeb-Edu vs Stack v2 fraction over 1,000 samples
- [X] T010 Implement `src/pcg_llm/data/streaming.py`: `HuggingFaceStreamingDataset` — streams FineWeb-Edu and The Stack v2 via HuggingFace `datasets` streaming API with configurable mixing ratios from `TrainingConfig.dataset_fineweb_frac` / `dataset_stack_frac`; pass T009

**Checkpoint**: Config validated, tokenizer working, dataset streaming verified. User story phases can now begin in parallel where their prerequisites allow.

---

## Phase 3: User Story 1 — Tiny PCG Free-Tier Proof of Concept (Priority: P1) 🎯 MVP

**Goal**: A complete, checkpoint-resilient training loop for the Tiny PCG (50–100M params) that runs on 2× T4 Kaggle GPUs and resumes from interruption.

**Independent Test**: `pytest tests/unit/ tests/integration/test_training_loop.py tests/integration/test_resume.py -m "not gpu and not slow"` for unit validation; then run full Kaggle Notebook smoke test per `quickstart.md` Track A.

### Tests for User Story 1 (TDD — write before implementation)

- [X] T011 [P] [US1] Write failing convergence tests in `tests/unit/test_deq_solver.py` (`@pytest.mark.convergence`): Anderson acceleration converges for a 2-layer linear system within 12 iterations at m=3; Broyden fallback triggers when Anderson exceeds max iters; Spectral Normalization reduces singular value to ≤1.0 after one forward pass; fixed-point condition `||Z - f(Z,X)||` < `solver_tolerance` is satisfied at output
- [X] T012 [P] [US1] Write failing tests in `tests/unit/test_adjacency.py`: Erdős–Rényi init produces correct sparsity level (±2%); Drop phase removes exactly 10% of active blocks; Grow phase activates exactly 10% of dormant blocks with highest gradient norm; sparsity stays in [0.80, 0.98] after 10 Drop-and-Grow cycles; freeze flag prevents further updates
- [X] T013 [P] [US1] Write failing tests in `tests/unit/test_loss.py`: CrossEntropy term is correct for known logits; L1 sparsity penalty equals sum of absolute structural weights; Variance Hinge term is zero when variance > τ and positive when variance < τ; combined loss decreases when model improves; anti-collapse guard triggers γ nudge when variance < τ
- [X] T014 [P] [US1] Write failing tests in `tests/unit/test_checkpoint.py`: checkpoint dict contains all required keys from `contracts/checkpoint.md`; SHA-256 in manifest matches file; atomic write leaves no `.pt.tmp` on success; corrupt `.pt` file (truncated) is detected via checksum and fallback to previous is triggered; manifest `latest_valid_step` equals most recent valid entry; resume loads from correct step

### Implementation for User Story 1

- [X] T015 [US1] Implement `src/pcg_llm/arch/node.py`: `PCGNode` — latent state estimator with configurable `hidden_dim`; BF16 parameters; Spectral Normalization wrapper applied to weight matrix W via `torch.nn.utils.spectral_norm`; `forward(x, neighbors)` → updated state with local prediction error `eᵢ`; pass relevant T011 sub-tests
- [X] T016 [US1] Implement `src/pcg_llm/arch/deq_solver.py`: `ConstrainedDEQSolver` — wraps `torchdeq` Anderson Acceleration solver with `m=3`, `β=1e-4`, `max_solver_iters` from `TrainingConfig` (passed as `max_iter` to torchdeq API); `solve(f_theta, z0, x)` returns fixed point `Z*`; Spectral Normalization of `f_theta` weights checked before each call; Broyden fallback on Anderson non-convergence; logs solver step count per call; pass T011
- [X] T017 [US1] Implement `src/pcg_llm/arch/adjacency.py`: `BlockSparseAdjacency` — stores `bool` mask and `float32` W_structure; `initialize_erdos_renyi(sparsity)` constructor; `message_pass(Z)` → aggregated neighbor states via block-sparse matmul (Triton kernel when available, dense fallback); gradient computation for Grow phase; `drop_and_grow(gradients)` method implementing RigL one cycle; `freeze()` method; pass T012
- [X] T018 [US1] Implement `src/pcg_llm/training/loss.py`: `FreeEnergyLoss` — three-term loss from spec FR-008: `CrossEntropy(Z*, Y) + λ·||A||₁ − γ·max(0, τ − Var(Z*))` with automatic γ nudge (+10%) when variance < τ for 10 consecutive steps; emits variance and γ values to `TrainingMetrics`; pass T013
- [X] T019 [US1] Implement `src/pcg_llm/training/scheduler.py`: `RigLSparsitySchedule` — cosine decay of re-routing fraction from `rigl_reroute_start` (0.30) to 0.0 over `rigl_freeze_step_frac × total_steps` steps; `step()` returns current fraction and freeze flag; serializable state dict for checkpointing; pass existing tests
- [X] T020 [US1] Implement `src/pcg_llm/training/curriculum.py`: `DataCurriculum` — manages epoch index, shard index, and per-epoch mixing ratios from `TrainingConfig`; `advance_shard()` increments shard counter; `get_mixing_ratio(epoch)` returns `(fineweb_frac, stack_frac)`; serializable state for checkpointing; pass existing tests
- [X] T021 [US1] Implement `src/pcg_llm/checkpointing/local.py`: `LocalCheckpointBackend` — writes `{step}.pt.tmp` then renames to `{step}.pt`; monitors disk quota via `shutil.disk_usage` and triggers shutdown callback when < 500MB free; `list_checkpoints()` returns all `.pt` files in checkpoint dir (ignores `.tmp`); pass T014 sub-tests
- [X] T022 [US1] Implement `src/pcg_llm/checkpointing/checkpoint.py`: `CheckpointManager` — orchestrates atomic save (build dict → write via backend → compute SHA-256 → update manifest); `load_latest()` reads manifest, verifies checksum of most recent valid entry, falls back to previous on mismatch; `build_checkpoint_dict(trainer, step)` assembles all required keys from `contracts/checkpoint.md`; pass T014
- [X] T023 [US1] Implement `src/pcg_llm/checkpointing/signals.py`: `SignalHandler` — registers `signal.SIGTERM` handler and `KeyboardInterrupt` catcher; on signal, calls `CheckpointManager.save()` then exits with code 2; logs "SIGTERM received, saving checkpoint..." before save
- [X] T024 [US1] Implement `src/pcg_llm/training/trainer.py`: `PCGTrainer` — main training loop; initializes `PCGNode`, `ConstrainedDEQSolver`, `BlockSparseAdjacency`, `FreeEnergyLoss`, `RigLSparsitySchedule`, `DataCurriculum`, `CheckpointManager`, `SignalHandler` from `TrainingConfig`; inner loop: tokenize batch → DEQ solve → compute loss → backward → optimizer step → RigL every 100 steps → checkpoint every N steps; emits `TrainingMetrics` each step; `resume_if_available()` on startup; pass integration tests *(Amended 2026-04-04: extended to implement DDP multi-GPU support — auto-detection of `LOCAL_RANK`, `DistributedDataParallel` wrapping of `PCGNode` and `output_proj`, `W_structure.requires_grad_(True)` fix, `W_structure.grad` manual `all_reduce` in `_optimizer_step`, RigL mask broadcast after `drop_and_grow`, rank-0-only checkpointing/logging/W&B; raw refs `_raw_node`/`_raw_output_proj` kept for checkpoint save/load)*
- [X] T025 [US1] Implement `src/pcg_llm/__main__.py` and `main.py`: CLI entry point with `train`, `evaluate`, `export-config` subcommands per `contracts/cli.md`; `--preset tiny/3b` flag loading named configs; all flags from cli contract; `--resume` default True; proper exit codes (0, 1, 2)
- [X] T026 [US1] Write and pass integration test `tests/integration/test_training_loop.py` (`@pytest.mark.slow`): run Tiny PCG for 100 steps on synthetic data; assert loss decreases, sparsity in range, checkpoint written at step 50 and 100, solver steps ≤ 15 (matching SC-003: mean solver steps ≤ 15 out of max 25)
- [X] T027 [US1] Write and pass integration test `tests/integration/test_resume.py` (`@pytest.mark.slow`): run for 150 steps, interrupt at step 100 by raising `KeyboardInterrupt`, resume, assert training continues from step 100 with matching loss value; assert no `.pt.tmp` orphan files remain
- [X] T054 [P] [US1] Write unit tests for `PCGNode` in `tests/unit/test_node.py`: `forward(x, neighbors)` returns correct output shape `[B, N, d]`; Spectral Normalization wrapper is present on the weight matrix; output BF16 dtype when model is in BF16; `eᵢ` (local prediction error) is non-zero for non-trivial inputs; gradient flows through `PCGNode.forward` (no detach)
- [X] T055 [P] [US1] Write and pass unit tests for FR-018 CPU offload in `tests/unit/test_adjacency.py`: `BlockSparseAdjacency` mask tensor can be moved to CPU mid-forward and back to GPU for drop-and-grow without data corruption; `message_pass(Z)` correctly handles Z on GPU while mask is on CPU (auto-moves mask to Z.device)
- [X] T056 [P] [US1] Write and pass unit tests for FR-019 gradient accumulation in `tests/unit/test_trainer_unit.py`: when `grad_accum_steps=4`, optimizer step is called once every 4 batches; loss value matches single-step with 4× batch size (up to BF16 tolerance); step counter increments every batch (not every optimizer step)

- [X] T060 [P] [US1] Capture DEQ solver baseline timing with all optimization flags OFF (v3.0 behavior): run `benchmarks/bench_deq_solver.py --flags-off` and write results to `benchmarks/results/deq_solver_baseline.json`; this MUST run before Phase 4 optimization tasks to establish the comparison baseline required by constitution Principle V and SC-007
- [X] T061 [P] [US1] Write DDP unit tests in `tests/unit/test_trainer_unit.py` (class `TestDDPSupport`): `_is_ddp=False` by default when `LOCAL_RANK` not set; `W_structure.requires_grad=True` after `__init__`; `W_structure.grad is not None` after `train_step` backward (regression for missing `requires_grad_(True)` bug); `_save_checkpoint` skipped when `_rank != 0`; `dist.all_reduce` called in `_optimizer_step` when DDP active; `dist.broadcast` called in `_broadcast_state_after_resume`; RigL mask broadcast triggered in `train_step` when DDP active; DDP+grad_accum optimizer step count verified (class `TestDDPGradAccumInteraction`)
- [X] T062 [P] [US1] Create `benchmarks/bench_ddp_vs_single.py`: measure tokens/s and peak VRAM for single-GPU vs DDP 2× on Tiny PCG dimensions; structured comparison table printed to stdout; writes `benchmarks/results/ddp_vs_single_baseline.json` with `--baseline` flag; skips DDP measurement gracefully when `torchrun` or 2nd GPU is unavailable; satisfies constitution Principle V for DDP throughput claims

**Checkpoint**: Tiny PCG trains end-to-end on CPU/single GPU, checkpoints atomically, and resumes correctly. US1 independently testable.

---

## Phase 4: User Story 2 — Additional Optimization Integration (Priority: P2)

**Goal**: All eight research-validated optimizations (Muon, nGPT, Monarch, FlexAttention, Anderson m=3, 8-bit optimizer, curriculum mixing, EAGLE draft calibration) are implemented as configurable flags and deliver ≥20% per-step speedup over baseline.

**Independent Test**: Run `pytest tests/unit/test_muon.py tests/unit/test_monarch.py tests/integration/test_eagle.py`; then run the benchmark suite and confirm ≥20% wall-clock improvement over baseline (all flags off = v3.0 behavior).

### Tests for User Story 2 (TDD — write before implementation)

- [X] T028 [P] [US2] Write failing tests in `tests/unit/test_muon.py`: Muon update for a 3×3 weight matrix is orthogonal (singular values of update matrix ≈ 1.0 after Newton-Schulz); Muon is NOT applied to 1-D bias tensors (must raise or skip); hybrid wrapper applies Muon to `nn.Linear.weight`, AdamW to `nn.Embedding.weight` and biases; after 10 steps, Muon model loss < AdamW-only model loss on synthetic quadratic task
- [X] T029 [P] [US2] Write failing tests in `tests/unit/test_monarch.py`: `MonarchProjection(512, 512)` output matches equivalent dense linear layer output for same weights (up to numerical tolerance 1e-4); FLOPs of `MonarchProjection(3072, 3072)` forward pass < FLOPs of `nn.Linear(3072, 3072)` forward pass (verified via `torch.profiler`); Spectral Normalization can be applied to each butterfly factor independently
- [X] T030 [P] [US2] Write failing tests in `tests/integration/test_eagle.py` (`@pytest.mark.slow`): `EAGLEExtrapolationHead` produces draft tree of shape `[B, K, L, d]`; Parallel Tree Attention verifies all K branches in a single forward pass; acceptance rate logging works; fallback to 3-step DEQ triggered when all branches rejected; draft length expands from 4 to 8 when acceptance rate EMA > 0.65

### Implementation for User Story 2

- [X] T031 [US2] Implement `src/pcg_llm/training/optimizer.py`: `MuonOptimizer` — Nesterov momentum + Newton-Schulz orthogonalization (5 iterations, standard approximation) for matrix parameters; `HybridOptimizer` — applies `MuonOptimizer` to all `nn.Linear.weight` parameters and `torch.optim.AdamW` (or `bitsandbytes.optim.AdamW8bit` when `optimizer_bits=8`) to all other parameters; automatic learning rate scaling (Muon LR = 10× base_lr); pass T028
- [X] T032 [US2] Implement `src/pcg_llm/training/normalization.py`: `nGPTNorm` — normalizes input tensor to unit norm along the last dimension; applied inside `f_theta` (after each message-passing step in `PCGNode`); compatible with BF16; when `normalize="ngpt"`, automatically scales base_lr by 10× in `PCGTrainer`; variance hinge interpretation updated (angular spread mode); pass existing unit tests
- [X] T033 [US2] Implement `src/pcg_llm/arch/monarch.py`: `MonarchProjection` — two sequential `torch.bmm` butterfly factor layers; weight init via `nn.init.orthogonal_` on each factor; `forward(x)` applies P₁·B₁·P₂·B₂ sequence; `torch.compile`-compatible (no Python loops in forward); Spectral Normalization on each factor independently; used in place of `nn.Linear` inside DEQ projection layers when `projection="monarch"`; pass T029
- [X] T034 [US2] Integrate Muon + nGPT + Monarch + 8-bit optimizer flags into `src/pcg_llm/training/trainer.py`: read flags from `TrainingConfig`; instantiate correct optimizer, normalization, and projection layers; log which optimizations are active at startup; ensure `projection="dense"` path produces identical output to pre-optimization baseline (regression test)
- [X] T035 [US2] Implement `src/pcg_llm/arch/eagle_head.py`: `EAGLEExtrapolationHead` — single transformer block (`nn.MultiheadAttention` + FFN) operating on PCG penultimate latent states; `generate_draft_tree(Z_penultimate)` → `EAGLEDraftTree` of shape `[B, K, draft_len, d]`; uses FlexAttention block-local mask (64-token window + global summary token at position 0) when `torch.version >= "2.5"`, falls back to causal mask otherwise; `accept_rate_ema` tracked and logged; `draft_len` auto-expanded when EMA > `eagle_accept_threshold`; pass T030
- [X] T036 [US2] Integrate EAGLE Head into inference path in `src/pcg_llm/training/trainer.py` and `__main__.py`: EAGLE head runs during `evaluate` subcommand; `PCGTrainer.generate(prompt_tokens)` method; acceptance rate reported to `TrainingMetrics`
- [X] T037 [US2] Create benchmark suite — `benchmarks/bench_deq_solver.py` (wall-clock + peak VRAM for 100 DEQ solve calls at Tiny PCG dimensions; baseline vs. optimized), `benchmarks/bench_rigl_cycle.py` (Drop-and-Grow step timing at 90% sparsity), `benchmarks/bench_eagle_throughput.py` (EAGLE draft + verification tokens/sec vs. greedy decoding); each benchmark prints a structured summary table and writes `benchmarks/results/{name}_baseline.json` on first run
- [X] T038 [P] [US2] Write and pass GPU tests `tests/gpu/test_block_sparse.py` and `tests/gpu/test_flex_attention.py` (`@pytest.mark.gpu`): block-sparse matmul via Triton kernel produces same output as dense equivalent (tolerance 1e-3); FlexAttention block-local mask has zero attention weight between tokens in different partitions and full attention within partition

**Checkpoint**: All optimization flags active, benchmarks show ≥20% improvement, EAGLE head generates and verifies draft trees. US2 independently testable.

---

## Phase 5: User Story 3 — GCP Preemptible 3B Model Training (Priority: P3)

**Goal**: Atomic GCS checkpoint writes with preemption resilience; automatic cross-instance resume; 3B preset validated.

**Independent Test**: `pytest tests/unit/test_checkpoint.py tests/integration/test_gcs_resume.py -k gcs`; then manually simulate preemption on a GCP instance per `quickstart.md` Track B, Step 4–5.

### Tests for User Story 3 (TDD — write before implementation)

- [X] T039 [P] [US3] Write failing tests in `tests/integration/test_gcs_resume.py` (`@pytest.mark.slow`): using `unittest.mock.patch` to mock the GCS SDK, verify atomic write (`.tmp` path written first, then renamed to final path); verify exponential-backoff retry fires on `google.api_core.exceptions.ServiceUnavailable`; verify resume from mocked GCS manifest selects correct latest-valid-step; verify corrupted checkpoint (wrong SHA-256) is skipped and previous valid checkpoint is loaded
- [X] T057 [P] [US3] Write unit tests for FR-017 gradient checkpointing correctness in `tests/unit/test_trainer_unit.py`: when `grad_checkpoint=True`, `PCGTrainer.train_step(batch)` produces numerically identical loss to `grad_checkpoint=False` (within BF16 tolerance 1e-2); verify peak VRAM is lower with checkpointing enabled (skip if not on GPU via `pytest.mark.gpu`)
- [X] T058 [P] [US3] Write unit tests for FR-025 disk quota trigger in `tests/unit/test_checkpoint.py`: mock `shutil.disk_usage` to return < 500MB free; assert `LocalCheckpointBackend` calls shutdown callback before write completes; mock returning < 1GB free and assert `DiskQuotaWarning` log message is emitted without shutdown
- [X] T059 [P] [US3] Write unit tests for FR-011 3-step DEQ fallback in `tests/unit/test_deq_solver.py`: when `PCGTrainer.generate()` falls back to DEQ on full EAGLE rejection, the solver is called with `max_solver_iters=3` cap regardless of configured production max; assert fallback produces a token (valid integer in vocab range)

### Implementation for User Story 3

- [X] T040 [US3] Implement `src/pcg_llm/checkpointing/gcs.py`: `GCSCheckpointBackend` — writes to GCS bucket using `google.cloud.storage` SDK; atomic write via temp blob name + `blob.rewrite()` rename; exponential backoff with `tenacity` or manual retry (max 5 attempts, base delay 1s, max delay 32s) for all GCS write operations; `list_checkpoints()` scans GCS bucket prefix; pass T039
- [X] T041 [US3] Extend `src/pcg_llm/checkpointing/signals.py`: add GCP preemption detection — register `SIGTERM` handler (GCP sends SIGTERM 30 seconds before reclaim); ensure checkpoint completes within 25 seconds (log warning at 20 seconds); set exit code 2 on clean checkpoint-save exit
- [X] T042 [US3] Add `gcp` optional dependency group to `pyproject.toml`: `google-cloud-storage>=2.16.0`, `tenacity>=8.2.0`; update `quickstart.md` Track B install step to use `pip install -e ".[gcp]"`
- [X] T043 [US3] Validate 3B preset end-to-end on CPU (smoke test, 10 steps): create `tests/integration/test_3b_smoke.py` (`@pytest.mark.slow`): instantiate full 3B `TrainingConfig` preset, run PCGTrainer for 10 steps on synthetic data, assert no OOM on CPU (uses grad checkpointing), assert checkpoint written, assert resume works

**Checkpoint**: GCS checkpoint backend implemented and tested with mocked GCS; SIGTERM handler saves within 25 seconds; 3B preset instantiates without error. US3 independently testable.

---

## Phase 6: User Story 4 — Training Health Monitoring (Priority: P4)

**Goal**: Real-time per-step metrics (solver steps, sparsity, node variance, EAGLE acceptance rate) logged to W&B or TensorBoard; auto-γ nudging; disk-quota guard; benchmark evaluation harness.

**Independent Test**: Run `pytest tests/unit/test_metrics.py`; then run a 200-step Tiny PCG training and confirm all four custom metrics appear in W&B dashboard.

### Tests for User Story 4 (TDD — write before implementation)

- [X] T044 [P] [US4] Write failing tests in `tests/unit/test_metrics.py`: `TrainingMetrics.record_step()` populates all fields from data-model entity 6; alert threshold check fires warning log when `solver_steps_mean > 0.8 × max_solver_iters`; alert fires when `node_variance < variance_floor`; EAGLE alert fires when `eagle_accept_rate_ema < 0.60` over 1,000-step window; γ auto-nudge in `FreeEnergyLoss` logs event to `TrainingMetrics`

### Implementation for User Story 4

- [X] T045 [US4] Implement `src/pcg_llm/monitoring/metrics.py`: `TrainingMetrics` — dataclass for all metrics from data-model entity 6; `record_step(step, solver_steps, sparsity, node_variance, eagle_rate, losses, gamma, rigl_state, vram_gb, throughput_tps)`; `check_alerts()` emits `logging.warning` for all threshold violations; `to_dict()` for W&B logging; pass T044
- [X] T046 [US4] Integrate W&B logging into `src/pcg_llm/training/trainer.py`: call `wandb.log(metrics.to_dict(), step=step)` when `config.wandb_project is not None`; `wandb.init()` at training start with config dict; W&B is optional (no error when `wandb` not installed — guard with `try/except ImportError`)
- [X] T047 [US4] Add disk quota monitor to `src/pcg_llm/checkpointing/local.py`: `_check_disk_quota()` called before each checkpoint write; raises `DiskQuotaWarning` (logged, not raised) at < 1GB free; calls `trainer.graceful_shutdown()` at < 500MB free (saves checkpoint then exits with code 2)
- [X] T048 [US4] Implement `src/pcg_llm/evaluation/harness.py`: `BenchmarkHarness` — wraps `lm_eval.evaluator.simple_evaluate()`; loads model from checkpoint path; applies 4-bit GPTQ quantization when `quantize="4bit"` via `bitsandbytes`; runs requested benchmark tasks; returns structured results dict; outputs JSON to file or stdout; used by `evaluate` CLI subcommand; guards with graceful ImportError for `lm_eval`
- [X] T049 [US4] Add EAGLE fine-tuning phase to `src/pcg_llm/training/trainer.py`: `fine_tune_eagle(steps=5000)` method that freezes PCG core, trains only `EAGLEExtrapolationHead` for N steps, monitors acceptance rate EMA, expands draft length from 4→8 when gate passes

**Checkpoint**: All four custom metrics log to W&B, auto-γ nudging triggers on collapse, disk quota guard works, benchmark harness runs ARC/GSM8K. US4 independently testable.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: CI pipeline, baseline benchmarks, type completeness, and changelog. No new functionality.

- [X] T050 Create `.github/workflows/ci.yml`: run `ruff check src/ tests/`, `ruff format --check src/ tests/`, `mypy src/`, `bandit -r src/ -ll`, `pytest tests/unit/ tests/integration/ --cov=src/pcg_llm --cov-fail-under=80 -m "not gpu and not slow"` on every push and PR; separate job for `@pytest.mark.slow` tests on schedule (nightly); GPU tests excluded from standard CI (require self-hosted runner)
- [X] T051 [P] Capture benchmark baselines: run `benchmarks/bench_deq_solver.py`, `bench_rigl_cycle.py`, `bench_eagle_throughput.py` with all optimization flags OFF (baseline v3.0 behavior), write results to `benchmarks/results/*_baseline.json`; then run with all flags ON and confirm ≥20% wall-clock improvement; commit baseline files
- [X] T052 [P] Full mypy strict pass: run `mypy src/ --disallow-untyped-defs --no-error-summary` and fix all remaining annotation gaps; add `py.typed` marker file to `src/pcg_llm/`; ensure no `# type: ignore` without inline justification
- [X] T053 Update `CHANGELOG.md` under `[Unreleased]`: document all new modules, optimization flags, CLI subcommands, checkpoint format, and free-tier training support added in this feature branch

---

## Dependencies

```
Phase 1 (T001–T004)
  └─► Phase 2 (T005–T010)
        └─► Phase 3 (T011–T027, T054–T056, T060–T062) ─── MVP ──► can deploy US1 independently
        └─► Phase 4 (T028–T038) ─── depends on Phase 3 complete (trainer.py exists)
        │     └─► T060 MUST precede Phase 4 (baseline benchmark capture — Constitution V)
        └─► Phase 5 (T039–T043, T057–T059) ─── depends on Phase 3 complete (CheckpointManager exists)
        └─► Phase 6 (T044–T049) ─── depends on Phase 3 complete (TrainingMetrics hooks exist)
              └─► Phase 7 (T050–T053) ─── all phases complete
```

Within Phase 3:
```
T011–T014 (tests) can run in parallel with each other
T015 (PCGNode) → T016 (DEQSolver) → T017 (Adjacency) → T018 (Loss)
T019 (Scheduler), T020 (Curriculum) — parallel with T015–T018
T021 (LocalBackend) → T022 (CheckpointManager) → T023 (SignalHandler)
T024 (Trainer) — depends on T016, T017, T018, T019, T020, T022, T023
T025 (CLI) — depends on T024
T026, T027 (integration tests) — depend on T025
T054 (PCGNode unit tests) — depends on T015 (PCGNode must exist)
T055 (CPU offload tests) — depends on T017 (Adjacency must exist)
T056 (grad accum tests) — depends on T024 (Trainer must exist)
T060 (baseline benchmark) — depends on T025 (CLI must exist); MUST precede T028
T061 (DDP tests) — depends on T024 (Trainer with DDP must exist)
T062 (DDP benchmark) — depends on T024 (Trainer with DDP must exist)
```

Within Phase 4:
```
T028–T030 (tests) — parallel with each other
T031 (Muon) → T034 (integrate into trainer)
T032 (nGPT), T033 (Monarch) — parallel with T031, all feed into T034
T035 (EAGLE Head) → T036 (EAGLE inference) — depends on T034
T037 (benchmarks) — depends on T034
T038 (GPU tests) — parallel with T035
```

---

## Parallel Execution Examples

### Phase 3 Sprint (US1 — after Phase 2 complete)

```
Stream A: T011 → T015 → T016 → T017 → T018 → T024 → T025 → T026 → T027
Stream B: T012 → T017 (shares, coordinate)
Stream C: T013 → T018 (shares)
Stream D: T014 → T021 → T022 → T023 (feeds T024)
Stream E: T019 + T020 (parallel, feed T024)
Stream F: T015 → T054 (PCGNode tests, after T015)
Stream G: T017 → T055 (CPU offload tests, after T017)
Stream H: T024 → T056 + T060 (grad accum tests + baseline benchmark, after T024/T025)
```

### Phase 4 Sprint (US2 — after Phase 3 + T060 baseline complete)

```
Stream A: T028 → T031 → T034 → T036 → T037
Stream B: T029 → T033 → T034 (coordinate with A)
Stream C: T030 → T035 → T036 (coordinate with A)
Stream D: T032 → T034 (coordinate with A)
Stream E: T038 (independent GPU tests, any time)
```

### Phase 5 Sprint (US3 — after Phase 3 complete)

```
Stream A: T039 → T040 → T041 → T042 → T043
Stream B: T057 (grad checkpoint test, parallel — needs T024)
Stream C: T058 (disk quota test, parallel — needs T021/T022)
Stream D: T059 (3-step DEQ fallback test, parallel — needs T016/T024)
```

---

## Implementation Strategy (MVP First)

| Milestone | Phases | Deliverable | Verification |
|---|---|---|---|
| **MVP** | 1 + 2 + 3 | Tiny PCG trains on Kaggle T4 with checkpoints | Kaggle Notebook smoke test, perplexity ≤ 50 |
| **Optimized** | + 4 | All 8 optimization flags active, EAGLE head | Benchmark ≥20% improvement confirmed |
| **GCP-Ready** | + 5 | 3B model trains on preemptible A100 | Manual preemption + resume test |
| **Observable** | + 6 | Full monitoring, W&B dashboard, eval harness | 200-step run with all metrics visible |
| **Production** | + 7 | CI green, mypy strict, benchmarks baselined | All CI checks pass |

**Suggested MVP scope for first implementation session**: Phases 1–3 (T001–T027). Delivers a working Tiny PCG training pipeline with checkpoint-and-resume on free-tier hardware. All remaining phases are incremental additions.

---

## Summary

| Metric | Value |
|---|---|
| Total tasks | 60 |
| Phase 1 (Setup) | 4 tasks |
| Phase 2 (Foundational) | 6 tasks |
| Phase 3 (US1 — MVP) | 21 tasks (T011–T027 + T054–T056 + T060) |
| Phase 4 (US2 — Optimizations) | 11 tasks |
| Phase 5 (US3 — GCP) | 8 tasks (T039–T043 + T057–T059) |
| Phase 6 (US4 — Monitoring) | 6 tasks |
| Phase 7 (Polish) | 4 tasks |
| Parallelizable tasks [P] | 29 tasks |
| TDD test tasks | 23 tasks (must fail before implementation) |
| New source files | 18 modules across 6 packages |
| New test files | 11 test files |
| New benchmark files | 3 benchmark scripts |
| *Tasks added by speckit-analyze remediation* | *7 tasks: T054–T060* |
