# Implementation Plan: PCG-LLM Optimization Research & Free-Tier Training

**Branch**: `001-pcg-llm-optimization-spec` | **Date**: 2026-04-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-pcg-llm-optimization-spec/spec.md`

## Summary

Build the full PCG-LLM training system on top of the v3.0 architecture specification, extending it with eight research-validated optimizations (Muon optimizer, nGPT normalization, Monarch matrices, FlexAttention EAGLE head, Anderson m=3, 8-bit optimizer, curriculum mixing, EAGLE draft calibration) and a complete free-tier training infrastructure (checkpoint-and-resume for Kaggle and GCP, SIGTERM handlers, GCS atomic writes, HuggingFace streaming). The Tiny PCG (50–100M params) is the first deliverable, validated on 2× T4 GPUs; the 3B model follows on GCP preemptible A100.

## Technical Context

**Language/Version**: Python 3.13 (constitution-mandated)
**Primary Dependencies**:
- `torch>=2.5.0` (required for FlexAttention; upgrade from pyproject.toml's 2.4.0)
- `torchdeq>=0.3.0` (Anderson/Broyden DEQ solvers; compatible with PyTorch 2.5+)
- `triton>=3.0.0` (Block-Sparse Triton kernels for adjacency ops)
- `einops>=0.7.0` (tensor reshaping for block-sparse operations)
- `bitsandbytes>=0.43.0` (8-bit Adam/Muon optimizer states)
- `datasets>=2.19.0` (HuggingFace streaming for FineWeb-Edu, The Stack v2)
- `wandb>=0.17.0` (training metrics dashboard)
- `lm-eval>=0.4.0` (lm-evaluation-harness for ARC, MMLU, GSM8K, HumanEval, HellaSwag)
- `google-cloud-storage>=2.16.0` (GCS checkpoint writes for GCP track)
- `transformers>=4.40.0` (Llama-3 Tiktoken tokenizer)

**Storage**: Local filesystem (Kaggle `/kaggle/working/checkpoints/`) and Google Cloud Storage bucket (GCP track). Checkpoint manifest stored as JSON alongside model shards.

**Testing**: pytest with markers: `unit`, `integration`, `convergence`, `gpu`, `slow`. Coverage floor 80% (constitution). DEQ solver and routing graph changes require `convergence` + `integration` tests.

**Target Platform**: Linux (Kaggle Notebooks, Google Colab, GCP preemptible VM). CUDA 12.x. Consumer GPU (T4 16GB) for Tiny PCG; A100 40GB for 3B model.

**Project Type**: ML research training framework + model library (`src/pcg_llm/`).

**Performance Goals**:
- Tiny PCG: validation perplexity ≤ 50 within 4 hours on 2× T4
- 3B model: ≥ 15 tokens/second at T4 inference (4-bit quantized)
- 3B model: within 5pp of Phi-3-mini on ARC-Challenge, GSM8K, HumanEval
- Per-step training time: ≥ 20% faster than v3.0 baseline with all optimizations enabled

**Constraints**:
- Kaggle: 30 GPU-hours/week, 16GB VRAM total (2× T4), ~20GB disk quota
- GCP free tier: $300 credit, preemptible V100 16GB or A100 40GB
- Memory budget for 3B model on single A100 40GB: ~38GB peak with all VRAM optimizations
- No custom CUDA — all operations must be implementable with Triton or standard PyTorch

**Scale/Scope**:
- Tiny PCG: 50–100M parameters, d=512, 32×32 blocks, 70% initial sparsity
- 3B PCG: 3.1B parameters, d=3072, 64×64 blocks, 90% initial sparsity
- Training corpus: 100B tokens FineWeb-Edu + The Stack v2 subset

**Terminology note**: The DEQ solver iteration cap is `max_solver_iters` throughout `TrainingConfig`, `tasks.md`, and the codebase. The torchdeq API calls this parameter `max_iter`. These are the same concept; use `max_solver_iters` in all project files.

**Multi-GPU (DDP)**: Auto-detected via `LOCAL_RANK` environment variable set by `torchrun`. When active, `PCGNode` and `output_proj` are wrapped in `DistributedDataParallel` with `broadcast_buffers=True` (required for spectral-norm `weight_u`/`weight_v` buffer synchronization). `W_structure` is NOT a module parameter and is therefore NOT covered by DDP's automatic gradient all-reduce; `_optimizer_step()` manually calls `dist.all_reduce(W_structure.grad)` before each optimizer step. RigL topology decisions are made on rank-0 only and the updated mask is broadcast to all ranks via `dist.broadcast`. All checkpointing, W&B logging, and console output is restricted to rank-0. Launch command: `torchrun --nproc_per_node=2 --master_port=29500 train_ddp.py`.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Code Quality (Ruff + Mypy + Bandit) | PASS | All new modules in `src/pcg_llm/` must pass Ruff, Mypy strict, Bandit. No exemptions without inline justification. |
| II. TDD (Red-Green-Refactor) | PASS | DEQ convergence tests, RigL sparsity tests, checkpoint round-trip tests, and EAGLE acceptance-rate tests must be written before implementation code. T054–T059 (remediation tasks) follow the same TDD mandate. |
| III. Review Standards | PASS | All changes to `src/` via PR with spec reference. |
| IV. Consistency (Python 3.13, src/pcg_llm/ layout, 100-char lines) | PASS | PyTorch 2.5 upgrade requires pyproject.toml bump. Otherwise no deviations. |
| V. Performance (benchmarks required for DEQ/graph changes) | PASS | `benchmarks/` suite required for DEQ solver, routing graph, and attention mechanism. **T060 captures baseline before Phase 4 optimization PRs** (mandatory per Principle V). |

**No gate violations. No Complexity Tracking required.**

**Post-Phase-1 re-check**: Re-verify after data-model and contracts are finalised, specifically that the checkpoint schema doesn't introduce a hidden dependency on implementation-specific tensor layouts.

**Post-speckit-analyze re-check (2026-04-04)**: All 5 constitution gates still PASS after remediation edits. T060 (baseline benchmark capture) was added specifically to satisfy Principle V's requirement that a baseline be captured before optimization PRs.

## Project Structure

### Documentation (this feature)

```text
specs/001-pcg-llm-optimization-spec/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   ├── cli.md           # Training script CLI interface
│   ├── checkpoint.md    # Checkpoint format schema
│   └── config.md        # TrainingConfig schema
└── tasks.md             # Phase 2 output (/speckit.tasks — NOT created here)
```

### Source Code (repository root)

```text
src/pcg_llm/
├── __init__.py                  # existing
├── arch/
│   ├── __init__.py
│   ├── node.py                  # PCGNode: latent state estimator
│   ├── deq_solver.py            # ConstrainedDEQSolver (Anderson + Broyden fallback)
│   ├── adjacency.py             # BlockSparseAdjacency + RigL Drop-and-Grow
│   ├── eagle_head.py            # EAGLEExtrapolationHead + Parallel Tree Attention
│   └── monarch.py               # MonarchProjection (butterfly factor decomposition)
├── training/
│   ├── __init__.py
│   ├── loss.py                  # FreeEnergyLoss (CrossEntropy + L1 sparsity + Variance Hinge)
│   ├── optimizer.py             # MuonOptimizer + hybrid AdamW wrapper
│   ├── scheduler.py             # RigLSparsitySchedule (cosine decay + freeze phase)
│   ├── curriculum.py            # DataCurriculum (epoch-based mixing ratios)
│   ├── trainer.py               # PCGTrainer (main training loop)
│   └── normalization.py         # nGPTNorm (hyperspherical unit-norm normalization)
├── checkpointing/
│   ├── __init__.py
│   ├── checkpoint.py            # CheckpointManager (atomic write, SHA-256, manifest)
│   ├── gcs.py                   # GCSCheckpointBackend (exponential-backoff GCS writes)
│   ├── local.py                 # LocalCheckpointBackend (Kaggle disk quota monitoring)
│   └── signals.py               # SignalHandler (SIGTERM + KeyboardInterrupt → checkpoint)
├── data/
│   ├── __init__.py
│   ├── streaming.py             # HuggingFaceStreamingDataset (FineWeb-Edu + Stack v2)
│   └── tokenizer.py             # Llama3TokenizerWrapper
├── evaluation/
│   ├── __init__.py
│   └── harness.py               # BenchmarkHarness (lm-eval + bitsandbytes 4-bit quantization)
├── monitoring/
│   ├── __init__.py
│   └── metrics.py               # TrainingMetrics (solver steps, sparsity, variance, EAGLE rate)
└── config.py                    # TrainingConfig (dataclass, all flags)

tests/
├── test_pcg_llm.py              # existing (extend)
├── unit/
│   ├── test_deq_solver.py       # convergence tests (pytest.mark.convergence)
│   ├── test_adjacency.py        # RigL sparsity tests + FR-018 CPU offload (T055)
│   ├── test_loss.py             # variance hinge, anti-collapse
│   ├── test_checkpoint.py       # round-trip, atomic write, SHA-256 + FR-025 quota (T058)
│   ├── test_monarch.py          # butterfly projection correctness
│   ├── test_muon.py             # optimizer update direction tests
│   ├── test_normalization.py    # nGPTNorm unit-norm, LayerNorm, build_norm factory
│   ├── test_scheduler.py        # RigLSparsitySchedule pre/post-freeze, state_dict
│   ├── test_curriculum.py       # DataCurriculum mixing ratios, shard advance, state_dict
│   ├── test_signals.py          # SignalHandler SIGTERM, check_and_save
│   ├── test_cli.py              # CLI subcommands, _parse_tokens, _resolve_config
│   ├── test_trainer_unit.py     # PCGTrainer instantiation, generate, train, fine_tune_eagle
│   │                            #   + FR-019 grad accumulation (T056) + FR-017 grad ckpt (T057)
│   └── test_node.py             # PCGNode forward shape, SpectralNorm, BF16, grad flow (T054)
├── integration/
│   ├── test_training_loop.py    # end-to-end Tiny PCG training (pytest.mark.slow)
│   ├── test_resume.py           # checkpoint interrupt-and-resume
│   ├── test_eagle.py            # draft tree generation + verification
│   ├── test_gcs_resume.py       # GCS atomic write + backoff + resume (T039)
│   └── test_3b_smoke.py         # 3B preset CPU smoke test (T043)
└── gpu/
    ├── test_block_sparse.py     # Triton kernel correctness (pytest.mark.gpu)
    └── test_flex_attention.py   # FlexAttention mask (pytest.mark.gpu)

benchmarks/
├── bench_deq_solver.py          # wall-clock + VRAM for DEQ (baseline reference, T060)
├── bench_rigl_cycle.py          # Drop-and-Grow step timing
├── bench_eagle_throughput.py    # EAGLE draft + verification tokens/sec (SC-010)
└── results/
    └── bench_deq_solver_baseline.json # Written by T060 before Phase 4 optimization PRs
```

**Structure Decision**: Single-project layout. `src/pcg_llm/` is organized by concern (arch, training, checkpointing, data, evaluation, monitoring) rather than by model size. All size-specific parameters (d, block_size, sparsity) are controlled via `TrainingConfig` at runtime. Tests are separated by speed/hardware requirement using pytest markers to allow fast unit runs on Kaggle/Colab without triggering GPU or slow tests.

---

## Remediation Additions (post-speckit-analyze, 2026-04-04)

The following tasks were added after a `/speckit-analyze` cross-artifact consistency check identified 21 findings (3 CRITICAL, 6 HIGH, 7 MEDIUM, 4 LOW). All 21 findings have been resolved. The 7 new tasks below are the buildable work items surfaced by that analysis.

### New Tasks Summary

| Task | Finding | Category | Description |
|------|---------|----------|-------------|
| T054 | C4 | Unit test gap | `PCGNode` unit tests — forward shape, SpectralNorm, BF16, grad flow |
| T055 | G1 | Coverage gap | FR-018 CPU offload: mask device transfer without data corruption |
| T056 | G2 | Coverage gap | FR-019 gradient accumulation: optimizer step cadence, step counter |
| T057 | G4 | Coverage gap | FR-017 gradient checkpointing correctness (loss equivalence ± BF16 tolerance) |
| T058 | G5 | Coverage gap | FR-025 disk quota 500MB trigger: shutdown callback + DiskQuotaWarning |
| T059 | G6 | Coverage gap | FR-011 3-step DEQ fallback: `max_solver_iters=3` override on EAGLE rejection |
| T060 | C1 | Constitution V | DEQ baseline benchmark capture BEFORE Phase 4 optimization PRs |

### Spec Changes Summary

The following spec.md sections were updated (no new requirements added — all are clarifications or corrections):

- **FR-003**: `max_solver_iters` terminology standardized (was `max_solver_iterations`)
- **FR-006**: `rigl_interval` config field referenced (was hardcoded "100 steps")
- **FR-010**: "PCG energy constraints" defined as `||Z* − f_θ|| < solver_tolerance`
- **FR-011**: "3-step DEQ fallback" clarified as `max_solver_iters=3` override
- **FR-013**: nGPT `base_lr × 10` auto-scaling is an implementation responsibility, not a config field
- **FR-020**: Marked as out-of-scope stretch goal (MUST NOT deploy without proof-of-concept)
- **FR-021**: "gating temperature" removed from checkpoint keys (undefined concept)
- **FR-022/FR-024**: Clarified as read/validate path and write/store path respectively
- **FR-031**: Quantization pipeline defined: bitsandbytes NF4/FP4 (not GPTQ)
- **SC-003**: References T026 and `max_solver_iters` directly
- **SC-004**: Two-tier sparsity bounds made explicit (operational [88%,95%] vs hard [80%,98%])
- **SC-009**: Marked as operational/budget metric (not buildable)
- **SC-010**: Points to T037 for verification
- **Assumptions**: torchdeq PyTorch compatibility updated to 2.5+
- **Key Entities**: Training Curriculum State — "gating temperature" removed
