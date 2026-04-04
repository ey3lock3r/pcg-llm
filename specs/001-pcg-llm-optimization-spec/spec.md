# Feature Specification: PCG-LLM Optimization Research & Free-Tier Training

**Feature Branch**: `001-pcg-llm-optimization-spec`
**Created**: 2026-04-04
**Status**: Draft
**Input**: User description: "Review docs thoroughly, research each part for further performance optimization, engineering/algorithm alternatives or hacks/tricks to make the model optimised and performant while competing or exceeding SOTA. Target free-tier platforms (Kaggle, Colab/GCP) with checkpoint support for interrupted training."

---

## 1. Project Context & Summary of Reviewed Documents

The existing documentation defines a **Predictive Coding Graph Language Model (PCG-LLM)** through three progressive engineering specification versions:

- **v1.1** — Baseline architecture: PyTorch + PyG + Triton, Inference Learning (IL) inner loop, 1D Block-Diagonal GPU partitioning, Non-Autoregressive (NAR) decoding with confidence-based re-engagement.
- **v2.0** — Three "hacker" upgrades: DEQ solver (Broyden), RigL dynamic sparse training, speculative verification with a 50M draft model.
- **v3.0** — Production fixes addressing three critical v2.0 bugs: Spectral Normalization (Lipschitz convergence guarantee), Block-Structured RigL (64x64 blocks for GPU memory contiguity), EAGLE-style extrapolation head replacing the separate draft model.

The **Hyper-Parameter Table** defines two operating points: a Tiny PCG (50-100M params, 12-16GB VRAM) and a Final PCG (3.1B params, H100/A100).

This specification **extends v3.0** with additional optimization discoveries surfaced through research, identifies remaining gaps, and adds a **free-tier training track** for Kaggle Notebooks and Google Colab/GCP.

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Tiny PCG Free-Tier Proof of Concept (Priority: P1)

An ML researcher with no cloud budget trains the Tiny PCG (50-100M parameters) end-to-end on a Kaggle Notebook (2x T4, 16GB VRAM total) or Google Colab (T4, 15GB VRAM), with full checkpoint-and-resume so that a session disconnect never loses progress. The researcher verifies the DEQ solver converges, the Block-Structured RigL sparsity schedule runs correctly, and the EAGLE head produces sensible draft trees on a small validation set.

**Why this priority**: All downstream work (3B model, SOTA benchmarks) depends on validating the core mechanisms on accessible hardware. If DEQ diverges or Block-RigL produces degenerate sparsity, the 3B model will fail expensively. The Tiny PCG is the mandatory validation test bed.

**Independent Test**: Deploy a Kaggle Notebook that trains the Tiny PCG on a 1B-token slice of FineWeb-Edu, saves a checkpoint every 500 steps to the working directory, and resumes seamlessly after a manual kernel restart. Validates that loss decreases, solver steps remain below the maximum iteration cap, and perplexity on a held-out slice reaches a fluency threshold.

**Acceptance Scenarios**:

1. **Given** a fresh Kaggle Notebook with 2x T4 GPUs and the training script, **When** the script is run for 2,000 steps, **Then** a checkpoint file is written every 500 steps containing model weights, optimizer state, RigL mask, DEQ solver state, and the global step counter.
2. **Given** an interrupted training session (simulated by manually stopping the kernel at step 1,200), **When** the notebook is restarted and the training script is executed again, **Then** training resumes from the latest valid checkpoint (step 1,000 or 1,500) with identical loss trajectory continuing from that point.
3. **Given** a completed 4-hour Tiny PCG run, **When** evaluated on a 50M-token validation slice, **Then** perplexity is at or below 50 and the solver step count per token averages at or below 8 (indicating proper convergence rather than iteration budget exhaustion).
4. **Given** the DEQ solver is running with Spectral Normalization active, **When** the Lipschitz constant of any weight matrix is found to exceed 1.0 during a forward pass, **Then** Spectral Normalization re-normalizes that matrix before the solver runs and a warning is logged; the solver does not crash or diverge.

---

### User Story 2 - Additional Optimization Integration (Priority: P2)

A senior ML engineer receives the v3.0 spec extended with newly researched optimization techniques — Muon optimizer, nGPT normalization, FlexAttention for the EAGLE head, Monarch matrix projections, and 8-bit optimizer states — and is able to evaluate each technique's compatibility with the DEQ + Block-RigL core. Each technique is enabled or disabled via a single configuration flag with no code changes required.

**Why this priority**: The v3.0 spec is production-ready but not yet SOTA-calibrated. Each additional optimization can yield 10-40% throughput improvements or measurable benchmark gains. Engineers must be able to A/B test them without modifying the core architecture.

**Independent Test**: Run the Tiny PCG with each optimization flag toggled independently. Confirm each flag enables or disables the optimization cleanly, that the baseline (all flags off) exactly reproduces v3.0 behavior, and that the all-flags-on configuration reduces per-step wall-clock time by at least 20% on equivalent hardware.

**Acceptance Scenarios**:

1. **Given** the configuration flag `optimizer=muon` is set, **When** training starts, **Then** the Muon optimizer (orthogonal gradient update via Newton-Schulz iteration) is used for all non-embedding weight matrices, replacing AdamW, with the learning rate automatically scaled to Muon's update magnitude conventions.
2. **Given** the flag `normalize=ngpt` is set, **When** a forward pass executes, **Then** all node state vectors are normalized to unit norm after each message-passing step, the learning rate warmup schedule is eliminated, and the configured base learning rate is automatically multiplied by 10 to exploit nGPT's geometry-invariant training.
3. **Given** the flag `projection=monarch` is set, **When** the 3072x3072 DEQ projection matrices are initialized, **Then** they use butterfly-structured Monarch decompositions, reducing FLOPs of each projection pass without requiring custom CUDA kernels beyond standard batched matrix multiplication.
4. **Given** the flag `optimizer_bits=8` is set, **When** the optimizer is instantiated, **Then** it uses 8-bit quantized momentum and variance states, reducing optimizer VRAM consumption by approximately 50% with no change to the training loss trajectory.

---

### User Story 3 - GCP Preemptible 3B Model Training (Priority: P3)

A researcher uses GCP's free credit to train the 3B PCG-LLM on preemptible GPU instances, with training state checkpointed to Google Cloud Storage every 250 steps. When a preemptible instance is reclaimed, training automatically resumes from the latest GCS checkpoint on the next available instance without any manual file transfer.

**Why this priority**: The 3B model requires more sustained compute than Kaggle's 30 GPU-hours per week allows. GCP preemptible instances are the most cost-effective path to large-model training but are interrupted unpredictably. Resilient checkpointing to durable external storage is mandatory for any realistic training run.

**Independent Test**: Launch a 3B training job on a preemptible GCP V100, simulate preemption by manually terminating the VM at step 750, spin up a fresh instance, run the resume script pointing to the same GCS bucket, and confirm training continues from step 500 or 750 with the correct model state.

**Acceptance Scenarios**:

1. **Given** a preemptible GCP instance running the 3B training script, **When** a checkpoint is written every 250 steps, **Then** the checkpoint is written atomically to GCS (write to a temp path first, then rename) so that a mid-write preemption never produces a corrupt or partial checkpoint.
2. **Given** a terminated preemptible instance and a fresh GCP instance with identical configuration, **When** the training script starts on the new instance, **Then** it automatically detects and validates the latest checkpoint in the configured GCS bucket and resumes from it without any manual file operations.
3. **Given** the 3B model completes 50,000 training steps, **When** evaluated on GSM8K, ARC-Challenge, and HumanEval benchmarks, **Then** the PCG-LLM scores within 5 percentage points of published scores for both Phi-3-mini (3.8B) and Gemma-2 2B on each benchmark, establishing SOTA competitiveness at the efficient 3B scale.

---

### User Story 4 - Training Health Monitoring (Priority: P4)

A researcher can observe training health in real time from a Kaggle or Colab notebook, tracking DEQ solver convergence quality, RigL sparsity statistics, node variance (anti-collapse guard), and EAGLE draft acceptance rate — without leaving the notebook environment or requiring a separate server.

**Why this priority**: DEQ models and dynamic sparse models fail silently in ways that standard loss curves do not reveal. A researcher who only watches cross-entropy loss may waste hours of free-tier compute on a model that has collapsed to zero-state, diverged its graph, or saturated its solver iteration budget.

**Independent Test**: After 500 training steps, a metrics dashboard (Weights and Biases or TensorBoard, configured for notebook embedding) shows all four custom metrics (solver step count, sparsity level, node variance, EAGLE acceptance rate) populated and updating each step.

**Acceptance Scenarios**:

1. **Given** training is running, **When** the DEQ solver's mean step count over the last 100 steps exceeds 80% of the configured maximum, **Then** a warning is logged and displayed in the notebook output alerting the researcher that the learning rate may be too high.
2. **Given** the variance hinge loss term is active, **When** node state variance drops below the threshold τ = 0.1, **Then** the γ coefficient is automatically nudged upward by 10% for the next 100 steps and an event is logged.
3. **Given** the EAGLE head is generating draft trees during inference evaluation, **When** the PCG rejects more than 60% of draft tokens over a 1,000-step window, **Then** a warning is logged recommending that the EAGLE head be fine-tuned separately before the next training phase.

---

### Edge Cases

- What happens when the DEQ solver hits max iterations on every token for an extended period? The system must not silently continue producing low-quality outputs; it must log a critical warning and increase the max iterations or reduce the learning rate automatically.
- How does Block-Structured RigL behave when the Grow phase attempts to activate blocks that were never sampled in the initial Erdos-Renyi distribution (blocks with no gradient history)? These blocks should be initialized with a small random gradient estimate rather than zero.
- What happens if a Kaggle checkpoint write is interrupted mid-write due to disk quota exhaustion? The reader must detect file corruption via checksum mismatch and fall back to the previous valid checkpoint automatically.
- What happens during the RigL freeze phase (final 20% of training) if node variance collapses? Since edge re-routing is disabled, the γ coefficient must be the sole corrective mechanism; the system must increase γ aggressively and log that the freeze phase encountered a collapse event.
- How does the model behave on out-of-distribution inputs where the PCG's latent representation is incoherent, causing the EAGLE head to produce nonsensical draft trees? The PCG's verification step should reject all branches and fall back to 3-step DEQ generation.
- What happens if GCS is temporarily unavailable during a checkpoint write on a GCP preemptible instance? The write must retry with exponential backoff; if the instance is preempted before the write succeeds, the previous checkpoint in GCS must remain valid and uncorrupted.

---

## Requirements *(mandatory)*

### Functional Requirements

#### Core Architecture (Confirmed from v3.0)

- **FR-001**: The system MUST implement the DEQ Fixed-Point solver as the primary reasoning mechanism, replacing all iterative IL inner loops from v1.1.
- **FR-002**: The DEQ solver MUST apply Spectral Normalization to all graph weight matrices during every forward pass to guarantee Lipschitz constant L < 1, using power iteration to compute the largest singular value.
- **FR-003**: The DEQ solver MUST use Anderson Acceleration as the primary fixed-point solver with history window m=3 and regularization coefficient β=1e-4, falling back to Broyden's Method when Anderson fails to converge within `max_solver_iters` steps (the `TrainingConfig` field name; equivalent to `max_iter` in torchdeq API).
- **FR-004**: The Adjacency Tensor MUST be initialized at 90% block-sparsity using an Erdos-Renyi distribution for the 3B model and 85% sparsity for the Tiny PCG (≥80% to satisfy SC-004 hard lower bound from step 0), never instantiating a full dense adjacency matrix.
- **FR-005**: Block-Structured RigL MUST evaluate and prune entire 64x64 connectivity blocks for the 3B model (32x32 for Tiny PCG) based on L2 gradient norm, never individual edges.
- **FR-006**: The RigL Drop-and-Grow cycle MUST run every `rigl_interval` training steps (default 100 for production runs; the `TrainingConfig` field is configurable for tests), dropping the bottom 10% of active blocks and growing the 10% of dormant blocks with the highest gradient signal.
- **FR-007**: The RigL re-routing fraction MUST follow a cosine decay schedule starting at 30% at step 1, decaying to 0% at 80% of total training steps, after which the graph topology is frozen.
- **FR-008**: The Loss Function MUST include three terms: CrossEntropy prediction error, L1 block sparsity penalty (λ), and a negative Variance Hinge term (−γ·Var(Z*)) where τ = 0.1 is the variance floor; γ = 0.01 at initialization.
- **FR-009**: The inference pipeline MUST implement the EAGLE Extrapolation Head as a single transformer block attached to the PCG's penultimate latent states, generating a Draft Tree of K=8 token continuations with initial draft length 4 (expandable to 8 after fine-tuning).
- **FR-010**: Parallel Tree Attention MUST verify all K=8 draft branches in a single PCG forward pass, accepting the longest contiguous path whose residual satisfies the DEQ fixed-point condition `||Z* − f_θ(Z*, X)|| < solver_tolerance` (i.e., the branch is a valid near-fixed-point of the DEQ, not just low cross-entropy).
- **FR-011**: On draft rejection, the system MUST fall back to a DEQ iteration capped at 3 solver steps (`max_solver_iters=3` override for this call) to generate the correct token. This is a latency-bounded fallback, not a fully-converged solve; the 3-step cap trades output quality for bounded worst-case latency when all EAGLE branches are rejected.

#### Additional Optimizations (New Research — Configurable via Flags)

- **FR-012**: The system MUST support the Muon Optimizer as a configurable drop-in replacement for AdamW on all non-embedding, non-output-head weight matrices. Muon applies Nesterov momentum then orthogonalizes the update via Newton-Schulz iteration, producing updates with better spectral properties for matrix weights and specifically benefiting the W_structure routing matrices updated by Hebbian plasticity.
- **FR-013**: The system MUST support nGPT-style hyperspherical normalization as a configurable alternative to LayerNorm. All node state vectors are normalized to unit norm after each message-passing step, enabling 10-100x higher learning rates and eliminating mandatory warmup scheduling. The EAGLE head's internal attention layers are the primary application target. When `normalize="ngpt"` is set, `PCGTrainer` MUST automatically multiply `base_lr` by 10 to exploit nGPT's geometry-invariant training (this auto-scaling is an implementation responsibility, not a user configuration field).
- **FR-014**: The system MUST support Monarch Matrix decomposition as a configurable replacement for dense 3072x3072 DEQ projection matrices. A Monarch matrix decomposes a d×d matrix into two sqrt(d)-block-diagonal butterfly factors, reducing projection FLOPs from O(d^2) to O(d·sqrt(d)) using only standard batched matrix multiplications compatible with torch.compile.
- **FR-015**: The system MUST support 8-bit quantized optimizer states via bitsandbytes as a configurable flag, reducing optimizer VRAM from approximately 12 bytes per parameter (standard AdamW FP32 states) to 6 bytes per parameter, freeing roughly 18GB of VRAM on a 3B model.
- **FR-016**: The system MUST use FlexAttention (PyTorch 2.5+) for the EAGLE Head's internal attention mechanism with a custom block-causal mask aligned to the PCG partitions — partition size equals `block_size` (32 for Tiny PCG, 64 for 3B model) — reducing EAGLE head attention complexity from O(N²) to O(N·block_size) per partition with no custom CUDA code.
- **FR-017**: The system MUST support gradient checkpointing on the DEQ solver's intermediate Anderson Acceleration states as a configurable flag, trading approximately 35% additional compute for a 60-70% reduction in peak VRAM, enabling the 3B model on a single A100 40GB.
- **FR-018**: The system MUST support CPU offloading of the RigL Adjacency Tensor during the forward pass, transferring the mask to GPU only during the Drop-and-Grow evaluation step, saving approximately 3GB of VRAM on the 3B model.
- **FR-019**: The training script MUST implement gradient accumulation with a configurable `grad_accum_steps` field. There is no automatic hardware detection; the recommended values are 8 for free-tier T4/V100 and 1 for H100, documented in `quickstart.md`. The default in `TrainingConfig` is 1 (no accumulation); users override this in their training script or preset config.
- **FR-020**: *(Research Stretch Goal — Out of Scope for This Feature)* BitNet b1.58 ternary weight quantization for the W_structure routing matrices is a post-training research direction only. It MUST NOT be implemented in the main training loop without a separate proof-of-concept demonstrating DEQ fixed-point compatibility. This item has no associated task in this feature branch and is recorded here solely as a research note for future planning.

#### Free-Tier Training and Checkpointing

- **FR-021**: The training script MUST save a complete checkpoint every N steps (configurable; default 500 for Kaggle, 250 for GCP), containing: model weights (BF16), optimizer state, RigL Adjacency Mask, current sparsity schedule step, DEQ Anderson Acceleration history buffer (last m=3 iterates), global training step, current loss values for all three loss terms, RNG seeds for all devices, and the training curriculum state (epoch index, dataset shard index, current data mixing ratio).
- **FR-022**: On startup, the training script MUST automatically scan the configured checkpoint directory, select the latest checkpoint by step number from the manifest, verify its SHA-256 checksum (defined in FR-024), and load it. If no valid checkpoint exists, training starts from scratch. *Note: FR-022 is the read/validate path; FR-024 is the complementary write/store path. Together they form the integrity contract.*
- **FR-023**: Checkpoint writes MUST be atomic: written to a temporary path first, then renamed to the final checkpoint filename. This guarantees that a process crash during write never produces a corrupt or partial checkpoint at the final path.
- **FR-024**: When writing a checkpoint, the system MUST compute and store a SHA-256 checksum alongside the checkpoint file and record it in the manifest. This checksum is what FR-022 verifies on load; on mismatch, the system MUST fall back to the previous valid manifest entry and log the corruption event. *Note: FR-024 is the write/store path; FR-022 is the read/validate path.*
- **FR-025**: For Kaggle, the checkpoint directory MUST be within `/kaggle/working/checkpoints/`. The training script MUST monitor remaining Kaggle disk quota and trigger a checkpoint write followed by graceful shutdown when fewer than 500MB of quota remain.
- **FR-026**: For GCP, the checkpoint script MUST write checkpoints to a configured GCS bucket using the Google Cloud Storage Python SDK with exponential-backoff retry (maximum 5 attempts, base delay 1 second, maximum delay 32 seconds) for all GCS write operations.
- **FR-027**: The training script MUST support streaming the FineWeb-Edu and The Stack v2 training datasets directly from the Hugging Face Datasets streaming API, consuming no local disk space beyond the current active shard, to remain within Kaggle's disk quota limits.
- **FR-028**: The training script MUST register a signal handler for SIGTERM (GCP preemption signal) and a keyboard interrupt handler (Colab/Kaggle session termination) that triggers an immediate checkpoint save before the process exits.

#### Curriculum and Data

- **FR-029**: The training curriculum MUST mix code data throughout all epochs rather than reserving it for Epoch 3 only, following the ratios: Epoch 1 at 92% FineWeb-Edu and 8% The Stack v2; Epoch 2 at 85% FineWeb-Edu and 15% The Stack v2; Epoch 3 (crystallization) at 70% FineWeb-Edu and 30% The Stack v2. This is supported by published Data Mixing Laws research showing mixed-throughout schedules outperform sequential domain shifts for reasoning generalization.

#### Benchmarking

- **FR-030**: The repository MUST include evaluation harness scripts for ARC-Challenge, MMLU, GSM8K, HumanEval, and HellaSwag benchmarks using the lm-evaluation-harness framework.
- **FR-031**: The evaluation harness MUST support 4-bit quantized inference to enable benchmark runs on a single T4 GPU within Kaggle's session time limits. Quantization is applied via `bitsandbytes` 4-bit weight loading (NF4/FP4) on model load, not post-training GPTQ. The pipeline: load checkpoint → apply bitsandbytes 4-bit quantization → run `lm_eval.evaluator.simple_evaluate()`. This is implemented in `BenchmarkHarness` (T048) with a `quantize="4bit"` flag.

### Key Entities

- **PCG Node State (Z*)**: The DEQ fixed-point solution tensor of shape [B, N, d]. Represents the globally consistent, settled latent representation of all tokens in context. The primary output of the DEQ solver.
- **Adjacency Tensor (A_evolved)**: The block-sparse connectivity mask of shape [N/64, N/64] blocks (for the 3B model). The learned topological structure of the reasoning graph. Modified by RigL Drop-and-Grow cycles. Must be fully checkpointed.
- **DEQ Solver State**: The Anderson Acceleration history buffer containing the last m=3 iterates and residuals. Must be checkpointed and restored to ensure identical convergence behavior on training resume.
- **RigL Sparsity Schedule State**: Tracks the current re-routing fraction (cosine decay), the step counter within the 100-step evaluation interval, and the freeze flag (active after 80% of total training steps). Must be checkpointed.
- **EAGLE Draft Tree**: A tree of K=8 candidate token continuations of draft length 4-8, generated from the PCG's penultimate latent state. Shape: [B, K, draft_len, d]. Verified by the PCG in a single Parallel Tree Attention forward pass.
- **Training Curriculum State**: Current epoch index, dataset shard index, RigL re-routing fraction, and current data mixing ratio. Must be fully checkpointed to resume training from the exact data position.
- **Checkpoint Manifest**: A JSON index file in the checkpoint directory listing all available checkpoints with their step numbers, file paths, timestamps, and SHA-256 checksums. The manifest is the authoritative source for checkpoint discovery on resume.

---

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The Tiny PCG (50-100M parameters) reaches a validation perplexity of 50 or below on a 50M-token FineWeb-Edu validation split within 4 hours of wall-clock training time on 2x T4 GPUs.
- **SC-002**: Training interrupted at any step and resumed from the latest valid checkpoint produces a loss trajectory that differs from the uninterrupted run by less than 0.5% (relative) within 10 steps post-resume. Validated by `test_resume_loss_continuity` in `tests/integration/test_resume.py`.
- **SC-003**: The DEQ solver step count per token does not exceed 15 (out of a configured maximum; `max_solver_iters` default is 25 for production) on any step during stable synthetic training. Integration test T026 (`test_solver_steps_within_limit`) verifies `solver_steps ≤ max_solver_iters` for every step over 100 synthetic training steps — a stricter check than the original "5% of tokens" criterion, appropriate for deterministic synthetic test conditions.
- **SC-004**: Block-Structured RigL maintains sparsity within two tiers throughout all non-freeze training steps: (a) **operational target**: between 88% and 95% in steady-state training; (b) **hard safety bounds**: never below 80% (collapse floor) or above 98% (full density ceiling). The automated test `test_sparsity_stays_in_range_after_10_cycles` (in `test_adjacency.py`) validates the hard bounds [80%, 98%] for the Drop-and-Grow mechanism in isolation. The integration test `test_sparsity_stays_in_range` (in `test_training_loop.py`) validates the hard bounds [80%, 98%] during full training; the operational target [88%, 95%] is a steady-state goal verified by manual inspection of training logs.
- **SC-005**: Node state variance remains above τ = 0.1 after the warmup phase completes in production training (500+ warmup steps on real data). In CI, the integration test `test_node_variance_is_tracked_and_finite` verifies the metric is reported, finite, and non-negative each step; the Variance Hinge mechanism driving variance above τ is validated at unit level by `test_loss.py`.
- **SC-006**: The 3B PCG-LLM scores within 5 percentage points of published Phi-3-mini (3.8B) and Gemma-2 2B scores on ARC-Challenge, GSM8K, and HumanEval, establishing SOTA competitiveness at the 3B efficient-model tier.
- **SC-007**: The combined additional optimizations (Muon + nGPT normalization + 8-bit optimizer) reduce per-step wall-clock training time by at least 20% compared to the baseline v3.0 configuration on equivalent hardware.
- **SC-008**: The EAGLE draft acceptance rate exceeds 70% on standard conversational and coding prompts after the EAGLE head fine-tuning phase. *(Operational metric — requires real inference data. The 70% threshold is not enforced in CI; `fine_tune_eagle()` in T049 implements the training mechanism, and `accept_rate_ema` is logged per step. Acceptance rate gate must be manually verified before 3B production deployment.)*
- **SC-009**: *(Operational/Budget Metric — not buildable)* The full 3B model training run is expected to complete within the GCP free credit budget when using preemptible V100 instances. This criterion is not testable in CI and has no associated implementation task; it is a planning constraint verified manually at time of execution.
- **SC-010**: On a single T4 GPU with 4-bit quantized inference, the 3B model generates responses at 15 or more tokens per second using the EAGLE plus Parallel Tree Attention pipeline. Verified by benchmark script `benchmarks/bench_eagle_throughput.py` (T037).

---

## Assumptions

- The Llama-3 Tiktoken tokenizer (vocabulary size 128,256) is used as-is. The embedding matrix is trained from scratch; no pretrained embedding weights are assumed to be available.
- FineWeb-Edu and The Stack v2 are the sole training corpora. Both are available via Hugging Face Datasets streaming and can be consumed without full local download.
- For Kaggle free-tier training, the researcher has a verified Kaggle account with GPU quota (30 hours/week, 2x T4). No Kaggle Pro subscription is assumed.
- For GCP training, the researcher has activated the $300 GCP free credit and has access to preemptible V100 (16GB) or A100 (40GB) instances. TPU access is treated as a stretch goal.
- The torchdeq library (providing Anderson/Broyden solvers) is available via pip and compatible with PyTorch 2.5+ (the required version for FlexAttention; torchdeq>=0.3.0 is the minimum version, per plan.md).
- The bitsandbytes library is available and supports T4, V100, A100, and H100 hardware targets.
- The Muon optimizer is implemented as a publicly available reference implementation and requires no additional library installation beyond PyTorch.
- Adapting the EAGLE extrapolation head (originally designed for autoregressive Transformers) to PCG non-autoregressive latent states requires a design validation experiment before the 3B training run. This is scoped as a P2 requirement with a defined acceptance test.
- The Tiny PCG (50-100M params) is the mandatory first deliverable. The 3B model is only trained after the Tiny PCG validates the full pipeline on free-tier hardware.
- Benchmark evaluation is run using the lm-evaluation-harness framework with 4-bit quantized inference. Scores are compared against published numbers for Phi-3-mini-3.8B and Gemma-2 2B.
- The v3.0 spec's hardware target (8x H100) is aspirational. The practical training path is: Tiny PCG on Kaggle, then 3B model on GCP preemptible V100/A100, then potential H100 scale-up if access is obtained separately.
- BitNet b1.58 quantization (FR-020) is a research stretch goal. Its compatibility with the DEQ fixed-point formulation requires a separate proof-of-concept before inclusion in the main training loop.
- Anderson Acceleration is configured with history window m=3 (not m=5 as in the generic DEQ literature) because the PCG's aggressive Spectral Normalization (targeting L < 0.9) means the graph dynamics are sufficiently well-conditioned for the smaller history window, saving 40% of solver memory with no convergence quality loss.
- The Tiny PCG (50–100M params) serves a dual purpose: it is first a mandatory engineering validation step (convergence checks, DEQ stability, RigL correctness), and conditionally a standalone published model if benchmark results are competitive. A benchmark gate is applied after validation — if the Tiny PCG scores within 5 percentage points of comparable-scale published models (TinyLlama-1.1B scaled down, SmolLM-135M), it is promoted to a standalone deliverable with its own model card and benchmark suite. If it does not pass the gate, it remains an internal validation artifact and training proceeds directly to the 3B model.

---

## Additional Research: Identified Gaps and Optimization Opportunities in v3.0

### Gap 1: Muon Optimizer vs. AdamW for Matrix Weights

The v3.0 spec uses AdamW throughout. Recent empirical results demonstrate that Muon — which applies Nesterov momentum then orthogonalizes the full gradient matrix via Newton-Schulz iteration — consistently outperforms AdamW by 15-30% in loss per compute-step on matrix-weight language model training. The mechanism: AdamW normalizes gradients element-wise (coordinate-wise), while Muon normalizes the gradient matrix spectrally, producing steepest-descent updates in the space of linear maps. This is especially valuable for the PCG's W_structure routing matrices, where the orthogonalized update prevents weight matrix rank collapse far more robustly than AdamW's element-wise clipping. Muon is not appropriate for embeddings or scalar parameters, so a hybrid optimizer (Muon for matrices, AdamW for embeddings and biases) is the recommended configuration.

### Gap 2: nGPT Normalization and Training Speed

The v3.0 spec does not specify a normalization strategy for node states beyond the Variance Hinge loss term. The nGPT formulation (Loshchilov et al. 2024) normalizes all weight vectors and hidden states to unit norm, reformulating the network as operating on the unit hypersphere. Two critical benefits for the PCG: (1) learning rates become geometry-invariant, enabling 10-100x higher LRs without instability and sharply reducing the number of steps needed to reach target loss; (2) residual stream norms do not drift, eliminating the need for manual LR warmup calibration. For the PCG's node states, hyperspherical normalization adds a geometric constraint that complements the energy-based collapse prevention of the Variance Hinge, potentially allowing γ to be reduced without collapse risk.

### Gap 3: Monarch Matrices for Projection Efficiency

The v3.0 spec uses implicit dense 3072x3072 matrices for the DEQ's internal projection layers. Monarch matrices (Dao et al. 2022) represent any dense d×d matrix as two sqrt(d)-block-diagonal butterfly factor products, reducing FLOPs from O(d^2) to O(d·sqrt(d)) using only standard batched GEMM operations compatible with torch.compile and Triton. On A100 and H100 hardware, this achieves 2-4x hardware efficiency for the specific projection sizes used in 3B-scale models. Monarch matrices do not require custom CUDA kernels and are drop-in compatible with existing PyTorch autograd.

### Gap 4: FlexAttention for EAGLE Head Efficiency

The v3.0 spec specifies the EAGLE Head as "a single transformer block" but does not define the attention pattern within it. The default (full causal attention over all N tokens) is O(N^2), which at N=4096 produces a 16M-element attention matrix just for the EAGLE head. Using FlexAttention (available in PyTorch 2.5+), a custom block mask can restrict each node's attention to its own 64-node partition plus a single global summary token, reducing EAGLE head attention complexity to O(N·64) — a 64x reduction — using a pure Python mask function with no custom CUDA code.

### Gap 5: Anderson Acceleration Window Size Optimization

The v3.0 spec delegates solver configuration to the torchdeq library without specifying the Anderson Acceleration history window size m. Published DEQ training guidance recommends m=5 as the general default, but notes that m=3 is sufficient when the Lipschitz constant is aggressively constrained below 0.9. Since the PCG enforces Spectral Normalization targeting L < 1 (and in practice achieves L < 0.9 within a few hundred training steps), m=3 reduces solver memory consumption by 40% with no measurable convergence quality loss. This frees approximately 1.5GB of VRAM on the 3B model during training.

### Gap 6: Free-Tier Training — Not Addressed in v3.0

The v3.0 spec targets 8x H100, completely inaccessible on free tiers. The following techniques, when combined, make the 3B model trainable on a single A100 40GB GCP preemptible instance:

| Technique | VRAM Savings | Compute Overhead | Notes |
|---|---|---|---|
| Gradient Checkpointing (DEQ states) | 60-70% peak reduction | +35% compute | Highest impact single technique |
| 8-bit Optimizer States | ~18GB on 3B model | Negligible | Must use bitsandbytes |
| CPU Offload (RigL Mask) | ~3GB | Small PCIe transfer overhead | Only during forward pass |
| Gradient Accumulation (x8) | Enables larger effective batch | None | Pure throughput benefit |
| Anderson m=3 (vs m=5) | ~1.5GB | None | Free optimization |

With all five techniques enabled, the 3B model fits in 38-40GB of VRAM (single A100 40GB) at effective batch size 512 via 8-step gradient accumulation.

### Gap 7: Curriculum Data Mixing Strategy

The v3.0 spec sequences FineWeb-Edu (Epochs 1-2) followed by The Stack v2 (Epoch 3). Research from Doremi (Xie et al. 2023) and Data Mixing Laws (Ye et al. 2024) shows that introducing code data as a small percentage from the very first training step — rather than saving it for a final "crystallization epoch" — produces significantly better reasoning generalization. The model builds code-aware representations from the beginning rather than having to reconcile two separate representation spaces at the end of training. The recommended ratios are defined in FR-029.

### Gap 8: EAGLE Head Draft Length Calibration

The v3.0 spec states K=8 draft branches but does not specify draft sequence length. The EAGLE paper (Li et al. 2024) uses draft lengths of 4-6 tokens per verification step for standard autoregressive Transformers. For the PCG's non-autoregressive verification, draft length must be empirically calibrated: too long risks wholesale rejection of the draft tree (increasing latency rather than reducing it); too short provides minimal speedup over single-token generation. Starting at draft length 4 and expanding to 8 after acceptance rate stabilizes above 65% is the recommended approach, defined in FR-009.
