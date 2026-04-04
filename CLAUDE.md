# pcg-llm Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-04-04

## Active Technologies
- Local filesystem (Kaggle `/kaggle/working/checkpoints/`) and Google Cloud Storage bucket (GCP track). Checkpoint manifest stored as JSON alongside model shards. (001-pcg-llm-optimization-spec)

- Python 3.13 (constitution-mandated) (001-pcg-llm-optimization-spec)

## Project Structure

```text
src/
tests/
```

## Commands

cd src; pytest; ruff check .

## Code Style

Python 3.13 (constitution-mandated): Follow standard conventions

## Recent Changes
- 001-pcg-llm-optimization-spec: Added Python 3.13 (constitution-mandated)

- 001-pcg-llm-optimization-spec: Added Python 3.13 (constitution-mandated)

<!-- MANUAL ADDITIONS START -->
## Full Technology Stack (PCG-LLM)

### Core ML
- PyTorch 2.5+ (`torch.compile` Inductor, FlexAttention, BF16 mixed precision)
- torchdeq 0.3+ (Anderson Acceleration + Broyden DEQ solvers)
- OpenAI Triton 3.0+ (Block-Sparse 64×64 custom kernels)
- einops 0.7+ (tensor reshaping)
- bitsandbytes 0.43+ (8-bit optimizer states)

### Training Optimizations (feature 001)
- Muon optimizer: use for all `nn.Linear` weight matrices; AdamW for embeddings/biases
- nGPT normalization: applied inside f_θ (normalizes node states within DEQ function)
- Monarch matrices: replace dense 3072×3072 projections; d=3072 → 64×64 butterfly blocks
- FlexAttention: EAGLE head mask (block-local 64-token + 1 global summary token)
- Anderson window m=3 (not m=5); β=1e-4

### Data
- `datasets` 2.19+ (HuggingFace streaming, FineWeb-Edu + The Stack v2)
- `transformers` 4.40+ (Llama-3 Tiktoken tokenizer, vocab=128256)

### Checkpointing
- Local backend: `/kaggle/working/checkpoints/` (Kaggle T4)
- GCS backend: `google-cloud-storage` 2.16+ with exponential-backoff retry
- Format: `torch.save` dict + SHA-256 + `manifest.json`; atomic write (.tmp → rename)

### Evaluation
- `lm-eval` 0.4+ (lm-evaluation-harness: ARC, MMLU, GSM8K, HumanEval, HellaSwag)
- 4-bit GPTQ inference for T4-compatible benchmark runs

### Monitoring
- `wandb` 0.17+ (W&B dashboard, embedded in Kaggle/Colab notebooks)

## Key Architecture Invariants
- DEQ fixed point: Z* = f_θ(Z*, X); Spectral Normalization enforces L < 1
- Block-Structured RigL: 64×64 blocks (3B) / 32×32 (Tiny); never instantiate dense adjacency
- Loss: CrossEntropy + λ·L1(A) − γ·Var(Z*); variance floor τ=0.1, γ=0.01
- EAGLE Head: single transformer block on penultimate PCG states; K=8 draft branches; L=4 initially
- Checkpoint interval: 500 steps (Kaggle) / 250 steps (GCP)
<!-- MANUAL ADDITIONS END -->
