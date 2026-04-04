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

Always use `uv` to run Python, tests, and tools — never the bare `python` or `pytest` binaries:

```bash
uv run pytest tests/unit/ --no-cov --tb=short   # unit tests
uv run pytest tests/unit/ tests/integration/ --no-cov --tb=short  # full (no GPU/slow)
uv run ruff check src/
uv run mypy src/
uv run python -c "..."                           # one-off Python
```

Install dependencies:
```bash
uv pip install -e ".[training]" --system   # Kaggle / system Python (no venv)
uv sync --extra training                   # local dev (inside venv)
```

**Never use `uv sync --system`** — the `--system` flag was removed from `uv sync` in newer uv versions. Use `uv pip install --system` instead.

## Code Style

Python 3.13 (constitution-mandated): Follow standard conventions

## Recent Changes
- 001-pcg-llm-optimization-spec: Added Python 3.13 (constitution-mandated)

- 001-pcg-llm-optimization-spec: Added Python 3.13 (constitution-mandated)

<!-- MANUAL ADDITIONS START -->
## Development Workflow

### Spec-Driven Development (required for all code changes)

Every non-trivial change MUST follow the speckit workflow in order:

1. **`/speckit-specify`** — write the spec (`specs/<branch>/spec.md`)
2. **`/speckit-plan`** — produce the implementation plan (`plan.md`)
3. **`/speckit-tasks`** — break the plan into tasks (`tasks.md`)
4. **`/speckit-analyze`** — cross-check artifacts for inconsistencies before touching code
5. **`/speckit-implement`** — execute the tasks

Do not skip steps or reorder them. The spec is the source of truth; implementation follows the spec, not the other way around.

### Known Bash / Tooling Pitfalls

**Encoding — always specify `encoding='utf-8'` when opening files in Python on Windows:**
```python
# CORRECT
with open('file.json', 'r', encoding='utf-8') as f: ...
with open('file.json', 'w', encoding='utf-8') as f: ...

# WRONG — default is cp1252 on Windows; corrupts multibyte UTF-8 (mojibake)
with open('file.json') as f: ...
```

**`json.dump` — use `ensure_ascii=False` to keep Unicode readable:**
```python
json.dump(obj, f, indent=2, ensure_ascii=False)
```

**PowerShell scripts from bash — use `-File`, not `-Command`:**
```bash
powershell -File .specify/scripts/powershell/check-prerequisites.ps1 -Json
```

**Pre-commit hook auto-fixes** — hooks (ruff-format, mixed-line-ending, end-of-file-fixer) modify staged files. After a hook failure, always re-`git add` the modified files before retrying `git commit`.

**`uv sync --system` does not exist** — see Commands section above.
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
