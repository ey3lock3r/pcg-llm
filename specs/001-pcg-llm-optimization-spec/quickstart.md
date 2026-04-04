# Quickstart: PCG-LLM Training on Free-Tier Platforms

**Feature**: `001-pcg-llm-optimization-spec`
**Date**: 2026-04-04

---

## Prerequisites

- Python 3.13
- Git access to this repository
- A Kaggle account (verified, GPU quota enabled) **or** a GCP account with $300 free credit activated

---

## Track A: Tiny PCG on Kaggle (50–100M params, 2× T4)

### Step 1: Set up a Kaggle Notebook

1. Go to kaggle.com → **New Notebook**
2. Under **Settings → Accelerator**, select **GPU T4 × 2**
3. Under **Settings → Internet**, enable internet access (required for dataset streaming)

### Step 2: Install dependencies

```python
# Cell 1 — Install
!pip install -q \
    "torch>=2.5.0" \
    torchdeq einops \
    datasets transformers \
    wandb \
    lm-eval
```

### Step 3: Clone the repository and install pcg_llm

```python
# Cell 2 — Setup
!git clone https://github.com/your-username/pcg-llm.git /kaggle/working/pcg-llm
!pip install -e /kaggle/working/pcg-llm
import os
os.chdir("/kaggle/working/pcg-llm")
```

### Step 4: Start training with the Tiny PCG preset

```python
# Cell 3 — Train (runs until session ends or total_tokens reached)
!python -m pcg_llm.training.trainer train \
    --preset tiny \
    --checkpoint-dir /kaggle/working/checkpoints \
    --wandb-project pcg-llm-tiny \
    --seed 42
```

Training will auto-resume if the kernel is restarted and this cell is re-run. The checkpoint at `/kaggle/working/checkpoints/` survives kernel restarts within the same Kaggle session.

### Step 5: Monitor training health

Key metrics to watch in W&B (or in the notebook output if W&B is not configured):
- `solver_steps_mean` — should stay below 10 for the Tiny PCG
- `sparsity` — should stay between 0.75 and 0.95
- `node_variance` — must stay above 0.1; if it drops, γ auto-adjusts
- `loss_total` — should decrease monotonically after warmup

### Step 6: Benchmark gate evaluation

After training completes (or at a fluency checkpoint where perplexity ≤ 50):

```python
# Cell 4 — Evaluate
!python -m pcg_llm.training.trainer evaluate \
    --checkpoint /kaggle/working/checkpoints/step-XXXXXX.pt \
    --tasks arc_challenge,gsm8k \
    --quantize 4bit
```

If scores are within 5pp of SmolLM-135M published results: the Tiny PCG is promoted to a standalone deliverable. Otherwise, proceed directly to Track B.

---

## Track B: 3B PCG on GCP Preemptible A100

### Step 1: Activate free credit and create a VM

```bash
# On your local machine or GCP Cloud Shell
gcloud compute instances create pcg-llm-train \
  --machine-type a2-highgpu-1g \
  --accelerator type=nvidia-tesla-a100,count=1 \
  --provisioning-model SPOT \
  --image-family pytorch-2-5-cu124 \
  --image-project deeplearning-platform-release \
  --boot-disk-size 100GB \
  --scopes storage-rw
```

### Step 2: SSH into the VM and install

```bash
gcloud compute ssh pcg-llm-train
git clone https://github.com/your-username/pcg-llm.git
cd pcg-llm
pip install -e ".[triton]"
pip install bitsandbytes google-cloud-storage wandb lm-eval
```

### Step 3: Create a GCS bucket for checkpoints

```bash
gsutil mb gs://pcg-llm-checkpoints
```

### Step 4: Start training with the 3B preset

```bash
python -m pcg_llm.training.trainer train \
  --preset 3b \
  --checkpoint-backend gcs \
  --checkpoint-dir gs://pcg-llm-checkpoints/3b \
  --wandb-project pcg-llm-3b \
  --seed 42
```

The SIGTERM handler catches preemption events. When the VM is reclaimed:
- The script saves a checkpoint to GCS (typically completes within ~30 seconds)
- The process exits with code 2

### Step 5: Resume after preemption

Spin up a new preemptible VM (repeat Step 1), install (Step 2), then run the same Step 4 command. The script auto-detects the latest checkpoint in `gs://pcg-llm-checkpoints/3b` and resumes.

### Step 6: Benchmark evaluation

```bash
python -m pcg_llm.training.trainer evaluate \
  --checkpoint gs://pcg-llm-checkpoints/3b/step-XXXXXX.pt \
  --tasks arc_challenge,gsm8k,humaneval,mmlu,hellaswag \
  --quantize 4bit \
  --output results.json
```

---

## Configuration Quick Reference

| Scenario | Command |
|---|---|
| Tiny PCG (Kaggle defaults) | `--preset tiny` |
| 3B model (GCP defaults) | `--preset 3b` |
| Disable Muon (use AdamW only) | `--optimizer adamw` |
| Disable nGPT | `--normalize standard` |
| Disable Monarch | `--projection dense` |
| No gradient checkpointing | `--no-grad-checkpoint` |
| Faster checkpoints (less frequent) | `--checkpoint-interval 1000` |
| Export config for reproducibility | `export-config --preset tiny --output my_config.json` |

---

## Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `solver_steps_mean` consistently at max | Learning rate too high | Reduce `--base-lr` by 2× |
| `node_variance` < 0.1 immediately | γ coefficient too low | Increase `--gamma-variance 0.05` |
| `sparsity` below 0.80 | RigL drop fraction too low | Increase `--rigl-drop-fraction 0.15` |
| Checkpoint load fails (checksum mismatch) | Disk corruption | Previous checkpoint auto-selected; check manifest |
| GCS write timeout on preemption | SIGTERM received too late | Reduce `--checkpoint-interval` to 100 |
| EAGLE accept rate < 0.40 | EAGLE head needs fine-tuning | Run EAGLE fine-tuning phase before deployment |
