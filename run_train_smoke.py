"""Quick smoke-run training script.

Runs a small PCG-LLM for 300 steps on synthetic data and prints detailed
per-step diagnostics so we can verify the model is actually learning.

Usage:
    uv run python run_train_smoke.py
"""

from __future__ import annotations

import logging
import sys
import time
from typing import cast

import torch

from pcg_llm.config import TrainingConfig
from pcg_llm.training.trainer import PCGTrainer

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,  # suppress library noise
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)

# ── config ────────────────────────────────────────────────────────────────────
config = TrainingConfig(
    hidden_dim=64,  # must be >= 64 for monarch; PCGNode uses block_size anyway
    vocab_size=1024,  # small vocab → fast softmax on CPU
    max_seq_len=128,  # 4 blocks of 32
    block_size=32,
    initial_sparsity=0.80,
    rigl_interval=50,
    rigl_drop_fraction=0.10,
    max_solver_iters=12,
    solver_tolerance=1e-2,
    lambda_sparse=0.01,
    gamma_variance=0.01,
    variance_floor=0.1,
    base_lr=3e-4,
    warmup_steps=10,
    total_tokens=10_000_000,
    batch_size=4,
    grad_accum_steps=1,
    optimizer="adamw",  # plain AdamW — no Muon complexity
    optimizer_bits=32,
    normalize="ngpt",  # the normalization that was previously missing
    projection="dense",  # dense proj (monarch needs hidden_dim ≥ 64 per spec)
    grad_checkpoint=False,
    cpu_offload_mask=False,
    checkpoint_interval=99999,
    checkpoint_backend="local",
    checkpoint_dir="/tmp/pcg_smoke",
    wandb_project=None,
)

torch.manual_seed(42)

trainer = PCGTrainer(config=config)

print(f"\n{'='*72}")
print("PCG-LLM Smoke Run")
print(
    f"  vocab={config.vocab_size}  seq_len={config.max_seq_len}  "
    f"blocks={config.max_seq_len // config.block_size}  "
    f"block_size={config.block_size}"
)
print(f"  normalize={config.normalize}  optimizer={config.optimizer}  lr={config.base_lr}")
print(f"  solver: max_iter={config.max_solver_iters}  tol={config.solver_tolerance}")
print(f"  device: {trainer.device}")

# Count parameters
total_params = (
    sum(p.numel() for p in trainer._raw_node.parameters())
    + sum(p.numel() for p in trainer._raw_output_proj.parameters())
    + sum(p.numel() for p in trainer._raw_embedding.parameters())
    + trainer.adjacency.W_structure.numel()
)
print(f"  total trainable params: {total_params:,}")
print(f"{'='*72}\n")

# Print header
HDR = (
    f"{'step':>5}  {'loss':>8}  {'ce':>8}  {'l1':>7}  {'var':>7}  "
    f"{'solver':>6}  {'conv?':>5}  {'spars':>5}  "
    f"|g|node  |g|emb   |g|Wstr  "
    f"{'Z*norm':>7}  {'dt':>5}"
)
SEP = "-" * len(HDR)
print(HDR)
print(SEP)

STEPS = 300
LOG_EVERY = 10

torch.manual_seed(42)

losses = []
t0 = time.perf_counter()

for step in range(1, STEPS + 1):
    # synthetic random-token batch
    batch = torch.randint(0, config.vocab_size, (config.batch_size, config.max_seq_len))

    # ── forward + backward ───────────────────────────────────────────────────
    trainer.optimizer.zero_grad()

    batch_dev = batch.to(trainer.device)
    B, L = batch_dev.shape
    num_blocks = L // config.block_size

    block_input_tokens = batch_dev[:, :: config.block_size]
    Z0 = trainer.embedding(block_input_tokens)

    def f_theta(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        aggregated = trainer.adjacency.message_pass(z)
        updated, _ = trainer.node(z, aggregated + x, apply_norm=False)
        return cast(torch.Tensor, updated)

    Z_star, solver_info = trainer.solver.solve(f_theta, Z0, Z0)
    Z_star = trainer._raw_node._norm(Z_star)  # type: ignore[operator]

    n_pred = num_blocks - 1
    logits = trainer.output_proj(Z_star[:, :-1, :])
    targets = batch_dev[:, config.block_size :: config.block_size]
    logits_flat = logits.reshape(B * n_pred, config.vocab_size)
    targets_flat = targets.reshape(B * n_pred)

    total_loss, components = trainer.loss_fn(
        logits_flat, targets_flat, trainer.adjacency.W_structure, Z_star
    )
    total_loss.backward()

    # gradient norms (before clipping)
    node_gnorm = (
        sum(p.grad.norm().item() ** 2 for p in trainer._raw_node.parameters() if p.grad is not None)
        ** 0.5
    )
    emb_gnorm = (
        trainer._raw_embedding.weight.grad.norm().item()  # type: ignore[operator]
        if trainer._raw_embedding.weight.grad is not None  # type: ignore[union-attr]
        else 0.0
    )
    wstr_gnorm = (
        trainer.adjacency.W_structure.grad.norm().item()
        if trainer.adjacency.W_structure.grad is not None
        else 0.0
    )

    torch.nn.utils.clip_grad_norm_(
        list(trainer._raw_node.parameters()) + list(trainer._raw_output_proj.parameters()),
        max_norm=1.0,
    )
    trainer.optimizer.step()
    trainer.step += 1

    # RigL
    if trainer.step > 0 and trainer.step % config.rigl_interval == 0:
        _, should_freeze = trainer.scheduler.step()
        if not trainer.adjacency.is_frozen:
            grads = (
                trainer.adjacency.W_structure.grad
                if trainer.adjacency.W_structure.grad is not None
                else torch.randn_like(trainer.adjacency.W_structure).abs()
            )
            trainer.adjacency.drop_and_grow(grads)

    loss_val = total_loss.item()
    losses.append(loss_val)

    if step % LOG_EVERY == 0:
        dt = time.perf_counter() - t0
        ce_val = components["cross_entropy"].item()
        l1_val = components["l1_sparsity"].item()
        var_val = components["variance"].item()
        conv = "Y" if solver_info.get("converged", False) else "N"
        s_steps = solver_info.get("solver_steps", 0)
        spars = trainer.adjacency.sparsity()
        z_norm = Z_star.norm(dim=-1).mean().item()  # mean block norm

        print(
            f"{step:>5}  {loss_val:>8.4f}  {ce_val:>8.4f}  {l1_val:>7.4f}  {var_val:>7.4f}  "
            f"{s_steps:>6}  {conv:>5}  {spars:>5.3f}  "
            f"{node_gnorm:>6.2e}  {emb_gnorm:>6.2e}  {wstr_gnorm:>6.2e}  "
            f"{z_norm:>7.4f}  {dt:>5.1f}s"
        )
        t0 = time.perf_counter()

print(SEP)

# ── summary ───────────────────────────────────────────────────────────────────
first10 = sum(losses[:10]) / 10
last10 = sum(losses[-10:]) / 10
midpoint = sum(losses[140:160]) / 20

print(f"\nLoss summary over {STEPS} steps:")
print(f"  steps  1-10  avg: {first10:.4f}")
print(f"  steps 141-160 avg: {midpoint:.4f}")
print(f"  steps 291-300 avg: {last10:.4f}")
print(f"  total reduction:  {first10 - last10:+.4f}  ({(first10-last10)/first10*100:.1f}%)")

if last10 < first10 * 0.95:
    print("\n✓ LOSS IS DECREASING — training is working")
else:
    print("\n✗ Loss NOT decreasing significantly — still broken")
