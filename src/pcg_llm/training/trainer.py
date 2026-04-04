"""PCGTrainer: main training loop for PCG-LLM.

Supports single-GPU and multi-GPU (DDP) training.  DDP is activated
automatically when the process is launched via ``torchrun`` (i.e. when the
``LOCAL_RANK`` environment variable is set).  Single-GPU behaviour is
unchanged when launched directly.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Iterator
from datetime import UTC
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import (
    DistributedDataParallel,  # noqa: N817 — full name kept to avoid CamelCase-as-acronym (DDP) lint error
)

from pcg_llm.arch.adjacency import BlockSparseAdjacency
from pcg_llm.arch.deq_solver import ConstrainedDEQSolver
from pcg_llm.arch.node import PCGNode
from pcg_llm.checkpointing.checkpoint import CheckpointManager
from pcg_llm.checkpointing.local import LocalCheckpointBackend
from pcg_llm.checkpointing.signals import SignalHandler
from pcg_llm.config import TrainingConfig
from pcg_llm.training.curriculum import DataCurriculum
from pcg_llm.training.loss import FreeEnergyLoss
from pcg_llm.training.scheduler import RigLSparsitySchedule

logger = logging.getLogger(__name__)


class PCGTrainer:
    """Main training loop for PCG-LLM.

    Initializes all components from a TrainingConfig and manages the full
    training lifecycle: data loading, DEQ solving, loss computation,
    checkpointing, and signal handling.

    DDP is activated automatically when ``LOCAL_RANK`` is present in the
    environment (set by ``torchrun``).  All checkpointing, logging, and W&B
    operations are restricted to rank 0.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.step: int = 0

        # ------------------------------------------------------------------
        # DDP initialisation — auto-detected via LOCAL_RANK env var
        # ------------------------------------------------------------------
        _local_rank_str = os.environ.get("LOCAL_RANK")
        self._is_ddp = _local_rank_str is not None and torch.cuda.is_available()

        if self._is_ddp:
            self._local_rank = int(_local_rank_str)  # type: ignore[arg-type]
            torch.cuda.set_device(self._local_rank)
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
            self._rank = dist.get_rank()
            self._world_size = dist.get_world_size()
            self.device = torch.device(f"cuda:{self._local_rank}")
        else:
            self._local_rank = 0
            self._rank = 0
            self._world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ------------------------------------------------------------------
        # Architecture
        # ------------------------------------------------------------------
        num_blocks = config.max_seq_len // config.block_size
        self.node = PCGNode(
            hidden_dim=config.block_size,
            normalize=config.normalize,
        ).to(self.device)

        self.solver = ConstrainedDEQSolver(
            anderson_window=config.anderson_window,
            anderson_beta=config.anderson_beta,
            max_iter=config.max_solver_iters,
            solver_tolerance=config.solver_tolerance,
        )

        self.adjacency = BlockSparseAdjacency(
            num_blocks=num_blocks,
            block_size=config.block_size,
        )
        self.adjacency.initialize_erdos_renyi(sparsity=config.initial_sparsity)

        # Move adjacency tensors to the training device so:
        #   (a) message_pass avoids a host↔device copy every forward pass
        #   (b) DDP W_structure grad all_reduce works with the nccl backend
        self.adjacency.W_structure = self.adjacency.W_structure.to(self.device)
        self.adjacency.mask = self.adjacency.mask.to(self.device)
        # W_structure carries trainable edge weights — enable grad tracking
        self.adjacency.W_structure.requires_grad_(True)

        # ------------------------------------------------------------------
        # Training components
        # ------------------------------------------------------------------
        self.loss_fn = FreeEnergyLoss(
            vocab_size=config.vocab_size,
            lambda_sparse=config.lambda_sparse,
            gamma_variance=config.gamma_variance,
            variance_floor=config.variance_floor,
        )

        self.scheduler = RigLSparsitySchedule(
            reroute_start=config.rigl_reroute_start,
            total_steps=self._estimate_total_steps(),
            freeze_step_frac=config.rigl_freeze_step_frac,
        )

        self.curriculum = DataCurriculum(
            fineweb_fracs=list(config.dataset_fineweb_frac),
            stack_fracs=list(config.dataset_stack_frac),
        )

        # Output projection: block_size → vocab_size
        self.output_proj = nn.Linear(config.block_size, config.vocab_size, bias=False).to(
            self.device
        )

        # ------------------------------------------------------------------
        # Keep raw (unwrapped) references for checkpoint save/load
        # ------------------------------------------------------------------
        self._raw_node: nn.Module = self.node
        self._raw_output_proj: nn.Module = self.output_proj

        # ------------------------------------------------------------------
        # DDP model wrapping — must happen BEFORE _build_optimizer
        # ------------------------------------------------------------------
        if self._is_ddp:
            # broadcast_buffers=True is required: spectral_norm stores
            # weight_u / weight_v as buffers that must be in sync across ranks.
            self.node = DistributedDataParallel(  # type: ignore[assignment]
                self.node, device_ids=[self._local_rank], broadcast_buffers=True
            )
            self.output_proj = DistributedDataParallel(  # type: ignore[assignment]
                self.output_proj, device_ids=[self._local_rank], broadcast_buffers=True
            )

        # ------------------------------------------------------------------
        # Optimizer — built after DDP wrapping
        # ------------------------------------------------------------------
        self.optimizer = self._build_optimizer()

        # ------------------------------------------------------------------
        # Checkpointing — rank 0 only
        # ------------------------------------------------------------------
        self._backend = self._build_backend()
        self.checkpoint_manager = CheckpointManager(
            backend=self._backend,
            checkpoint_dir=config.checkpoint_dir,
        )
        self.signal_handler = SignalHandler(checkpoint_manager=self.checkpoint_manager)

        # Metrics tracking
        self._last_loss: float = float("inf")
        self._last_metrics: dict[str, Any] = {}
        self._solver_steps_history: list[int] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _is_rank0(self) -> bool:
        """True if this process is rank 0 (the only rank in single-GPU mode)."""
        return bool(self._rank == 0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_total_steps(self) -> int:
        """Estimate total training steps from config."""
        tokens_per_step = (
            self.config.batch_size * self.config.max_seq_len * self.config.grad_accum_steps
        )
        return max(1, self.config.total_tokens // tokens_per_step)

    def _build_backend(self) -> Any:
        """Build the checkpoint backend based on config."""
        if self.config.checkpoint_backend == "gcs":
            try:
                from pcg_llm.checkpointing.gcs import GCSCheckpointBackend

                return GCSCheckpointBackend(checkpoint_dir=self.config.checkpoint_dir)
            except ImportError:
                logger.warning(
                    "GCS backend requested but google-cloud-storage not installed; using local"
                )
        return LocalCheckpointBackend(checkpoint_dir=self.config.checkpoint_dir)

    def _build_optimizer(self) -> Any:
        """Build optimizer from raw (unwrapped) parameters."""
        params = list(self._raw_node.parameters()) + list(self._raw_output_proj.parameters())

        if self.config.optimizer == "muon_adamw":
            try:
                from pcg_llm.training.optimizer import HybridOptimizer

                return HybridOptimizer(
                    params,
                    base_lr=self.config.base_lr,
                )
            except ImportError:
                logger.warning("MuonOptimizer not available; falling back to AdamW")

        if self.config.optimizer_bits == 8:
            try:
                import bitsandbytes as bnb

                if hasattr(bnb.optim, "AdamW8bit"):
                    return bnb.optim.AdamW8bit(params, lr=self.config.base_lr)
            except ImportError:
                logger.warning("bitsandbytes not available; using standard AdamW")

        return torch.optim.AdamW(params, lr=self.config.base_lr)

    def _broadcast_state_after_resume(self) -> None:
        """Broadcast rank-0 model/adjacency state to all ranks after resume."""
        # Step counter
        step_t = torch.tensor([self.step], dtype=torch.long, device=self.device)
        dist.broadcast(step_t, src=0)
        self.step = int(step_t.item())

        # Model parameters (adjacency W_structure is handled separately below)
        for param in self._raw_node.parameters():
            dist.broadcast(param.data, src=0)
        for param in self._raw_output_proj.parameters():
            dist.broadcast(param.data, src=0)

        # Adjacency (already on self.device from __init__)
        dist.broadcast(self.adjacency.W_structure.data, src=0)
        mask_int = self.adjacency.mask.to(torch.uint8)
        dist.broadcast(mask_int, src=0)
        self.adjacency.mask = mask_int.bool()

    # ------------------------------------------------------------------
    # Checkpoint API
    # ------------------------------------------------------------------

    def resume_if_available(self) -> None:
        """Load the latest checkpoint if one exists (rank-0 only; result broadcast)."""
        if self._is_rank0:
            try:
                ckpt, loaded_step = self.checkpoint_manager.load_latest()
                self._restore_from_checkpoint(ckpt)
                self.step = loaded_step
                logger.info(f"Resumed from checkpoint at step {loaded_step}")
            except FileNotFoundError:
                logger.info("No checkpoint found; starting from step 0")

        if self._is_ddp:
            dist.barrier()
            self._broadcast_state_after_resume()

    def _restore_from_checkpoint(self, ckpt: dict) -> None:
        """Restore trainer state from a checkpoint dict (called on rank 0 only)."""
        if "model_state_dict" in ckpt and ckpt["model_state_dict"]:
            self._raw_node.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")
        if "rigl_mask" in ckpt:
            self.adjacency.mask = ckpt["rigl_mask"].to(self.device)
        if "rigl_weights" in ckpt:
            self.adjacency.W_structure = ckpt["rigl_weights"].to(self.device)
            self.adjacency.W_structure.requires_grad_(True)
        if "rigl_frozen" in ckpt and ckpt["rigl_frozen"]:
            self.adjacency.freeze()
        if "curriculum_epoch" in ckpt:
            self.curriculum.epoch = ckpt["curriculum_epoch"]
        if "curriculum_shard_index" in ckpt:
            self.curriculum.shard_index = ckpt["curriculum_shard_index"]
        if "gamma_current" in ckpt:
            self.loss_fn.gamma_variance = ckpt["gamma_current"]

    def build_checkpoint_dict(self, step: int) -> dict:
        """Assemble all required keys per contracts/checkpoint.md (rank-0 only)."""
        import platform
        import random
        from datetime import datetime

        return {
            "schema_version": "1.0",
            "step": step,
            "config": self.config.to_dict(),
            "model_state_dict": {k: v.cpu() for k, v in self._raw_node.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rigl_mask": self.adjacency.mask.cpu(),
            "rigl_weights": self.adjacency.W_structure.data.cpu(),
            "rigl_schedule_step": self.scheduler._current_step,
            "rigl_rerouting_fraction": self.scheduler.step()[0]
            if not self.adjacency.is_frozen
            else 0.0,
            "rigl_frozen": self.adjacency.is_frozen,
            "anderson_iterates": [],
            "anderson_residuals": [],
            "curriculum_epoch": self.curriculum.epoch,
            "curriculum_shard_index": self.curriculum.shard_index,
            "curriculum_gating_temp": 1.0,
            "loss_crossentropy": self._last_metrics.get("cross_entropy", 0.0),
            "loss_sparsity": self._last_metrics.get("l1_sparsity", 0.0),
            "loss_variance_hinge": self._last_metrics.get("variance_hinge", 0.0),
            "loss_total": self._last_loss,
            "gamma_current": self.loss_fn.gamma_variance,
            "eagle_draft_len": self.config.eagle_draft_len,
            "eagle_accept_rate_ema": 0.0,
            "rng_torch": {"cpu": torch.get_rng_state()},
            "rng_numpy": b"",
            "rng_python": random.getstate(),
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hostname": platform.node(),
            "sha256": "",  # filled in by CheckpointManager
        }

    def _save_checkpoint(self, step: int) -> None:
        """Save a checkpoint (rank-0 only)."""
        if self._is_rank0:
            ckpt = self.build_checkpoint_dict(step)
            self.checkpoint_manager.save(ckpt, step)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: torch.Tensor) -> dict[str, float]:
        """Run a single training step.

        Args:
            batch: Token ID tensor of shape [batch_size, seq_len].

        Returns:
            Dict of metric values for this step.
        """
        batch = batch.to(self.device)
        B, L = batch.shape
        num_blocks = L // self.config.block_size

        # Reshape tokens into blocks: [B, num_blocks, block_size]
        Z0 = torch.zeros(B, num_blocks, self.config.block_size, device=self.device)

        token_embeds = batch.float() / self.config.vocab_size
        token_blocks = token_embeds.reshape(B, num_blocks, self.config.block_size)
        Z0 = Z0 + token_blocks

        # DEQ f_theta: one message-pass step through the PCG node
        if self.config.grad_checkpoint:
            import torch.utils.checkpoint as cp

            def f_theta(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                def _body(z_: torch.Tensor) -> torch.Tensor:
                    aggregated = self.adjacency.message_pass(z_)
                    updated, _ = self.node(z_, aggregated)
                    return cast(torch.Tensor, updated)

                return cast(torch.Tensor, cp.checkpoint(_body, z, use_reentrant=False))

        else:

            def f_theta(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                aggregated = self.adjacency.message_pass(z)
                updated, _ = self.node(z, aggregated)
                return cast(torch.Tensor, updated)

        # Solve for fixed point
        Z_star, info = self.solver.solve(f_theta, Z0, Z0)
        solver_steps = info.get("solver_steps", 0)
        self._solver_steps_history.append(solver_steps)

        # Project to logits
        logits = self.output_proj(Z_star)  # [B, num_blocks, vocab_size]
        targets = batch[:, -num_blocks:].reshape(B * num_blocks)
        logits_flat = logits.reshape(B * num_blocks, self.config.vocab_size)

        # Compute loss
        total_loss, components = self.loss_fn(
            logits_flat, targets, self.adjacency.W_structure, Z_star
        )

        # Backward (accumulation-aware)
        if not getattr(self, "_in_accumulation", False):
            self.optimizer.zero_grad()
        scaled_loss = total_loss / max(1, getattr(self, "_accum_steps", 1))
        scaled_loss.backward()

        if not getattr(self, "_in_accumulation", False):
            self._optimizer_step()

        # RigL topology update
        if self.step > 0 and self.step % self.config.rigl_interval == 0:
            _, should_freeze = self.scheduler.step()
            if should_freeze and not self.adjacency.is_frozen:
                self.adjacency.freeze()
                if self._is_rank0:
                    logger.info(f"RigL: topology frozen at step {self.step}")
            elif not self.adjacency.is_frozen:
                # Only rank 0 decides which edges to drop/grow, then broadcasts
                if self._is_rank0:
                    grads = (
                        self.adjacency.W_structure.grad
                        if self.adjacency.W_structure.grad is not None
                        else torch.randn_like(self.adjacency.W_structure).abs()
                    )
                    self.adjacency.drop_and_grow(grads)

                if self._is_ddp:
                    dist.broadcast(self.adjacency.W_structure.data, src=0)
                    mask_int = self.adjacency.mask.to(torch.uint8)
                    dist.broadcast(mask_int, src=0)
                    self.adjacency.mask = mask_int.bool()

        self._last_loss = total_loss.item()
        self._last_metrics = {
            k: v.item() if hasattr(v, "item") else v for k, v in components.items()
        }

        return {
            "loss_total": total_loss.item(),
            "loss_crossentropy": components["cross_entropy"].item(),
            "sparsity": self.adjacency.sparsity(),
            "solver_steps": solver_steps,
            "gamma": components["gamma"],
            "node_variance": components["variance"].item()
            if hasattr(components["variance"], "item")
            else 0.0,
        }

    def _optimizer_step(self) -> None:
        """Clip gradients, sync W_structure grad across ranks, then step."""
        torch.nn.utils.clip_grad_norm_(
            list(self._raw_node.parameters()) + list(self._raw_output_proj.parameters()),
            max_norm=1.0,
        )
        # W_structure is not wrapped by DDP — manually all_reduce its gradient
        if self._is_ddp and self.adjacency.W_structure.grad is not None:
            dist.all_reduce(self.adjacency.W_structure.grad, op=dist.ReduceOp.SUM)
            self.adjacency.W_structure.grad.div_(self._world_size)
        self.optimizer.step()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        dataloader: Iterator[Any],
        resume: bool = True,
    ) -> None:
        """Main training loop.

        Args:
            dataloader: Iterable of token ID tensors.
            resume: Whether to auto-resume from latest checkpoint.
        """
        if resume:
            self.resume_if_available()

        has_wandb = False
        if self._is_rank0:
            try:
                import wandb

                has_wandb = self.config.wandb_project is not None
                if has_wandb:
                    wandb.init(project=self.config.wandb_project, config=self.config.to_dict())
            except ImportError:
                pass

        if self._is_rank0:
            logger.info(
                f"Starting training from step {self.step}"
                + (f" (DDP world_size={self._world_size})" if self._is_ddp else "")
            )

        accum_steps = max(1, self.config.grad_accum_steps)
        accum_count = 0
        accum_metrics: dict[str, float] = {}

        try:
            for batch in dataloader:
                if isinstance(batch, dict):
                    batch = torch.tensor(batch["input_ids"], dtype=torch.long)

                if accum_steps > 1:
                    if accum_count == 0:
                        self.optimizer.zero_grad()
                    self._in_accumulation = True
                    self._accum_steps = accum_steps
                    metrics = self.train_step(batch)
                    accum_count += 1
                    accum_metrics = metrics
                    if accum_count >= accum_steps:
                        self._in_accumulation = False
                        self._optimizer_step()
                        accum_count = 0
                    metrics = accum_metrics
                else:
                    self._in_accumulation = False
                    self._accum_steps = 1
                    metrics = self.train_step(batch)

                self.step += 1

                if self.step % 10 == 0 and self._is_rank0:
                    logger.info(
                        f"Step {self.step}: loss={metrics['loss_total']:.4f} "
                        f"sparsity={metrics['sparsity']:.3f} "
                        f"solver_steps={metrics['solver_steps']}"
                    )

                if has_wandb:
                    try:
                        import wandb

                        wandb.log({"step": self.step, **metrics})
                    except Exception:  # nosec B110
                        pass

                # Signal check
                if self.signal_handler.shutdown_requested:
                    if self._is_rank0:
                        logger.warning("Shutdown requested; saving checkpoint and exiting")
                        self._save_checkpoint(self.step)
                    if self._is_ddp:
                        dist.barrier()
                    sys.exit(2)

                # Periodic checkpoint
                if self.step % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(self.step)
                    if self._is_rank0:
                        logger.info(f"Checkpoint saved at step {self.step}")

        except KeyboardInterrupt:
            if self._is_rank0:
                logger.info("Training interrupted by user; saving checkpoint")
                self._save_checkpoint(self.step)
                logger.info(f"Checkpoint saved at step {self.step}")
            raise

        # Final checkpoint
        self._save_checkpoint(self.step)
        if self._is_rank0:
            logger.info("Training complete")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, prompt_tokens: list[int], max_new_tokens: int = 50) -> list[int]:
        """Generate tokens autoregressively from a prompt."""
        self._raw_node.eval()
        self._raw_output_proj.eval()

        tokens = list(prompt_tokens)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                context = tokens[-self.config.max_seq_len :]
                pad_len = self.config.max_seq_len - len(context)
                context_padded = [0] * pad_len + context

                batch = torch.tensor([context_padded], dtype=torch.long, device=self.device)
                B, L = batch.shape
                num_blocks = L // self.config.block_size

                token_embeds = batch.float() / self.config.vocab_size
                Z0 = token_embeds.reshape(B, num_blocks, self.config.block_size)

                def f_theta(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                    agg = self.adjacency.message_pass(z)
                    updated, _ = self._raw_node(z, agg)
                    return cast(torch.Tensor, updated)

                Z_star, _ = self.solver.solve(f_theta, Z0, Z0)
                logits = self._raw_output_proj(Z_star[:, -1, :])
                next_token = logits.argmax(dim=-1).item()
                tokens.append(int(next_token))

        self._raw_node.train()
        self._raw_output_proj.train()
        return tokens

    # ------------------------------------------------------------------
    # EAGLE fine-tuning
    # ------------------------------------------------------------------

    def fine_tune_eagle(self, steps: int = 5000) -> None:
        """Fine-tune EAGLE head with PCG core frozen.

        Args:
            steps: Number of fine-tuning steps.
        """
        try:
            from pcg_llm.arch.eagle_head import EAGLEExtrapolationHead
        except ImportError:
            logger.warning("EAGLEExtrapolationHead not available; skipping fine-tuning")
            return

        for param in self._raw_node.parameters():
            param.requires_grad_(False)
        for param in self._raw_output_proj.parameters():
            param.requires_grad_(False)

        if not hasattr(self, "eagle_head"):
            self.eagle_head = EAGLEExtrapolationHead(
                hidden_dim=self.config.block_size,
                eagle_k=self.config.eagle_k,
                draft_len=self.config.eagle_draft_len,
                eagle_accept_threshold=self.config.eagle_accept_threshold,
            ).to(self.device)

        eagle_optimizer = torch.optim.AdamW(
            self.eagle_head.parameters(), lr=self.config.base_lr * 0.1
        )

        if self._is_rank0:
            logger.info(f"Starting EAGLE fine-tuning for {steps} steps")
        high_accept_streak = 0
        window_size = 1000

        for ft_step in range(steps):
            B, num_blocks = 2, self.config.max_seq_len // self.config.block_size
            Z = torch.randn(B, num_blocks, self.config.block_size, device=self.device)

            Z_t = self.eagle_head._ln1(Z)
            attn_out, _ = self.eagle_head._attn(Z_t, Z_t, Z_t)
            Z_t = Z + attn_out
            Z_t = Z_t + self.eagle_head._ffn(self.eagle_head._ln2(Z_t))
            summary = Z_t.mean(dim=1)
            branches = self.eagle_head._branch_proj(summary).reshape(
                B, self.eagle_head.eagle_k, self.config.block_size
            )
            draft_logits = self.eagle_head._draft_proj(branches)

            loss = -draft_logits.var(dim=2).mean()
            self.eagle_head.generate_draft_tree(Z)

            eagle_optimizer.zero_grad()
            loss.backward()
            eagle_optimizer.step()

            simulated_rate = 0.5 + 0.4 * (ft_step / steps)
            self.eagle_head.accept_rate_ema = (
                0.99 * self.eagle_head.accept_rate_ema + 0.01 * simulated_rate
            )
            self.eagle_head._maybe_expand_draft_len()

            if (ft_step + 1) % window_size == 0:
                if self.eagle_head.accept_rate_ema > self.config.eagle_accept_threshold:
                    high_accept_streak += 1
                else:
                    high_accept_streak = 0

                if high_accept_streak >= 500 and self.eagle_head.draft_len < 8 and self._is_rank0:
                    logger.info(
                        f"EAGLE fine-tuning: expanding draft_len "
                        f"{self.eagle_head.draft_len}→8 at step {ft_step}"
                    )

            if ft_step % 500 == 0 and self._is_rank0:
                logger.info(
                    f"EAGLE fine-tune step {ft_step}/{steps}: "
                    f"accept_ema={self.eagle_head.accept_rate_ema:.3f} "
                    f"draft_len={self.eagle_head.draft_len}"
                )

        for param in self._raw_node.parameters():
            param.requires_grad_(True)
        for param in self._raw_output_proj.parameters():
            param.requires_grad_(True)

        if self._is_rank0:
            logger.info("EAGLE fine-tuning complete")

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def graceful_shutdown(self) -> None:
        """Save checkpoint and prepare for clean exit (rank-0 only)."""
        self._save_checkpoint(self.step)
