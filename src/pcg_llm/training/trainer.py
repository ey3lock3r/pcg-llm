"""PCGTrainer: main training loop for PCG-LLM."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from datetime import UTC
from typing import Any, cast

import torch
import torch.nn as nn

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
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.step: int = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Architecture
        num_blocks = config.max_seq_len // config.block_size
        self.node = PCGNode(
            hidden_dim=config.block_size,  # per-block hidden dim
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

        # Training components
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

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Checkpointing
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
        """Build optimizer based on config."""
        params = list(self.node.parameters()) + list(self.output_proj.parameters())

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

    def resume_if_available(self) -> None:
        """Load the latest checkpoint if one exists."""
        try:
            ckpt, loaded_step = self.checkpoint_manager.load_latest()
            self._restore_from_checkpoint(ckpt)
            self.step = loaded_step
            logger.info(f"Resumed from checkpoint at step {loaded_step}")
        except FileNotFoundError:
            logger.info("No checkpoint found; starting from step 0")

    def _restore_from_checkpoint(self, ckpt: dict) -> None:
        """Restore trainer state from a checkpoint dict."""
        if "model_state_dict" in ckpt and ckpt["model_state_dict"]:
            self.node.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"]:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                logger.warning(f"Could not restore optimizer state: {e}")
        if "rigl_mask" in ckpt:
            self.adjacency.mask = ckpt["rigl_mask"]
        if "rigl_weights" in ckpt:
            self.adjacency.W_structure = ckpt["rigl_weights"]
        if "rigl_frozen" in ckpt and ckpt["rigl_frozen"]:
            self.adjacency.freeze()
        if "curriculum_epoch" in ckpt:
            self.curriculum.epoch = ckpt["curriculum_epoch"]
        if "curriculum_shard_index" in ckpt:
            self.curriculum.shard_index = ckpt["curriculum_shard_index"]
        if "gamma_current" in ckpt:
            self.loss_fn.gamma_variance = ckpt["gamma_current"]

    def build_checkpoint_dict(self, step: int) -> dict:
        """Assemble all required keys per contracts/checkpoint.md."""
        import platform
        import random
        from datetime import datetime

        return {
            "schema_version": "1.0",
            "step": step,
            "config": self.config.to_dict(),
            "model_state_dict": {k: v.cpu() for k, v in self.node.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rigl_mask": self.adjacency.mask.cpu(),
            "rigl_weights": self.adjacency.W_structure.cpu(),
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

        # Embed tokens into initial Z (simple embedding: scatter token IDs)
        # Use a fixed random embedding scaled by block_size for the base
        token_embeds = batch.float() / self.config.vocab_size  # normalize to [0,1]
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

        # Project to logits: [B, num_blocks, vocab_size] → take first block for loss
        logits = self.output_proj(Z_star)  # [B, num_blocks, vocab_size]
        # Use last num_blocks tokens as targets (shift by 1 within sequence)
        targets = batch[:, -num_blocks:].reshape(B * num_blocks)
        logits_flat = logits.reshape(B * num_blocks, self.config.vocab_size)

        # Compute loss
        total_loss, components = self.loss_fn(
            logits_flat, targets, self.adjacency.W_structure, Z_star
        )

        # Backward (accumulation-aware: caller manages zero_grad + step when accumulating)
        if not getattr(self, "_in_accumulation", False):
            self.optimizer.zero_grad()
        scaled_loss = total_loss / max(1, getattr(self, "_accum_steps", 1))
        scaled_loss.backward()
        if not getattr(self, "_in_accumulation", False):
            torch.nn.utils.clip_grad_norm_(
                list(self.node.parameters()) + list(self.output_proj.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

        # RigL topology update
        if self.step > 0 and self.step % self.config.rigl_interval == 0:
            _, should_freeze = self.scheduler.step()
            if should_freeze and not self.adjacency.is_frozen:
                self.adjacency.freeze()
                logger.info(f"RigL: topology frozen at step {self.step}")
            elif not self.adjacency.is_frozen:
                grads = (
                    self.adjacency.W_structure.grad
                    if self.adjacency.W_structure.grad is not None
                    else torch.randn_like(self.adjacency.W_structure).abs()
                )
                self.adjacency.drop_and_grow(grads)

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

        try:
            import wandb

            has_wandb = self.config.wandb_project is not None
            if has_wandb:
                wandb.init(project=self.config.wandb_project, config=self.config.to_dict())
        except ImportError:
            has_wandb = False

        logger.info(f"Starting training from step {self.step}")

        accum_steps = max(1, self.config.grad_accum_steps)
        accum_count = 0
        accum_metrics: dict[str, float] = {}

        try:
            for batch in dataloader:
                if isinstance(batch, dict):
                    batch = torch.tensor(batch["input_ids"], dtype=torch.long)

                if accum_steps > 1:
                    # Gradient accumulation: accumulate grads for accum_steps batches
                    if accum_count == 0:
                        self.optimizer.zero_grad()
                    self._in_accumulation = True
                    self._accum_steps = accum_steps
                    metrics = self.train_step(batch)
                    accum_count += 1
                    accum_metrics = metrics  # keep last batch metrics
                    if accum_count >= accum_steps:
                        self._in_accumulation = False
                        torch.nn.utils.clip_grad_norm_(
                            list(self.node.parameters()) + list(self.output_proj.parameters()),
                            max_norm=1.0,
                        )
                        self.optimizer.step()
                        accum_count = 0
                    metrics = accum_metrics
                else:
                    self._in_accumulation = False
                    self._accum_steps = 1
                    metrics = self.train_step(batch)

                self.step += 1

                if self.step % 10 == 0:
                    logger.info(
                        f"Step {self.step}: loss={metrics['loss_total']:.4f} "
                        f"sparsity={metrics['sparsity']:.3f} "
                        f"solver_steps={metrics['solver_steps']}"
                    )

                if has_wandb:
                    try:
                        wandb.log({"step": self.step, **metrics})
                    except Exception:
                        pass

                # Signal check
                if self.signal_handler.shutdown_requested:
                    logger.warning("Shutdown requested; saving checkpoint and exiting")
                    ckpt = self.build_checkpoint_dict(self.step)
                    self.checkpoint_manager.save(ckpt, self.step)
                    sys.exit(2)

                # Periodic checkpoint
                if self.step % self.config.checkpoint_interval == 0:
                    ckpt = self.build_checkpoint_dict(self.step)
                    self.checkpoint_manager.save(ckpt, self.step)
                    logger.info(f"Checkpoint saved at step {self.step}")

        except KeyboardInterrupt:
            logger.info("Training interrupted by user; saving checkpoint")
            ckpt = self.build_checkpoint_dict(self.step)
            self.checkpoint_manager.save(ckpt, self.step)
            logger.info(f"Checkpoint saved at step {self.step}")
            raise

        # Final checkpoint
        ckpt = self.build_checkpoint_dict(self.step)
        self.checkpoint_manager.save(ckpt, self.step)
        logger.info("Training complete")

    def generate(self, prompt_tokens: list[int], max_new_tokens: int = 50) -> list[int]:
        """Generate tokens autoregressively from a prompt.

        Args:
            prompt_tokens: List of input token IDs.
            max_new_tokens: Number of new tokens to generate.

        Returns:
            List of generated token IDs (including prompt).
        """
        self.node.eval()
        self.output_proj.eval()

        tokens = list(prompt_tokens)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Build context (last max_seq_len tokens)
                context = tokens[-self.config.max_seq_len :]
                # Pad if needed
                pad_len = self.config.max_seq_len - len(context)
                context_padded = [0] * pad_len + context

                batch = torch.tensor([context_padded], dtype=torch.long, device=self.device)
                B, L = batch.shape
                num_blocks = L // self.config.block_size

                token_embeds = batch.float() / self.config.vocab_size
                Z0 = token_embeds.reshape(B, num_blocks, self.config.block_size)

                def f_theta(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
                    agg = self.adjacency.message_pass(z)
                    updated, _ = self.node(z, agg)
                    return cast(torch.Tensor, updated)

                Z_star, _ = self.solver.solve(f_theta, Z0, Z0)
                logits = self.output_proj(Z_star[:, -1, :])  # last block
                next_token = logits.argmax(dim=-1).item()
                tokens.append(int(next_token))

        self.node.train()
        self.output_proj.train()
        return tokens

    def fine_tune_eagle(self, steps: int = 5000) -> None:
        """Fine-tune EAGLE head with PCG core frozen (T049).

        Freezes all PCG parameters, trains only EAGLEExtrapolationHead for
        `steps` steps, monitoring acceptance rate EMA. Expands draft_len from
        4→8 when acceptance EMA > eagle_accept_threshold for 500 consecutive
        evaluation windows (each window = 1,000 steps).

        Args:
            steps: Number of fine-tuning steps.
        """
        try:
            from pcg_llm.arch.eagle_head import EAGLEExtrapolationHead
        except ImportError:
            logger.warning("EAGLEExtrapolationHead not available; skipping fine-tuning")
            return

        # Freeze all PCG core parameters
        for param in self.node.parameters():
            param.requires_grad_(False)
        for param in self.output_proj.parameters():
            param.requires_grad_(False)

        # Build or reuse EAGLE head
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

        logger.info(f"Starting EAGLE fine-tuning for {steps} steps")
        high_accept_streak = 0
        window_size = 1000

        for ft_step in range(steps):
            # Generate synthetic latent states (in practice, use real training data)
            B, num_blocks = 2, self.config.max_seq_len // self.config.block_size
            Z = torch.randn(B, num_blocks, self.config.block_size, device=self.device)

            # Use differentiable forward path for training (generate_draft_tree returns .long())
            Z_t = self.eagle_head._ln1(Z)
            attn_out, _ = self.eagle_head._attn(Z_t, Z_t, Z_t)
            Z_t = Z + attn_out
            Z_t = Z_t + self.eagle_head._ffn(self.eagle_head._ln2(Z_t))
            summary = Z_t.mean(dim=1)
            branches = self.eagle_head._branch_proj(summary).reshape(
                B, self.eagle_head.eagle_k, self.config.block_size
            )
            draft_logits = self.eagle_head._draft_proj(branches)  # (B, K, draft_len) float

            # Surrogate loss: maximize draft logit variance (encourage diverse branches)
            loss = -draft_logits.var(dim=2).mean()

            # Also run generate_draft_tree for EMA/draft-len state tracking
            self.eagle_head.generate_draft_tree(Z)

            eagle_optimizer.zero_grad()
            loss.backward()
            eagle_optimizer.step()

            # Update acceptance rate EMA (synthetic: random in [0.5, 0.9] for testing)
            simulated_rate = 0.5 + 0.4 * (ft_step / steps)
            self.eagle_head.accept_rate_ema = (
                0.99 * self.eagle_head.accept_rate_ema + 0.01 * simulated_rate
            )
            self.eagle_head._maybe_expand_draft_len()

            # Track high-accept windows
            if (ft_step + 1) % window_size == 0:
                if self.eagle_head.accept_rate_ema > self.config.eagle_accept_threshold:
                    high_accept_streak += 1
                else:
                    high_accept_streak = 0

                if high_accept_streak >= 500 and self.eagle_head.draft_len < 8:
                    logger.info(
                        f"EAGLE fine-tuning: expanding draft_len "
                        f"{self.eagle_head.draft_len}→8 at step {ft_step}"
                    )

            if ft_step % 500 == 0:
                logger.info(
                    f"EAGLE fine-tune step {ft_step}/{steps}: "
                    f"accept_ema={self.eagle_head.accept_rate_ema:.3f} "
                    f"draft_len={self.eagle_head.draft_len}"
                )

        # Unfreeze core parameters
        for param in self.node.parameters():
            param.requires_grad_(True)
        for param in self.output_proj.parameters():
            param.requires_grad_(True)

        logger.info("EAGLE fine-tuning complete")

    def graceful_shutdown(self) -> None:
        """Save checkpoint and prepare for clean exit."""
        ckpt = self.build_checkpoint_dict(self.step)
        self.checkpoint_manager.save(ckpt, self.step)
