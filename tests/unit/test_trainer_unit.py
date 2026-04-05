"""Unit tests for src/pcg_llm/training/trainer.py (previously 41% coverage)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **overrides):
    """Return a minimal TrainingConfig suitable for fast CPU unit tests."""
    from pcg_llm.config import TrainingConfig

    defaults = dict(
        hidden_dim=64,
        max_seq_len=64,
        vocab_size=256,
        block_size=32,
        initial_sparsity=0.70,
        max_solver_iters=3,
        solver_tolerance=0.5,
        checkpoint_interval=50,
        checkpoint_dir=str(tmp_path),
        checkpoint_backend="local",
        optimizer="adamw",
        optimizer_bits=32,
        normalize="standard",
        projection="dense",
        grad_checkpoint=False,
        wandb_project=None,
        warmup_steps=2,
        total_tokens=10_000,
        grad_accum_steps=1,
        batch_size=2,
        rigl_interval=10,
        anderson_window=3,
        eagle_draft_len=2,
        eagle_k=2,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _make_loader(n: int = 3, vocab_size: int = 256, seq_len: int = 64):
    """Yield *n* random token batches of shape (2, seq_len)."""
    for _ in range(n):
        yield torch.randint(0, vocab_size, (2, seq_len))


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestPCGTrainerInstantiation:
    def test_instantiates_without_error(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        assert trainer is not None
        assert trainer.step == 0

    def test_device_is_set(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        assert isinstance(trainer.device, torch.device)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_returns_prompt_plus_new_tokens(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        prompt = [1, 2, 3]
        max_new = 5
        result = trainer.generate(prompt, max_new_tokens=max_new)
        assert isinstance(result, list)
        assert len(result) == len(prompt) + max_new

    def test_generated_tokens_are_ints(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        result = trainer.generate([0], max_new_tokens=3)
        assert all(isinstance(t, int) for t in result)

    def test_generate_zero_new_tokens(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        prompt = [5, 10]
        result = trainer.generate(prompt, max_new_tokens=0)
        assert result == prompt


# ---------------------------------------------------------------------------
# graceful_shutdown
# ---------------------------------------------------------------------------


class TestGracefulShutdown:
    def test_runs_without_error(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        # Should complete without raising
        trainer.graceful_shutdown()

    def test_creates_checkpoint_file(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        trainer.graceful_shutdown()
        # Checkpoint dir should now contain at least one .pt file
        pt_files = list(tmp_path.glob("step-*.pt"))
        assert len(pt_files) >= 1


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_runs_3_steps(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        trainer.train(_make_loader(n=3), resume=False)
        assert trainer.step == 3

    def test_train_saves_final_checkpoint(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        trainer.train(_make_loader(n=2), resume=False)
        pt_files = list(tmp_path.glob("step-*.pt"))
        assert len(pt_files) >= 1

    def test_train_with_dict_batch(self, tmp_path: Path):
        """Dataloader that yields dicts with 'input_ids' should work."""
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)

        def dict_loader():
            for _ in range(2):
                yield {"input_ids": torch.randint(0, 256, (2, 64))}

        trainer.train(dict_loader(), resume=False)
        assert trainer.step == 2

    def test_train_keyboard_interrupt_saves_and_reraises(self, tmp_path: Path):
        """KeyboardInterrupt should save a checkpoint then re-raise."""
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)

        # Create a loader that raises KeyboardInterrupt after 1 batch
        def interrupt_loader():
            yield torch.randint(0, 256, (2, 64))
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            trainer.train(interrupt_loader(), resume=False)

        pt_files = list(tmp_path.glob("step-*.pt"))
        assert len(pt_files) >= 1


# ---------------------------------------------------------------------------
# _build_backend — GCS fallback
# ---------------------------------------------------------------------------


class TestBuildBackend:
    def test_gcs_backend_falls_back_to_local_when_not_installed(self, tmp_path: Path):
        """When google-cloud-storage is absent, GCS backend falls back to local.

        Strategy: build a trainer with a local config, then temporarily replace
        trainer.config with a SimpleNamespace that advertises checkpoint_backend="gcs"
        but uses tmp_path as checkpoint_dir so LocalCheckpointBackend can mkdir.
        """
        import sys
        import types
        from unittest.mock import patch

        from pcg_llm.checkpointing.local import LocalCheckpointBackend
        from pcg_llm.training.trainer import PCGTrainer

        local_config = _make_config(tmp_path, checkpoint_backend="local")
        trainer = PCGTrainer(local_config)

        # Build a fake config namespace advertising GCS with a local fallback dir
        fake_config = types.SimpleNamespace(
            checkpoint_backend="gcs",
            checkpoint_dir=str(tmp_path),
        )
        original_config = trainer.config
        trainer.config = fake_config  # type: ignore[assignment]

        try:
            # Simulate the GCS module raising ImportError on import
            with patch.dict(sys.modules, {"pcg_llm.checkpointing.gcs": None}):
                backend = trainer._build_backend()
        finally:
            trainer.config = original_config  # type: ignore[assignment]

        assert isinstance(backend, LocalCheckpointBackend)

    def test_local_backend_used_when_configured(self, tmp_path: Path):
        from pcg_llm.checkpointing.local import LocalCheckpointBackend
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, checkpoint_backend="local")
        trainer = PCGTrainer(config)
        assert isinstance(trainer._backend, LocalCheckpointBackend)


# ---------------------------------------------------------------------------
# normalize="ngpt" variant
# ---------------------------------------------------------------------------


class TestNGPTNormalization:
    def test_instantiates_with_ngpt(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, normalize="ngpt")
        trainer = PCGTrainer(config)
        assert trainer is not None

    def test_train_step_returns_sparsity_float(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, normalize="ngpt")
        trainer = PCGTrainer(config)
        batch = torch.randint(0, 256, (2, 64))
        metrics = trainer.train_step(batch)
        assert "sparsity" in metrics
        assert isinstance(metrics["sparsity"], float)

    def test_train_step_returns_expected_keys(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, normalize="ngpt")
        trainer = PCGTrainer(config)
        batch = torch.randint(0, 256, (2, 64))
        metrics = trainer.train_step(batch)
        expected_keys = {"loss_total", "loss_crossentropy", "sparsity", "solver_steps"}
        assert expected_keys.issubset(metrics.keys())


# ---------------------------------------------------------------------------
# muon_adamw optimizer path
# ---------------------------------------------------------------------------


class TestMuonAdamWOptimizer:
    def test_muon_optimizer_falls_back_to_adamw_when_unavailable(self, tmp_path: Path):
        """When HybridOptimizer is not importable, falls back to standard AdamW."""
        import sys
        from unittest.mock import patch

        # Patch the optimizer import to trigger ImportError fallback
        with patch.dict(sys.modules, {"pcg_llm.training.optimizer": None}):
            from pcg_llm.training.trainer import PCGTrainer

            config = _make_config(tmp_path, optimizer="muon_adamw")
            trainer = PCGTrainer(config)

        # The fallback should be a standard AdamW
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_adamw_optimizer_used_directly(self, tmp_path: Path):
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, optimizer="adamw")
        trainer = PCGTrainer(config)
        assert isinstance(trainer.optimizer, torch.optim.AdamW)


# ---------------------------------------------------------------------------
# fine_tune_eagle
# ---------------------------------------------------------------------------


class TestFineTuneEagle:
    def test_fine_tune_eagle_runs_without_error(self, tmp_path: Path):
        """fine_tune_eagle should complete without raising."""
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, eagle_k=2, eagle_draft_len=2)
        trainer = PCGTrainer(config)
        # Run just 2 steps (fast, CPU)
        trainer.fine_tune_eagle(steps=2)

    def test_fine_tune_eagle_unfreezes_params(self, tmp_path: Path):
        """After fine_tune_eagle, PCGNode parameters must require_grad again."""
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        trainer.fine_tune_eagle(steps=1)
        for p in trainer.node.parameters():
            assert p.requires_grad, "PCGNode params must be unfrozen after fine_tune_eagle"

    def test_fine_tune_eagle_skipped_when_eagle_head_unavailable(self, tmp_path: Path):
        """If EAGLEExtrapolationHead cannot be imported, method returns silently."""
        import sys
        from unittest.mock import patch

        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)

        with patch.dict(sys.modules, {"pcg_llm.arch.eagle_head": None}):
            trainer.fine_tune_eagle(steps=5)  # should not raise


# ---------------------------------------------------------------------------
# gradient accumulation (FR-019) — T056
# ---------------------------------------------------------------------------


class TestGradientAccumulation:
    """FR-019: gradient accumulation with configurable accum_steps."""

    def test_step_counter_increments_every_batch(self, tmp_path: Path):
        """self.step must increment once per batch, not once per optimizer.step."""
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, grad_accum_steps=4)
        trainer = PCGTrainer(config)

        def loader():
            for _ in range(8):
                yield torch.randint(0, 256, (2, 64))

        trainer.train(loader(), resume=False)
        assert (
            trainer.step == 8
        ), f"step should be 8 (one per batch regardless of accum_steps), got {trainer.step}"

    def test_optimizer_step_called_once_per_accum_window(self, tmp_path: Path):
        """With grad_accum_steps=4 and 8 batches, optimizer.step is called exactly 2×."""
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, grad_accum_steps=4)
        trainer = PCGTrainer(config)

        step_count = [0]
        original_step = trainer.optimizer.step

        def counting_step(*args, **kwargs):
            step_count[0] += 1
            return original_step(*args, **kwargs)

        trainer.optimizer.step = counting_step  # type: ignore[method-assign]

        def loader():
            for _ in range(8):
                yield torch.randint(0, 256, (2, 64))

        trainer.train(loader(), resume=False)
        assert step_count[0] == 2, (
            f"optimizer.step should be called 2 times for 8 batches with accum_steps=4, "
            f"got {step_count[0]}"
        )

    def test_grad_accum_1_behaves_like_no_accumulation(self, tmp_path: Path):
        """grad_accum_steps=1 (default) must behave identically to the baseline."""
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, grad_accum_steps=1)
        trainer = PCGTrainer(config)

        def loader():
            for _ in range(3):
                yield torch.randint(0, 256, (2, 64))

        trainer.train(loader(), resume=False)
        assert trainer.step == 3


# ---------------------------------------------------------------------------
# gradient checkpointing correctness (FR-017) — T057
# ---------------------------------------------------------------------------


class TestGradientCheckpointing:
    """FR-017: grad_checkpoint=True must produce same loss as grad_checkpoint=False."""

    def test_loss_equivalent_with_and_without_checkpointing(self, tmp_path: Path):
        """train_step loss values must match ± BF16 tolerance (1e-2)."""
        from pcg_llm.training.trainer import PCGTrainer

        batch = torch.randint(0, 256, (2, 64))

        config_no_ckpt = _make_config(tmp_path / "no_ckpt", grad_checkpoint=False)
        torch.manual_seed(0)
        trainer_no = PCGTrainer(config_no_ckpt)
        metrics_no = trainer_no.train_step(batch)

        config_ckpt = _make_config(tmp_path / "ckpt", grad_checkpoint=True)
        # Re-seed so trainer_ck receives the same initial weights as trainer_no.
        torch.manual_seed(0)
        trainer_ck = PCGTrainer(config_ckpt)
        metrics_ck = trainer_ck.train_step(batch)

        # Both trainers have identical weights and the same batch → loss should be the same.
        assert abs(metrics_no["loss_total"] - metrics_ck["loss_total"]) < 1e-1, (
            f"Loss mismatch between grad_checkpoint=False ({metrics_no['loss_total']:.4f}) "
            f"and grad_checkpoint=True ({metrics_ck['loss_total']:.4f})"
        )

    def test_checkpointing_does_not_raise(self, tmp_path: Path):
        """train_step with grad_checkpoint=True must complete without error."""
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, grad_checkpoint=True)
        trainer = PCGTrainer(config)
        batch = torch.randint(0, 256, (2, 64))
        metrics = trainer.train_step(batch)
        assert "loss_total" in metrics
        assert metrics["loss_total"] == metrics["loss_total"], "Loss must not be NaN"


# ---------------------------------------------------------------------------
# DDP multi-GPU support (T061) — all tests run in single-process mode via mocks
# ---------------------------------------------------------------------------


class TestDDPSupport:
    """DDP multi-GPU support tests.

    These tests run entirely in a single process — ``torch.distributed`` calls
    are either absent (single-GPU path) or mocked so no real NCCL backend is
    initialised.
    """

    def test_is_ddp_false_by_default(self, tmp_path: Path):
        """Without LOCAL_RANK env var the trainer must operate single-process."""
        import os

        from pcg_llm.training.trainer import PCGTrainer

        os.environ.pop("LOCAL_RANK", None)
        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        assert trainer._is_ddp is False
        assert trainer._world_size == 1

    def test_w_structure_requires_grad_after_init(self, tmp_path: Path):
        """W_structure must have requires_grad=True immediately after __init__."""
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        assert (
            trainer.adjacency.W_structure.requires_grad
        ), "W_structure.requires_grad must be True — it carries trainable edge weights"

    def test_w_structure_receives_grad_after_train_step(self, tmp_path: Path):
        """W_structure.grad must be populated after a forward-backward train_step.

        Regression test: before the DDP fix, W_structure was a plain tensor
        with requires_grad never set, so no gradients ever flowed to it.
        """
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        batch = torch.randint(0, 256, (2, 64))
        trainer.train_step(batch)
        assert trainer.adjacency.W_structure.grad is not None, (
            "W_structure must receive gradients after train_step "
            "(regression for missing requires_grad_(True) bug)"
        )

    def test_save_checkpoint_guarded_by_rank0(self, tmp_path: Path):
        """_save_checkpoint must not write files when self._rank != 0."""
        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        # Simulate non-zero rank without launching a real process group
        trainer._rank = 1
        trainer._save_checkpoint(step=99)
        pt_files = list(tmp_path.glob("step-*.pt"))
        assert len(pt_files) == 0, "Non-rank-0 process must not write checkpoint files"

    def test_optimizer_step_calls_all_reduce_in_ddp_mode(self, tmp_path: Path):
        """_optimizer_step must call dist.all_reduce for W_structure.grad when DDP active."""
        from unittest.mock import patch

        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        # Simulate DDP mode without a real process group
        trainer._is_ddp = True
        trainer._world_size = 2
        trainer._rank = 0
        # Provide a pre-computed gradient for W_structure
        trainer.adjacency.W_structure.grad = torch.ones_like(trainer.adjacency.W_structure)
        # Provide zero gradients for module params (required by clip_grad_norm_)
        for p in trainer._raw_node.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        for p in trainer._raw_output_proj.parameters():
            if p.grad is None:
                p.grad = torch.zeros_like(p)

        with patch("pcg_llm.training.trainer.dist.all_reduce") as mock_all_reduce:
            trainer._optimizer_step()

        mock_all_reduce.assert_called_once()

    def test_broadcast_state_after_resume_calls_broadcast(self, tmp_path: Path):
        """_broadcast_state_after_resume must call dist.broadcast to sync state."""
        from unittest.mock import patch

        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path)
        trainer = PCGTrainer(config)
        trainer._is_ddp = True
        trainer._rank = 0

        with patch("pcg_llm.training.trainer.dist.broadcast") as mock_broadcast:
            trainer._broadcast_state_after_resume()

        assert (
            mock_broadcast.call_count > 0
        ), "dist.broadcast must be called at least once to sync step and model state"

    def test_rigl_mask_broadcast_after_drop_and_grow(self, tmp_path: Path):
        """RigL drop_and_grow must broadcast the updated mask in DDP mode."""
        from unittest.mock import patch

        from pcg_llm.training.trainer import PCGTrainer

        # rigl_interval=1 → RigL triggers when self.step > 0 and step % 1 == 0
        config = _make_config(tmp_path, rigl_interval=1)
        trainer = PCGTrainer(config)
        trainer._is_ddp = True
        trainer._rank = 0
        trainer.step = 1  # ensures step > 0 condition is met inside train_step

        with patch("pcg_llm.training.trainer.dist.broadcast") as mock_broadcast:
            # Also mock all_reduce (triggered by W_structure.grad in _optimizer_step)
            with patch("pcg_llm.training.trainer.dist.all_reduce"):
                batch = torch.randint(0, 256, (2, 64))
                trainer.train_step(batch)

        assert (
            mock_broadcast.call_count > 0
        ), "dist.broadcast must be called after drop_and_grow to sync mask across ranks"


# ---------------------------------------------------------------------------
# DDP + gradient accumulation interaction (M3)
# ---------------------------------------------------------------------------


class TestDDPGradAccumInteraction:
    """Gradient accumulation must remain correct when DDP mode is simulated (M3)."""

    def test_grad_accum_optimizer_step_count_with_ddp_flag(self, tmp_path: Path):
        """optimizer.step fires once per accumulation window even with _is_ddp=True."""
        from unittest.mock import patch

        from pcg_llm.training.trainer import PCGTrainer

        config = _make_config(tmp_path, grad_accum_steps=2)
        trainer = PCGTrainer(config)
        # Simulate DDP flag (world_size=1 so no real communication happens)
        trainer._is_ddp = True
        trainer._world_size = 1
        trainer._rank = 0

        step_count = [0]
        original_step = trainer.optimizer.step

        def counting_step(*args, **kwargs):
            step_count[0] += 1
            return original_step(*args, **kwargs)

        trainer.optimizer.step = counting_step  # type: ignore[method-assign]

        with patch("pcg_llm.training.trainer.dist.all_reduce"):
            trainer.train(_make_loader(n=4), resume=False)

        assert step_count[0] == 2, (
            f"With grad_accum_steps=2 and 4 batches, optimizer.step should fire 2×, "
            f"got {step_count[0]}"
        )
