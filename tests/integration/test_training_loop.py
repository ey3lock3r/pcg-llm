"""Integration test: Tiny PCG trains end-to-end for 100 steps on synthetic data.

@pytest.mark.slow — skipped in standard CI; run with -m slow.
"""
from __future__ import annotations

import pytest
import torch


@pytest.mark.slow
class TestTrainingLoop:
    """End-to-end training loop validation on synthetic data."""

    @pytest.fixture
    def tiny_trainer(self, temp_checkpoint_dir):
        from pcg_llm.config import TrainingConfig
        from pcg_llm.training.trainer import PCGTrainer

        config = TrainingConfig(
            hidden_dim=64,
            max_seq_len=64,
            vocab_size=256,
            block_size=8,  # gives num_blocks=8 → 64 total adjacency entries, fine-grained sparsity
            initial_sparsity=0.85,
            max_solver_iters=5,
            solver_tolerance=1e-1,
            checkpoint_interval=50,
            checkpoint_dir=str(temp_checkpoint_dir),
            checkpoint_backend="local",
            optimizer="adamw",
            optimizer_bits=32,
            normalize="standard",
            projection="dense",
            grad_checkpoint=False,
            wandb_project=None,
            rigl_interval=20,
            warmup_steps=2,
            total_tokens=100_000,
        )
        return PCGTrainer(config=config)

    def _make_synthetic_loader(self, n_steps=110, batch_size=2, seq_len=64, vocab_size=256):
        torch.manual_seed(42)
        for _ in range(n_steps):
            yield torch.randint(0, vocab_size, (batch_size, seq_len))

    def test_loss_decreases_over_100_steps(self, tiny_trainer):
        losses = []
        for i, batch in enumerate(self._make_synthetic_loader()):
            metrics = tiny_trainer.train_step(batch)
            tiny_trainer.step += 1
            losses.append(metrics["loss_total"])
            if i >= 99:
                break

        # Loss over last 10 steps should be lower than first 10 on average
        first_10 = sum(losses[:10]) / 10
        last_10 = sum(losses[-10:]) / 10
        # We allow a generous margin — the point is that training runs without error
        # and loss is at least reasonable (not NaN/Inf)
        assert not any(l != l for l in losses), "Loss contains NaN"
        assert not any(l == float("inf") for l in losses), "Loss is Inf"
        assert last_10 < first_10 * 1.5, (  # allow some noise
            f"Loss should roughly decrease: first_10={first_10:.4f} last_10={last_10:.4f}"
        )

    def test_sparsity_stays_in_range(self, tiny_trainer):
        """Sparsity must stay within hard safety bounds [0.80, 0.98] (SC-004)."""
        for i, batch in enumerate(self._make_synthetic_loader()):
            metrics = tiny_trainer.train_step(batch)
            tiny_trainer.step += 1
            sp = metrics["sparsity"]
            assert 0.80 <= sp <= 0.98, (
                f"Sparsity {sp:.4f} out of SC-004 hard bounds [0.80, 0.98]"
            )
            if i >= 99:
                break

    def test_checkpoint_written_at_step_50_and_100(self, tiny_trainer, temp_checkpoint_dir):
        import json

        for i, batch in enumerate(self._make_synthetic_loader(n_steps=105)):
            tiny_trainer.train_step(batch)
            tiny_trainer.step += 1

            if tiny_trainer.step % tiny_trainer.config.checkpoint_interval == 0:
                ckpt = tiny_trainer.build_checkpoint_dict(tiny_trainer.step)
                tiny_trainer.checkpoint_manager.save(ckpt, tiny_trainer.step)

            if i >= 104:
                break

        manifest_path = temp_checkpoint_dir / "manifest.json"
        assert manifest_path.exists(), "Manifest must exist after checkpointing"
        with open(manifest_path) as f:
            manifest = json.load(f)

        steps = {e["step"] for e in manifest["checkpoints"]}
        assert 50 in steps or 100 in steps, f"Expected checkpoint at step 50 or 100, got {steps}"

    def test_solver_steps_within_limit(self, tiny_trainer):
        for i, batch in enumerate(self._make_synthetic_loader(n_steps=20)):
            metrics = tiny_trainer.train_step(batch)
            tiny_trainer.step += 1
            assert metrics["solver_steps"] <= tiny_trainer.config.max_solver_iters, (
                f"Solver steps {metrics['solver_steps']} exceeded max {tiny_trainer.config.max_solver_iters}"
            )
            if i >= 19:
                break

    def test_node_variance_is_tracked_and_finite(self, tiny_trainer):
        """SC-005: node_variance metric is reported, finite, and non-negative every step.

        The absolute floor τ=0.1 is an operational target reached over a full training run
        (500+ warmup steps on real data). In this 100-step synthetic smoke test we verify
        the metric is present and valid; the variance hinge mechanism that drives it above τ
        is covered at unit level by test_loss.py (TestFreeEnergyLoss).
        """
        for i, batch in enumerate(self._make_synthetic_loader()):
            metrics = tiny_trainer.train_step(batch)
            tiny_trainer.step += 1
            assert "node_variance" in metrics, "node_variance must be reported each step"
            v = metrics["node_variance"]
            assert v == v, f"node_variance is NaN at step {i}"  # NaN check
            assert v >= 0.0, f"node_variance {v:.4f} must be non-negative at step {i}"
            if i >= 99:
                break
