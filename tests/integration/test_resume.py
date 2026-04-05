"""Integration test: interrupt-and-resume training.

@pytest.mark.slow
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.slow
class TestTrainingResume:
    """Verify checkpoint-based resume restores state correctly."""

    def _make_config(self, checkpoint_dir: str):
        from pcg_llm.config import TrainingConfig

        return TrainingConfig(
            hidden_dim=64,
            max_seq_len=64,
            vocab_size=256,
            block_size=32,
            initial_sparsity=0.85,
            max_solver_iters=5,
            solver_tolerance=1e-1,
            checkpoint_interval=100,
            checkpoint_dir=checkpoint_dir,
            checkpoint_backend="local",
            optimizer="adamw",
            optimizer_bits=32,
            normalize="standard",
            projection="dense",
            grad_checkpoint=False,
            wandb_project=None,
            rigl_interval=50,
            warmup_steps=2,
            total_tokens=100_000,
        )

    def _make_loader(self, n, seed=42):
        torch.manual_seed(seed)
        for _ in range(n):
            yield torch.randint(0, 256, (2, 64))

    def test_resume_continues_from_correct_step(self, temp_checkpoint_dir):
        from pcg_llm.training.trainer import PCGTrainer

        config = self._make_config(str(temp_checkpoint_dir))

        # Phase 1: Train 100 steps
        trainer1 = PCGTrainer(config=config)
        for i, batch in enumerate(self._make_loader(100)):
            trainer1.train_step(batch)
            trainer1.step += 1

        # Save checkpoint at step 100
        ckpt = trainer1.build_checkpoint_dict(100)
        trainer1.checkpoint_manager.save(ckpt, 100)

        # Phase 2: Resume and verify step continues from 100
        trainer2 = PCGTrainer(config=config)
        trainer2.resume_if_available()

        assert trainer2.step == 100, f"Resume should restore to step 100, got {trainer2.step}"

    def test_no_tmp_orphan_files_after_interrupt(self, temp_checkpoint_dir):
        from pcg_llm.training.trainer import PCGTrainer

        config = self._make_config(str(temp_checkpoint_dir))
        trainer = PCGTrainer(config=config)

        # Simulate training with KeyboardInterrupt
        try:
            for i, batch in enumerate(self._make_loader(50)):
                trainer.train_step(batch)
                trainer.step += 1
                if trainer.step == 40:
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            ckpt = trainer.build_checkpoint_dict(trainer.step)
            trainer.checkpoint_manager.save(ckpt, trainer.step)

        # Check no .tmp files remain
        tmp_files = list(temp_checkpoint_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Orphaned .tmp files: {tmp_files}"

    def test_resume_loss_continuity(self, temp_checkpoint_dir):
        """SC-002: after resume, loss trajectory must differ by < 0.5% within 10 steps."""
        from pcg_llm.training.trainer import PCGTrainer

        config = self._make_config(str(temp_checkpoint_dir))

        # Pre-generate all batches so both runs see EXACTLY the same sequences.
        # (Re-seeding a generator mid-run gives different batches than continuing
        # a seeded generator from step 51 onward, which would cause spurious
        # divergence once training is actually working.)
        all_train_batches = list(self._make_loader(60, seed=42))
        eval_batches = list(self._make_loader(10, seed=99))

        # Uninterrupted reference run: train 60 steps, record last 10 losses.
        # Fix the model-initialisation seed so trainer_ref and trainer1 start
        # from identical weights.
        torch.manual_seed(0)
        trainer_ref = PCGTrainer(config=config)
        for batch in all_train_batches:
            trainer_ref.train_step(batch)
            trainer_ref.step += 1
        ref_losses = []
        for batch in eval_batches:
            m = trainer_ref.train_step(batch)
            trainer_ref.step += 1
            ref_losses.append(m["loss_total"])

        # Interrupted run: train 50 steps, save, resume, train steps 51-60
        # using the SAME batches trainer_ref saw at those steps.
        torch.manual_seed(0)
        trainer1 = PCGTrainer(config=config)
        for batch in all_train_batches[:50]:
            trainer1.train_step(batch)
            trainer1.step += 1
        ckpt = trainer1.build_checkpoint_dict(50)
        trainer1.checkpoint_manager.save(ckpt, 50)

        trainer2 = PCGTrainer(config=config)
        trainer2.resume_if_available()
        assert trainer2.step == 50, "Step should be 50 after resume"

        # Catch up from step 50→60 with the exact same batches as trainer_ref.
        for batch in all_train_batches[50:]:
            trainer2.train_step(batch)
            trainer2.step += 1

        resumed_losses = []
        for batch in eval_batches:
            m = trainer2.train_step(batch)
            trainer2.step += 1
            resumed_losses.append(m["loss_total"])

        # SC-002: loss difference must be < 0.5% at each post-resume step.
        assert all(
            l == l and l != float("inf") for l in resumed_losses
        ), "Loss must be finite after resume"
        for step_i, (ref, res) in enumerate(zip(ref_losses, resumed_losses, strict=False)):
            if ref > 0:
                rel_diff = abs(ref - res) / ref
                assert rel_diff < 0.005, (
                    f"SC-002: loss diff at post-resume step {step_i}: "
                    f"ref={ref:.4f} resumed={res:.4f} diff={rel_diff:.4%} (limit 0.5%)"
                )
