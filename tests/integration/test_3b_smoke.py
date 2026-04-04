"""Smoke test for 3B preset on CPU (T043)."""
from __future__ import annotations
import pytest
import torch

@pytest.mark.slow
class TestSmoke3BPreset:
    def test_3b_config_instantiates(self) -> None:
        from pcg_llm.config import TrainingConfig
        cfg = TrainingConfig.from_preset("3b")
        assert cfg.hidden_dim == 3072
        assert cfg.checkpoint_backend == "gcs"

    def test_trainer_instantiates_with_3b_config(self, temp_checkpoint_dir) -> None:
        """PCGTrainer with 3B config (override checkpoint to local for test)."""
        import dataclasses
        from pcg_llm.config import TrainingConfig
        from pcg_llm.training.trainer import PCGTrainer

        # Override to local checkpoint for CPU test
        cfg = dataclasses.replace(
            TrainingConfig.from_preset("3b"),
            checkpoint_backend="local",
            checkpoint_dir=str(temp_checkpoint_dir),
            optimizer="adamw",
            optimizer_bits=32,
            normalize="standard",
            projection="dense",
        )
        trainer = PCGTrainer(config=cfg)
        assert trainer is not None
        assert trainer.config.hidden_dim == 3072
