"""Failing tests for TrainingConfig — TDD Phase 2 (T005).

All tests here should fail until TrainingConfig is implemented (T006).
"""

from __future__ import annotations

import pytest


class TestTrainingConfigValidation:
    """Tests for TrainingConfig validation rules per contracts/config.md."""

    def test_import_training_config(self) -> None:
        from pcg_llm.config import TrainingConfig  # noqa: F401

    def test_tiny_preset_instantiates(self) -> None:
        from pcg_llm.config import TrainingConfig

        cfg = TrainingConfig.from_preset("tiny")
        assert cfg.hidden_dim == 512
        assert cfg.max_seq_len == 2048
        assert cfg.block_size == 32
        assert cfg.initial_sparsity == 0.85  # SC-004: hard lower bound is 80%; start above it
        assert cfg.checkpoint_backend == "local"

    def test_3b_preset_instantiates(self) -> None:
        from pcg_llm.config import TrainingConfig

        cfg = TrainingConfig.from_preset("3b")
        assert cfg.hidden_dim == 3072
        assert cfg.max_seq_len == 4096
        assert cfg.block_size == 64
        assert cfg.initial_sparsity == 0.90
        assert cfg.checkpoint_backend == "gcs"

    def test_unknown_preset_raises(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="Unknown preset"):
            TrainingConfig.from_preset("unknown")

    def test_mixing_fractions_sum_to_one(self) -> None:
        from pcg_llm.config import TrainingConfig

        cfg = TrainingConfig.from_preset("tiny")
        for fw, sv in zip(cfg.dataset_fineweb_frac, cfg.dataset_stack_frac, strict=False):
            assert abs(fw + sv - 1.0) < 1e-9, f"Fractions must sum to 1.0, got {fw + sv}"

    def test_mixing_fractions_invalid_raises(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="[Mm]ixing|[Ff]rac|sum"):
            TrainingConfig(
                hidden_dim=64,
                max_seq_len=128,
                block_size=32,
                dataset_fineweb_frac=[0.80, 0.80, 0.70],
                dataset_stack_frac=[0.08, 0.15, 0.30],
            )

    def test_block_size_must_divide_max_seq_len(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="[Bb]lock|[Dd]ivide|[Ss]eq"):
            TrainingConfig(
                hidden_dim=64,
                max_seq_len=100,  # not divisible by block_size=32
                block_size=32,
            )

    def test_optimizer_bits_must_be_32_or_8(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="[Oo]ptimizer.*[Bb]it|[Bb]it.*[Oo]ptimizer"):
            TrainingConfig(hidden_dim=64, max_seq_len=128, block_size=32, optimizer_bits=16)

    def test_gcs_backend_requires_gs_prefix(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="gs://"):
            TrainingConfig(
                hidden_dim=64,
                max_seq_len=128,
                block_size=32,
                checkpoint_backend="gcs",
                checkpoint_dir="/local/path",  # must start with gs://
            )

    def test_monarch_min_dim_guard(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="[Mm]onarch|[Dd]im|64"):
            TrainingConfig(
                hidden_dim=32,  # < 64 minimum for Monarch
                max_seq_len=128,
                block_size=32,
                projection="monarch",
            )

    def test_monarch_min_dim_64_is_valid(self) -> None:
        from pcg_llm.config import TrainingConfig

        cfg = TrainingConfig(
            hidden_dim=64,
            max_seq_len=128,
            block_size=32,
            projection="monarch",
        )
        assert cfg.projection == "monarch"

    def test_invalid_normalize_value_raises(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="[Nn]ormalize"):
            TrainingConfig(hidden_dim=64, max_seq_len=128, block_size=32, normalize="invalid")

    def test_invalid_projection_value_raises(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="[Pp]rojection"):
            TrainingConfig(hidden_dim=64, max_seq_len=128, block_size=32, projection="invalid")

    def test_sparsity_must_be_strictly_between_0_and_1(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="[Ss]parsity"):
            TrainingConfig(hidden_dim=64, max_seq_len=128, block_size=32, initial_sparsity=1.0)

        with pytest.raises(ValueError, match="[Ss]parsity"):
            TrainingConfig(hidden_dim=64, max_seq_len=128, block_size=32, initial_sparsity=0.0)

    def test_anderson_window_bounds(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="[Aa]nderson|[Ww]indow"):
            TrainingConfig(hidden_dim=64, max_seq_len=128, block_size=32, anderson_window=0)

        with pytest.raises(ValueError, match="[Aa]nderson|[Ww]indow"):
            TrainingConfig(hidden_dim=64, max_seq_len=128, block_size=32, anderson_window=11)

    def test_eagle_draft_len_bounds(self) -> None:
        from pcg_llm.config import TrainingConfig

        with pytest.raises(ValueError, match="[Ee]agle|[Dd]raft"):
            TrainingConfig(hidden_dim=64, max_seq_len=128, block_size=32, eagle_draft_len=0)

        with pytest.raises(ValueError, match="[Ee]agle|[Dd]raft"):
            TrainingConfig(hidden_dim=64, max_seq_len=128, block_size=32, eagle_draft_len=17)

    def test_config_is_frozen(self) -> None:
        from pcg_llm.config import TrainingConfig

        cfg = TrainingConfig.from_preset("tiny")
        with pytest.raises((AttributeError, TypeError)):
            cfg.hidden_dim = 999  # type: ignore[misc]

    def test_config_to_dict_round_trip(self) -> None:
        from pcg_llm.config import TrainingConfig

        cfg = TrainingConfig.from_preset("tiny")
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["hidden_dim"] == 512
        # from_dict not tested here, but to_dict must exist
