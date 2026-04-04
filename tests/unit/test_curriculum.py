"""Tests for DataCurriculum — covering validation, shard advancement,
mixing ratio plateau, state_dict, and load_state_dict."""

from __future__ import annotations

import pytest


class TestDataCurriculumConstruction:
    """Tests for constructor validation (lines 31, 33)."""

    def test_import(self) -> None:
        from pcg_llm.training.curriculum import DataCurriculum  # noqa: F401

    def test_empty_fineweb_fracs_raises_value_error(self) -> None:
        """Passing an empty fineweb_fracs list must raise ValueError."""
        from pcg_llm.training.curriculum import DataCurriculum

        with pytest.raises(ValueError, match="fineweb_fracs"):
            DataCurriculum(fineweb_fracs=[], stack_fracs=[0.5])

    def test_empty_stack_fracs_raises_value_error(self) -> None:
        """Passing an empty stack_fracs list must raise ValueError."""
        from pcg_llm.training.curriculum import DataCurriculum

        with pytest.raises(ValueError, match="stack_fracs"):
            DataCurriculum(fineweb_fracs=[0.5], stack_fracs=[])

    def test_valid_construction_succeeds(self) -> None:
        """A curriculum with non-empty lists should construct without error."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8, 0.6], stack_fracs=[0.2, 0.4])
        assert cur.epoch == 0
        assert cur.shard_index == 0

    def test_both_empty_raises_value_error(self) -> None:
        """Both lists empty should still raise (fineweb_fracs is checked first)."""
        from pcg_llm.training.curriculum import DataCurriculum

        with pytest.raises(ValueError):
            DataCurriculum(fineweb_fracs=[], stack_fracs=[])


class TestDataCurriculumAdvanceShard:
    """Tests for advance_shard (line 45)."""

    def test_advance_shard_increments_shard_index(self) -> None:
        """advance_shard() should increment shard_index by 1."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8], stack_fracs=[0.2])
        assert cur.shard_index == 0
        cur.advance_shard()
        assert cur.shard_index == 1

    def test_advance_shard_multiple_times(self) -> None:
        """Multiple calls to advance_shard should accumulate correctly."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8], stack_fracs=[0.2])
        for i in range(5):
            cur.advance_shard()
        assert cur.shard_index == 5

    def test_advance_shard_does_not_affect_epoch(self) -> None:
        """advance_shard should not change the epoch counter."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8], stack_fracs=[0.2])
        cur.advance_shard()
        assert cur.epoch == 0


class TestDataCurriculumMixingRatio:
    """Tests for get_mixing_ratio (lines 63-65)."""

    def test_get_mixing_ratio_epoch_0(self) -> None:
        """get_mixing_ratio(0) should return the first entry from each list."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8, 0.6], stack_fracs=[0.2, 0.4])
        fw, st = cur.get_mixing_ratio(0)
        assert fw == 0.8
        assert st == 0.2

    def test_get_mixing_ratio_epoch_1(self) -> None:
        """get_mixing_ratio(1) should return the second entry from each list."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8, 0.6], stack_fracs=[0.2, 0.4])
        fw, st = cur.get_mixing_ratio(1)
        assert fw == 0.6
        assert st == 0.4

    def test_get_mixing_ratio_returns_tuple(self) -> None:
        """get_mixing_ratio should return a 2-tuple."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.7], stack_fracs=[0.3])
        result = cur.get_mixing_ratio(0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_mixing_ratio_plateau_beyond_list_length(self) -> None:
        """Epoch beyond list length should return the last entry (plateau)."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8, 0.6], stack_fracs=[0.2, 0.4])
        # List has 2 entries (indices 0 and 1); epoch 5 should clamp to index 1
        fw, st = cur.get_mixing_ratio(5)
        assert fw == 0.6
        assert st == 0.4

    def test_get_mixing_ratio_plateau_single_entry(self) -> None:
        """Single-entry lists should always return that entry regardless of epoch."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.9], stack_fracs=[0.1])
        for epoch in [0, 1, 10, 100]:
            fw, st = cur.get_mixing_ratio(epoch)
            assert fw == 0.9, f"Expected 0.9 at epoch {epoch}, got {fw}"
            assert st == 0.1, f"Expected 0.1 at epoch {epoch}, got {st}"


class TestDataCurriculumStateDict:
    """Tests for state_dict() and load_state_dict() (lines 77, 88-89)."""

    def test_state_dict_contains_epoch_and_shard_index(self) -> None:
        """state_dict() must return a dict with 'epoch' and 'shard_index' keys."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8], stack_fracs=[0.2])
        sd = cur.state_dict()
        assert isinstance(sd, dict)
        assert "epoch" in sd
        assert "shard_index" in sd

    def test_state_dict_initial_values(self) -> None:
        """Initial state_dict should have epoch=0 and shard_index=0."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8], stack_fracs=[0.2])
        sd = cur.state_dict()
        assert sd["epoch"] == 0
        assert sd["shard_index"] == 0

    def test_state_dict_reflects_mutations(self) -> None:
        """state_dict should reflect mutations to epoch and shard_index."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8], stack_fracs=[0.2])
        cur.epoch = 3
        cur.advance_shard()
        cur.advance_shard()
        sd = cur.state_dict()
        assert sd["epoch"] == 3
        assert sd["shard_index"] == 2

    def test_load_state_dict_restores_epoch(self) -> None:
        """load_state_dict should restore the epoch counter."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8], stack_fracs=[0.2])
        cur.load_state_dict({"epoch": 7, "shard_index": 0})
        assert cur.epoch == 7

    def test_load_state_dict_restores_shard_index(self) -> None:
        """load_state_dict should restore the shard_index counter."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8], stack_fracs=[0.2])
        cur.load_state_dict({"epoch": 0, "shard_index": 12})
        assert cur.shard_index == 12

    def test_round_trip_state_dict(self) -> None:
        """Saving and loading state_dict should fully round-trip."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur_a = DataCurriculum(fineweb_fracs=[0.8, 0.6], stack_fracs=[0.2, 0.4])
        cur_a.epoch = 5
        for _ in range(3):
            cur_a.advance_shard()

        cur_b = DataCurriculum(fineweb_fracs=[0.8, 0.6], stack_fracs=[0.2, 0.4])
        cur_b.load_state_dict(cur_a.state_dict())

        assert cur_b.epoch == cur_a.epoch
        assert cur_b.shard_index == cur_a.shard_index

    def test_load_state_dict_coerces_to_int(self) -> None:
        """load_state_dict should coerce string/float values to int."""
        from pcg_llm.training.curriculum import DataCurriculum

        cur = DataCurriculum(fineweb_fracs=[0.8], stack_fracs=[0.2])
        cur.load_state_dict({"epoch": "4", "shard_index": 2.0})
        assert cur.epoch == 4
        assert cur.shard_index == 2
