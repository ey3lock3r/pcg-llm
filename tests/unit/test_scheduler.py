"""Tests for RigLSparsitySchedule — covering all lines including freeze path,
state_dict, and load_state_dict."""

from __future__ import annotations

import math

import pytest


class TestRigLSparsityScheduleBasic:
    """Tests for normal (pre-freeze) step behaviour."""

    def test_import(self) -> None:
        from pcg_llm.training.scheduler import RigLSparsitySchedule  # noqa: F401

    def test_step_before_freeze_returns_fraction_and_false(self) -> None:
        """Early steps should return a positive fraction and frozen=False."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched = RigLSparsitySchedule(
            reroute_start=0.30, total_steps=1000, freeze_step_frac=0.80
        )
        fraction, frozen = sched.step()
        assert frozen is False
        assert fraction > 0.0
        assert fraction <= 0.30

    def test_first_step_fraction_close_to_reroute_start(self) -> None:
        """At step 1 the cosine fraction should be close to reroute_start."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched = RigLSparsitySchedule(
            reroute_start=0.30, total_steps=1000, freeze_step_frac=0.80
        )
        fraction, frozen = sched.step()
        freeze_step = int(0.80 * 1000)  # 800
        expected = 0.30 * 0.5 * (1.0 + math.cos(math.pi * 1 / freeze_step))
        assert frozen is False
        assert abs(fraction - expected) < 1e-9

    def test_step_counter_advances(self) -> None:
        """_current_step must increment with each call to step()."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched = RigLSparsitySchedule(total_steps=100)
        assert sched._current_step == 0
        sched.step()
        assert sched._current_step == 1
        sched.step()
        assert sched._current_step == 2

    def test_fraction_decreases_over_steps(self) -> None:
        """The re-routing fraction should decrease monotonically before freeze."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched = RigLSparsitySchedule(
            reroute_start=0.30, total_steps=100, freeze_step_frac=0.80
        )
        fractions = []
        for _ in range(10):
            f, frozen = sched.step()
            if frozen:
                break
            fractions.append(f)

        for i in range(len(fractions) - 1):
            assert fractions[i] >= fractions[i + 1], (
                f"Fraction should be non-increasing: {fractions[i]} < {fractions[i+1]}"
            )


class TestRigLSparsityScheduleFreeze:
    """Tests for the topology-freeze path (lines 55-56)."""

    def test_step_at_freeze_step_returns_zero_and_true(self) -> None:
        """Exactly at freeze_step the scheduler must return (0.0, True)."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        total_steps = 100
        freeze_step_frac = 0.50
        sched = RigLSparsitySchedule(
            reroute_start=0.30,
            total_steps=total_steps,
            freeze_step_frac=freeze_step_frac,
        )
        freeze_step = int(freeze_step_frac * total_steps)  # 50

        # Advance to the step just before freeze
        for _ in range(freeze_step - 1):
            f, frozen = sched.step()
            assert frozen is False, f"Should not be frozen before step {freeze_step}"

        # This step hits freeze_step
        f, frozen = sched.step()
        assert frozen is True
        assert f == 0.0

    def test_step_after_freeze_returns_zero_and_true(self) -> None:
        """All steps beyond freeze_step must return (0.0, True)."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched = RigLSparsitySchedule(
            reroute_start=0.30, total_steps=10, freeze_step_frac=0.50
        )
        # Advance well past the freeze point (freeze_step = 5)
        results = [sched.step() for _ in range(10)]
        post_freeze = [(f, frozen) for f, frozen in results if frozen]
        assert len(post_freeze) > 0, "Expected at least one frozen step"
        for f, frozen in post_freeze:
            assert f == 0.0
            assert frozen is True

    def test_freeze_with_frac_one_freezes_immediately(self) -> None:
        """freeze_step_frac=1.0 means freeze_step == total_steps; step 1 is before freeze."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched = RigLSparsitySchedule(
            reroute_start=0.30, total_steps=5, freeze_step_frac=1.0
        )
        # freeze_step = int(1.0 * 5) = 5; steps 1-4 are before freeze
        for _ in range(4):
            _, frozen = sched.step()
            assert frozen is False
        # step 5 hits freeze_step
        _, frozen = sched.step()
        assert frozen is True


class TestRigLSparsityScheduleStateDict:
    """Tests for state_dict() and load_state_dict() serialisation (lines 76, 90-93)."""

    def test_state_dict_contains_required_keys(self) -> None:
        """state_dict() must return a dict with all four expected keys."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched = RigLSparsitySchedule(
            reroute_start=0.25, total_steps=500, freeze_step_frac=0.75
        )
        sd = sched.state_dict()
        assert isinstance(sd, dict)
        for key in ("current_step", "reroute_start", "total_steps", "freeze_step_frac"):
            assert key in sd, f"Missing key '{key}' in state_dict"

    def test_state_dict_values_match_constructor(self) -> None:
        """state_dict values should reflect construction parameters."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched = RigLSparsitySchedule(
            reroute_start=0.20, total_steps=200, freeze_step_frac=0.60
        )
        sd = sched.state_dict()
        assert sd["current_step"] == 0
        assert sd["reroute_start"] == 0.20
        assert sd["total_steps"] == 200
        assert sd["freeze_step_frac"] == 0.60

    def test_state_dict_current_step_updates_after_steps(self) -> None:
        """After calling step() N times, state_dict['current_step'] == N."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched = RigLSparsitySchedule(total_steps=1000)
        for _ in range(7):
            sched.step()
        assert sched.state_dict()["current_step"] == 7

    def test_load_state_dict_restores_step_counter(self) -> None:
        """load_state_dict should restore _current_step from saved state."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched = RigLSparsitySchedule(
            reroute_start=0.30, total_steps=1000, freeze_step_frac=0.80
        )
        for _ in range(42):
            sched.step()

        saved = sched.state_dict()

        # Create a fresh scheduler and restore
        sched2 = RigLSparsitySchedule()
        sched2.load_state_dict(saved)
        assert sched2._current_step == 42

    def test_load_state_dict_restores_all_fields(self) -> None:
        """load_state_dict should restore reroute_start, total_steps, freeze_step_frac."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        saved = {
            "current_step": 99,
            "reroute_start": 0.15,
            "total_steps": 300,
            "freeze_step_frac": 0.70,
        }
        sched = RigLSparsitySchedule()
        sched.load_state_dict(saved)

        assert sched._current_step == 99
        assert sched.reroute_start == 0.15
        assert sched.total_steps == 300
        assert sched.freeze_step_frac == 0.70

    def test_round_trip_state_dict(self) -> None:
        """Saving and loading state_dict should produce identical next-step output."""
        from pcg_llm.training.scheduler import RigLSparsitySchedule

        sched_a = RigLSparsitySchedule(
            reroute_start=0.30, total_steps=1000, freeze_step_frac=0.80
        )
        for _ in range(10):
            sched_a.step()

        sched_b = RigLSparsitySchedule()
        sched_b.load_state_dict(sched_a.state_dict())

        f_a, frozen_a = sched_a.step()
        f_b, frozen_b = sched_b.step()
        assert abs(f_a - f_b) < 1e-9
        assert frozen_a == frozen_b
