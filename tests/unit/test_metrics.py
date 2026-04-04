"""Tests for TrainingMetrics — TDD Phase 6 (T044)."""

from __future__ import annotations

import logging


class TestTrainingMetrics:
    def test_import(self) -> None:
        from pcg_llm.monitoring.metrics import TrainingMetrics  # noqa

    def test_record_step_populates_all_fields(self) -> None:
        from pcg_llm.monitoring.metrics import TrainingMetrics

        m = TrainingMetrics(max_solver_iters=12, variance_floor=0.1)
        m.record_step(
            step=1,
            solver_steps=6,
            sparsity=0.85,
            node_variance=0.2,
            eagle_rate=0.7,
            loss_ce=1.0,
            loss_sparsity=0.01,
            loss_variance=0.0,
            loss_total=1.01,
            gamma=0.01,
            rigl_reroute_fraction=0.30,
            rigl_frozen=False,
            vram_gb=3.0,
            throughput_tps=1000.0,
        )
        d = m.to_dict()
        assert d["step"] == 1
        assert d["solver_steps_mean"] == 6
        assert d["sparsity"] == 0.85

    def test_alert_fires_when_solver_steps_high(self, caplog) -> None:
        from pcg_llm.monitoring.metrics import TrainingMetrics

        m = TrainingMetrics(max_solver_iters=12, variance_floor=0.1)
        with caplog.at_level(logging.WARNING):
            m.record_step(
                step=1,
                solver_steps=11,  # > 80% of 12 = 9.6
                sparsity=0.85,
                node_variance=0.2,
                eagle_rate=0.7,
                loss_ce=1.0,
                loss_sparsity=0.0,
                loss_variance=0.0,
                loss_total=1.0,
                gamma=0.01,
                rigl_reroute_fraction=0.3,
                rigl_frozen=False,
                vram_gb=3.0,
                throughput_tps=1000.0,
            )
            m.check_alerts()
        assert any(
            "solver" in r.message.lower() for r in caplog.records
        ), "Should warn when solver_steps_mean > 80% of max"

    def test_alert_fires_when_variance_low(self, caplog) -> None:
        from pcg_llm.monitoring.metrics import TrainingMetrics

        m = TrainingMetrics(max_solver_iters=12, variance_floor=0.1)
        with caplog.at_level(logging.WARNING):
            m.record_step(
                step=1,
                solver_steps=3,
                sparsity=0.85,
                node_variance=0.05,  # < 0.1 floor
                eagle_rate=0.7,
                loss_ce=1.0,
                loss_sparsity=0.0,
                loss_variance=0.0,
                loss_total=1.0,
                gamma=0.01,
                rigl_reroute_fraction=0.3,
                rigl_frozen=False,
                vram_gb=3.0,
                throughput_tps=1000.0,
            )
            m.check_alerts()
        assert any("variance" in r.message.lower() for r in caplog.records)

    def test_eagle_alert_fires_when_rate_below_threshold(self, caplog) -> None:
        from pcg_llm.monitoring.metrics import TrainingMetrics

        m = TrainingMetrics(max_solver_iters=12, variance_floor=0.1)
        # Record 1000+ steps with low eagle rate
        for i in range(1001):
            m.record_step(
                step=i,
                solver_steps=3,
                sparsity=0.85,
                node_variance=0.2,
                eagle_rate=0.50,  # < 0.60 threshold
                loss_ce=1.0,
                loss_sparsity=0.0,
                loss_variance=0.0,
                loss_total=1.0,
                gamma=0.01,
                rigl_reroute_fraction=0.3,
                rigl_frozen=False,
                vram_gb=3.0,
                throughput_tps=1000.0,
            )
        with caplog.at_level(logging.WARNING):
            m.check_alerts()
        assert any(
            "eagle" in r.message.lower() or "accept" in r.message.lower() for r in caplog.records
        )
