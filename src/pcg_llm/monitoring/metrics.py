"""TrainingMetrics — live training state and alerting.

Tracks solver convergence, sparsity, EAGLE acceptance, loss, and system
metrics per step.  Fires log-level warnings when thresholds are breached.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_EAGLE_ACCEPT_THRESHOLD = 0.60
_EAGLE_ALERT_MIN_STEPS = 1000
_SOLVER_ALERT_FRAC = 0.80


class TrainingMetrics:
    """Mutable live-metrics container for a single training run.

    Args:
        max_solver_iters: Upper bound on DEQ solver iterations (from config).
        variance_floor: Minimum acceptable node variance (from config).
    """

    def __init__(self, max_solver_iters: int, variance_floor: float) -> None:
        self.max_solver_iters = max_solver_iters
        self.variance_floor = variance_floor

        # --- Latest step snapshot ---
        self.step: int = 0
        self.solver_steps_mean: float = 0.0
        self.sparsity: float = 0.0
        self.node_variance: float = 0.0
        self.eagle_accept_rate_ema: float = 0.0
        self.loss_ce: float = 0.0
        self.loss_sparsity: float = 0.0
        self.loss_variance: float = 0.0
        self.loss_total: float = 0.0
        self.gamma: float = 0.0
        self.rigl_reroute_fraction: float = 0.0
        self.rigl_frozen: bool = False
        self.vram_gb: float = 0.0
        self.throughput_tps: float = 0.0

        # --- Running history for rolling alerts ---
        self._eagle_rate_history: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_step(
        self,
        step: int,
        solver_steps: float,
        sparsity: float,
        node_variance: float,
        eagle_rate: float,
        loss_ce: float,
        loss_sparsity: float,
        loss_variance: float,
        loss_total: float,
        gamma: float,
        rigl_reroute_fraction: float,
        rigl_frozen: bool,
        vram_gb: float,
        throughput_tps: float,
    ) -> None:
        """Update all tracking fields for the current training step.

        Args:
            step: Global training step counter.
            solver_steps: Mean DEQ solver iterations for this step.
            sparsity: Current adjacency matrix sparsity in [0, 1].
            node_variance: Mean variance of DEQ fixed-point node states.
            eagle_rate: EAGLE draft acceptance rate this step.
            loss_ce: Cross-entropy loss component.
            loss_sparsity: L1 sparsity loss component.
            loss_variance: Variance hinge loss component.
            loss_total: Total combined loss.
            gamma: Current variance-loss coefficient.
            rigl_reroute_fraction: Current RigL re-routing fraction.
            rigl_frozen: Whether RigL topology is frozen.
            vram_gb: GPU VRAM used in GB.
            throughput_tps: Training throughput in tokens per second.
        """
        self.step = step
        self.solver_steps_mean = float(solver_steps)
        self.sparsity = float(sparsity)
        self.node_variance = float(node_variance)
        self.eagle_accept_rate_ema = float(eagle_rate)
        self.loss_ce = float(loss_ce)
        self.loss_sparsity = float(loss_sparsity)
        self.loss_variance = float(loss_variance)
        self.loss_total = float(loss_total)
        self.gamma = float(gamma)
        self.rigl_reroute_fraction = float(rigl_reroute_fraction)
        self.rigl_frozen = bool(rigl_frozen)
        self.vram_gb = float(vram_gb)
        self.throughput_tps = float(throughput_tps)

        self._eagle_rate_history.append(float(eagle_rate))

    def check_alerts(self) -> None:
        """Check threshold conditions and emit log warnings for violations.

        Checks:
        - ``solver_steps_mean > 0.8 * max_solver_iters`` → WARNING
        - ``node_variance < variance_floor`` → WARNING
        - EAGLE accept rate EMA < 0.60 over 1000+ recorded steps → WARNING
        """
        solver_threshold = _SOLVER_ALERT_FRAC * self.max_solver_iters
        if self.solver_steps_mean > solver_threshold:
            logger.warning(
                "Solver steps mean %.1f exceeds %.0f%% of max (%d); "
                "DEQ may not be converging. Consider reducing learning rate or "
                "increasing max_solver_iters.",
                self.solver_steps_mean,
                _SOLVER_ALERT_FRAC * 100,
                self.max_solver_iters,
            )

        if self.node_variance < self.variance_floor:
            logger.warning(
                "Node variance %.4f is below variance floor %.4f; "
                "node states may be collapsing. Consider increasing gamma.",
                self.node_variance,
                self.variance_floor,
            )

        if len(self._eagle_rate_history) >= _EAGLE_ALERT_MIN_STEPS:
            recent = self._eagle_rate_history[-_EAGLE_ALERT_MIN_STEPS:]
            mean_rate = sum(recent) / len(recent)
            if mean_rate < _EAGLE_ACCEPT_THRESHOLD:
                logger.warning(
                    "EAGLE accept rate EMA %.3f (mean over last %d steps) is below "
                    "threshold %.2f; draft speculation is not effective.",
                    mean_rate,
                    _EAGLE_ALERT_MIN_STEPS,
                    _EAGLE_ACCEPT_THRESHOLD,
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the current metrics.

        Suitable for logging to W&B or other monitoring systems.
        """
        return {
            "step": self.step,
            "solver_steps_mean": self.solver_steps_mean,
            "sparsity": self.sparsity,
            "node_variance": self.node_variance,
            "eagle_accept_rate_ema": self.eagle_accept_rate_ema,
            "loss_ce": self.loss_ce,
            "loss_sparsity": self.loss_sparsity,
            "loss_variance": self.loss_variance,
            "loss_total": self.loss_total,
            "gamma": self.gamma,
            "rigl_reroute_fraction": self.rigl_reroute_fraction,
            "rigl_frozen": self.rigl_frozen,
            "vram_gb": self.vram_gb,
            "throughput_tps": self.throughput_tps,
        }
