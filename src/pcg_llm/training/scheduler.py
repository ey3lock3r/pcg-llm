"""RigL sparsity schedule with cosine decay of the re-routing fraction."""

from __future__ import annotations

import math


class RigLSparsitySchedule:
    """Cosine-annealed re-routing fraction for RigL sparse-training.

    The fraction of connections to re-route starts at *reroute_start* and
    decays to zero following a cosine curve over ``freeze_step_frac *
    total_steps`` steps.  After that point the topology is frozen and the
    scheduler always returns ``(0.0, True)``.

    The step counter is advanced by calling :meth:`step`, which also returns
    the current fraction and freeze flag.  This mirrors the PyTorch LR-scheduler
    pattern so callers can write::

        fraction, frozen = schedule.step()
        if not frozen:
            adjacency.drop_and_grow(grads, fraction, fraction)

    Args:
        reroute_start: Initial re-routing fraction (e.g. 0.30 = 30 %).
        total_steps: Total number of training steps.
        freeze_step_frac: Fraction of total steps after which topology freezes.
    """

    def __init__(
        self,
        reroute_start: float = 0.30,
        total_steps: int = 1000,
        freeze_step_frac: float = 0.80,
    ) -> None:
        self.reroute_start = reroute_start
        self.total_steps = total_steps
        self.freeze_step_frac = freeze_step_frac
        self._current_step: int = 0

    def step(self) -> tuple[float, bool]:
        """Advance the step counter and return the current schedule values.

        Returns:
            A tuple ``(fraction, frozen)`` where:

            - *fraction* is the cosine-annealed re-routing fraction in
              ``[0, reroute_start]``.
            - *frozen* is ``True`` once the freeze step has been reached,
              after which *fraction* is always ``0.0``.
        """
        self._current_step += 1
        freeze_step = int(self.freeze_step_frac * self.total_steps)

        if self._current_step >= freeze_step:
            return (0.0, True)

        fraction = (
            self.reroute_start
            * 0.5
            * (1.0 + math.cos(math.pi * self._current_step / freeze_step))
        )
        return (fraction, False)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a snapshot of the scheduler state.

        Returns:
            Dict with keys ``"current_step"``, ``"reroute_start"``,
            ``"total_steps"``, and ``"freeze_step_frac"``.
        """
        return {
            "current_step": self._current_step,
            "reroute_start": self.reroute_start,
            "total_steps": self.total_steps,
            "freeze_step_frac": self.freeze_step_frac,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore scheduler state from a dict produced by :meth:`state_dict`.

        Args:
            d: Dict with keys ``"current_step"``, ``"reroute_start"``,
               ``"total_steps"``, and ``"freeze_step_frac"``.
        """
        self._current_step = int(d["current_step"])
        self.reroute_start = float(d["reroute_start"])
        self.total_steps = int(d["total_steps"])
        self.freeze_step_frac = float(d["freeze_step_frac"])
