"""Data curriculum managing epoch, shard, and dataset mixing ratios."""

from __future__ import annotations


class DataCurriculum:
    """Manages epoch progression, shard indexing, and FineWeb/Stack mixing ratios.

    The curriculum is designed for streaming datasets where the model trains
    on increasing-difficulty data over epochs.  Callers advance the shard
    pointer as each shard is consumed and query the mixing ratio to determine
    how much FineWeb-Edu vs. The Stack v2 data to draw each step.

    When the requested *epoch* index exceeds the length of the fraction lists
    the last entry is reused (plateau behaviour), so callers can safely pass
    epochs beyond the explicitly configured range.

    Args:
        fineweb_fracs: Per-epoch mixing fraction for FineWeb-Edu data.
            ``fineweb_fracs[0]`` is used for epoch 0, and so on.
        stack_fracs: Per-epoch mixing fraction for The Stack v2 data.
            Must have the same length as *fineweb_fracs*.
    """

    def __init__(
        self,
        fineweb_fracs: list[float],
        stack_fracs: list[float],
    ) -> None:
        if not fineweb_fracs:
            raise ValueError("fineweb_fracs must be non-empty")
        if not stack_fracs:
            raise ValueError("stack_fracs must be non-empty")
        self.fineweb_fracs = list(fineweb_fracs)
        self.stack_fracs = list(stack_fracs)
        self.epoch: int = 0
        self.shard_index: int = 0

    # ------------------------------------------------------------------
    # Progression
    # ------------------------------------------------------------------

    def advance_shard(self) -> None:
        """Increment the shard index by one."""
        self.shard_index += 1

    # ------------------------------------------------------------------
    # Mixing ratio query
    # ------------------------------------------------------------------

    def get_mixing_ratio(self, epoch: int) -> tuple[float, float]:
        """Return the (FineWeb, Stack) mixing fractions for a given epoch.

        If *epoch* >= the number of configured entries, the last entry is
        returned (plateau / no extrapolation).

        Args:
            epoch: Zero-based epoch index to query.

        Returns:
            A tuple ``(fineweb_frac, stack_frac)``.
        """
        idx = min(epoch, len(self.fineweb_fracs) - 1)
        stack_idx = min(epoch, len(self.stack_fracs) - 1)
        return (self.fineweb_fracs[idx], self.stack_fracs[stack_idx])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a snapshot of the curriculum state.

        Returns:
            Dict with keys ``"epoch"`` and ``"shard_index"``.
        """
        return {
            "epoch": self.epoch,
            "shard_index": self.shard_index,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore curriculum state from a dict produced by :meth:`state_dict`.

        Args:
            d: Dict with keys ``"epoch"`` and ``"shard_index"``.
        """
        self.epoch = int(d["epoch"])
        self.shard_index = int(d["shard_index"])
