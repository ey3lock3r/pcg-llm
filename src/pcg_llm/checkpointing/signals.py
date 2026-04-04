"""SignalHandler: SIGTERM / KeyboardInterrupt → checkpoint save.

Handles both Kaggle (KeyboardInterrupt) and GCP preemptible (SIGTERM) scenarios.
GCP sends SIGTERM 30 seconds before reclaim; checkpoint must complete within 25s.
"""

from __future__ import annotations

import logging
import signal
import time
from typing import Any

logger = logging.getLogger(__name__)

# GCP gives 30s warning; we target completing within 25s and warn at 20s.
GCP_CHECKPOINT_DEADLINE_SECONDS = 25.0
GCP_CHECKPOINT_WARN_SECONDS = 20.0


class SignalHandler:
    """Registers SIGTERM handler for graceful checkpoint save on preemption.

    GCP preemptible VMs send SIGTERM ~30 seconds before the instance is reclaimed.
    This handler:
      1. Sets _shutdown_requested = True on SIGTERM.
      2. Times the checkpoint save and logs a warning if it exceeds 20s.
      3. Exits with code 2 on clean checkpoint-save exit.
    """

    def __init__(self, checkpoint_manager: Any) -> None:
        self._manager = checkpoint_manager
        self._shutdown_requested = False
        self._sigterm_received_at: float | None = None
        self._register()

    def _register(self) -> None:
        """Register SIGTERM handler; silently skip on unsupported platforms."""
        try:
            signal.signal(signal.SIGTERM, self._handle_sigterm)
        except (OSError, ValueError):
            # SIGTERM registration not supported on this platform (e.g. Windows threads)
            pass

    def _handle_sigterm(self, signum: int, frame: Any) -> None:
        logger.warning("SIGTERM received, saving checkpoint...")
        self._shutdown_requested = True
        self._sigterm_received_at = time.monotonic()

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    def check_and_save(self, checkpoint_dict: dict, step: int) -> bool:
        """If shutdown was requested, save a checkpoint and return True.

        Measures save duration and logs a warning if the GCP 20-second soft
        deadline is exceeded. The caller should call sys.exit(2) when this
        returns True.

        Returns:
            True if shutdown was requested and checkpoint was saved; False otherwise.
        """
        if not self._shutdown_requested:
            return False

        t_start = time.monotonic()
        self._manager.save(checkpoint_dict, step)
        elapsed = time.monotonic() - t_start

        if elapsed >= GCP_CHECKPOINT_WARN_SECONDS:
            logger.warning(
                f"Checkpoint save took {elapsed:.1f}s — exceeded the {GCP_CHECKPOINT_WARN_SECONDS}s "
                f"soft deadline. Consider reducing --checkpoint-interval to keep saves under "
                f"{GCP_CHECKPOINT_DEADLINE_SECONDS}s for GCP preemption safety."
            )

        # Report time since SIGTERM for GCP deadline awareness
        if self._sigterm_received_at is not None:
            total_since_sigterm = time.monotonic() - self._sigterm_received_at
            if total_since_sigterm > GCP_CHECKPOINT_DEADLINE_SECONDS:
                logger.error(
                    f"Total time since SIGTERM: {total_since_sigterm:.1f}s — "
                    f"may exceed GCP 30-second reclaim window."
                )

        return True
