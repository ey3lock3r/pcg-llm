from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

_1GB = 1 * 1024 * 1024 * 1024
_500MB = 500 * 1024 * 1024


class DiskQuotaWarning(UserWarning):
    """Warning raised when disk space is critically low."""


class LocalCheckpointBackend:
    def __init__(
        self,
        checkpoint_dir: str,
        shutdown_callback: Callable[[], None] | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._shutdown_callback = shutdown_callback

    def write(self, step: int, data: bytes) -> Path:
        """Write bytes to {step:06d}.pt atomically via tmp → rename."""
        self._check_disk_quota()
        path = self.checkpoint_dir / f"step-{step:06d}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        tmp_path.write_bytes(data)
        tmp_path.replace(path)  # atomic on POSIX; best-effort on Windows
        logger.info("Checkpoint written: %s", path)
        return path

    def list_checkpoints(self) -> list[Path]:
        """Return all .pt checkpoint files sorted by step (ascending)."""
        return sorted(self.checkpoint_dir.glob("step-*.pt"))

    def _check_disk_quota(self) -> None:
        """Log warnings when disk free space is low."""
        usage = shutil.disk_usage(self.checkpoint_dir)
        free = usage.free
        if free < _500MB:
            logger.critical(
                "Disk space critically low: %.1f MB free (< 500 MB). Triggering graceful shutdown.",
                free / (1024 * 1024),
            )
            if self._shutdown_callback is not None:
                self._shutdown_callback()
        elif free < _1GB:
            logger.warning(
                "Disk space low: %.1f MB free (< 1 GB).",
                free / (1024 * 1024),
            )
