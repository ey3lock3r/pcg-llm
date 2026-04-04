"""GCS checkpoint backend with atomic write and exponential-backoff retry.

Requires ``google-cloud-storage`` 2.16+. If not installed the module can still
be imported but instantiation will raise ``ImportError`` with a helpful message.

Retry strategy: up to 5 attempts, base 1 s, max 32 s, on
``google.api_core.exceptions.ServiceUnavailable``.  Uses ``tenacity`` when
available; falls back to a manual retry loop otherwise.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: google-cloud-storage
# ---------------------------------------------------------------------------

try:
    from google.cloud import storage

    _GCS_AVAILABLE = True
except ImportError:
    storage = None
    _GCS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional dependency: tenacity
# ---------------------------------------------------------------------------

try:
    import tenacity

    _TENACITY_AVAILABLE = True
except ImportError:
    tenacity = None
    _TENACITY_AVAILABLE = False

_MAX_ATTEMPTS = 5
_BASE_DELAY_S = 1.0
_MAX_DELAY_S = 32.0


def _is_transient(exc: BaseException) -> bool:
    """Return True if *exc* is a transient GCS ServiceUnavailable error."""
    try:
        from google.api_core import exceptions as gapi_exc

        return isinstance(exc, gapi_exc.ServiceUnavailable)
    except ImportError:
        return False


def _retry_manual(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Simple manual exponential-backoff retry loop (fallback when tenacity absent)."""
    delay = _BASE_DELAY_S
    last_exc: BaseException | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except BaseException as exc:
            if not _is_transient(exc):
                raise
            last_exc = exc
            if attempt == _MAX_ATTEMPTS:
                break
            sleep_time = min(delay, _MAX_DELAY_S)
            logger.warning(
                "GCS transient error (attempt %d/%d), retrying in %.1fs: %s",
                attempt,
                _MAX_ATTEMPTS,
                sleep_time,
                exc,
            )
            time.sleep(sleep_time)
            delay *= 2.0
    raise last_exc  # type: ignore[misc]


if _TENACITY_AVAILABLE:
    import tenacity as _tenacity

    def _retry_upload(fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Retry wrapper using tenacity."""
        retryer = _tenacity.retry(
            retry=_tenacity.retry_if_exception(_is_transient),
            wait=_tenacity.wait_exponential(
                multiplier=_BASE_DELAY_S, min=_BASE_DELAY_S, max=_MAX_DELAY_S
            ),
            stop=_tenacity.stop_after_attempt(_MAX_ATTEMPTS),
            reraise=True,
        )
        return retryer(fn)(*args, **kwargs)

else:
    _retry_upload = _retry_manual


# ---------------------------------------------------------------------------
# GCSCheckpointBackend
# ---------------------------------------------------------------------------


class GCSCheckpointBackend:
    """Checkpoint backend that writes to Google Cloud Storage.

    Args:
        checkpoint_dir: Must start with ``gs://``.  Trailing slash is
            normalised automatically.

    Raises:
        ImportError: if ``google-cloud-storage`` is not installed.
        ValueError: if *checkpoint_dir* does not start with ``gs://``.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        if storage is None:
            raise ImportError(
                "google-cloud-storage is required for GCSCheckpointBackend. "
                "Install it with: pip install google-cloud-storage>=2.16"
            )
        if not checkpoint_dir.startswith("gs://"):
            raise ValueError(f"checkpoint_dir must start with 'gs://', got '{checkpoint_dir}'")
        # Normalise: ensure trailing slash on the prefix portion
        self.checkpoint_dir = (
            checkpoint_dir if checkpoint_dir.endswith("/") else checkpoint_dir + "/"
        )
        self._client = storage.Client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write(self, step: int, data: bytes) -> str:
        """Write *data* to GCS atomically via a tmp blob + rewrite rename.

        Args:
            step: Training step number, used to name the checkpoint file.
            data: Raw bytes (output of ``torch.save`` into a BytesIO buffer).

        Returns:
            The final GCS blob path as a string (``gs://bucket/prefix/step-NNN.pt``).
        """
        bucket_name, prefix = self._parse_gs_path()
        tmp_blob_name = f"{prefix}step-{step:06d}.pt.tmp"
        final_blob_name = f"{prefix}step-{step:06d}.pt"

        bucket = self._client.bucket(bucket_name)
        # Obtain blob handles up front so each retry does not re-call bucket.blob()
        tmp_blob = bucket.blob(tmp_blob_name)

        def _do_upload() -> None:
            tmp_blob.upload_from_string(data, content_type="application/octet-stream")
            logger.debug("GCS tmp blob written: gs://%s/%s", bucket_name, tmp_blob_name)

        _retry_upload(_do_upload)

        def _do_rename() -> None:
            # GCS "rename" = server-side copy (rewrite) + delete source
            bucket.copy_blob(tmp_blob, bucket, final_blob_name)
            tmp_blob.delete()
            logger.info("GCS checkpoint written: gs://%s/%s", bucket_name, final_blob_name)

        _retry_upload(_do_rename)

        return f"gs://{bucket_name}/{final_blob_name}"

    def list_checkpoints(self) -> list[str]:
        """Return all ``.pt`` blob paths sorted by step (ascending).

        Returns:
            A list of GCS paths of the form ``gs://bucket/prefix/step-NNN.pt``.
        """
        bucket_name, prefix = self._parse_gs_path()
        bucket = self._client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=prefix))

        step_pattern = re.compile(r"step-(\d+)\.pt$")
        results: list[tuple[int, str]] = []
        for blob in blobs:
            m = step_pattern.search(blob.name)
            if m:
                step_num = int(m.group(1))
                results.append((step_num, f"gs://{bucket_name}/{blob.name}"))

        results.sort(key=lambda t: t[0])
        return [path for _, path in results]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_gs_path(self) -> tuple[str, str]:
        """Split ``gs://bucket/prefix/`` into ``(bucket_name, prefix)``.

        The prefix always ends with ``/`` (or is empty for bucket root).
        """
        without_scheme = self.checkpoint_dir[len("gs://") :]
        slash_idx = without_scheme.find("/")
        if slash_idx == -1:
            return without_scheme, ""
        bucket_name = without_scheme[:slash_idx]
        prefix = without_scheme[slash_idx + 1 :]  # includes trailing slash
        return bucket_name, prefix
