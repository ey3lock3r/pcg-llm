from __future__ import annotations

import hashlib
import io
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch

from pcg_llm.checkpointing.local import LocalCheckpointBackend

logger = logging.getLogger(__name__)

_EMPTY_MANIFEST: dict = {
    "schema_version": "1.0",
    "latest_valid_step": None,
    "checkpoints": [],
}


class CheckpointManager:
    def __init__(
        self,
        backend: LocalCheckpointBackend | Any,
        checkpoint_dir: str,
    ) -> None:
        self.backend = backend
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.checkpoint_dir / "manifest.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, checkpoint_dict: dict, step: int) -> Path:
        """Serialize *checkpoint_dict*, write atomically, update manifest.

        Returns the path to the written ``.pt`` file.
        """
        # Step 1: serialize to bytes via BytesIO
        buf = io.BytesIO()
        torch.save(checkpoint_dict, buf)
        data = buf.getvalue()

        # Step 2: write via backend (atomic tmp → rename)
        path = self.backend.write(step, data)

        # Step 3: compute SHA-256 from the file as written to disk
        sha256 = self._sha256_of_file(path)

        # Step 4: load manifest, add entry, save manifest
        manifest = self._load_manifest()
        timestamp = datetime.now(tz=UTC).isoformat()

        entry = {
            "step": step,
            "path": str(path),
            "sha256": sha256,
            "timestamp_utc": timestamp,
            "loss_total": float(checkpoint_dict.get("loss_total", 0.0)),
            "sparsity": float(checkpoint_dict.get("sparsity", 0.0)),
            "solver_steps_mean": float(checkpoint_dict.get("solver_steps_mean", 0.0)),
            "valid": True,
        }

        # Replace any existing entry for this step
        manifest["checkpoints"] = [e for e in manifest["checkpoints"] if e.get("step") != step]
        manifest["checkpoints"].append(entry)
        manifest["latest_valid_step"] = step

        self._save_manifest(manifest)
        logger.info("Checkpoint saved at step %d: %s (sha256=%s)", step, path, sha256)
        return path

    def load_latest(self) -> tuple[dict, int]:
        """Load the most recent valid checkpoint, verifying SHA-256.

        If SHA-256 verification fails, the entry is marked ``valid=False``
        in the manifest and the previous checkpoint is attempted.

        Raises:
            FileNotFoundError: if no valid checkpoints exist.
        """
        manifest = self._load_manifest()
        checkpoints = manifest.get("checkpoints", [])

        # Sort descending by step so we try the latest first
        candidates = sorted(
            [e for e in checkpoints if e.get("valid", True)],
            key=lambda e: e["step"],
            reverse=True,
        )

        if not candidates:
            raise FileNotFoundError("No valid checkpoints found in manifest.")

        for entry in candidates:
            path = Path(entry["path"])
            expected_sha256 = entry.get("sha256", "")

            if not path.exists():
                logger.warning("Checkpoint file missing: %s — skipping.", path)
                entry["valid"] = False
                self._save_manifest(manifest)
                continue

            actual_sha256 = self._sha256_of_file(path)
            if actual_sha256 != expected_sha256:
                logger.error(
                    "SHA-256 mismatch for step %d: expected %s, got %s — marking invalid.",
                    entry["step"],
                    expected_sha256,
                    actual_sha256,
                )
                entry["valid"] = False
                self._save_manifest(manifest)
                continue

            checkpoint_dict = torch.load(path, weights_only=False)  # nosec B614
            step = entry["step"]
            logger.info("Loaded checkpoint step %d from %s", step, path)
            return checkpoint_dict, step

        raise FileNotFoundError(
            "All checkpoint entries failed SHA-256 verification or are missing."
        )

    # ------------------------------------------------------------------
    # Manifest helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict:
        """Read manifest.json; return an empty manifest structure if absent."""
        if not self._manifest_path.exists():
            import copy

            return copy.deepcopy(_EMPTY_MANIFEST)
        try:
            return cast(dict[Any, Any], json.loads(self._manifest_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read manifest (%s); starting fresh.", exc)
            import copy

            return copy.deepcopy(_EMPTY_MANIFEST)

    def _save_manifest(self, manifest: dict) -> None:
        """Write manifest atomically (tmp → rename)."""
        tmp_path = self._manifest_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp_path.replace(self._manifest_path)

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _sha256_of_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
