"""Tests for CheckpointManager — TDD Phase 3 (T014)."""

from __future__ import annotations

import hashlib
import json

import torch


class TestCheckpointManager:
    """Verify atomic write, SHA-256 verification, manifest, and resume logic."""

    def _make_minimal_checkpoint_dict(self, step: int = 0) -> dict:
        """Minimal valid checkpoint dict per contracts/checkpoint.md."""
        return {
            "schema_version": "1.0",
            "step": step,
            "config": {"hidden_dim": 64, "vocab_size": 128256},
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "rigl_mask": torch.zeros(4, 4, dtype=torch.bool),
            "rigl_weights": torch.zeros(4, 4),
            "rigl_schedule_step": 0,
            "rigl_rerouting_fraction": 0.30,
            "rigl_frozen": False,
            "anderson_iterates": [],
            "anderson_residuals": [],
            "curriculum_epoch": 0,
            "curriculum_shard_index": 0,
            "curriculum_gating_temp": 1.0,
            "loss_crossentropy": 1.0,
            "loss_sparsity": 0.0,
            "loss_variance_hinge": 0.0,
            "loss_total": 1.0,
            "gamma_current": 0.01,
            "eagle_draft_len": 4,
            "eagle_accept_rate_ema": 0.0,
            "rng_torch": {},
            "rng_numpy": b"",
            "rng_python": (),
            "timestamp_utc": "2026-04-04T00:00:00Z",
            "hostname": "test-host",
            "sha256": "",  # filled in by CheckpointManager
        }

    def test_import(self) -> None:
        from pcg_llm.checkpointing.checkpoint import CheckpointManager  # noqa: F401

    def test_checkpoint_dict_contains_all_required_keys(self, temp_checkpoint_dir) -> None:
        """Contract: all required keys per contracts/checkpoint.md."""
        required_keys = [
            "schema_version",
            "step",
            "config",
            "model_state_dict",
            "optimizer_state_dict",
            "rigl_mask",
            "rigl_weights",
            "rigl_schedule_step",
            "rigl_rerouting_fraction",
            "rigl_frozen",
            "anderson_iterates",
            "anderson_residuals",
            "curriculum_epoch",
            "curriculum_shard_index",
            "curriculum_gating_temp",
            "loss_crossentropy",
            "loss_sparsity",
            "loss_variance_hinge",
            "loss_total",
            "gamma_current",
            "eagle_draft_len",
            "eagle_accept_rate_ema",
            "rng_torch",
            "rng_numpy",
            "rng_python",
            "timestamp_utc",
            "hostname",
            "sha256",
        ]
        ckpt = self._make_minimal_checkpoint_dict(step=100)
        for key in required_keys:
            assert key in ckpt, f"Required key '{key}' missing from checkpoint dict"

    def test_sha256_in_manifest_matches_file(self, temp_checkpoint_dir) -> None:
        from pcg_llm.checkpointing.checkpoint import CheckpointManager
        from pcg_llm.checkpointing.local import LocalCheckpointBackend

        backend = LocalCheckpointBackend(checkpoint_dir=str(temp_checkpoint_dir))
        manager = CheckpointManager(backend=backend, checkpoint_dir=str(temp_checkpoint_dir))

        ckpt = self._make_minimal_checkpoint_dict(step=50)
        saved_path = manager.save(ckpt, step=50)

        # Read the manifest
        manifest_path = temp_checkpoint_dir / "manifest.json"
        assert manifest_path.exists(), "Manifest must be written after checkpoint save"
        with open(manifest_path) as f:
            manifest = json.load(f)

        entry = next(e for e in manifest["checkpoints"] if e["step"] == 50)
        # Compute SHA-256 of saved file
        with open(saved_path, "rb") as f:
            actual_sha = hashlib.sha256(f.read()).hexdigest()

        assert (
            entry["sha256"] == actual_sha
        ), f"Manifest SHA-256 {entry['sha256']} != file SHA-256 {actual_sha}"

    def test_atomic_write_leaves_no_tmp_on_success(self, temp_checkpoint_dir) -> None:
        from pcg_llm.checkpointing.checkpoint import CheckpointManager
        from pcg_llm.checkpointing.local import LocalCheckpointBackend

        backend = LocalCheckpointBackend(checkpoint_dir=str(temp_checkpoint_dir))
        manager = CheckpointManager(backend=backend, checkpoint_dir=str(temp_checkpoint_dir))

        ckpt = self._make_minimal_checkpoint_dict(step=100)
        manager.save(ckpt, step=100)

        # No .tmp files should remain
        tmp_files = list(temp_checkpoint_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Orphaned .tmp files found: {tmp_files}"

    def test_corrupt_checkpoint_detected_via_checksum(self, temp_checkpoint_dir) -> None:
        """Truncated .pt file is detected and fallback to previous checkpoint is used."""
        from pcg_llm.checkpointing.checkpoint import CheckpointManager
        from pcg_llm.checkpointing.local import LocalCheckpointBackend

        backend = LocalCheckpointBackend(checkpoint_dir=str(temp_checkpoint_dir))
        manager = CheckpointManager(backend=backend, checkpoint_dir=str(temp_checkpoint_dir))

        # Save two valid checkpoints
        ckpt1 = self._make_minimal_checkpoint_dict(step=50)
        ckpt2 = self._make_minimal_checkpoint_dict(step=100)
        manager.save(ckpt1, step=50)
        path2 = manager.save(ckpt2, step=100)

        # Corrupt the second checkpoint
        with open(path2, "ab") as f:
            f.write(b"\x00\x00\x00CORRUPT")

        # Load latest should detect corruption and fall back to step 50
        loaded, loaded_step = manager.load_latest()
        assert (
            loaded_step == 50
        ), f"Should fall back to step 50 after step-100 corruption, got {loaded_step}"

    def test_manifest_latest_valid_step_is_most_recent_valid(self, temp_checkpoint_dir) -> None:
        from pcg_llm.checkpointing.checkpoint import CheckpointManager
        from pcg_llm.checkpointing.local import LocalCheckpointBackend

        backend = LocalCheckpointBackend(checkpoint_dir=str(temp_checkpoint_dir))
        manager = CheckpointManager(backend=backend, checkpoint_dir=str(temp_checkpoint_dir))

        for step in [50, 100, 150]:
            ckpt = self._make_minimal_checkpoint_dict(step=step)
            manager.save(ckpt, step=step)

        manifest_path = temp_checkpoint_dir / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["latest_valid_step"] == 150

    def test_resume_loads_from_correct_step(self, temp_checkpoint_dir) -> None:
        from pcg_llm.checkpointing.checkpoint import CheckpointManager
        from pcg_llm.checkpointing.local import LocalCheckpointBackend

        backend = LocalCheckpointBackend(checkpoint_dir=str(temp_checkpoint_dir))
        manager = CheckpointManager(backend=backend, checkpoint_dir=str(temp_checkpoint_dir))

        for step in [100, 200]:
            ckpt = self._make_minimal_checkpoint_dict(step=step)
            manager.save(ckpt, step=step)

        loaded, loaded_step = manager.load_latest()
        assert loaded_step == 200, f"Resume must load from step 200, got {loaded_step}"
        assert loaded["step"] == 200


# ---------------------------------------------------------------------------
# Disk quota trigger (FR-025) — T058
# ---------------------------------------------------------------------------


class TestDiskQuotaTriggerFR025:
    """FR-025: LocalCheckpointBackend must call shutdown_callback at < 500 MB free."""

    def test_shutdown_callback_called_when_below_500mb(self, tmp_path) -> None:
        """When disk_free < 500 MB, shutdown_callback must be invoked."""
        import shutil
        from unittest.mock import MagicMock, patch

        from pcg_llm.checkpointing.local import _500MB, LocalCheckpointBackend

        shutdown_called = MagicMock()
        backend = LocalCheckpointBackend(
            checkpoint_dir=str(tmp_path),
            shutdown_callback=shutdown_called,
        )

        # Simulate < 500 MB free
        fake_usage = shutil.disk_usage.__class__  # usage named-tuple type
        mock_free = _500MB - 1  # 1 byte below threshold
        with patch("shutil.disk_usage", return_value=type("usage", (), {"free": mock_free})()):
            backend._check_disk_quota()

        shutdown_called.assert_called_once()

    def test_shutdown_callback_not_called_when_above_500mb(self, tmp_path) -> None:
        """When disk_free >= 500 MB (but < 1 GB), only a warning is logged — no shutdown."""
        from unittest.mock import MagicMock, patch

        from pcg_llm.checkpointing.local import _500MB, LocalCheckpointBackend

        shutdown_called = MagicMock()
        backend = LocalCheckpointBackend(
            checkpoint_dir=str(tmp_path),
            shutdown_callback=shutdown_called,
        )

        # Simulate 600 MB free (between 500 MB and 1 GB)
        mock_free = _500MB + 100 * 1024 * 1024  # 600 MB
        with patch("shutil.disk_usage", return_value=type("usage", (), {"free": mock_free})()):
            backend._check_disk_quota()

        shutdown_called.assert_not_called()

    def test_disk_quota_warning_logged_below_1gb(self, tmp_path, caplog) -> None:
        """When disk_free < 1 GB (but >= 500 MB), DiskQuotaWarning-level log is emitted."""
        import logging
        from unittest.mock import patch

        from pcg_llm.checkpointing.local import _500MB, LocalCheckpointBackend

        backend = LocalCheckpointBackend(checkpoint_dir=str(tmp_path))

        mock_free = _500MB + 50 * 1024 * 1024  # 550 MB (< 1 GB, >= 500 MB)
        with caplog.at_level(logging.WARNING, logger="pcg_llm.checkpointing.local"):
            with patch(
                "shutil.disk_usage",
                return_value=type("usage", (), {"free": mock_free})(),
            ):
                backend._check_disk_quota()

        assert any(
            "low" in r.message.lower() for r in caplog.records
        ), "A low-disk warning must be logged when free < 1 GB"

    def test_no_shutdown_callback_does_not_raise(self, tmp_path) -> None:
        """LocalCheckpointBackend without shutdown_callback must not raise at < 500 MB."""
        from unittest.mock import patch

        from pcg_llm.checkpointing.local import _500MB, LocalCheckpointBackend

        backend = LocalCheckpointBackend(checkpoint_dir=str(tmp_path))  # no callback

        mock_free = _500MB - 1
        with patch("shutil.disk_usage", return_value=type("usage", (), {"free": mock_free})()):
            backend._check_disk_quota()  # must not raise
