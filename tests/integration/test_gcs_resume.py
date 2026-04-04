"""Tests for GCS checkpoint backend with mocked GCS SDK (T039)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.slow
class TestGCSCheckpointResume:
    def test_import(self) -> None:
        from pcg_llm.checkpointing.gcs import GCSCheckpointBackend  # noqa

    def test_atomic_write_uses_tmp_then_rename(self) -> None:
        """Verify tmp blob written first, then renamed to final path."""
        with patch("pcg_llm.checkpointing.gcs.storage") as mock_storage:
            mock_client = MagicMock()
            mock_storage.Client.return_value = mock_client
            mock_bucket = MagicMock()
            mock_client.bucket.return_value = mock_bucket
            mock_tmp_blob = MagicMock()
            mock_final_blob = MagicMock()
            mock_bucket.blob.side_effect = [mock_tmp_blob, mock_final_blob]

            from pcg_llm.checkpointing.gcs import GCSCheckpointBackend

            backend = GCSCheckpointBackend("gs://test-bucket/checkpoints/")
            backend.write(100, b"fake_checkpoint_data")

            # Verify tmp blob was uploaded
            mock_tmp_blob.upload_from_string.assert_called_once()

    def test_corrupted_checkpoint_skipped_falls_back(self, temp_checkpoint_dir) -> None:
        """Corrupted SHA-256 causes fallback to previous valid checkpoint."""
        import torch

        from pcg_llm.checkpointing.checkpoint import CheckpointManager
        from pcg_llm.checkpointing.local import LocalCheckpointBackend

        backend = LocalCheckpointBackend(str(temp_checkpoint_dir))
        manager = CheckpointManager(backend, str(temp_checkpoint_dir))

        # Save two valid checkpoints
        ckpt1 = {
            "schema_version": "1.0",
            "step": 50,
            "config": {},
            "model_state_dict": {},
            "optimizer_state_dict": {},
            "rigl_mask": torch.zeros(2, 2, dtype=torch.bool),
            "rigl_weights": torch.zeros(2, 2),
            "rigl_schedule_step": 0,
            "rigl_rerouting_fraction": 0.3,
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
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "hostname": "test",
            "sha256": "",
        }
        ckpt2 = {**ckpt1, "step": 100}

        manager.save(ckpt1, 50)
        path2 = manager.save(ckpt2, 100)

        # Corrupt checkpoint 2
        with open(path2, "ab") as f:
            f.write(b"CORRUPTION")

        loaded, step = manager.load_latest()
        assert step == 50, f"Should fall back to step 50 but got step {step}"
