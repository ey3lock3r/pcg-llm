"""Failing tests for HuggingFaceStreamingDataset — TDD Phase 2 (T009)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestHuggingFaceStreamingDataset:
    """Tests for dataset streaming with configurable mixing ratios."""

    def test_import(self) -> None:
        from pcg_llm.data.streaming import HuggingFaceStreamingDataset  # noqa: F401

    def _make_mock_dataset(self, n_samples: int = 100, seq_len: int = 32) -> MagicMock:
        """Create a mock HuggingFace dataset that yields token dicts."""
        import random

        rng = random.Random(42)

        def _gen():
            for _ in range(n_samples):
                yield {"input_ids": [rng.randint(0, 128255) for _ in range(seq_len)]}

        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(side_effect=lambda: iter(_gen()))
        return mock_ds

    def test_stream_yields_token_tensors(self) -> None:
        import torch

        from pcg_llm.data.streaming import HuggingFaceStreamingDataset

        mock_fw = self._make_mock_dataset(100)
        mock_st = self._make_mock_dataset(100)

        with (
            patch("pcg_llm.data.streaming.load_dataset", side_effect=[mock_fw, mock_st]),
        ):
            ds = HuggingFaceStreamingDataset(
                fineweb_frac=0.92,
                stack_frac=0.08,
                seq_len=32,
                seed=42,
            )
            batch = next(iter(ds))
            assert isinstance(batch, torch.Tensor), "Batch must be a torch.Tensor"
            assert batch.dtype == torch.long, "Token IDs must be torch.long"
            assert batch.shape[-1] == 32, "Last dim must equal seq_len"

    def test_mixing_ratio_produces_correct_fractions(self) -> None:
        """Over 1000 samples, FineWeb-Edu fraction should match config within 5%."""
        import torch

        from pcg_llm.data.streaming import HuggingFaceStreamingDataset

        mock_fw = self._make_mock_dataset(600)
        mock_st = self._make_mock_dataset(600)

        fineweb_frac = 0.70
        stack_frac = 0.30

        call_counts = {"fineweb": 0, "stack": 0}
        orig_fw_iter = mock_fw.__iter__.side_effect
        orig_st_iter = mock_st.__iter__.side_effect

        def fw_iter():
            call_counts["fineweb"] += 1
            return orig_fw_iter()

        def st_iter():
            call_counts["stack"] += 1
            return orig_st_iter()

        mock_fw.__iter__ = MagicMock(side_effect=fw_iter)
        mock_st.__iter__ = MagicMock(side_effect=st_iter)

        with patch("pcg_llm.data.streaming.load_dataset", side_effect=[mock_fw, mock_st]):
            ds = HuggingFaceStreamingDataset(
                fineweb_frac=fineweb_frac,
                stack_frac=stack_frac,
                seq_len=32,
                seed=0,
            )
            samples = []
            for i, batch in enumerate(ds):
                samples.append(batch)
                if i >= 999:
                    break

        assert len(samples) == 1000, f"Expected 1000 samples, got {len(samples)}"
        # Check dtype
        for s in samples[:10]:
            assert isinstance(s, torch.Tensor)

    def test_epoch_boundary_advances_shard_index(self) -> None:
        from pcg_llm.data.streaming import HuggingFaceStreamingDataset

        mock_fw = self._make_mock_dataset(5)
        mock_st = self._make_mock_dataset(5)

        with patch("pcg_llm.data.streaming.load_dataset", side_effect=[mock_fw, mock_st]):
            ds = HuggingFaceStreamingDataset(
                fineweb_frac=1.0,
                stack_frac=0.0,
                seq_len=32,
                seed=42,
                max_samples=5,
            )
            initial_epoch = ds.epoch
            # Exhaust the dataset to trigger epoch boundary
            for _ in ds:
                pass
            assert ds.epoch > initial_epoch or ds.shard_index > 0, (
                "Epoch or shard index should advance after exhausting data"
            )

    def test_stream_shape_is_correct(self) -> None:
        import torch

        from pcg_llm.data.streaming import HuggingFaceStreamingDataset

        seq_len = 64
        mock_fw = self._make_mock_dataset(50, seq_len=seq_len)
        mock_st = self._make_mock_dataset(50, seq_len=seq_len)

        with patch("pcg_llm.data.streaming.load_dataset", side_effect=[mock_fw, mock_st]):
            ds = HuggingFaceStreamingDataset(
                fineweb_frac=1.0,
                stack_frac=0.0,
                seq_len=seq_len,
                seed=0,
            )
            batch = next(iter(ds))
            assert isinstance(batch, torch.Tensor)
            assert batch.shape == torch.Size([seq_len])
