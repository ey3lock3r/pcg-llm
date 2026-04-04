"""HuggingFaceStreamingDataset: streams FineWeb-Edu and The Stack v2.

Implements configurable mixing ratios from TrainingConfig per data-model.md.
Uses HuggingFace `datasets` streaming API with deterministic interleaving.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

import torch


def load_dataset(*args, **kwargs):  # type: ignore[no-untyped-def]
    """Thin wrapper around datasets.load_dataset for testability (mock target)."""
    try:
        from datasets import load_dataset as _load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets>=2.19.0 is required for streaming. "
            "Install with: pip install 'pcg-llm[training]'"
        ) from exc
    return _load_dataset(*args, **kwargs)


class HuggingFaceStreamingDataset:
    """Interleaved streaming dataset mixing FineWeb-Edu and The Stack v2.

    Yields individual token ID tensors of shape [seq_len] (dtype=torch.long).
    Source selection per sample is governed by (fineweb_frac, stack_frac).

    Args:
        fineweb_frac: Fraction of samples drawn from FineWeb-Edu (0.0–1.0).
        stack_frac: Fraction of samples from The Stack v2 (must sum to 1.0 with fineweb_frac).
        seq_len: Sequence length for each yielded tensor.
        seed: Random seed for deterministic mixing.
        max_samples: If set, stop after this many samples (useful for testing).
    """

    FINEWEB_NAME = "HuggingFaceFW/fineweb-edu"
    STACK_NAME = "bigcode/the-stack-v2-train-smol-ids"

    def __init__(
        self,
        fineweb_frac: float,
        stack_frac: float,
        seq_len: int,
        seed: int = 42,
        max_samples: int | None = None,
    ) -> None:
        if abs(fineweb_frac + stack_frac - 1.0) > 1e-9:
            raise ValueError(
                f"fineweb_frac + stack_frac must equal 1.0, "
                f"got {fineweb_frac} + {stack_frac} = {fineweb_frac + stack_frac}"
            )
        self.fineweb_frac = fineweb_frac
        self.stack_frac = stack_frac
        self.seq_len = seq_len
        self.seed = seed
        self.max_samples = max_samples

        # Epoch / shard tracking for checkpoint serialization
        self.epoch: int = 0
        self.shard_index: int = 0

        self._rng = random.Random(seed)
        self._fineweb_iter: Iterator | None = None
        self._stack_iter: Iterator | None = None
        self._init_iterators()

    def _init_iterators(self) -> None:
        """Load (or reload) streaming iterators for both datasets."""
        kwargs: dict[str, object] = {"streaming": True, "trust_remote_code": True}
        self._fineweb_ds = load_dataset(self.FINEWEB_NAME, split="train", **kwargs)
        self._stack_ds = load_dataset(self.STACK_NAME, split="train", **kwargs)
        self._fineweb_iter = iter(self._fineweb_ds)
        self._stack_iter = iter(self._stack_ds)

    def _next_from_source(self, source: str) -> list[int] | None:
        """Pull the next sample from the specified source.

        Returns None if the iterator is exhausted.
        """
        it = self._fineweb_iter if source == "fineweb" else self._stack_iter
        if it is None:
            return None
        try:
            row = next(it)
        except (StopIteration, RuntimeError):
            return None

        # Extract token IDs; different datasets use different column names.
        ids: list[int] = []
        for col in ("input_ids", "tokens", "token_ids", "text"):
            if col in row:
                val = row[col]
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], int):
                    ids = val
                    break
                if isinstance(val, str):
                    # Raw text fallback — shouldn't happen with tokenized datasets
                    ids = [ord(c) % 128256 for c in val[: self.seq_len]]
                    break

        if not ids:
            return []

        # Truncate or pad to seq_len
        if len(ids) >= self.seq_len:
            return ids[: self.seq_len]
        # Pad with zeros if shorter than seq_len
        return ids + [0] * (self.seq_len - len(ids))

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield token tensors mixing both sources according to configured fractions."""
        count = 0
        exhausted_epochs = 0

        while True:
            # Choose source stochastically according to mixing fractions
            if self._rng.random() < self.fineweb_frac:
                source = "fineweb"
            else:
                source = "stack"

            ids = self._next_from_source(source)

            if ids is None:
                # This source is exhausted — advance shard and try the other
                self.shard_index += 1
                exhausted_epochs += 1
                other = "stack" if source == "fineweb" else "fineweb"
                ids = self._next_from_source(other)
                if ids is None:
                    # Both sources exhausted; start new epoch
                    self.epoch += 1
                    exhausted_epochs = 0
                    try:
                        self._init_iterators()
                    except (StopIteration, RuntimeError, Exception):
                        break  # Cannot reload (e.g. mock exhausted in tests)
                    continue

            # Check max_samples after exhaustion detection but before yielding
            if self.max_samples is not None and count >= self.max_samples:
                break

            if ids is not None and len(ids) == self.seq_len:
                yield torch.tensor(ids, dtype=torch.long)
                count += 1
