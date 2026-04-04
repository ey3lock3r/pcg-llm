"""Failing tests for Llama3TokenizerWrapper — TDD Phase 2 (T007).

Tests should fail until tokenizer.py is implemented (T008).
"""

from __future__ import annotations

import pytest


transformers = pytest.importorskip("transformers", reason="transformers not installed; skip tokenizer tests")


class TestLlama3TokenizerWrapper:
    """Tests for the Llama-3 tokenizer wrapper per contracts and data-model."""

    @pytest.fixture
    def tokenizer(self):
        """Instantiate the tokenizer wrapper."""
        from pcg_llm.data.tokenizer import Llama3TokenizerWrapper

        return Llama3TokenizerWrapper()

    def test_import(self) -> None:
        from pcg_llm.data.tokenizer import Llama3TokenizerWrapper  # noqa: F401

    def test_vocab_size_is_128256(self, tokenizer) -> None:
        assert tokenizer.vocab_size == 128256

    def test_encode_returns_list_of_ints(self, tokenizer) -> None:
        ids = tokenizer.encode("Hello, world!")
        assert isinstance(ids, list)
        assert all(isinstance(t, int) for t in ids)
        assert len(ids) > 0

    def test_decode_returns_string(self, tokenizer) -> None:
        ids = tokenizer.encode("Hello, world!")
        decoded = tokenizer.decode(ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_encode_decode_round_trip(self, tokenizer) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        # Round trip should preserve the original text (possibly with minor whitespace)
        assert text in decoded or decoded.strip() == text.strip()

    def test_encode_returns_ids_within_vocab(self, tokenizer) -> None:
        ids = tokenizer.encode("Hello, world!")
        for token_id in ids:
            assert 0 <= token_id < tokenizer.vocab_size

    def test_bos_token_id_exists(self, tokenizer) -> None:
        assert hasattr(tokenizer, "bos_token_id")
        assert isinstance(tokenizer.bos_token_id, int)

    def test_eos_token_id_exists(self, tokenizer) -> None:
        assert hasattr(tokenizer, "eos_token_id")
        assert isinstance(tokenizer.eos_token_id, int)

    def test_bos_eos_are_within_vocab(self, tokenizer) -> None:
        assert 0 <= tokenizer.bos_token_id < tokenizer.vocab_size
        assert 0 <= tokenizer.eos_token_id < tokenizer.vocab_size

    def test_encode_empty_string(self, tokenizer) -> None:
        ids = tokenizer.encode("")
        assert isinstance(ids, list)

    def test_encode_special_characters(self, tokenizer) -> None:
        ids = tokenizer.encode("<|endoftext|>")
        assert isinstance(ids, list)
        assert len(ids) >= 1
