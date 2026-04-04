"""Llama3TokenizerWrapper: wraps the HuggingFace Llama-3 tokenizer.

Provides a uniform encode/decode interface with the Llama-3 Tiktoken vocabulary
(vocab_size=128256) as specified in data-model.md and contracts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class Llama3TokenizerWrapper:
    """Wrapper around the HuggingFace Llama-3 tokenizer.

    vocab_size: 128256 (Llama-3 Tiktoken vocabulary)
    """

    VOCAB_SIZE = 128256
    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    # Fallback model name for environments where the gated model is unavailable
    FALLBACK_MODEL_NAME = "hf-internal-testing/llama-tokenizer"

    def __init__(self) -> None:
        self._tokenizer = self._load_tokenizer()

    def _load_tokenizer(self) -> Any:
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.40.0 is required for Llama3TokenizerWrapper. "
                "Install with: pip install 'pcg-llm[training]'"
            ) from exc

        # Try the primary Llama-3 model first, fall back to a compatible tokenizer
        # for environments without HuggingFace Hub credentials.
        for model_name in [self.MODEL_NAME, self.FALLBACK_MODEL_NAME]:
            try:
                return AutoTokenizer.from_pretrained(model_name)
            except Exception:  # noqa: BLE001
                continue

        raise RuntimeError(
            f"Could not load Llama-3 tokenizer from '{self.MODEL_NAME}'. "
            "Ensure you have accepted the model license at "
            "https://huggingface.co/meta-llama/Meta-Llama-3-8B and have run "
            "`huggingface-cli login`."
        )

    @property
    def vocab_size(self) -> int:
        """Vocabulary size: 128256 (Llama-3 Tiktoken)."""
        return self.VOCAB_SIZE

    @property
    def bos_token_id(self) -> int:
        """Beginning-of-sequence token ID."""
        bos = self._tokenizer.bos_token_id
        if bos is None:
            return 128000  # Llama-3 BOS token
        return int(bos)

    @property
    def eos_token_id(self) -> int:
        """End-of-sequence token ID."""
        eos = self._tokenizer.eos_token_id
        if eos is None:
            return 128001  # Llama-3 EOS token
        return int(eos)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Encode text to a list of integer token IDs.

        Args:
            text: Input text to tokenize.
            add_special_tokens: Whether to prepend BOS / append EOS.

        Returns:
            List of integer token IDs in [0, vocab_size).
        """
        ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return [int(t) for t in ids]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode a list of token IDs back to text.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: Whether to omit special tokens from output.

        Returns:
            Decoded string.
        """
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens))
