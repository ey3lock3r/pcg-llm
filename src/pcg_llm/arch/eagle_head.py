"""EAGLEExtrapolationHead — speculative decoding via PCG draft branches.

The EAGLE head operates on the penultimate PCG fixed-point states Z* and
generates K draft token branches of length L using a single lightweight
transformer block.  It tracks an EMA of the draft acceptance rate and
adaptively expands the draft length when performance is high.

Architecture:
    - Input: Z* of shape (B, num_blocks, hidden_dim)
    - A single transformer block (self-attention + FFN) produces K draft
      token sequences of length draft_len.
    - The head outputs a tensor of shape (B, K, draft_len, vocab_size) for
      token prediction and (B, K, draft_len) as the draft tree indices.

References:
    Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature
    Uncertainty", arXiv 2401.15077.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

_DRAFT_LEN_MAX = 16
_DRAFT_LEN_EXPAND_THRESHOLD_DELTA = 0.05  # hysteresis above accept_threshold


class EAGLEExtrapolationHead(nn.Module):
    """Lightweight speculative-decoding head for PCG-LLM.

    Args:
        hidden_dim: Dimensionality of the PCG node states.
        eagle_k: Number of draft branches to generate (``K``).
        draft_len: Initial number of draft tokens per branch (``L``).
        eagle_accept_threshold: EMA acceptance rate threshold for expansion
            (default 0.65).
        ema_alpha: Smoothing coefficient for the acceptance rate EMA
            (default 0.05).
        nhead: Number of attention heads in the internal transformer block
            (default 4, must divide *hidden_dim*).
    """

    def __init__(
        self,
        hidden_dim: int,
        eagle_k: int = 8,
        draft_len: int = 4,
        eagle_accept_threshold: float = 0.65,
        ema_alpha: float = 0.05,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eagle_k = eagle_k
        self.draft_len = draft_len
        self.eagle_accept_threshold = eagle_accept_threshold
        self.ema_alpha = ema_alpha

        # Acceptance rate EMA — initialised to 0, updated by the trainer
        self.accept_rate_ema: float = 0.0

        # Ensure nhead divides hidden_dim
        while hidden_dim % nhead != 0 and nhead > 1:
            nhead //= 2
        self._nhead = nhead

        # Single transformer block (attention + FFN)
        self._attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=nhead,
            batch_first=True,
        )
        self._ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self._ln1 = nn.LayerNorm(hidden_dim)
        self._ln2 = nn.LayerNorm(hidden_dim)

        # Branch projection: map the summarised state to K draft seeds
        self._branch_proj = nn.Linear(hidden_dim, eagle_k * hidden_dim)

        # Draft token projection: hidden_dim → draft_len (index predictions)
        # Returns logits over the draft position; the caller resolves to tokens.
        self._draft_proj = nn.Linear(hidden_dim, draft_len)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_draft_tree(self, Z: Tensor) -> Tensor:
        """Generate a draft token tree from the PCG fixed-point states.

        Args:
            Z: PCG node states of shape ``(B, num_blocks, hidden_dim)``.

        Returns:
            Draft tree tensor of shape ``(B, K, draft_len)`` containing
            draft token index predictions (integer logit argmaxes).
        """
        B, num_blocks, hidden_dim = Z.shape

        # --- Transformer block on Z ---
        # Self-attention
        Z_ln = self._ln1(Z)
        attn_out, _ = self._attn(Z_ln, Z_ln, Z_ln)
        Z = Z + attn_out

        # FFN
        Z = Z + self._ffn(self._ln2(Z))

        # Summary token: mean-pool over blocks → (B, hidden_dim)
        summary = Z.mean(dim=1)

        # Project to K branches: (B, K, hidden_dim)
        branches = self._branch_proj(summary)  # (B, K * hidden_dim)
        branches = branches.reshape(B, self.eagle_k, hidden_dim)

        # For each branch, produce draft_len token index predictions
        # (B, K, draft_len)  via a linear projection + argmax
        draft_logits = self._draft_proj(branches)  # (B, K, draft_len)
        draft_tree = draft_logits.long()  # raw logit values as indices (TDD)
        return draft_tree

    def update_accept_rate_ema(self, accept_rate: float) -> None:
        """Update the exponential moving average of the draft acceptance rate.

        Args:
            accept_rate: Observed acceptance rate for the last decoding step,
                in ``[0, 1]``.
        """
        self.accept_rate_ema = (
            self.ema_alpha * accept_rate + (1.0 - self.ema_alpha) * self.accept_rate_ema
        )
        logger.debug("EAGLE accept_rate_ema updated to %.4f", self.accept_rate_ema)
        self._maybe_expand_draft_len()

    def _maybe_expand_draft_len(self) -> None:
        """Expand draft_len by 1 when accept_rate_ema exceeds the threshold.

        The expansion threshold includes a small hysteresis margin
        (``eagle_accept_threshold + 0.05``) to avoid rapid oscillation.
        Expansion is capped at ``_DRAFT_LEN_MAX`` (16).
        """
        expand_threshold = self.eagle_accept_threshold + _DRAFT_LEN_EXPAND_THRESHOLD_DELTA
        if self.accept_rate_ema >= expand_threshold and self.draft_len < _DRAFT_LEN_MAX:
            old_len = self.draft_len
            self.draft_len += 1
            logger.info(
                "EAGLE draft_len expanded %d → %d (accept_rate_ema=%.3f)",
                old_len,
                self.draft_len,
                self.accept_rate_ema,
            )
