"""MonarchProjection — structured butterfly (Monarch) matrix multiplication.

Monarch matrices replace dense d×d projections with two chained butterfly
factor matrices, reducing FLOPs from O(d²) to O(d·√d) while retaining
expressiveness close to a full-rank projection.

Architecture:
    For an input dimension *d_in* and output dimension *d_out*, we decompose
    the projection into two butterfly factors:
        F1: (d_in // block_size, block_size, block_size)
        F2: (d_out // block_size, block_size, block_size)
    applied sequentially with an intermediate reshape.

The block size is chosen as ``max(8, int(d_in ** 0.5))`` rounded to the
nearest power of 2, subject to the constraint that both *d_in* and *d_out*
are divisible by the block size.

Spectral normalisation can optionally be applied to each factor independently
via ``with_spectral_norm=True``.

References:
    Dao et al., "Monarch: Expressive Structured Matrices for Efficient and
    Accurate Training", ICML 2022.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _nearest_pow2(n: int) -> int:
    """Return the largest power of 2 that is <= *n*."""
    if n <= 1:
        return 1
    return 2 ** int(math.log2(n))


def _choose_block_size(d_in: int, d_out: int) -> int:
    """Choose a butterfly block size that divides both *d_in* and *d_out*."""
    # Start with sqrt(d_in) rounded to nearest power-of-2, minimum 8
    candidate = max(8, _nearest_pow2(int(math.isqrt(d_in))))
    # Walk down until we find a divisor of both dims
    while candidate >= 2:
        if d_in % candidate == 0 and d_out % candidate == 0:
            return candidate
        candidate //= 2
    # Fallback: block size 1 always divides; degenerate but valid
    return 1


class MonarchProjection(nn.Module):
    """Structured Monarch matrix projection replacing a dense nn.Linear.

    Args:
        d_in: Input feature dimension.
        d_out: Output feature dimension.
        with_spectral_norm: If ``True``, apply spectral normalisation to
            each butterfly factor independently (default ``False``).

    Raises:
        ValueError: if *d_in* or *d_out* is not divisible by the chosen
            block size (should not occur given ``_choose_block_size``).
    """

    def __init__(
        self,
        d_in: int,
        d_out: int,
        with_spectral_norm: bool = False,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.block_size = _choose_block_size(d_in, d_out)
        self.with_spectral_norm = with_spectral_norm

        b = self.block_size
        n_in = d_in // b   # number of input blocks
        n_out = d_out // b  # number of output blocks

        # Factor 1: n_in independent (b × b) matrices operating on the input
        # Factor 2: n_out independent (b × b) matrices producing the output
        #
        # We store each factor as a (num_blocks, b, b) parameter tensor.
        # The forward pass applies them with batched matmul.

        f1_weight = torch.empty(n_in, b, b)
        f2_weight = torch.empty(n_out, b, b)
        nn.init.orthogonal_(f1_weight.view(n_in * b, b))
        nn.init.orthogonal_(f2_weight.view(n_out * b, b))

        if with_spectral_norm:
            # Apply spectral norm by treating each (b, b) block as a linear map.
            # We reshape to 2-D, apply spectral_norm via a helper Linear, then
            # store the result as a plain parameter (spectral norm is re-applied
            # each forward pass via the SN wrapper linears).
            self._f1_linears = nn.ModuleList([
                nn.utils.spectral_norm(nn.Linear(b, b, bias=False))
                for _ in range(n_in)
            ])
            self._f2_linears = nn.ModuleList([
                nn.utils.spectral_norm(nn.Linear(b, b, bias=False))
                for _ in range(n_out)
            ])
            # Initialise their weights from our orthogonal init
            with torch.no_grad():
                for i, lin in enumerate(self._f1_linears):
                    lin.weight.copy_(f1_weight[i])
                for i, lin in enumerate(self._f2_linears):
                    lin.weight.copy_(f2_weight[i])
            self.factor1: nn.Parameter | None = None
            self.factor2: nn.Parameter | None = None
        else:
            self.factor1 = nn.Parameter(f1_weight)
            self.factor2 = nn.Parameter(f2_weight)
            self._f1_linears = nn.ModuleList()
            self._f2_linears = nn.ModuleList()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Apply the Monarch projection to *x*.

        Args:
            x: Input tensor of shape ``(..., d_in)``.

        Returns:
            Output tensor of shape ``(..., d_out)``.
        """
        batch_shape = x.shape[:-1]
        b = self.block_size
        n_in = self.d_in // b
        n_out = self.d_out // b

        # Reshape: (..., d_in) → (..., n_in, b)
        h = x.reshape(*batch_shape, n_in, b)

        if self.with_spectral_norm:
            # Apply each block's spectrally-normalised linear independently
            # h: (..., n_in, b) → (..., n_in, b)
            out_blocks = []
            for i, lin in enumerate(self._f1_linears):
                # Extract block i: (..., b)
                block = h[..., i, :]  # shape: (*batch_shape, b)
                out_blocks.append(lin(block))
            # Stack back: (*batch_shape, n_in, b)
            h = torch.stack(out_blocks, dim=-2)
        else:
            # Batched matmul: factor1 has shape (n_in, b, b)
            # h: (..., n_in, b) → (..., n_in, b) via h @ factor1.T per block
            # Expand batch dims for broadcasting
            h = torch.einsum("...ib,ibc->...ic", h, self.factor1)  # type: ignore[arg-type]

        # Reshape intermediate: (..., n_in, b) → (..., n_out, b)
        # This is the "butterfly" permutation — flatten then re-block
        h = h.reshape(*batch_shape, self.d_in)
        # Pad or truncate if d_in != d_out
        if self.d_in != self.d_out:
            if self.d_in < self.d_out:
                pad_size = self.d_out - self.d_in
                h = F.pad(h, (0, pad_size))
            else:
                h = h[..., : self.d_out]
        h = h.reshape(*batch_shape, n_out, b)

        if self.with_spectral_norm:
            out_blocks = []
            for i, lin in enumerate(self._f2_linears):
                block = h[..., i, :]
                out_blocks.append(lin(block))
            h = torch.stack(out_blocks, dim=-2)
        else:
            h = torch.einsum("...ib,ibc->...ic", h, self.factor2)  # type: ignore[arg-type]

        # Reshape: (..., n_out, b) → (..., d_out)
        out = h.reshape(*batch_shape, self.d_out)
        return out
