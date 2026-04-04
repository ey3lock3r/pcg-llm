"""nGPTNorm: hyperspherical unit-norm normalization for PCG-LLM nodes.

Applied inside the DEQ f_θ function (after each message-passing step in PCGNode)
per research.md Decision 3: placement inside DEQ preserves fixed-point semantics.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch import Tensor


class nGPTNorm(nn.Module):
    """Hyperspherical unit-norm normalization (nGPT style).

    Normalizes the last dimension of the input tensor to unit norm (L2).
    When applied inside f_θ, the DEQ fixed point Z* is constrained to the unit
    sphere, which strengthens convergence via compactness (bounded dynamics).

    Under nGPT mode the Variance Hinge loss is reinterpreted as angular spread:
    variance of node states ≡ mean pairwise cosine dissimilarity.
    The threshold τ=0.1 translates to requiring mean cosine similarity < 0.9.

    Args:
        dim: Dimension along which to normalize (default: -1, i.e., last dim).
        eps: Small constant for numerical stability.
    """

    def __init__(self, dim: int = -1, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x to unit norm along self.dim.

        Args:
            x: Input tensor of any shape.

        Returns:
            Tensor of same shape with unit L2 norm along self.dim.
        """
        return cast(Tensor, x / (x.norm(dim=self.dim, keepdim=True) + self.eps))

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


class LayerNorm(nn.Module):
    """Standard LayerNorm wrapper used when normalize='standard'.

    Thin wrapper around nn.LayerNorm for uniform interface with nGPTNorm.
    """

    def __init__(self, normalized_shape: int | list[int], eps: float = 1e-5) -> None:
        super().__init__()
        self._norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(Tensor, self._norm(x))


def build_norm(normalize: str, hidden_dim: int) -> nn.Module:
    """Factory: return the appropriate norm module based on config flag.

    Args:
        normalize: 'ngpt' for hyperspherical norm, 'standard' for LayerNorm.
        hidden_dim: Feature dimension for LayerNorm shape.

    Returns:
        An nn.Module implementing the chosen normalization.

    Raises:
        ValueError: If normalize value is not recognized.
    """
    if normalize == "ngpt":
        return nGPTNorm(dim=-1)
    if normalize == "standard":
        return LayerNorm(normalized_shape=hidden_dim)
    raise ValueError(f"Unknown normalize value '{normalize}'. Choose 'ngpt' or 'standard'.")
