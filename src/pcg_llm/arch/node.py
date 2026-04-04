"""PCGNode — latent state estimator for the PCG-LLM architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils
from torch import Tensor


class PCGNode(nn.Module):
    """Latent state estimator for a single PCG node.

    Applies a spectrally-normalised linear projection followed by tanh
    activation to produce an updated latent state, and also returns the
    prediction error (residual) between the previous and updated states.

    The weight matrix ``self.W`` has spectral normalisation registered via
    :func:`torch.nn.utils.spectral_norm` so that the Lipschitz constant of the
    linear map is bounded by 1, which guarantees the DEQ contraction condition.

    When the input ``z`` is in BF16 the forward pass casts the linear layer
    output back to BF16 to preserve dtype consistency throughout the network.

    Args:
        hidden_dim: Dimensionality of the node's latent state vector.
        normalize: Normalization variant (currently ``"standard"`` is supported;
            reserved for future nGPT variants).
    """

    def __init__(self, hidden_dim: int, normalize: str = "standard") -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.normalize = normalize

        # Register spectral norm on the linear layer; bias=False per spec.
        linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W: nn.Module = nn.utils.spectral_norm(linear)

    def forward(self, z: Tensor, neighbors: Tensor) -> tuple[Tensor, Tensor]:
        """Compute the updated latent state and prediction error.

        The updated state is computed as::

            updated_state = tanh(W_sn @ (z + neighbors))

        where ``W_sn`` is the spectrally-normalised version of ``self.W``.
        The prediction error is the residual::

            e_i = z - updated_state

        Args:
            z: Current latent state of shape ``[..., hidden_dim]``.
            neighbors: Aggregated neighbour messages of shape matching *z*.

        Returns:
            A tuple ``(updated_state, prediction_error_e)`` both with the same
            shape as *z*.
        """
        input_dtype = z.dtype
        combined = z + neighbors

        # Cast to float32 for the linear layer if needed, then restore dtype.
        if input_dtype == torch.bfloat16:
            combined = combined.to(torch.float32)

        updated = torch.tanh(self.W(combined))

        if input_dtype == torch.bfloat16:
            updated = updated.to(torch.bfloat16)

        prediction_error_e = z - updated
        return updated, prediction_error_e
