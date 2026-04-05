"""PCGNode — latent state estimator for the PCG-LLM architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.utils
from torch import Tensor

from pcg_llm.training.normalization import build_norm


class PCGNode(nn.Module):
    """Latent state estimator for a single PCG node.

    Applies a spectrally-normalised linear projection followed by tanh
    activation and an optional normalisation layer to produce an updated
    latent state.  When ``normalize='ngpt'`` the output is projected onto
    the unit hypersphere (nGPT style), which keeps node states bounded and
    ensures the DEQ fixed point lies on a compact manifold — preventing
    tanh saturation and vanishing gradients.  When ``normalize='standard'``
    a LayerNorm is applied instead.

    The weight matrix ``self.W`` has spectral normalisation registered via
    :func:`torch.nn.utils.spectral_norm` so that the Lipschitz constant of the
    linear map is bounded by 1, which guarantees the DEQ contraction condition.

    When the input ``z`` is in BF16 the forward pass casts the linear layer
    output back to BF16 to preserve dtype consistency throughout the network.

    Args:
        hidden_dim: Dimensionality of the node's latent state vector.
        normalize: ``'ngpt'`` for unit-sphere normalisation (recommended);
            ``'standard'`` for LayerNorm.
    """

    def __init__(self, hidden_dim: int, normalize: str = "standard") -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.normalize = normalize

        # Register spectral norm on the linear layer; bias=False per spec.
        linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W: nn.Module = nn.utils.spectral_norm(linear)

        # Normalisation applied after tanh — nGPT projects onto unit sphere,
        # standard applies LayerNorm.  Applied inside f_θ per research.md.
        self._norm: nn.Module = build_norm(normalize, hidden_dim)

    def forward(
        self,
        z: Tensor,
        neighbors: Tensor,
        apply_norm: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Compute the updated latent state and prediction error.

        The updated state is computed as::

            updated_state = norm(tanh(W_sn @ (z + neighbors)))

        where ``W_sn`` is the spectrally-normalised version of ``self.W``
        and ``norm`` is the configured normalisation (nGPT or LayerNorm).
        The prediction error is the residual::

            e_i = z - updated_state

        Args:
            z: Current latent state of shape ``[..., hidden_dim]``.
            neighbors: Aggregated neighbour messages of shape matching *z*.
            apply_norm: Whether to apply the post-tanh normalisation.  Set to
                ``False`` during DEQ solver iterations so that the Anderson
                mixing works in Euclidean space (nGPT maps iterates off the
                unit sphere, preventing convergence detection). Apply it once
                outside the solver on the final fixed point.

        Returns:
            A tuple ``(updated_state, prediction_error_e)`` both with the same
            shape as *z*.
        """
        input_dtype = z.dtype
        combined = z + neighbors

        # Cast to float32 for the linear layer if needed, then restore dtype.
        if input_dtype == torch.bfloat16:
            combined = combined.to(torch.float32)

        pre_norm = torch.tanh(self.W(combined))

        # Apply normalisation after tanh to keep states bounded.
        # Skipped during DEQ iterations (apply_norm=False) so that Anderson
        # mixing operates in plain Euclidean space; applied once post-solver.
        updated = self._norm(pre_norm) if apply_norm else pre_norm

        if input_dtype == torch.bfloat16:
            updated = updated.to(torch.bfloat16)

        prediction_error_e = z - updated
        return updated, prediction_error_e
