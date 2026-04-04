"""Free-energy loss: CrossEntropy + λ·||A||₁ − γ·Var(Z*)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


class FreeEnergyLoss:
    """Three-term loss combining language-modelling, sparsity, and variance objectives.

    The total loss is::

        L = CE(logits, targets)
          + λ · ||adj_weights||₁
          + γ · max(0, τ − Var(Z*))

    The third term is a *hinge penalty* that encourages the DEQ fixed-point
    variance to stay above the floor ``τ``.  Minimising the hinge is equivalent
    to maximising ``Var(Z*)`` until the floor is met.

    An adaptive schedule is applied to ``γ``: if the variance remains below
    the floor for ``gamma_nudge_threshold`` consecutive forward passes, ``γ``
    is multiplied by ``gamma_nudge_factor`` to strengthen the regularisation.

    Args:
        vocab_size: Vocabulary size (used only for documentation; CE handles it).
        lambda_sparse: Weight λ on the L1 sparsity term.
        gamma_variance: Initial weight γ on the variance hinge term.
        variance_floor: Variance floor τ below which the hinge activates.
        gamma_nudge_factor: Multiplicative factor applied to γ when the streak
            threshold is exceeded.
        gamma_nudge_threshold: Number of consecutive low-variance steps before
            γ is nudged upward.
    """

    def __init__(
        self,
        vocab_size: int,
        lambda_sparse: float = 0.01,
        gamma_variance: float = 0.01,
        variance_floor: float = 0.1,
        gamma_nudge_factor: float = 1.1,
        gamma_nudge_threshold: int = 10,
    ) -> None:
        self.vocab_size = vocab_size
        self._lambda_sparse = lambda_sparse
        self.gamma_variance: float = gamma_variance
        self.variance_floor = variance_floor
        self.gamma_nudge_factor = gamma_nudge_factor
        self.gamma_nudge_threshold = gamma_nudge_threshold
        self._low_variance_streak: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def lambda_sparse(self) -> float:
        """Weight λ on the L1 adjacency sparsity term."""
        return self._lambda_sparse

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        logits: Tensor,
        targets: Tensor,
        adj_weights: Tensor,
        Z_star: Tensor,
    ) -> tuple[Tensor, dict]:
        """Compute the three-term free-energy loss.

        Args:
            logits: Token log-probabilities of shape ``[B, vocab_size]``.
            targets: Ground-truth token indices of shape ``[B]`` (``torch.long``).
            adj_weights: Structural adjacency weights of shape ``[N, N]``
                (e.g. ``BlockSparseAdjacency.W_structure``).
            Z_star: DEQ fixed-point states of shape ``[B, N, d]``.

        Returns:
            A tuple ``(total_loss, metrics)`` where *metrics* is a dict with
            keys: ``"cross_entropy"``, ``"l1_sparsity"``, ``"variance_hinge"``,
            ``"variance"``, and ``"gamma"``.
        """
        # --- Cross-entropy ---
        ce_loss = F.cross_entropy(logits, targets)

        # --- L1 sparsity ---
        l1_loss = self._lambda_sparse * adj_weights.abs().sum()

        # --- Variance hinge ---
        # var over node and feature dimensions, then mean over batch
        var_z = Z_star.var(dim=[1, 2]).mean()
        hinge = torch.clamp(
            torch.tensor(self.variance_floor, device=Z_star.device, dtype=Z_star.dtype) - var_z,
            min=0.0,
        )
        variance_hinge_loss = self.gamma_variance * hinge

        # --- Total ---
        total = ce_loss + l1_loss + variance_hinge_loss

        # --- Adaptive γ schedule ---
        var_z_scalar = var_z.item()
        if var_z_scalar < self.variance_floor:
            self._low_variance_streak += 1
        else:
            self._low_variance_streak = 0

        if self._low_variance_streak >= self.gamma_nudge_threshold:
            self.gamma_variance = self.gamma_variance * self.gamma_nudge_factor
            self._low_variance_streak = 0

        metrics = {
            "cross_entropy": ce_loss,
            "l1_sparsity": l1_loss,
            "variance_hinge": variance_hinge_loss,
            "variance": var_z,
            "gamma": self.gamma_variance,
        }
        return total, metrics

    # Make instances directly callable like nn.Module.
    def __call__(
        self,
        logits: Tensor,
        targets: Tensor,
        adj_weights: Tensor,
        Z_star: Tensor,
    ) -> tuple[Tensor, dict]:
        """Alias for :meth:`forward`."""
        return self.forward(logits, targets, adj_weights, Z_star)
