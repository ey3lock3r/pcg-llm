"""Block-sparse adjacency matrix for the PCG-LLM graph structure."""

from __future__ import annotations

import torch
from torch import Tensor


class BlockSparseAdjacency:
    """Block-sparse adjacency matrix with RigL-style drop-and-grow support.

    The adjacency is stored as a boolean *mask* (which edges are active) and a
    float32 weight matrix *W_structure* (the trainable edge strengths).  The
    granularity is at the *block* level: each entry ``[i, j]`` in the matrices
    corresponds to a full ``block_size × block_size`` dense sub-matrix.

    This avoids ever instantiating a dense ``(num_blocks * block_size)²``
    adjacency tensor, which is crucial for 3B-parameter models where
    ``num_blocks`` can be in the hundreds.

    Args:
        num_blocks: Number of block-rows (and columns) in the adjacency.
        block_size: Size of each square block (e.g. 64 for 3B, 32 for Tiny).
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size

        # Boolean activity mask — False means the edge is inactive (pruned).
        self.mask: Tensor = torch.zeros(num_blocks, num_blocks, dtype=torch.bool)
        # Float32 structural weights — zero for inactive edges.
        self.W_structure: Tensor = torch.zeros(num_blocks, num_blocks, dtype=torch.float32)
        self._frozen: bool = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_erdos_renyi(self, sparsity: float) -> None:
        """Randomly initialise the adjacency with an Erdős–Rényi graph.

        Each block-edge is independently activated with probability
        ``1 - sparsity``.  Active edges receive i.i.d. Normal weights;
        inactive edges are forced to zero.

        Args:
            sparsity: Fraction of entries that should be *inactive* (False).
                Must be in ``[0, 1)``.
        """
        if not 0.0 <= sparsity < 1.0:
            raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
        # Each entry is True (active) with probability (1 - sparsity)
        rand = torch.rand(self.num_blocks, self.num_blocks)
        self.mask = rand >= sparsity
        weights = torch.randn(self.num_blocks, self.num_blocks)
        self.W_structure = torch.where(self.mask, weights, torch.zeros_like(weights))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def sparsity(self) -> float:
        """Fraction of block-edges that are currently *inactive*.

        Returns:
            A float in ``[0, 1]``.
        """
        total = self.mask.numel()
        if total == 0:
            return 0.0
        inactive = (~self.mask).sum().item()
        return float(inactive) / float(total)

    @property
    def is_frozen(self) -> bool:
        """True if the adjacency topology is frozen (no further drop/grow)."""
        return self._frozen

    # ------------------------------------------------------------------
    # Message passing
    # ------------------------------------------------------------------

    def message_pass(self, Z: Tensor) -> Tensor:
        """Aggregate neighbour messages for each node.

        For each batch element and each block-node ``i``, the output is the
        sum of ``Z[b, j, :]`` for all active neighbours ``j`` of ``i``::

            out[b, i, :] = Σ_{j: mask[i,j]} Z[b, j, :]

        The implementation uses a dense fallback: the mask is cast to float
        and broadcast-multiplied with Z, then summed over the neighbour
        dimension.  This avoids custom sparse CUDA kernels while remaining
        correct.

        Args:
            Z: Node state tensor of shape ``[B, num_blocks, block_size]``.

        Returns:
            Aggregated messages of the same shape as *Z*.
        """
        # mask: [num_blocks, num_blocks] → [1, num_blocks, num_blocks, 1]
        mask_f = self.mask.float().to(Z.device)  # [N, N]
        # Z: [B, N, D]; we want out[b, i, d] = sum_j mask[i,j] * Z[b, j, d]
        # Einsum: out[b, i, d] = mask[i, j] * Z[b, j, d] summed over j
        return torch.einsum("ij,bjd->bid", mask_f, Z)

    # ------------------------------------------------------------------
    # RigL drop-and-grow
    # ------------------------------------------------------------------

    @torch.no_grad()
    def drop_and_grow(
        self,
        gradients: Tensor,
        drop_fraction: float = 0.10,
        grow_fraction: float = 0.10,
    ) -> None:
        """Prune low-magnitude edges and activate high-gradient dormant edges.

        If the adjacency is frozen this method returns immediately without
        making any changes.

        **Drop phase**: Among currently-active block-edges, the
        ``round(active * drop_fraction)`` edges with the smallest
        ``|W_structure|`` are deactivated (mask set to False, weight set to 0).

        **Grow phase**: Among currently-inactive block-edges, the
        ``round(dormant * grow_fraction)`` edges with the largest
        ``|gradients|`` are activated (mask set to True, weight initialised to
        the corresponding gradient value to provide a warm start).

        Args:
            gradients: Gradient magnitude tensor of shape
                ``[num_blocks, num_blocks]`` — typically the gradient of the
                loss w.r.t. ``W_structure`` for dormant connections.
            drop_fraction: Fraction of active edges to drop.
            grow_fraction: Fraction of dormant edges to activate.
        """
        if self._frozen:
            return

        # ---- Drop phase ----
        active_mask = self.mask  # [N, N] bool
        active_indices = active_mask.nonzero(as_tuple=False)  # [num_active, 2]
        num_active = active_indices.shape[0]

        n_drop = 0
        if num_active > 0:
            n_drop = round(num_active * drop_fraction)
            if n_drop > 0:
                active_weights = self.W_structure[active_mask]  # [num_active]
                # Sort ascending by absolute weight; drop the smallest
                sorted_idx = active_weights.abs().argsort()
                drop_idx = sorted_idx[:n_drop]
                drop_coords = active_indices[drop_idx]
                self.mask[drop_coords[:, 0], drop_coords[:, 1]] = False
                self.W_structure[drop_coords[:, 0], drop_coords[:, 1]] = 0.0

        # ---- Grow phase ----
        dormant_mask = ~self.mask  # [N, N] bool
        dormant_indices = dormant_mask.nonzero(as_tuple=False)  # [num_dormant, 2]
        num_dormant = dormant_indices.shape[0]

        if num_dormant > 0:
            n_grow = round(num_dormant * grow_fraction)
            # Cap grow to enforce minimum 80% sparsity (active ≤ 20% of total entries)
            max_active = int(0.20 * self.num_blocks * self.num_blocks)
            active_after_drop = num_active - n_drop
            n_grow = min(n_grow, max(0, max_active - active_after_drop))
            if n_grow > 0:
                grad_flat = gradients.to(dormant_mask.device)
                dormant_grads = grad_flat[dormant_mask]  # [num_dormant]
                # Sort descending by absolute gradient; grow the largest
                sorted_idx = dormant_grads.abs().argsort(descending=True)
                grow_idx = sorted_idx[:n_grow]
                grow_coords = dormant_indices[grow_idx]
                self.mask[grow_coords[:, 0], grow_coords[:, 1]] = True
                # Initialise new weights from gradient signal
                self.W_structure[grow_coords[:, 0], grow_coords[:, 1]] = grad_flat[
                    grow_coords[:, 0], grow_coords[:, 1]
                ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Permanently lock the current topology; drop_and_grow becomes a no-op."""
        self._frozen = True

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return a serialisable snapshot of the adjacency state.

        Returns:
            Dict with keys ``"mask"``, ``"W_structure"``, and ``"frozen"``.
        """
        return {
            "mask": self.mask,
            "W_structure": self.W_structure,
            "frozen": self._frozen,
        }

    def load_state_dict(self, d: dict) -> None:
        """Restore state from a dict produced by :meth:`state_dict`.

        Args:
            d: Dict with keys ``"mask"``, ``"W_structure"``, and ``"frozen"``.
        """
        self.mask = d["mask"]
        self.W_structure = d["W_structure"]
        self._frozen = bool(d["frozen"])
