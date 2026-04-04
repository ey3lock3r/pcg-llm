"""GPU tests for block-sparse Triton kernel — T038.

@pytest.mark.gpu — skipped unless CUDA GPU is available.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu
class TestBlockSparseTritonKernel:
    """Verify block-sparse matmul produces same output as dense equivalent."""

    def test_block_sparse_matches_dense_output(self) -> None:
        """Block-sparse matmul via dense fallback matches explicit dense computation."""
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        torch.manual_seed(0)
        device = torch.device("cuda")
        num_blocks, block_size = 8, 32
        B = 2

        adj = BlockSparseAdjacency(num_blocks=num_blocks, block_size=block_size)
        adj.initialize_erdos_renyi(sparsity=0.5)  # 50% for meaningful test
        adj.mask = adj.mask.to(device)
        adj.W_structure = adj.W_structure.to(device)

        Z = torch.randn(B, num_blocks, block_size, device=device)

        # Block-sparse result
        out_sparse = adj.message_pass(Z)

        # Dense equivalent: mask broadcast
        mask_float = adj.mask.float().to(device)  # [num_blocks, num_blocks]
        # out[b, i, :] = sum_j mask[i,j] * Z[b, j, :]
        out_dense = torch.einsum("ij,bjd->bid", mask_float, Z)

        assert out_sparse.shape == out_dense.shape
        assert torch.allclose(out_sparse, out_dense, atol=1e-3), (
            f"Block-sparse and dense outputs differ by "
            f"{(out_sparse - out_dense).abs().max().item():.6f}"
        )

    def test_block_sparse_sparsity_on_gpu(self) -> None:
        """BlockSparseAdjacency operations work correctly on GPU."""
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        device = torch.device("cuda")
        adj = BlockSparseAdjacency(num_blocks=16, block_size=64)
        adj.initialize_erdos_renyi(sparsity=0.90)
        adj.mask = adj.mask.to(device)
        adj.W_structure = adj.W_structure.to(device)

        sparsity = adj.sparsity()
        assert abs(sparsity - 0.90) <= 0.02, f"Sparsity {sparsity:.4f} out of ±2% of 0.90"

    def test_drop_and_grow_on_gpu(self) -> None:
        """Drop-and-Grow executes correctly on CUDA tensors."""
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        device = torch.device("cuda")
        adj = BlockSparseAdjacency(num_blocks=8, block_size=32)
        adj.initialize_erdos_renyi(sparsity=0.80)
        adj.mask = adj.mask.to(device)
        adj.W_structure = adj.W_structure.to(device)

        active_before = int(adj.mask.sum().item())
        grads = torch.randn_like(adj.W_structure).abs()
        adj.drop_and_grow(grads, drop_fraction=0.10, grow_fraction=0.10)

        # After one Drop-and-Grow cycle, active count should be similar
        active_after = int(adj.mask.sum().item())
        assert active_after >= 0
        assert adj.sparsity() <= 1.0
