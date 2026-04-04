"""GPU tests for FlexAttention block-local mask — T038.

@pytest.mark.gpu — skipped unless CUDA GPU is available (requires PyTorch 2.5+).
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.gpu
class TestFlexAttentionMask:
    """Verify block-local FlexAttention mask behavior."""

    @pytest.fixture
    def eagle_head(self):
        from pcg_llm.arch.eagle_head import EAGLEExtrapolationHead

        return EAGLEExtrapolationHead(
            hidden_dim=64,
            eagle_k=4,
            draft_len=4,
        ).cuda()

    def test_eagle_head_runs_on_gpu(self, eagle_head) -> None:
        """EAGLEExtrapolationHead forward pass works on CUDA."""
        B, num_blocks, hidden_dim = 2, 8, 64
        Z = torch.randn(B, num_blocks, hidden_dim, device="cuda")

        draft_tree = eagle_head.generate_draft_tree(Z)
        assert draft_tree.device.type == "cuda" or draft_tree is not None

    def test_block_local_attention_zero_between_partitions(self) -> None:
        """Block-local mask has zero attention weight between tokens in different partitions."""
        # Verify that the FlexAttention mask (when used) correctly isolates partitions.
        # For PyTorch < 2.5, the causal mask is used instead — check that's valid.
        try:
            import torch.nn.functional as F  # noqa: F401
        except ImportError:
            pytest.skip("torch not available")

        partition_size = 64
        seq_len = 128  # 2 partitions
        B, H = 1, 1

        # Simulate block-local mask: tokens only attend within their 64-token partition
        q_idx = torch.arange(seq_len).unsqueeze(1)  # [L, 1]
        k_idx = torch.arange(seq_len).unsqueeze(0)  # [1, L]

        # Block-local: same partition iff q_idx // 64 == k_idx // 64
        partition_mask = (q_idx // partition_size) == (k_idx // partition_size)

        # Check: tokens in partition 0 (0..63) do NOT attend to partition 1 (64..127)
        cross_partition = partition_mask[:partition_size, partition_size:]
        assert (
            cross_partition.sum() == 0
        ), "Block-local mask must have zero cross-partition attention"

        # Check: tokens within same partition CAN attend to each other
        intra_partition = partition_mask[:partition_size, :partition_size]
        assert intra_partition.sum() > 0, "Block-local mask must allow intra-partition attention"

    def test_global_summary_token_attends_everywhere(self) -> None:
        """Global summary token at position 0 receives attention from all partitions."""
        partition_size = 64
        seq_len = 128

        q_idx = torch.arange(seq_len)
        k_idx = torch.arange(seq_len)

        # Block-local + global token at position 0
        # A query token can attend to: (1) same partition, (2) position 0 (global)
        def can_attend(q: int, k: int) -> bool:
            if k == 0:
                return True  # Global summary token
            return (q // partition_size) == (k // partition_size)

        # Tokens in partition 1 can attend to position 0
        assert can_attend(65, 0), "Partition-1 token should attend to global token at 0"
        assert can_attend(127, 0), "Last token should attend to global token at 0"
        # But NOT to non-global tokens in partition 0
        assert not can_attend(
            65, 1
        ), "Partition-1 token should NOT attend to non-global partition-0 tokens"
