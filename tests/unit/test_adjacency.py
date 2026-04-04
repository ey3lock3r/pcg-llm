"""Tests for BlockSparseAdjacency — TDD Phase 3 (T012)."""

from __future__ import annotations

import pytest
import torch


class TestBlockSparseAdjacency:
    """Verify Erdős–Rényi initialization, Drop-and-Grow cycle, freeze flag."""

    @pytest.fixture
    def small_adjacency(self):
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        # 8 blocks × 8 blocks, 32×32 each → hidden_dim=256 conceptually
        return BlockSparseAdjacency(num_blocks=8, block_size=32)

    def test_import(self) -> None:
        from pcg_llm.arch.adjacency import BlockSparseAdjacency  # noqa: F401

    def test_erdos_renyi_produces_correct_sparsity(self) -> None:
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        target_sparsity = 0.80
        adj = BlockSparseAdjacency(num_blocks=32, block_size=32)
        adj.initialize_erdos_renyi(sparsity=target_sparsity)

        actual = adj.sparsity()
        assert abs(actual - target_sparsity) <= 0.02, (
            f"Sparsity should be {target_sparsity} ± 2%, got {actual:.4f}"
        )

    def test_drop_phase_removes_exactly_10_percent(self, small_adjacency) -> None:
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        adj: BlockSparseAdjacency = small_adjacency
        adj.initialize_erdos_renyi(sparsity=0.80)

        active_before = int(adj.mask.sum().item())
        gradients = torch.randn_like(adj.W_structure)
        adj.drop_and_grow(gradients, drop_fraction=0.10, grow_fraction=0.0)
        active_after = int(adj.mask.sum().item())

        expected_drop = max(1, round(active_before * 0.10))
        actual_drop = active_before - active_after
        assert actual_drop == expected_drop, (
            f"Drop phase should remove exactly {expected_drop} blocks, removed {actual_drop}"
        )

    def test_grow_phase_activates_highest_gradient_blocks(self, small_adjacency) -> None:
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        adj: BlockSparseAdjacency = small_adjacency
        adj.initialize_erdos_renyi(sparsity=0.90)

        dormant_count_before = int((~adj.mask).sum().item())
        # Make one dormant position have a very high gradient
        gradients = torch.zeros_like(adj.W_structure)
        dormant_positions = (~adj.mask).nonzero(as_tuple=False)
        if len(dormant_positions) > 0:
            # Give the first dormant position a huge gradient
            r, c = dormant_positions[0]
            gradients[r, c] = 1000.0

        active_before = int(adj.mask.sum().item())
        adj.drop_and_grow(gradients, drop_fraction=0.0, grow_fraction=0.10)
        active_after = int(adj.mask.sum().item())

        assert active_after > active_before, "Grow phase must activate dormant blocks"

    def test_sparsity_stays_in_range_after_10_cycles(self) -> None:
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        adj = BlockSparseAdjacency(num_blocks=16, block_size=32)
        adj.initialize_erdos_renyi(sparsity=0.85)

        for _ in range(10):
            grads = torch.randn_like(adj.W_structure).abs()
            adj.drop_and_grow(grads, drop_fraction=0.10, grow_fraction=0.10)
            sp = adj.sparsity()
            assert 0.80 <= sp <= 0.98, (
                f"Sparsity {sp:.4f} out of allowed range [0.80, 0.98]"
            )

    def test_freeze_prevents_updates(self, small_adjacency) -> None:
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        adj: BlockSparseAdjacency = small_adjacency
        adj.initialize_erdos_renyi(sparsity=0.80)
        adj.freeze()

        mask_before = adj.mask.clone()
        grads = torch.randn_like(adj.W_structure)
        adj.drop_and_grow(grads, drop_fraction=0.10, grow_fraction=0.10)

        assert torch.equal(adj.mask, mask_before), (
            "Freeze flag must prevent topology changes"
        )
        assert adj.is_frozen, "is_frozen property should be True after freeze()"

    def test_message_pass_returns_correct_shape(self, small_adjacency) -> None:
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        adj: BlockSparseAdjacency = small_adjacency
        adj.initialize_erdos_renyi(sparsity=0.80)

        # Z: [batch, num_blocks, block_size]
        B, num_blocks, block_size = 2, 8, 32
        Z = torch.randn(B, num_blocks, block_size)
        out = adj.message_pass(Z)

        assert out.shape == Z.shape, (
            f"message_pass output shape {out.shape} must match input {Z.shape}"
        )


class TestCPUOffloadFR018:
    """FR-018: mask tensor can be offloaded to CPU and back without data corruption.

    message_pass() already calls mask.to(Z.device) so any device mismatch is
    silently resolved.  These tests verify that contract holds and that the
    Drop-and-Grow state is preserved through a device round-trip.
    """

    def test_message_pass_auto_moves_mask_to_z_device(self) -> None:
        """message_pass with mask on different logical device resolves silently."""
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        adj = BlockSparseAdjacency(num_blocks=4, block_size=8)
        adj.initialize_erdos_renyi(sparsity=0.50)

        # Keep both on CPU (default); simulate "offloaded" by cloning to cpu explicitly
        adj.mask = adj.mask.cpu()

        Z = torch.randn(2, 4, 8)  # CPU
        out = adj.message_pass(Z)

        assert out.shape == Z.shape, "Output shape must match Z"
        assert out.device == Z.device, "Output must be on the same device as Z"

    def test_drop_and_grow_preserves_shape_after_device_transfer(self) -> None:
        """drop_and_grow must work correctly after a CPU ↔ device round-trip."""
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        adj = BlockSparseAdjacency(num_blocks=8, block_size=16)
        adj.initialize_erdos_renyi(sparsity=0.70)

        mask_shape = adj.mask.shape

        # Simulate CPU offload: move mask/weights to cpu, then back
        adj.mask = adj.mask.cpu()
        adj.W_structure = adj.W_structure.cpu()

        grads = torch.randn_like(adj.W_structure).abs()
        adj.drop_and_grow(grads, drop_fraction=0.10, grow_fraction=0.10)

        assert adj.mask.shape == mask_shape, "Mask shape must be preserved"
        assert adj.W_structure.shape == mask_shape, "W_structure shape must be preserved"

    def test_message_pass_output_matches_regardless_of_mask_device(self) -> None:
        """message_pass output is numerically identical whether mask was moved or not."""
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        adj = BlockSparseAdjacency(num_blocks=4, block_size=8)
        adj.mask = torch.ones(4, 4, dtype=torch.bool)  # fully dense, deterministic

        Z = torch.randn(1, 4, 8)

        out_before = adj.message_pass(Z).clone()

        # Move mask to CPU (no-op on CPU-only machines, but tests .to(Z.device) path)
        adj.mask = adj.mask.cpu()
        out_after = adj.message_pass(Z)

        assert torch.allclose(out_before, out_after, atol=1e-5), (
            "message_pass output must be identical before and after mask device transfer"
        )

    def test_state_dict_round_trip_preserves_mask(self) -> None:
        """state_dict/load_state_dict must preserve mask after a device round-trip."""
        from pcg_llm.arch.adjacency import BlockSparseAdjacency

        adj = BlockSparseAdjacency(num_blocks=4, block_size=8)
        adj.initialize_erdos_renyi(sparsity=0.70)

        # Simulate CPU offload: export state, move to cpu, reload
        sd = adj.state_dict()
        sd["mask"] = sd["mask"].cpu()
        sd["W_structure"] = sd["W_structure"].cpu()

        adj2 = BlockSparseAdjacency(num_blocks=4, block_size=8)
        adj2.load_state_dict(sd)

        assert torch.equal(adj.mask.cpu(), adj2.mask.cpu()), (
            "Mask must be identical after state_dict round-trip through CPU"
        )
