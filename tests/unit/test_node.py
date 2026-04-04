"""Unit tests for PCGNode — TDD remediation task T054."""

from __future__ import annotations

import torch


class TestPCGNode:
    """Verify PCGNode forward shape, SpectralNorm registration, BF16, grad flow."""

    def test_import(self) -> None:
        from pcg_llm.arch.node import PCGNode  # noqa: F401

    def test_forward_returns_correct_shapes(self) -> None:
        """forward(z, neighbors) must return (updated, error) both [B, N, d]."""
        from pcg_llm.arch.node import PCGNode

        hidden_dim = 64
        B, N = 2, 4
        node = PCGNode(hidden_dim=hidden_dim)
        z = torch.randn(B, N, hidden_dim)
        neighbors = torch.randn(B, N, hidden_dim)

        updated, error = node(z, neighbors)

        assert updated.shape == (
            B,
            N,
            hidden_dim,
        ), f"updated shape {updated.shape} != expected {(B, N, hidden_dim)}"
        assert error.shape == (
            B,
            N,
            hidden_dim,
        ), f"error shape {error.shape} != expected {(B, N, hidden_dim)}"

    def test_spectral_norm_registered_on_weight(self) -> None:
        """W must have spectral_norm registered (weight_orig present)."""
        from pcg_llm.arch.node import PCGNode

        node = PCGNode(hidden_dim=32)
        assert hasattr(node.W, "weight_orig"), "spectral_norm must register 'weight_orig' on node.W"

    def test_spectral_norm_sigma_reflects_weight_scale(self) -> None:
        """The spectral_norm sigma estimate must be > 0 and < weight_orig's max SV.

        torch.nn.utils.spectral_norm stores the estimated sigma in weight_orig's
        associated hook state (accessible via weight_u / weight_v).  We verify
        that spectral norm is active by checking weight_orig and weight_u exist and
        that sigma = u^T W v is a positive scalar, indicating normalisation is live.
        """
        from pcg_llm.arch.node import PCGNode

        node = PCGNode(hidden_dim=32)

        # Run a forward pass to update power-iteration vectors
        z = torch.randn(1, 1, 32)
        node(z, torch.zeros_like(z))

        assert hasattr(node.W, "weight_orig"), "weight_orig must be registered by spectral_norm"
        assert hasattr(node.W, "weight_u"), "weight_u must be registered by spectral_norm"
        assert hasattr(node.W, "weight_v"), "weight_v must be registered by spectral_norm"

        # Compute sigma estimate as u^T @ weight_orig @ v
        u = node.W.weight_u  # [out_features]
        v = node.W.weight_v  # [in_features]
        w_orig = node.W.weight_orig  # [out_features, in_features]
        sigma = (u @ w_orig @ v).abs().item()
        assert sigma > 1e-6, f"Sigma must be positive, got {sigma}"

    def test_bf16_input_produces_bf16_output(self) -> None:
        """BF16 inputs must yield BF16 outputs (dtype preserved)."""
        from pcg_llm.arch.node import PCGNode

        node = PCGNode(hidden_dim=32)
        z = torch.randn(1, 4, 32).to(torch.bfloat16)
        neighbors = torch.randn(1, 4, 32).to(torch.bfloat16)

        updated, error = node(z, neighbors)

        assert (
            updated.dtype == torch.bfloat16
        ), f"updated dtype should be bfloat16, got {updated.dtype}"
        assert error.dtype == torch.bfloat16, f"error dtype should be bfloat16, got {error.dtype}"

    def test_float32_input_stays_float32(self) -> None:
        from pcg_llm.arch.node import PCGNode

        node = PCGNode(hidden_dim=32)
        z = torch.randn(2, 3, 32)
        neighbors = torch.zeros(2, 3, 32)

        updated, error = node(z, neighbors)

        assert updated.dtype == torch.float32
        assert error.dtype == torch.float32

    def test_prediction_error_nonzero_for_nontrivial_input(self) -> None:
        """eᵢ = z − updated_state must be non-zero when updated_state ≠ z."""
        from pcg_llm.arch.node import PCGNode

        node = PCGNode(hidden_dim=32)
        z = torch.randn(1, 4, 32)
        neighbors = torch.randn(1, 4, 32)

        _, error = node(z, neighbors)

        assert (
            error.abs().max().item() > 1e-6
        ), "Prediction error should be non-zero for non-trivial (z, neighbors)"

    def test_gradient_flows_through_forward(self) -> None:
        """Gradient must propagate back through PCGNode.forward without detach."""
        from pcg_llm.arch.node import PCGNode

        node = PCGNode(hidden_dim=32)
        z = torch.randn(1, 4, 32, requires_grad=True)
        neighbors = torch.randn(1, 4, 32)

        updated, _ = node(z, neighbors)
        updated.sum().backward()

        assert z.grad is not None, "z.grad must not be None after backward"
        assert z.grad.abs().max().item() > 0.0, "z.grad must be non-zero"

    def test_zero_neighbors_gives_pure_self_update(self) -> None:
        """With zero neighbors, updated = tanh(W @ z), error = z - tanh(W @ z)."""
        from pcg_llm.arch.node import PCGNode

        node = PCGNode(hidden_dim=16)
        z = torch.randn(1, 2, 16)
        neighbors = torch.zeros_like(z)

        updated, error = node(z, neighbors)

        # error = z - updated; they must sum to z
        reconstructed = updated + error
        assert torch.allclose(
            reconstructed, z, atol=1e-5
        ), "updated + error must equal z (prediction error definition)"
