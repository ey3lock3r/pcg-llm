"""Tests for normalization modules: nGPTNorm, LayerNorm, build_norm."""

from __future__ import annotations

import pytest
import torch


class TestNGPTNorm:
    """Tests for nGPTNorm hyperspherical unit-norm normalization."""

    def test_import(self) -> None:
        from pcg_llm.training.normalization import nGPTNorm  # noqa: F401

    def test_output_is_unit_norm(self) -> None:
        """Output along last dim should have L2 norm == 1.0."""
        from pcg_llm.training.normalization import nGPTNorm

        norm = nGPTNorm()
        x = torch.randn(4, 8, 64)
        y = norm(x)
        norms = y.norm(dim=-1)
        assert torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5
        ), f"Expected unit norms, got min={norms.min().item():.6f} max={norms.max().item():.6f}"

    def test_output_shape_unchanged(self) -> None:
        """Output shape must match input shape."""
        from pcg_llm.training.normalization import nGPTNorm

        norm = nGPTNorm()
        x = torch.randn(3, 16)
        assert norm(x).shape == x.shape

    def test_zero_vector_no_nan(self) -> None:
        """A zero vector must not produce NaN; norm must stay finite."""
        from pcg_llm.training.normalization import nGPTNorm

        norm = nGPTNorm(eps=1e-6)
        x = torch.zeros(1, 64)
        y = norm(x)
        assert not torch.isnan(y).any(), "nGPTNorm produced NaN for zero input"
        assert torch.isfinite(y).all(), "nGPTNorm produced non-finite value for zero input"

    def test_zero_vector_norm_finite(self) -> None:
        """Output norm for zero vector must be finite (not inf, not NaN)."""
        from pcg_llm.training.normalization import nGPTNorm

        norm = nGPTNorm(eps=1e-6)
        x = torch.zeros(2, 32)
        y = norm(x)
        out_norms = y.norm(dim=-1)
        assert torch.isfinite(out_norms).all()

    def test_custom_dim(self) -> None:
        """nGPTNorm with dim=1 should normalise along dim=1."""
        from pcg_llm.training.normalization import nGPTNorm

        norm = nGPTNorm(dim=1)
        x = torch.randn(8, 16)
        y = norm(x)
        norms = y.norm(dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_extra_repr(self) -> None:
        """extra_repr should include dim and eps values."""
        from pcg_llm.training.normalization import nGPTNorm

        norm = nGPTNorm(dim=-1, eps=1e-6)
        repr_str = norm.extra_repr()
        assert "dim=-1" in repr_str
        assert "eps=1e-06" in repr_str or "eps=1e-6" in repr_str


class TestLayerNorm:
    """Tests for the LayerNorm wrapper."""

    def test_import(self) -> None:
        from pcg_llm.training.normalization import LayerNorm  # noqa: F401

    def test_forward_runs_without_error(self) -> None:
        """LayerNorm forward pass should complete without raising."""
        from pcg_llm.training.normalization import LayerNorm

        ln = LayerNorm(normalized_shape=64)
        x = torch.randn(4, 64)
        y = ln(x)
        assert y is not None

    def test_output_shape_matches_input(self) -> None:
        """Output tensor shape must match input tensor shape."""
        from pcg_llm.training.normalization import LayerNorm

        ln = LayerNorm(normalized_shape=32)
        x = torch.randn(2, 8, 32)
        assert ln(x).shape == x.shape

    def test_output_is_normalized(self) -> None:
        """LayerNorm output should have mean ≈ 0 and std ≈ 1 over normalized dim."""
        from pcg_llm.training.normalization import LayerNorm

        torch.manual_seed(42)
        ln = LayerNorm(normalized_shape=128)
        x = torch.randn(8, 128) * 10.0 + 5.0
        y = ln(x)
        # After LayerNorm the mean of each sample row should be near 0
        means = y.mean(dim=-1)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-4)


class TestBuildNorm:
    """Tests for the build_norm factory function."""

    def test_import(self) -> None:
        from pcg_llm.training.normalization import build_norm  # noqa: F401

    def test_build_norm_ngpt_returns_ngptnorm(self) -> None:
        """build_norm('ngpt', 64) should return an nGPTNorm instance."""
        from pcg_llm.training.normalization import build_norm, nGPTNorm

        module = build_norm("ngpt", 64)
        assert isinstance(module, nGPTNorm)

    def test_build_norm_standard_returns_layernorm(self) -> None:
        """build_norm('standard', 64) should return a LayerNorm instance."""
        from pcg_llm.training.normalization import LayerNorm, build_norm

        module = build_norm("standard", 64)
        assert isinstance(module, LayerNorm)

    def test_build_norm_invalid_raises_value_error(self) -> None:
        """build_norm with an unrecognised key should raise ValueError."""
        from pcg_llm.training.normalization import build_norm

        with pytest.raises(ValueError, match="invalid"):
            build_norm("invalid", 64)

    def test_build_norm_empty_string_raises_value_error(self) -> None:
        """build_norm with empty string should raise ValueError."""
        from pcg_llm.training.normalization import build_norm

        with pytest.raises(ValueError):
            build_norm("", 64)

    def test_build_norm_ngpt_is_callable_module(self) -> None:
        """The returned nGPTNorm module should be usable as a callable."""
        import torch.nn as nn

        from pcg_llm.training.normalization import build_norm

        module = build_norm("ngpt", 64)
        assert isinstance(module, nn.Module)
        x = torch.randn(2, 64)
        y = module(x)
        assert y.shape == x.shape

    def test_build_norm_standard_is_callable_module(self) -> None:
        """The returned LayerNorm module should be usable as a callable."""
        import torch.nn as nn

        from pcg_llm.training.normalization import build_norm

        module = build_norm("standard", 64)
        assert isinstance(module, nn.Module)
        x = torch.randn(2, 64)
        y = module(x)
        assert y.shape == x.shape
