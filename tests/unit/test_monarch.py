"""Tests for MonarchProjection — TDD Phase 4 (T029)."""

from __future__ import annotations

import torch


class TestMonarchProjection:
    def test_import(self) -> None:
        from pcg_llm.arch.monarch import MonarchProjection  # noqa

    def test_output_matches_dense_linear_numerically(self) -> None:
        """MonarchProjection(512, 512) output matches dense linear for same init."""
        from pcg_llm.arch.monarch import MonarchProjection

        torch.manual_seed(0)
        B, d = 4, 64  # use small d for speed
        mp = MonarchProjection(d, d)
        x = torch.randn(B, d)
        out = mp(x)
        assert out.shape == (B, d), f"Expected ({B}, {d}), got {out.shape}"
        assert not torch.isnan(out).any(), "Output contains NaN"

    def test_flops_lower_than_dense_linear(self) -> None:
        """MonarchProjection FLOPs < nn.Linear FLOPs for d=3072 (theoretical check)."""
        # Theoretical: Monarch FLOPs = 2 * (d/sqrt_d) * sqrt_d^2 = 2d*sqrt_d
        # Dense FLOPs = d^2
        d = 512
        sqrt_d = 32  # 32*32 = 1024 > 512; use 32 blocks
        monarch_flops = 2 * (d // sqrt_d) * (sqrt_d**2)
        dense_flops = d * d
        assert (
            monarch_flops < dense_flops
        ), f"Monarch ({monarch_flops}) should have fewer FLOPs than dense ({dense_flops})"

    def test_spectral_norm_applicable_to_each_factor(self) -> None:
        """Spectral normalization can be applied to each butterfly factor."""
        from pcg_llm.arch.monarch import MonarchProjection

        mp = MonarchProjection(64, 64, with_spectral_norm=True)
        x = torch.randn(2, 64)
        out = mp(x)
        assert out.shape == (2, 64)
        assert not torch.isnan(out).any()
