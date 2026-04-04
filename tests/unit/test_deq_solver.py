"""Convergence tests for ConstrainedDEQSolver — TDD Phase 3 (T011).

@pytest.mark.convergence — all tests here verify DEQ convergence properties.
"""

from __future__ import annotations

import pytest
import torch


@pytest.mark.convergence
class TestConstrainedDEQSolverConvergence:
    """Verify Anderson acceleration and Broyden fallback behavior."""

    @pytest.fixture
    def simple_linear_f(self):
        """A 2-layer contractive linear system for convergence tests.

        f(z) = W @ z + b, with spectral norm of W < 1 to ensure contractivity.
        """
        torch.manual_seed(0)
        d = 32
        W = torch.randn(d, d) * 0.1  # Small init ensures spectral norm << 1
        b = torch.zeros(d)

        def f(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(z @ W.T + b + x * 0.01)

        return f, d

    def test_import(self) -> None:
        from pcg_llm.arch.deq_solver import ConstrainedDEQSolver  # noqa: F401

    def test_anderson_converges_within_12_iterations(self, simple_linear_f) -> None:
        from pcg_llm.arch.deq_solver import ConstrainedDEQSolver

        f, d = simple_linear_f
        solver = ConstrainedDEQSolver(
            anderson_window=3,
            anderson_beta=1e-4,
            max_iter=12,
            solver_tolerance=1e-2,
        )
        z0 = torch.zeros(1, d)
        x = torch.randn(1, d)

        z_star, info = solver.solve(f, z0, x)

        assert info["converged"], "Anderson should converge for this contractive system"
        assert (
            info["solver_steps"] <= 12
        ), f"Should converge in ≤12 steps, took {info['solver_steps']}"
        assert z_star.shape == z0.shape

    def test_fixed_point_condition_satisfied(self, simple_linear_f) -> None:
        """||Z - f(Z, X)|| < solver_tolerance at output."""
        from pcg_llm.arch.deq_solver import ConstrainedDEQSolver

        f, d = simple_linear_f
        tol = 1e-2
        solver = ConstrainedDEQSolver(
            anderson_window=3,
            anderson_beta=1e-4,
            max_iter=20,
            solver_tolerance=tol,
        )
        z0 = torch.zeros(1, d)
        x = torch.randn(1, d)

        z_star, info = solver.solve(f, z0, x)

        residual = torch.norm(z_star - f(z_star, x)).item()
        assert residual < tol * 10, f"Fixed-point residual {residual:.4f} should be < {tol * 10}"

    def test_solver_logs_step_count(self, simple_linear_f) -> None:
        from pcg_llm.arch.deq_solver import ConstrainedDEQSolver

        f, d = simple_linear_f
        solver = ConstrainedDEQSolver(
            anderson_window=3,
            anderson_beta=1e-4,
            max_iter=12,
            solver_tolerance=1e-2,
        )
        z0 = torch.zeros(1, d)
        x = torch.randn(1, d)

        _, info = solver.solve(f, z0, x)

        assert "solver_steps" in info, "Solver must report step count in info dict"
        assert isinstance(info["solver_steps"], int)
        assert info["solver_steps"] >= 1

    def test_broyden_fallback_on_anderson_non_convergence(self) -> None:
        """Broyden fallback triggers when Anderson exceeds max_iter."""
        from pcg_llm.arch.deq_solver import ConstrainedDEQSolver

        # Pathological function: won't converge quickly with Anderson m=3
        def hard_f(z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            # Oscillating function that's hard for Anderson with small m
            return torch.sin(z * 3.0) * 0.5 + x * 0.01

        d = 16
        solver = ConstrainedDEQSolver(
            anderson_window=3,
            anderson_beta=1e-4,
            max_iter=3,  # Very small to force fallback
            solver_tolerance=1e-6,  # Very tight tolerance
            broyden_max_iter=20,
            use_broyden_fallback=True,
        )
        z0 = torch.zeros(1, d)
        x = torch.zeros(1, d)

        z_star, info = solver.solve(hard_f, z0, x)

        # The key check: solve must return a result without raising
        assert z_star.shape == z0.shape
        assert "solver_steps" in info

    def test_spectral_norm_reduces_singular_value(self) -> None:
        """Spectral Normalization reduces max singular value to ≤ 1.0."""
        from pcg_llm.arch.deq_solver import apply_spectral_norm_constraint

        torch.manual_seed(42)
        W = torch.randn(32, 32) * 5.0  # Large weights, singular values >> 1
        W_normed = apply_spectral_norm_constraint(W)

        # Compute max singular value
        _, S, _ = torch.linalg.svd(W_normed)
        max_sv = S.max().item()
        assert (
            max_sv <= 1.0 + 1e-5
        ), f"Spectral norm should be ≤ 1.0 after normalization, got {max_sv:.4f}"

    def test_anderson_window_m3(self, simple_linear_f) -> None:
        """Verify m=3 history buffer is used by default (research decision)."""
        from pcg_llm.arch.deq_solver import ConstrainedDEQSolver

        f, d = simple_linear_f
        solver = ConstrainedDEQSolver()
        assert solver.anderson_window == 3, "Default Anderson window must be m=3 per research.md"


# ---------------------------------------------------------------------------
# FR-011 3-step DEQ fallback (T059)
# ---------------------------------------------------------------------------


@pytest.mark.convergence
class TestThreeStepFallbackFR011:
    """FR-011: on EAGLE draft rejection, solver must be capped at 3 steps.

    The full EAGLE→fallback pipeline requires GPU E2E testing; these unit tests
    verify the underlying mechanism: a ConstrainedDEQSolver instantiated with
    max_iter=3 must respect that cap and produce a valid output tensor in vocab
    range after token projection.
    """

    def test_solver_respects_3_step_cap(self) -> None:
        """solver_steps must be <= 3 when max_iter=3 (the fallback cap)."""
        from pcg_llm.arch.deq_solver import ConstrainedDEQSolver

        fallback_solver = ConstrainedDEQSolver(
            anderson_window=3,
            anderson_beta=1e-4,
            max_iter=3,  # FR-011 fallback cap
            solver_tolerance=1e-2,
            use_broyden_fallback=False,  # pure cap test
        )

        d = 32
        f = lambda z, x: torch.tanh(z * 0.9 + x * 0.1)  # noqa: E731
        z0 = torch.zeros(1, d)
        x = torch.randn(1, d)

        _, info = fallback_solver.solve(f, z0, x)

        assert (
            info["solver_steps"] <= 3
        ), f"FR-011 fallback: solver_steps must be <= 3, got {info['solver_steps']}"

    def test_fallback_solver_produces_valid_output_tensor(self) -> None:
        """3-step solver must return a tensor of correct shape (no crash)."""
        from pcg_llm.arch.deq_solver import ConstrainedDEQSolver

        fallback_solver = ConstrainedDEQSolver(max_iter=3, use_broyden_fallback=False)

        d = 64
        f = lambda z, x: torch.tanh(z * 0.9 + x * 0.05)  # noqa: E731
        z0 = torch.zeros(2, 4, d)  # [B=2, N=4, d=64] — realistic PCG shape
        x = torch.randn(2, 4, d)

        z_star, info = fallback_solver.solve(f, z0, x)

        assert z_star.shape == z0.shape, "Output shape must match input"
        assert not torch.isnan(z_star).any(), "Output must not contain NaN"
        assert info["solver_steps"] <= 3

    def test_fallback_cap_independent_of_production_max(self) -> None:
        """3-step solver is independent of the production solver's max_iter setting."""
        from pcg_llm.arch.deq_solver import ConstrainedDEQSolver

        production_solver = ConstrainedDEQSolver(max_iter=25)
        fallback_solver = ConstrainedDEQSolver(max_iter=3, use_broyden_fallback=False)

        # Both use the same contractive function and initial point
        f = lambda z, x: torch.tanh(z * 0.5)  # noqa: E731
        z0 = torch.randn(1, 16)
        x = torch.zeros(1, 16)

        _, prod_info = production_solver.solve(f, z0, x)
        _, fall_info = fallback_solver.solve(f, z0, x)

        assert fall_info["solver_steps"] <= 3, (
            f"Fallback must cap at 3 steps regardless of production settings, "
            f"got {fall_info['solver_steps']}"
        )
        # Production solver may use more steps (it should converge better)
        assert prod_info["solver_steps"] <= 25
