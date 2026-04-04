"""Tests for FreeEnergyLoss — TDD Phase 3 (T013)."""

from __future__ import annotations

import pytest
import torch


class TestFreeEnergyLoss:
    """Verify three-term loss and anti-collapse mechanics."""

    @pytest.fixture
    def loss_fn(self):
        from pcg_llm.training.loss import FreeEnergyLoss

        return FreeEnergyLoss(
            vocab_size=100,
            lambda_sparse=0.01,
            gamma_variance=0.01,
            variance_floor=0.1,
        )

    def test_import(self) -> None:
        from pcg_llm.training.loss import FreeEnergyLoss  # noqa: F401

    def test_cross_entropy_term_is_correct(self, loss_fn) -> None:
        """CrossEntropy term matches torch.nn.CrossEntropyLoss for known logits."""
        import torch.nn.functional as F

        from pcg_llm.training.loss import FreeEnergyLoss

        torch.manual_seed(0)
        B, V = 4, 100
        logits = torch.randn(B, V)
        targets = torch.randint(0, V, (B,))

        loss_fn_custom: FreeEnergyLoss = loss_fn

        # Compute using the custom loss (disable L1 and variance terms)
        adj_weights = torch.zeros(8, 8)  # zero → L1 = 0
        Z_star = torch.randn(B, 8, 32)  # high variance → variance hinge = 0
        Z_star = Z_star + 2.0  # ensure variance > threshold

        loss, components = loss_fn_custom(logits, targets, adj_weights, Z_star)

        # Reference CE
        ref_ce = F.cross_entropy(logits, targets)
        assert abs(components["cross_entropy"].item() - ref_ce.item()) < 1e-4, (
            f"CE term {components['cross_entropy'].item():.6f} != ref {ref_ce.item():.6f}"
        )

    def test_l1_sparsity_penalty_equals_sum_of_abs_weights(self, loss_fn) -> None:
        """L1 penalty = lambda * sum(|A|)."""
        from pcg_llm.training.loss import FreeEnergyLoss

        B, V = 2, 100
        logits = torch.randn(B, V)
        targets = torch.randint(0, V, (B,))
        adj_weights = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
        Z_star = torch.randn(B, 2, 32) + 2.0

        loss_fn_l1: FreeEnergyLoss = loss_fn
        _, components = loss_fn_l1(logits, targets, adj_weights, Z_star)

        expected_l1 = loss_fn_l1.lambda_sparse * adj_weights.abs().sum()
        assert abs(components["l1_sparsity"].item() - expected_l1.item()) < 1e-5

    def test_variance_hinge_zero_when_variance_above_threshold(self, loss_fn) -> None:
        """Variance hinge should be zero when Var(Z*) > tau."""
        B, N, d = 2, 8, 32
        Z_high_var = torch.randn(B, N, d) * 5.0  # Very high variance
        logits = torch.randn(B, 100)
        targets = torch.randint(0, 100, (B,))
        adj = torch.zeros(N, N)

        _, components = loss_fn(logits, targets, adj, Z_high_var)

        # hinge = max(0, tau - Var(Z*)) should be ~0 for high-variance Z
        assert components["variance_hinge"].item() >= 0, "Hinge must be non-negative"
        # For very high variance, penalty should be near 0
        assert components["variance_hinge"].item() < 0.01, (
            "Variance hinge should be ~0 when Var(Z*) >> tau"
        )

    def test_variance_hinge_positive_when_variance_below_threshold(self, loss_fn) -> None:
        """Variance hinge is positive when Var(Z*) < tau."""
        B, N, d = 2, 8, 32
        Z_low_var = torch.zeros(B, N, d)  # Zero variance
        logits = torch.randn(B, 100)
        targets = torch.randint(0, 100, (B,))
        adj = torch.zeros(N, N)

        _, components = loss_fn(logits, targets, adj, Z_low_var)

        assert components["variance_hinge"].item() > 0, (
            "Variance hinge must be positive when Var(Z*) = 0 < tau=0.1"
        )

    def test_combined_loss_decreases_when_model_improves(self) -> None:
        """Combined loss should decrease as logits improve."""
        from pcg_llm.training.loss import FreeEnergyLoss

        loss_fn = FreeEnergyLoss(vocab_size=10, lambda_sparse=0.0, gamma_variance=0.0)

        B = 4
        targets = torch.zeros(B, dtype=torch.long)
        adj = torch.zeros(4, 4)
        Z_star = torch.randn(B, 4, 8) + 2.0

        # Bad logits: uniform
        logits_bad = torch.ones(B, 10)
        loss_bad, _ = loss_fn(logits_bad, targets, adj, Z_star)

        # Good logits: class 0 much higher
        logits_good = torch.zeros(B, 10)
        logits_good[:, 0] = 10.0
        loss_good, _ = loss_fn(logits_good, targets, adj, Z_star)

        assert loss_good.item() < loss_bad.item(), (
            "Loss must decrease when logits improve"
        )

    def test_anti_collapse_guard_triggers_gamma_nudge(self) -> None:
        """Gamma auto-nudge +10% when variance < tau for 10 consecutive steps."""
        from pcg_llm.training.loss import FreeEnergyLoss

        loss_fn = FreeEnergyLoss(
            vocab_size=10,
            lambda_sparse=0.0,
            gamma_variance=0.01,
            variance_floor=0.5,  # High threshold to easily trigger
        )

        initial_gamma = loss_fn.gamma_variance
        B = 2
        # Z_star with very low variance (below threshold)
        Z_low_var = torch.zeros(B, 4, 8)
        logits = torch.randn(B, 10)
        targets = torch.randint(0, 10, (B,))
        adj = torch.zeros(4, 4)

        # Run 10 consecutive steps with low variance
        for _ in range(10):
            loss_fn(logits, targets, adj, Z_low_var)

        assert loss_fn.gamma_variance > initial_gamma, (
            "Gamma must auto-nudge (+10%) after 10 consecutive low-variance steps"
        )
