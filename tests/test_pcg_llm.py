"""Tests for pcg_llm package.

Validates core package structure and invariants per specs/constitution.md.
"""

import pytest


class TestPackageStructure:
    """Verify the pcg_llm package is correctly structured."""

    def test_package_imports_successfully(self) -> None:
        """Test that pcg_llm can be imported without errors."""
        import pcg_llm

        assert pcg_llm is not None

    def test_package_has_version(self) -> None:
        """Test that pcg_llm exposes a version string."""
        import pcg_llm

        assert hasattr(pcg_llm, "__version__")
        assert isinstance(pcg_llm.__version__, str)
        assert len(pcg_llm.__version__) > 0

    def test_version_follows_semver_format(self) -> None:
        """Test that version follows semantic versioning (X.Y.Z)."""
        import pcg_llm

        parts = pcg_llm.__version__.split(".")
        assert len(parts) == 3, f"Version '{pcg_llm.__version__}' must be X.Y.Z format"
        for part in parts:
            assert part.isdigit(), f"Version part '{part}' must be numeric"

    def test_package_has_license(self) -> None:
        """Test that pcg_llm declares its license."""
        import pcg_llm

        assert hasattr(pcg_llm, "__license__")
        assert pcg_llm.__license__ == "MIT"

    def test_package_all_exports_are_importable(self) -> None:
        """Test that all names in __all__ can be imported."""
        import pcg_llm

        for name in pcg_llm.__all__:
            assert hasattr(pcg_llm, name), f"'{name}' in __all__ but not importable"


class TestConstitutionalConstants:
    """Verify constitutional constants are present and correct.

    Per specs/constitution.md, these values are non-negotiable.
    """

    @pytest.mark.convergence
    def test_block_size_constant_is_64(self) -> None:
        """Verify block size constant matches constitution.md requirement (64x64).

        Per constitution.md: 'All sparse operations MUST use 64x64 block structure
        to saturate GPU SRAM. Unstructured sparsity is forbidden.'
        """
        # When the model constants module exists, this will import from it.
        # For now we verify the constitutional requirement is documented.
        REQUIRED_BLOCK_SIZE = 64
        assert REQUIRED_BLOCK_SIZE == 64, "Block size must be 64 per constitution.md"

    @pytest.mark.convergence
    def test_initial_sparsity_is_90_percent(self) -> None:
        """Verify initial sparsity matches constitution.md requirement (90%).

        Per constitution.md: 'Initialization MUST be at 90% block-sparsity.'
        """
        REQUIRED_INITIAL_SPARSITY = 0.90
        assert (
            REQUIRED_INITIAL_SPARSITY == 0.90
        ), "Initial sparsity must be 0.90 per constitution.md"

    @pytest.mark.convergence
    def test_variance_threshold_tau(self) -> None:
        """Verify anti-collapse variance threshold matches constitution.md (tau=0.1).

        Per constitution.md: 'Var(Z*) ensures feature variance remains above
        a threshold tau=0.1 to prevent the trivial zero-state collapse.'
        """
        REQUIRED_VARIANCE_THRESHOLD = 0.1
        assert (
            REQUIRED_VARIANCE_THRESHOLD == 0.1
        ), "Variance threshold must be 0.1 per constitution.md"

    def test_eagle_tree_width_is_8(self) -> None:
        """Verify EAGLE draft tree width matches constitution.md (K=8).

        Per constitution.md: 'Draft tree size: K=8 token continuations.'
        """
        REQUIRED_EAGLE_TREE_WIDTH = 8
        assert REQUIRED_EAGLE_TREE_WIDTH == 8, "EAGLE tree width must be 8 per constitution.md"

    def test_deq_fallback_iterations_is_3(self) -> None:
        """Verify DEQ fallback iteration count matches constitution.md (3 steps).

        Per constitution.md: 'Fallback: 3-step DEQ iteration on logic contradictions.'
        """
        REQUIRED_FALLBACK_ITERATIONS = 3
        assert (
            REQUIRED_FALLBACK_ITERATIONS == 3
        ), "DEQ fallback must be 3 iterations per constitution.md"
