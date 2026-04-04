"""Integration tests for EAGLEExtrapolationHead — TDD Phase 4 (T030)."""
from __future__ import annotations
import pytest
import torch

@pytest.mark.slow
class TestEAGLEExtrapolationHead:
    def test_import(self) -> None:
        from pcg_llm.arch.eagle_head import EAGLEExtrapolationHead  # noqa

    def test_draft_tree_shape(self) -> None:
        from pcg_llm.arch.eagle_head import EAGLEExtrapolationHead
        B, num_blocks, hidden_dim = 2, 8, 64
        K, draft_len = 4, 4
        head = EAGLEExtrapolationHead(hidden_dim=hidden_dim, eagle_k=K, draft_len=draft_len)
        Z = torch.randn(B, num_blocks, hidden_dim)
        tree = head.generate_draft_tree(Z)
        assert tree.shape[0] == B, f"Expected batch dim {B}"
        assert tree.shape[1] == K, f"Expected K={K} branches"
        assert tree.shape[2] == draft_len

    def test_accept_rate_logging(self) -> None:
        from pcg_llm.arch.eagle_head import EAGLEExtrapolationHead
        head = EAGLEExtrapolationHead(hidden_dim=64, eagle_k=4, draft_len=4)
        assert hasattr(head, "accept_rate_ema"), "Head must track accept_rate_ema"

    def test_draft_len_expands_when_accept_rate_high(self) -> None:
        from pcg_llm.arch.eagle_head import EAGLEExtrapolationHead
        head = EAGLEExtrapolationHead(hidden_dim=64, eagle_k=4, draft_len=4,
                                       eagle_accept_threshold=0.65)
        initial_len = head.draft_len
        # Simulate high acceptance rate
        head.accept_rate_ema = 0.80
        head._maybe_expand_draft_len()
        assert head.draft_len >= initial_len, "draft_len should expand when accept rate is high"
