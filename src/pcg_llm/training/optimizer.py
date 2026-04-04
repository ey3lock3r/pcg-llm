"""Muon optimizer and HybridOptimizer for PCG-LLM training.

Muon uses Newton-Schulz orthogonalisation to produce orthogonal update
directions for 2-D weight matrices (nn.Linear weights).  It must NOT be
applied to 1-D tensors such as biases or embedding weights.

HybridOptimizer applies Muon to all 2-D nn.Linear.weight tensors and
AdamW to everything else (embeddings, biases, LayerNorm scales, etc.).

References:
    Kosson et al., "Muon: Momentum Orthogonalized by Newton-Schulz"
    https://github.com/KellerJordan/modded-nanogpt
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import overload

import torch
from torch import Tensor
from torch.optim import Optimizer

# ---------------------------------------------------------------------------
# Newton-Schulz iteration for approximate matrix orthogonalisation
# ---------------------------------------------------------------------------


def _newton_schulz_5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Return an approximate orthogonal matrix closest to *G* via NS iteration.

    Uses the degree-5 polynomial iteration from the Muon paper:
        X ← a*X + b*X*(X^T X) + c*X*(X^T X)^2
    with coefficients (a=3.4445, b=-4.7750, c=2.0315).

    The gradient is normalised before iteration to keep the spectral radius
    near 1, which is required for convergence.

    Args:
        G: A 2-D gradient tensor of shape ``(m, n)``.
        steps: Number of Newton-Schulz iterations (default 5).
        eps: Small constant to avoid division by zero during normalisation.

    Returns:
        An approximately orthogonal matrix of the same shape as *G*.
    """
    assert G.ndim == 2, f"Newton-Schulz requires a 2-D tensor, got shape {G.shape}"
    # Work in float32 for numerical stability
    X = G.float()
    # Normalise
    norm = X.norm()
    X = X / (norm + eps)
    # Polynomial coefficients (Muon paper)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ A @ X)
    return X.to(G.dtype)


# ---------------------------------------------------------------------------
# MuonOptimizer
# ---------------------------------------------------------------------------


class MuonOptimizer(Optimizer):
    """Momentum Orthogonalised by Newton-Schulz (Muon) for 2-D weight matrices.

    Only accepts parameter tensors with ``ndim == 2``.  Attempting to pass
    1-D tensors (biases) raises ``ValueError``.

    Args:
        params: Iterable of 2-D ``nn.Parameter`` tensors.
        lr: Learning rate.
        momentum: SGD momentum coefficient (default 0.95).
        ns_steps: Number of Newton-Schulz iterations (default 5).

    Raises:
        ValueError: if any parameter has ``ndim != 2``.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.01,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ) -> None:
        param_list = list(params)
        for p in param_list:
            if p.ndim != 2:
                raise ValueError(
                    f"MuonOptimizer only accepts 2-D parameter tensors "
                    f"(weight matrices); got shape {tuple(p.shape)}. "
                    "Use AdamW for 1-D tensors such as biases."
                )
        defaults = {"lr": lr, "momentum": momentum, "ns_steps": ns_steps}
        super().__init__(param_list, defaults)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                # Orthogonalise the momentum buffer
                update = _newton_schulz_5(buf, steps=ns_steps)
                p.add_(update, alpha=-lr)
        return loss


# ---------------------------------------------------------------------------
# HybridOptimizer
# ---------------------------------------------------------------------------


class HybridOptimizer:
    """Hybrid optimizer: Muon for 2-D Linear weights, AdamW for everything else.

    Accepts a flat list of parameters.  Internally partitions them into:
    - *muon_params*: tensors with ``ndim == 2`` (treated as weight matrices).
    - *adamw_params*: all other tensors (embeddings, biases, LayerNorm, etc.).

    If there are no 2-D parameters, only AdamW is created.

    Args:
        params: Iterable of parameters (all types accepted).
        base_lr: Learning rate passed to both sub-optimizers.
        momentum: Muon momentum (default 0.95).
        weight_decay: AdamW weight decay (default 0.01).
        betas: AdamW betas (default (0.9, 0.999)).
        ns_steps: Newton-Schulz iterations for Muon (default 5).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        base_lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        betas: tuple[float, float] = (0.9, 0.999),
        ns_steps: int = 5,
    ) -> None:
        param_list = list(params)
        muon_params = [p for p in param_list if p.ndim == 2]
        adamw_params = [p for p in param_list if p.ndim != 2]

        self._optimizers: list[Optimizer] = []

        if muon_params:
            self._muon_opt: MuonOptimizer | None = MuonOptimizer(
                muon_params, lr=base_lr, momentum=momentum, ns_steps=ns_steps
            )
            self._optimizers.append(self._muon_opt)
        else:
            self._muon_opt = None

        if adamw_params:
            self._adamw_opt: torch.optim.AdamW | None = torch.optim.AdamW(
                adamw_params, lr=base_lr, betas=betas, weight_decay=weight_decay
            )
            self._optimizers.append(self._adamw_opt)
        else:
            self._adamw_opt = None

        # If there are only 2-D params, still create an AdamW with empty group
        # so callers always have access to .step() without extra None checks.
        if not self._optimizers:
            # degenerate: no params at all
            self._adamw_opt = torch.optim.AdamW([], lr=base_lr)
            self._optimizers.append(self._adamw_opt)

    # ------------------------------------------------------------------
    # Optimizer protocol
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @overload
    def step(self, closure: None = ...) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        result: float | None = None
        for opt in self._optimizers:
            result = opt.step(closure)
        return result

    def state_dict(self) -> dict:
        return {
            "muon": self._muon_opt.state_dict() if self._muon_opt is not None else None,
            "adamw": self._adamw_opt.state_dict() if self._adamw_opt is not None else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self._muon_opt is not None and state.get("muon") is not None:
            self._muon_opt.load_state_dict(state["muon"])
        if self._adamw_opt is not None and state.get("adamw") is not None:
            self._adamw_opt.load_state_dict(state["adamw"])
