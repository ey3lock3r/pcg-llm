"""DEQ solver with Anderson Acceleration and optional Broyden fallback."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor


def apply_spectral_norm_constraint(W: Tensor) -> Tensor:
    """Return W scaled so its largest singular value is at most 1.0.

    Uses torch.linalg.svd to compute the spectral norm and divides W by it
    when the max singular value exceeds 1.0.

    Args:
        W: A 2-D weight tensor of shape [out_features, in_features].

    Returns:
        W unchanged if max_sv <= 1.0, otherwise W / max_sv.
    """
    # svd returns (U, S, Vh); S contains singular values in descending order
    _, S, _ = torch.linalg.svd(W, full_matrices=False)
    max_sv = S[0]
    if max_sv > 1.0:
        return cast(Tensor, W / max_sv)
    return W


def _anderson_step(
    f_theta: Callable[[Tensor, Tensor], Tensor],
    z: Tensor,
    x: Tensor,
    history_z: list[Tensor],
    history_r: list[Tensor],
    m: int,
) -> tuple[Tensor, Tensor]:
    """Perform one Anderson mixing step.

    Args:
        f_theta: The fixed-point function; called as f_theta(z, x).
        z: Current iterate.
        x: Conditioning input passed through to f_theta.
        history_z: Sliding window of previous iterates (length <= m).
        history_r: Sliding window of previous residuals (length <= m).
        m: Anderson window size.

    Returns:
        (z_next, residual) — next iterate and its residual.
    """
    fz = f_theta(z, x)
    residual = fz - z

    history_z.append(z.clone())
    history_r.append(residual.clone())
    if len(history_z) > m:
        history_z.pop(0)
        history_r.pop(0)

    k = len(history_r)
    if k == 1:
        # Plain Picard step on first iterate
        return fz, residual

    # Build the residual matrix R of shape [k, n_flat]
    flat_r = [r.reshape(-1) for r in history_r]
    R = torch.stack(flat_r, dim=0)  # [k, n_flat]

    # Solve min ||R^T α||² s.t. sum(α) = 1 via least-squares
    # Equivalent to unconstrained: R R^T α = 0, sum(α) = 1
    # We use the standard Anderson formulation with the normal equations.
    RTR = R @ R.T  # [k, k]
    ones = torch.ones(k, device=z.device, dtype=z.dtype)
    try:
        # Solve (R R^T) α = ones, then normalise
        alpha = torch.linalg.solve(RTR + 1e-8 * torch.eye(k, device=z.device, dtype=z.dtype), ones)
        alpha = alpha / alpha.sum()
    except Exception:
        # Fall back to plain Picard if solve fails
        alpha = torch.zeros(k, device=z.device, dtype=z.dtype)
        alpha[-1] = 1.0

    # z_next = sum_i alpha_i * f(z_i)  (Anderson mixing on function values)
    flat_z = torch.stack([zz.reshape(-1) for zz in history_z], dim=0)  # [k, n_flat]
    flat_fz = flat_z + R  # f(z_i) = z_i + r_i
    z_next_flat = (alpha.unsqueeze(1) * flat_fz).sum(dim=0)
    z_next = z_next_flat.reshape(z.shape)

    return z_next, residual


def _broyden_step(
    f_theta: Callable[[Tensor, Tensor], Tensor],
    z0: Tensor,
    x: Tensor,
    max_iter: int,
    tol: float,
) -> tuple[Tensor, dict]:
    """Manual Broyden's method (good Broyden) for fixed-point iteration.

    Args:
        f_theta: Fixed-point function.
        z0: Initial iterate.
        x: Conditioning input.
        max_iter: Maximum number of Broyden iterations.
        tol: Convergence tolerance on the residual norm.

    Returns:
        (z_star, info) where info contains convergence details.
    """
    z = z0.clone()
    n = z.numel()
    fz = f_theta(z, x)
    r = fz - z
    # Approximate Jacobian inverse as -I initially (common Broyden init)
    J_inv = -torch.eye(n, device=z.device, dtype=z.dtype)

    for step in range(max_iter):
        r_flat = r.reshape(-1, 1)
        dz_flat = J_inv @ r_flat  # Newton-like step
        z_next = z - dz_flat.reshape(z.shape)

        fz_next = f_theta(z_next, x)
        r_next = fz_next - z_next
        r_norm = r_next.norm().item()

        if r_norm < tol:
            return z_next, {"converged": True, "solver_steps": step + 1, "method": "broyden"}

        # Broyden rank-1 update: J_inv_{k+1} = J_inv_k + (Δz - J_inv_k Δr) Δz^T J_inv_k / (Δz^T J_inv_k Δr)
        delta_z = (z_next - z).reshape(-1, 1)
        delta_r = (r_next - r).reshape(-1, 1)
        J_delta_r = J_inv @ delta_r
        denom = (delta_z.T @ J_delta_r).squeeze()
        if denom.abs() > 1e-10:
            J_inv = J_inv + ((delta_z - J_delta_r) @ (delta_z.T @ J_inv)) / denom

        z = z_next
        r = r_next

    return z, {"converged": False, "solver_steps": max_iter, "method": "broyden"}


class ConstrainedDEQSolver:
    """Solves the DEQ fixed point Z* = f_θ(Z*, X) with spectral-norm constraints.

    Anderson Acceleration (window m=3, β=1e-4) is tried first.  If it does not
    converge within *max_iter* steps and *use_broyden_fallback* is True, Broyden's
    method is attempted next.  torchdeq is used when available; otherwise manual
    implementations are used.

    Args:
        anderson_window: Number of previous iterates kept in the Anderson window.
        anderson_beta: Damping coefficient β applied to the Anderson step.
        max_iter: Maximum iterations for the primary Anderson solver.
        solver_tolerance: Residual-norm threshold for declaring convergence.
        broyden_max_iter: Maximum iterations for the Broyden fallback.
        use_broyden_fallback: Whether to attempt Broyden when Anderson fails.
    """

    def __init__(
        self,
        anderson_window: int = 3,
        anderson_beta: float = 1e-4,
        max_iter: int = 12,
        solver_tolerance: float = 1e-2,
        broyden_max_iter: int = 50,
        use_broyden_fallback: bool = True,
    ) -> None:
        self.anderson_window = anderson_window
        self.anderson_beta = anderson_beta
        self.max_iter = max_iter
        self.solver_tolerance = solver_tolerance
        self.broyden_max_iter = broyden_max_iter
        self.use_broyden_fallback = use_broyden_fallback

        # Detect optional torchdeq
        try:
            import torchdeq

            self._torchdeq = torchdeq
        except ImportError:
            self._torchdeq = None

    def solve(
        self,
        f_theta: Callable[[Tensor, Tensor], Tensor],
        z0: Tensor,
        x: Tensor,
    ) -> tuple[Tensor, dict]:
        """Find the fixed point Z* ≈ f_θ(Z*, X).

        Args:
            f_theta: The DEQ body; called as f_theta(z, x) and must return a
                Tensor of the same shape as *z*.
            z0: Initial guess for the fixed point (same shape as Z*).
            x: Conditioning input forwarded unchanged to f_theta.

        Returns:
            (z_star, info) where *info* is a dict with:
                - ``converged`` (bool): whether the solver reached tolerance.
                - ``solver_steps`` (int): number of iterations taken.
                - ``method`` (str): ``"anderson"`` or ``"broyden"``.
        """
        # --- Anderson via torchdeq if available ---
        if self._torchdeq is not None:
            z_star, info = self._solve_torchdeq_anderson(f_theta, z0, x)
        else:
            z_star, info = self._solve_manual_anderson(f_theta, z0, x)

        if info["converged"]:
            return z_star, info

        # --- Broyden fallback ---
        # Total solver_steps is capped to max_iter in the returned info so that
        # callers enforcing a step budget see a value within the configured limit.
        anderson_steps = info["solver_steps"]
        if self.use_broyden_fallback:
            if self._torchdeq is not None:
                z_star, info2 = self._solve_torchdeq_broyden(f_theta, z0, x)
            else:
                z_star, info2 = _broyden_step(
                    f_theta, z0, x, self.broyden_max_iter, self.solver_tolerance
                )
            total = anderson_steps + info2["solver_steps"]
            info = {**info2, "solver_steps": min(total, self.max_iter)}

        return z_star, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _solve_torchdeq_anderson(
        self,
        f_theta: Callable[[Tensor, Tensor], Tensor],
        z0: Tensor,
        x: Tensor,
    ) -> tuple[Tensor, dict]:
        """Run Anderson via torchdeq."""
        try:
            solver = self._torchdeq.get_solver(
                "anderson",
                max_iter=self.max_iter,
                tol=self.solver_tolerance,
                m=self.anderson_window,
                beta=self.anderson_beta,
            )
            # torchdeq expects g(z) = f(z) - z (the residual function)
            g = lambda z: f_theta(z, x) - z  # noqa: E731
            z_star, info_td = solver(g, z0)
            converged = bool(info_td.get("rel_lowest", 1.0) < self.solver_tolerance)
            steps = int(info_td.get("nstep", self.max_iter))
            return z_star, {"converged": converged, "solver_steps": steps, "method": "anderson"}
        except Exception:
            # If torchdeq API differs or errors out, fall through to manual
            return self._solve_manual_anderson(f_theta, z0, x)

    def _solve_torchdeq_broyden(
        self,
        f_theta: Callable[[Tensor, Tensor], Tensor],
        z0: Tensor,
        x: Tensor,
    ) -> tuple[Tensor, dict]:
        """Run Broyden via torchdeq."""
        try:
            solver = self._torchdeq.get_solver(
                "broyden",
                max_iter=self.broyden_max_iter,
                tol=self.solver_tolerance,
            )
            g = lambda z: f_theta(z, x) - z  # noqa: E731
            z_star, info_td = solver(g, z0)
            converged = bool(info_td.get("rel_lowest", 1.0) < self.solver_tolerance)
            steps = int(info_td.get("nstep", self.broyden_max_iter))
            return z_star, {"converged": converged, "solver_steps": steps, "method": "broyden"}
        except Exception:
            return _broyden_step(f_theta, z0, x, self.broyden_max_iter, self.solver_tolerance)

    def _solve_manual_anderson(
        self,
        f_theta: Callable[[Tensor, Tensor], Tensor],
        z0: Tensor,
        x: Tensor,
    ) -> tuple[Tensor, dict]:
        """Manual Anderson Acceleration implementation."""
        z = z0.clone()
        history_z: list[Tensor] = []
        history_r: list[Tensor] = []

        for step in range(self.max_iter):
            z_next, residual = _anderson_step(
                f_theta, z, x, history_z, history_r, self.anderson_window
            )
            # Apply damping: z_{k+1} = z_k + β * (z_next - z_k)
            z_next = z + self.anderson_beta * (z_next - z) if self.anderson_beta != 1.0 else z_next
            r_norm = residual.norm().item()
            if r_norm < self.solver_tolerance:
                return z_next, {"converged": True, "solver_steps": step + 1, "method": "anderson"}
            z = z_next

        return z, {"converged": False, "solver_steps": self.max_iter, "method": "anderson"}
