"""Predictive Coding Graph Language Model (PCG-LLM).

A non-hierarchical, dynamically routed neural architecture that replaces
backpropagation with Inference Learning and Deep Equilibrium (DEQ) solving.

Architecture Overview:
    - PCG substrate: universally connected graph of latent state estimators
    - Training: Inference Learning via free energy minimization (no backprop)
    - Sparsity: Block-structured 64x64 RigL Drop-and-Grow
    - Convergence: Spectral Normalization enforces Lipschitz L < 1
    - Inference: EAGLE extrapolation head + Parallel Tree Attention

See specs/constitution.md for non-negotiable architectural constraints.
See docs/Engineering Requirements Document_ PCG-LLM v3.md for full spec.
"""

__version__ = "0.1.0"
__author__ = "pcg-llm contributors"
__license__ = "MIT"

__all__ = ["__version__"]
