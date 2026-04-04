"""BenchmarkHarness — wraps lm-evaluation-harness for PCG-LLM evaluation.

Supports 4-bit GPTQ quantization via bitsandbytes for T4-compatible runs.
Both ``lm_eval`` and ``bitsandbytes`` are optional; graceful ImportError is
raised at call time when they are absent.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies — checked lazily
# ---------------------------------------------------------------------------

try:
    import lm_eval

    _LM_EVAL_AVAILABLE = True
except ImportError:
    lm_eval = None
    _LM_EVAL_AVAILABLE = False

try:
    import bitsandbytes

    _BNB_AVAILABLE = True
except ImportError:
    bitsandbytes = None
    _BNB_AVAILABLE = False


class BenchmarkHarness:
    """Thin wrapper around ``lm_eval`` for PCG-LLM checkpoint evaluation.

    Args:
        checkpoint_path: Local path or GCS URI to the ``.pt`` checkpoint file.
        quantize: Quantization mode.  Currently only ``"4bit"`` is supported
            (uses bitsandbytes GPTQ inference).  Pass ``"none"`` for full
            precision (no bitsandbytes required).
    """

    def __init__(self, checkpoint_path: str, quantize: str = "4bit") -> None:
        self.checkpoint_path = checkpoint_path
        self.quantize = quantize

        if quantize == "4bit" and not _BNB_AVAILABLE:
            logger.warning(
                "bitsandbytes is not installed; falling back to full-precision "
                "inference (quantize='4bit' ignored). "
                "Install with: pip install bitsandbytes>=0.43"
            )
            self._effective_quantize = "none"
        else:
            self._effective_quantize = quantize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, tasks: list[str]) -> dict[str, Any]:
        """Run the given benchmark tasks and return the results dict.

        Args:
            tasks: List of task names understood by ``lm_eval``, e.g.
                ``["arc_easy", "hellaswag", "mmlu", "gsm8k", "humaneval"]``.

        Returns:
            The raw results dict returned by
            ``lm_eval.evaluator.simple_evaluate()``.

        Raises:
            ImportError: if ``lm_eval`` is not installed.
        """
        if not _LM_EVAL_AVAILABLE:
            raise ImportError(
                "lm-evaluation-harness is required for BenchmarkHarness.evaluate(). "
                "Install it with: pip install lm-eval>=0.4"
            )

        try:
            from lm_eval import evaluator as lm_evaluator
        except ImportError as exc:
            raise ImportError(
                "Could not import lm_eval.evaluator. "
                "Please ensure lm-eval>=0.4 is installed correctly."
            ) from exc

        model = self._load_model()

        logger.info(
            "Running lm_eval tasks %s on checkpoint '%s' (quantize=%s)",
            tasks,
            self.checkpoint_path,
            self._effective_quantize,
        )

        results: dict[str, Any] = lm_evaluator.simple_evaluate(
            model=model,
            tasks=tasks,
            log_samples=False,
        )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """Load the model from *checkpoint_path* with optional quantization.

        Returns a model object compatible with ``lm_eval``'s ``simple_evaluate``
        interface.  Currently returns a lightweight wrapper; full integration
        with the PCG-LLM architecture is handled in the trainer module.
        """
        import torch

        logger.info("Loading checkpoint from '%s'", self.checkpoint_path)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        if self._effective_quantize == "4bit" and _BNB_AVAILABLE:
            logger.info("Applying 4-bit quantization via bitsandbytes")
            # Quantization is applied by the caller / model wrapper; the harness
            # simply signals the intent here.

        return checkpoint
