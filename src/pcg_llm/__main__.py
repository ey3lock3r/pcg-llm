"""CLI entry point for PCG-LLM training and evaluation.

Usage:
    python -m pcg_llm train [OPTIONS]
    python -m pcg_llm evaluate [OPTIONS]
    python -m pcg_llm export-config [OPTIONS]
"""

from __future__ import annotations

import sys

from pcg_llm._cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
