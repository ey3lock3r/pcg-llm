#!/usr/bin/env python3
"""Repository root entry point. Delegates to pcg_llm CLI."""

import sys

from pcg_llm._cli import main

if __name__ == "__main__":
    sys.exit(main())
