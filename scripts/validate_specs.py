"""Validate spec files under specs/ for required sections."""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_SECTIONS = ["## Requirements"]

ROOT = Path(__file__).parent.parent
SPECS_DIR = ROOT / "specs"


def validate_spec(path: Path) -> list[str]:
    """Return a list of error strings for the given spec file."""
    errors: list[str] = []
    text = path.read_text(encoding="utf-8")
    for section in REQUIRED_SECTIONS:
        if section not in text:
            errors.append(f"{path}: missing section '{section}'")
    return errors


def main() -> int:
    spec_files = list(SPECS_DIR.rglob("spec.md"))
    if not spec_files:
        print("No spec files found — nothing to validate.")
        return 0

    all_errors: list[str] = []
    for spec_file in spec_files:
        all_errors.extend(validate_spec(spec_file))

    if all_errors:
        for err in all_errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1

    print(f"Validated {len(spec_files)} spec file(s) — all OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
