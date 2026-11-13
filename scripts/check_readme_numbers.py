#!/usr/bin/env python3
"""Check that the numbers quoted in README.md are the numbers in the committed results files.

Three kinds of check, all failing with exit status 1:

1. The tables pasted into README.md equal the tables in results/*/tables.md (the same test as
   ``make_tables.py --check``).
2. Each entry of CHECKS names a results file, a text that must occur in that file (a table row or
   a value) and a text that must occur in README.md. When a result is regenerated and its number
   changes, the first part fails; when the README is edited by hand and no longer quotes the
   number, the second fails.
3. Every relative link or image in README.md points to a file that exists.

(Modelled on scripts/check_readme_numbers.py of market-risk-engine.) Standard library only.

Usage: python3 scripts/check_readme_numbers.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import make_tables

ROOT = Path(__file__).resolve().parents[1]

# (results file, text that must occur in it, text that must occur in README.md)
CHECKS: list[tuple[str, str, str]] = []


def broken_links(readme: str) -> list[str]:
    """Relative links and images in the README whose target does not exist."""
    missing = []
    for target in re.findall(r"\]\(([^)\s]+)\)", readme):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        if not (ROOT / target.split("#", 1)[0]).exists():
            missing.append(target)
    return missing


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    problems = 0
    for target in broken_links(readme):
        print(f"README.md links to {target!r}, which does not exist")
        problems += 1
    try:
        if make_tables.fill(readme, make_tables.read_tables(ROOT / "results")) != readme:
            print("README.md tables differ from results/*/tables.md; run scripts/make_tables.py")
            problems += 1
    except ValueError as err:
        print(f"tables: {err}")
        problems += 1
    cache: dict[str, str] = {}
    for path, in_file, in_readme in CHECKS:
        if path not in cache:
            try:
                cache[path] = (ROOT / path).read_text(encoding="utf-8")
            except FileNotFoundError:
                print(f"{path}: file not found")
                problems += 1
                continue
        if in_file not in cache[path]:
            print(f"{path}: {in_file!r} not found")
            problems += 1
        if in_readme not in readme:
            print(f"README.md does not contain {in_readme!r} (checked against {path})")
            problems += 1
    print(f"{len(CHECKS)} number checks and the table and link checks, {problems} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
