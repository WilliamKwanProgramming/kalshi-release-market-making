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
CHECKS: list[tuple[str, str, str]] = [
    # D, the oracle and the proxy (tables M1, M4)
    ("results/markouts/tables.md", "| saturated events, % | 83.7 |",
     "84% of the planted events end outside [0.01, 0.99]"),
    # coverage and power (tables M5, M8)
    ("results/markouts/tables.md", "percent of 100 replications whose 95% interval contains",
     "each rate in the table comes from 100 replications"),
    ("results/markouts/tables.md", "Monte Carlo standard error at most 5.0 points",
     "so it is uncertain by up to 5 points"),
    ("results/markouts/tables.md",
     "| planted | (a) raw D, near (0, 30] | 91 / 66 | 95 / 69 | 96 / 76 |",
     "close to 95% from about 100 events"),
    ("results/markouts/tables.md", "| (a) raw D, near (0, 30] | 50 | 99% | 76% at n = 25 |",
     "For the planted effect a few dozen events"),
    # the quoting comparison and the trading window (tables B1, B2, B8)
    ("results/backtest/tables.md", "| 12 | 10 | -1.22 (0.12) |",
     "with 12 hours of trading before each release, the premium by bucket loses"),
    ("results/backtest/tables.md", "| 30 | 10 | +1.37 (0.11) |",
     "the rehearsal below has 30 hours of trading"),
    ("results/backtest/tables.md", "| 12 | 10 | -1.22 (0.12) |",
     "with 12 hours of trading before each release and wins with 30 (tables B1, B8"),
    ("results/backtest/tables.md", "negative in 8 of 8 world and policy cells",
     "every policy loses money (B1, B2)"),
    ("results/backtest/tables.md", "negative in 8 of 8 world and policy cells",
     "every policy loses money in the backtest (table B2)"),
    # recalibration (table C1)
    ("results/calibration/tables.md", "| 1.0 | 100 | 50 | -2.76 (0.35) |",
     "in every row (its Brier skill is negative)"),
    ("results/calibration/tables.md", "| 1.0 | 200 | 150 | -1.89 (0.14) |",
     "in every row (its Brier skill is negative)"),
    ("results/calibration/tables.md", "| 1.0 | 500 | 450 | -1.03 (0.05) |",
     "in every row (its Brier skill is negative)"),
    ("results/calibration/tables.md", "| 1.0 | 1000 | 950 | -0.65 (0.03) |",
     "in every row (its Brier skill is negative)"),
    ("results/calibration/tables.md",
     "| 0.7 | 1000 | 950 | 1.05 (0.06) | 1.71 | 24.0 (3.0) |",
     "(0.7) and 1000 events it is supported in 24% of the datasets"),
    # the rehearsal (tables R2 to R8)
    ("results/rehearsal/rehearsal.manifest.txt", "world: 200 events of 30.0 hours",
     "a world of 200 events (30 hours of trading before each release)"),
    ("results/rehearsal/tables.md", "| P2 | 0.0001 | 0.0003 | supported |",
     "its test of this comparison (P2) came out the other way"),
]


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
