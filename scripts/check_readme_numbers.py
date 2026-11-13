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
    # the quoting comparisons, the premium sizes and the trading window (tables B1, B2, B7 to B9)
    ("results/backtest/tables.md", "| null | +1.22 (0.05) | +0.88 to +1.44 | 0 | 10 | 0 |",
     "`release_aware` earns more than `flat_matched` in both worlds (B1)"),
    ("results/backtest/tables.md", "| planted | +2.83 (0.14) | +2.05 to +3.80 | 0 | 10 | 0 |",
     "`release_aware` earns more than `flat_matched` in both worlds (B1)"),
    ("results/backtest/tables.md", "The pooled premium estimated at the end is 2.80 cents",
     "(2.80 and 1.65 cents in the planted world)"),
    ("results/backtest/tables.md", "is 1.65 cents: a ratio of 1.70",
     "(2.80 and 1.65 cents in the planted world)"),
    ("results/backtest/tables.md", "negative in 10 of 10 world and policy cells",
     "every policy loses money (B2)"),
    ("results/backtest/tables.md", "negative in 10 of 10 world and policy cells",
     "every policy loses money in the backtest (table B2)"),
    ("results/backtest/tables.md",
     "| null | `release_aware` | -9.10 (0.35) | 2.20 | 104.6 | 657.0 | -1.39 | -9.22 | 0.937 "
     "| 4.73 |",
     "`release_aware` quotes 4.73 cents on average against 4.81 for `flat_matched` in the null"),
    ("results/backtest/tables.md",
     "| null | `flat_matched` | -10.32 (0.33) | 2.37 | 115.1 | 719.2 | -1.44 | -10.46 | 0.937 "
     "| 4.81 |",
     "`release_aware` quotes 4.73 cents on average against 4.81 for `flat_matched` in the null"),
    ("results/backtest/tables.md",
     "| planted | `release_aware` | -11.14 (0.21) | 2.09 | 103.3 | 649.3 | -1.72 | -10.58 | 0.917 "
     "| 4.97 |",
     "world, and 4.97 against 5.37 in the planted world"),
    ("results/backtest/tables.md",
     "| planted | `flat_matched` | -13.97 (0.29) | 2.18 | 111.6 | 696.8 | -2.00 | -13.25 | 0.917 "
     "| 5.37 |",
     "world, and 4.97 against 5.37 in the planted world"),
    ("results/backtest/tables.md",
     "| 12 | 10 | +2.83 (0.14) | 0 | 10 | -1.22 (0.12) | 9 | 0 | 2.80 | 1.65 |",
     "`release_aware` earns less than `pooled_premium` and with 30 hours more"),
    ("results/backtest/tables.md",
     "| 30 | 10 | +1.71 (0.10) | 0 | 10 | +1.37 (0.11) | 0 | 10 | 1.77 | 1.64 |",
     "`release_aware` earns less than `pooled_premium` and with 30 hours more"),
    ("results/backtest/tables.md", "changes sign at F = 1.47",
     "it earns more until that premium is scaled by about 1.5 and less beyond (B9)"),
    ("results/backtest/tables.md", "changes sign at F = 1.47",
     "less once that premium is scaled by about 1.5"),
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
    # the mutation check
    ("results/mutation_check.txt", "437 of 437 mutants caught.",
     "437 hand-written mutants, all caught"),
    # the rehearsal (tables R2 to R8)
    ("results/rehearsal/rehearsal.manifest.txt", "world: 200 events of 30.0 hours",
     "a world of 200 events (30 hours of trading before each release)"),
    ("results/rehearsal/tables.md", "| P2 | 0.0001 | 0.0003 | supported |",
     "P2 comes out supported in R8"),
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
