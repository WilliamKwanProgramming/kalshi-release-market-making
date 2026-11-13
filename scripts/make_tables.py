#!/usr/bin/env python3
"""Paste the tables written by the experiments into README.md.

Every experiment writes results/<name>/tables.md. A table there starts with a heading line
``### <ID>. <title>`` (an ID is one capital letter and a number, such as ``B2``) and runs to the
next such heading. README.md marks where a table goes with a pair of comment lines

    <!-- table:B2 -->
    <!-- /table:B2 -->

and this script replaces everything between a pair by the table of that ID, so no number inside
a README table is typed by hand. Standard library only.

Usage:
    python3 scripts/make_tables.py            rewrite README.md
    python3 scripts/make_tables.py --check    exit status 1 if README.md is not up to date
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^### ([A-Z]\d+)\. ", re.MULTILINE)
MARKED = re.compile(r"<!-- table:([A-Z]\d+) -->.*?<!-- /table:\1 -->", re.DOTALL)


def read_tables(results: Path) -> dict[str, str]:
    """Map table ID to its Markdown block, from every results/*/tables.md."""
    tables: dict[str, str] = {}
    for path in sorted(results.glob("*/tables.md")):
        text = path.read_text(encoding="utf-8")
        starts = [(m.start(), m.group(1)) for m in HEADING.finditer(text)]
        for i, (begin, table_id) in enumerate(starts):
            end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
            if table_id in tables:
                raise ValueError(f"table {table_id} occurs twice (second time in {path})")
            tables[table_id] = text[begin:end].strip()
    return tables


def fill(readme: str, tables: dict[str, str]) -> str:
    """README text with every marked pair replaced by the table of its ID."""
    opens = re.findall(r"<!-- table:([A-Z]\d+) -->", readme)
    closes = re.findall(r"<!-- /table:([A-Z]\d+) -->", readme)
    if opens != closes:
        raise ValueError(f"table markers do not pair up: opening {opens}, closing {closes}")

    def block(match: re.Match[str]) -> str:
        table_id = match.group(1)
        if table_id not in tables:
            raise ValueError(f"README.md asks for table {table_id}, which no tables.md defines")
        return f"<!-- table:{table_id} -->\n\n{tables[table_id]}\n\n<!-- /table:{table_id} -->"

    return MARKED.sub(block, readme)


def main() -> int:
    """Paste the tables of ``results/*/tables.md`` into the README, or compare with ``--check``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="do not write, only compare")
    ap.add_argument("--readme", type=Path, default=REPO / "README.md")
    ap.add_argument("--results", type=Path, default=REPO / "results")
    args = ap.parse_args()
    old = args.readme.read_text(encoding="utf-8")
    try:
        new = fill(old, read_tables(args.results))
    except ValueError as err:
        print(f"make_tables: {err}")
        return 1
    if args.check:
        if new != old:
            print(f"{args.readme.name} does not match the tables in {args.results}; "
                  "run scripts/make_tables.py")
            return 1
        return 0
    if new != old:
        args.readme.write_text(new, encoding="utf-8")
    print(f"{args.readme.name}: {len(MARKED.findall(old))} tables pasted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
