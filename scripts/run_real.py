#!/usr/bin/env python3
"""Run docs/protocol.md on a database of real Kalshi data; see ``kalshi_mm.run_real``.

    python3 scripts/run_real.py --db data/kalshi.sqlite
"""

from __future__ import annotations

import sys
from pathlib import Path


def _run() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from kalshi_mm.run_real import main

    return main()


if __name__ == "__main__":
    sys.exit(_run())
