"""Test configuration.

Puts this checkout's ``src/`` first on ``sys.path`` and drops any ``kalshi_mm`` module imported
earlier, so the tests always exercise the sources next to them (a mutation check runs the suite
on a modified copy and relies on this).
"""

import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
for _name in [m for m in sys.modules if m == "kalshi_mm" or m.startswith("kalshi_mm.")]:
    del sys.modules[_name]
