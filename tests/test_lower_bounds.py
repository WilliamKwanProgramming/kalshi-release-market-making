"""The pins of constraints/lowest.txt are the lower bounds of pyproject.toml, one for one."""

import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def bounds() -> dict[str, str]:
    """``name: version`` of every ``name>=version`` requirement of pyproject.toml."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    requirements = [*project["dependencies"],
                    *[r for group in project["optional-dependencies"].values() for r in group]]
    found = {}
    for requirement in requirements:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)>=([0-9][0-9A-Za-z.]*)", requirement.strip())
        assert match, f"not a plain lower bound: {requirement!r}"
        found[match.group(1).lower()] = match.group(2)
    return found


def pins() -> dict[str, str]:
    """``name: version`` of every ``name==version`` line of constraints/lowest.txt."""
    found = {}
    for line in (REPO / "constraints" / "lowest.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([0-9][0-9A-Za-z.]*)", line)
            assert match, f"not an exact pin: {line!r}"
            found[match.group(1).lower()] = match.group(2)
    return found


def test_every_lower_bound_is_pinned_at_its_value_and_nothing_else_is_pinned():
    assert pins() == bounds()


def test_the_bounds_name_the_four_libraries_and_the_linter():
    assert set(bounds()) == {"numpy", "pandas", "scipy", "pytest", "ruff"}


def test_the_ci_job_installs_with_the_constraints_file():
    ci = (REPO / ".github" / "workflows" / "ci.yml").read_text()
    assert 'pip install -c constraints/lowest.txt -e ".[test]"' in ci
    assert "pytest -q -W error" in ci
