"""Tests of the two scripts that keep README.md and results/ in step.

scripts/make_tables.py pastes the tables of results/*/tables.md into README.md, and
scripts/check_readme_numbers.py is the check that CI runs on the result. A check that nobody tests
could pass for the wrong reason, so each failure it is meant to catch is made here on a small
README and a small results directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

TABLE_A1 = "### A1. First (synthetic data)\n\nReading: one.\n\n| a | b |\n|---|---|\n| 1 | 2 |"
TABLE_A2 = "### A2. Second (synthetic data)\n\nReading: two.\n\n| c |\n|---|\n| 3 |"


@pytest.fixture
def scripts(monkeypatch):
    """The two scripts as modules, imported fresh and removed again afterwards."""
    monkeypatch.syspath_prepend(str(SCRIPTS))
    for name in ("make_tables", "check_readme_numbers"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    import check_readme_numbers
    import make_tables

    yield make_tables, check_readme_numbers
    for name in ("make_tables", "check_readme_numbers"):
        sys.modules.pop(name, None)


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def readme_with(table_id: str, body: str = "OLD") -> str:
    return f"intro\n\n<!-- table:{table_id} -->\n{body}\n<!-- /table:{table_id} -->\n\noutro\n"


def test_fill_replaces_what_is_between_the_markers_and_is_idempotent(scripts):
    make_tables, _ = scripts
    filled = make_tables.fill(readme_with("A1"), {"A1": TABLE_A1})
    assert filled == (
        f"intro\n\n<!-- table:A1 -->\n\n{TABLE_A1}\n\n<!-- /table:A1 -->\n\noutro\n")
    assert make_tables.fill(filled, {"A1": TABLE_A1}) == filled


def test_fill_changes_each_pair_to_the_table_of_its_own_id(scripts):
    make_tables, _ = scripts
    text = readme_with("A1") + readme_with("A2")
    filled = make_tables.fill(text, {"A1": TABLE_A1, "A2": TABLE_A2})
    assert filled.index(TABLE_A1) < filled.index(TABLE_A2)
    assert filled.count(TABLE_A1) == 1 and filled.count(TABLE_A2) == 1


def test_fill_refuses_markers_that_do_not_pair_up_and_ids_without_a_table(scripts):
    make_tables, _ = scripts
    with pytest.raises(ValueError, match="do not pair up"):
        make_tables.fill("<!-- table:A1 -->\n", {"A1": TABLE_A1})
    with pytest.raises(ValueError, match="do not pair up"):
        make_tables.fill("<!-- table:A1 -->\n<!-- /table:A2 -->\n", {"A1": TABLE_A1})
    with pytest.raises(ValueError, match="A2"):
        make_tables.fill(readme_with("A2"), {"A1": TABLE_A1})


def test_a_readme_without_markers_is_left_alone(scripts):
    make_tables, _ = scripts
    assert make_tables.fill("no tables here\n", {"A1": TABLE_A1}) == "no tables here\n"


def test_read_tables_splits_each_file_at_its_headings(scripts, tmp_path):
    make_tables, _ = scripts
    write(tmp_path / "x" / "tables.md", TABLE_A1 + "\n\n" + TABLE_A2 + "\n")
    write(tmp_path / "y" / "tables.md", "### B1. Third\n\nbody\n")
    assert make_tables.read_tables(tmp_path) == {
        "A1": TABLE_A1, "A2": TABLE_A2, "B1": "### B1. Third\n\nbody"}


def test_read_tables_refuses_an_id_that_two_files_define(scripts, tmp_path):
    make_tables, _ = scripts
    write(tmp_path / "x" / "tables.md", TABLE_A1)
    write(tmp_path / "y" / "tables.md", TABLE_A1)
    with pytest.raises(ValueError, match="A1 occurs twice"):
        make_tables.read_tables(tmp_path)


def test_make_tables_writes_the_readme_and_check_mode_only_compares(
        scripts, tmp_path, monkeypatch, capsys):
    make_tables, _ = scripts
    results = tmp_path / "results"
    write(results / "x" / "tables.md", TABLE_A1)
    readme = write(tmp_path / "README.md", readme_with("A1"))
    base = ["make_tables.py", "--readme", str(readme), "--results", str(results)]
    monkeypatch.setattr(sys, "argv", [*base, "--check"])
    assert make_tables.main() == 1
    assert "does not match" in capsys.readouterr().out
    assert readme.read_text(encoding="utf-8") == readme_with("A1")  # --check does not write
    monkeypatch.setattr(sys, "argv", base)
    assert make_tables.main() == 0
    assert TABLE_A1 in readme.read_text(encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*base, "--check"])
    assert make_tables.main() == 0


def test_make_tables_reports_a_table_that_no_results_file_defines(
        scripts, tmp_path, monkeypatch, capsys):
    make_tables, _ = scripts
    results = tmp_path / "results"
    write(results / "x" / "tables.md", TABLE_A1)
    readme = write(tmp_path / "README.md", readme_with("A2"))
    monkeypatch.setattr(sys, "argv", [
        "make_tables.py", "--check", "--readme", str(readme), "--results", str(results)])
    assert make_tables.main() == 1
    assert "A2" in capsys.readouterr().out


def make_repo(tmp_path: Path, readme_body: str = TABLE_A1, extra: str = "") -> Path:
    """A small repository: README.md with one table and a results file that has the same."""
    text = readme_with("A1", f"\n{readme_body}\n") + extra
    write(tmp_path / "README.md", text)
    write(tmp_path / "results" / "x" / "tables.md", TABLE_A1 + "\n")
    write(tmp_path / "results" / "x" / "value.txt", "value: 17.25 cents\n")
    write(tmp_path / "docs" / "model.md", "model\n")
    return tmp_path


def run_check(check, tmp_path, monkeypatch, checks, capsys):
    monkeypatch.setattr(check, "ROOT", tmp_path)
    monkeypatch.setattr(check, "CHECKS", checks)
    code = check.main()
    return code, capsys.readouterr().out


def test_the_check_passes_on_a_consistent_repository(scripts, tmp_path, monkeypatch, capsys):
    _, check = scripts
    make_repo(tmp_path, extra="The value is 17.25 cents. See [the model](docs/model.md#top).\n")
    checks = [("results/x/value.txt", "value: 17.25 cents", "is 17.25 cents")]
    code, out = run_check(check, tmp_path, monkeypatch, checks, capsys)
    assert code == 0, out
    assert "1 number checks and the table and link checks, 0 problems" in out


def test_the_check_fails_when_a_pasted_table_differs_from_its_results_file(
        scripts, tmp_path, monkeypatch, capsys):
    _, check = scripts
    make_repo(tmp_path, readme_body=TABLE_A1.replace("| 1 | 2 |", "| 1 | 3 |"))
    code, out = run_check(check, tmp_path, monkeypatch, [], capsys)
    assert code == 1
    assert "tables differ" in out


def test_the_check_fails_when_the_result_no_longer_has_the_number(
        scripts, tmp_path, monkeypatch, capsys):
    _, check = scripts
    make_repo(tmp_path, extra="The value is 17.25 cents.\n")
    checks = [("results/x/value.txt", "value: 18.00 cents", "is 17.25 cents")]
    code, out = run_check(check, tmp_path, monkeypatch, checks, capsys)
    assert code == 1
    assert "'value: 18.00 cents' not found" in out


def test_the_check_fails_when_the_readme_no_longer_quotes_the_number(
        scripts, tmp_path, monkeypatch, capsys):
    _, check = scripts
    make_repo(tmp_path, extra="The value is 20 cents.\n")
    checks = [("results/x/value.txt", "value: 17.25 cents", "is 17.25 cents")]
    code, out = run_check(check, tmp_path, monkeypatch, checks, capsys)
    assert code == 1
    assert "README.md does not contain 'is 17.25 cents'" in out


def test_the_check_fails_when_a_results_file_is_missing(scripts, tmp_path, monkeypatch, capsys):
    _, check = scripts
    make_repo(tmp_path, extra="The value is 17.25 cents.\n")
    checks = [("results/y/missing.txt", "value", "is 17.25 cents")]
    code, out = run_check(check, tmp_path, monkeypatch, checks, capsys)
    assert code == 1
    assert "results/y/missing.txt: file not found" in out


def test_the_check_fails_on_a_link_to_a_file_that_does_not_exist(
        scripts, tmp_path, monkeypatch, capsys):
    _, check = scripts
    make_repo(tmp_path, extra="See [the protocol](docs/protocol.md) and [web](https://x.org/a).\n")
    code, out = run_check(check, tmp_path, monkeypatch, [], capsys)
    assert code == 1
    assert "docs/protocol.md" in out
    assert "x.org" not in out


def test_broken_links_ignores_web_anchor_and_mail_links_and_strips_anchors(scripts, tmp_path):
    _, check = scripts
    write(tmp_path / "docs" / "model.md", "model\n")
    readme = ("[a](https://x.org) [b](http://x.org) [c](#top) [d](mailto:a@b.org) "
              "[e](docs/model.md#section) ![f](docs/missing.png) [g](docs/other.md)")
    check.ROOT = tmp_path
    assert check.broken_links(readme) == ["docs/missing.png", "docs/other.md"]
