"""The printouts and the tag check of the real-data runner (kalshi_mm.run_real)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_mm import run_real
from kalshi_mm.run_real import describe_dataset, tag_problem

HOUR = 3600.0


def _hand_dataset():
    events = pd.DataFrame(
        {"event": ["A1", "A2", "B1"], "release_ts": [1_000_000.0, 2_000_000.0, 3_000_000.0],
         "series": ["CPI", "CPI", "JOBS"]}
    )
    markets = pd.DataFrame(
        {"market": ["m1", "m2", "m3", "m4"], "event": ["A1", "A1", "A2", "B1"],
         "outcome": [1, 0, np.nan, 1], "open_ts": 0.0,
         "close_ts": [1_000_000.0 + HOUR, 1_000_000.0 - 2 * HOUR, 2_000_000.0,
                      3_000_000.0 + 10 * HOUR]}
    )
    tape = pd.DataFrame(
        {"event": ["A1", "A1", "A1", "B1"], "market": ["m1", "m1", "m2", "m4"],
         "ts": [10.0, 20.0, 30.0, 40.0], "seq": [0, 1, 0, 0], "price": 0.5, "size": 1.0,
         "side": [1, -1, 1, 1]}
    )
    return tape, markets, events


def test_describe_dataset_counts_by_series_and_quantiles_of_the_gap():
    text = describe_dataset(*_hand_dataset())
    rows = {line.split()[0]: line.split()[1:] for line in text.splitlines()
            if line.split() and line.split()[0] in {"CPI", "JOBS", "all"}}
    # events, markets, settled markets, trades
    assert rows == {"CPI": ["2", "3", "2", "3"], "JOBS": ["1", "1", "1", "1"],
                    "all": ["3", "4", "3", "4"]}
    # gaps in hours are -2, 0, 1 and 10
    assert ("  min -2.00  5% -1.70  25% -0.50  median 0.50  75% 3.25  95% 8.65  max 10.00"
            in text)
    assert "markets closing before the release: 1 of 4" in text


def test_describe_dataset_names_the_first_middle_and_last_event():
    text = describe_dataset(*_hand_dataset())
    lines = text.splitlines()
    start = lines.index("Events to compare with the Kalshi website:")
    assert [line.split(", release ")[0].strip() for line in lines[start + 1:]] == [
        "A1 (CPI)", "A2 (CPI)", "B1 (JOBS)"]
    assert lines[start + 1].endswith("2 markets, 3 trades")
    assert lines[start + 2].endswith("1 markets, 0 trades")
    assert "1970-01-12 13:46 UTC" in lines[start + 1]  # 1,000,000 s after the epoch


def test_describe_dataset_with_one_event_names_it_once():
    tape, markets, events = _hand_dataset()
    keep = events["event"] == "B1"
    text = describe_dataset(tape[tape["event"] == "B1"], markets[markets["event"] == "B1"],
                            events[keep])
    assert text.count("B1 (JOBS)") == 1


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        check=True, capture_output=True,
    )


def test_tag_problem_in_a_temporary_repository(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "docs").mkdir()
    (repo / "src" / "a.py").write_text("x = 1\n")
    (repo / "docs" / "protocol.md").write_text("protocol\n")
    (repo / "docs" / "model.md").write_text("model\n")
    (repo / "README.md").write_text("readme\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    assert "does not exist" in tag_problem(repo)
    _git(repo, "tag", "real-run")
    assert tag_problem(repo) is None
    (repo / "README.md").write_text("changed\n")  # not a file that decides a result
    assert tag_problem(repo) is None
    (repo / "src" / "a.py").write_text("x = 2\n")
    message = tag_problem(repo)
    assert "1 file(s) differ from the tag real-run" in message and "src/a.py" in message
    _git(repo, "checkout", "--", "src/a.py")
    assert tag_problem(repo) is None
    (repo / "src" / "new.py").write_text("y = 1\n")  # untracked
    assert "src/new.py" in tag_problem(repo)
    (repo / "src" / "new.py").unlink()
    (repo / "docs" / "protocol.md").write_text("edited protocol\n")
    assert "docs/protocol.md" in tag_problem(repo)


def test_the_guarded_paths_cover_everything_that_decides_a_result():
    needed = {"src", "scripts", "docs/protocol.md", "docs/model.md", "pyproject.toml"}
    assert needed <= set(run_real.GUARDED)
