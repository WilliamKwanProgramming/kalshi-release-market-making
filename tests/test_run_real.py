"""The printouts and the tag check of the real-data runner (kalshi_mm.run_real)."""

from __future__ import annotations

import datetime
import json
import os
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_mm import run_real
from kalshi_mm.panel import begin_fetch_run, create_schema, end_fetch_run, write_dataset
from kalshi_mm.run_real import (
    data_checks,
    describe_checks,
    describe_dataset,
    off_tick,
    off_tick_share,
    tag_created,
    tag_problem,
)
from kalshi_mm.world import WorldParams, generate_world

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


def _git(repo: Path, *args: str, when: str | None = None) -> None:
    """Run git in ``repo``; ``when`` (ISO 8601) is the committer and tagger date."""
    env = {**os.environ, "GIT_COMMITTER_DATE": when} if when else None
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        check=True, capture_output=True, env=env,
    )


def _annotate(repo: Path, when: str) -> None:
    """The tag ``real-run`` on HEAD, annotated and created at ``when``."""
    _git(repo, "tag", "-a", "real-run", "-m", "the code and the protocol of the run", when=when)


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


# ------------------------------------------------------------------------- checks of the data
def test_off_tick_flags_prices_more_than_the_tolerance_from_the_grid():
    prices = [0.56, 0.01, 0.99, 0.565, 0.5605, 0.56 + 0.9e-9, 0.56 - 0.9e-9, 0.56 + 1.1e-9,
              0.56 - 1.1e-9, 0.123456]
    assert off_tick(prices).tolist() == [False, False, False, True, True, False, False, True,
                                         True, True]
    assert run_real.TICK == 0.01 and run_real.TICK_TOLERANCE == 1e-9
    assert off_tick([0.565], tick=0.005).tolist() == [False]


def _api_fields():
    """Stored fields of the hand dataset: m1's three trades and m4's one."""
    trades = pd.DataFrame({
        "market": ["m1", "m1", "m2", "m4", "m4"], "ts": [10.0, 10.0, 30.0, 40.0, 50.0],
        "price": [0.5, 0.5, 0.555, 0.5, 0.0101], "size": 1.0, "side": [1, 1, 1, 1, -1],
        "trade_id": ["a", "b", "c", "d", "e"],
        "endpoint": ["historical", "live", "both", "both", None]})
    markets = pd.DataFrame({"market": ["m1", "m2", "m3", "m4"],
                            "price_level_structure": ["linear_cent", None, "deci_cent",
                                                      "linear_cent"],
                            "price_ranges": None})
    return trades, markets


def test_data_checks_count_by_series_and_in_total():
    _, markets, events = _hand_dataset()
    checks = data_checks(markets, events, *_api_fields())
    assert checks["by_series"] == {
        "CPI": {"trades kept": 3, "off the 0.01 tick": 1, "trade_id from both endpoints": 1,
                "groups alike across endpoints": 1},
        "JOBS": {"trades kept": 2, "off the 0.01 tick": 1, "trade_id from both endpoints": 1,
                 "groups alike across endpoints": 0},
        "all": {"trades kept": 5, "off the 0.01 tick": 2, "trade_id from both endpoints": 2,
                "groups alike across endpoints": 1}}
    assert checks["markets by price_level_structure"] == {"deci_cent": 1, "linear_cent": 2,
                                                          "not recorded": 1}
    text = describe_checks(checks)
    assert "The backtest quotes on a 0.01 USD tick; 2 of the 5 kept trades are at prices off it." \
        in text
    assert "deci_cent 1, linear_cent 2, not recorded 1" in text


def _real_like_database(path):
    """A small synthetic world stored as if it came from the API, with two prices off the tick,
    one trade marked as read from both endpoints and a price grid on every market."""
    world = generate_world(WorldParams(n_events=26, window_hours=8.0), seed=3)
    tape = world.tape.copy()
    tape["trade_id"] = [f"id{i}" for i in range(len(tape))]
    tape["endpoint"] = "historical"
    tape.loc[tape.index[0], "endpoint"] = "both"
    tape.loc[tape.index[[5, 9]], "price"] += 0.005
    markets = world.markets.assign(price_level_structure="linear_cent", price_ranges="[]")
    with closing(sqlite3.connect(path)) as conn:
        create_schema(conn)
        write_dataset(conn, tape, markets, world.events, source="kalshi-public-api")
        conn.commit()
    return len(tape)


SMALL = ["n_boot=200", "recal_min_releases=12", "recal_refit_every=4", "recal_hours=2",
         "quote.min_history_events=5"]


def test_run_real_prints_and_writes_the_checks_of_the_data(tmp_path, capsys, monkeypatch):
    n = _real_like_database(tmp_path / "kalshi.sqlite")
    monkeypatch.setattr(run_real, "tag_problem", lambda: None)
    out = tmp_path / "real"
    assert run_real.main(["--db", str(tmp_path / "kalshi.sqlite"), "--out", str(out),
                          *[f"--set={a}" for a in SMALL]]) == 0
    printed = capsys.readouterr().out
    assert f"The backtest quotes on a 0.01 USD tick; 2 of the {n} kept trades" in printed
    assert "Markets by price_level_structure: linear_cent" in printed
    summary = json.loads((out / "summary.json").read_text())
    assert summary["data_checks"]["by_series"]["all"] == {
        "trades kept": n, "off the 0.01 tick": 2, "trade_id from both endpoints": 1,
        "groups alike across endpoints": 0}
    tables = (out / "tables.md").read_text()
    assert "### K9. Checks of the data (real data)" in tables
    assert f"2 of the {n:,} kept trades are at prices off it" in tables
    assert "off the 0.01 tick" not in (out / "DEVIATIONS.md").read_text()  # not a departure
    manifest = (out / "run_real.manifest.txt").read_text()
    assert "check, off the 0.01 tick: 2" in manifest


def test_run_real_labels_p2_exploratory_when_too_many_trades_are_off_the_tick(
        tmp_path, capsys, monkeypatch):
    _real_like_database(tmp_path / "kalshi.sqlite")
    monkeypatch.setattr(run_real, "tag_problem", lambda: None)
    out = tmp_path / "real"
    monkeypatch.setattr(run_real, "off_tick_share", lambda checks: 0.0501)  # above the 5% limit
    assert run_real.main(["--db", str(tmp_path / "kalshi.sqlite"), "--out", str(out),
                          *[f"--set={a}" for a in SMALL]]) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["family"]["P2"]["verdict"] == "exploratory"
    tables = (out / "tables.md").read_text()
    assert ("P2 is exploratory because 5.0% of the kept trades are off the 0.01 USD tick, more "
            "than the 5% that the protocol allows.") in tables
    assert "off the 0.01 USD tick" not in (out / "DEVIATIONS.md").read_text()  # not a departure


def test_run_real_does_not_label_p2_when_the_off_tick_share_is_within_the_limit(
        tmp_path, capsys, monkeypatch):
    _real_like_database(tmp_path / "kalshi.sqlite")
    monkeypatch.setattr(run_real, "tag_problem", lambda: None)
    out = tmp_path / "real"
    monkeypatch.setattr(run_real, "off_tick_share", lambda checks: 0.05)  # at the limit: allowed
    assert run_real.main(["--db", str(tmp_path / "kalshi.sqlite"), "--out", str(out),
                          *[f"--set={a}" for a in SMALL]]) == 0
    assert "of the kept trades are off the 0.01 USD tick" not in (out / "tables.md").read_text()
    text = (out / "tables.md").read_text()
    assert "against the 5% above which P2 is labelled exploratory" in text  # K9 gives the limit


def test_run_real_refuses_a_database_of_the_older_schema(tmp_path, capsys):
    path = tmp_path / "old.sqlite"
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("INSERT INTO meta VALUES ('source', 'kalshi-public-api')")
        conn.execute("CREATE TABLE markets (market TEXT PRIMARY KEY, event TEXT, outcome INTEGER,"
                     " open_ts REAL, close_ts REAL)")
        conn.commit()
    assert run_real.main(["--db", str(path), "--out", str(tmp_path / "out")]) == 2
    assert "earlier version of the schema" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


# --------------------------------------------------------------- the record of the fetches
def _head(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def _run(commit, dirty="false", started="2026-11-13T10:00:00Z"):
    return {"fetch_started_utc": started, "fetch_finished_utc": "2026-11-13T11:00:00Z",
            "git_commit": commit, "git_dirty": dirty}


def test_provenance_problems_in_a_temporary_repository(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    (repo / "README.md").write_text("readme\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    tagged = _head(repo)
    assert any("cannot be compared" in p for p in run_real.provenance_problems([_run(tagged)],
                                                                               repo))
    _annotate(repo, "2026-11-13T09:00:00Z")  # an hour before the fetches below
    assert run_real.provenance_problems([_run(tagged)], repo) == []
    (repo / "README.md").write_text("changed\n")  # not a guarded file
    _git(repo, "commit", "-q", "-am", "readme")
    readme = _head(repo)
    assert run_real.provenance_problems([_run(tagged), _run(readme)], repo) == []
    (repo / "src" / "a.py").write_text("x = 2\n")
    _git(repo, "commit", "-q", "-am", "code")
    code = _head(repo)
    problems = run_real.provenance_problems([_run(tagged), _run(code), _run(code)], repo)
    assert problems == [f"a fetch ran from commit {code}, whose guarded files differ from the "
                        "tag real-run"]  # one line per commit
    dirty = run_real.provenance_problems(
        [_run(tagged), _run(tagged, "true", "2026-11-14T09:00:00Z")], repo)
    assert dirty == ["1 of 2 fetches ran from a repository with uncommitted changes "
                     "(git_dirty true), the first started 2026-11-14T09:00:00Z"]
    assert run_real.provenance_problems([_run(tagged, "unknown")], repo) == []
    unknown = run_real.provenance_problems([_run("unknown", "unknown")], repo)
    assert unknown == ["a fetch has no recorded commit (git was unavailable)"]
    missing = run_real.provenance_problems([_run("0" * 40)], repo)
    assert len(missing) == 1 and "cannot be compared with the tag real-run" in missing[0]
    assert "no record of its fetch" in run_real.provenance_problems([], repo)[0]


def _repository(tmp_path):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("x = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    return repo, _head(repo)


def test_tag_created_is_the_date_of_an_annotated_tag_and_nothing_else(tmp_path):
    repo, _ = _repository(tmp_path)
    assert tag_created(repo) is None  # no tag
    _git(repo, "tag", "real-run")
    assert tag_created(repo) is None  # a lightweight tag has no creation time
    _git(repo, "tag", "-d", "real-run")
    _annotate(repo, "2026-11-13T09:30:00Z")
    created = tag_created(repo)
    assert created == datetime.datetime(2026, 11, 13, 9, 30, tzinfo=datetime.UTC)
    assert created.tzinfo is not None  # aware: comparable with the times of the fetches


def test_a_lightweight_tag_is_a_departure_because_it_has_no_creation_time(tmp_path):
    repo, commit = _repository(tmp_path)
    _git(repo, "tag", "real-run")
    assert run_real.tag_problem(repo) is None  # the content check passes
    assert run_real.provenance_problems([_run(commit)], repo) == [
        "the tag real-run is not annotated (git tag -a), so it has no creation time and it "
        "cannot be shown that it was made before the fetch"]


def test_a_fetch_that_started_before_the_tag_was_created_is_a_departure(tmp_path):
    repo, commit = _repository(tmp_path)
    _annotate(repo, "2026-11-13T10:00:00Z")
    assert run_real.provenance_problems([_run(commit, started="2026-11-13T10:00:00Z")],
                                        repo) == []  # at the same second: not before
    before = run_real.provenance_problems([_run(commit, started="2026-11-13T09:59:59Z")], repo)
    assert before == ["a fetch started 2026-11-13T09:59:59+00:00, before the tag real-run was "
                      "created (2026-11-13T10:00:00+00:00)"]
    # the earliest of several fetches counts, whatever its place in the record
    runs = [_run(commit, started="2026-11-14T10:00:00Z"),
            _run(commit, started="2026-11-12T23:00:00Z")]
    assert run_real.provenance_problems(runs, repo) == [
        "a fetch started 2026-11-12T23:00:00+00:00, before the tag real-run was created "
        "(2026-11-13T10:00:00+00:00)"]
    # a start time without an offset is read as UTC, the zone of the record
    assert run_real.provenance_problems([_run(commit, started="2026-11-13T09:00:00")], repo)
    assert run_real.provenance_problems([_run(commit, started="2026-11-13T11:00:00")], repo) == []
    # a time zone is respected: 06:00 at -05:00 is 11:00 UTC
    assert run_real.provenance_problems(
        [_run(commit, started="2026-11-13T06:00:00-05:00")], repo) == []


def test_a_start_time_that_cannot_be_read_is_a_departure(tmp_path):
    repo, commit = _repository(tmp_path)
    _annotate(repo, "2026-11-13T10:00:00Z")
    for bad in ("yesterday", None):
        problems = run_real.provenance_problems([_run(commit, started=bad)], repo)
        assert problems == [f"a fetch has no readable start time ({bad!r}), so it cannot be "
                            "compared with the creation time of the tag real-run"]


def test_without_the_tag_the_date_gives_no_departure_of_its_own(tmp_path):
    repo, commit = _repository(tmp_path)
    problems = run_real.provenance_problems([_run(commit)], repo)
    assert len(problems) == 1 and "cannot be compared with the tag real-run" in problems[0]


def test_off_tick_share_is_the_share_of_the_kept_trades_off_the_tick():
    _, markets, events = _hand_dataset()
    checks = data_checks(markets, events, *_api_fields())
    assert off_tick_share(checks) == 2 / 5
    none = {"by_series": {"all": {"trades kept": 0, "off the 0.01 tick": 0}}}
    assert off_tick_share(none) is None  # nothing was kept: no share, no label


def test_run_real_prints_the_fetches_and_writes_their_departures(tmp_path, capsys, monkeypatch):
    _real_like_database(tmp_path / "kalshi.sqlite")
    with closing(sqlite3.connect(tmp_path / "kalshi.sqlite")) as conn:
        begin_fetch_run(conn, started_utc="2026-11-13T10:00:00Z", git_commit="abc123",
                        git_dirty="true")
        end_fetch_run(conn, finished_utc="2026-11-13T11:00:00Z")
    monkeypatch.setattr(run_real, "tag_problem", lambda: None)
    monkeypatch.setattr(run_real, "provenance_problems",
                        lambda runs: [f"seen {len(runs)} run, dirty {runs[0]['git_dirty']}"])
    out = tmp_path / "real"
    assert run_real.main(["--db", str(tmp_path / "kalshi.sqlite"), "--out", str(out),
                          *[f"--set={a}" for a in SMALL]]) == 0
    printed = capsys.readouterr().out
    assert ("started 2026-11-13T10:00:00Z, finished 2026-11-13T11:00:00Z, commit abc123, "
            "uncommitted changes true") in printed
    assert "departure: seen 1 run, dirty true" in printed
    assert "- seen 1 run, dirty true" in (out / "DEVIATIONS.md").read_text()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["data_checks"]["fetch_runs"][0]["git_commit"] == "abc123"
    assert "seen 1 run, dirty true" in summary["deviations"]
    manifest = (out / "run_real.manifest.txt").read_text()
    assert "fetch 1: fetch_started_utc 2026-11-13T10:00:00Z" in manifest


def test_a_database_without_a_record_of_its_fetch_is_a_departure(tmp_path, monkeypatch):
    _real_like_database(tmp_path / "kalshi.sqlite")
    monkeypatch.setattr(run_real, "tag_problem", lambda: None)
    out = tmp_path / "real"
    assert run_real.main(["--db", str(tmp_path / "kalshi.sqlite"), "--out", str(out),
                          *[f"--set={a}" for a in SMALL]]) == 0
    assert "no record of its fetch" in (out / "DEVIATIONS.md").read_text()
