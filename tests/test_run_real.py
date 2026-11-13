"""The printouts and the tag check of the real-data runner (kalshi_mm.run_real)."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from kalshi_mm import run_real
from kalshi_mm.panel import create_schema, write_dataset
from kalshi_mm.run_real import data_checks, describe_checks, describe_dataset, off_tick, tag_problem
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
