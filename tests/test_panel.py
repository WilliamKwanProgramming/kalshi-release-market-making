"""SQLite storage of the data contract (src/kalshi_mm/panel.py; docs/model.md sections 4 and 5).

The synthetic world of ``kalshi_mm.world`` gives datasets of known content; small hand-built
datasets give exact values for the constraints and the replacement rules.
"""

import sqlite3

import numpy as np
import pandas as pd
import pytest

from kalshi_mm import panel
from kalshi_mm.world import WorldParams, generate_world

SEED = 20261113
TAPE_COLUMNS = ["event", "market", "ts", "seq", "price", "size", "side"]
MARKET_COLUMNS = ["market", "event", "outcome", "open_ts", "close_ts"]
EVENT_COLUMNS = ["event", "release_ts", "series"]


@pytest.fixture(scope="module")
def world():
    return generate_world(WorldParams(n_events=30), SEED)


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    panel.create_schema(connection)
    yield connection
    connection.close()


def small_dataset():
    """Two markets of one event: M1 has two trades and the outcome 1, M2 one trade and 0."""
    events = pd.DataFrame({"event": ["E1"], "release_ts": [1000.0], "series": ["S"]})
    markets = pd.DataFrame({
        "market": ["M1", "M2"], "event": ["E1", "E1"], "outcome": np.array([1, 0], dtype=np.int64),
        "open_ts": [0.0, 0.0], "close_ts": [1000.0, 1000.0]})
    tape = pd.DataFrame({
        "event": ["E1", "E1", "E1"], "market": ["M1", "M1", "M2"], "ts": [10.0, 20.0, 15.0],
        "seq": np.array([0, 1, 0], dtype=np.int64), "price": [0.40, 0.45, 0.30],
        "size": [5.0, 1.0, 2.0], "side": np.array([1, -1, 1], dtype=np.int8)})
    return tape, markets, events


def counts(conn):
    return [conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("events", "markets", "trades", "meta")]


# ------------------------------------------------------------------------------- round trip
def test_round_trip_of_a_synthetic_world_is_exact(conn, world):
    panel.write_dataset(conn, world.tape, world.markets, world.events, source="synthetic")
    tape, markets, events = panel.read_dataset(conn)
    pd.testing.assert_frame_equal(tape, world.tape[TAPE_COLUMNS], check_exact=True)
    pd.testing.assert_frame_equal(markets, world.markets[MARKET_COLUMNS], check_exact=True)
    pd.testing.assert_frame_equal(events, world.events[EVENT_COLUMNS], check_exact=True)
    assert panel.get_source(conn) == "synthetic"
    assert tape["side"].dtype == np.int8 and tape["seq"].dtype == np.int64
    assert len(tape) > 3000  # a world of 30 events is not trivially small


def test_the_database_has_no_latent_columns(conn, world):
    panel.write_dataset(conn, world.tape, world.markets, world.events, source="synthetic")
    for table in ("events", "markets", "trades"):
        names = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        assert not [n for n in names if n.startswith("latent_")]
    tape, markets, events = panel.read_dataset(conn)
    assert not [c for f in (tape, markets, events) for c in f.columns if c.startswith("latent_")]


def test_frames_in_any_row_order_come_back_in_the_order_of_the_contract(conn, world):
    frames = (world.tape, world.markets, world.events)
    shuffled = [f.sample(frac=1.0, random_state=3) for f in frames]
    panel.write_dataset(conn, *shuffled, source="synthetic")
    tape, markets, events = panel.read_dataset(conn)
    pd.testing.assert_frame_equal(tape, world.tape[TAPE_COLUMNS], check_exact=True)
    pd.testing.assert_frame_equal(markets, world.markets[MARKET_COLUMNS], check_exact=True)
    pd.testing.assert_frame_equal(events, world.events[EVENT_COLUMNS], check_exact=True)


def test_events_and_markets_are_ordered_by_release_time_not_by_name(conn):
    tape, markets, events = small_dataset()
    events = pd.DataFrame({"event": ["B", "A"], "release_ts": [100.0, 200.0], "series": "S"})
    markets = pd.DataFrame({"market": ["B-1", "A-1", "A-0"], "event": ["B", "A", "A"],
                            "outcome": [1, 0, 1], "open_ts": 0.0, "close_ts": 100.0})
    tape = tape.iloc[:0]
    panel.write_dataset(conn, tape, markets, events, source="synthetic")
    _, markets_back, events_back = panel.read_dataset(conn)
    assert events_back["event"].tolist() == ["B", "A"]  # release 100 before release 200
    assert markets_back["market"].tolist() == ["B-1", "A-0", "A-1"]  # by event release, then name


def test_integer_valued_times_are_stored_as_numbers_and_read_as_float64(conn):
    tape, markets, events = small_dataset()
    tape["ts"] = np.array([10, 20, 15], dtype=np.int64)
    markets["open_ts"] = np.array([0, 0], dtype=np.int64)
    events["release_ts"] = np.array([1000], dtype=np.int64)
    panel.write_dataset(conn, tape, markets, events, source="synthetic")
    back, _, _ = panel.read_dataset(conn)
    assert back["ts"].dtype == np.float64 and back["ts"].tolist() == [10.0, 20.0, 15.0]
    types = conn.execute("SELECT DISTINCT typeof(ts), typeof(seq), typeof(price), typeof(size), "
                         "typeof(side) FROM trades").fetchall()
    assert types == [("real", "integer", "real", "real", "integer")]


def test_missing_outcomes_are_nan_and_the_other_outcomes_survive(conn):
    tape, markets, events = small_dataset()
    markets["outcome"] = pd.array([1, None], dtype="Int64")
    panel.write_dataset(conn, tape, markets, events, source="synthetic")
    _, back, _ = panel.read_dataset(conn)
    assert back["outcome"].dtype == np.float64
    assert back["outcome"].iloc[0] == 1.0 and np.isnan(back["outcome"].iloc[1])
    assert conn.execute("SELECT outcome FROM markets ORDER BY market").fetchall() == [(1,), (None,)]
    markets["outcome"] = np.array([np.nan, 0.0])  # a float column with NaN is the same thing
    panel.write_dataset(conn, tape, markets, events, source="synthetic")
    assert conn.execute("SELECT outcome FROM markets ORDER BY market").fetchall() == [(None,), (0,)]


def test_an_empty_database_reads_empty_frames_with_the_contract_columns(conn):
    tape, markets, events = panel.read_dataset(conn)
    assert list(tape.columns) == TAPE_COLUMNS and len(tape) == 0
    assert list(markets.columns) == MARKET_COLUMNS and len(markets) == 0
    assert list(events.columns) == EVENT_COLUMNS and len(events) == 0
    assert tape["ts"].dtype == np.float64 and tape["seq"].dtype == np.int64
    assert tape["side"].dtype == np.int8 and markets["outcome"].dtype == np.int64


# ------------------------------------------------------------------------------------ source
def test_source_is_stored_and_read_back(conn):
    with pytest.raises(ValueError, match="no source"):
        panel.get_source(conn)
    panel.write_dataset(conn, *small_dataset(), source="kalshi-public-api")
    assert panel.get_source(conn) == "kalshi-public-api"
    assert conn.execute("SELECT key, value FROM meta").fetchall() == [
        ("source", "kalshi-public-api")]


def test_writing_with_another_source_raises_and_changes_nothing(conn):
    tape, markets, events = small_dataset()
    panel.write_dataset(conn, tape, markets, events, source="synthetic")
    before = counts(conn)
    markets2 = markets.assign(market=["N1", "N2"])
    tape2 = tape.assign(market=["N1", "N1", "N2"])
    with pytest.raises(ValueError, match="synthetic"):
        panel.write_dataset(conn, tape2, markets2, events, source="kalshi-public-api")
    assert counts(conn) == before and panel.get_source(conn) == "synthetic"
    panel.write_dataset(conn, tape2, markets2, events, source="synthetic")  # same source is fine
    assert counts(conn)[1] == 4


@pytest.mark.parametrize("bad", ["", "real", "Synthetic", None, 3])
def test_unknown_sources_are_refused(conn, bad):
    with pytest.raises(ValueError, match="source"):
        panel.write_dataset(conn, *small_dataset(), source=bad)
    assert counts(conn) == [0, 0, 0, 0]


# ----------------------------------------------------------------- replacing and idempotence
def test_writing_the_same_dataset_twice_changes_nothing(conn, world):
    panel.write_dataset(conn, world.tape, world.markets, world.events, source="synthetic")
    first_counts, first = counts(conn), panel.read_dataset(conn)
    panel.write_dataset(conn, world.tape, world.markets, world.events, source="synthetic")
    assert counts(conn) == first_counts
    for a, b in zip(first, panel.read_dataset(conn), strict=True):
        pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_the_trades_of_a_written_market_are_replaced_the_others_are_kept(conn):
    tape, markets, events = small_dataset()
    panel.write_dataset(conn, tape, markets, events, source="synthetic")
    new = tape[tape["market"] == "M1"].iloc[:1].assign(price=0.77)
    panel.write_dataset(conn, new, markets.iloc[:1], events, source="synthetic")
    back, _, _ = panel.read_dataset(conn)
    assert back[["market", "ts", "price"]].values.tolist() == [
        ["M1", 10.0, 0.77], ["M2", 15.0, 0.30]]
    panel.write_dataset(conn, tape.iloc[:0], markets.iloc[:1], events, source="synthetic")
    assert panel.read_dataset(conn)[0]["market"].tolist() == ["M2"]  # M1 is now written empty


def test_a_later_write_updates_an_outcome_that_was_missing(conn):
    tape, markets, events = small_dataset()
    markets["outcome"] = pd.array([None, 0], dtype="Int64")
    panel.write_dataset(conn, tape, markets, events, source="kalshi-public-api")
    markets["outcome"] = pd.array([1, 0], dtype="Int64")
    panel.write_dataset(conn, tape, markets, events, source="kalshi-public-api")
    assert panel.read_dataset(conn)[1]["outcome"].tolist() == [1, 0]


def test_loaded_markets_and_stored_release_times(conn):
    assert panel.loaded_markets(conn) == set() and panel.stored_release_times(conn) == {}
    panel.write_dataset(conn, *small_dataset(), source="synthetic")
    assert panel.loaded_markets(conn) == {"M1", "M2"}
    assert panel.stored_release_times(conn) == {"E1": 1000.0}


# --------------------------------------------------------------------------------- constraints
@pytest.mark.parametrize("column, value", [
    ("price", 1.0), ("price", 0.0), ("price", -0.2), ("price", 1.5), ("price", np.nan),
    ("size", 0.0), ("size", -1.0), ("size", np.nan), ("ts", np.nan), ("side", 0), ("side", 2),
])
def test_a_trade_that_breaks_a_constraint_raises_and_nothing_is_written(conn, column, value):
    tape, markets, events = small_dataset()
    tape.loc[1, column] = value if column != "side" else np.int8(value)
    tape["side"] = tape["side"].astype(np.int8)
    with pytest.raises(sqlite3.IntegrityError):
        panel.write_dataset(conn, tape, markets, events, source="synthetic")
    assert counts(conn) == [0, 0, 0, 0]  # the events, markets and source were rolled back too


def test_a_repeated_market_ts_seq_raises(conn):
    tape, markets, events = small_dataset()
    tape.loc[1, ["ts", "seq"]] = [10.0, 0]  # the same key as the first trade of M1
    tape["seq"] = tape["seq"].astype(np.int64)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        panel.write_dataset(conn, tape, markets, events, source="synthetic")
    assert counts(conn) == [0, 0, 0, 0]


def test_equal_times_are_allowed_when_seq_differs(conn):
    tape, markets, events = small_dataset()
    tape.loc[1, ["ts", "seq"]] = [10.0, 1]
    tape["seq"] = tape["seq"].astype(np.int64)
    panel.write_dataset(conn, tape, markets, events, source="synthetic")
    assert counts(conn)[2] == 3


@pytest.mark.parametrize("sql, args", [
    ("INSERT INTO trades (market, ts, seq, price, size, side) VALUES (?, ?, ?, ?, ?, ?)",
     ("NOPE", 1.0, 0, 0.5, 1.0, 1)),  # no such market
    ("INSERT INTO markets (market, event, outcome, open_ts, close_ts) VALUES (?, ?, ?, ?, ?)",
     ("M9", "NOPE", 1, 0.0, 1.0)),  # no such event
    ("INSERT INTO markets (market, event, outcome, open_ts, close_ts) VALUES (?, ?, ?, ?, ?)",
     ("M9", "E1", 2, 0.0, 1.0)),  # outcome must be 0, 1 or NULL
    ("INSERT INTO markets (market, event, outcome, open_ts, close_ts) VALUES (?, ?, ?, ?, ?)",
     ("M9", "E1", 1, None, 1.0)),
    ("INSERT INTO events (event, series, release_ts) VALUES (?, ?, ?)", ("E9", "S", None)),
    ("INSERT INTO events (event, series, release_ts) VALUES (?, ?, ?)", ("E9", None, 1.0)),
    ("INSERT INTO trades (market, ts, seq, price, size, side) VALUES (?, ?, ?, ?, ?, ?)",
     ("M1", np.int64(7), 0, 0.5, 1.0, 1)),  # a numpy integer would be stored as a BLOB
    ("INSERT INTO trades (market, ts, seq, price, size, side) VALUES (?, ?, ?, ?, ?, ?)",
     ("M1", 7.0, np.int64(5), 0.5, 1.0, 1)),
    ("INSERT INTO trades (market, ts, seq, price, size, side) VALUES (?, ?, ?, ?, ?, ?)",
     ("M1", 7.0, 5, "cheap", 1.0, 1)),  # text that is not a number
    ("INSERT INTO trades (market, ts, seq, price, size, side) VALUES (?, ?, ?, ?, ?, ?)",
     ("M1", 7.0, 5, 0.5, "many", 1)),
])
def test_the_database_itself_refuses_bad_rows(conn, sql, args):
    panel.write_dataset(conn, *small_dataset(), source="synthetic")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(sql, args)


def test_foreign_keys_are_on_after_create_schema_and_after_write_dataset(tmp_path):
    path = tmp_path / "panel.sqlite"
    first = sqlite3.connect(path)
    panel.create_schema(first)
    assert first.execute("PRAGMA foreign_keys").fetchone() == (1,)
    first.close()
    second = sqlite3.connect(path)  # a new connection starts with foreign keys off
    assert second.execute("PRAGMA foreign_keys").fetchone() == (0,)
    panel.write_dataset(second, *small_dataset(), source="synthetic")
    assert second.execute("PRAGMA foreign_keys").fetchone() == (1,)
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        second.execute("DELETE FROM events")  # the markets still point at the event
    second.close()


def test_create_schema_twice_is_harmless_and_keeps_the_rows(conn):
    panel.write_dataset(conn, *small_dataset(), source="synthetic")
    panel.create_schema(conn)
    assert counts(conn) == [1, 2, 3, 1]


# ------------------------------------------------------------------------ contract violations
def test_missing_columns_raise_a_value_error(conn):
    tape, markets, events = small_dataset()
    for name, frames in (("tape", (tape.drop(columns="size"), markets, events)),
                         ("markets", (tape, markets.drop(columns="outcome"), events)),
                         ("events", (tape, markets, events.drop(columns="series")))):
        with pytest.raises(ValueError, match=name):
            panel.write_dataset(conn, *frames, source="synthetic")


def test_a_trade_must_belong_to_a_written_market_with_the_same_event(conn):
    tape, markets, events = small_dataset()
    with pytest.raises(ValueError, match="not in markets"):
        panel.write_dataset(conn, tape.assign(market="M3"), markets, events, source="synthetic")
    with pytest.raises(ValueError, match="differs from the event"):
        panel.write_dataset(conn, tape.assign(event="E2"), markets, events, source="synthetic")
    assert counts(conn) == [0, 0, 0, 0]


def test_repeated_identifiers_in_events_or_markets_raise(conn):
    tape, markets, events = small_dataset()
    with pytest.raises(ValueError, match="repeated"):
        panel.write_dataset(conn, tape, markets.assign(market="M1"), events, source="synthetic")
    with pytest.raises(ValueError, match="repeated"):
        panel.write_dataset(conn, tape, markets, pd.concat([events, events]), source="synthetic")


@pytest.mark.parametrize("column", ["seq", "side"])
def test_integer_columns_must_be_integers(conn, column):
    tape, markets, events = small_dataset()
    tape[column] = tape[column].astype(float)
    with pytest.raises(ValueError, match=column):
        panel.write_dataset(conn, tape, markets, events, source="synthetic")


def test_outcomes_must_be_zero_one_or_missing(conn):
    tape, markets, events = small_dataset()
    for bad in (2.0, 0.5, -1.0):
        with pytest.raises(ValueError, match="outcome"):
            panel.write_dataset(conn, tape, markets.assign(outcome=[bad, 1.0]), events,
                                source="synthetic")
    assert counts(conn) == [0, 0, 0, 0]


def test_text_columns_must_hold_text(conn):
    tape, markets, events = small_dataset()
    tape["market"], markets["market"] = [1, 1, 2], [1, 2]  # consistent, but not text
    with pytest.raises(ValueError, match="text"):
        panel.write_dataset(conn, tape, markets, events, source="synthetic")


# ---------------------------------------------------------------------------------- trade ids
def test_trade_ids_are_stored_with_the_trades_but_not_returned(conn):
    tape, markets, events = small_dataset()
    tape["trade_id"] = ["a", "b", None]
    panel.write_dataset(conn, tape, markets, events, source="kalshi-public-api")
    assert conn.execute("SELECT trade_id FROM trades ORDER BY market, ts").fetchall() == [
        ("a",), ("b",), (None,)]
    assert list(panel.read_dataset(conn)[0].columns) == TAPE_COLUMNS


def test_a_trade_id_cannot_be_stored_twice_in_a_market(conn):
    tape, markets, events = small_dataset()
    tape["trade_id"] = ["same", "same", "same"]  # twice in M1, once in M2: only M1 is a clash
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        panel.write_dataset(conn, tape, markets, events, source="kalshi-public-api")
    tape["trade_id"] = ["same", "other", "same"]
    panel.write_dataset(conn, tape, markets, events, source="kalshi-public-api")  # M2 may reuse it
    assert counts(conn)[2] == 3
    tape["trade_id"] = [None, None, None]  # missing identifiers never clash
    panel.write_dataset(conn, tape, markets, events, source="kalshi-public-api")


# --------------------------------------------------------------------------------- the index
def test_the_last_buy_before_a_time_is_an_index_seek_without_a_sort(conn):
    plan = " | ".join(row[3] for row in conn.execute(
        "EXPLAIN QUERY PLAN SELECT price FROM trades WHERE market = ? AND side = 1 AND ts < ? "
        "ORDER BY ts DESC, seq DESC LIMIT 1", ("M1", 5.0)))
    assert "COVERING INDEX trades_by_side" in plan and "TEMP B-TREE" not in plan
