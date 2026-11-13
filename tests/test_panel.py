"""SQLite storage of the data contract and the markout panel in SQL (src/kalshi_mm/panel.py;
docs/model.md sections 4 and 5).

The synthetic world of ``kalshi_mm.world`` gives datasets of known content; small hand-built
datasets give exact values for the constraints, the replacement rules and the edge cases of the
panel. The panel computed in SQL is compared with ``kalshi_mm.markout.trade_markouts`` on the same
data, exactly (equal frames, NaN in the same places): on a synthetic world, on a hand-built tape
whose cells are also worked out on paper, and on many small random worlds with equal timestamps,
one-sided stretches, trades at the release and ages that equal the staleness limit.
"""

import sqlite3
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from kalshi_mm import panel
from kalshi_mm.markout import BUCKET_EDGES_MIN, bucket_index, trade_markouts
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


def test_the_price_grid_of_a_market_is_stored_when_given_and_null_otherwise(conn):
    tape, markets, events = small_dataset()
    panel.write_dataset(conn, tape, markets, events, source="kalshi-public-api")
    query = "SELECT market, price_level_structure, price_ranges FROM markets ORDER BY market"
    assert conn.execute(query).fetchall() == [("M1", None, None), ("M2", None, None)]
    markets["price_level_structure"] = ["linear_cent", None]
    markets["price_ranges"] = ['[{"start": "0.0000", "end": "1.0000", "step": "0.0100"}]', np.nan]
    panel.write_dataset(conn, tape, markets, events, source="kalshi-public-api")  # updates
    assert conn.execute(query).fetchall() == [
        ("M1", "linear_cent", '[{"start": "0.0000", "end": "1.0000", "step": "0.0100"}]'),
        ("M2", None, None)]
    assert list(panel.read_dataset(conn)[1].columns) == MARKET_COLUMNS


# The markets table as the first version of the schema created it.
OLD_MARKETS = """CREATE TABLE markets (market TEXT PRIMARY KEY, event TEXT NOT NULL,
    outcome INTEGER, open_ts REAL NOT NULL, close_ts REAL NOT NULL)"""


def test_a_database_of_the_older_schema_is_refused_with_a_clear_message_and_left_alone(tmp_path):
    path = tmp_path / "old.sqlite"
    with closing(sqlite3.connect(path)) as old:
        old.execute(OLD_MARKETS)
        old.commit()
    with closing(sqlite3.connect(path)) as conn:
        problem = panel.schema_problem(conn)
        assert "earlier version of the schema" in problem
        assert "markets.price_level_structure" in problem and "markets.price_ranges" in problem
        with pytest.raises(panel.OldSchemaError, match="build a new database"):
            panel.create_schema(conn)
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert tables == {"markets", "sqlite_autoindex_markets_1"}  # nothing was created


def test_a_new_or_current_database_has_no_schema_problem(conn):
    assert panel.schema_problem(conn) is None
    with closing(sqlite3.connect(":memory:")) as empty:
        assert panel.schema_problem(empty) is None  # no table yet: create_schema makes them


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


# ------------------------------------------------------------------------- the panel in SQL
NAN = np.nan
PANEL_COLUMNS = [*TAPE_COLUMNS, "tau_min", "bucket", "mid_before", "mid_after", "eff_half_spread",
                 "adverse_selection", "realized_spread", "settlement_markout"]


def python_panel(conn, **kwargs):
    """The panel of ``trade_markouts`` for the tape stored in the database."""
    tape, markets, events = panel.read_dataset(conn)
    return trade_markouts(tape, events, markets, **kwargs)


def assert_panel_equals_python(conn, **kwargs):
    """The SQL panel is the frame of ``trade_markouts``: columns, dtypes, index, values and the
    places of the NaN, compared exactly."""
    got = panel.trade_panel_sql(conn, **kwargs)
    pd.testing.assert_frame_equal(got, python_panel(conn, **kwargs), check_exact=True)
    return got


def frames_from(prints, releases, outcomes):
    """Dataset frames from ``prints = {market: (event, [(ts, side, price), ...])}`` (in time
    order), the release time of each event and the outcome of each market (None: missing). The
    trades of a market are numbered in order, so equal times are told apart by ``seq``."""
    rows = [(event, market, ts, seq, price, 1.0 + seq % 3, side)
            for market, (event, trades) in prints.items()
            for seq, (ts, side, price) in enumerate(trades)]
    tape = pd.DataFrame(rows, columns=TAPE_COLUMNS).astype(
        {"ts": "float64", "seq": "int64", "price": "float64", "size": "float64", "side": "int8"})
    event_of = {market: event for market, (event, _) in prints.items()}
    markets = pd.DataFrame({
        "market": list(prints), "event": [event_of[m] for m in prints],
        "outcome": [NAN if outcomes[m] is None else float(outcomes[m]) for m in prints],
        "open_ts": 0.0, "close_ts": [float(releases[event_of[m]]) for m in prints]})
    events = pd.DataFrame({"event": list(releases), "release_ts": [float(t) for t in
                                                                   releases.values()],
                           "series": "S"})
    return tape, markets, events


# Market M1 (event E1, release 10000, outcome 1): three prints at one time, two sells at another,
# a buy exactly 300 s after them, a last sell that is 3600 s old at 5100 and 3601 s old at 5101,
# a print exactly at the release (dropped, and used by no mid) and one after it (dropped).
HAND_M1 = [
    (1000.0, 1, 0.50), (1000.0, -1, 0.40), (1000.0, 1, 0.60),
    (1500.0, -1, 0.45), (1500.0, -1, 0.47),
    (1800.0, 1, 0.62),
    (5100.0, 1, 0.55), (5101.0, 1, 0.56),
    (9000.0, 1, 0.70), (9500.0, -1, 0.66), (9800.0, -1, 0.65),
    (10000.0, 1, 0.99),
    (10050.0, -1, 0.01),
]
# Market M2 (event E1, outcome missing): a stretch with buys only, then the first sell.
HAND_M2 = [(2000.0, 1, 0.30), (2100.0, 1, 0.32), (2200.0, 1, 0.31), (2300.0, -1, 0.28),
           (2400.0, 1, 0.33)]
# Market M3 (event E2, release 100000, outcome 0): a trade exactly on each bucket edge and one half
# a second earlier (which is in the next bucket), one half a second before the release, one at
# the release and one after it.
HAND_R3 = 100000.0
HAND_M3_TIMES = [t for edge in reversed(BUCKET_EDGES_MIN[1:-1])
                 for t in (HAND_R3 - 60.0 * edge - 0.5, HAND_R3 - 60.0 * edge)]
HAND_M3_TIMES += [HAND_R3 - 0.5, HAND_R3, HAND_R3 + 0.5]
HAND_M3 = [(t, 1 if i % 2 == 0 else -1, 0.40 + 0.01 * i) for i, t in enumerate(HAND_M3_TIMES)]
# Market M4 (event E3, release 50000, outcome 1): every trade is at or after the release.
HAND_M4 = [(50000.0, 1, 0.50), (50000.0, -1, 0.40), (50001.0, 1, 0.60)]
HAND_KEPT = [*range(0, 11), *range(13, 31)]  # rows of the whole tape before their release


def hand_world():
    prints = {"M1": ("E1", HAND_M1), "M2": ("E1", HAND_M2), "M3": ("E2", HAND_M3),
              "M4": ("E3", HAND_M4)}
    releases = {"E1": 10000.0, "E2": HAND_R3, "E3": 50000.0, "E4": 70000.0}  # E4 has no market
    return frames_from(prints, releases, {"M1": 1, "M2": None, "M3": 0, "M4": 1})


@pytest.fixture
def hand_conn(conn):
    panel.write_dataset(conn, *hand_world(), source="synthetic")
    return conn


# Worked out on paper for horizon 300 s and staleness 3600 s. Examples, M1: row 3 (a sell at 1500)
# has mid_before = (0.60 + 0.40) / 2 from the prints before 1500 (the sell at 1500 is not one of
# them) and mid_after = (0.62 + 0.47) / 2 from the prints up to 1800, the buy at 1800 included;
# row 10 (a sell at 9800) has its horizon cut at the release 10000, where the buy of 0.99 does not
# count; row 6 has no mid_after because the last sell is 3900 s old at 5400.
M1_CELLS = {
    "tau_min": np.array([9000, 9000, 9000, 8500, 8500, 8200, 4900, 4899, 1000, 500, 200]) / 60.0,
    "bucket": [4, 4, 4, 4, 4, 4, 4, 4, 2, 1, 0],
    "mid_before": [NAN, NAN, NAN, 0.50, 0.50, 0.535, 0.545, NAN, NAN, NAN, 0.68],
    "mid_after": [0.50, 0.50, 0.50, 0.545, 0.545, 0.545, NAN, NAN, NAN, 0.675, 0.675],
    "eff_half_spread": [NAN, NAN, NAN, 0.05, 0.03, 0.085, 0.005, NAN, NAN, NAN, 0.03],
    "adverse_selection": [NAN, NAN, NAN, -0.045, -0.045, 0.01, NAN, NAN, NAN, NAN, 0.005],
    "realized_spread": [0.0, 0.10, 0.10, 0.095, 0.075, 0.075, NAN, NAN, NAN, 0.015, 0.025],
    "settlement_markout": [-0.50, 0.60, -0.40, 0.55, 0.53, -0.38, -0.45, -0.44, -0.30, 0.34,
                           0.35],
}
# M2: the first sell is at 2300, so only the last row has a mid_before; the sell at 2300 is
# exactly at the end of the horizon of the first row and counts there.
M2_CELLS = {
    "tau_min": np.array([8000, 7900, 7800, 7700, 7600]) / 60.0,
    "bucket": [4, 4, 4, 4, 4],
    "mid_before": [NAN, NAN, NAN, NAN, 0.295],
    "mid_after": [0.295, 0.305, 0.305, 0.305, 0.305],
    "eff_half_spread": [NAN, NAN, NAN, NAN, 0.035],
    "adverse_selection": [NAN, NAN, NAN, NAN, 0.01],
    "realized_spread": [0.005, 0.015, 0.005, 0.025, 0.025],
    "settlement_markout": [NAN] * 5,
}


def test_hand_world_cells_agree_with_the_numbers_worked_out_on_paper(hand_conn):
    got = panel.trade_panel_sql(hand_conn)
    assert got.index.tolist() == HAND_KEPT  # M1's last two trades and all of M4 are dropped
    assert list(got.columns) == PANEL_COLUMNS
    for market, cells in (("M1", M1_CELLS), ("M2", M2_CELLS)):
        part = got[got["market"] == market]
        assert len(part) == len(cells["bucket"])
        for column, expected in cells.items():
            np.testing.assert_allclose(part[column].to_numpy(dtype=float), expected, rtol=0.0,
                                       atol=1e-12, err_msg=f"{market} {column}")
    m1 = got[got["market"] == "M1"]
    assert m1["side"].tolist() == [1, -1, 1, -1, -1, 1, 1, 1, 1, -1, -1]
    assert m1["ts"].tolist() == [t for t, _, _ in HAND_M1[:11]]
    assert m1["seq"].tolist() == list(range(11))  # equal times keep their seq


def test_every_hand_world_cell_is_checked_for_undefined_values_too(hand_conn):
    got = panel.trade_panel_sql(hand_conn)
    part = got[got["market"].isin(["M1", "M2"])]
    for column in ("mid_before", "mid_after", "eff_half_spread", "adverse_selection",
                   "realized_spread", "settlement_markout"):
        assert part[column].isna().any() and part[column].notna().any(), column


@pytest.mark.parametrize("horizon_s, stale_s", [
    (300.0, 3600.0), (60.0, 600.0), (900.0, 3600.0), (1.0, 3600.0), (299.0, 3600.0),
    (300.0, 0.0), (300.0, 300.0), (300.0, 3599.0), (300.0, 3601.0), (np.inf, 3600.0),
    (300.0, np.inf), (np.inf, np.inf),
])
def test_hand_world_equals_trade_markouts(hand_conn, horizon_s, stale_s):
    assert_panel_equals_python(hand_conn, horizon_s=horizon_s, stale_s=stale_s)


def test_an_age_of_exactly_the_staleness_limit_is_fresh(hand_conn):
    """M1 row 6 (5100) has a last sell that is 3600 s old, row 7 (5101) one that is 3601 s old."""
    for stale_s, row_6, row_7 in ((3599.0, NAN, NAN), (3600.0, 0.545, NAN), (3601.0, 0.545, 0.51)):
        got = panel.trade_panel_sql(hand_conn, stale_s=stale_s)
        m1 = got[got["market"] == "M1"]["mid_before"].to_numpy()
        np.testing.assert_allclose(m1[[6, 7]], [row_6, row_7], rtol=0.0, atol=1e-12)


def test_the_horizon_ends_with_an_inclusive_time_and_stops_at_the_release(hand_conn):
    """The buy at 1800 is exactly 300 s after the trades at 1500: it is in their mid_after for a
    horizon of 300 s and not for 299 s; a horizon past the release reads the proxy at 10000, from
    the buy at 9000 and the sell at 9800 (not from the buy of 0.99 at 10000)."""
    for horizon_s, row_3, row_10 in ((300.0, 0.545, 0.675), (299.0, 0.535, 0.675),
                                     (1e9, 0.675, 0.675)):
        got = panel.trade_panel_sql(hand_conn, horizon_s=horizon_s)
        m1 = got[got["market"] == "M1"]["mid_after"].to_numpy()
        np.testing.assert_allclose(m1[[3, 10]], [row_3, row_10], rtol=0.0, atol=1e-12)


def test_bucket_edges_are_right_closed_like_bucket_index(hand_conn):
    got = panel.trade_panel_sql(hand_conn)
    m3 = got[got["market"] == "M3"]
    # each edge belongs to the lower bucket; half a second earlier is in the next one up
    assert m3["bucket"].tolist() == [6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0]
    assert m3["tau_min"].iloc[[1, 3, 5, 7, 9, 11]].tolist() == [360.0, 180.0, 60.0, 30.0, 15.0, 5.0]
    np.testing.assert_array_equal(bucket_index(got["tau_min"].to_numpy()), got["bucket"].to_numpy())


def test_prints_at_or_after_the_release_change_nothing_before_it(hand_conn):
    base = panel.trade_panel_sql(hand_conn)
    changed = hand_conn.execute(
        "UPDATE trades SET price = 1.0 - price, side = -side WHERE ts >= "
        "(SELECT e.release_ts FROM markets AS m JOIN events AS e ON e.event = m.event "
        "WHERE m.market = trades.market)").rowcount
    assert changed == 2 + 2 + 3  # M1: 10000 and 10050; M3: 100000, 100000.5; M4: all three
    pd.testing.assert_frame_equal(panel.trade_panel_sql(hand_conn), base, check_exact=True)
    hand_conn.rollback()


def test_a_market_with_trades_only_at_or_after_the_release_gives_no_rows(conn):
    tape, markets, events = hand_world()
    keep = tape["market"] == "M4"
    panel.write_dataset(conn, tape[keep], markets[markets["market"] == "M4"], events,
                        source="synthetic")
    got = assert_panel_equals_python(conn)
    assert len(got) == 0 and list(got.columns) == PANEL_COLUMNS


def test_text_columns_have_one_dtype_whether_or_not_there_are_rows(conn):
    empty = panel.read_dataset(conn)
    panel.write_dataset(conn, *small_dataset(), source="synthetic")
    full = panel.read_dataset(conn)
    for frame_empty, frame_full in zip(empty, full, strict=True):
        for column in set(frame_empty.columns) & {"event", "market", "series"}:
            assert frame_empty[column].dtype == frame_full[column].dtype, column


def test_an_empty_database_gives_an_empty_panel(conn):
    got = assert_panel_equals_python(conn)
    assert len(got) == 0 and list(got.columns) == PANEL_COLUMNS
    assert got["side"].dtype == np.int8 and got["bucket"].dtype == np.int64
    assert got["mid_before"].dtype == np.float64


# ------------------------------------------------------------- the panel on a synthetic world
def test_panel_equals_trade_markouts_on_a_synthetic_world_of_30_events(conn, world):
    panel.write_dataset(conn, world.tape, world.markets, world.events, source="synthetic")
    got = assert_panel_equals_python(conn)
    # the same panel from the frames the world was made of, never stored (latent columns dropped)
    direct = trade_markouts(world.tape, world.events, world.markets)
    pd.testing.assert_frame_equal(got, direct, check_exact=True)
    assert len(got) == len(world.tape) > 3000
    assert got["mid_before"].isna().any() and got["adverse_selection"].notna().sum() > 3000
    assert not [c for c in got.columns if c.startswith("latent_")]
    assert got["settlement_markout"].notna().all()  # the world settles every market


@pytest.mark.parametrize("horizon_s, stale_s", [(60.0, 600.0), (900.0, 3600.0), (300.0, 120.0),
                                                (5.0, 30.0), (np.inf, np.inf)])
def test_panel_equals_trade_markouts_when_horizon_and_staleness_change(conn, world, horizon_s,
                                                                       stale_s):
    panel.write_dataset(conn, world.tape, world.markets, world.events, source="synthetic")
    got = assert_panel_equals_python(conn, horizon_s=horizon_s, stale_s=stale_s)
    assert got["mid_before"].isna().any() and got["mid_before"].notna().any()


def test_missing_outcomes_make_only_the_settlement_markout_missing(conn, world):
    markets = world.markets.copy()
    markets["outcome"] = markets["outcome"].astype("float64")
    markets.loc[markets.index[::3], "outcome"] = np.nan
    panel.write_dataset(conn, world.tape, markets, world.events, source="synthetic")
    got = assert_panel_equals_python(conn)
    missing = got["market"].isin(set(markets.loc[markets["outcome"].isna(), "market"]))
    assert missing.sum() > 1000 and (~missing).sum() > 1000
    assert got.loc[missing, "settlement_markout"].isna().all()
    assert got.loc[~missing, "settlement_markout"].notna().all()
    assert got.loc[missing, "adverse_selection"].notna().any()


# ----------------------------------------------------- many small random worlds, with ties
HORIZONS = [20.0, 60.0, 100.0, 300.0, np.inf]
STALES = [0.0, 40.0, 100.0, 600.0, np.inf]


def random_world(seed):
    """Up to five markets in three events, with up to 25 trades each at times that are multiples
    of 20 s: many equal times, ages that equal the staleness limits, prints exactly at the end of a
    horizon and at the release, trades after it, one-sided stretches and markets without trades."""
    rng = np.random.default_rng(seed)
    releases = {f"E{k}": 20.0 * int(rng.integers(30, 70)) for k in range(3)}
    prints, outcomes = {}, {}
    for k in range(int(rng.integers(2, 6))):
        n = int(rng.integers(0, 26))
        times = np.sort(rng.integers(0, 80, n)) * 20.0
        sides = rng.choice([1, -1], n)
        prices = rng.integers(1, 100, n) / 100.0
        prints[f"M{k}"] = (f"E{int(rng.integers(0, 3))}",
                           list(zip(times.tolist(), sides.tolist(), prices.tolist(), strict=True)))
        outcomes[f"M{k}"] = [None, 0, 1][int(rng.integers(0, 3))]
    return frames_from(prints, releases, outcomes)


def test_panel_equals_trade_markouts_on_random_small_worlds():
    rng = np.random.default_rng(SEED)
    rows = both = undefined = 0
    for seed in range(80):
        tape, markets, events = random_world(seed)
        with closing(sqlite3.connect(":memory:")) as db:
            panel.create_schema(db)
            panel.write_dataset(db, tape.sample(frac=1.0, random_state=seed), markets, events,
                                source="synthetic")
            for _ in range(3):
                got = assert_panel_equals_python(
                    db, horizon_s=HORIZONS[int(rng.integers(len(HORIZONS)))],
                    stale_s=STALES[int(rng.integers(len(STALES)))])
                rows += len(got)
                both += int(got["adverse_selection"].notna().sum())
                undefined += int(got["mid_before"].isna().sum())
    assert rows > 3000 and both > 800 and undefined > 800  # defined and undefined cases both occur


# --------------------------------------------------------------------- look-ahead, in SQL
def keyed(frame):
    return frame.set_index(["market", "ts", "seq"])


def same_numbers(a, b):
    return np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True)


@pytest.mark.parametrize("statement, flips", [
    ("UPDATE trades SET price = 1.0 - price, side = -side WHERE market = ? AND ts > ?", True),
    ("DELETE FROM trades WHERE market = ? AND ts > ?", False),
])
def test_changing_or_deleting_later_prints_leaves_earlier_rows_alone(conn, world, statement,
                                                                     flips):
    """The statements run inside the database on the prints after a cut (flipping their side and
    price, or deleting them). A trade at or before the cut keeps its mid_before, because that uses
    the prints strictly before it; a trade whose horizon ends by the cut keeps its whole row."""
    panel.write_dataset(conn, world.tape, world.markets, world.events, source="synthetic")
    base = keyed(panel.trade_panel_sql(conn))
    names = base.index.get_level_values("market")
    times = base.index.get_level_values("ts")
    for market in sorted(world.markets["market"])[::8]:
        cut = float(world.tape.loc[world.tape["market"] == market, "ts"].median())
        conn.execute(statement, (market, cut))
        altered = keyed(panel.trade_panel_sql(conn))
        conn.rollback()
        early = (names == market) & (times <= cut)
        finished = (names == market) & (times + 300.0 <= cut)
        late = (names == market) & (times > cut)
        assert early.sum() > 5 and finished.sum() > 5 and late.sum() > 5
        assert same_numbers(altered.loc[base.index[early], "mid_before"],
                            base.loc[early, "mid_before"])
        pd.testing.assert_frame_equal(altered.loc[base.index[finished]], base.loc[finished],
                                      check_exact=True)
        if flips:  # the guard has power: the rows after the cut do change
            assert not same_numbers(altered.loc[base.index[late], "mid_before"],
                                    base.loc[late, "mid_before"])
        else:
            assert len(altered) == len(base) - late.sum()


# ------------------------------------------------------------- arguments and the statement
@pytest.mark.parametrize("kwargs, match", [
    ({"horizon_s": 0.0}, "horizon_s"), ({"horizon_s": -1.0}, "horizon_s"),
    ({"horizon_s": float("nan")}, "horizon_s"), ({"stale_s": -1.0}, "stale_s"),
    ({"stale_s": float("nan")}, "stale_s"),
])
def test_invalid_arguments_are_refused(conn, kwargs, match):
    with pytest.raises(ValueError, match=match):
        panel.trade_panel_sql(conn, **kwargs)


def test_an_old_sqlite_is_refused_with_a_clear_message(conn, monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 34, 1))
    monkeypatch.setattr(sqlite3, "sqlite_version", "3.34.1")
    with pytest.raises(RuntimeError, match=r"SQLite 3\.35.*3\.34\.1"):
        panel.trade_panel_sql(conn)


def test_the_panel_only_reads_the_database(hand_conn):
    before = hand_conn.total_changes
    panel.trade_panel_sql(hand_conn)
    assert hand_conn.total_changes == before and not hand_conn.in_transaction
    names = {row[0] for row in hand_conn.execute("SELECT name FROM sqlite_master")}
    assert not [n for n in names if "panel" in n or "last_prints" in n]


def test_the_panel_statement_does_one_index_seek_per_lookup_and_builds_no_index(conn):
    plan = [row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + panel.PANEL_SQL,
                                           {"horizon_s": 300.0, "stale_s": 3600.0})]
    seeks = [p for p in plan if p.startswith("SEARCH s ")]
    # four lookups (last buy and last sell, before and after), each evaluated once per trade
    assert len([p for p in plan if "CORRELATED SCALAR SUBQUERY" in p]) == 4
    assert len(seeks) == 4 and all("COVERING INDEX trades_by_side" in p for p in seeks)
    assert not [p for p in plan if "AUTOMATIC" in p or p.startswith("SCAN s")]
