"""SQLite storage of the data contract (docs/model.md section 4, tape; section 5, markouts).

A dataset is three pandas frames, as in the data contract of the package (all times are Unix
seconds, prices are USD per contract on the YES side):

* ``events``: ``event``, ``release_ts``, ``series``;
* ``markets``: ``market``, ``event``, ``outcome`` (0, 1 or missing), ``open_ts``, ``close_ts``;
* ``tape``: ``event``, ``market``, ``ts``, ``seq``, ``price``, ``size``, ``side`` (+1 the taker
  bought YES, -1 the taker sold YES), sorted by (``market``, ``ts``, ``seq``).

The database has one table for each, ``events``, ``markets`` and ``trades``, and a ``meta`` table
with the key ``source``. ``source`` says where the rows come from: ``synthetic`` for rows made by
``kalshi_mm.world`` and ``kalshi-public-api`` for rows written from Kalshi's public API. The
analysis reads it with :func:`get_source` to label its tables, and a database never mixes the two.

The ``event`` of a trade is not stored twice: the trade has a market, the market has an event.
Columns of a frame that the tables do not have, such as the ``latent_*`` columns of a synthetic
tape, are ignored and never stored, so the analysis cannot read them from the database. The one
exception is ``trade_id`` (Kalshi's identifier of a trade): it is stored when the tape has it, and
a unique index refuses a second trade with the same identifier in a market; :func:`read_dataset`
does not return it.

SQLite does not enforce column types. A numpy integer bound as a parameter is stored as a BLOB
without an error (checked on SQLite 3.45), so the numeric columns carry ``typeof`` checks next to
the range checks, and :func:`write_dataset` converts every value to a Python ``int`` or ``float``.

:func:`trade_panel_sql` builds the markout panel of docs/model.md section 5 inside the database:
the mid proxy of section 4 before and after each trade, and the four markouts. It is the same
computation as ``kalshi_mm.markout.trade_markouts``, written as one SQL statement
(:data:`PANEL_SQL`), so that the panel is built without a loop in Python and without loading the
tape first; the tests compare the two exactly.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import numpy as np
import pandas as pd

__all__ = [
    "PANEL_SQL",
    "SCHEMA_SQL",
    "SOURCES",
    "SOURCE_KALSHI_API",
    "SOURCE_SYNTHETIC",
    "create_schema",
    "get_source",
    "loaded_markets",
    "read_dataset",
    "stored_release_times",
    "trade_panel_sql",
    "write_dataset",
]

SOURCE_SYNTHETIC = "synthetic"
SOURCE_KALSHI_API = "kalshi-public-api"
SOURCES = (SOURCE_SYNTHETIC, SOURCE_KALSHI_API)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    event      TEXT PRIMARY KEY,
    series     TEXT NOT NULL,
    release_ts REAL NOT NULL CHECK (typeof(release_ts) = 'real')
);

CREATE TABLE IF NOT EXISTS markets (
    market   TEXT PRIMARY KEY,
    event    TEXT NOT NULL REFERENCES events (event),
    outcome  INTEGER CHECK (outcome IN (0, 1)),
    open_ts  REAL NOT NULL CHECK (typeof(open_ts) = 'real'),
    close_ts REAL NOT NULL CHECK (typeof(close_ts) = 'real')
);

CREATE TABLE IF NOT EXISTS trades (
    market   TEXT    NOT NULL REFERENCES markets (market),
    ts       REAL    NOT NULL CHECK (typeof(ts) = 'real'),
    seq      INTEGER NOT NULL CHECK (typeof(seq) = 'integer'),
    price    REAL    NOT NULL CHECK (typeof(price) = 'real' AND price > 0 AND price < 1),
    size     REAL    NOT NULL CHECK (typeof(size) = 'real' AND size > 0),
    side     INTEGER NOT NULL CHECK (side IN (-1, 1)),
    trade_id TEXT,
    PRIMARY KEY (market, ts, seq)
) WITHOUT ROWID;

-- "the last buy before ts" is a seek: one market, one side, then ts and seq in order
CREATE INDEX IF NOT EXISTS trades_by_side ON trades (market, side, ts, seq, price);
CREATE UNIQUE INDEX IF NOT EXISTS trades_by_id ON trades (market, trade_id)
    WHERE trade_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS markets_by_event ON markets (event);
"""

# The dtype pandas gives a column of Python strings: object before pandas 3, ``str`` from pandas 3.
# A query that returns no rows gives object columns whatever the version, so the frames made here
# are cast to it, which keeps an empty frame's dtypes the same as a full one's.
_TEXT = pd.Series([""]).dtype

_EVENT_COLUMNS = ("event", "release_ts", "series")
_MARKET_COLUMNS = ("market", "event", "outcome", "open_ts", "close_ts")
_TAPE_COLUMNS = ("event", "market", "ts", "seq", "price", "size", "side")
_CHUNK_ROWS = 50_000  # rows converted to Python objects at a time

_UPSERT_EVENT = """
INSERT INTO events (event, series, release_ts) VALUES (?, ?, ?)
ON CONFLICT (event) DO UPDATE SET series = excluded.series, release_ts = excluded.release_ts
"""
_UPSERT_MARKET = """
INSERT INTO markets (market, event, outcome, open_ts, close_ts) VALUES (?, ?, ?, ?, ?)
ON CONFLICT (market) DO UPDATE SET event = excluded.event, outcome = excluded.outcome,
    open_ts = excluded.open_ts, close_ts = excluded.close_ts
"""
_INSERT_TRADE = """
INSERT INTO trades (market, ts, seq, price, size, side, trade_id) VALUES (?, ?, ?, ?, ?, ?, ?)
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Switch foreign keys on for this connection and create the tables and indexes if missing.

    Foreign key enforcement is a setting of the connection, not of the file; a pending
    transaction on ``conn`` is committed (``executescript`` does that).
    """
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)


def get_source(conn: sqlite3.Connection) -> str:
    """The ``source`` given to :func:`write_dataset`; ``ValueError`` if the database has none."""
    row = conn.execute("SELECT value FROM meta WHERE key = 'source'").fetchone()
    if row is None:
        raise ValueError("the database has no source: nothing was written with write_dataset")
    return str(row[0])


def loaded_markets(conn: sqlite3.Connection) -> set[str]:
    """Identifiers of the markets in the database (a resumed fetch skips these)."""
    return {row[0] for row in conn.execute("SELECT market FROM markets")}


def stored_release_times(conn: sqlite3.Connection) -> dict[str, float]:
    """Release time (Unix seconds) of every stored event, by event identifier."""
    return {row[0]: row[1] for row in conn.execute("SELECT event, release_ts FROM events")}


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} lacks the columns {missing}")


def _require_unique(frame: pd.DataFrame, column: str, name: str) -> None:
    if not frame[column].is_unique:
        raise ValueError(f"{name} has repeated values in the column {column!r}")


def _floats(column: pd.Series, name: str) -> list[float]:
    """The column as Python floats; NaN stays NaN and is refused by the NOT NULL constraint."""
    try:
        return column.to_numpy(dtype=np.float64, na_value=np.nan).tolist()
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric: {error}") from error


def _integers(column: pd.Series, name: str) -> list[int]:
    if not pd.api.types.is_integer_dtype(column):
        raise ValueError(f"{name} must have an integer dtype, it has {column.dtype}")
    return column.to_numpy(dtype=np.int64).tolist()


def _strings(column: pd.Series, name: str) -> list[str]:
    values = column.tolist()
    if not all(isinstance(v, str) for v in values):
        raise ValueError(f"{name} must hold text only")
    return values


def _outcomes(column: pd.Series) -> list[int | None]:
    """0, 1 or None (missing); any other value is an error, not silently truncated."""
    values = _floats(column, "outcome")
    if not all(np.isnan(v) or v in (0.0, 1.0) for v in values):
        raise ValueError("outcome must be 0, 1 or missing")
    return [None if np.isnan(v) else int(v) for v in values]


def _trade_ids(tape: pd.DataFrame) -> list[str | None]:
    if "trade_id" not in tape.columns:
        return [None] * len(tape)
    return [None if pd.isna(v) else str(v) for v in tape["trade_id"].tolist()]


def _check_events_of_trades(tape: pd.DataFrame, markets: pd.DataFrame) -> None:
    """Every trade's market is in ``markets`` and the trade's ``event`` is that market's event."""
    event_of = markets.set_index("market")["event"]
    if not tape["market"].isin(event_of.index).all():
        raise ValueError("the tape has trades of a market that is not in markets")
    if (tape["market"].map(event_of) != tape["event"]).any():
        raise ValueError("a trade's event differs from the event of its market")


def _trade_rows(tape: pd.DataFrame) -> Iterator[list[tuple[object, ...]]]:
    """The trades as rows of Python objects for ``executemany``, one chunk at a time."""
    ids = _trade_ids(tape)
    for start in range(0, len(tape), _CHUNK_ROWS):
        part = tape.iloc[start : start + _CHUNK_ROWS]
        yield list(
            zip(
                _strings(part["market"], "tape market"),
                _floats(part["ts"], "ts"),
                _integers(part["seq"], "seq"),
                _floats(part["price"], "price"),
                _floats(part["size"], "size"),
                _integers(part["side"], "side"),
                ids[start : start + _CHUNK_ROWS],
                strict=True,
            )
        )


def write_dataset(
    conn: sqlite3.Connection,
    tape: pd.DataFrame,
    markets: pd.DataFrame,
    events: pd.DataFrame,
    *,
    source: str,
) -> None:
    """Store a dataset in one transaction (all of it or none of it).

    ``source`` is ``synthetic`` or ``kalshi-public-api`` and is recorded under the key ``source``
    of ``meta`` the first time; a later call with another source raises ``ValueError``. Events and
    markets are inserted or updated by their identifiers; the stored trades of every market in
    ``markets`` are replaced by the trades of ``tape``, so writing the same dataset twice leaves
    the database unchanged. A trade of a market that is not in ``markets``, or whose ``event``
    differs from its market's event, raises ``ValueError``; a value that breaks a constraint of the
    schema raises ``sqlite3.IntegrityError`` and nothing is written.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    _require_columns(events, _EVENT_COLUMNS, "events")
    _require_columns(markets, _MARKET_COLUMNS, "markets")
    _require_columns(tape, _TAPE_COLUMNS, "tape")
    _require_unique(events, "event", "events")
    _require_unique(markets, "market", "markets")
    _check_events_of_trades(tape, markets)
    event_rows = list(
        zip(
            _strings(events["event"], "event"),
            _strings(events["series"], "series"),
            _floats(events["release_ts"], "release_ts"),
            strict=True,
        )
    )
    market_ids = _strings(markets["market"], "market")
    market_rows = list(
        zip(
            market_ids,
            _strings(markets["event"], "market event"),
            _outcomes(markets["outcome"]),
            _floats(markets["open_ts"], "open_ts"),
            _floats(markets["close_ts"], "close_ts"),
            strict=True,
        )
    )
    conn.execute("PRAGMA foreign_keys = ON")
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        raise RuntimeError("foreign keys cannot be switched on: is a transaction open on conn?")
    with conn:
        stored = conn.execute("SELECT value FROM meta WHERE key = 'source'").fetchone()
        if stored is None:
            conn.execute("INSERT INTO meta (key, value) VALUES ('source', ?)", (source,))
        elif stored[0] != source:
            raise ValueError(f"the database holds {stored[0]!r} data, refusing {source!r} data")
        conn.executemany(_UPSERT_EVENT, event_rows)
        conn.executemany(_UPSERT_MARKET, market_rows)
        conn.executemany("DELETE FROM trades WHERE market = ?", [(m,) for m in market_ids])
        for rows in _trade_rows(tape):
            conn.executemany(_INSERT_TRADE, rows)


def read_dataset(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(tape, markets, events)`` of the database, as frames of the data contract.

    The tape is sorted by (``market``, ``ts``, ``seq``), events by (``release_ts``, ``event``) and
    markets by (release of their event, ``event``, ``market``); every frame has a default index.
    ``outcome`` is ``int64``, or ``float64`` with NaN when a market has none. The tape has exactly
    the seven columns of the contract.
    """
    tape = pd.read_sql_query(
        "SELECT m.event AS event, t.market AS market, t.ts AS ts, t.seq AS seq, t.price AS price,"
        " t.size AS size, t.side AS side"
        " FROM trades AS t JOIN markets AS m ON m.market = t.market"
        " ORDER BY t.market, t.ts, t.seq",
        conn,
    ).astype(
        {"event": _TEXT, "market": _TEXT, "ts": "float64", "seq": "int64", "price": "float64",
         "size": "float64", "side": "int8"}
    )
    markets = pd.read_sql_query(
        "SELECT m.market AS market, m.event AS event, m.outcome AS outcome, m.open_ts AS open_ts,"
        " m.close_ts AS close_ts"
        " FROM markets AS m JOIN events AS e ON e.event = m.event"
        " ORDER BY e.release_ts, m.event, m.market",
        conn,
    ).astype({"market": _TEXT, "event": _TEXT, "open_ts": "float64", "close_ts": "float64"})
    has_missing = bool(markets["outcome"].isna().any())
    markets["outcome"] = markets["outcome"].astype("float64" if has_missing else "int64")
    events = pd.read_sql_query(
        "SELECT event, release_ts, series FROM events ORDER BY release_ts, event", conn
    ).astype({"event": _TEXT, "series": _TEXT, "release_ts": "float64"})
    return tape, markets, events


# The markout panel. Every comparison of times is spelled out because the strict and the inclusive
# one differ in effect: mid_before reads the prints strictly before the trade, mid_after the prints
# at or before min(ts + horizon, release) and strictly before the release.
PANEL_SQL = """
WITH
-- Every trade in tape order. tape_row is its row number in the whole tape as read_dataset returns
-- it, counted before any trade is dropped; it becomes the index of the result.
tape AS (
    SELECT ROW_NUMBER() OVER (ORDER BY t.market, t.ts, t.seq) - 1 AS tape_row,
           m.event, t.market, t.ts, t.seq, t.price, t.size, t.side,
           e.release_ts, m.outcome
    FROM trades AS t
    JOIN markets AS m ON m.market = t.market
    JOIN events AS e ON e.event = m.event
),

-- The trades strictly before their release (a trade at the release is dropped). tau_min is the
-- time to the release in minutes. after_ts is the time at which the mid after the trade is read:
-- one horizon later, but not later than the release.
before_release AS (
    SELECT *,
           (release_ts - ts) / 60.0 AS tau_min,
           MIN(ts + :horizon_s, release_ts) AS after_ts
    FROM tape
    WHERE ts < release_ts
),

-- For each trade, the price of the last buy and of the last sell print of its market, read at two
-- times. The last print is found first (an index seek, newest first, the later seq wins a tie in
-- ts), and only then checked for age: the price is NULL when there is no print or the print is
-- more than stale_s seconds older than the time it is read at. MATERIALIZED makes SQLite compute
-- these four seeks once per trade; inlined, it would repeat them wherever a mid is used below.
last_prints AS MATERIALIZED (
    SELECT x.*,
           -- the prints strictly before the trade, aged at the trade's time
           (SELECT CASE WHEN x.ts - s.ts <= :stale_s THEN s.price END
              FROM trades AS s
             WHERE s.market = x.market AND s.side = 1 AND s.ts < x.ts
             ORDER BY s.ts DESC, s.seq DESC LIMIT 1) AS buy_before,
           (SELECT CASE WHEN x.ts - s.ts <= :stale_s THEN s.price END
              FROM trades AS s
             WHERE s.market = x.market AND s.side = -1 AND s.ts < x.ts
             ORDER BY s.ts DESC, s.seq DESC LIMIT 1) AS sell_before,
           -- the prints at or before after_ts and strictly before the release, aged at after_ts
           (SELECT CASE WHEN x.after_ts - s.ts <= :stale_s THEN s.price END
              FROM trades AS s
             WHERE s.market = x.market AND s.side = 1
               AND s.ts <= x.after_ts AND s.ts < x.release_ts
             ORDER BY s.ts DESC, s.seq DESC LIMIT 1) AS buy_after,
           (SELECT CASE WHEN x.after_ts - s.ts <= :stale_s THEN s.price END
              FROM trades AS s
             WHERE s.market = x.market AND s.side = -1
               AND s.ts <= x.after_ts AND s.ts < x.release_ts
             ORDER BY s.ts DESC, s.seq DESC LIMIT 1) AS sell_after
    FROM before_release AS x
),

-- The mid proxy: the mean of the last buy and the last sell. A NULL price (no print, or a stale
-- one) makes the sum NULL, so the mid is undefined, as in the NaN of the Python version.
mids AS (
    SELECT *,
           0.5 * (buy_before + sell_before) AS mid_before,
           0.5 * (buy_after + sell_after) AS mid_after
    FROM last_prints
)

-- The markouts. side is +1 when the taker bought YES and -1 when the taker sold, outcome is the
-- 1 or 0 the market settled at (NULL if unknown). The buckets are right-closed intervals of
-- tau_min with the edges 5, 15, 30, 60, 180 and 360 minutes.
SELECT tape_row, event, market, ts, seq, price, size, side, tau_min,
       CASE WHEN tau_min <= 5 THEN 0
            WHEN tau_min <= 15 THEN 1
            WHEN tau_min <= 30 THEN 2
            WHEN tau_min <= 60 THEN 3
            WHEN tau_min <= 180 THEN 4
            WHEN tau_min <= 360 THEN 5
            ELSE 6 END AS bucket,
       mid_before,
       mid_after,
       side * (price - mid_before) AS eff_half_spread,
       side * (mid_after - mid_before) AS adverse_selection,
       side * (price - mid_after) AS realized_spread,
       side * (price - outcome) AS settlement_markout
FROM mids
ORDER BY tape_row
"""

_PANEL_TYPES = {
    "event": _TEXT,
    "market": _TEXT,
    "ts": "float64",
    "seq": "int64",
    "price": "float64",
    "size": "float64",
    "side": "int8",
    "tau_min": "float64",
    "bucket": "int64",
    "mid_before": "float64",
    "mid_after": "float64",
    "eff_half_spread": "float64",
    "adverse_selection": "float64",
    "realized_spread": "float64",
    "settlement_markout": "float64",
}
_MIN_SQLITE_FOR_PANEL = (3, 35, 0)  # WITH ... AS MATERIALIZED


def trade_panel_sql(
    conn: sqlite3.Connection, *, horizon_s: float = 300.0, stale_s: float = 3600.0
) -> pd.DataFrame:
    """Markouts of every trade before its release, computed in SQL (docs/model.md sections 4, 5).

    For a trade at time ``t`` with taker side ``s`` (+1 the taker bought YES, -1 sold), price ``p``
    and release time ``T``, with horizon ``h = horizon_s`` and ``m(q)`` the mid proxy at time ``q``,
    the mean of the prices of the last buy and the last sell print, each at most ``stale_s``
    seconds old at ``q`` (an age of exactly ``stale_s`` is still fresh):

        mid_before         = m(t) from the prints strictly before t
        mid_after          = m(min(t + h, T)) from the prints at or before that time and strictly
                             before T
        eff_half_spread    = s (p - mid_before)
        adverse_selection  = s (mid_after - mid_before)
        realized_spread    = s (p - mid_after)
        settlement_markout = s (p - y)         y = 1 if the market settled YES, else 0

    A value is NaN where a mid it needs is undefined or the outcome is missing. Trades at or after
    the release are dropped. ``tau_min = (T - t) / 60`` and ``bucket`` is the index of its
    right-closed interval (0,5], (5,15], (15,30], (30,60], (60,180], (180,360], (360,inf).

    The result is the frame that ``kalshi_mm.markout.trade_markouts(tape, events, markets,
    horizon_s=horizon_s, stale_s=stale_s)`` returns for ``tape, markets, events =
    read_dataset(conn)``: the same columns, dtypes and row order (market, ``ts``, ``seq``), the
    same index (a trade's index label is its row number in the whole tape, so the dropped trades
    leave gaps) and the same values, which the tests compare exactly. The statement is
    :data:`PANEL_SQL`; it only reads the database. It needs SQLite 3.35 or later.
    """
    if not horizon_s > 0.0:
        raise ValueError("horizon_s must be positive")
    if not stale_s >= 0.0:
        raise ValueError("stale_s must be non-negative")
    if sqlite3.sqlite_version_info < _MIN_SQLITE_FOR_PANEL:
        raise RuntimeError(
            f"trade_panel_sql needs SQLite 3.35 or later, this is {sqlite3.sqlite_version}"
        )
    frame = pd.read_sql_query(
        PANEL_SQL, conn, params={"horizon_s": float(horizon_s), "stale_s": float(stale_s)}
    )
    index = pd.Index(frame.pop("tape_row").to_numpy(dtype=np.int64))
    frame = frame.astype(_PANEL_TYPES)
    frame.index = index
    return frame
