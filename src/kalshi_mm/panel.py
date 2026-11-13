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
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import numpy as np
import pandas as pd

__all__ = [
    "SCHEMA_SQL",
    "SOURCES",
    "SOURCE_KALSHI_API",
    "SOURCE_SYNTHETIC",
    "create_schema",
    "get_source",
    "loaded_markets",
    "read_dataset",
    "stored_release_times",
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
        {"ts": "float64", "seq": "int64", "price": "float64", "size": "float64", "side": "int8"}
    )
    markets = pd.read_sql_query(
        "SELECT m.market AS market, m.event AS event, m.outcome AS outcome, m.open_ts AS open_ts,"
        " m.close_ts AS close_ts"
        " FROM markets AS m JOIN events AS e ON e.event = m.event"
        " ORDER BY e.release_ts, m.event, m.market",
        conn,
    ).astype({"open_ts": "float64", "close_ts": "float64"})
    has_missing = bool(markets["outcome"].isna().any())
    markets["outcome"] = markets["outcome"].astype("float64" if has_missing else "int64")
    events = pd.read_sql_query(
        "SELECT event, release_ts, series FROM events ORDER BY release_ts, event", conn
    ).astype({"release_ts": "float64"})
    return tape, markets, events
