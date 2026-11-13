"""Turn Kalshi API records into the data contract (docs/protocol.md, "Data"; docs/data.md).

Field mapping (the documentation these names come from is listed in docs/data.md):

* trade ``created_time`` -> ``ts``, Unix seconds, microseconds kept;
* trade ``yes_price_dollars`` -> ``price``, USD per YES contract;
* trade ``count_fp`` -> ``size``, contracts (fractions are allowed);
* trade ``taker_outcome_side`` -> ``side``, +1 for ``yes`` and -1 for ``no``; the deprecated
  ``taker_side`` is read only when the new field is absent;
* trade ``trade_id`` -> ``trade_id`` (stored, not part of the contract);
* market ``ticker`` -> ``market``; ``event_ticker`` -> ``event``; ``open_time`` -> ``open_ts``;
  ``close_time`` -> ``close_ts``;
* market ``series_ticker`` if the record has it, else the event ticker up to its first hyphen
  -> ``series``;
* market ``result`` -> ``outcome``: 1 for ``yes``, 0 for ``no``, missing for anything else;
* market ``price_level_structure`` and ``price_ranges`` (the price grid: ranges of prices, each
  with its ``step`` in USD) -> stored as given, ``price_ranges`` as JSON text; not used by the
  analysis, which quotes on a grid of 0.01 USD (``docs/data.md``, Limitations).

A record that cannot be read raises :class:`RejectedRecordError`; :func:`build_dataset` counts every
rejection by reason in its report and drops nothing without counting it.

The release time of an event is never taken from a market: it comes from the release calendar
(:func:`load_release_calendar`), a file checked by hand against the official calendars.
:func:`write_calendar_draft` writes a draft from the markets' expected expiration times to help
with that; the loader refuses any file whose name contains "draft".

:func:`build_dataset` applies the filters of docs/protocol.md in this order, and the report counts
what each one removes. Markets: unreadable record; event not in the calendar; release outside the
window ``[window_start, window_end)``. Trades of the markets that remain: unreadable record
(price outside (0, 1), size not positive, ...); repeated ``trade_id`` (the first record is kept);
block trade; at or after the release (``ts >= release``); then a market with fewer than
``min_trades`` trades left is dropped with its trades. Trades are numbered ``seq = 0, 1, ...`` in
the order (``ts``, ``trade_id``), so the numbering does not depend on the order of the pages.

Fetching. :func:`list_series_markets` lists a series from ``GET /markets`` and ``GET
/historical/markets`` (one record per ticker). :func:`fetch_series` reads the trades of the markets
that can enter the dataset (a readable record, an event in the calendar, a release inside the
window) from the endpoint or endpoints that hold them (:func:`trade_queries`: trades created before
the cutoff of ``GET /historical/cutoff`` are on the historical endpoint only). :func:`ingest_series`
writes every market that passes the protocol filters to SQLite as soon as its trades are read, so
that an interrupted run resumes: a market already in the database is not fetched again. A market
dropped for having too few trades is not stored, so a resumed run fetches it again, and the report
of a run counts only what that run read. Tested on canned responses shaped like the documentation;
not run against the live service.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import sqlite3
from collections.abc import Callable, Container, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from kalshi_mm import panel
from kalshi_mm.kalshi_api import HistoricalCutoff, KalshiClient, parse_iso_datetime, parse_iso_time

__all__ = [
    "ALREADY_LOADED",
    "MARKET_MARGIN_S",
    "MIN_TRADES",
    "OUT_OF_SCOPE",
    "PROTOCOL_END",
    "PROTOCOL_SERIES",
    "PROTOCOL_START",
    "WANTED",
    "BuildReport",
    "CalendarChangedError",
    "Classified",
    "DraftSummary",
    "Fetched",
    "IngestSummary",
    "RejectedRecordError",
    "TradeQuery",
    "Window",
    "build_dataset",
    "classify_markets",
    "fetch_series",
    "ingest_series",
    "list_series_markets",
    "load_release_calendar",
    "parse_market",
    "parse_trade",
    "trade_queries",
    "utc_text",
    "write_calendar_draft",
]

# docs/protocol.md, "Data": the series (tickers of Kalshi's fee schedule of 2026-07-07) and the
# releases to load, the last day included.
PROTOCOL_SERIES = ("KXCPI", "KXCPIYOY", "KXPAYROLLS", "KXGDP", "KXU3", "KXFEDDECISION")
PROTOCOL_START = "2024-01-01"
PROTOCOL_END = "2026-10-31"
MIN_TRADES = 50  # docs/protocol.md, "Markets kept"
MARKET_MARGIN_S = 7 * 86400.0  # a draft calendar lists events up to a week outside the window

_CALENDAR_COLUMNS = ("event_ticker", "release_time_utc", "source_url")
_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")
_TAPE_COLUMNS = ["event", "market", "ts", "seq", "price", "size", "side", "trade_id"]


class RejectedRecordError(ValueError):
    """An API record that cannot become a row of the data contract; ``reason`` says why."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def _number(record: Mapping[str, Any], name: str, reason: str) -> float:
    value = record.get(name)
    if value is None or isinstance(value, bool):
        raise RejectedRecordError(reason, f"{name} is {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise RejectedRecordError(reason, f"{name} = {value!r} is not a number") from None
    if not math.isfinite(number):
        raise RejectedRecordError(reason, f"{name} = {value!r} is not finite")
    return number


def _text(record: Mapping[str, Any], name: str, reason: str) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value.strip():
        raise RejectedRecordError(reason, f"{name} is {value!r}")
    return value


def _time(record: Mapping[str, Any], name: str, reason: str) -> float:
    try:
        return parse_iso_time(record.get(name))
    except ValueError as error:
        raise RejectedRecordError(reason, f"{name}: {error}") from None


def parse_trade(record: Mapping[str, Any]) -> dict[str, Any]:
    """One trade record as ``ts``, ``price``, ``size``, ``side``, ``trade_id``, ``is_block_trade``.

    ``side`` is +1 if ``taker_outcome_side`` is ``yes`` (the taker is long YES: it bought YES or
    sold NO) and -1 if it is ``no``; only when that field is absent is the deprecated
    ``taker_side`` read.
    ``block_flag_missing`` is true when the record has no boolean ``is_block_trade`` (then the
    trade counts as not a block trade, and the report says how many did).

    Raises :class:`RejectedRecordError` with the first reason that applies, checked in this order:
    ``bad_price`` (missing or not a finite number), ``price_outside_0_1`` (not strictly between 0
    and 1), ``bad_size``, ``size_not_positive``, ``bad_time`` (no offset, unreadable), ``bad_side``.
    """
    price = _number(record, "yes_price_dollars", "bad_price")
    if not 0.0 < price < 1.0:
        raise RejectedRecordError("price_outside_0_1", f"yes_price_dollars = {price}")
    size = _number(record, "count_fp", "bad_size")
    if not size > 0.0:
        raise RejectedRecordError("size_not_positive", f"count_fp = {size}")
    ts = _time(record, "created_time", "bad_time")
    taker = record.get("taker_outcome_side")
    if taker is None:  # only an older record lacks the new field
        taker = record.get("taker_side")
    if taker == "yes":
        side = 1
    elif taker == "no":
        side = -1
    else:
        raise RejectedRecordError("bad_side", f"taker side is {taker!r}")
    trade_id = record.get("trade_id")
    block = record.get("is_block_trade")
    return {
        "ts": ts,
        "price": price,
        "size": size,
        "side": side,
        "trade_id": None if trade_id in (None, "") else str(trade_id),
        "is_block_trade": block is True,
        "block_flag_missing": not isinstance(block, bool),
    }


def parse_market(record: Mapping[str, Any]) -> dict[str, Any]:
    """One market record as ``market``, ``event``, ``series``, ``outcome``, ``open_ts`` and
    ``close_ts``, plus ``result`` (as given), ``expected_expiration_ts`` (None when the record
    has none or it cannot be read), ``price_level_structure`` (the text as given, None when absent
    or not text) and ``price_ranges`` (the JSON text of the value as given, None when absent).

    ``outcome`` is 1 if ``result`` is ``yes``, 0 if ``no`` and None for anything else: a market
    that has not settled has ``result`` empty, a scalar market has ``scalar``.

    Raises :class:`RejectedRecordError`: ``missing_ticker``, ``missing_event_ticker``,
    ``bad_open_time``, ``bad_close_time`` or ``close_before_open``.
    """
    ticker = _text(record, "ticker", "missing_ticker")
    event = _text(record, "event_ticker", "missing_event_ticker")
    open_ts = _time(record, "open_time", "bad_open_time")
    close_ts = _time(record, "close_time", "bad_close_time")
    if close_ts < open_ts:
        raise RejectedRecordError("close_before_open", f"{ticker} closes before it opens")
    series = record.get("series_ticker")
    result = record.get("result")
    result = result if isinstance(result, str) else ""
    try:
        expected = parse_iso_time(record.get("expected_expiration_time"))
    except ValueError:
        expected = None
    structure = record.get("price_level_structure")
    ranges = record.get("price_ranges")
    return {
        "market": ticker,
        "event": event,
        "series": series if isinstance(series, str) and series else event.split("-", 1)[0],
        "outcome": {"yes": 1, "no": 0}.get(result),
        "result": result,
        "open_ts": open_ts,
        "close_ts": close_ts,
        "expected_expiration_ts": expected,
        "price_level_structure": structure if isinstance(structure, str) else None,
        "price_ranges": None if ranges is None else json.dumps(ranges),
    }


# ------------------------------------------------------------------------- release calendar
def _calendar_error(path: Path, problems: list[str]) -> ValueError:
    more = f"; and {len(problems) - 5} more" if len(problems) > 5 else ""
    return ValueError(f"{path.name}: " + "; ".join(problems[:5]) + more)


def load_release_calendar(path: str | Path) -> pd.DataFrame:
    """The release calendar: ``event_ticker``, ``release_time_utc`` (UTC datetimes), ``source_url``.

    The CSV needs those three columns (others are ignored), one row per event. Every
    ``release_time_utc`` needs an explicit offset (``2026-10-24T12:30:00Z``, or ``-04:00``, which
    is converted): a time without one could be Eastern time typed in by mistake. Raises
    ``ValueError`` for a file whose name contains "draft" (a draft is not a checked calendar), a
    missing column, an empty file, a repeated or blank event, an unreadable time or a blank source
    URL; the message gives the lines.
    """
    path = Path(path)
    if "draft" in path.name.lower():
        raise ValueError(
            f"{path.name} is a draft: check every release time against the official calendars "
            "and save the file under a name without 'draft'"
        )
    try:
        raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        raise ValueError(f"{path.name} is empty") from None
    missing = [c for c in _CALENDAR_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name} lacks the columns {missing}")
    if raw.empty:
        raise ValueError(f"{path.name} has no rows")
    tickers = raw["event_ticker"].str.strip()
    urls = raw["source_url"].str.strip()
    problems: list[str] = []
    for i in np.flatnonzero(tickers == ""):
        problems.append(f"line {i + 2}: event_ticker is blank")
    repeated = sorted(set(tickers[tickers.duplicated(keep=False) & (tickers != "")]))
    if repeated:
        problems.append(f"event_ticker repeated: {', '.join(repeated)}")
    moments: list[datetime.datetime | None] = []
    for i, text in enumerate(raw["release_time_utc"].str.strip()):
        try:
            moments.append(parse_iso_datetime(text))
        except ValueError as error:
            moments.append(None)
            problems.append(f"line {i + 2}: release_time_utc: {error}")
    for i in np.flatnonzero(urls == ""):
        problems.append(f"line {i + 2}: source_url is blank")
    if problems:
        raise _calendar_error(path, problems)
    return pd.DataFrame(
        {
            "event_ticker": tickers.to_numpy(),
            "release_time_utc": pd.to_datetime(pd.Series(moments), utc=True),
            "source_url": urls.to_numpy(),
        }
    )


class DraftSummary(NamedTuple):
    """What :func:`write_calendar_draft` did: events written, events left out because no market
    gave an expected expiration time, and market records that could not be read."""

    written: int
    omitted: list[str]
    unreadable: int


def utc_text(seconds: float) -> str:
    """Unix seconds as an ISO 8601 UTC time with ``Z``, the form of the release calendar."""
    moment = datetime.datetime.fromtimestamp(seconds, datetime.UTC)
    return moment.isoformat().replace("+00:00", "Z")


def write_calendar_draft(markets: Iterable[Mapping[str, Any]], path: str | Path) -> DraftSummary:
    """Write a draft calendar from the market records: one row per event, the earliest
    ``expected_expiration_time`` of its markets as ``release_time_utc``, ``source_url`` empty.

    The file name must contain "draft", and :func:`load_release_calendar` refuses it: the times
    are Kalshi's expectation of when its markets expire, not the official release times, and must
    be checked against the BLS, BEA and Federal Reserve calendars first. An event whose markets
    have no expected expiration time is left out (listed in the summary).
    """
    path = Path(path)
    if "draft" not in path.name.lower():
        raise ValueError(f"the file name of a draft calendar must contain 'draft', got {path.name}")
    earliest: dict[str, float] = {}
    events: set[str] = set()
    unreadable = 0
    for record in markets:
        try:
            market = parse_market(record)
        except RejectedRecordError:
            unreadable += 1
            continue
        events.add(market["event"])
        expected = market["expected_expiration_ts"]
        if expected is not None:
            earliest[market["event"]] = min(expected, earliest.get(market["event"], expected))
    rows = sorted((ts, event) for event, ts in earliest.items())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(_CALENDAR_COLUMNS)
        writer.writerows((event, utc_text(ts), "") for ts, event in rows)
    return DraftSummary(len(rows), sorted(events - earliest.keys()), unreadable)


# ----------------------------------------------------------------------------- the window
@dataclass(frozen=True)
class Window:
    """Release times ``start_ts <= T < end_ts`` in Unix seconds."""

    start_ts: float
    end_ts: float

    def __post_init__(self) -> None:
        if not self.start_ts < self.end_ts:
            raise ValueError("the window must start before it ends")

    @classmethod
    def from_dates(cls, start: str, end: str) -> Window:
        """From midnight UTC at the start of the day ``start`` to midnight UTC at the end of the
        day ``end``: both dates (``YYYY-MM-DD``) are inside the window."""
        first = datetime.date.fromisoformat(start)
        last = datetime.date.fromisoformat(end) + datetime.timedelta(days=1)
        midnight = datetime.time(0, 0, tzinfo=datetime.UTC)
        return cls(
            datetime.datetime.combine(first, midnight).timestamp(),
            datetime.datetime.combine(last, midnight).timestamp(),
        )


# ---------------------------------------------------------------------------- the report
@dataclass
class BuildReport:
    """How many records each rule of :func:`build_dataset` removed. Reports add up.

    ``markets_in`` = ``sum(markets_rejected.values())`` + ``markets_no_release`` +
    ``markets_outside_window`` + ``markets_too_few_trades`` + ``markets_kept``; and ``trades_in``
    (the trade records of the markets that passed the first three rules) = ``sum(trades_rejected
    .values())`` + ``trades_duplicate`` + ``trades_block`` + ``trades_at_or_after_release`` +
    ``trades_in_dropped_markets`` + ``trades_kept``.
    """

    min_trades: int = MIN_TRADES
    markets_in: int = 0
    markets_rejected: dict[str, int] = field(default_factory=dict)
    markets_no_release: int = 0  # the event is not in the release calendar
    markets_outside_window: int = 0  # the release is not in the window
    markets_too_few_trades: int = 0  # fewer than min_trades trades are left
    markets_kept: int = 0
    markets_outcome_missing: int = 0  # of the kept: result is not yes or no
    markets_scalar_result: int = 0  # of those: result is scalar
    trades_in: int = 0
    trades_rejected: dict[str, int] = field(default_factory=dict)
    trades_duplicate: int = 0  # a trade_id seen before in the same market
    trades_block: int = 0
    trades_at_or_after_release: int = 0
    trades_in_dropped_markets: int = 0  # the trades left in a market dropped for too few
    trades_kept: int = 0
    trades_block_flag_missing: int = 0  # of the kept: the record has no is_block_trade field

    def __add__(self, other: BuildReport) -> BuildReport:
        if self.min_trades != other.min_trades:
            raise ValueError("reports with different min_trades cannot be added")
        total = BuildReport(min_trades=self.min_trades)
        for f in fields(BuildReport):
            if f.name == "min_trades":
                continue
            a, b = getattr(self, f.name), getattr(other, f.name)
            if isinstance(a, dict):
                setattr(total, f.name, {k: a.get(k, 0) + b.get(k, 0) for k in sorted(a | b)})
            else:
                setattr(total, f.name, a + b)
        return total

    def summary(self) -> list[str]:
        """The counts as lines of text, one for the markets and one for the trades."""

        def detail(counts: dict[str, int]) -> str:
            return ", ".join(f"{n} {reason}" for reason, n in sorted(counts.items())) or "none"

        return [
            f"markets: {self.markets_in} read; removed: {sum(self.markets_rejected.values())} "
            f"unreadable ({detail(self.markets_rejected)}), {self.markets_no_release} event not "
            f"in the calendar, {self.markets_outside_window} release outside the window, "
            f"{self.markets_too_few_trades} with fewer than {self.min_trades} trades; "
            f"kept {self.markets_kept} ({self.markets_outcome_missing} without an outcome, "
            f"{self.markets_scalar_result} of them scalar)",
            f"trades: {self.trades_in} read; removed: {sum(self.trades_rejected.values())} "
            f"unreadable ({detail(self.trades_rejected)}), {self.trades_duplicate} repeated "
            f"trade_id, {self.trades_block} block trades, {self.trades_at_or_after_release} at or "
            f"after the release, {self.trades_in_dropped_markets} in dropped markets; "
            f"kept {self.trades_kept} ({self.trades_block_flag_missing} without an "
            f"is_block_trade field)",
        ]


def _count(counts: dict[str, int], reason: str) -> None:
    counts[reason] = counts.get(reason, 0) + 1


# ------------------------------------------------------------------------ build the dataset
def _release_times(calendar: pd.DataFrame) -> dict[str, float]:
    """Unix seconds of each calendar event; a repeated event is an error."""
    if not calendar["event_ticker"].is_unique:
        raise ValueError("the calendar has repeated events")
    seconds = (calendar["release_time_utc"] - _EPOCH) / pd.Timedelta(seconds=1)
    return dict(zip(calendar["event_ticker"], seconds.astype(float), strict=True))


def _clean_trades(
    records: Iterable[Mapping[str, Any]], release_ts: float, report: BuildReport
) -> list[dict[str, Any]]:
    """The trades of one market that pass the trade rules, sorted by (``ts``, ``trade_id``)."""
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for record in records:
        report.trades_in += 1
        try:
            trade = parse_trade(record)
        except RejectedRecordError as rejection:
            _count(report.trades_rejected, rejection.reason)
            continue
        if trade["trade_id"] is not None:
            if trade["trade_id"] in seen:
                report.trades_duplicate += 1
                continue
            seen.add(trade["trade_id"])
        if trade["is_block_trade"]:
            report.trades_block += 1
        elif trade["ts"] >= release_ts:
            report.trades_at_or_after_release += 1
        else:
            kept.append(trade)
    kept.sort(key=lambda t: (t["ts"], t["trade_id"] or ""))
    return kept


def build_dataset(
    markets_records: Iterable[Mapping[str, Any]],
    trades_records_by_market: Mapping[str, Iterable[Mapping[str, Any]]],
    calendar: pd.DataFrame,
    *,
    min_trades: int = MIN_TRADES,
    window_start: float,
    window_end: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, BuildReport]:
    """``(tape, markets, events, report)`` from API records, with the protocol filters applied.

    ``trades_records_by_market`` maps a market ticker to its trade records; ``calendar`` is the
    frame of :func:`load_release_calendar`; the window is in Unix seconds, ``window_start`` inside
    and ``window_end`` outside. The filters and their order are in the module docstring. The
    frames are those of the data contract (``tape`` has one more column, ``trade_id``, and
    ``markets`` two, ``price_level_structure`` and ``price_ranges``, which ``panel.write_dataset``
    stores and ``panel.read_dataset`` does not return) in the order
    ``panel.read_dataset`` gives: tape by (market, ts, seq), events by (release, event), markets by
    (release of the event, event, market). An event is kept when one of its markets is.
    """
    report = BuildReport(min_trades=min_trades)
    release_of = _release_times(calendar)
    kept_markets: list[dict[str, Any]] = []
    kept_trades: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for record in markets_records:
        report.markets_in += 1
        try:
            market = parse_market(record)
        except RejectedRecordError as rejection:
            _count(report.markets_rejected, rejection.reason)
            continue
        release = release_of.get(market["event"])
        if release is None:
            report.markets_no_release += 1
            continue
        if not window_start <= release < window_end:
            report.markets_outside_window += 1
            continue
        trades = _clean_trades(trades_records_by_market.get(market["market"], ()), release, report)
        if len(trades) < min_trades:
            report.markets_too_few_trades += 1
            report.trades_in_dropped_markets += len(trades)
            continue
        market["release_ts"] = release
        kept_markets.append(market)
        kept_trades.append((market, trades))
        report.markets_kept += 1
        report.trades_kept += len(trades)
        report.markets_outcome_missing += market["outcome"] is None
        report.markets_scalar_result += market["result"] == "scalar"
        report.trades_block_flag_missing += sum(t["block_flag_missing"] for t in trades)
    return (*_frames(kept_markets, kept_trades), report)


def _frames(
    markets: list[dict[str, Any]], trades: list[tuple[dict[str, Any], list[dict[str, Any]]]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The three frames of the data contract from the kept markets and their clean trades."""
    markets = sorted(markets, key=lambda m: (m["release_ts"], m["event"], m["market"]))
    events = {m["event"]: (m["release_ts"], m["series"]) for m in markets}
    event_rows = sorted((release, event, series) for event, (release, series) in events.items())
    has_missing = any(m["outcome"] is None for m in markets)
    markets_frame = pd.DataFrame(
        {
            "market": [m["market"] for m in markets],
            "event": [m["event"] for m in markets],
            "outcome": np.array(
                [np.nan if m["outcome"] is None else m["outcome"] for m in markets],
                dtype=np.float64 if has_missing else np.int64,
            ),
            "open_ts": np.array([m["open_ts"] for m in markets], dtype=np.float64),
            "close_ts": np.array([m["close_ts"] for m in markets], dtype=np.float64),
            "price_level_structure": pd.Series(
                [m["price_level_structure"] for m in markets], dtype=object),
            "price_ranges": pd.Series([m["price_ranges"] for m in markets], dtype=object),
        }
    )
    events_frame = pd.DataFrame(
        {
            "event": [event for _, event, _ in event_rows],
            "release_ts": np.array([release for release, _, _ in event_rows], dtype=np.float64),
            "series": [series for _, _, series in event_rows],
        }
    )
    rows = sorted(
        ((market["market"], market["event"], seq, t) for market, ts in trades
         for seq, t in enumerate(ts)),
        key=lambda row: (row[0], row[3]["ts"], row[2]),
    )
    tape = pd.DataFrame(
        {
            "event": [event for _, event, _, _ in rows],
            "market": [market for market, _, _, _ in rows],
            "ts": np.array([t["ts"] for _, _, _, t in rows], dtype=np.float64),
            "seq": np.array([seq for _, _, seq, _ in rows], dtype=np.int64),
            "price": np.array([t["price"] for _, _, _, t in rows], dtype=np.float64),
            "size": np.array([t["size"] for _, _, _, t in rows], dtype=np.float64),
            "side": np.array([t["side"] for _, _, _, t in rows], dtype=np.int8),
            "trade_id": [t["trade_id"] for _, _, _, t in rows],
        }
    )[_TAPE_COLUMNS]
    return tape, markets_frame, events_frame


# ------------------------------------------------------------------------------- fetching
WANTED = "wanted"  # the market can enter the dataset: its trades are fetched
ALREADY_LOADED = "already loaded"  # the database has it from an earlier run
OUT_OF_SCOPE = "out of scope"  # unreadable, or its event has no release in the calendar window


class TradeQuery(NamedTuple):
    """One request for the trades of a market: the endpoint (``"historical"`` or ``"live"``) and
    the range of trade times in whole Unix seconds (``min_ts`` None: from the start)."""

    endpoint: str
    min_ts: int | None
    max_ts: int


def trade_queries(release_ts: float, cutoff_ts: float) -> list[TradeQuery]:
    """The requests that together cover every trade of a market before its release.

    Trades created before the cutoff (``trades_created_ts`` of ``GET /historical/cutoff``) are on
    the historical endpoint only, later ones on the live endpoint. A release at or before the
    cutoff needs the historical endpoint alone. A later release needs both: the historical one for
    the trades before the cutoff and the live one from the cutoff to the release.

    The documentation describes ``min_ts`` and ``max_ts`` as "after" and "before" a Unix time and
    does not say whether the bound itself is included, so the ranges are widened until they are
    right for either reading: the upper bounds (the cutoff for the historical endpoint, the
    release) are rounded up to a whole second, and the live range starts a whole second before the
    cutoff, so that a trade created exactly at the cutoff is read. A trade can then come from both
    endpoints; :func:`build_dataset` removes repeated ``trade_id`` and trades at or after the
    release.
    """
    last = math.ceil(release_ts)
    if release_ts <= cutoff_ts:
        return [TradeQuery("historical", None, last)]
    return [
        TradeQuery("historical", None, math.ceil(cutoff_ts)),
        TradeQuery("live", math.floor(cutoff_ts) - 1, last),
    ]


def _is_settled(record: Mapping[str, Any]) -> bool:
    return record.get("result") in ("yes", "no")


def _event_time(record: Mapping[str, Any]) -> float | None:
    """When a market's event is expected: the expected expiration time, else the close time."""
    try:
        market = parse_market(record)
    except RejectedRecordError:
        return None
    expected = market["expected_expiration_ts"]
    return market["close_ts"] if expected is None else expected


def list_series_markets(
    client: KalshiClient, series: str, *, window: Window | None = None
) -> list[dict[str, Any]]:
    """The market records of a series from ``GET /markets`` and ``GET /historical/markets``.

    One record per ticker, sorted by ticker (records without a ticker follow, in the order read).
    A ticker on both endpoints, which happens when a market is archived while the listing runs,
    keeps the live record unless only the archived one has a settled ``result``. The listing has
    no time filter, because the calendar and not a market's close time decides which markets belong
    to the window. With ``window`` (used for the draft calendar, which has no release times yet)
    records whose event time (the expected expiration time, else the close time) is more than
    :data:`MARKET_MARGIN_S` outside the window are left out, and so are none that cannot be read.
    """
    chosen: dict[str, dict[str, Any]] = {}
    unnamed: list[dict[str, Any]] = []
    for records in (
        client.iter_markets(series_ticker=series),
        client.iter_historical_markets(series_ticker=series),
    ):
        for record in records:
            ticker = record.get("ticker")
            if not isinstance(ticker, str) or not ticker:
                unnamed.append(record)
            elif ticker not in chosen or (_is_settled(record) and not _is_settled(chosen[ticker])):
                chosen[ticker] = record
    listed = [chosen[ticker] for ticker in sorted(chosen)] + unnamed
    if window is not None:
        low, high = window.start_ts - MARKET_MARGIN_S, window.end_ts + MARKET_MARGIN_S
        listed = [r for r in listed if (t := _event_time(r)) is None or low <= t < high]
    return listed


class Classified(NamedTuple):
    """A market record and what a fetch does with it: ``state`` is :data:`WANTED`,
    :data:`ALREADY_LOADED` or :data:`OUT_OF_SCOPE`; ``release_ts`` is the release of the event of a
    wanted market and None otherwise."""

    market: dict[str, Any]
    state: str
    release_ts: float | None


def classify_markets(
    records: Iterable[dict[str, Any]],
    calendar: pd.DataFrame,
    window: Window,
    *,
    loaded: Container[str] = frozenset(),
) -> list[Classified]:
    """Sort market records into those to fetch and those not to, before any trade is requested.

    A market in ``loaded`` is :data:`ALREADY_LOADED`. Otherwise it is :data:`WANTED` when
    :func:`parse_market` reads it, its event is in ``calendar`` and the release of the event is
    inside ``window`` (the first three rules of :func:`build_dataset`), else :data:`OUT_OF_SCOPE`.
    """
    release_of = _release_times(calendar)
    classified = []
    for record in records:
        ticker = record.get("ticker")
        if isinstance(ticker, str) and ticker in loaded:
            classified.append(Classified(record, ALREADY_LOADED, None))
            continue
        try:
            event = parse_market(record)["event"]
        except RejectedRecordError:
            classified.append(Classified(record, OUT_OF_SCOPE, None))
            continue
        release = release_of.get(event)
        if release is None or not window.start_ts <= release < window.end_ts:
            classified.append(Classified(record, OUT_OF_SCOPE, None))
        else:
            classified.append(Classified(record, WANTED, release))
    return classified


class Fetched(NamedTuple):
    """A market record with its trade records. ``trades`` is a list only when ``state`` is
    :data:`WANTED`; for the other states no trade was requested."""

    market: dict[str, Any]
    trades: list[dict[str, Any]] | None
    state: str


def fetch_series(
    client: KalshiClient,
    series: str,
    calendar: pd.DataFrame,
    window: Window,
    *,
    cutoff: HistoricalCutoff,
    loaded: Container[str] = frozenset(),
) -> Iterator[Fetched]:
    """Every market record of the series with its trades, one market at a time.

    The listing is :func:`list_series_markets`; the markets are sorted by
    :func:`classify_markets`; the trades of a wanted market are read from the endpoint or endpoints
    of :func:`trade_queries` for ``cutoff.trades_created_ts``. A generator: no trade of a market is
    requested before the consumer asks for it, so a consumer that writes each market to the
    database before asking for the next loses at most one market if the run is interrupted.
    """
    for item in classify_markets(
        list_series_markets(client, series), calendar, window, loaded=loaded
    ):
        if item.state != WANTED or item.release_ts is None:
            yield Fetched(item.market, None, item.state)
            continue
        ticker = item.market["ticker"]
        trades: list[dict[str, Any]] = []
        for query in trade_queries(item.release_ts, cutoff.trades_created_ts):
            read = (
                client.iter_historical_trades
                if query.endpoint == "historical"
                else client.iter_trades
            )
            trades.extend(read(ticker=ticker, min_ts=query.min_ts, max_ts=query.max_ts))
        yield Fetched(item.market, trades, WANTED)


class IngestSummary(NamedTuple):
    """What :func:`ingest_series` did. ``report`` counts what the filters removed among the
    ``out_of_scope`` and ``fetched`` markets of this run; ``already_loaded`` markets were read by an
    earlier run and are not in it. ``written`` markets (those that passed every filter) are in the
    database. ``cutoff_before`` and ``cutoff_after`` are read before and after the run."""

    report: BuildReport
    listed: int
    out_of_scope: int
    already_loaded: int
    fetched: int
    written: int
    cutoff_before: HistoricalCutoff
    cutoff_after: HistoricalCutoff

    @property
    def cutoff_moved(self) -> bool:
        """True if the cutoffs that decide which endpoint holds a trade or a market moved."""
        before, after = self.cutoff_before, self.cutoff_after
        return (before.trades_created_ts, before.market_settled_ts) != (
            after.trades_created_ts,
            after.market_settled_ts,
        )


class CalendarChangedError(ValueError):
    """The release calendar disagrees with the database about the release time of an event."""


def _check_calendar_drift(stored: Mapping[str, float], release_of: Mapping[str, float]) -> None:
    changed = sorted(e for e, ts in stored.items() if e in release_of and release_of[e] != ts)
    if changed:
        more = f" and {len(changed) - 5} more" if len(changed) > 5 else ""
        raise CalendarChangedError(
            f"the calendar gives other release times than the database for "
            f"{', '.join(changed[:5])}{more}: the stored trades were cut at the old times; use a "
            "new database or restore the calendar"
        )


def ingest_series(
    conn: sqlite3.Connection,
    client: KalshiClient,
    series: str,
    calendar: pd.DataFrame,
    window: Window,
    *,
    min_trades: int = MIN_TRADES,
    log: Callable[[str], None] = print,
) -> IngestSummary:
    """Read a series from the API and store the markets that pass the protocol filters.

    Every market is processed in turn: its trades are read, :func:`build_dataset` applies the
    filters to that one market (the counts add up across markets, ``BuildReport.__add__``) and, if
    the market is kept, ``panel.write_dataset`` stores it in its own transaction. Markets already
    in the database are skipped, so a run that stopped (a ``KalshiApiError``, an interrupt) is
    resumed by running it again. A calendar that gives another release time than the database for
    an event raises :class:`CalendarChangedError` before anything is requested. One line per
    market read goes to ``log``.
    """
    release_of = _release_times(calendar)
    panel.create_schema(conn)
    _check_calendar_drift(panel.stored_release_times(conn), release_of)
    cutoff = client.historical_cutoff()
    loaded = panel.loaded_markets(conn)
    log(
        f"{series}: trades created before {utc_text(cutoff.trades_created_ts)} are on the "
        f"historical endpoint; {len(loaded)} markets are already in the database"
    )
    report = BuildReport(min_trades=min_trades)
    counts = {WANTED: 0, ALREADY_LOADED: 0, OUT_OF_SCOPE: 0}
    written = 0
    for item in fetch_series(client, series, calendar, window, cutoff=cutoff, loaded=loaded):
        counts[item.state] += 1
        if item.state == ALREADY_LOADED:
            continue
        ticker = str(item.market.get("ticker"))
        tape, markets, events, part = build_dataset(
            [item.market],
            {ticker: item.trades or []},
            calendar,
            min_trades=min_trades,
            window_start=window.start_ts,
            window_end=window.end_ts,
        )
        report += part
        if len(markets):
            panel.write_dataset(conn, tape, markets, events, source=panel.SOURCE_KALSHI_API)
            written += 1
        if item.state == WANTED:
            left = part.trades_kept + part.trades_in_dropped_markets  # one of the two is zero
            fate = "stored" if len(markets) else f"dropped (fewer than {min_trades})"
            log(f"{ticker}: {part.trades_in} trades read, {left} left after the filters, {fate}")
    return IngestSummary(
        report=report,
        listed=sum(counts.values()),
        out_of_scope=counts[OUT_OF_SCOPE],
        already_loaded=counts[ALREADY_LOADED],
        fetched=counts[WANTED],
        written=written,
        cutoff_before=cutoff,
        cutoff_after=client.historical_cutoff(),
    )
