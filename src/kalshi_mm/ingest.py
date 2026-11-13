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
* market ``result`` -> ``outcome``: 1 for ``yes``, 0 for ``no``, missing for anything else.

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

:func:`fetch_series` and :func:`ingest_series` read a series from the live and historical
endpoints and write it to SQLite market by market, so that an interrupted run resumes.
Tested on canned responses only; not run against the live service.
"""

from __future__ import annotations

import csv
import datetime
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from kalshi_mm.kalshi_api import parse_iso_datetime, parse_iso_time

__all__ = [
    "MIN_TRADES",
    "PROTOCOL_END",
    "PROTOCOL_SERIES",
    "PROTOCOL_START",
    "BuildReport",
    "DraftSummary",
    "RejectedRecordError",
    "Window",
    "build_dataset",
    "load_release_calendar",
    "parse_market",
    "parse_trade",
    "write_calendar_draft",
]

# docs/protocol.md, "Data": the series (tickers of Kalshi's fee schedule of 2026-07-07) and the
# releases to load, the last day included.
PROTOCOL_SERIES = ("KXCPI", "KXCPIYOY", "KXPAYROLLS", "KXGDP", "KXU3", "KXFEDDECISION")
PROTOCOL_START = "2024-01-01"
PROTOCOL_END = "2026-10-31"
MIN_TRADES = 50  # docs/protocol.md, "Markets kept"

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

    ``side`` is +1 if ``taker_outcome_side`` is ``yes`` (the taker bought YES: it hit the ask)
    and -1 if it is ``no``; only when that field is absent is the deprecated ``taker_side`` read.
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
    ``close_ts``, plus ``result`` (as given) and ``expected_expiration_ts`` (None when the record
    has none or it cannot be read).

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
    return {
        "market": ticker,
        "event": event,
        "series": series if isinstance(series, str) and series else event.split("-", 1)[0],
        "outcome": {"yes": 1, "no": 0}.get(result),
        "result": result,
        "open_ts": open_ts,
        "close_ts": close_ts,
        "expected_expiration_ts": expected,
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


def _utc_text(seconds: float) -> str:
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
        writer.writerows((event, _utc_text(ts), "") for ts, event in rows)
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
    frames are those of the data contract (``tape`` has one more column, ``trade_id``, which
    ``panel.write_dataset`` stores and ``panel.read_dataset`` does not return) in the order
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
