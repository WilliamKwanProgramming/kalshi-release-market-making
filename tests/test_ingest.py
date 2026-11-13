"""Turning Kalshi API records into the data contract (src/kalshi_mm/ingest.py) and the command
line that fetches them (scripts/fetch_kalshi.py).

The records are canned and shaped like the documentation read on 2026-11-13 (docs/data.md).
Expected times come from ``calendar.timegm``, which shares no code with the parser under test.
Every filter of ``build_dataset`` has a case that sits exactly on its boundary. The fetching tests
run the client against a fake of the endpoints (``FakeApi``) that answers from canned data, and
compare what lands in SQLite, read back, with frames built by hand from the same canned data.
"""

import calendar
import dataclasses
import importlib.util
import json
import pathlib
import re
import sqlite3
import subprocess
import types
import urllib.parse
from contextlib import closing

import numpy as np
import pandas as pd
import pytest

from kalshi_mm import ingest, panel
from kalshi_mm.ingest import (
    BuildReport,
    CalendarChangedError,
    RejectedRecordError,
    Window,
    build_dataset,
    classify_markets,
    fetch_series,
    ingest_series,
    list_series_markets,
    load_release_calendar,
    parse_market,
    parse_trade,
    trade_queries,
    write_calendar_draft,
)
from kalshi_mm.kalshi_api import HistoricalCutoff, KalshiApiError, KalshiClient

REPO = pathlib.Path(__file__).resolve().parents[1]
RELEASE = "2026-10-24T12:30:00Z"
RELEASE_TS = float(calendar.timegm((2026, 10, 24, 12, 30, 0)))  # 1792845000.0
EVENT = "KXCPI-26OCT"
MARKET = "KXCPI-26OCT-T0.3"
WINDOW = {"window_start": RELEASE_TS - 86400.0, "window_end": RELEASE_TS + 86400.0}
TAPE_COLUMNS = ["event", "market", "ts", "seq", "price", "size", "side", "trade_id", "endpoint"]
OTHER = "KXCPI-26OCT-T0.4"


def unix(*parts):
    return float(calendar.timegm(parts))


def iso(seconds):
    """A Z timestamp of whole or microsecond seconds before the release, for building records."""
    micro = round((RELEASE_TS - seconds) * 1e6)
    whole, frac = divmod(micro, 1_000_000)
    before = RELEASE_TS - whole
    stamp = pd.Timestamp(before, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")
    return stamp + (f".{(1_000_000 - frac) % 1_000_000:06d}Z" if frac else "Z")


def trade_record(n=0, *, ts="2026-10-23T14:00:00Z", price="0.5600", count="10.00",
                 side="yes", block=False, trade_id=None, ticker=MARKET):
    record = {"trade_id": trade_id if trade_id is not None else f"t{n:05d}", "ticker": ticker,
              "count_fp": count, "yes_price_dollars": price, "no_price_dollars": "0.4400",
              "taker_outcome_side": side, "taker_book_side": "bid" if side == "yes" else "ask",
              "created_time": ts, "is_block_trade": block}
    return record


def market_record(ticker=MARKET, event=EVENT, *, result="yes", open_time="2026-09-01T12:00:00Z",
                  close_time=RELEASE, expected=RELEASE, **extra):
    record = {"ticker": ticker, "event_ticker": event, "status": "finalized", "result": result,
              "open_time": open_time, "close_time": close_time,
              "latest_expiration_time": "2026-10-31T12:30:00Z",
              "expected_expiration_time": expected, "settlement_ts": "2026-10-24T13:00:00Z",
              "strike_type": "greater", "floor_strike": 0.3}
    return {**record, **extra}


def write_calendar(tmp_path, rows, name="release_calendar.csv"):
    path = tmp_path / name
    lines = ["event_ticker,release_time_utc,source_url", *[",".join(r) for r in rows]]
    path.write_text("\n".join(lines) + "\n")
    return path


def calendar_frame(tmp_path, rows=None):
    rows = rows if rows is not None else [(EVENT, RELEASE, "https://www.bls.gov/schedule/")]
    return load_release_calendar(write_calendar(tmp_path, rows))


def many_trades(n, *, start=3600.0, step=10.0, **kwargs):
    """``n`` kept trades in the hour before the release, one every ``step`` seconds."""
    return [trade_record(i, ts=iso(start - i * step), **kwargs) for i in range(n)]


# ------------------------------------------------------------------------------ parse_trade
def test_parse_trade_maps_every_field():
    record = trade_record(7, ts="2026-10-23T14:00:00.123456Z", price="0.5600", count="12.50",
                          side="no", block=False, trade_id="abc-1")
    parsed = parse_trade(record)
    assert parsed["ts"] == unix(2026, 10, 23, 14, 0, 0) + 0.123456
    assert parsed["price"] == 0.56 and parsed["size"] == 12.5 and parsed["side"] == -1
    assert parsed["trade_id"] == "abc-1"
    assert parsed["is_block_trade"] is False and parsed["block_flag_missing"] is False


def test_the_price_is_the_yes_price_whatever_the_side():
    """A taker who sells YES at the bid prints the YES price of the bid: no 1 - p conversion."""
    buy = parse_trade(trade_record(side="yes", price="0.5600"))
    sell = parse_trade(trade_record(side="no", price="0.5600"))
    assert buy["price"] == sell["price"] == 0.56 and (buy["side"], sell["side"]) == (1, -1)


@pytest.mark.parametrize("outcome_side, side", [("yes", 1), ("no", -1)])
def test_side_comes_from_taker_outcome_side(outcome_side, side):
    record = trade_record(side=outcome_side)
    record["taker_book_side"] = "ask" if outcome_side == "yes" else "bid"  # opposite on purpose
    assert parse_trade(record)["side"] == side  # the book side is not read


@pytest.mark.parametrize("old, side", [("yes", 1), ("no", -1)])
def test_the_deprecated_taker_side_is_read_only_when_the_new_field_is_absent(old, side):
    record = trade_record()
    del record["taker_outcome_side"]
    record["taker_side"] = old
    assert parse_trade(record)["side"] == side
    record["taker_outcome_side"] = None  # a null new field counts as absent
    assert parse_trade(record)["side"] == side


def test_the_new_side_field_wins_over_the_deprecated_one():
    record = trade_record(side="yes")
    record["taker_side"] = "no"
    assert parse_trade(record)["side"] == 1


@pytest.mark.parametrize("value", ["", "bid", "YES", "buy", 1, ["yes"]])
def test_an_invalid_new_side_field_is_rejected_without_falling_back(value):
    record = trade_record()
    record["taker_outcome_side"] = value
    record["taker_side"] = "yes"
    with pytest.raises(RejectedRecordError) as info:
        parse_trade(record)
    assert info.value.reason == "bad_side"


def test_a_record_with_no_side_field_at_all_is_rejected():
    record = trade_record()
    del record["taker_outcome_side"], record["taker_book_side"]
    with pytest.raises(RejectedRecordError) as info:
        parse_trade(record)
    assert info.value.reason == "bad_side"


@pytest.mark.parametrize("price, reason", [
    ("0", "price_outside_0_1"), ("0.0000", "price_outside_0_1"), ("1", "price_outside_0_1"),
    ("1.0000", "price_outside_0_1"), ("-0.1000", "price_outside_0_1"),
    ("1.5000", "price_outside_0_1"), ("56", "price_outside_0_1"),
    ("", "bad_price"), (None, "bad_price"), ("abc", "bad_price"), ("nan", "bad_price"),
    ("inf", "bad_price"), (True, "bad_price"), ([0.5], "bad_price"),
])
def test_prices_outside_the_open_unit_interval_or_unreadable_are_rejected(price, reason):
    record = trade_record(price=price)
    with pytest.raises(RejectedRecordError) as info:
        parse_trade(record)
    assert info.value.reason == reason


def test_a_missing_price_field_is_rejected():
    record = trade_record()
    del record["yes_price_dollars"]
    with pytest.raises(RejectedRecordError) as info:
        parse_trade(record)
    assert info.value.reason == "bad_price"


@pytest.mark.parametrize("price", ["0.0001", "0.0100", "0.9900", "0.9999", "0.500001"])
def test_prices_just_inside_the_interval_are_accepted(price):
    assert parse_trade(trade_record(price=price))["price"] == float(price)


@pytest.mark.parametrize("count, reason", [
    ("0.00", "size_not_positive"), ("0", "size_not_positive"), ("-1.00", "size_not_positive"),
    ("-0.01", "size_not_positive"), ("", "bad_size"), (None, "bad_size"), ("many", "bad_size"),
    ("nan", "bad_size"), ("inf", "bad_size"),
])
def test_sizes_that_are_not_positive_or_unreadable_are_rejected(count, reason):
    with pytest.raises(RejectedRecordError) as info:
        parse_trade(trade_record(count=count))
    assert info.value.reason == reason


def test_the_smallest_size_is_one_hundredth_of_a_contract():
    assert parse_trade(trade_record(count="0.01"))["size"] == 0.01


@pytest.mark.parametrize("stamp", [None, "", "2026-10-23T14:00:00", "2026-10-23", "garbage", 12345])
def test_unreadable_times_and_times_without_an_offset_are_rejected(stamp):
    with pytest.raises(RejectedRecordError) as info:
        parse_trade(trade_record(ts=stamp))
    assert info.value.reason == "bad_time"


def test_the_first_failing_check_names_the_reason():
    bad = trade_record(price="2", count="-1", ts="garbage", side="x")
    with pytest.raises(RejectedRecordError) as info:
        parse_trade(bad)
    assert info.value.reason == "price_outside_0_1"
    bad["yes_price_dollars"] = "0.5"
    with pytest.raises(RejectedRecordError) as info:
        parse_trade(bad)
    assert info.value.reason == "size_not_positive"
    bad["count_fp"] = "1"
    with pytest.raises(RejectedRecordError) as info:
        parse_trade(bad)
    assert info.value.reason == "bad_time"
    bad["created_time"] = RELEASE
    with pytest.raises(RejectedRecordError) as info:
        parse_trade(bad)
    assert info.value.reason == "bad_side"


@pytest.mark.parametrize("trade_id, expected", [("x1", "x1"), (None, None), ("", None), (42, "42")])
def test_the_trade_id_is_text_or_none(trade_id, expected):
    record = trade_record()
    record["trade_id"] = trade_id
    assert parse_trade(record)["trade_id"] == expected


@pytest.mark.parametrize("flag, is_block, missing", [
    (True, True, False), (False, False, False), (None, False, True), ("true", False, True),
    (1, False, True),
])
def test_the_block_flag_is_read_only_when_it_is_a_boolean(flag, is_block, missing):
    record = trade_record()
    record["is_block_trade"] = flag
    parsed = parse_trade(record)
    assert (parsed["is_block_trade"], parsed["block_flag_missing"]) == (is_block, missing)
    del record["is_block_trade"]
    assert parse_trade(record)["block_flag_missing"] is True


# ----------------------------------------------------------------------------- parse_market
def test_parse_market_maps_every_field():
    parsed = parse_market(market_record(result="no", expected="2026-10-24T12:31:00Z"))
    assert parsed == {
        "market": MARKET, "event": EVENT, "series": "KXCPI", "outcome": 0, "result": "no",
        "open_ts": unix(2026, 9, 1, 12, 0, 0), "close_ts": RELEASE_TS,
        "expected_expiration_ts": RELEASE_TS + 60.0, "price_level_structure": None,
        "price_ranges": None}


# The price grid of a market as the Market object of the documentation describes it: a structure
# name and ranges of prices with the step of each, all in USD as text.
GRID = {"price_level_structure": "tapered_deci_cent",
        "price_ranges": [{"start": "0.0000", "end": "0.1000", "step": "0.0010"},
                         {"start": "0.1000", "end": "0.9000", "step": "0.0100"},
                         {"start": "0.9000", "end": "1.0000", "step": "0.0010"}]}


def test_the_price_grid_is_kept_as_given():
    parsed = parse_market(market_record(**GRID))
    assert parsed["price_level_structure"] == "tapered_deci_cent"
    assert json.loads(parsed["price_ranges"]) == GRID["price_ranges"]
    assert parse_market(market_record(price_ranges=[]))["price_ranges"] == "[]"
    # a structure that is not text is not kept; ranges of any shape are kept as JSON text
    odd = parse_market(market_record(price_level_structure=3, price_ranges={"step": 0.01}))
    assert odd["price_level_structure"] is None and odd["price_ranges"] == '{"step": 0.01}'


@pytest.mark.parametrize("result, outcome", [
    ("yes", 1), ("no", 0), ("", None), ("scalar", None), (None, None), ("YES", None),
    ("void", None), (1, None),
])
def test_the_outcome_is_one_for_yes_zero_for_no_and_missing_otherwise(result, outcome):
    assert parse_market(market_record(result=result))["outcome"] == outcome


def test_a_record_without_a_result_field_has_a_missing_outcome():
    record = market_record()
    del record["result"]
    assert parse_market(record)["outcome"] is None and parse_market(record)["result"] == ""


@pytest.mark.parametrize("event, series", [
    ("KXCPI-26OCT", "KXCPI"), ("KXCPIYOY-26OCT", "KXCPIYOY"),
    ("KXFEDDECISION-26OCT", "KXFEDDECISION"),
    ("PLAIN", "PLAIN"), ("A-B-C", "A"),
])
def test_the_series_is_the_event_ticker_up_to_the_first_hyphen(event, series):
    assert parse_market(market_record(event=event))["series"] == series


def test_a_series_ticker_field_wins_when_the_record_has_one():
    assert parse_market(market_record(series_ticker="KXOTHER"))["series"] == "KXOTHER"
    assert parse_market(market_record(series_ticker=""))["series"] == "KXCPI"
    assert parse_market(market_record(series_ticker=None))["series"] == "KXCPI"


@pytest.mark.parametrize("expected", [None, "", "garbage", "2026-10-24T12:30:00", 5])
def test_an_absent_or_unreadable_expected_expiration_time_is_none(expected):
    assert parse_market(market_record(expected=expected))["expected_expiration_ts"] is None
    record = market_record()
    del record["expected_expiration_time"]
    assert parse_market(record)["expected_expiration_ts"] is None


@pytest.mark.parametrize("change, reason", [
    ({"ticker": ""}, "missing_ticker"), ({"ticker": None}, "missing_ticker"),
    ({"event_ticker": " "}, "missing_event_ticker"), ({"event_ticker": 3}, "missing_event_ticker"),
    ({"open_time": None}, "bad_open_time"), ({"open_time": "2026-09-01"}, "bad_open_time"),
    ({"close_time": "soon"}, "bad_close_time"), ({"close_time": "2026-10-24T12:30:00"},
                                                 "bad_close_time"),
    ({"close_time": "2026-08-01T00:00:00Z"}, "close_before_open"),
])
def test_unreadable_market_records_are_rejected(change, reason):
    with pytest.raises(RejectedRecordError) as info:
        parse_market(market_record(**change))
    assert info.value.reason == reason


def test_a_market_that_closes_the_moment_it_opens_is_readable():
    stamp = "2026-10-24T12:30:00Z"
    assert parse_market(market_record(open_time=stamp, close_time=stamp))["close_ts"] == RELEASE_TS


# ------------------------------------------------------------------------- release calendar
def test_a_valid_calendar_is_loaded_with_utc_times_and_stripped_text(tmp_path):
    path = write_calendar(tmp_path, [
        (" KXCPI-26OCT ", "2026-10-24T12:30:00Z", " https://www.bls.gov/schedule/ "),
        ("KXPAYROLLS-26OCT", "2026-10-02T08:30:00-04:00", "https://www.bls.gov/"),
        ("KXFEDDECISION-26OCT", "2026-10-28 18:00:00+00:00", "https://www.federalreserve.gov/"),
    ])
    frame = load_release_calendar(path)
    assert list(frame.columns) == ["event_ticker", "release_time_utc", "source_url"]
    assert frame["event_ticker"].tolist() == [
        "KXCPI-26OCT", "KXPAYROLLS-26OCT", "KXFEDDECISION-26OCT"]
    assert frame["source_url"].tolist()[0] == "https://www.bls.gov/schedule/"
    assert str(frame["release_time_utc"].dt.tz) == "UTC"
    seconds = (frame["release_time_utc"] - pd.Timestamp("1970-01-01", tz="UTC")
               ) / pd.Timedelta(seconds=1)
    assert seconds.tolist() == [
        RELEASE_TS, unix(2026, 10, 2, 12, 30, 0), unix(2026, 10, 28, 18, 0, 0)]


def test_extra_columns_and_a_byte_order_mark_are_tolerated(tmp_path):
    path = tmp_path / "release_calendar.csv"
    path.write_bytes("﻿event_ticker,release_time_utc,source_url,note\n".encode()
                     + f"{EVENT},{RELEASE},https://x.gov/,checked on the 3rd\n".encode())
    frame = load_release_calendar(path)
    assert list(frame.columns) == ["event_ticker", "release_time_utc", "source_url"]
    assert len(frame) == 1


@pytest.mark.parametrize("name", ["release_calendar.draft.csv", "DRAFT_calendar.csv",
                                  "my_Draft.csv", "calendar-draft"])
def test_a_file_with_draft_in_its_name_is_refused(tmp_path, name):
    path = write_calendar(tmp_path, [(EVENT, RELEASE, "https://x.gov/")], name=name)
    with pytest.raises(ValueError, match="draft"):
        load_release_calendar(path)


def test_a_missing_file_is_a_file_error_not_a_silent_empty_calendar(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_release_calendar(tmp_path / "release_calendar.csv")


@pytest.mark.parametrize("content, match", [
    ("", "is empty"),
    ("event_ticker,release_time_utc,source_url\n", "has no rows"),
    ("event_ticker,release_time_utc\nA,2026-10-24T12:30:00Z\n", "lacks the columns.*source_url"),
    ("event,release_time_utc,source_url\nA,2026-10-24T12:30:00Z,u\n", "lacks the columns"),
])
def test_calendar_files_without_the_required_shape_are_refused(tmp_path, content, match):
    path = tmp_path / "release_calendar.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=match):
        load_release_calendar(path)


@pytest.mark.parametrize("rows, match", [
    ([("A", RELEASE, "u"), ("B", RELEASE, "u"), ("A", "2026-11-01T00:00:00Z", "u")],
     "event_ticker repeated: A"),
    ([("A", RELEASE, "u"), (" A", RELEASE, "u")], "event_ticker repeated: A"),
    ([("A", RELEASE, "u"), ("", RELEASE, "u")], "line 3: event_ticker is blank"),
    ([("A", "2026-10-24T12:30:00", "u")], "line 2: release_time_utc.*no UTC offset"),
    ([("A", RELEASE, "u"), ("B", "tomorrow", "u")], "line 3: release_time_utc"),
    ([("A", "", "u")], "line 2: release_time_utc"),
    ([("A", RELEASE, ""), ("B", RELEASE, "  ")], "line 2: source_url is blank; line 3"),
])
def test_calendar_validation_errors_name_the_problem_and_the_line(tmp_path, rows, match):
    with pytest.raises(ValueError, match=match):
        load_release_calendar(write_calendar(tmp_path, rows))


def test_many_calendar_problems_are_summarised(tmp_path):
    rows = [(f"E{i}", "never", "u") for i in range(9)]
    with pytest.raises(ValueError, match=r"line 2.*line 6: release_time_utc: .*; and 4 more$"):
        load_release_calendar(write_calendar(tmp_path, rows))


# ----------------------------------------------------------------------------- calendar draft
def test_a_draft_uses_the_earliest_expected_expiration_of_each_event(tmp_path):
    markets = [
        market_record("A-1", "KXCPI-26OCT", expected="2026-10-24T12:35:00Z"),
        market_record("A-2", "KXCPI-26OCT", expected="2026-10-24T12:30:00Z"),
        market_record("B-1", "KXPAYROLLS-26OCT", expected="2026-10-02T12:30:00.5Z"),
        market_record("C-1", "KXU3-26OCT", expected=None),  # no expected time: left out
        market_record("C-2", "KXU3-26OCT", expected=None),
        market_record("D-1", "KXGDP-26OCT", expected="2026-10-29T12:30:00Z"),
        {"ticker": "", "event_ticker": "X"},  # unreadable
    ]
    path = tmp_path / "release_calendar.draft.csv"
    summary = write_calendar_draft(markets, path)
    assert summary.written == 3 and summary.omitted == ["KXU3-26OCT"] and summary.unreadable == 1
    assert path.read_text().splitlines() == [
        "event_ticker,release_time_utc,source_url",
        "KXPAYROLLS-26OCT,2026-10-02T12:30:00.500000Z,",
        "KXCPI-26OCT,2026-10-24T12:30:00Z,",
        "KXGDP-26OCT,2026-10-29T12:30:00Z,",
    ]


def test_the_name_of_a_draft_must_say_draft_and_the_loader_refuses_the_file(tmp_path):
    with pytest.raises(ValueError, match="draft"):
        write_calendar_draft([market_record()], tmp_path / "release_calendar.csv")
    assert not (tmp_path / "release_calendar.csv").exists()
    path = tmp_path / "out" / "release_calendar.draft.csv"  # the folder is created
    write_calendar_draft([market_record()], path)
    with pytest.raises(ValueError, match="draft"):
        load_release_calendar(path)


def test_a_draft_that_is_checked_and_renamed_loads_with_the_same_times(tmp_path):
    """Check the checker: the draft's text is read back as the times it was written from."""
    draft = tmp_path / "release_calendar.draft.csv"
    markets = [market_record(expected="2026-10-24T12:30:00.123456Z")]
    write_calendar_draft(markets, draft)
    checked = tmp_path / "release_calendar.csv"
    checked.write_text(draft.read_text().replace(",\n", ",https://www.bls.gov/schedule/\n"))
    frame = load_release_calendar(checked)
    seconds = (frame["release_time_utc"] - pd.Timestamp("1970-01-01", tz="UTC")
               ) / pd.Timedelta(seconds=1)
    assert seconds.tolist() == [RELEASE_TS + 0.123456]


def test_a_draft_of_no_events_has_only_the_header(tmp_path):
    path = tmp_path / "x.draft.csv"
    summary = write_calendar_draft([market_record(expected=None)], path)
    assert summary == (0, [EVENT], 0)
    assert path.read_text() == "event_ticker,release_time_utc,source_url\n"


# ------------------------------------------------------------------------------- the window
def test_window_from_dates_includes_both_end_dates():
    window = Window.from_dates("2024-01-01", "2026-10-31")
    assert window.start_ts == unix(2024, 1, 1, 0, 0, 0)
    assert window.end_ts == unix(2026, 11, 1, 0, 0, 0)  # midnight after the last day


@pytest.mark.parametrize("start, end", [("2026-10-31", "2024-01-01"), ("2024-01-01", "2023-12-31")])
def test_a_window_must_start_before_it_ends(start, end):
    with pytest.raises(ValueError):
        Window.from_dates(start, end)
    with pytest.raises(ValueError):
        Window(5.0, 5.0)
    assert Window.from_dates("2024-01-01", "2024-01-01").end_ts - unix(2024, 1, 1, 0, 0, 0) == 86400


def test_the_protocol_constants_are_those_of_the_protocol_document():
    text = (REPO / "docs" / "protocol.md").read_text()
    series_line = re.search(r"^- Series .*?(?=^- )", text, flags=re.S | re.M).group(0)
    assert re.findall(r"`(KX\w+)`", series_line) == list(ingest.PROTOCOL_SERIES)
    releases = re.search(r"^- Releases: .*$", text, flags=re.M).group(0)
    assert ingest.PROTOCOL_START in releases and ingest.PROTOCOL_END in releases
    assert f"at least {ingest.MIN_TRADES} trades" in text


# ------------------------------------------------------------------------------ build_dataset
def build(tmp_path, markets, trades, rows=None, **kwargs):
    kwargs = {**WINDOW, **kwargs}
    return build_dataset(markets, trades, calendar_frame(tmp_path, rows), **kwargs)


def assert_report_adds_up(report):
    assert report.markets_in == (sum(report.markets_rejected.values()) + report.markets_no_release
                                 + report.markets_outside_window + report.markets_too_few_trades
                                 + report.markets_kept)
    assert report.trades_in == (sum(report.trades_rejected.values()) + report.trades_duplicate
                                + report.trades_block + report.trades_at_or_after_release
                                + report.trades_in_dropped_markets + report.trades_kept)


def test_a_clean_market_gives_the_contract_frames_by_hand(tmp_path):
    trades = [
        trade_record(1, ts="2026-10-24T12:00:00Z", price="0.4000", count="5.00", side="yes"),
        trade_record(2, ts="2026-10-24T12:10:00.250000Z", price="0.3500", count="2.50", side="no"),
        trade_record(3, ts="2026-10-24T12:29:59.999999Z", price="0.9000", count="1.00", side="yes"),
    ]
    tape, markets, events, report = build(tmp_path, [market_record()], {MARKET: trades},
                                          min_trades=3)
    assert list(tape.columns) == TAPE_COLUMNS
    assert tape["event"].tolist() == [EVENT] * 3 and tape["market"].tolist() == [MARKET] * 3
    assert tape["ts"].tolist() == [
        unix(2026, 10, 24, 12, 0, 0), unix(2026, 10, 24, 12, 10, 0) + 0.25,
        unix(2026, 10, 24, 12, 29, 59) + 0.999999]
    assert tape["seq"].tolist() == [0, 1, 2] and tape["price"].tolist() == [0.40, 0.35, 0.90]
    assert tape["size"].tolist() == [5.0, 2.5, 1.0] and tape["side"].tolist() == [1, -1, 1]
    assert tape["trade_id"].tolist() == ["t00001", "t00002", "t00003"]
    dtypes = (tape["ts"].dtype, tape["seq"].dtype, tape["side"].dtype)
    assert dtypes == (np.float64, np.int64, np.int8)
    assert markets.to_dict("list") == {
        "market": [MARKET], "event": [EVENT], "outcome": [1],
        "open_ts": [unix(2026, 9, 1, 12, 0, 0)], "close_ts": [RELEASE_TS],
        "price_level_structure": [None], "price_ranges": [None]}
    assert events.to_dict("list") == {"event": [EVENT], "release_ts": [RELEASE_TS],
                                      "series": ["KXCPI"]}
    assert markets["outcome"].dtype == np.int64 and events["release_ts"].dtype == np.float64
    assert report.markets_kept == 1 and report.trades_kept == 3 and report.trades_in == 3
    assert_report_adds_up(report)


# ------------------------------------------------ each filter, on and either side of its boundary
def test_a_trade_at_exactly_the_release_is_dropped_and_one_microsecond_before_is_kept(tmp_path):
    trades = [trade_record(1, ts="2026-10-24T12:29:59.999999Z"),
              trade_record(2, ts="2026-10-24T12:30:00Z"),
              trade_record(3, ts="2026-10-24T12:30:00.000001Z"),
              trade_record(4, ts="2026-10-24T12:30:00.000000Z", trade_id="t-at-release-again")]
    tape, _, _, report = build(tmp_path, [market_record()], {MARKET: trades}, min_trades=1)
    assert tape["trade_id"].tolist() == ["t00001"]
    assert report.trades_at_or_after_release == 3 and report.trades_kept == 1
    assert_report_adds_up(report)


def test_exactly_min_trades_is_kept_and_one_fewer_is_dropped(tmp_path):
    fifty, forty_nine = many_trades(50), many_trades(49)
    tape, markets, _, report = build(tmp_path, [market_record()], {MARKET: fifty})
    assert len(tape) == 50 and len(markets) == 1 and report.markets_kept == 1
    tape, markets, events, report = build(tmp_path, [market_record()], {MARKET: forty_nine})
    assert len(tape) == 0 and len(markets) == 0 and len(events) == 0
    assert report.markets_too_few_trades == 1 and report.trades_in_dropped_markets == 49
    assert report.trades_kept == 0 and report.markets_kept == 0
    assert_report_adds_up(report)
    tape, *_ = build(tmp_path, [market_record()], {MARKET: forty_nine}, min_trades=49)
    assert len(tape) == 49  # the minimum is a parameter


def test_the_minimum_counts_only_the_trades_that_survive_the_other_filters(tmp_path):
    """50 records, but 3 are block trades, 2 repeat an id and 1 is at the release: 44 are left."""
    trades = many_trades(44)
    trades += [trade_record(100 + i, ts=iso(100.0 + i), block=True) for i in range(3)]
    trades += [dict(trades[0]), dict(trades[1])]
    trades += [trade_record(200, ts=RELEASE)]
    assert len(trades) == 50
    tape, markets, _, report = build(tmp_path, [market_record()], {MARKET: trades})
    assert len(tape) == 0 and report.markets_too_few_trades == 1
    assert (report.trades_block, report.trades_duplicate) == (3, 2)
    assert report.trades_at_or_after_release == 1
    assert report.trades_in_dropped_markets == 44
    assert_report_adds_up(report)


def test_release_window_start_is_inside_and_window_end_is_outside(tmp_path):
    rows = [("E-start", "2026-10-24T00:00:00Z", "u"), ("E-before", "2026-10-23T23:59:59Z", "u"),
            ("E-end", "2026-10-26T00:00:00Z", "u"), ("E-last", "2026-10-25T23:59:59Z", "u")]
    start, end = unix(2026, 10, 24, 0, 0, 0), unix(2026, 10, 26, 0, 0, 0)
    markets = [market_record(f"M-{e}", e) for e in ("E-start", "E-before", "E-end", "E-last")]
    trades = {f"M-{e}": [trade_record(i, ts="2026-10-20T00:00:00Z", ticker=f"M-{e}")
                         for i in range(2)]
              for e in ("E-start", "E-before", "E-end", "E-last")}
    _, kept, events, report = build(tmp_path, markets, trades, rows, min_trades=1,
                                    window_start=start, window_end=end)
    assert sorted(kept["market"]) == ["M-E-last", "M-E-start"]
    assert events["event"].tolist() == ["E-start", "E-last"]  # in release order
    assert report.markets_outside_window == 2 and report.markets_kept == 2
    assert_report_adds_up(report)


def test_a_market_whose_event_is_not_in_the_calendar_is_counted_not_dropped_silently(tmp_path):
    markets = [market_record(), market_record("OTHER-1", "KXOTHER-26OCT")]
    trades = {MARKET: many_trades(50), "OTHER-1": many_trades(50, ticker="OTHER-1")}
    tape, kept, _, report = build(tmp_path, markets, trades)
    assert kept["market"].tolist() == [MARKET] and set(tape["market"]) == {MARKET}
    assert report.markets_no_release == 1 and report.markets_in == 2
    assert report.trades_in == 50  # the trades of a market without a release are never read
    assert_report_adds_up(report)


def test_block_trades_are_dropped_and_counted_the_others_are_kept(tmp_path):
    trades = many_trades(50)
    trades[10] = {**trades[10], "is_block_trade": True}
    trades[20] = {**trades[20], "is_block_trade": True}
    tape, _, _, report = build(tmp_path, [market_record()], {MARKET: trades}, min_trades=48)
    assert len(tape) == 48 and report.trades_block == 2
    assert not set(tape["trade_id"]) & {"t00010", "t00020"}
    _, _, _, report = build(tmp_path, [market_record()], {MARKET: trades}, min_trades=49)
    assert report.markets_too_few_trades == 1  # 48 left, 49 needed
    assert_report_adds_up(report)


def test_a_missing_block_flag_keeps_the_trade_and_is_counted(tmp_path):
    trades = many_trades(50)
    for trade in trades[:5]:
        del trade["is_block_trade"]
    tape, _, _, report = build(tmp_path, [market_record()], {MARKET: trades})
    assert len(tape) == 50 and report.trades_block_flag_missing == 5 and report.trades_block == 0


def test_repeated_trade_ids_are_removed_and_the_first_record_is_kept(tmp_path):
    trades = many_trades(50)
    again = dict(trades[5], yes_price_dollars="0.9900")  # same id, other price: the first stays
    tape, _, _, report = build(tmp_path, [market_record()], {MARKET: [*trades, again, trades[6]]})
    assert len(tape) == 50 and report.trades_duplicate == 2 and report.trades_in == 52
    assert tape.loc[tape["trade_id"] == "t00005", "price"].tolist() == [0.56]
    assert_report_adds_up(report)


def test_trades_without_an_id_are_never_treated_as_repeats(tmp_path):
    trades = many_trades(50)
    for trade in trades:
        trade["trade_id"] = None
    tape, _, _, report = build(tmp_path, [market_record()], {MARKET: [*trades, *trades[:3]]},
                               min_trades=50)
    assert len(tape) == 53 and report.trades_duplicate == 0 and tape["trade_id"].isna().all()


def test_the_same_trade_id_in_two_markets_is_not_a_repeat(tmp_path):
    rows = [(EVENT, RELEASE, "u")]
    markets = [market_record(), market_record(OTHER)]
    trades = {MARKET: many_trades(50), OTHER: many_trades(50, ticker=OTHER)}
    tape, kept, _, report = build(tmp_path, markets, trades, rows)
    assert len(tape) == 100 and report.trades_duplicate == 0 and len(kept) == 2


def test_unreadable_trades_and_markets_are_counted_by_reason(tmp_path):
    trades = many_trades(50)
    trades += [trade_record(300, price="1.0"), trade_record(301, price="0"),
               trade_record(302, count="0"), trade_record(303, ts="garbage"),
               trade_record(304, side="maybe"), trade_record(305, price="abc")]
    markets = [market_record(), market_record("BAD-1", "KXCPI-26OCT", close_time="never"),
               {"ticker": "BAD-2"}]
    tape, kept, _, report = build(tmp_path, markets, {MARKET: trades})
    assert len(tape) == 50
    assert report.trades_rejected == {"price_outside_0_1": 2, "size_not_positive": 1,
                                      "bad_time": 1, "bad_side": 1, "bad_price": 1}
    assert report.markets_rejected == {"bad_close_time": 1, "missing_event_ticker": 1}
    assert report.trades_in == 56 and report.markets_in == 3
    assert_report_adds_up(report)


def test_markets_without_a_yes_or_no_result_are_kept_with_a_missing_outcome_and_counted(tmp_path):
    markets = [market_record(MARKET, result="yes"), market_record("KXCPI-26OCT-T0.4", result=""),
               market_record("KXCPI-26OCT-T0.5", result="scalar")]
    trades = {m["ticker"]: many_trades(50, ticker=m["ticker"]) for m in markets}
    _, kept, _, report = build(tmp_path, markets, trades)
    assert len(kept) == 3 and kept["outcome"].dtype == np.float64
    assert kept["outcome"].isna().tolist() == [False, True, True]
    assert report.markets_outcome_missing == 2 and report.markets_scalar_result == 1


def test_an_event_is_kept_when_one_of_its_markets_is(tmp_path):
    markets = [market_record(MARKET), market_record(OTHER)]
    trades = {MARKET: many_trades(50), OTHER: many_trades(10, ticker=OTHER)}
    tape, kept, events, report = build(tmp_path, markets, trades)
    assert kept["market"].tolist() == [MARKET] and events["event"].tolist() == [EVENT]
    assert report.markets_too_few_trades == 1
    markets = [market_record(OTHER)]
    _, kept, events, _ = build(tmp_path, markets, {OTHER: many_trades(10)})
    assert len(kept) == 0 and len(events) == 0  # no market: no event


def test_a_market_without_any_trade_records_is_dropped_by_the_minimum(tmp_path):
    _, kept, _, report = build(tmp_path, [market_record()], {})
    assert len(kept) == 0 and report.markets_too_few_trades == 1 and report.trades_in == 0
    _, kept, _, report = build(tmp_path, [market_record()], {}, min_trades=0)
    assert len(kept) == 1  # a minimum of zero keeps it, with no trades


# -------------------------------------------------------------------- order and determinism
def test_trades_are_numbered_by_time_then_trade_id_whatever_the_order_of_the_records(tmp_path):
    base = [trade_record(1, ts="2026-10-24T12:00:00Z", trade_id="b"),
            trade_record(2, ts="2026-10-24T12:00:00Z", trade_id="a"),
            trade_record(3, ts="2026-10-24T11:00:00Z", trade_id="z"),
            trade_record(4, ts="2026-10-24T12:00:01Z", trade_id="c")]
    expected_ids = ["z", "a", "b", "c"]
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2]):
        tape, *_ = build(tmp_path, [market_record()], {MARKET: [base[i] for i in order]},
                         min_trades=4)
        assert tape["trade_id"].tolist() == expected_ids and tape["seq"].tolist() == [0, 1, 2, 3]


def test_trades_without_ids_keep_their_input_order_at_equal_times(tmp_path):
    first = trade_record(1, ts="2026-10-24T12:00:00Z", price="0.30")
    second = trade_record(2, ts="2026-10-24T12:00:00Z", price="0.60")
    for trade in (first, second):
        trade["trade_id"] = None
    tape, *_ = build(tmp_path, [market_record()], {MARKET: [first, second]}, min_trades=2)
    assert tape["price"].tolist() == [0.30, 0.60]


def test_frames_are_sorted_like_panel_read_dataset(tmp_path):
    rows = [("KXB-26OCT", "2026-10-24T12:30:00Z", "u"), ("KXA-26NOV", "2026-10-25T12:30:00Z", "u"),
            ("KXA-26OCT", "2026-10-24T12:30:00Z", "u")]
    names = [("B-2", "KXB-26OCT"), ("A-9", "KXA-26NOV"), ("A-1", "KXA-26OCT"), ("B-1", "KXB-26OCT")]
    markets = [market_record(m, e, close_time="2026-10-25T12:30:00Z") for m, e in names]
    trades = {m: many_trades(50, ticker=m) for m, _ in names}
    tape, kept, events, _ = build(tmp_path, markets, trades, rows,
                                  window_end=RELEASE_TS + 3 * 86400.0)
    assert events["event"].tolist() == ["KXA-26OCT", "KXB-26OCT", "KXA-26NOV"]  # release, then name
    assert kept["market"].tolist() == ["A-1", "B-1", "B-2", "A-9"]
    assert tape["market"].is_monotonic_increasing  # by market name
    assert (tape.groupby("market")["seq"].apply(lambda s: s.tolist() == list(range(len(s))))).all()


def test_changing_trades_at_or_after_the_release_changes_nothing_that_is_kept(tmp_path):
    """Look-ahead guard: the kept data depend only on what happened before the release."""
    base = many_trades(50)
    late = [trade_record(900, ts="2026-10-24T12:30:00Z", price="0.10"),
            trade_record(901, ts="2026-10-24T13:00:00Z", price="0.20", side="no")]
    other = [trade_record(900, ts="2026-10-24T12:30:00Z", price="0.99", side="no", count="7.00"),
             trade_record(902, ts="2026-10-25T09:00:00Z", price="0.50")]
    a = build(tmp_path, [market_record()], {MARKET: base + late})
    b = build(tmp_path, [market_record()], {MARKET: base + other})
    c = build(tmp_path, [market_record()], {MARKET: base})
    for x, y in ((a, b), (a, c)):
        for frame_x, frame_y in zip(x[:3], y[:3], strict=True):
            pd.testing.assert_frame_equal(frame_x, frame_y, check_exact=True)


def test_the_result_does_not_depend_on_the_order_of_the_market_records(tmp_path):
    markets = [market_record(f"KXCPI-26OCT-T{i}") for i in range(4)]
    trades = {m["ticker"]: many_trades(50, ticker=m["ticker"]) for m in markets}
    a = build(tmp_path, markets, trades)
    b = build(tmp_path, markets[::-1], trades)
    for frame_a, frame_b in zip(a[:3], b[:3], strict=True):
        pd.testing.assert_frame_equal(frame_a, frame_b, check_exact=True)


def test_nothing_kept_gives_empty_frames_with_the_contract_columns(tmp_path):
    tape, kept, events, report = build(tmp_path, [], {})
    assert list(tape.columns) == TAPE_COLUMNS
    assert list(kept.columns) == ["market", "event", "outcome", "open_ts", "close_ts",
                                  "price_level_structure", "price_ranges"]
    assert list(events.columns) == ["event", "release_ts", "series"]
    assert len(tape) == len(kept) == len(events) == 0 and report.markets_in == 0
    assert tape["ts"].dtype == np.float64 and tape["side"].dtype == np.int8
    assert tape["seq"].dtype == np.int64 and events["release_ts"].dtype == np.float64


def test_a_calendar_with_a_repeated_event_is_refused_even_if_built_by_hand():
    frame = pd.DataFrame({"event_ticker": ["A", "A"],
                          "release_time_utc": pd.to_datetime([RELEASE, RELEASE], utc=True),
                          "source_url": ["u", "u"]})
    with pytest.raises(ValueError, match="repeated"):
        build_dataset([], {}, frame, **WINDOW)


def test_the_release_comes_from_the_calendar_not_from_the_market_close_time(tmp_path):
    """The market closes a day after the release; the calendar says when the release was."""
    market = market_record(close_time="2026-10-25T12:30:00Z", expected="2026-10-25T12:30:00Z")
    trades = many_trades(50) + [trade_record(500, ts="2026-10-24T12:30:00Z"),
                                trade_record(501, ts="2026-10-24T18:00:00Z")]
    tape, kept, events, report = build(tmp_path, [market], {MARKET: trades})
    assert events["release_ts"].tolist() == [RELEASE_TS]
    assert kept["close_ts"].tolist() == [RELEASE_TS + 86400.0]
    assert len(tape) == 50 and report.trades_at_or_after_release == 2


# ------------------------------------------------------------------------------- the report
def test_reports_add_field_by_field_and_merge_reason_counts():
    a = BuildReport(markets_in=2, markets_kept=1, trades_rejected={"bad_time": 2},
                    markets_rejected={"bad_open_time": 1})
    b = BuildReport(markets_in=3, markets_kept=2, trades_rejected={"bad_time": 1, "bad_side": 4})
    total = a + b
    assert total.markets_in == 5 and total.markets_kept == 3
    assert total.trades_rejected == {"bad_side": 4, "bad_time": 3}
    assert total.markets_rejected == {"bad_open_time": 1}
    assert a.trades_rejected == {"bad_time": 2}  # the operands are unchanged
    every = {f.name for f in dataclasses.fields(BuildReport)} - {"min_trades"}
    ones = BuildReport(**{n: ({"x": 1} if isinstance(getattr(BuildReport(), n), dict) else 1)
                          for n in every})
    twice = ones + ones
    assert all(getattr(twice, n) == ({"x": 2} if n.endswith("rejected") else 2) for n in every)
    with pytest.raises(ValueError, match="min_trades"):
        BuildReport(min_trades=10) + BuildReport(min_trades=20)


def test_the_summary_lines_state_the_counts(tmp_path):
    trades = many_trades(50) + [trade_record(300, price="1.0"), trade_record(301, block=True)]
    _, _, _, report = build(tmp_path, [market_record()], {MARKET: trades})
    markets_line, trades_line, endpoints_line = report.summary()
    assert "1 read" in markets_line and "kept 1" in markets_line
    assert "52 read" in trades_line and "1 price_outside_0_1" in trades_line
    assert "1 block trades" in trades_line and "kept 50" in trades_line
    assert endpoints_line.startswith("both trade endpoints: 0 trade_id returned by both")
    assert "; 0 groups of kept trades" in endpoints_line


# --------------------------------------------------------- trades read from both endpoints
def marked(record, endpoint):
    """A trade record as fetch_series passes it on: marked with the endpoint it came from."""
    return {**record, ingest.ENDPOINT_FIELD: endpoint}


def test_an_unmarked_record_has_no_endpoint_and_a_marked_one_keeps_its_mark():
    assert parse_trade(trade_record())["endpoint"] is None
    assert parse_trade(marked(trade_record(), "historical"))["endpoint"] == "historical"
    assert parse_trade(marked(trade_record(), "live"))["endpoint"] == "live"
    assert parse_trade(marked(trade_record(), "both"))["endpoint"] is None  # not a read mark


def test_one_trade_id_from_both_endpoints_is_kept_once_marked_both_and_counted(tmp_path):
    trades = [marked(t, "historical") for t in many_trades(50)]
    again = marked(dict(trades[7]), "live")  # the same trade from the live endpoint
    third = marked(dict(trades[7]), "live")  # and once more: still one id on both endpoints
    tape, _, _, report = build(tmp_path, [market_record()], {MARKET: [*trades, again, third]})
    assert len(tape) == 50 and report.trades_duplicate == 2
    assert report.trades_id_on_both_endpoints == 1 and report.trades_lookalike_groups == 0
    assert tape.loc[tape["trade_id"] == "t00007", "endpoint"].tolist() == ["both"]
    assert (tape["endpoint"] == "historical").sum() == 49
    assert_report_adds_up(report)


def test_a_repeat_from_the_same_endpoint_is_not_counted_as_both(tmp_path):
    trades = [marked(t, "live") for t in many_trades(50)]
    tape, _, _, report = build(tmp_path, [market_record()], {MARKET: [*trades, trades[3]]})
    assert report.trades_duplicate == 1 and report.trades_id_on_both_endpoints == 0
    assert set(tape["endpoint"]) == {"live"}


def test_different_ids_with_identical_fields_from_the_two_endpoints_are_kept_and_counted(tmp_path):
    trades = [marked(t, "historical") for t in many_trades(50)]
    twin = marked({**trades[4], "trade_id": "other-id"}, "live")
    tape, _, _, report = build(tmp_path, [market_record()], {MARKET: [*trades, twin]})
    assert len(tape) == 51 and report.trades_duplicate == 0
    assert report.trades_lookalike_groups == 1 and report.trades_id_on_both_endpoints == 0
    # a twin that differs in one field (size, price, side or time) is no group
    for change in ({"count_fp": "11.00"}, {"yes_price_dollars": "0.5700"},
                   {"taker_outcome_side": "no"}, {"created_time": iso(9999.0)}):
        _, _, _, report = build(tmp_path, [market_record()],
                                {MARKET: [*trades, {**twin, **change}]})
        assert report.trades_lookalike_groups == 0, change


def test_identical_fields_within_one_endpoint_are_kept_and_not_counted(tmp_path):
    trades = [marked(t, "live") for t in many_trades(50)]
    twins = [{**trades[4], "trade_id": "other-id"}, {**trades[4], "trade_id": "third-id"}]
    tape, _, _, report = build(tmp_path, [market_record()], {MARKET: [*trades, *twins]})
    assert len(tape) == 52 and report.trades_lookalike_groups == 0


def test_lookalike_groups_counts_groups_not_trades_and_needs_both_endpoints():
    base = {"ts": 1.0, "price": 0.5, "size": 2.0, "side": 1}
    h, live, both = ({**base, "endpoint": e} for e in ("historical", "live", "both"))
    assert ingest.lookalike_groups([h, live]) == 1
    assert ingest.lookalike_groups([h, live, {**live}]) == 1  # three trades, one group
    assert ingest.lookalike_groups([h, {**h}]) == 0
    assert ingest.lookalike_groups([both, {**h}]) == 1  # the id on both counts as each
    assert ingest.lookalike_groups([both]) == 0  # one trade is no group
    assert ingest.lookalike_groups([h, {**base, "endpoint": None}]) == 0
    assert ingest.lookalike_groups([{**h, "market": "A"}, {**live, "market": "B"}]) == 0
    assert ingest.lookalike_groups([h, live, {**h, "side": -1}, {**live, "side": -1}]) == 2


# ------------------------------------------------------------------- the fake endpoints
BASE = "https://example.test/trade-api/v2"
CUTOFFS = {"market_settled_ts": "2026-09-15T00:00:00Z",
           "trades_created_ts": "2026-10-20T00:00:00Z",
           "orders_updated_ts": "2026-10-30T00:00:00Z"}
CUTOFF = HistoricalCutoff(market_settled_ts=unix(2026, 9, 15, 0, 0, 0),
                          trades_created_ts=unix(2026, 10, 20, 0, 0, 0),
                          orders_updated_ts=unix(2026, 10, 30, 0, 0, 0))
CUTOFF_TS = CUTOFF.trades_created_ts
R_SEP = unix(2026, 9, 11, 12, 30, 0)
SEP, OCT, JUL, NOV = "KXCPI-26SEP", "KXCPI-26OCT", "KXCPI-26JUL", "KXCPI-26NOV"
S3, S4, S9, O3, O4 = (f"{SEP}-T0.3", f"{SEP}-T0.4", f"{SEP}-T0.9", f"{OCT}-T0.3", f"{OCT}-T0.4")
J3, N3 = f"{JUL}-T0.3", f"{NOV}-T0.3"
FETCH_WINDOW = Window.from_dates("2026-09-01", "2026-10-31")
SOURCE_URL = "https://www.bls.gov/schedule/"
CALENDAR_ROWS = (
    (SEP, "2026-09-11T12:30:00Z", SOURCE_URL),
    (OCT, RELEASE, SOURCE_URL),
    (JUL, "2026-07-15T12:30:00Z", SOURCE_URL),
)
TRADE_PATHS = ("/historical/trades", "/markets/trades")
STORED_COLUMNS = "market, ts, seq, price, size, side, trade_id"


def zulu(seconds):
    """A timestamp with ``Z`` of whole Unix seconds."""
    return pd.Timestamp(seconds, unit="s", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def no_log(line):
    """A log that drops its lines."""


class FakeApi:
    """The public endpoints, answering from canned records.

    ``live_markets`` and ``historical_markets`` map a series to its market records, ``trades`` maps
    a ticker to its trade records. The historical trade endpoint serves the trades created before
    ``CUTOFF_TS`` and the live one those created at or after it (``overlap``: both serve a trade
    created exactly at the cutoff; with ``live_id_suffix`` the live endpoint gives that trade
    another ``trade_id``). ``min_ts`` and ``max_ts`` are inclusive, or exclusive with
    ``strict_bounds`` (the documentation does not say which), and the newest trades come first. A
    page holds at most ``page_size`` records whatever ``limit`` says. ``cutoffs``
    lists the answers of the cutoff endpoint, the last one repeated. After ``fail_after`` requests
    every answer is HTTP 500. ``urls`` records every request.
    """

    def __init__(self, live_markets, historical_markets, trades, *, cutoffs=(CUTOFFS,),
                 page_size=3, overlap=False, strict_bounds=False, fail_after=None,
                 live_id_suffix=""):
        self.live_markets, self.historical_markets = live_markets, historical_markets
        self.trades, self.cutoffs, self.page_size = trades, list(cutoffs), page_size
        self.overlap, self.strict_bounds, self.fail_after = overlap, strict_bounds, fail_after
        self.live_id_suffix = live_id_suffix
        self.urls = []

    @staticmethod
    def split(url):
        """``(path, query)`` of a URL, the path without the base and the query as a dict."""
        parts = urllib.parse.urlsplit(url)
        query = {k: v[0] for k, v in urllib.parse.parse_qs(parts.query).items()}
        return parts.path.removeprefix("/trade-api/v2"), query

    def __call__(self, url):
        self.urls.append(url)
        if self.fail_after is not None and len(self.urls) > self.fail_after:
            return 500, {}, b'{"message": "internal error"}'
        path, query = self.split(url)
        if path == "/historical/cutoff":
            answer = self.cutoffs.pop(0) if len(self.cutoffs) > 1 else self.cutoffs[0]
            return 200, {}, json.dumps(answer).encode()
        if path in ("/markets", "/historical/markets"):
            source = self.live_markets if path == "/markets" else self.historical_markets
            key, records = "markets", source.get(query["series_ticker"], [])
        elif path in TRADE_PATHS:
            key, records = "trades", self._trades(path, query)
        else:
            return 404, {}, b'{"message": "not found"}'
        start, size = int(query.get("cursor", 0)), min(int(query["limit"]), self.page_size)
        more = start + size < len(records)
        body = {key: records[start:start + size], "cursor": str(start + size) if more else ""}
        return 200, {}, json.dumps(body).encode()

    def _trades(self, path, query):
        low, high = float(query.get("min_ts", "-inf")), float(query.get("max_ts", "inf"))
        found = []
        for record in self.trades.get(query["ticker"], []):
            ts = pd.Timestamp(record["created_time"]).timestamp()
            held = ts < CUTOFF_TS if path == "/historical/trades" else ts >= CUTOFF_TS
            inside = low < ts < high if self.strict_bounds else low <= ts <= high
            if inside and (held or (self.overlap and ts == CUTOFF_TS)):
                if path == "/markets/trades" and ts == CUTOFF_TS and self.live_id_suffix:
                    record = {**record, "trade_id": record["trade_id"] + self.live_id_suffix}
                found.append((ts, record))
        found.sort(key=lambda pair: (pair[0], pair[1]["trade_id"]), reverse=True)
        return [record for _, record in found]

    def requests(self, *paths):
        """``(path, query)`` of the first page of every request to one of the paths."""
        split = [self.split(url) for url in self.urls]
        return [(p, q) for p, q in split if p in paths and "cursor" not in q]

    def trade_requests(self):
        """The set of ``(path, ticker, min_ts, max_ts)`` of the trade requests."""
        return {(p, q["ticker"], q.get("min_ts"), q.get("max_ts"))
                for p, q in self.requests(*TRADE_PATHS)}

    def tickers_requested(self):
        """The tickers whose trades were requested."""
        return {ticker for _, ticker, _, _ in self.trade_requests()}


class RunStoppedError(Exception):
    """What a killed process or a Ctrl-C would raise inside the client."""


def stop_after(api, requests):
    """A transport that answers as ``api`` does for ``requests`` requests and then raises."""
    served = []

    def transport(url):
        if len(served) >= requests:
            raise RunStoppedError(url)
        served.append(url)
        return api(url)

    return transport


def make_client(transport, **kwargs):
    return KalshiClient(BASE, transport=transport, sleep=lambda seconds: None, **kwargs)


@pytest.fixture
def db():
    with closing(sqlite3.connect(":memory:")) as connection:
        yield connection


def canned_trades(ticker, times):
    """Trade records at the given Unix times and the rows they must become: ``(ts, price, size,
    side, trade_id)``. Sides alternate and prices cycle through 0.40 to 0.59."""
    records, rows = [], []
    for i, ts in enumerate(times):
        side = 1 if i % 2 == 0 else -1
        price, count, trade_id = f"{0.40 + 0.01 * (i % 20):.4f}", f"{1 + i % 5}.00", f"{ticker}-{i}"
        records.append(trade_record(ts=zulu(ts), price=price, count=count, trade_id=trade_id,
                                    side="yes" if side == 1 else "no", ticker=ticker))
        rows.append((ts, float(price), float(count), side, trade_id))
    return records, rows


def never_kept(ticker, release):
    """A block trade, a trade at the release and one after it: no rule keeps them. The last is
    beyond the range a fetch asks for, so the endpoints do not serve it."""
    return [trade_record(ts=zulu(release - 3000.0), block=True, trade_id=f"{ticker}-block",
                         ticker=ticker),
            trade_record(ts=zulu(release), trade_id=f"{ticker}-at-release", ticker=ticker),
            trade_record(ts=zulu(release + 5.0), trade_id=f"{ticker}-after", ticker=ticker)]


def canned_market(ticker, event, close, result="yes", **extra):
    """A market record that opens on 2026-08-01 and closes, and is expected to expire, at
    ``close``; ``extra`` replaces fields (``open_time``, ``expected``, ...)."""
    fields = {"open_time": "2026-08-01T12:00:00Z", "expected": close, **extra}
    return market_record(ticker, event, result=result, close_time=close, **fields)


def canned_world():
    """The records of a series and the frames a run must store from them.

    September's CPI releases before the trade cutoff (its trades are on the historical endpoint),
    October's after it (its trades are on both, one exactly at the cutoff). The markets: S3 has 60
    trades to keep, plus a repeated record, one with price 1, a block trade and one at the release;
    S4 has 55 and settled no; S9 has 10 (too few); O3 has 61 to keep, a block trade and one at the
    release, and O4 has exactly 50 and has not settled. July's market is outside the window,
    November's event is not in the calendar and two records cannot be read.
    """
    sep_times = [R_SEP - 10800.0 + 120.0 * i for i in range(60)]
    oct_before = [CUTOFF_TS - 43200.0 + 600.0 * i for i in range(30)]
    oct_after = [CUTOFF_TS + 600.0 * j for j in range(31)]  # the first is exactly at the cutoff
    sep3, rows_s3 = canned_trades(S3, sep_times)
    sep4, rows_s4 = canned_trades(S4, sep_times[:55])
    oct3, rows_o3 = canned_trades(O3, oct_before + oct_after)
    oct4, rows_o4 = canned_trades(O4, oct_before[:25] + oct_after[:25])
    bad_price = trade_record(ts=zulu(R_SEP - 4000.0), price="1.0000", trade_id="bad", ticker=S3)
    trades = {
        S3: [*sep3, dict(sep3[3]), bad_price, *never_kept(S3, R_SEP)],
        S4: sep4,
        S9: canned_trades(S9, sep_times[:10])[0],
        O3: [*oct3, *never_kept(O3, RELEASE_TS)],
        O4: oct4,
        J3: canned_trades(J3, sep_times)[0],
        N3: canned_trades(N3, sep_times)[0],
    }
    historical = [canned_market(S3, SEP, zulu(R_SEP)), canned_market(S4, SEP, zulu(R_SEP), "no"),
                  canned_market(S9, SEP, zulu(R_SEP), "no"),
                  canned_market(J3, JUL, "2026-07-15T12:30:00Z", open_time="2026-06-01T12:00:00Z")]
    live = [canned_market(O3, OCT, RELEASE), canned_market(O4, OCT, RELEASE, ""),
            canned_market(N3, NOV, "2026-11-12T13:30:00Z"),
            {"ticker": "KXCPI-NO-EVENT", "status": "open"}, {"event_ticker": SEP}]

    def tape_frame(ticker, event, rows):
        rows = sorted(rows, key=lambda r: (r[0], r[4]))
        return pd.DataFrame({
            "event": event, "market": ticker, "ts": [r[0] for r in rows],
            "seq": np.arange(len(rows), dtype=np.int64), "price": [r[1] for r in rows],
            "size": [r[2] for r in rows], "side": np.array([r[3] for r in rows], dtype=np.int8)})

    tape = pd.concat([tape_frame(O3, OCT, rows_o3), tape_frame(O4, OCT, rows_o4),
                      tape_frame(S3, SEP, rows_s3), tape_frame(S4, SEP, rows_s4)],
                     ignore_index=True)
    markets = pd.DataFrame({
        "market": [S3, S4, O3, O4], "event": [SEP, SEP, OCT, OCT],
        "outcome": [1.0, 0.0, 1.0, np.nan], "open_ts": unix(2026, 8, 1, 12, 0, 0),
        "close_ts": [R_SEP, R_SEP, RELEASE_TS, RELEASE_TS]})
    events = pd.DataFrame({"event": [SEP, OCT], "release_ts": [R_SEP, RELEASE_TS],
                           "series": "KXCPI"})
    return types.SimpleNamespace(live=live, historical=historical, trades=trades, tape=tape,
                                 markets=markets, events=events)


def make_api(world, **kwargs):
    return FakeApi({"KXCPI": world.live}, {"KXCPI": world.historical}, world.trades, **kwargs)


def run_ingest(db, tmp_path, transport, rows=CALENDAR_ROWS, log=no_log, **kwargs):
    return ingest_series(db, make_client(transport), "KXCPI", calendar_frame(tmp_path, rows),
                         FETCH_WINDOW, log=log, **kwargs)


def assert_stored(db, world):
    """The database reads back as the frames of ``canned_world``, to the last bit."""
    stored = panel.read_dataset(db)
    for got, want in zip(stored, (world.tape, world.markets, world.events), strict=True):
        pd.testing.assert_frame_equal(got, want, check_exact=True)


def stored_rows(db):
    return db.execute(f"SELECT {STORED_COLUMNS} FROM trades ORDER BY market, ts, seq").fetchall()


# ---------------------------------------------------------------------------- trade_queries
def test_a_release_before_the_cutoff_needs_only_the_historical_endpoint():
    assert trade_queries(100.0, 200.0) == [("historical", None, 100)]
    assert trade_queries(200.0, 200.0) == [("historical", None, 200)]  # trades are before it


def test_a_release_after_the_cutoff_needs_both_endpoints_split_at_the_cutoff():
    assert trade_queries(201.0, 200.0) == [("historical", None, 200), ("live", 199, 201)]


def test_query_ranges_are_rounded_outwards_to_whole_seconds():
    got = trade_queries(1000.5, 500.25)
    assert got == [("historical", None, 501), ("live", 499, 1001)]
    assert all(isinstance(q.max_ts, int) for q in got)
    assert trade_queries(1000.5, 2000.0) == [("historical", None, 1001)]
    assert trade_queries(1000.0, 500.0) == [("historical", None, 500), ("live", 499, 1000)]


# ------------------------------------------------------------------------ list_series_markets
def listing(live, historical, **kwargs):
    client = make_client(FakeApi({"S": live}, {"S": historical}, {}))
    return list_series_markets(client, "S", **kwargs)


def test_the_listing_keeps_one_record_per_ticker_sorted_by_ticker():
    live = [canned_market("B-1", "E", RELEASE), canned_market("A-1", "E", RELEASE)]
    historical = [canned_market("C-1", "E", RELEASE), canned_market("A-1", "E", RELEASE, "no")]
    got = listing(live, historical)
    assert [(r["ticker"], r["result"]) for r in got] == [("A-1", "yes"), ("B-1", "yes"),
                                                         ("C-1", "yes")]


def test_an_archived_record_replaces_a_live_one_only_if_it_is_settled_and_the_live_one_is_not():
    live = [canned_market("A-1", "E", RELEASE, ""), canned_market("B-1", "E", RELEASE, ""),
            canned_market("C-1", "E", RELEASE, "yes"), canned_market("D-1", "E", RELEASE, "")]
    historical = [canned_market("A-1", "E", RELEASE, "no"),
                  canned_market("B-1", "E", RELEASE, "scalar"),
                  canned_market("C-1", "E", RELEASE, "no"), canned_market("D-1", "E", RELEASE, "")]
    assert {r["ticker"]: r["result"] for r in listing(live, historical)} == {
        "A-1": "no", "B-1": "", "C-1": "yes", "D-1": ""}


def test_a_ticker_repeated_inside_one_listing_is_settled_if_any_record_is():
    live = [canned_market("A-1", "E", RELEASE, ""), canned_market("A-1", "E", RELEASE, "yes"),
            canned_market("A-1", "E", RELEASE, "")]
    assert [r["result"] for r in listing(live, [])] == ["yes"]


def test_records_without_a_ticker_are_kept_after_the_others_in_the_order_read():
    nameless, blank = {"event_ticker": "E", "note": 1}, {"ticker": "", "event_ticker": "E"}
    got = listing([nameless, canned_market("B-1", "E", RELEASE)],
                  [blank, canned_market("A-1", "E", RELEASE)])
    assert [r.get("ticker") for r in got] == ["A-1", "B-1", None, ""]


def test_an_empty_listing_is_an_empty_list():
    assert listing([], []) == []


def test_a_draft_window_keeps_events_up_to_seven_days_outside_it():
    window = Window.from_dates("2026-09-01", "2026-09-30")  # [2026-09-01, 2026-10-01)
    assert ingest.MARKET_MARGIN_S == 7 * 86400.0
    closes = {"low-edge": "2026-08-25T00:00:00Z", "below": "2026-08-24T23:59:59Z",
              "inside": "2030-01-01T00:00:00Z", "above": "2026-10-07T23:59:59Z",
              "high-edge": "2026-10-08T00:00:00Z", "far": "2026-12-09T00:00:00Z"}
    records = [canned_market(name, "E", close, expected=None) for name, close in closes.items()]
    records[2]["expected_expiration_time"] = "2026-09-20T00:00:00Z"  # it wins over the close time
    unreadable = {"ticker": "unreadable", "event_ticker": "E", "open_time": "garbage"}
    kept = [r["ticker"] for r in listing(records + [unreadable], [], window=window)]
    # the edge a week before the window is inside, the edge a week after it is outside
    assert kept == ["above", "inside", "low-edge", "unreadable"]
    assert len(listing(records + [unreadable], [])) == 7  # no window, no filter


# ------------------------------------------------------------------------- classify_markets
def test_markets_are_sorted_into_wanted_loaded_and_out_of_scope(tmp_path):
    rows = [*CALENDAR_ROWS, ("KXCPI-26AUG", zulu(FETCH_WINDOW.start_ts), SOURCE_URL),
            ("KXCPI-26DEC", zulu(FETCH_WINDOW.end_ts), SOURCE_URL)]
    records = [canned_market(S3, SEP, RELEASE), canned_market(O3, OCT, RELEASE),
               canned_market(J3, JUL, RELEASE), canned_market(N3, NOV, RELEASE),
               canned_market("U-1", "KXCPI-26AUG", RELEASE),
               canned_market("U-2", "KXCPI-26DEC", RELEASE),
               {"ticker": "KXCPI-X"}, {"event_ticker": SEP}, canned_market(S4, SEP, RELEASE)]
    got = classify_markets(records, calendar_frame(tmp_path, rows), FETCH_WINDOW, loaded={S4})
    assert [(c.state, c.release_ts) for c in got] == [
        (ingest.WANTED, R_SEP), (ingest.WANTED, RELEASE_TS),
        (ingest.OUT_OF_SCOPE, None),  # July: the release is before the window
        (ingest.OUT_OF_SCOPE, None),  # November: the event is not in the calendar
        (ingest.WANTED, FETCH_WINDOW.start_ts),  # a release at the start of the window is inside
        (ingest.OUT_OF_SCOPE, None),  # a release at the end of the window is outside
        (ingest.OUT_OF_SCOPE, None), (ingest.OUT_OF_SCOPE, None),  # unreadable records
        (ingest.ALREADY_LOADED, None)]
    assert [c.market for c in got] == records


def test_a_stored_market_is_not_fetched_even_if_its_record_cannot_be_read_any_more(tmp_path):
    got = classify_markets([{"ticker": S3}], calendar_frame(tmp_path), FETCH_WINDOW, loaded={S3})
    assert [c.state for c in got] == [ingest.ALREADY_LOADED]


# ----------------------------------------------------------------------------- fetch_series
def fetch(api, tmp_path, **kwargs):
    return fetch_series(make_client(api), "KXCPI", calendar_frame(tmp_path, CALENDAR_ROWS),
                        FETCH_WINDOW, cutoff=CUTOFF, **kwargs)


def test_fetch_series_requests_the_trades_of_a_market_only_when_the_consumer_asks(tmp_path):
    api = make_api(canned_world())
    seen = []
    for item in fetch(api, tmp_path):
        seen.append((item.market.get("ticker"), item.state, item.trades is None))
        if item.state == ingest.WANTED:
            break
    assert seen == [(J3, ingest.OUT_OF_SCOPE, True), (N3, ingest.OUT_OF_SCOPE, True),
                    (O3, ingest.WANTED, False)]
    assert api.tickers_requested() == {O3}  # nothing was read ahead


def test_fetch_series_reads_the_pages_of_each_market_from_the_endpoints_that_hold_them(tmp_path):
    api = make_api(canned_world())
    got = {item.market.get("ticker"): item for item in fetch(api, tmp_path, loaded={S4})}
    assert len(got) == 9 and got[S4].state == ingest.ALREADY_LOADED and got[S4].trades is None
    # S3: 60 + repeat + price 1 + block + release; the trade after the release is not asked for
    assert [len(got[t].trades) for t in (O3, O4, S3, S9)] == [63, 50, 64, 10]
    ids = {t["trade_id"] for t in got[O3].trades}
    # the last trade before the cutoff and the one exactly at it, then the two the filters remove
    assert {f"{O3}-29", f"{O3}-30", f"{O3}-block", f"{O3}-at-release"} <= ids
    assert f"{O3}-after" not in ids
    assert all(got[t].trades is None for t in (J3, N3, "KXCPI-NO-EVENT", None))
    assert api.tickers_requested() == {O3, O4, S3, S9}


# ----------------------------------------------------------------------------- ingest_series
def test_a_run_stores_what_the_filters_keep_and_reads_back_as_the_expected_frames(tmp_path, db):
    world, lines = canned_world(), []
    api = make_api(world)
    summary = run_ingest(db, tmp_path, api, log=lines.append)
    assert_stored(db, world)
    assert panel.get_source(db) == "kalshi-public-api"
    assert (summary.listed, summary.out_of_scope, summary.already_loaded, summary.fetched,
            summary.written) == (9, 4, 0, 5, 4)
    assert summary.cutoff_before == summary.cutoff_after == CUTOFF and not summary.cutoff_moved
    r = summary.report
    assert r.markets_in == 9
    assert r.markets_rejected == {"missing_event_ticker": 1, "missing_ticker": 1}
    assert (r.markets_no_release, r.markets_outside_window, r.markets_too_few_trades,
            r.markets_kept, r.markets_outcome_missing) == (1, 1, 1, 4, 1)
    assert r.trades_in == 242 and r.trades_rejected == {"price_outside_0_1": 1}
    assert (r.trades_duplicate, r.trades_block, r.trades_at_or_after_release,
            r.trades_in_dropped_markets, r.trades_kept) == (1, 2, 2, 10, 226)
    assert_report_adds_up(r)
    sep_release, cutoff = str(int(R_SEP)), str(int(CUTOFF_TS))
    live_from, oct_release = str(int(CUTOFF_TS) - 1), str(int(RELEASE_TS))
    assert api.trade_requests() == {
        ("/historical/trades", ticker, None, sep_release) for ticker in (S3, S4, S9)
    } | {
        ("/historical/trades", ticker, None, cutoff) for ticker in (O3, O4)
    } | {
        ("/markets/trades", ticker, live_from, oct_release) for ticker in (O3, O4)
    }
    assert len(api.requests("/historical/cutoff")) == 2  # before and after the run
    assert lines == [
        "KXCPI: trades created before 2026-10-20T00:00:00Z are on the historical endpoint; "
        "0 markets are already in the database",
        f"{O3}: 63 trades read, 61 left after the filters, stored",
        f"{O4}: 50 trades read, 50 left after the filters, stored",
        f"{S3}: 64 trades read, 60 left after the filters, stored",
        f"{S4}: 55 trades read, 55 left after the filters, stored",
        f"{S9}: 10 trades read, 10 left after the filters, dropped (fewer than 50)"]


def test_a_trade_on_both_endpoints_at_the_cutoff_is_stored_once(tmp_path, db):
    world = canned_world()
    summary = run_ingest(db, tmp_path, make_api(world, overlap=True))
    assert_stored(db, world)
    # O3 and O4 each have a trade created exactly at the cutoff: read twice, counted once
    assert summary.report.trades_in == 244 and summary.report.trades_duplicate == 3
    assert summary.report.trades_id_on_both_endpoints == 2
    assert summary.report.trades_lookalike_groups == 0
    stored = dict(db.execute("SELECT trade_id, endpoint FROM trades WHERE market IN (?, ?)",
                             (O3, O4)).fetchall())
    at_cutoff = {f"{O3}-30", f"{O4}-25"}  # the trades created exactly at the cutoff
    assert {k for k, v in stored.items() if v == "both"} == at_cutoff
    assert {v for k, v in stored.items() if k not in at_cutoff} == {"historical", "live"}


def test_where_each_stored_trade_was_read_from(tmp_path, db):
    run_ingest(db, tmp_path, make_api(canned_world()))
    got = dict(((m, e), n) for m, e, n in db.execute(
        "SELECT market, endpoint, COUNT(*) FROM trades GROUP BY market, endpoint"))
    # September is before the cutoff; October has 30 (25) trades before it and 31 (25) at or after
    assert got == {(S3, "historical"): 60, (S4, "historical"): 55, (O3, "historical"): 30,
                   (O3, "live"): 31, (O4, "historical"): 25, (O4, "live"): 25}


def test_a_trade_with_another_id_on_each_endpoint_is_kept_twice_and_counted(tmp_path, db):
    world = canned_world()
    summary = run_ingest(db, tmp_path, make_api(world, overlap=True, live_id_suffix="-live"))
    r = summary.report
    assert r.trades_id_on_both_endpoints == 0 and r.trades_duplicate == 1  # S3's own repeat
    assert r.trades_lookalike_groups == 2 and r.trades_kept == 228  # one extra in O3 and in O4
    twins = db.execute("SELECT market, trade_id, endpoint FROM trades WHERE ts = ? "
                       "ORDER BY market, trade_id", (CUTOFF_TS,)).fetchall()
    assert twins == [(O3, f"{O3}-30", "historical"), (O3, f"{O3}-30-live", "live"),
                     (O4, f"{O4}-25", "historical"), (O4, f"{O4}-25-live", "live")]


def test_the_price_grid_of_each_market_is_stored_and_not_returned(tmp_path, db):
    world = canned_world()
    world.live[0] = {**world.live[0], **GRID}  # O3 has a grid, the others do not
    run_ingest(db, tmp_path, make_api(world))
    assert_stored(db, world)  # the frames of the contract do not change
    rows = dict(((m, (s, r)) for m, s, r in db.execute(
        "SELECT market, price_level_structure, price_ranges FROM markets")))
    assert rows[O3][0] == "tapered_deci_cent" and json.loads(rows[O3][1]) == GRID["price_ranges"]
    assert rows[O4] == rows[S3] == (None, None)


@pytest.mark.parametrize("strict_bounds", [False, True])
@pytest.mark.parametrize("overlap", [False, True])
def test_the_trades_around_the_cutoff_are_read_whichever_way_the_bounds_are_treated(
        tmp_path, db, overlap, strict_bounds):
    """The documentation does not say whether ``min_ts`` and ``max_ts`` are included; a trade
    created exactly at the cutoff, and the trade at the release, must give the same dataset
    under both readings."""
    world = canned_world()
    run_ingest(db, tmp_path, make_api(world, overlap=overlap, strict_bounds=strict_bounds))
    assert_stored(db, world)


def test_a_second_run_changes_nothing_and_does_not_fetch_stored_markets_again(tmp_path, db):
    world = canned_world()
    run_ingest(db, tmp_path, make_api(world))
    before = stored_rows(db)
    api = make_api(world)
    second = run_ingest(db, tmp_path, api)
    assert_stored(db, world)
    assert stored_rows(db) == before
    # S9 was dropped for having too few trades, so it was not stored: it is read and dropped again
    assert (second.already_loaded, second.fetched, second.written) == (4, 1, 0)
    assert api.tickers_requested() == {S9}
    assert second.report.markets_in == 5 and second.report.markets_too_few_trades == 1
    assert second.report.trades_in == 10 and second.report.markets_kept == 0


def test_a_run_that_stops_at_any_request_resumes_to_the_same_database(tmp_path):
    world = canned_world()
    with closing(sqlite3.connect(":memory:")) as clean:
        clean_api = make_api(world)
        run_ingest(clean, tmp_path, clean_api)
        assert_stored(clean, world)
        total = len(clean_api.urls)
    assert total > 80
    sizes = world.tape["market"].value_counts()
    for stop in range(1, total, 5):
        with closing(sqlite3.connect(":memory:")) as db:
            with pytest.raises(RunStoppedError):
                run_ingest(db, tmp_path, stop_after(make_api(world), stop))
            half = panel.read_dataset(db)
            stored = set(half[1]["market"])
            # a market is stored whole or not at all
            assert dict(half[0]["market"].value_counts()) == {m: sizes[m] for m in stored if m}
            api = make_api(world)
            resumed = run_ingest(db, tmp_path, api)
            assert resumed.already_loaded == len(stored)
            assert not api.tickers_requested() & stored
            assert_stored(db, world)


def test_a_calendar_with_another_release_time_than_the_database_is_refused_first(tmp_path, db):
    run_ingest(db, tmp_path, make_api(canned_world()))
    moved = [(SEP, "2026-09-11T13:30:00Z", SOURCE_URL), CALENDAR_ROWS[1], CALENDAR_ROWS[2]]
    api = make_api(canned_world())
    with pytest.raises(CalendarChangedError, match=SEP):
        run_ingest(db, tmp_path, api, rows=moved)
    assert api.urls == []  # nothing was requested
    assert_stored(db, canned_world())  # and nothing was changed


def test_the_calendar_may_leave_out_events_that_are_stored(tmp_path, db):
    run_ingest(db, tmp_path, make_api(canned_world()))
    again = run_ingest(db, tmp_path, make_api(canned_world()), rows=[CALENDAR_ROWS[2]])
    assert again.already_loaded == 4
    assert_stored(db, canned_world())


def test_a_moved_trade_cutoff_is_reported_but_a_moved_orders_cutoff_is_not(tmp_path):
    for change, moved in (({"trades_created_ts": "2026-10-21T00:00:00Z"}, True),
                          ({"market_settled_ts": "2026-09-16T00:00:00Z"}, True),
                          ({"orders_updated_ts": "2026-10-31T00:00:00Z"}, False)):
        with closing(sqlite3.connect(":memory:")) as db:
            api = make_api(canned_world(), cutoffs=[CUTOFFS, {**CUTOFFS, **change}])
            summary = run_ingest(db, tmp_path, api)
            assert summary.cutoff_moved is moved
            assert summary.cutoff_before == CUTOFF


def test_a_series_without_markets_stores_nothing(tmp_path, db):
    summary = run_ingest(db, tmp_path, FakeApi({}, {}, {}))
    assert (summary.listed, summary.written, summary.report.markets_in) == (0, 0, 0)
    assert panel.loaded_markets(db) == set()
    with pytest.raises(ValueError, match="no source"):
        panel.get_source(db)  # nothing was written, so no source was recorded


def test_the_minimum_number_of_trades_can_be_changed(tmp_path, db):
    summary = run_ingest(db, tmp_path, make_api(canned_world()), min_trades=10)
    assert summary.written == 5 and summary.report.markets_too_few_trades == 0
    assert panel.read_dataset(db)[0]["market"].value_counts()[S9] == 10
    with closing(sqlite3.connect(":memory:")) as other:
        summary = run_ingest(other, tmp_path, make_api(canned_world()), min_trades=11)
        assert summary.written == 4 and summary.report.markets_too_few_trades == 1


def test_a_failing_endpoint_stops_the_run_with_the_clients_error(tmp_path, db):
    with pytest.raises(KalshiApiError, match="HTTP 500"):
        run_ingest(db, tmp_path, make_api(canned_world(), fail_after=0))
    assert panel.loaded_markets(db) == set()


# ------------------------------------------------------------------------ the command line
@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("fetch_kalshi",
                                                  REPO / "scripts" / "fetch_kalshi.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(script, tmp_path, *extra, transport=None, rows=CALENDAR_ROWS, **client_kwargs):
    """Run the script's ``main`` on the fake endpoints and return its exit status and output.

    The calendar is written to ``tmp_path`` unless ``rows`` is None; ``extra`` arguments come after
    the usual ones and win over them."""
    transport = transport if transport is not None else make_api(canned_world())
    if rows is not None:
        write_calendar(tmp_path, rows)
    arguments = ["--series", "KXCPI", "--start", "2026-09-01", "--end", "2026-10-31",
                 "--db", str(tmp_path / "k.sqlite"),
                 "--calendar", str(tmp_path / "release_calendar.csv"), *extra]
    lines = []
    status = script.main(arguments, client=make_client(transport, **client_kwargs),
                         out=lines.append)
    return status, "\n".join(lines)


def test_the_defaults_are_the_series_dates_and_minimum_of_the_protocol(script):
    args = script.build_parser().parse_args([])
    assert args.series == list(ingest.PROTOCOL_SERIES) and args.min_trades == ingest.MIN_TRADES
    assert (args.start, args.end) == (ingest.PROTOCOL_START, ingest.PROTOCOL_END)
    assert args.db == pathlib.Path("data/kalshi.sqlite")
    assert args.calendar == pathlib.Path("data/release_calendar.csv")
    assert args.calendar_draft is None and args.dry_run is False


def test_a_run_prints_the_agreement_notice_and_the_counts_and_stores_the_data(script, tmp_path):
    status, text = run_cli(script, tmp_path)
    assert status == 0
    assert "Developer Agreement, v1.1" in text and "Kalshi-Developer-Agreement.pdf" in text
    assert "Section 3.1 prohibits" in text and "sends no credentials" in text
    assert ("KXCPI: 9 market records: 4 out of scope, 0 already in the database, 5 fetched, "
            "4 stored") in text
    assert "markets: 9 read" in text and "trades: 242 read" in text and "kept 226" in text
    assert "both trade endpoints: 0 trade_id returned by both" in text
    assert "all series (1), read in this run:" in text
    assert text.splitlines()[-1] == f"done; the database is {tmp_path / 'k.sqlite'}"
    with closing(sqlite3.connect(tmp_path / "k.sqlite")) as conn:
        assert_stored(conn, canned_world())
        assert panel.get_source(conn) == "kalshi-public-api"


def test_the_minimum_number_of_trades_and_the_database_folder_are_taken_from_the_arguments(
        script, tmp_path):
    database = tmp_path / "nested" / "folder" / "k.sqlite"
    status, text = run_cli(script, tmp_path, "--min-trades", "10", "--db", str(database))
    assert status == 0 and "5 fetched, 5 stored" in text
    with closing(sqlite3.connect(database)) as conn:
        assert len(panel.loaded_markets(conn)) == 5


def test_several_series_are_fetched_in_turn(script, tmp_path):
    status, text = run_cli(script, tmp_path, "--series", "KXCPI", "KXU3")
    assert status == 0 and "KXCPI: 9 market records" in text and "KXU3: 0 market records" in text
    assert "KXU3: no market was stored" in text and "KXCPI: no market was stored" not in text
    total = text.split("all series (2), read in this run:")[1]
    assert "markets: 9 read" in total and "kept 226" in total


def test_the_counts_of_both_endpoints_are_printed_per_series_and_in_total(script, tmp_path):
    api = make_api(canned_world(), overlap=True, live_id_suffix="-live")
    status, text = run_cli(script, tmp_path, transport=api)
    lines = [line for line in text.splitlines() if "both trade endpoints" in line]
    assert len(lines) == 2  # the series, then the total
    assert all("0 trade_id returned by both" in line and "; 2 groups of kept trades" in line
               for line in lines)


def test_running_the_same_command_again_resumes(script, tmp_path):
    run_cli(script, tmp_path)
    api = make_api(canned_world())
    status, text = run_cli(script, tmp_path, transport=api)
    assert status == 0 and "4 already in the database, 1 fetched, 0 stored" in text
    assert "reported as empty" not in text  # four markets are stored: the series is not empty
    assert api.tickers_requested() == {S9}


def test_a_dry_run_lists_what_would_be_fetched_and_requests_no_trade(script, tmp_path):
    api = make_api(canned_world())
    status, text = run_cli(script, tmp_path, "--dry-run", transport=api)
    assert status == 0 and api.trade_requests() == set()
    assert ("KXCPI: 9 market records. Markets to fetch: 5, in 2 events (3 on the historical "
            "endpoint only, 2 on both). Already in") in text
    assert "Unreadable, outside the calendar or outside the window: 4." in text
    assert "No trade was requested and nothing was written" in text
    assert not (tmp_path / "k.sqlite").exists()


def test_a_dry_run_skips_the_markets_already_stored_and_leaves_the_database_alone(script, tmp_path):
    run_cli(script, tmp_path)
    before = (tmp_path / "k.sqlite").read_bytes()
    api = make_api(canned_world())
    status, text = run_cli(script, tmp_path, "--dry-run", transport=api)
    assert status == 0 and "Markets to fetch: 1, in 1 events" in text
    assert "Already in" in text and f"{tmp_path / 'k.sqlite'}: 4." in text
    assert api.trade_requests() == set() and (tmp_path / "k.sqlite").read_bytes() == before


def test_a_dry_run_on_a_file_without_the_tables_skips_nothing(script, tmp_path):
    with closing(sqlite3.connect(tmp_path / "k.sqlite")) as conn:
        conn.execute("CREATE TABLE unrelated (x INTEGER)")
        conn.commit()
    status, text = run_cli(script, tmp_path, "--dry-run")
    assert status == 0 and "Markets to fetch: 5" in text and f"{tmp_path / 'k.sqlite'}: 0." in text


def test_a_draft_calendar_is_written_from_the_markets_near_the_window(script, tmp_path):
    path = tmp_path / "release_calendar_draft.csv"
    api = make_api(canned_world())
    status, text = run_cli(script, tmp_path, "--calendar-draft", str(path), rows=None,
                           transport=api)
    assert status == 0 and api.trade_requests() == set()
    assert path.read_text().splitlines() == [
        "event_ticker,release_time_utc,source_url", f"{SEP},2026-09-11T12:30:00Z,",
        f"{OCT},2026-10-24T12:30:00Z,"]  # July and November are more than a week outside
    assert "wrote 2 events" in text and "2 market records could not be read" in text
    assert "The draft is not a calendar" in text
    with pytest.raises(ValueError, match="draft"):
        load_release_calendar(path)  # the analysis refuses it


def test_a_draft_dry_run_writes_nothing(script, tmp_path):
    path = tmp_path / "calendar_draft.csv"
    status, text = run_cli(script, tmp_path, "--calendar-draft", str(path), "--dry-run", rows=None)
    assert status == 0 and not path.exists() and "dry run" in text


@pytest.mark.parametrize("arguments", [
    ["--calendar-draft", "calendar.csv"],
    ["--start", "2026-10-31", "--end", "2026-09-01"],
    ["--start", "2026-13-01"],
    ["--min-trades", "0"],
])
def test_bad_arguments_stop_before_any_request(script, capsys, arguments):
    api = make_api(canned_world())
    with pytest.raises(SystemExit) as info:
        script.main(arguments, client=make_client(api), out=no_log)
    assert info.value.code == 2 and api.urls == [] and capsys.readouterr().err


def test_an_api_failure_stops_with_status_one_and_what_was_stored_is_kept(script, tmp_path):
    with closing(sqlite3.connect(":memory:")) as clean:
        clean_api = make_api(canned_world())
        run_ingest(clean, tmp_path, clean_api)
    failing = make_api(canned_world(), fail_after=len(clean_api.urls) // 2)
    status, text = run_cli(script, tmp_path, transport=failing, max_retries=0)
    assert status == 1 and "HTTP 500" in text
    assert "What was stored is kept; run the same command again to resume." in text
    with closing(sqlite3.connect(tmp_path / "k.sqlite")) as conn:
        assert 0 < len(panel.loaded_markets(conn)) < 4
    status, text = run_cli(script, tmp_path)
    assert status == 0
    with closing(sqlite3.connect(tmp_path / "k.sqlite")) as conn:
        assert_stored(conn, canned_world())


def test_a_missing_or_draft_calendar_is_refused_before_any_request(script, tmp_path):
    api = make_api(canned_world())
    status, text = run_cli(script, tmp_path, transport=api, rows=None)
    assert status == 2 and "calendar:" in text and api.urls == []
    draft = write_calendar(tmp_path, CALENDAR_ROWS, name="release_calendar_draft.csv")
    status, text = run_cli(script, tmp_path, "--calendar", str(draft), transport=api, rows=None)
    assert status == 2 and "is a draft" in text and api.urls == []


def test_a_calendar_with_a_wrong_row_is_refused_with_the_line(script, tmp_path):
    rows = [(SEP, "2026-09-11 12:30", SOURCE_URL), CALENDAR_ROWS[1]]
    api = make_api(canned_world())
    status, text = run_cli(script, tmp_path, transport=api, rows=rows)
    assert status == 2 and "line 2: release_time_utc" in text and api.urls == []


def test_a_changed_calendar_stops_with_status_two(script, tmp_path):
    run_cli(script, tmp_path)
    moved = [(SEP, "2026-09-11T13:30:00Z", SOURCE_URL), CALENDAR_ROWS[1], CALENDAR_ROWS[2]]
    api = make_api(canned_world())
    status, text = run_cli(script, tmp_path, transport=api, rows=moved)
    assert status == 2 and "stopped" in text and SEP in text and api.urls == []


def test_a_database_of_the_older_schema_stops_with_status_two_before_any_request(script,
                                                                               tmp_path):
    with closing(sqlite3.connect(tmp_path / "k.sqlite")) as conn:
        conn.execute("CREATE TABLE markets (market TEXT PRIMARY KEY, event TEXT, outcome INTEGER,"
                     " open_ts REAL, close_ts REAL)")
        conn.commit()
    api = make_api(canned_world())
    status, text = run_cli(script, tmp_path, transport=api)
    assert status == 2 and "earlier version of the schema" in text and api.urls == []


def test_an_empty_series_is_reported_as_empty_and_the_run_still_ends_well(script, tmp_path):
    status, text = run_cli(script, tmp_path, transport=FakeApi({}, {}, {}))
    assert status == 0 and "the series is reported as empty" in text
    assert text.splitlines()[-1].startswith("done;")


def test_a_moved_cutoff_is_a_warning(script, tmp_path):
    moved = {**CUTOFFS, "trades_created_ts": "2026-10-21T00:00:00Z"}
    status, text = run_cli(script, tmp_path,
                           transport=make_api(canned_world(), cutoffs=[CUTOFFS, moved]))
    assert status == 0 and "warning: the historical cutoff moved" in text


# ------------------------------------------------------------------- the record of the fetches
def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c",
                    "user.email=test@example.com", "-c", "commit.gpgsign=false", *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    """A git repository with one commit and a clean tree."""
    path = tmp_path / "repo"
    path.mkdir()
    (path / "a.txt").write_text("a\n")
    git(path, "init", "-q")
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "first")
    return path


def head(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def test_git_state_reads_the_commit_and_whether_the_tree_is_dirty(script, repo, tmp_path):
    assert script.git_state(repo) == (head(repo), "false")
    (repo / "new.txt").write_text("untracked\n")
    assert script.git_state(repo) == (head(repo), "true")
    (repo / "new.txt").unlink()
    (repo / "a.txt").write_text("changed\n")
    assert script.git_state(repo) == (head(repo), "true")
    plain = tmp_path / "plain"
    plain.mkdir()
    assert script.git_state(plain) == ("unknown", "unknown")


def test_git_state_is_unknown_when_git_cannot_run(script, repo, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(script.subprocess, "run", missing)
    assert script.git_state(repo) == ("unknown", "unknown")


def test_utc_now_is_iso_utc_to_the_second(script):
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", script.utc_now())


def test_each_fetch_is_recorded_and_a_second_one_does_not_replace_the_first(script, tmp_path,
                                                                            repo, monkeypatch):
    times = iter(["2026-11-13T10:00:00Z", "2026-11-13T10:05:00Z", "2026-11-14T09:00:00Z",
                  "2026-11-14T09:01:00Z"])
    monkeypatch.setattr(script, "utc_now", lambda: next(times))
    first_commit = head(repo)
    lines = []
    write_calendar(tmp_path, CALENDAR_ROWS)
    arguments = ["--series", "KXCPI", "--start", "2026-09-01", "--end", "2026-10-31",
                 "--db", str(tmp_path / "k.sqlite"),
                 "--calendar", str(tmp_path / "release_calendar.csv")]
    assert script.main(arguments, client=make_client(make_api(canned_world())),
                       out=lines.append, repo=repo) == 0
    assert f"fetch recorded: commit {first_commit}, uncommitted changes: false" in lines
    (repo / "a.txt").write_text("changed\n")
    git(repo, "commit", "-q", "-am", "second")
    (repo / "b.txt").write_text("dirty\n")
    assert script.main(arguments, client=make_client(make_api(canned_world())),
                       out=lines.append, repo=repo) == 0
    with closing(sqlite3.connect(tmp_path / "k.sqlite")) as conn:
        meta = dict(conn.execute("SELECT key, value FROM meta WHERE key LIKE 'fetch_%' "
                                 "AND key != 'fetch_runs' OR key LIKE 'git_%'"))
        runs = panel.fetch_runs(conn)
    assert meta == {"fetch_started_utc": "2026-11-13T10:00:00Z",
                    "fetch_finished_utc": "2026-11-13T10:05:00Z",
                    "git_commit": first_commit, "git_dirty": "false"}
    assert runs == [
        {"fetch_started_utc": "2026-11-13T10:00:00Z", "fetch_finished_utc": "2026-11-13T10:05:00Z",
         "git_commit": first_commit, "git_dirty": "false"},
        {"fetch_started_utc": "2026-11-14T09:00:00Z", "fetch_finished_utc": "2026-11-14T09:01:00Z",
         "git_commit": head(repo), "git_dirty": "true"}]
    assert first_commit != head(repo)


def test_a_fetch_that_fails_is_recorded_with_the_time_it_stopped(script, tmp_path, repo):
    failing = make_api(canned_world(), fail_after=6)
    status, _ = run_cli(script, tmp_path, transport=failing, max_retries=0)
    assert status == 1
    with closing(sqlite3.connect(tmp_path / "k.sqlite")) as conn:
        runs = panel.fetch_runs(conn)
    assert len(runs) == 1 and runs[0]["fetch_finished_utc"] is not None


def test_a_dry_run_and_a_draft_record_no_fetch(script, tmp_path):
    run_cli(script, tmp_path)
    status, _ = run_cli(script, tmp_path, "--dry-run")
    assert status == 0
    with closing(sqlite3.connect(tmp_path / "k.sqlite")) as conn:
        assert len(panel.fetch_runs(conn)) == 1
