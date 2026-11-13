"""Turning Kalshi API records into the data contract (src/kalshi_mm/ingest.py).

The records are canned and shaped like the documentation read on 2026-11-13 (docs/data.md).
Expected times come from ``calendar.timegm``, which shares no code with the parser under test.
Every filter of ``build_dataset`` has a case that sits exactly on its boundary.
"""

import calendar
import dataclasses
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from kalshi_mm import ingest
from kalshi_mm.ingest import (
    BuildReport,
    RejectedRecordError,
    Window,
    build_dataset,
    load_release_calendar,
    parse_market,
    parse_trade,
    write_calendar_draft,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
RELEASE = "2026-10-24T12:30:00Z"
RELEASE_TS = float(calendar.timegm((2026, 10, 24, 12, 30, 0)))  # 1792845000.0
EVENT = "KXCPI-26OCT"
MARKET = "KXCPI-26OCT-T0.3"
WINDOW = {"window_start": RELEASE_TS - 86400.0, "window_end": RELEASE_TS + 86400.0}
TAPE_COLUMNS = ["event", "market", "ts", "seq", "price", "size", "side", "trade_id"]
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
        "expected_expiration_ts": RELEASE_TS + 60.0}


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
        "open_ts": [unix(2026, 9, 1, 12, 0, 0)], "close_ts": [RELEASE_TS]}
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
    assert list(kept.columns) == ["market", "event", "outcome", "open_ts", "close_ts"]
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
    markets_line, trades_line = report.summary()
    assert "1 read" in markets_line and "kept 1" in markets_line
    assert "52 read" in trades_line and "1 price_outside_0_1" in trades_line
    assert "1 block trades" in trades_line and "kept 50" in trades_line
