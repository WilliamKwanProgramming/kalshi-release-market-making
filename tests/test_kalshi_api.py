"""The client of Kalshi's public endpoints, run against canned responses (no network).

The responses are shaped like the documentation read on 2026-11-13 (docs/data.md); nothing here
shows that the live service behaves the same way. A fake transport returns the canned answers in
order and records every URL it is asked for.
"""

import http.client
import io
import json
import pathlib
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import pytest

from kalshi_mm import kalshi_api
from kalshi_mm.kalshi_api import HistoricalCutoff, KalshiApiError, KalshiClient, parse_iso_time

BASE = "https://example.test/trade-api/v2"


class FakeTransport:
    """Returns the queued answers one by one (an exception in the queue is raised)."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        assert self.answers, f"unexpected extra request: {url}"
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    def queries(self):
        return [urllib.parse.parse_qs(urllib.parse.urlsplit(u).query, keep_blank_values=True)
                for u in self.urls]

    def paths(self):
        return [urllib.parse.urlsplit(u).path for u in self.urls]


def reply(payload, status=200, headers=None):
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return status, headers or {}, body


def trade(n):
    return {"trade_id": f"t{n}", "ticker": "KXCPI-26OCT-T0.3", "count_fp": "10.00",
            "yes_price_dollars": "0.5600", "no_price_dollars": "0.4400",
            "taker_outcome_side": "yes", "taker_book_side": "bid",
            "created_time": "2026-10-23T14:00:00.123456Z", "is_block_trade": False}


def make_client(transport, **kwargs):
    sleeps = []
    client = KalshiClient(BASE, transport=transport, sleep=sleeps.append, **kwargs)
    return client, sleeps


# ---------------------------------------------------------------------------- time stamps
@pytest.mark.parametrize("text, expected", [
    ("2026-10-24T12:30:00Z", 1792845000.0),
    ("2026-10-24T12:30:00+00:00", 1792845000.0),
    ("2026-10-24T08:30:00-04:00", 1792845000.0),
    ("2026-10-24 12:30:00+00:00", 1792845000.0),
    ("2026-10-24T12:30:00.123456Z", 1792845000.123456),
    ("2026-10-24T12:30:00.5Z", 1792845000.5),
    ("2026-10-24T12:30:00.123456789Z", 1792845000.123456),  # digits beyond the sixth are dropped
    ("1970-01-01T00:00:00Z", 0.0),
])
def test_parse_iso_time(text, expected):
    assert parse_iso_time(text) == expected


@pytest.mark.parametrize("bad", ["2026-10-24T12:30:00", "2026-10-24", "garbage", "", None, 1.5e9])
def test_parse_iso_time_refuses_unreadable_or_offsetless_times(bad):
    with pytest.raises(ValueError):
        parse_iso_time(bad)


def test_microseconds_survive_the_conversion_to_float_seconds():
    a = parse_iso_time("2026-10-24T12:30:00.000001Z")
    b = parse_iso_time("2026-10-24T12:30:00.000002Z")
    assert b > a and parse_iso_time("2026-10-24T12:30:00.000000Z") < a


# ----------------------------------------------------------------------------- pagination
def test_pages_are_followed_with_the_cursor_until_it_is_empty():
    fake = FakeTransport(reply({"trades": [trade(1), trade(2)], "cursor": "c1"}),
                         reply({"trades": [trade(3)], "cursor": "c2"}),
                         reply({"trades": [trade(4)], "cursor": ""}))
    client, sleeps = make_client(fake)
    got = list(client.iter_trades(ticker="KXCPI-26OCT-T0.3", limit=2))
    assert [t["trade_id"] for t in got] == ["t1", "t2", "t3", "t4"]
    first, second, third = fake.queries()
    assert "cursor" not in first  # the first page has no cursor
    assert second["cursor"] == ["c1"] and third["cursor"] == ["c2"]
    assert all(q["limit"] == ["2"] and q["ticker"] == ["KXCPI-26OCT-T0.3"]
               for q in (first, second, third))
    assert not fake.answers and sleeps == []


@pytest.mark.parametrize("last", [{"cursor": ""}, {"cursor": None}, {}])
def test_an_empty_null_or_missing_cursor_ends_the_pages(last):
    fake = FakeTransport(reply({"trades": [trade(1)], "cursor": "c1"}),
                         reply({"trades": [trade(2)], **last}))
    client, _ = make_client(fake)
    assert len(list(client.iter_trades())) == 2 and len(fake.urls) == 2


def test_a_single_empty_page_gives_no_records():
    fake = FakeTransport(reply({"trades": [], "cursor": ""}))
    assert list(make_client(fake)[0].iter_trades()) == []


def test_a_consumer_that_stops_early_does_not_fetch_further_pages():
    fake = FakeTransport(reply({"trades": [trade(1), trade(2)], "cursor": "c1"}))
    client, _ = make_client(fake)
    pages = client.iter_trades()
    assert fake.urls == []  # nothing is requested until the first record is asked for
    assert next(pages)["trade_id"] == "t1"
    assert len(fake.urls) == 1


def test_the_same_cursor_twice_raises_instead_of_looping_forever():
    fake = FakeTransport(reply({"trades": [trade(1)], "cursor": "c1"}),
                         reply({"trades": [trade(2)], "cursor": "c1"}))
    with pytest.raises(KalshiApiError, match="same cursor twice"):
        list(make_client(fake)[0].iter_trades())


def test_a_cursor_that_is_not_text_raises():
    fake = FakeTransport(reply({"trades": [trade(1)], "cursor": 12}))
    with pytest.raises(KalshiApiError, match="not a string"):
        list(make_client(fake)[0].iter_trades())


def test_market_pages_work_like_trade_pages_on_all_four_list_endpoints():
    for name, path, key in (("iter_markets", "/markets", "markets"),
                            ("iter_historical_markets", "/historical/markets", "markets"),
                            ("iter_trades", "/markets/trades", "trades"),
                            ("iter_historical_trades", "/historical/trades", "trades")):
        fake = FakeTransport(reply({key: [{"n": 1}], "cursor": "a b&c"}),
                             reply({key: [{"n": 2}], "cursor": ""}))
        client, _ = make_client(fake)
        assert list(getattr(client, name)()) == [{"n": 1}, {"n": 2}]
        assert fake.paths() == [f"/trade-api/v2{path}"] * 2


# -------------------------------------------------------------------------- query strings
def test_query_strings_are_encoded_and_unset_values_are_left_out():
    fake = FakeTransport(reply({"markets": [], "cursor": ""}))
    client, _ = make_client(fake)
    list(client.iter_markets(series_ticker="KX CPI&x=1", status="settled",
                             tickers=["A-1", "B,2"], min_settled_ts=1700000000,
                             max_settled_ts=np.int64(1700003600)))
    url = fake.urls[0]
    assert url.startswith(f"{BASE}/markets?") and " " not in url and "&x=1" not in url
    (query,) = fake.queries()
    assert query == {"limit": ["1000"], "series_ticker": ["KX CPI&x=1"], "status": ["settled"],
                     "tickers": ["A-1,B,2"], "min_settled_ts": ["1700000000"],
                     "max_settled_ts": ["1700003600"]}


def test_a_cursor_with_special_characters_comes_back_unchanged_in_the_next_request():
    cursor = "abc+/=&? %é"
    fake = FakeTransport(reply({"trades": [], "cursor": cursor}),
                         reply({"trades": [], "cursor": ""}))
    list(make_client(fake)[0].iter_trades())
    assert fake.queries()[1]["cursor"] == [cursor]


def test_booleans_and_integers_are_written_the_way_the_documentation_does():
    fake = FakeTransport(reply({"trades": [], "cursor": ""}), reply({"trades": [], "cursor": ""}))
    client, _ = make_client(fake)
    list(client.iter_trades(is_block_trade=False, min_ts=1, max_ts=2))
    list(client.iter_historical_trades(is_block_trade=True))
    first, second = fake.queries()
    assert first["is_block_trade"] == ["false"] and first["min_ts"] == ["1"]
    assert first["max_ts"] == ["2"] and second["is_block_trade"] == ["true"]
    assert "min_ts" not in second


def test_the_five_endpoints_use_the_documented_paths_and_a_trailing_slash_is_harmless():
    fake = FakeTransport(
        reply({"market_settled_ts": "2026-01-01T00:00:00Z",
               "trades_created_ts": "2026-02-01T00:00:00Z",
               "orders_updated_ts": "2026-03-01T00:00:00Z"}),
        reply({"markets": [], "cursor": ""}), reply({"markets": [], "cursor": ""}),
        reply({"trades": [], "cursor": ""}), reply({"trades": [], "cursor": ""}))
    client = KalshiClient(BASE + "/", transport=fake, sleep=lambda s: None)
    client.historical_cutoff()
    list(client.iter_markets())
    list(client.iter_historical_markets())
    list(client.iter_trades())
    list(client.iter_historical_trades())
    assert fake.paths() == ["/trade-api/v2/historical/cutoff", "/trade-api/v2/markets",
                            "/trade-api/v2/historical/markets", "/trade-api/v2/markets/trades",
                            "/trade-api/v2/historical/trades"]
    assert not any("//" in u.split("://", 1)[1] for u in fake.urls)


@pytest.mark.parametrize("call", [
    lambda c: c.iter_trades(limit=0), lambda c: c.iter_trades(limit=1001),
    lambda c: c.iter_trades(limit=True), lambda c: c.iter_markets(limit=2.5),
    lambda c: c.iter_markets(status="finalized"), lambda c: c.iter_trades(min_ts=1.5),
    lambda c: c.iter_trades(max_ts="1700000000"), lambda c: c.iter_markets(min_close_ts=True),
])
def test_arguments_that_the_documentation_does_not_allow_raise_before_any_request(call):
    fake = FakeTransport()
    with pytest.raises(ValueError):
        call(make_client(fake)[0])
    assert fake.urls == []


def test_a_base_url_without_https_is_refused():
    with pytest.raises(ValueError, match="https"):
        KalshiClient("http://example.test/trade-api/v2")


# --------------------------------------------------------------------------------- retries
def test_429_then_success_waits_for_the_retry_after_header():
    fake = FakeTransport(reply({"error": "too many requests"}, 429, {"Retry-After": "7"}),
                         reply({"trades": [trade(1)], "cursor": ""}))
    client, sleeps = make_client(fake)
    assert len(list(client.iter_trades())) == 1
    assert sleeps == [7.0] and len(fake.urls) == 2 and fake.urls[0] == fake.urls[1]


def test_the_retry_after_header_is_found_whatever_its_case_and_decimals_are_allowed():
    fake = FakeTransport(reply({}, 429, {"retry-after": "2.5"}),
                         reply({"trades": [], "cursor": ""}))
    client, sleeps = make_client(fake)
    list(client.iter_trades())
    assert sleeps == [2.5]


@pytest.mark.parametrize("value", ["soon", "-3", "nan", "inf", "Wed, 21 Oct 2026 07:28:00 GMT", ""])
def test_an_unusable_retry_after_header_falls_back_to_the_backoff(value):
    fake = FakeTransport(reply({}, 429, {"Retry-After": value}),
                         reply({"trades": [], "cursor": ""}))
    client, sleeps = make_client(fake)
    list(client.iter_trades())
    assert len(sleeps) == 1 and 0.5 <= sleeps[0] <= 1.0  # the first backoff: 1 s times 0.5 to 1


def test_429_without_retry_after_backs_off_exponentially_with_jitter_and_a_cap():
    n = 6
    fake = FakeTransport(*[reply({}, 429)] * n, reply({"trades": [], "cursor": ""}))
    client, sleeps = make_client(fake, max_retries=n, base_delay_s=1.0, max_delay_s=10.0)
    list(client.iter_trades())
    ceilings = [min(10.0, 2.0**k) for k in range(n)]  # 1, 2, 4, 8, 10, 10
    assert len(sleeps) == n
    for wait, ceiling in zip(sleeps, ceilings, strict=True):
        assert 0.5 * ceiling <= wait <= ceiling
    assert len(set(sleeps)) == n  # jitter: no two waits coincide


def test_the_waits_are_repeatable_for_a_seed_and_differ_between_seeds():
    def waits(seed):
        fake = FakeTransport(*[reply({}, 503)] * 3, reply({"trades": [], "cursor": ""}))
        client, sleeps = make_client(fake, jitter_seed=seed)
        list(client.iter_trades())
        return sleeps

    assert waits(5) == waits(5) and waits(5) != waits(6)


def test_a_retry_after_shorter_than_the_backoff_does_not_shorten_the_wait():
    fake = FakeTransport(*[reply({}, 429)] * 4, reply({"trades": [], "cursor": ""}))
    fake.answers[3] = reply({}, 429, {"Retry-After": "0"})
    client, sleeps = make_client(fake, max_retries=4)
    list(client.iter_trades())
    assert sleeps[3] >= 0.5 * 8.0  # the fourth wait is at least half of 1 * 2^3


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_5xx_is_retried_and_a_later_success_is_returned(status):
    fake = FakeTransport(reply({"error": "boom"}, status),
                         reply({"trades": [trade(1)], "cursor": ""}))
    client, sleeps = make_client(fake)
    assert len(list(client.iter_trades())) == 1 and len(sleeps) == 1


def test_5xx_until_the_retry_limit_raises_a_clear_error_after_exactly_that_many_tries():
    fake = FakeTransport(*[reply({"error": "boom"}, 500)] * 4)
    client, sleeps = make_client(fake, max_retries=3)
    with pytest.raises(KalshiApiError, match=r"giving up after 4 attempts.*HTTP 500") as info:
        list(client.iter_trades(ticker="X"))
    assert info.value.status == 500 and "ticker=X" in info.value.url
    assert len(fake.urls) == 4 and len(sleeps) == 3 and not fake.answers


def test_no_retries_allowed_means_one_attempt_and_no_sleep():
    fake = FakeTransport(reply({}, 500))
    client, sleeps = make_client(fake, max_retries=0)
    with pytest.raises(KalshiApiError, match="giving up after 1 attempts"):
        list(client.iter_trades())
    assert sleeps == []


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_other_client_errors_are_not_retried(status):
    fake = FakeTransport(reply({"code": "not_found", "message": "no such market"}, status))
    client, sleeps = make_client(fake)
    with pytest.raises(KalshiApiError, match=f"HTTP {status}: not_found: no such market") as info:
        list(client.iter_trades(ticker="NOPE"))
    assert info.value.status == status and len(fake.urls) == 1 and sleeps == []


def test_an_error_body_that_is_not_json_is_shown_truncated():
    fake = FakeTransport(reply(b"<html>" + b"x" * 500 + b"</html>", 404))
    with pytest.raises(KalshiApiError) as info:
        list(make_client(fake)[0].iter_trades())
    assert "<html>" in str(info.value) and len(str(info.value)) < 400


@pytest.mark.parametrize("error", [ConnectionResetError("reset"), TimeoutError("slow"),
                                   urllib.error.URLError("dns")])
def test_network_errors_are_retried_like_5xx(error):
    fake = FakeTransport(error, reply({"trades": [trade(1)], "cursor": ""}))
    client, sleeps = make_client(fake)
    assert len(list(client.iter_trades())) == 1 and len(sleeps) == 1


def test_a_network_error_every_time_ends_in_a_kalshi_api_error_without_a_status():
    fake = FakeTransport(*[ConnectionResetError("reset")] * 3)
    client, _ = make_client(fake, max_retries=2)
    with pytest.raises(KalshiApiError, match="network error: ConnectionResetError") as info:
        list(client.iter_trades())
    assert info.value.status is None


def test_other_exceptions_from_the_transport_are_not_swallowed():
    fake = FakeTransport(ValueError("a bug in the transport"))
    with pytest.raises(ValueError, match="a bug"):
        list(make_client(fake)[0].iter_trades())


# ------------------------------------------------------------------------ malformed bodies
@pytest.mark.parametrize("body, match", [
    (b"<html>gateway</html>", "not valid JSON"),
    (b"", "not valid JSON"),
    (b'{"trades": [', "not valid JSON"),
    (b"[1, 2, 3]", "not a JSON object"),
    (b'"text"', "not a JSON object"),
    (b'{"cursor": ""}', "no list of records"),
    (b'{"trades": {"a": 1}, "cursor": ""}', "no list of records"),
    (b'{"trades": [1, 2], "cursor": ""}', "no list of records"),
    (b'{"markets": [], "cursor": ""}', "no list of records"),  # the wrong key for trades
])
def test_a_malformed_body_raises_and_is_not_retried(body, match):
    fake = FakeTransport(reply(body))
    client, sleeps = make_client(fake)
    with pytest.raises(KalshiApiError, match=match):
        list(client.iter_trades())
    assert len(fake.urls) == 1 and sleeps == []


# ------------------------------------------------------------------------------- the cutoff
def test_the_cutoff_is_parsed_to_unix_seconds():
    fake = FakeTransport(reply({"market_settled_ts": "2026-09-01T00:00:00Z",
                                "trades_created_ts": "2026-10-01T12:00:00.5Z",
                                "orders_updated_ts": "2026-10-30T00:00:00+00:00"}))
    cutoff = make_client(fake)[0].historical_cutoff()
    assert cutoff == HistoricalCutoff(1788220800.0, 1790856000.5, 1793318400.0)
    assert fake.queries() == [{}]  # no parameters


@pytest.mark.parametrize("body", [
    {"market_settled_ts": "2026-09-01T00:00:00Z", "trades_created_ts": "2026-10-01T00:00:00Z"},
    {"market_settled_ts": "yesterday", "trades_created_ts": "2026-10-01T00:00:00Z",
     "orders_updated_ts": "2026-10-30T00:00:00Z"},
    {"market_settled_ts": "2026-09-01T00:00:00Z", "trades_created_ts": "2026-10-01T00:00:00",
     "orders_updated_ts": "2026-10-30T00:00:00Z"},
    {"market_settled_ts": 1788220800, "trades_created_ts": "2026-10-01T00:00:00Z",
     "orders_updated_ts": "2026-10-30T00:00:00Z"},
])
def test_a_cutoff_response_that_is_not_as_documented_raises(body):
    with pytest.raises(KalshiApiError, match="not as documented"):
        make_client(FakeTransport(reply(body)))[0].historical_cutoff()


# ------------------------------------------------------------------- credentials and headers
class FakeResponse:
    def __init__(self, status=200, headers=None, body=b"{}", fail_with=None):
        self.status, self.headers, self._body, self._fail = status, headers or {}, body, fail_with

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        if self._fail:
            raise self._fail
        return self._body


def test_the_default_transport_sends_a_user_agent_and_accept_and_nothing_else(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout):
        seen.append((request.full_url, dict(request.header_items()), request.get_method(), timeout))
        return FakeResponse(200, {"Content-Type": "application/json"}, b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, headers, body = kalshi_api.urllib_transport(f"{BASE}/markets?limit=1", timeout=12.0)
    assert (status, body) == (200, b'{"ok": true}')
    assert headers == {"Content-Type": "application/json"}
    ((url, sent, method, timeout),) = seen
    assert url == f"{BASE}/markets?limit=1" and method == "GET" and timeout == 12.0
    assert set(sent) == {"User-agent", "Accept"} and sent["User-agent"].startswith("kalshi-mm/")
    assert sent["Accept"] == "application/json"


def test_the_client_passes_its_timeout_to_the_default_transport(monkeypatch):
    seen = []

    def fake_urlopen(request, timeout):
        seen.append(timeout)
        return FakeResponse(200, {}, b'{"trades": [], "cursor": ""}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    list(KalshiClient(BASE, timeout_s=7.5).iter_trades())
    assert seen == [7.5]


def test_the_default_transport_returns_refused_statuses_as_answers(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests",
                                     {"Retry-After": "3"}, io.BytesIO(b'{"error": "slow down"}'))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    status, headers, body = kalshi_api.urllib_transport(f"{BASE}/markets")
    assert status == 429 and headers == {"Retry-After": "3"} and b"slow down" in body


def test_the_default_transport_turns_a_cut_off_response_into_a_connection_error(monkeypatch):
    def fake_urlopen(request, timeout):
        return FakeResponse(fail_with=http.client.IncompleteRead(b"abc", 10))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(ConnectionError, match="IncompleteRead"):
        kalshi_api.urllib_transport(f"{BASE}/markets")


def test_the_module_has_no_code_that_reads_or_sends_credentials():
    """The documentation declares no security on these operations. Check the checker: the
    forbidden words are spelled out, so adding a header or an environment lookup fails here."""
    source = pathlib.Path(kalshi_api.__file__).read_text()
    for word in ("Authorization", "Bearer", "KALSHI-ACCESS", "environ", "getenv", "api_key",
                 "private_key", "password"):
        assert word not in source, word


def test_every_request_is_a_get_of_a_url_and_nothing_else(monkeypatch):
    """The transport gets one argument, the URL; the query holds only documented parameters."""
    documented = {"limit", "cursor", "ticker", "min_ts", "max_ts", "is_block_trade",
                  "series_ticker", "event_ticker", "status", "tickers", "min_close_ts",
                  "max_close_ts", "min_settled_ts", "max_settled_ts"}
    fake = FakeTransport(reply({"markets": [], "cursor": ""}), reply({"trades": [], "cursor": ""}))
    client, _ = make_client(fake)
    list(client.iter_markets(series_ticker="S", status="open"))
    list(client.iter_trades(ticker="T", min_ts=1, max_ts=2, is_block_trade=False))
    for query in fake.queries():
        assert set(query) <= documented
