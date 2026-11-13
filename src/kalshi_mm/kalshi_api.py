"""Client for Kalshi's public market-data endpoints. Read-only, no credentials.

Endpoints, relative to ``https://external-api.kalshi.com/trade-api/v2`` (docs/data.md has the
documentation pages and the date they were read):

===========================  ====================================================
``GET /markets/trades``      :meth:`KalshiClient.iter_trades`
``GET /markets``             :meth:`KalshiClient.iter_markets`
``GET /historical/trades``   :meth:`KalshiClient.iter_historical_trades`
``GET /historical/markets``  :meth:`KalshiClient.iter_historical_markets`
``GET /historical/cutoff``   :meth:`KalshiClient.historical_cutoff`
===========================  ====================================================

The specification declares no security on these operations, so the client sends no credentials: it
has no code that reads, stores or sends a key, and the only request headers are ``User-Agent``
and ``Accept``. A page is requested with ``limit`` (at most 1000) and the ``cursor``
of the previous page; the last page has an empty (or missing) cursor.

Failures. HTTP 429 and 5xx answers, and network errors (a reset connection, a timeout), are
retried at most ``max_retries`` times. The wait before retry ``k`` (``k = 0, 1, ...``) is
``min(max_delay_s, base_delay_s 2^k)`` times a random factor between 0.5 and 1 (the factor comes
from a generator seeded with ``jitter_seed``, so a run is repeatable), or the ``Retry-After``
header in seconds if that is longer. Any other status (404, 400, ...) raises at once; so does a
body that is not the JSON object the documentation describes. Every failure is a
:class:`KalshiApiError`.

Tested against canned responses shaped like the documentation; not run against the live service.
"""

from __future__ import annotations

import datetime
import functools
import http.client
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from kalshi_mm import __version__

__all__ = [
    "BASE_URL",
    "HistoricalCutoff",
    "KalshiApiError",
    "KalshiClient",
    "parse_iso_time",
    "urllib_transport",
]

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
USER_AGENT = f"kalshi-mm/{__version__} (research; read-only public endpoints)"
MAX_PAGE_SIZE = 1000
MARKET_STATUSES = ("unopened", "open", "paused", "closed", "settled")

Response = tuple[int, Mapping[str, str], bytes]  # status, headers, body
Transport = Callable[[str], Response]
_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)


class KalshiApiError(Exception):
    """A request that failed for good: a refused status, a bad body or too many retries."""

    def __init__(self, message: str, *, url: str | None = None, status: int | None = None):
        super().__init__(message if url is None else f"{message} ({url})")
        self.url = url
        self.status = status


@dataclass(frozen=True)
class HistoricalCutoff:
    """The three cutoffs of ``GET /historical/cutoff`` in Unix seconds.

    Markets settled before ``market_settled_ts`` are only on ``/historical/markets``; trades
    created before ``trades_created_ts`` are only on ``/historical/trades``. The cutoffs advance
    over time.
    """

    market_settled_ts: float
    trades_created_ts: float
    orders_updated_ts: float


def parse_iso_time(text: str) -> float:
    """Unix seconds of an ISO 8601 timestamp that carries an offset (``Z`` or ``+HH:MM``).

    Microseconds are kept; digits beyond the sixth are dropped. A timestamp without an offset, or
    one that cannot be read, raises ``ValueError``: a guessed time zone would shift every release.
    """
    if not isinstance(text, str):
        raise ValueError(f"a timestamp must be a string, got {text!r}")
    moment = datetime.datetime.fromisoformat(text)
    if moment.tzinfo is None:
        raise ValueError(f"the timestamp {text!r} has no UTC offset (write it like ...T12:30:00Z)")
    return (moment - _EPOCH) / datetime.timedelta(seconds=1)


def urllib_transport(url: str, timeout: float = 30.0) -> Response:
    """GET ``url`` with ``urllib``; return ``(status, headers, body)``, also for 4xx and 5xx.

    The headers are ``User-Agent`` and ``Accept``, nothing else. A connection that fails raises
    ``OSError`` (``ConnectionError`` for a response that is cut short).
    """
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as error:  # a refused status is an answer, not a failure
        with error:
            return error.code, dict(error.headers.items()), error.read()
    except http.client.HTTPException as error:  # for example IncompleteRead
        raise ConnectionError(f"{type(error).__name__}: {error}") from error


def _query_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Sequence) and not isinstance(value, str):
        return ",".join(str(v) for v in value)
    return str(value)


def _whole_seconds(name: str, value: int | None) -> int | None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int | np.integer)):
        raise ValueError(f"{name} must be a whole number of Unix seconds, got {value!r}")
    return None if value is None else int(value)


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """The ``Retry-After`` header as seconds; a date, a negative or a garbled value is ignored."""
    for key, value in headers.items():
        if key.lower() == "retry-after":
            try:
                seconds = float(value)
            except (TypeError, ValueError):
                return None
            return seconds if math.isfinite(seconds) and seconds >= 0.0 else None
    return None


def _error_text(body: bytes) -> str:
    """The ``code`` and ``message`` of an error body, or its first 200 characters."""
    try:
        parsed = json.loads(body)
        if isinstance(parsed, dict) and ("message" in parsed or "code" in parsed):
            return f"{parsed.get('code', '')}: {parsed.get('message', '')}".strip(": ")
    except ValueError:
        pass
    return body[:200].decode("utf-8", errors="replace")


class KalshiClient:
    """Reads the public endpoints; see the module docstring for the retry rules.

    ``transport`` is a callable that takes a URL and returns ``(status, headers, body)``; the
    default is :func:`urllib_transport` with ``timeout_s``. ``sleep`` is called with the number of
    seconds to wait before a retry.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 5,
        base_delay_s: float = 1.0,
        max_delay_s: float = 60.0,
        jitter_seed: int = 0,
        timeout_s: float = 30.0,
    ):
        if not base_url.startswith("https://"):
            raise ValueError("base_url must start with https://")
        if max_retries < 0 or base_delay_s <= 0.0 or max_delay_s < base_delay_s or timeout_s <= 0:
            raise ValueError("max_retries >= 0, 0 < base_delay_s <= max_delay_s, timeout_s > 0")
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s
        self._transport = transport or functools.partial(urllib_transport, timeout=timeout_s)
        self._sleep = sleep
        self._rng = np.random.default_rng(jitter_seed)

    # ------------------------------------------------------------------------ one request
    def _url(self, path: str, params: Mapping[str, Any]) -> str:
        pairs = [(k, _query_value(v)) for k, v in params.items() if v is not None]
        query = urllib.parse.urlencode(pairs)
        return f"{self.base_url}{path}" + (f"?{query}" if query else "")

    def _wait(self, attempt: int, retry_after: float | None) -> float:
        ceiling = min(self.max_delay_s, self.base_delay_s * 2.0**attempt)
        backoff = ceiling * (0.5 + 0.5 * float(self._rng.random()))
        return backoff if retry_after is None else max(backoff, retry_after)

    def _get(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """The decoded JSON object of one GET request, retried as the module docstring says."""
        url = self._url(path, params)
        problem, status = "", None
        for attempt in range(self.max_retries + 1):
            retry_after = None
            try:
                status, headers, body = self._transport(url)
            except OSError as error:
                problem, status = f"network error: {type(error).__name__}: {error}", None
            else:
                if 200 <= status < 300:
                    return self._decode(url, body)
                if status != 429 and status < 500:
                    raise KalshiApiError(
                        f"HTTP {status}: {_error_text(body)}", url=url, status=status
                    )
                problem, retry_after = f"HTTP {status}", _retry_after(headers)
            if attempt < self.max_retries:
                self._sleep(self._wait(attempt, retry_after))
        raise KalshiApiError(
            f"giving up after {self.max_retries + 1} attempts, the last was: {problem}",
            url=url,
            status=status,
        )

    @staticmethod
    def _decode(url: str, body: bytes) -> dict[str, Any]:
        try:
            decoded = json.loads(body)
        except ValueError as error:
            raise KalshiApiError(f"the response is not valid JSON: {error}", url=url) from error
        if not isinstance(decoded, dict):
            raise KalshiApiError("the response is not a JSON object", url=url)
        return decoded

    # ------------------------------------------------------------------------- pagination
    def _pages(self, path: str, key: str, params: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
        """The records under ``key``, page after page, until the cursor is empty or missing."""
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page_params = {**params, "cursor": cursor}
            page = self._get(path, page_params)
            records = page.get(key)
            if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
                raise KalshiApiError(
                    f"the response has no list of records under {key!r}",
                    url=self._url(path, page_params),
                )
            yield from records
            cursor = page.get("cursor")
            if not cursor:
                return
            if not isinstance(cursor, str):
                raise KalshiApiError(
                    f"the cursor is not a string: {cursor!r}", url=self._url(path, page_params)
                )
            if cursor in seen:
                raise KalshiApiError(
                    "the server returned the same cursor twice", url=self._url(path, page_params)
                )
            seen.add(cursor)

    @staticmethod
    def _limit(limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be an integer from 1 to {MAX_PAGE_SIZE}, got {limit!r}")
        return limit

    # -------------------------------------------------------------------------- endpoints
    def historical_cutoff(self) -> HistoricalCutoff:
        """``GET /historical/cutoff``: the three cutoffs, parsed to Unix seconds."""
        path = "/historical/cutoff"
        body = self._get(path, {})
        try:
            return HistoricalCutoff(
                market_settled_ts=parse_iso_time(body["market_settled_ts"]),
                trades_created_ts=parse_iso_time(body["trades_created_ts"]),
                orders_updated_ts=parse_iso_time(body["orders_updated_ts"]),
            )
        except (KeyError, ValueError) as error:
            raise KalshiApiError(
                f"the cutoff response is not as documented: {error!r}", url=self._url(path, {})
            ) from error

    def iter_trades(
        self,
        *,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        is_block_trade: bool | None = None,
        limit: int = MAX_PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        """``GET /markets/trades``: trade records, following the cursor to the last page."""
        return self._pages(
            "/markets/trades",
            "trades",
            {
                "limit": self._limit(limit),
                "ticker": ticker,
                "min_ts": _whole_seconds("min_ts", min_ts),
                "max_ts": _whole_seconds("max_ts", max_ts),
                "is_block_trade": is_block_trade,
            },
        )

    def iter_historical_trades(
        self,
        *,
        ticker: str | None = None,
        min_ts: int | None = None,
        max_ts: int | None = None,
        is_block_trade: bool | None = None,
        limit: int = MAX_PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        """``GET /historical/trades``: trades older than the cutoff, same parameters and records
        as :meth:`iter_trades`."""
        return self._pages(
            "/historical/trades",
            "trades",
            {
                "limit": self._limit(limit),
                "ticker": ticker,
                "min_ts": _whole_seconds("min_ts", min_ts),
                "max_ts": _whole_seconds("max_ts", max_ts),
                "is_block_trade": is_block_trade,
            },
        )

    def iter_markets(
        self,
        *,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = None,
        tickers: Sequence[str] | None = None,
        min_close_ts: int | None = None,
        max_close_ts: int | None = None,
        min_settled_ts: int | None = None,
        max_settled_ts: int | None = None,
        limit: int = MAX_PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        """``GET /markets``: market records that are not yet archived (settled after the cutoff).

        The documentation lets the close-time filters go with ``status`` closed or empty and the
        settled-time filters with ``status`` settled or empty, and no two groups together.
        """
        if status is not None and status not in MARKET_STATUSES:
            raise ValueError(f"status must be one of {MARKET_STATUSES}, got {status!r}")
        return self._pages(
            "/markets",
            "markets",
            {
                "limit": self._limit(limit),
                "series_ticker": series_ticker,
                "event_ticker": event_ticker,
                "status": status,
                "tickers": tickers,
                "min_close_ts": _whole_seconds("min_close_ts", min_close_ts),
                "max_close_ts": _whole_seconds("max_close_ts", max_close_ts),
                "min_settled_ts": _whole_seconds("min_settled_ts", min_settled_ts),
                "max_settled_ts": _whole_seconds("max_settled_ts", max_settled_ts),
            },
        )

    def iter_historical_markets(
        self,
        *,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        tickers: Sequence[str] | None = None,
        limit: int = MAX_PAGE_SIZE,
    ) -> Iterator[dict[str, Any]]:
        """``GET /historical/markets``: archived markets. The documentation calls its filters
        mutually exclusive and has no time filter; filter the records by time yourself."""
        return self._pages(
            "/historical/markets",
            "markets",
            {
                "limit": self._limit(limit),
                "series_ticker": series_ticker,
                "event_ticker": event_ticker,
                "tickers": tickers,
            },
        )
