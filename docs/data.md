# Data

`scripts/fetch_kalshi.py` reads Kalshi's public market-data endpoints, `src/kalshi_mm/ingest.py`
applies the filters of `docs/protocol.md` and `src/kalshi_mm/panel.py` stores the result in SQLite,
which is what the analysis reads. The client was tested against canned responses shaped like the
documentation; not run against the live service. No number in this repository comes from Kalshi data.

## Endpoints

The documentation was read on 2026-11-13. Base URL `https://external-api.kalshi.com/trade-api/v2`.
Every request is a GET without credentials. The reference pages declare no security requirement and
the first-request guide (https://docs.kalshi.com/getting_started/making_your_first_request.md)
suggests testing with the public GetMarkets endpoint, but no page says in words that no key is needed.

| Endpoint | Reference page |
|---|---|
| `/markets` (not yet archived; filter `series_ticker`) | https://docs.kalshi.com/api-reference/market/get-markets.md |
| `/historical/markets` (archived) | https://docs.kalshi.com/api-reference/historical/get-historical-markets.md |
| `/markets/trades` (filters `ticker`, `min_ts`, `max_ts`) | https://docs.kalshi.com/api-reference/market/get-trades.md |
| `/historical/trades` (archived) | https://docs.kalshi.com/api-reference/historical/get-historical-trades.md |
| `/historical/cutoff` (where the two tiers meet) | https://docs.kalshi.com/api-reference/historical/get-historical-cutoff-timestamps.md |

## Where each column comes from

| Trade field | Tape column | Rule |
|---|---|---|
| `created_time` | `ts` | ISO 8601 to Unix seconds, microseconds kept; no offset: rejected |
| `yes_price_dollars` | `price` | USD per YES contract, strictly between 0 and 1, else rejected |
| `count_fp` | `size` | contracts, a string such as `"10.00"`; zero or less: rejected |
| `taker_outcome_side` | `side` | `yes` is +1 (the taker is long YES: bought YES or sold NO), `no` is -1; the deprecated `taker_side` is read only when this is absent |
| `ticker` | `market` | the `event` column is the `event_ticker` of that market |
| `trade_id` | `trade_id` | stored in SQLite, not in the contract frame; a repeat within a market is removed |
| `is_block_trade` | none | `true`: dropped. A record without the field counts as not a block trade, and is counted |

`no_price_dollars` and `taker_book_side` are not read: the price is the YES price whichever side
traded. `seq` counts 0, 1, ... in the order (`ts`, `trade_id`) within a market. A trade at or after the
release is dropped.

| Market field | Column | Rule |
|---|---|---|
| `ticker`, `event_ticker` | `markets.market`, `markets.event` | both required |
| `result` | `markets.outcome` | `yes` is 1, `no` is 0; `""` and `scalar` are missing, and counted |
| `open_time`, `close_time` | `markets.open_ts`, `markets.close_ts` | Unix seconds; closing before opening: rejected |
| `series_ticker` | `events.series` | the documented market has no such field: the event ticker up to its first hyphen |
| none | `events.release_ts` | not an API field: it comes from the release calendar (below) |

## The historical cutoff

Kalshi moves older data from the live endpoints to the historical ones
(https://docs.kalshi.com/getting_started/historical_data.md). `GET /historical/cutoff` returns the
date-times `market_settled_ts`, `trades_created_ts` and `orders_updated_ts`. Markets settled before
`market_settled_ts` are only on `/historical/markets`; trades created before `trades_created_ts` are
only on `/historical/trades`. The cutoffs advance over time; no schedule is given. For a fetch:

- A series is listed from both market endpoints, one record per ticker: the live one, unless only the
  archived record has a `result` of `yes` or `no`.
- Trades are requested only for markets that can enter the dataset (a readable record, an event in the
  calendar, a release inside the window). A release at or before `trades_created_ts` needs the
  historical endpoint only; a later one needs it up to the cutoff and the live one from one second
  before the cutoff to the release.
- The pages say `min_ts` and `max_ts` filter "after" and "before" a time, not whether the time itself
  counts. The ranges are widened to be right either way, so a trade can arrive from both endpoints;
  repeated `trade_id` values are removed.
- The cutoff is read before and after each series. If it moved, the script warns: trades created
  between the old and the new cutoff may be missing for markets read after the move.

## Paging, rate limits and retries

Pages hold up to 1000 records and are followed with the `cursor` until it is empty or null
(https://docs.kalshi.com/getting_started/pagination.md). The rate-limit page
(https://docs.kalshi.com/getting_started/rate_limits.md) gives token budgets for authenticated tiers
(the lowest, Basic: 200 read tokens per second at 10 per request), nothing for requests without
credentials, and says to back off exponentially on HTTP 429, which carries no `Retry-After` header.
The client retries 429, any 5xx and network errors up to 5 times, waiting `min(60 s, 1 s * 2^k)` times
a seeded jitter between 0.5 and 1 before retry `k` (at least a `Retry-After` value, if sent). Other
4xx answers, a body that is not a JSON object and the last failed retry raise `KalshiApiError`.
Requests are not paced.

## SQLite schema (`panel.SCHEMA_SQL`)

```
events  (event PRIMARY KEY, series, release_ts REAL)
markets (market PRIMARY KEY, event -> events, outcome 0/1/NULL, open_ts REAL, close_ts REAL)
trades  (market -> markets, ts REAL, seq INTEGER, price REAL, size REAL, side -1/1, trade_id,
         PRIMARY KEY (market, ts, seq)) WITHOUT ROWID; unique (market, trade_id) when not NULL
meta    (key PRIMARY KEY, value)   -- key "source": "synthetic" or "kalshi-public-api"
```

Column types are checked (`typeof`), prices lie in (0, 1), sizes are positive, and one database holds
one source. Each market that passes the filters is written in its own transaction, replacing its
earlier trades. `panel.trade_panel_sql` computes the markout panel of `docs/model.md` section 5
inside SQLite (version 3.35 or later).

## The release calendar

`data/release_calendar.csv`: columns `event_ticker`, `release_time_utc`, `source_url`; one row per
event, a time with an explicit offset (`Z`, or `-04:00`/`-05:00`, converted) and a non-blank URL.
`load_release_calendar` names the lines that break these rules and refuses a name containing "draft".
No market field gives the release time, so this file decides where every trade is cut; a wrong row
moves its event's cut. To build it:

1. `PYTHONPATH=src python scripts/fetch_kalshi.py --calendar-draft data/release_calendar_draft.csv`
   writes one row per event from the earliest `expected_expiration_time` of its markets, with
   `source_url` empty. The analysis refuses it; it is a starting point.
2. For each event copy the official date and time of publication: for the consumer price index and
   the employment situation (`KXCPI`, `KXCPIYOY`, `KXPAYROLLS`, `KXU3`) from the Bureau of Labor
   Statistics release schedule, for gross domestic product (`KXGDP`) from the Bureau of Economic
   Analysis release schedule (the market's rules text says which estimate), for the decision
   (`KXFEDDECISION`) from the Federal Reserve's FOMC meeting calendar. No URL is given for these
   schedules: they could not be fetched when this was written.
3. Convert the time to UTC (Eastern time is UTC-4 while daylight saving time is in force, else UTC-5),
   paste the URL of the page you copied it from into `source_url` and save the file without "draft"
   in its name. Example row, made-up values: `KXCPI-26OCT,2026-10-14T12:30:00Z,https://example.org/x`.

## Running it

From the repository root: `PYTHONPATH=src python scripts/fetch_kalshi.py`. Options: `--series T [T ...]`
(default `KXCPI KXCPIYOY KXPAYROLLS KXGDP KXU3 KXFEDDECISION`), `--start` and `--end` (UTC days, both
included; `2024-01-01` and `2026-10-31`), `--db` (`data/kalshi.sqlite`), `--calendar`
(`data/release_calendar.csv`), `--calendar-draft PATH` (writes the draft and stops), `--min-trades`
(50) and `--dry-run` (lists what would be fetched; requests no trade, writes nothing). A run that
stops (the API failed after its retries, exit status 1; or you interrupted it) keeps what it stored;
run the same command again and the stored markets are skipped. A market dropped for too few trades
is not stored, so it is read again; the counts of a run cover that run only. Exit status 2: unusable
arguments or calendar, or a calendar that gives a stored event another release time. The protocol's
rule that the mid proxy is defined at a trade in the last 12 hours before the release is not applied
here; it needs the mid proxy.

## Limitations

- The client has never run against the live service. Field names follow the pages read on 2026-11-13
  and may have changed: the changelog (https://docs.kalshi.com/changelog/index.md) dates
  `taker_outcome_side` to 2026-05-06 and `is_block_trade` to 2026-05-29, and the trade pages said
  `taker_side` "will not be removed before May 14, 2026". No authentication is used: no key is read,
  stored or sent; the headers are `User-Agent` and `Accept`.
- The documentation does not say: whether `min_ts` and `max_ts` include their bound; the order of
  trades within a page; the rate limit without credentials; whether a trade near the cutoff has the
  same `trade_id` on both endpoints; which filters of `/historical/markets` can be combined ("Filters
  are mutually exclusive"). `/markets` answers are cached briefly, so a market that has just settled
  could show an empty `result`, and then has no outcome in the data.
- Terms. Kalshi's Developer Agreement (v1.1,
  https://kalshi-public-docs.s3.amazonaws.com/Kalshi-Developer-Agreement.pdf) limits use of the API,
  in section 3, to facilitating a member's own trading, and lists in 3.1 as "explicitly prohibited"
  collecting, caching, aggregating or storing data accessed via the API "except for purposes of
  facilitating your own trading on Kalshi", and sharing such data with third parties without prior
  written authorization. The welcome page (https://docs.kalshi.com/welcome.md) says that continuing to
  use the API means agreeing to it. The agreement does not say whether it covers the public endpoints.
  No conclusion is drawn here; read it before you run the script. Fetched data go to `data/`, which
  git ignores, and are never committed or shared.
