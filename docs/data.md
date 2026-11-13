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
| none | `endpoint` | not an API field: `historical` or `live`, the endpoint the trade was read from, or `both` when the same `trade_id` came from the two; stored in SQLite, not in the contract frame |

`no_price_dollars` and `taker_book_side` are not read: the price is the YES price whichever side
traded. `seq` counts 0, 1, ... in the order (`ts`, `trade_id`) within a market. A trade at or after the
release is dropped.

| Market field | Column | Rule |
|---|---|---|
| `ticker`, `event_ticker` | `markets.market`, `markets.event` | both required |
| `result` | `markets.outcome` | `yes` is 1, `no` is 0; `""` and `scalar` are missing, and counted |
| `open_time`, `close_time` | `markets.open_ts`, `markets.close_ts` | Unix seconds; closing before opening: rejected |
| `series_ticker` | `events.series` | the documented market has no such field: the event ticker up to its first hyphen |
| `price_level_structure`, `price_ranges` | `markets.price_level_structure`, `markets.price_ranges` | the market's price grid, stored as given (`price_ranges` as JSON text); not read by the analysis |
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
  repeated `trade_id` values are removed. The fetch prints, per series and for all series, two counts:
  the `trade_id` values returned by both endpoints (kept once), and the groups of kept trades with
  different `trade_id` but the same market, time, price, size and taker side, read from the two
  endpoints. The second kind is kept: it is either one trade that the endpoints gave two ids, counted
  twice, or genuine trades that look alike, and the data cannot tell which.
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
markets (market PRIMARY KEY, event -> events, outcome 0/1/NULL, open_ts REAL, close_ts REAL,
         price_level_structure, price_ranges)
trades  (market -> markets, ts REAL, seq INTEGER, price REAL, size REAL, side -1/1, trade_id,
         endpoint historical/live/both/NULL,
         PRIMARY KEY (market, ts, seq)) WITHOUT ROWID; unique (market, trade_id) when not NULL
meta    (key PRIMARY KEY, value)   -- key "source": "synthetic" or "kalshi-public-api"
```

Each fetch records itself in `meta`: `fetch_started_utc` and `fetch_finished_utc` (ISO 8601, UTC),
`git_commit` (`git rev-parse HEAD` of the repository the script ran from) and `git_dirty` (`true` if
`git status --porcelain` lists anything, else `false`; both `unknown` when git cannot tell). These
four keys hold the first fetch and are never replaced; `fetch_runs` holds every fetch, the first
included, as a JSON list of the same four fields. A fetch that stops records the time it stopped.
`scripts/run_real.py` prints the list and writes a line to `DEVIATIONS.md` when the database has no
record, when a fetch had uncommitted changes, or when the commit of a fetch is unknown, unknown to
the repository or differs from the tag `real-run` in the files the tag check guards. It also
compares the start of the earliest fetch with the time at which the tag was created, which only an
annotated tag (`git tag -a`) records: a fetch that started before the tag, a fetch whose start time
cannot be read, and a tag that is not annotated are departures too. A start time without an offset
is read as UTC.

Column types are checked (`typeof`), prices lie in (0, 1), sizes are positive, and one database holds
one source. Each market that passes the filters is written in its own transaction, replacing its
earlier trades. A database written before a column was added to the schema is refused with a message
that names the missing columns (`panel.schema_problem`); build a new one. `panel.trade_panel_sql`
computes the markout panel of `docs/model.md` section 5 inside SQLite (version 3.35 or later).

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
- Tick. The backtest quotes on a grid of 0.01 USD. Kalshi's Market object has `price_level_structure`
  and `price_ranges` (each range with `start`, `end` and `step`, in USD), and some structures allow
  steps of 0.005, 0.002, 0.001 or 0.0001 USD (https://docs.kalshi.com/getting_started/subpenny_pricing,
  read on 2026-11-13). Both fields are stored per market. `scripts/run_real.py` counts, per series and
  in total, the kept trades whose `yes_price_dollars` is not a whole multiple of 0.01 USD (tolerance
  1e-9 USD) and the markets by `price_level_structure`, and prints and writes the counts (table K9,
  `summary.json`). One rule is fixed in advance (`docs/protocol.md`): when more than 5% of the kept
  trades are off the 0.01 USD tick, P2 is labelled exploratory, because the backtest quotes and
  fills on that tick; no trade is removed. Not checked: whether a trade's price lies on its own
  market's `price_ranges`, whether the grid changed during a market's life (only the grid of the
  record read at the fetch is stored), and what a sub-cent grid would do to the quotes and fills of
  the backtest.
- Both trade endpoints. `scripts/run_real.py` also counts the kept trades whose `trade_id` both trade
  endpoints returned and the groups of kept trades alike in everything but `trade_id` across the two
  endpoints. Such groups are kept, so if Kalshi gave one trade two ids, it is counted twice; the
  count bounds how many trades that can be.
- The documentation does not say: whether `min_ts` and `max_ts` include their bound; the order of
  trades within a page; the rate limit without credentials; whether a trade near the cutoff has the
  same `trade_id` on both endpoints (counted, above); which filters of `/historical/markets` can be
  combined ("Filters are mutually exclusive"). `/markets` answers are cached briefly, so a market
  that has just settled could show an empty `result`, and then has no outcome in the data.
- Terms. Kalshi's Developer Agreement (v1.1,
  https://kalshi-public-docs.s3.amazonaws.com/Kalshi-Developer-Agreement.pdf) limits use of the API,
  in section 3, to facilitating a member's own trading, and lists in 3.1 as "explicitly prohibited"
  collecting, caching, aggregating or storing data accessed via the API "except for purposes of
  facilitating your own trading on Kalshi", and sharing such data with third parties without prior
  written authorization. The welcome page (https://docs.kalshi.com/welcome.md) says that continuing to
  use the API means agreeing to it. The agreement does not say whether it covers the public endpoints.
  No conclusion is drawn here; read it before you run the script. Fetched data go to `data/`, which
  git ignores, and are never committed or shared.
