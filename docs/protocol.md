# Protocol for the real-data run

This protocol was written before any Kalshi data was loaded by this repository, and before the
analysis code existed. Every choice that could change a result is fixed here. Its first commit is
`eee69c7`; `git log -p docs/protocol.md` shows every later change, and the list at the end of this
file says what each one was. The real-data run uses the repository at a commit that is tagged
`real-run` before any data are fetched (`git tag real-run && git push origin real-run`); the code
and this file must not change between that tag and the run. Any departure is listed in
`results/real/DEVIATIONS.md` with the reason and the date, and the affected result is called
exploratory. `scripts/run_real.py` writes that file itself whenever a setting differs from the
values fixed here.

Status: not run. Nothing in the README about real data is a result.

## Data

- Source: Kalshi's public REST API, `https://external-api.kalshi.com/trade-api/v2`. Markets from
  `GET /markets` and, for markets settled before the live cutoff, `GET /historical/markets`; trades
  from `GET /markets/trades` and `GET /historical/trades`; the cutoffs from `GET /historical/cutoff`.
  The loader (`scripts/fetch_kalshi.py`) has been tested against mocked responses shaped like the
  documentation read on 2026-11-13. It has not been run against the live service.
- Series (tickers as listed in Kalshi's fee schedule of 2026-07-07): `KXCPI`, `KXCPIYOY`,
  `KXPAYROLLS`, `KXGDP`, `KXU3`, `KXFEDDECISION`. A series that returns no settled markets is
  reported as empty, not replaced.
- Releases: every event whose release time $T_e$ lies in 2024-01-01 to 2026-10-31 (UTC).
- Release calendar: `data/release_calendar.csv` with columns `event_ticker`, `release_time_utc`,
  `source_url`, filled in from the official calendars of the Bureau of Labor Statistics, the Bureau
  of Economic Analysis and the Federal Reserve. The loader writes a draft from each market's
  expected expiration time; the draft is not accepted by the analysis until it is checked against
  the official calendars and saved under the name above. The analysis prints the distribution of
  the gap between $T_e$ and each market's close time.
- Releases and clusters: events with the same $T_e$ are one release, because they share the news
  (the CPI and the year-on-year CPI events; payrolls and the unemployment rate). They are merged
  before any statistic is computed. The bootstrap resamples releases, and "per event" below means
  per release.
- Markets kept: at least 50 trades before $T_e$, and the mid proxy of `docs/model.md` (section 4)
  defined at the time of at least one trade inside the last 12 hours before $T_e$. Counts removed
  by each rule are printed.
- Trades kept: those with `created_time` before $T_e$. Block trades (`is_block_trade` true) are
  dropped because they are negotiated off the book and are not what a resting quote could fill.
  Prices are `yes_price_dollars`; the taker's side is `taker_outcome_side` (older records with
  only `taker_side` use that).
- Kalshi data are not committed. They go to `data/`, which is ignored by git.

## Primary analyses

All three are computed by `scripts/run_real.py`, with 10,000 bootstrap replications over events
and seed 20261113. Intervals are 95% percentile intervals. The three tests form one family and are
adjusted with Holm's method at the 5% level.

**P1. Adverse selection near the print.** Using the definitions of `docs/model.md` (section 5), the
pooled difference $D = \overline{\mathrm{AS}}_{\text{near}} - \overline{\mathrm{AS}}_{\text{quiet}}$ of
the 5-minute adverse selection in cents per contract, near $=(0,30]$ minutes to release and quiet
$= (360,\infty)$. The hypothesis is $D > 0$. The ratio is reported only if the quiet mean exceeds
0.2 cent. Also reported, not tested: each bucket, each series, the 1- and 15-minute horizons, and
the settlement markout.

**P2. Time-conditional pricing of adverse selection.** The walk-forward backtest of
`docs/model.md` (sections 6 and 7) over the last 12 hours before $T_e$, with $\gamma = 0.002$ per
contract per USD, $k = 100$ per USD, quote size 10, position limit 100, refresh 60 seconds, latency
1 second, strict trade-through fills, maker fee coefficient 0.0175, no quoting when the mid proxy
is below 0.03 or above 0.97. The first 20 releases (in order of $T_e$) are burn-in: they are run,
because later premiums are estimated from them, but they are not part of the comparison. The
quantity is the net settled P&L per release, summed over the release's markets, for `release_aware`
minus `pooled_premium`, paired by release. The hypothesis is that the mean difference is positive. Also reported, not tested: `always_on` and `pull_only`, fills
and cents per contract filled, each series, the two halves of the sample split at the median release
date, the touch fill rule, and the maker fee coefficient 0.

**P3. Recalibration.** The skill $\mathrm{BSS}$ of the walk-forward logistic recalibration of the
mid proxy at 24 hours before $T_e$ (prints up to and including that instant, none older than one
hour; a market with no proxy then is left out of P3 and counted), against the raw proxy, pooled
over markets with releases as clusters. Coefficients are refitted every 10 releases from at least
50 earlier ones. The hypothesis is $\mathrm{BSS} > 0$. Reported, not tested: the
fitted $(a, b)$ over time and the BSS by series.

A test is "supported" only if its Holm-adjusted p-value is below 0.05. The p-value is one-sided:
the share of bootstrap replications of the statistic that are at or below zero (the level at which
the one-sided percentile interval stops excluding zero), never below 1/10,001. With fewer than 100
releases in the sample, P1 and P2 are labelled exploratory whatever the p-values.

## What the repository may say afterwards

- A number in the README or on a resume comes from a file in `results/real/` written by the
  command above, and is named as real-data.
- A test that is not supported is reported as not supported, with its interval, next to the ones
  that are.
- Nothing is claimed about profit: the backtest does not move prices, ignores the queue, and
  assumes the maker fee schedule above.

## Checks before reading any result

1. The loader's counts of series, events, markets and trades are printed and match a spot check
   of three events on the Kalshi website.
2. The distribution of $T_e$ minus market close time is printed and read.
3. The synthetic end-to-end test (`tests/test_pipeline.py`) passes on the machine that runs the
   analysis.

## Changes since the first commit

All were made before any real-data result existed.

- 2026-11-13: the commit tag `real-run` replaces "the commit that contains this file", because the
  analysis code was written after this file; events with the same $T_e$ are merged into one
  release; the proxy rule for markets kept, the P3 price time, the burn-in of 20 releases and the
  refit schedule are stated explicitly (they are the values in `docs/model.md`).
