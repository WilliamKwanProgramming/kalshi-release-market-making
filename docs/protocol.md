# Protocol for the real-data run

This protocol was written before any Kalshi data was loaded by this repository, and before the
analysis code existed. Every choice that could change a result is fixed here. Its first commit is
`eee69c7`; `git log -p docs/protocol.md` shows every later change, and the list at the end of this
file says what each one was. The real-data run uses the repository at a commit that is tagged
`real-run` before any data are fetched (`git tag -a real-run -m "before the fetch" && git push
origin real-run`); the code and this file must not change between that tag and the run. The tag is
annotated because only an annotated tag records when it was made: `scripts/run_real.py` compares
that time with the start of each fetch, and a fetch that started before it, or a tag that is not
annotated, is a departure. Any departure is listed in `results/real/DEVIATIONS.md` with the reason
and the date, and the affected result is called exploratory. `scripts/run_real.py` writes that file
itself whenever a setting differs from the values fixed here.

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
0.2 cent. Also reported, not tested: each bucket, each series, the 1- and 15-minute horizons, the
settlement markout, $D$ of the markout divided by $4\hat m(1-\hat m)$ (cents per contract at a
price of 50 cents; trades whose mid proxy is outside $[0.03, 0.97]$ are left out), and $D$ with the
near window $(5, 30]$ minutes.

**P2. A premium by minutes to release against a flat premium of the same average over the
clock.** The walk-forward backtest of
`docs/model.md` (sections 6 and 7) over the last 12 hours before $T_e$, with $\gamma = 0.002$ per
contract per USD, $k = 100$ per USD, quote size 10, position limit 100, refresh 60 seconds, latency
1 second, strict trade-through fills, maker fee coefficient 0.0175, no quoting when the mid proxy
is below 0.03 or above 0.97. The first 20 releases (in order of $T_e$) are burn-in: they are run,
because later premiums are estimated from them, but they are not part of the comparison. The
quantity is the net settled P&L per release, summed over the release's markets, for `release_aware`
minus `flat_matched`, paired by release. `flat_matched` quotes one premium that is the average of
the bucket premiums over the refreshes of the quoting window (`docs/model.md`, section 6), so the
two policies quote the same premium per unit of $4m(1-m)$ averaged over the clock of the quoting
window and differ in when they quote it. They do not quote the same premium where the quotes are
hit: the trades, and so the fills, are not spread evenly over the clock, and `release_aware` quotes
more premium at the times at which the trades are. The hypothesis is that the mean difference is
positive. Also reported, not tested: the premium per unit of $4m(1-m)$ at the fills of every
policy (the mean over the contracts filled, table K5); the primary difference with
the flat premium scaled by the ratio of the premiums at the fills of `release_aware` and
`flat_matched`, so that the two quote the same premium at their fills (the scale is printed in the
name of the row); every other pair of the five policies, among them `release_aware` minus
`pooled_premium` (the pooled premium is an average over every trade of every earlier event and is
not the same size as the average of the bucket premiums over the quoting window, so the sign of
that difference follows the sizes; see the changes below), `always_on` and `pull_only`, fills and
cents per contract filled, each series, the two halves of the sample split at the median release
date, the touch fill rule, and the maker fee coefficient 0.

What a supported P2 can show. Only that, at the same average premium over the clock, a premium by
minutes to the release estimated from earlier events earned more than a flat one in this backtest,
whose fills are strict trade-throughs of a tape that does not react to the quotes. On the synthetic
worlds the same quantity is positive where no informed share is planted (the null world, table B1
of the README), and it is not positive against a flat premium that is scaled to quote the same
premium at the fills (tables B9 and B10, the latter exploratory), because every policy loses money
there and the policy that quotes more premium at its fills loses less. A supported P2 is therefore
not evidence of informed trading, and not evidence that a premium by minutes to the release pays at
an equal premium at the fills. Neither level is neutral: the average over the clock is fixed in
advance and is not what the P&L responds to, and the premium at the fills is what the P&L responds
to but is partly an outcome of the schedule, because a policy is filled less where it quotes more.

**P3. Recalibration.** The skill $\mathrm{BSS}$ of the walk-forward logistic recalibration of the
mid proxy at 24 hours before $T_e$ (prints up to and including that instant, none older than one
hour; a market with no proxy then is left out of P3 and counted), against the raw proxy, pooled
over markets with releases as clusters. Coefficients are refitted every 10 releases from at least
50 earlier ones. A release enters a fit only if it took place strictly before the price time of the
release being scored (24 hours before it): the outcome of a release in between was not known when
the price was observed. The first release scored is the first one with at least 50 releases before
its price time. The hypothesis is $\mathrm{BSS} > 0$. Reported, not tested: the
fitted $(a, b)$ over time and the BSS by series.

A test is "supported" only if its Holm-adjusted p-value is below 0.05. The p-value is one-sided:
the share of bootstrap replications of the statistic that are at or below zero (the level at which
the one-sided percentile interval stops excluding zero), never below 1/10,001. P1 and P2 are
labelled exploratory, whatever the p-values, when fewer than 100 releases enter the test: the
releases kept for P1 and the releases after the burn-in for P2. P3 is not labelled this way. P2 is
also labelled exploratory when more than 5% of the kept trades are off the 0.01 USD tick (check 4
below), because the backtest quotes and fills on that tick.

The number of releases is known from the official calendars and not from the data: the range above
holds about 100 scheduled releases (CPI about 34, payrolls about 34, GDP about 11 and Federal
Reserve decisions about 23, the year-on-year CPI and the unemployment rate sharing the release of
CPI and of payrolls), before the rules for the markets kept remove any. P1 is therefore expected to
be at the threshold, and P2, which needs the 20 releases of the burn-in as well, to be labelled
exploratory on this sample; P3 needs 50 earlier releases and would score about 50. The range and
the threshold are not changed because of this.

## What the repository may say afterwards

- A number in the README or on a resume comes from a file in `results/real/` written by the
  command above, and is named as real-data.
- A test that is not supported is reported as not supported, with its interval, next to the ones
  that are.
- A supported P2 is described as a premium by minutes to the release earning more than a flat
  premium of the same average over the clock, with the premium at the fills and the scaled row
  next to it; it is not described as a measure of informed trading.
- Nothing is claimed about profit: the backtest does not move prices, ignores the queue, and
  assumes the maker fee schedule above.

## Checks before reading any result

1. The loader's counts of series, events, markets and trades are printed and match a spot check
   of three events on the Kalshi website.
2. The distribution of $T_e$ minus market close time is printed and read.
3. The synthetic end-to-end test (`tests/test_pipeline.py`) passes on the machine that runs the
   analysis.
4. The counts of kept trades off the 0.01 USD tick, of trade ids returned by both trade endpoints
   and of groups of trades alike across the two endpoints are printed, by series, and read (table
   K9); they are reported, not used to remove trades. The one rule fixed in advance is the label of
   P2 above: more than 5% of the kept trades off the tick make P2 exploratory. The share is in
   K9, and the sample table K1 names it when it is over the limit.
5. The record of the fetches (start, end, commit, uncommitted changes) is printed; a fetch from a
   commit other than the tag `real-run` in the guarded files, or with uncommitted changes, or one
   that started before the annotated tag was created, is a departure written to `DEVIATIONS.md`,
   and so is a tag that is not annotated.

## Changes since the first commit

All were made before any real-data result existed.

- 2026-11-13: the commit tag `real-run` replaces "the commit that contains this file", because the
  analysis code was written after this file; events with the same $T_e$ are merged into one
  release; the proxy rule for markets kept, the P3 price time, the burn-in of 20 releases and the
  refit schedule are stated explicitly (the burn-in of 20 releases and the minimum of 50 releases
  are also in `docs/model.md`; the other values are in this file and in `Settings` of
  `src/kalshi_mm/protocol.py`).
- 2026-11-13, after the synthetic world was built: the scaled $D$ and the $(5, 30]$ window were added
  to the quantities reported for P1, because on the synthetic world the surge of the trade rate and
  the level of the price change $D$ without a change of the informed share (`docs/model.md`
  section 5). The tested quantity and its hypothesis did not change.
- 2026-11-13, after the synthetic backtest results and an independent review of the repository,
  before any real data: the tested comparison of P2 changed from `release_aware` minus
  `pooled_premium` to `release_aware` minus `flat_matched`. The review found that the first
  comparison followed the average size of the premium and not whether the premium depends on the
  time to the release. The pooled premium is an average over every trade of every earlier event;
  the bucket premiums averaged over the quoting window by the clock are smaller (on the synthetic
  planted world 1.65 against 2.80 cents per unit of $4m(1-m)$); every policy loses money per
  contract there, so the policy that quotes wider loses less, and the sign of the difference
  reversed when the pooled premium was scaled (first review) or the history lengthened (table B8
  of the README; the entry of the second review below corrects the citation of B9, which scales
  the flat premium of the new pair). `flat_matched` is the flat premium with the same average size
  (per unit of $4m(1-m)$, averaged over the clock) as the bucket premiums over the quoting window,
  estimated from earlier events only. Its definition was written down before the results for it
  were generated and was not tuned, but it was proposed by the first review, which had already
  reported the sign of the comparison at this level (see the entry of the second review).
  `pooled_premium` stays as a policy and its pair is reported. The hypothesis, the test (cluster
  bootstrap over releases, one-sided) and the Holm family did not change. A "supported" P2 says
  that the premium by bucket earns more than a flat premium of the same average over the clock in
  this backtest; it does not say that the premium measures informed trading. The backtest also
  reports the mean quoted spread of every policy (table B2 of the README), because the match is
  per unit of $4m(1-m)$ and the premium in USD also depends on the price.
- 2026-11-13, same review: the exploratory label counts the releases that enter each test (for P2
  the releases after the burn-in) and not the releases kept; and the P3 fit leaves out the
  releases that took place between the price time and the release being scored (the first version
  fitted on every earlier release, which used an outcome that was not known at the price time
  when two releases are less than 24 hours apart). No synthetic result changes: the synthetic
  releases are 168 hours apart.
- 2026-11-13, same review: two checks were added before reading any result (items 4 and 5 of
  "Checks before reading any result"): the counts of trades off the 0.01 USD tick and of trades
  seen on both trade endpoints, reported and not removed, and the record of the fetches, whose
  departures go to `DEVIATIONS.md`. No analysis choice changed.
- 2026-11-13, after a second independent review of the repository, before any real data. (1) The
  review found that "the same premium on average" holds over the clock of the quoting window and
  not at the fills, where `release_aware` quotes more premium per unit of $4m(1-m)$ than
  `flat_matched` (tables B2 and B10 of the README); the sentences that said that the comparison
  separates when a premium is applied from how large it is were replaced by the statement above,
  the backtest reports the premium at the fills of every policy, P2 reports the primary difference
  with the flat premium scaled to the premium at the fills of `release_aware`, and the synthetic
  study has the exploratory table B10. The quantity, the hypothesis, the test and the Holm family
  did not change. (2) P2 was retitled and "What a supported P2 can show" and the matching bullet
  of "What the repository may say afterwards" were added. (3) The expectation that P2 is labelled
  exploratory on this sample was added to the labelling rule. (4) The first review proposed the
  comparator `flat_matched` and had already reported the sign of the comparison at this level on the
  synthetic worlds (+2.80 USD per event planted and +1.26 null, at 12 hours), so the choice of this
  pair was not blind to that result; the sign of the new pair also reverses when the flat premium
  is scaled by about 1.5 (table B9), and the entry above cited B9 for the scaling of the pooled
  premium, which only the first review has. (5) P2 is labelled exploratory when more than 5% of the
  kept trades are off the 0.01 USD tick, and the tag `real-run` is annotated and its creation time
  compared with the start of every fetch. No synthetic result of the earlier tables changed.
