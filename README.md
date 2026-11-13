# Quoting the Print: Market Making on Kalshi Macro Contracts

[![CI](https://github.com/WilliamKwanProgramming/kalshi-release-market-making/actions/workflows/ci.yml/badge.svg)](https://github.com/WilliamKwanProgramming/kalshi-release-market-making/actions/workflows/ci.yml)

A Python library that measures adverse selection in Kalshi's release-day contracts (CPI, payrolls, GDP, unemployment, FOMC) from the public trade tape, quotes them with an Avellaneda-Stoikov rule whose premium depends on the minutes to the release, and backtests that quoting against the same tape. It is validated on a synthetic world with known truth; **no Kalshi data have been analysed in this repository**, and the real-data run is specified in [docs/protocol.md](docs/protocol.md) and has not been done.

## Status

| Feature | Implemented | Tested | Run on synthetic data with known truth | Run on real data |
|---|---|---|---|---|
| Kalshi public API client, fetch script, SQLite store, protocol filters | yes | yes (canned responses shaped like the documentation) | n/a | not run (the client has never run against the live service) |
| Synthetic world: logit diffusion, Glosten-Milgrom order flow, planted release effect | yes | yes (calibration and closed forms) | yes | n/a |
| Mid proxy from prints, markouts, release effect D with cluster bootstrap | yes | yes (hand tape, plain-loop check, look-ahead guards) | yes (tables M) | not run |
| Avellaneda-Stoikov quoting, four premium policies, trade-through backtest with Kalshi fees | yes | yes (closed forms, look-ahead guards) | yes (tables B) | not run |
| Favourite-longshot recalibration and Brier skill | yes | yes | yes (tables C) | not run |
| Protocol analysis P1 to P3, Holm family, tables, runner | yes | yes (end-to-end test) | yes (rehearsal, tables R) | not run |
| Mutation check of the test suite | yes | n/a | n/a | n/a |
| Release calendar (BLS, BEA, Federal Reserve) | no | no | no | not run: supplied by the user |

Model: [docs/model.md](docs/model.md). Protocol for the real data: [docs/protocol.md](docs/protocol.md). Data
loading: [docs/data.md](docs/data.md).

## Layout

    src/kalshi_mm/              the library: world, tape, markout, quoting, backtest, calibration, stats,
                                protocol (the three tests), panel and ingest (data loading), kalshi_api
    src/kalshi_mm/experiments/  one module per results directory (recovery, backtest_study, calibration_study, rehearsal)
    tests/                      unit, property, end-to-end and look-ahead tests; mutation_check.py
    scripts/                    reproduce.sh, make_tables.py, check_readme_numbers.py, fetch_kalshi.py, run_real.py
    results/                    outputs of the synthetic runs, each with its manifest (seed, commit, versions)

## Results

Every table below comes from a synthetic world that `src/kalshi_mm/world.py` builds, so the truth is known and each effect is one the generator plants. A table shows what the code does when the answer is known; it says nothing about how Kalshi behaves. The tables are pasted from `results/*/tables.md` by `scripts/make_tables.py`, and CI fails if this file differs from them. The other tables of those files are not repeated here.

### The release effect D is a cost measured on the tape, not the informed share

D is the mean 5-minute adverse selection within 30 minutes of the release minus the mean 360 minutes or more before it, in cents per contract. In the world with a planted informed share D is positive, but the world with a trade-rate surge and no informed-share profile already gives a large part of it, and the world with neither gives a slightly negative D. So a positive D on the tape does not by itself show informed trading, and the protocol reports scaled and uncut versions next to it.

<!-- table:M3 -->
<!-- /table:M3 -->

<!-- table:M4 -->
<!-- /table:M4 -->

### Intervals and sample size

The bootstrap over releases covers about as often as it should once a sample has around 100 events. An interval that treats every trade as independent covers too rarely for the raw D of the planted world and too often for the world with neither effect. For the planted effect a few dozen events give high power.

<!-- table:M5 -->
<!-- /table:M5 -->

<!-- table:M8 -->
<!-- /table:M8 -->

### Quoting

Four rules quote the same synthetic markets: always on, pulled in the last minutes, a premium for adverse selection pooled over all minutes to the release, and a premium estimated by bucket of minutes to the release. The comparison the protocol tests is the last against the third.

In the default world, with 12 hours of trading before each release, the premium by bucket loses to the pooled premium, and every policy loses money (B1, B2). B8 was added after those results, because the rehearsal below has 30 hours of trading and shows the opposite sign. It is exploratory, and it does not say which difference between the two windows causes the flip.

<!-- table:B1 -->
<!-- /table:B1 -->

<!-- table:B2 -->
<!-- /table:B2 -->

<!-- table:B8 -->
<!-- /table:B8 -->

### Recalibration

A favourite-longshot slope is planted in the prices 24 hours before the release; the walk-forward recalibration is scored by the Brier skill against the raw price. Where there is no slope the recalibration scores below the raw price in every row (its Brier skill is negative). With the strongest slope planted (0.7) and 1000 events it is supported in 24% of the datasets.

<!-- table:C1 -->
<!-- /table:C1 -->

### The whole pipeline on one synthetic world

The rehearsal writes a world of 200 events (30 hours of trading before each release) into SQLite, reads it back and runs the same protocol steps as the real-data runner. It shows what the output looks like and that the steps fit together; it is one world and one seed.

<!-- table:R2 -->
<!-- /table:R2 -->

<!-- table:R6 -->
<!-- /table:R6 -->

<!-- table:R8 -->
<!-- /table:R8 -->

## Reproduce

    python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[test]"
    pytest -q                                  # unit, property and end-to-end tests
    bash scripts/reproduce.sh                  # every results/ directory on synthetic data
    python3 tests/mutation_check.py --out results/mutation_check.txt
    python3 scripts/check_readme_numbers.py

`scripts/reproduce.sh` writes `results/MANIFEST.txt` (commands, seeds, commit, versions, wall and
CPU time); every results directory also has its own manifest. Nothing in `results/` is typed by
hand, and CI fails if a table in this README differs from `results/*/tables.md`.

The real-data run is a separate, one-time procedure with its own checks
([docs/protocol.md](docs/protocol.md), [docs/data.md](docs/data.md)):

    python3 scripts/fetch_kalshi.py ...        # public API to data/kalshi.sqlite (git-ignored)
    git tag real-run                           # before the data are fetched; the code must not change
    python3 scripts/run_real.py --db data/kalshi.sqlite

## Method (short)

* **World.** The price of a binary contract is a logistic function of a log-odds diffusion with the
  drift that makes the price a martingale. Each trade moves the log-odds by an amount that depends
  on the informed share of the order flow, which rises towards the release (Glosten and Milgrom
  1985); the trade rate surges at the release. Other makers quote at a fixed margin around the
  fair price.
* **Mid proxy.** The data are trades only, so the mid is the average of the last buy print and the
  last sell print when both are recent.
* **Markouts.** Effective half-spread, adverse selection at 5 minutes (cut at the release),
  realized spread and settlement markout, by minutes to the release. The release effect D is the
  mean adverse selection within 30 minutes of the release minus the mean over 360 minutes or more
  before it.
* **Quoting.** Avellaneda and Stoikov (2008) reservation price and spread, with the premium for
  adverse selection estimated from earlier releases: one pooled number, or one per bucket of
  minutes to the release. Four policies: always on, pulled in the last minutes, pooled premium,
  release-aware premium.
* **Fills and fees.** A print strictly beyond a quote fills it at the quote; maker fees follow
  Kalshi's schedule; inventory is held to resolution.
* **Inference.** Bootstrap over releases (events with the same release time are one release), a
  Holm correction over the three tested quantities, one-sided p-values as the share of bootstrap
  replications at or below zero.
* **Calibration.** Logistic recalibration of the mid proxy 24 hours before the release, fitted
  walk-forward, scored by the Brier skill against the raw proxy.

Details and every parameter: [docs/model.md](docs/model.md).

## Limitations

* **Synthetic data only.** Every result above comes from a world that the generator builds, so a
  recovered effect shows that the estimator works in that world, not that Kalshi behaves like it.
  The real-data run has a protocol ([docs/protocol.md](docs/protocol.md)) and has not been done;
  nothing here is a statement about Kalshi.
* **Trades only.** The public API gives trades, not the order book. The mid is a proxy built from
  the last buy and the last sell, which is stale where trading is thin, and the quoted spread is
  never seen.
* **The release effect D is not the informed share.** It is a difference of tape markouts. A surge
  of the trade rate and the level of the price change it without a change of the informed share
  (tables M); the scaled version removes the price level only.
* **The backtest does not trade against anyone.** A quote fills when a print goes strictly through
  it, at the quote; there is no queue, no market impact and no reaction of other makers. Profit
  figures are therefore about the rule, not about what an account would earn.
* **The quoting comparison flips with the trading window.** The premium by bucket loses to the
  pooled premium with 12 hours of trading before each release and wins with 30 (tables B1, B8, the
  latter exploratory), and every policy loses money in the backtest (table B2). The study does not
  say why, so nothing here shows that a release-aware premium pays.
* **Fees.** The maker fee follows the schedule that took effect on 7 July 2026 and is an
  assumption for earlier trades.
* **Release times come from the user.** The loader writes a draft from market expiry times; the
  analysis does not accept it until it is checked against the official calendars.
* **Three tests, one-sided.** The tested family is small by design; P3 prices are mid proxies, not
  prices anyone could trade at.

## References

- Avellaneda, M. and Stoikov, S. (2008). High-frequency trading in a limit order book.
  *Quantitative Finance* 8(3), 217-224.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability.
  *Monthly Weather Review* 78(1), 1-3.
- Bürgi, C., Deng, W. and Whelan, K. (2025). Makers and Takers: The Economics of the Kalshi
  Prediction Market. CESifo Working Paper 12122.
- Cameron, A. C., Gelbach, J. B. and Miller, D. L. (2008). Bootstrap-based improvements for
  inference with clustered errors. *Review of Economics and Statistics* 90(3), 414-427.
- Dalen, S. (2025). Toward Black Scholes for Prediction Markets: A Unified Kernel and Market
  Maker's Handbook. arXiv:2510.15205.
- Glosten, L. R. and Milgrom, P. R. (1985). Bid, ask and transaction prices in a specialist market
  with heterogeneously informed traders. *Journal of Financial Economics* 14(1), 71-100.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal
  of Statistics* 6(2), 65-70.
- Snowberg, E. and Wolfers, J. (2010). Explaining the favorite-long shot bias: is it risk-love or
  misperceptions? *Journal of Political Economy* 118(4), 723-746.
- Kalshi, API documentation (`https://docs.kalshi.com`) and fee schedule of 7 July 2026, both read
  on 2026-11-13.

## Disclaimer

Research code, not investment advice and not a trading system. This project is not affiliated
with or endorsed by Kalshi. Kalshi's data are subject to Kalshi's terms of use and are not
redistributed here; `data/` is git-ignored.
