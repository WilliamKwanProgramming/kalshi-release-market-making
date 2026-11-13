# Quoting the Print: Market Making on Kalshi Macro Contracts

[![CI](https://github.com/WilliamKwanProgramming/kalshi-release-market-making/actions/workflows/ci.yml/badge.svg)](https://github.com/WilliamKwanProgramming/kalshi-release-market-making/actions/workflows/ci.yml)

A Python library that measures adverse selection in Kalshi's release-day contracts (CPI, payrolls, GDP, unemployment, FOMC) from the public trade tape, quotes them with an Avellaneda-Stoikov rule whose premium depends on the minutes to the release, and backtests that quoting against the same tape. It is checked against a synthetic world with known truth; **no Kalshi data have been analysed in this repository**, and the real-data run is specified in [docs/protocol.md](docs/protocol.md) and has not been done.

## Status

| Feature | Implemented | Tested | Run on synthetic data with known truth | Run on real data |
|---|---|---|---|---|
| Kalshi public API client, fetch script, SQLite store, protocol filters | yes | yes (canned responses shaped like the documentation) | partly (the SQLite store, in the rehearsal; not the client, the fetch script or the filters) | not run (the client has never run against the live service) |
| Synthetic world: logit diffusion, Glosten-Milgrom order flow, planted release effect | yes | yes (calibration and closed forms) | yes | n/a |
| Mid proxy from prints, markouts, release effect D with cluster bootstrap | yes | yes (hand tape, plain-loop check, look-ahead guards) | yes (tables M) | not run |
| Avellaneda-Stoikov quoting, five policies, trade-through backtest with Kalshi fees | yes | yes (closed forms, look-ahead guards) | yes (tables B) | not run |
| Favourite-longshot recalibration and Brier skill | yes | yes | yes (tables C) | not run |
| Protocol analysis P1 to P3, Holm family, tables, runner | yes | yes (end-to-end test) | yes (rehearsal, tables R) | not run |
| Mutation check of the test suite | yes (492 hand-written mutants, all caught) | n/a | n/a | n/a |
| Release calendar (BLS, BEA, Federal Reserve) | no | no | no | not run: supplied by the user |

Model: [docs/model.md](docs/model.md). Protocol for the real data: [docs/protocol.md](docs/protocol.md). Data loading: [docs/data.md](docs/data.md).

## Layout

* `src/kalshi_mm/`: the library (world, tape, markout, quoting, backtest, calibration, stats, the
  three protocol tests, the loader and the Kalshi API client); `experiments/` has one module per
  results directory.
* `tests/`: unit, property, end-to-end and look-ahead tests, and `mutation_check.py`.
* `scripts/`: `reproduce.sh`, `make_tables.py`, `check_readme_numbers.py`, `fetch_kalshi.py` and
  `run_real.py`; `constraints/lowest.txt` pins the lowest supported versions for one CI job.
* `results/`: the outputs of the synthetic runs, each with a manifest (seed, commit, versions).

## Results

Every table below comes from a synthetic world that `src/kalshi_mm/world.py` builds, so the truth is known and each effect is one the generator plants. A table shows what the code does when the answer is known; it says nothing about how Kalshi behaves. The tables are pasted from `results/*/tables.md` by `scripts/make_tables.py`, and CI fails if this file differs from them. The other tables of those files are not repeated here.

### The release effect D is a cost measured on the tape, not the informed share

D is the mean 5-minute adverse selection within 30 minutes of the release minus the mean 360 minutes or more before it, in cents per contract. In the world with a planted informed share D is positive on the tape, and table M1 shows where that comes from. The informed share rises towards the release, but the cost per contract measured with the true price (the oracle) falls over the last hour, because the price saturates there (84% of the planted events end outside [0.01, 0.99], table M4) and p(1-p) shrinks; the scaled column, which divides by 4m(1-m), does rise. The oracle D is negative. The positive D on the tape arises because the proxy under-measures the quiet window by far more than the window near the release. The world with a trade-rate surge and no informed-share profile gives a large part of the planted D (M4), and the world with neither gives a slightly negative D (M3). So a positive D on the tape does not by itself show informed trading, and the protocol reports scaled and uncut versions next to it.

<!-- table:M1 -->

### M1. Adverse selection by minutes to release, planted world: tape proxy, oracle and price impact (synthetic data)

Reading: cents per contract over 3000 events, informed share at the mean time to release of the bucket; the tape proxy minus the oracle is -0.02 near (0, 30] and -1.21 in the quiet window, so D is +0.91 on the tape, -0.28 by the oracle and -0.28 by the price impact.

| minutes to release | trades | tape AS | oracle AS | tape - oracle | impact | scaled tape AS | informed share |
|---|---|---|---|---|---|---|---|
| (0,5] | 49896 | +0.83 | +0.94 | -0.12 | +0.94 | +8.85 | 0.234 |
| (5,15] | 89705 | +1.50 | +1.49 | +0.02 | +1.49 | +8.43 | 0.194 |
| (15,30] | 111374 | +1.97 | +1.97 | -0.00 | +1.99 | +6.72 | 0.145 |
| (30,60] | 158096 | +2.00 | +2.04 | -0.05 | +2.03 | +4.67 | 0.096 |
| (60,180] | 250756 | +1.39 | +1.64 | -0.26 | +1.65 | +2.48 | 0.056 |
| (180,360] | 117490 | +0.72 | +1.66 | -0.94 | +1.65 | +1.09 | 0.050 |
| >360 | 169887 | +0.66 | +1.88 | -1.21 | +1.88 | +0.87 | 0.050 |
| near (0,30] | 250975 | +1.57 | +1.59 | -0.02 | +1.60 | +7.44 | 0.176 |
| D = near - quiet |  | +0.91 | -0.28 | +1.19 | -0.28 | +6.57 | 0.126 |

Does not show: how the proxy behaves on a real tape (stale prints, ticks, thin books); the oracle is the latent mid before the first trade at or after the horizon, and after the last trade when none is left.

<!-- /table:M1 -->

<!-- table:M3 -->

### M3. Population release effect by world and estimand (synthetic data)

Reading: D = near minus quiet in cents per contract over 3000 events per world, with its 95% cluster-bootstrap interval; raw D is positive with the surge alone (spec_null, +0.41 cents) and negative with neither (flat_null, -0.16 cents), so D is a per-contract cost measured on the tape, not an estimate of the informed share; scaled D near (5, 30] removes the price-level artefact (flat_null +0.03 cents, with an interval that contains zero) but not the surge (spec_null +1.61 cents, interval +1.56 to +1.66).

| world | estimand | D | 95% interval | near mean | quiet mean |
|---|---|---|---|---|---|
| planted | (a) raw D, near (0, 30] | +0.91 | +0.84 to +0.99 | +1.57 | +0.66 |
| planted | (b) scaled D, near (0, 30] | +6.57 | +6.36 to +6.77 | +7.44 | +0.87 |
| planted | (c) raw D, near (5, 30] | +1.10 | +1.01 to +1.18 | +1.76 | +0.66 |
| planted | (d) scaled D, near (5, 30] | +6.42 | +6.20 to +6.64 | +7.29 | +0.87 |
| spec_null | (a) raw D, near (0, 30] | +0.41 | +0.37 to +0.44 | +1.07 | +0.66 |
| spec_null | (b) scaled D, near (0, 30] | +1.55 | +1.51 to +1.60 | +2.43 | +0.87 |
| spec_null | (c) raw D, near (5, 30] | +0.44 | +0.41 to +0.48 | +1.11 | +0.66 |
| spec_null | (d) scaled D, near (5, 30] | +1.61 | +1.56 to +1.66 | +2.48 | +0.87 |
| flat_null | (a) raw D, near (0, 30] | -0.16 | -0.19 to -0.13 | +0.49 | +0.65 |
| flat_null | (b) scaled D, near (0, 30] | -0.04 | -0.08 to +0.00 | +0.82 | +0.86 |
| flat_null | (c) raw D, near (5, 30] | -0.12 | -0.15 to -0.09 | +0.53 | +0.65 |
| flat_null | (d) scaled D, near (5, 30] | +0.03 | -0.01 to +0.08 | +0.89 | +0.86 |

Does not show: whether D means the same on real tapes; the planted default saturates prices near the end (84% of its events end outside [0.01, 0.99]), which shrinks p(1 - p) toward the release.

<!-- /table:M3 -->

<!-- table:M4 -->

### M4. Share of the planted raw D that the trade-rate surge gives, and price saturation (synthetic data)

Reading: D(spec_null) / D(planted) for estimand (a), from 3000 events per world, is 45% (95% interval 41% to 48%, paired cluster bootstrap, B = 1000); saturation is the share of events whose last latent mid is outside [0.01, 0.99].

| quantity | planted | spec_null | flat_null |
|---|---|---|---|
| events | 3000 | 3000 | 3000 |
| saturated events, % | 83.7 | 13.9 | 1.6 |
| raw D (a), cents | +0.91 | +0.41 | -0.16 |

Does not show: how much of D on real data is surge, price level or informed trading; the split holds for this world's parameters.

<!-- /table:M4 -->

### Intervals and sample size

The bootstrap over releases covers too rarely with 25 events and close to 95% from about 100 events on; each rate in the table comes from 100 replications, so it is uncertain by up to 5 points. An interval that treats every trade as independent covers too rarely for the raw D of the planted world and too often for the world with neither effect. For the planted effect a few dozen events give high power (table M8 in `results/markouts/tables.md`).

<!-- table:M5 -->

### M5. Coverage of the population D by the cluster-bootstrap and the naive interval (synthetic data)

Reading: cluster bootstrap / naive trade-level interval, percent of 100 replications whose 95% interval contains the population value of its own world and estimand (B = 1000); at n = 200 the cluster interval covers 94% to 99% and the naive one 70% to 100% (Monte Carlo standard error at most 5.0 points). The design asked for 200 replications per world; 100 were run to stay within the 6-minute budget.

| world | estimand | n = 25 | n = 50 | n = 100 | n = 200 | n = 400 |
|---|---|---|---|---|---|---|
| planted | (a) raw D, near (0, 30] | 91 / 66 | 95 / 69 | 96 / 76 | 94 / 70 | 95 / 69 |
| planted | (b) scaled D, near (0, 30] | 88 / 86 | 90 / 91 | 95 / 90 | 96 / 93 | 95 / 94 |
| planted | (c) raw D, near (5, 30] | 92 / 71 | 96 / 77 | 97 / 72 | 95 / 72 | 94 / 67 |
| planted | (d) scaled D, near (5, 30] | 85 / 88 | 94 / 93 | 97 / 94 | 96 / 96 | 95 / 93 |
| spec_null | (a) raw D, near (0, 30] | 90 / 77 | 98 / 77 | 95 / 81 | 97 / 82 | 97 / 86 |
| spec_null | (b) scaled D, near (0, 30] | 90 / 93 | 97 / 96 | 97 / 98 | 98 / 95 | 96 / 94 |
| spec_null | (c) raw D, near (5, 30] | 90 / 79 | 96 / 82 | 96 / 82 | 97 / 86 | 95 / 87 |
| spec_null | (d) scaled D, near (5, 30] | 92 / 93 | 99 / 96 | 98 / 99 | 99 / 97 | 96 / 96 |
| flat_null | (a) raw D, near (0, 30] | 89 / 96 | 95 / 99 | 95 / 98 | 97 / 99 | 93 / 95 |
| flat_null | (b) scaled D, near (0, 30] | 87 / 99 | 95 / 99 | 93 / 100 | 99 / 100 | 92 / 97 |
| flat_null | (c) raw D, near (5, 30] | 90 / 97 | 93 / 99 | 96 / 98 | 98 / 99 | 93 / 94 |
| flat_null | (d) scaled D, near (5, 30] | 87 / 98 | 95 / 99 | 92 / 98 | 98 / 100 | 93 / 97 |

Does not show: coverage of the true D; the population value comes from 3000 events and its own error lowers the nominal 95% to about 93.4% at n = 400.

<!-- /table:M5 -->

### Quoting

Five rules quote the same synthetic markets: always on, pulled in the last minutes, a premium for adverse selection pooled over all minutes to the release (`pooled_premium`), a flat premium equal to the average of the bucket premiums over the clock of the quoting window (`flat_matched`), and a premium estimated by bucket of minutes to the release (`release_aware`). The comparison the protocol tests is `release_aware` against `flat_matched`. At that level `release_aware` earns more than `flat_matched` in both worlds (B1), the null world included, whose buckets differ too (the trade-rate surge and the price level). The two premiums are equal on average over the clock and not where the quotes are hit. `release_aware` quotes no wider on average (4.73 cents against 4.81 for `flat_matched` in the null world, and 4.97 against 5.37 in the planted world), but per unit of 4m(1 - m) it quotes more premium at its fills: `release_aware` quotes 1.60 cents at its fills against 1.25 for `flat_matched` in the null world and 2.54 against 1.65 in the planted world (B2). With the flat premium scaled to the premium at the fills of `release_aware`, `release_aware` earns less in both worlds, 0.17 and 0.38 USD per event (B10, exploratory). So the tables show neither that the planted informed share makes a premium by bucket pay, nor that its timing pays at an equal premium at the fills; which level to hold equal is a choice, and the premium at the fills is partly an outcome of the schedule.

That pair replaced a first one, `release_aware` against `pooled_premium`, after the first results and an independent review, and before any real data. The pooled premium is an average over every trade of every earlier event, larger than the average of the bucket premiums over the quoting window (2.80 and 1.65 cents in the planted world), and every policy loses money (B2), so the policy that quotes wider loses less. With 12 hours of trading `release_aware` earns less than `pooled_premium` and with 30 hours more, while the matched premium hardly moves (B8); and against `flat_matched` it earns more until that premium is scaled by about 1.5 and less beyond (B9). A second independent review found that the matched pair equalises the premium over the clock and not at the fills; the primary pair and its test were kept as they had been fixed, and B10 was added. B8 to B10 are exploratory.

<!-- table:B1 -->

### B1. `release_aware` minus `flat_matched`, net P&L per event (synthetic data)

Reading: In the null world `release_aware` earned 1.22 USD per event more than `flat_matched` (mean of 10 datasets, range +0.88 to +1.44; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10). In the planted world `release_aware` earned 2.83 USD per event more than `flat_matched` (mean of 10 datasets, range +2.05 to +3.80; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10). `flat_matched` quotes one flat premium equal to the average of the bucket premiums over the clock of the quoting window (both premiums are per unit of 4m(1 - m); the quoted spreads in USD also depend on the price, see B2). That is the same premium averaged over the clock and not the same premium where the quotes are hit: B2 gives the premium at the fills of each policy and B10 repeats the comparison at an equal premium at the fills. Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| World | Mean difference, USD per event (MC SE over datasets) | Range over datasets | 95% interval below zero | 95% interval above zero | 95% interval contains zero | Mean interval half-width, USD |
|---|---|---|---|---|---|---|
| null | +1.22 (0.05) | +0.88 to +1.44 | 0 | 10 | 0 | 0.56 |
| planted | +2.83 (0.14) | +2.05 to +3.80 | 0 | 10 | 0 | 0.95 |

Does not show: whether the planted informed share is what makes the bucket premiums useful: the null world's buckets also differ, because of the trade-rate surge and the price level; whether the difference comes from when the premium is applied or from the larger premium that `release_aware` has at its fills (B10); and whether either policy earns or loses money on real Kalshi markets: the tape, the fill rule and the quote parameters are assumptions.

<!-- /table:B1 -->

<!-- table:B2 -->

### B2. The five policies, means over the datasets of each world (synthetic data)

Reading: Net P&L per event is negative in 10 of 10 world and policy cells; the highest is `pooled_premium` in the null world (-8.19 USD) and the lowest `always_on` in the planted world (-21.64 USD). In the null world the mean quoted spread, in cents, is `release_aware` 4.73, `flat_matched` 4.81, `pooled_premium` 5.58. In the planted world the mean quoted spread, in cents, is `release_aware` 4.97, `flat_matched` 5.37, `pooled_premium` 6.99. In the null world the premium per unit of 4m(1 - m) at the fills, over the contracts filled, in cents, is `release_aware` 1.60, `flat_matched` 1.25, `pooled_premium` 1.80. In the planted world the premium per unit of 4m(1 - m) at the fills, over the contracts filled, in cents, is `release_aware` 2.54, `flat_matched` 1.65, `pooled_premium` 2.80.

| World | Policy | Net P&L per event, USD (MC SE over datasets) | Fees per event, USD | Fills per event | Contracts per event | Net cents per contract | Marked P&L per event, USD | Share of intervals with a quote | Mean quoted spread, cents | Premium at the fills, cents per unit of 4m(1 - m) |
|---|---|---|---|---|---|---|---|---|---|---|
| null | `always_on` | -16.12 (0.51) | 2.93 | 143.4 | 896.7 | -1.80 | -16.19 | 0.937 | 3.02 | 0.00 |
| null | `pull_only` | -15.69 (0.52) | 2.86 | 139.1 | 871.1 | -1.80 | -15.85 | 0.931 | 3.02 | 0.00 |
| null | `pooled_premium` | -8.19 (0.32) | 2.12 | 102.0 | 637.1 | -1.29 | -8.27 | 0.937 | 5.58 | 1.80 |
| null | `flat_matched` | -10.32 (0.33) | 2.37 | 115.1 | 719.2 | -1.44 | -10.46 | 0.937 | 4.81 | 1.25 |
| null | `release_aware` | -9.10 (0.35) | 2.20 | 104.6 | 657.0 | -1.39 | -9.22 | 0.937 | 4.73 | 1.60 |
| planted | `always_on` | -21.64 (0.54) | 2.76 | 139.8 | 875.4 | -2.47 | -20.81 | 0.917 | 3.02 | 0.00 |
| planted | `pull_only` | -21.13 (0.53) | 2.74 | 138.3 | 866.9 | -2.44 | -20.32 | 0.916 | 3.02 | 0.00 |
| planted | `pooled_premium` | -9.92 (0.26) | 1.78 | 91.0 | 566.9 | -1.75 | -9.23 | 0.917 | 6.99 | 2.80 |
| planted | `flat_matched` | -13.97 (0.29) | 2.18 | 111.6 | 696.8 | -2.00 | -13.25 | 0.917 | 5.37 | 1.65 |
| planted | `release_aware` | -11.14 (0.21) | 2.09 | 103.3 | 649.3 | -1.72 | -10.58 | 0.917 | 4.97 | 2.54 |

Does not show: which part of a loss is adverse selection: the table has no decomposition of the losses; the spread is that of the quotes posted, ask minus bid at the refreshes with both sides quoted, not the width at which fills happened; and the premium at the fills is per unit of 4m(1 - m), so in USD it also depends on the price.

<!-- /table:B2 -->

<!-- table:B8 -->

### B8. Exploratory: `release_aware` minus `flat_matched` and `release_aware` minus `pooled_premium` by hours of trading before each release, planted world (synthetic data)

Reading: Exploratory: added after the first results because B1 and the rehearsal of the protocol analysis disagreed in sign for the first primary pair; it was not part of the design fixed beforehand. With 12 hours of trading before each release `release_aware` earned 2.83 USD per event more than `flat_matched` (the primary difference; mean of 10 datasets; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10) and `release_aware` earned 1.22 USD per event less than `pooled_premium` (the secondary difference; the 95% interval lies below zero in 9 of 10 datasets and above zero in 0). With 30 hours of trading before each release `release_aware` earned 1.71 USD per event more than `flat_matched` (the primary difference; mean of 10 datasets; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10) and `release_aware` earned 1.37 USD per event more than `pooled_premium` (the secondary difference; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10). From 12 to 30 hours of trading the premiums estimated at the end move as follows: the pooled premium from 2.80 to 1.77 cents (-37%), the matched premium from 1.65 to 1.64 cents (-1%), the premium of the bucket (0,5] from 8.95 to 8.37 cents (-6%) and the premium of the bucket >360 from 0.88 to 0.86 cents (-2%). Across the windows the primary difference stays positive and the secondary difference does not keep one sign. Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| Hours of trading before each release | Datasets | Primary, `release_aware` minus `flat_matched`: mean difference, USD per event (MC SE over datasets) | Primary: 95% interval below zero | Primary: 95% interval above zero | Secondary, `release_aware` minus `pooled_premium`: mean difference, USD per event (MC SE over datasets) | Secondary: 95% interval below zero | Secondary: 95% interval above zero | Pooled premium at the end, cents | Matched premium at the end, cents | Premium of the bucket (0,5] at the end, cents | Premium of the bucket >360 at the end, cents |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 10 | +2.83 (0.14) | 0 | 10 | -1.22 (0.12) | 9 | 0 | 2.80 | 1.65 | 8.95 | 0.88 |
| 30 | 10 | +1.71 (0.10) | 0 | 10 | +1.37 (0.11) | 0 | 10 | 1.77 | 1.64 | 8.37 | 0.86 |

Does not show: what either difference is made of: B9 and B10 change the level of the flat premium and describe what happens, they do not test it; and anything about real markets: the hours of trading are those of the synthetic world.

<!-- /table:B8 -->

<!-- table:B9 -->

### B9. Exploratory: `release_aware` minus `flat_matched` by the scale of the flat premium, planted world (synthetic data)

Reading: Exploratory: added after an independent review showed that the first primary pair followed the level of the pooled premium. The flat premium of `flat_matched` is multiplied by F and F = 1 is the primary comparison. At F = 1 `release_aware` earned 2.83 USD per event more than `flat_matched` (mean of 10 datasets; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10). At the largest F = 2 `release_aware` earned 2.50 USD per event less than `flat_matched` (mean of 10 datasets; the 95% interval lies below zero in 10 of 10 datasets and above zero in 0). The mean difference changes sign at F = 1.47, by linear interpolation between the means at F = 1.25 and F = 1.5; this describes the grid and is not an estimate with an interval. The mean pooled premium at the end is 2.80 cents and the mean matched premium 1.65 cents, a ratio of 1.70, so at the end of a dataset `pooled_premium` quotes the premium that `flat_matched` quotes at F = 1.70. Every policy loses money in the planted world (B2), so a wider quote loses less. B10 gives the scale at which the two policies quote the same premium at their fills. Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| Scale F of the flat premium | Flat premium quoted at the end, cents | `release_aware` minus `flat_matched`: mean difference, USD per event (MC SE over datasets) | Range over datasets | 95% interval below zero | 95% interval above zero |
|---|---|---|---|---|---|
| 0.5 | 0.82 | +6.32 (0.17) | +5.39 to +7.31 | 0 | 10 |
| 0.75 | 1.23 | +4.52 (0.17) | +3.66 to +5.57 | 0 | 10 |
| 1 | 1.65 | +2.83 (0.14) | +2.05 to +3.80 | 0 | 10 |
| 1.25 | 2.06 | +1.24 (0.11) | +0.72 to +2.03 | 0 | 10 |
| 1.5 | 2.47 | -0.16 (0.12) | -0.78 to +0.65 | 1 | 0 |
| 1.75 | 2.88 | -1.40 (0.14) | -1.81 to -0.24 | 9 | 0 |
| 2 | 3.29 | -2.50 (0.13) | -3.00 to -1.44 | 10 | 0 |

Does not show: that either policy earns money on real markets, and that the planted informed share is what makes the bucket premiums useful at F = 1: the null world has buckets that differ too (see the null row of B1).

<!-- /table:B9 -->

<!-- table:B10 -->

### B10. Exploratory: `release_aware` minus `flat_matched` at an equal premium at the fills (synthetic data)

Reading: Exploratory: added after a second independent review showed that `flat_matched` is matched to `release_aware` on the average of the premium over the clock and not where the quotes are hit (B1, B2). In the null world `release_aware` quotes 1.60 cents per unit of 4m(1 - m) at its fills against 1.25 for `flat_matched` (+28%). With the flat premium scaled by F_eq = 1.28 both quote about the same premium at their fills (1.60 and 1.60 cents) and `release_aware` earned 0.17 USD per event less than `flat_matched` (mean of 10 datasets; the 95% interval lies below zero in 3 of 10 datasets and above zero in 0). In the planted world `release_aware` quotes 2.54 cents per unit of 4m(1 - m) at its fills against 1.65 for `flat_matched` (+54%). With the flat premium scaled by F_eq = 1.54 both quote about the same premium at their fills (2.54 and 2.54 cents) and `release_aware` earned 0.38 USD per event less than `flat_matched` (mean of 10 datasets; the 95% interval lies below zero in 1 of 10 datasets and above zero in 0). F_eq is the premium per unit of 4m(1 - m) at the fills of `release_aware` divided by that of `flat_matched`, from one run of each; `flat_matched` is then run again with its flat premium multiplied by F_eq. Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| World | Scale F_eq of the flat premium, mean (range over datasets) | Premium at the fills of `release_aware`, cents | Premium at the fills of `flat_matched` at F = 1, cents | Premium at the fills of `flat_matched` at F_eq, cents | `release_aware` minus `flat_matched` at F_eq: mean difference, USD per event (MC SE over datasets) | Range over datasets | 95% interval below zero | 95% interval above zero |
|---|---|---|---|---|---|---|---|---|
| null | 1.28 (1.26 to 1.30) | 1.60 | 1.25 | 1.60 | -0.17 (0.08) | -0.59 to +0.06 | 3 | 0 |
| planted | 1.54 (1.50 to 1.59) | 2.54 | 1.65 | 2.54 | -0.38 (0.12) | -1.04 to +0.38 | 1 | 0 |

Does not show: which level is the right one to hold fixed: the average over the clock is fixed in advance and is not what the P&L responds to, and the premium at the fills is what it responds to but is partly an outcome of the schedule, because a policy is filled less where it quotes more premium; that F_eq is known beforehand: it is estimated from the same run as the difference, and the intervals treat it as given; and anything about real markets.

<!-- /table:B10 -->

### Recalibration

A favourite-longshot slope is planted in the prices 24 hours before the release; the walk-forward recalibration is scored by the Brier skill against the raw price. Where there is no slope the recalibration scores below the raw price in every row (its Brier skill is negative). With the strongest slope planted (0.7) and 1000 events it is supported in 24% of the datasets.

<!-- table:C1 -->

### C1. Walk-forward Brier skill score by price slope and number of events (synthetic data)

Reading: mean walk-forward BSS in percent (Monte Carlo standard error in brackets) and the share of 200 replications with one-sided cluster-bootstrap p below 0.05 ("supported", B = 500); the fits use all earlier events, at least 50, refit every 10. In the null rows (beta = 1) "supported" is 0.0% to 0.5% and the mean BSS is -2.76% to -0.65%.

| beta | events | scored | mean BSS, % | population BSS, % | supported, % | mean interval width, points |
|---|---|---|---|---|---|---|
| 1.0 | 100 | 50 | -2.76 (0.35) | 0.00 | 0.5 (0.5) | 18.18 |
| 1.0 | 200 | 150 | -1.89 (0.14) | 0.00 | 0.0 (0.0) | 8.36 |
| 1.0 | 500 | 450 | -1.03 (0.05) | 0.00 | 0.0 (0.0) | 3.78 |
| 1.0 | 1000 | 950 | -0.65 (0.03) | 0.00 | 0.0 (0.0) | 1.91 |
| 0.9 | 100 | 50 | -2.97 (0.38) | 0.15 | 1.5 (0.9) | 18.97 |
| 0.9 | 200 | 150 | -1.76 (0.14) | 0.15 | 0.5 (0.5) | 8.51 |
| 0.9 | 500 | 450 | -0.81 (0.05) | 0.15 | 0.0 (0.0) | 4.00 |
| 0.9 | 1000 | 950 | -0.44 (0.03) | 0.15 | 0.0 (0.0) | 2.18 |
| 0.8 | 100 | 50 | -2.39 (0.37) | 0.68 | 1.5 (0.9) | 20.27 |
| 0.8 | 200 | 150 | -1.27 (0.17) | 0.68 | 1.0 (0.7) | 10.15 |
| 0.8 | 500 | 450 | -0.20 (0.08) | 0.68 | 3.0 (1.2) | 5.07 |
| 0.8 | 1000 | 950 | 0.05 (0.05) | 0.68 | 4.0 (1.4) | 2.77 |
| 0.7 | 100 | 50 | -1.14 (0.41) | 1.71 | 2.5 (1.1) | 22.38 |
| 0.7 | 200 | 150 | -0.33 (0.21) | 1.71 | 3.5 (1.3) | 11.44 |
| 0.7 | 500 | 450 | 0.79 (0.11) | 1.71 | 8.0 (1.9) | 5.75 |
| 0.7 | 1000 | 950 | 1.05 (0.06) | 1.71 | 24.0 (3.0) | 3.65 |

Does not show: whether real Kalshi prices have a favourite-longshot slope, or one that the logistic form fits; both are planted here (x ~ N(0, 1.5^2), price noise 0.1 on the log-odds scale).

<!-- /table:C1 -->

### The whole pipeline on one synthetic world

The rehearsal writes a world of 200 events (30 hours of trading before each release) into SQLite, reads it back and runs the same protocol steps as the real-data runner. It shows what the output looks like and that the steps fit together; it is one world and one seed. P2 comes out supported in R8, as the primary comparison does in both worlds of B1, so a supported P2 on real data would not by itself show informed trading, nor that a premium by minutes to the release pays at an equal premium at the fills (B10). The detail of P2 and of the secondary comparisons is in `results/rehearsal/tables.md`.

<!-- table:R2 -->

### R2. P1: adverse selection near the release (synthetic data)

Reading: the 5-minute adverse selection is 0.91 cents per contract near the release and 0.49 when quiet, so D = 0.42 cents [0.19, 0.66] (cluster bootstrap over releases, 10,000 replications, seed 20261113; 200 releases).

| Quantity | Value |
|---|---|
| Mean adverse selection, near (0, 30] minutes (cents per contract) | 0.91 (16,587 trades) |
| Mean adverse selection, quiet (over 360 minutes) (cents per contract) | 0.49 (46,818 trades) |
| D = near minus quiet (cents per contract), 95% interval | 0.42 [0.19, 0.66] |
| Ratio near / quiet, 95% interval | 1.85 [1.39, 2.33] |
| One-sided p (share of replications with D at or below 0) | 0.0001 |
| Holm-adjusted p (family P1, P2, P3) | 0.0003 |
| Verdict | supported |

Does not show: that D comes from informed trading: it is a difference of markouts measured on the tape.

<!-- /table:R2 -->

<!-- table:R8 -->

### R8. The tested family (synthetic data)

Reading: P1, P2 and P3 are one family; a test is supported only if its Holm-adjusted p-value is below 0.05. P2 is release_aware minus flat_matched; release_aware minus pooled_premium is one of the other pairs of the P2 table and is not tested.

| Test | One-sided p | Holm-adjusted p | Verdict |
|---|---|---|---|
| P1 | 0.0001 | 0.0003 | supported |
| P2 | 0.0001 | 0.0003 | supported |
| P3 | 0.6544 | 0.6544 | not supported |

Does not show: the size of any effect: see the intervals in the tables above; that a supported P2 shows informed trading or that a premium by minutes to the release pays: on the synthetic worlds the primary difference is positive with no informed share (table B1 of the README) and not positive at an equal premium at the fills (table B10); and the sign of the pooled comparison: it follows the level of the pooled premium.

<!-- /table:R8 -->

## Reproduce

    python3 -m venv .venv && . .venv/bin/activate && pip install -e ".[test]"
    pytest -q                                  # unit, property and end-to-end tests
    bash scripts/reproduce.sh                  # every results/ directory on synthetic data
    python3 tests/mutation_check.py --out results/mutation_check.txt
    python3 scripts/check_readme_numbers.py

`scripts/reproduce.sh` writes `results/MANIFEST.txt` (commands, seeds, commit, versions, time);
every results directory has its own manifest too. Nothing in `results/` is typed by hand, and CI
fails if a table in this README differs from `results/*/tables.md`.

The real-data run is a separate, one-time procedure with its own checks
([docs/protocol.md](docs/protocol.md), [docs/data.md](docs/data.md)). It needs a release calendar
that has been checked by hand against the official calendars, and Kalshi's Developer Agreement may
restrict storing data from its API ([docs/data.md](docs/data.md) quotes it): read it first.

    git tag -a real-run -m "before the fetch" && git push origin real-run   # freezes the code
    python3 scripts/fetch_kalshi.py --dry-run      # lists what would be fetched, requests no trade
    python3 scripts/fetch_kalshi.py                # public API to data/kalshi.sqlite (git-ignored)
    python3 scripts/run_real.py --db data/kalshi.sqlite

## Method (short)

* **World.** The price of a binary contract is a logistic function of a log-odds diffusion with the
  drift that makes the price a martingale. Each trade moves the log-odds by an amount that depends
  on the informed share of the order flow, which rises towards the release (Glosten and Milgrom
  1985); the trade rate surges at the release. Other makers quote at a fixed margin.
* **Measures.** The data are trades only, so the mid is the average of the last buy print and the
  last sell print when both are recent. Adverse selection is the 5-minute markout (cut at the
  release) by minutes to the release; the release effect D is its mean within 30 minutes of the
  release minus its mean 360 minutes or more before it.
* **Quoting.** Avellaneda and Stoikov (2008) reservation price and spread, with a premium for
  adverse selection estimated from earlier releases: pooled, flat (the average of the bucket
  premiums over the clock) or by bucket of minutes to the release. A print strictly beyond a quote
  fills it at the quote; maker fees follow Kalshi's schedule; inventory is held to resolution.
* **Inference.** Bootstrap over releases (events with the same release time are one release), Holm
  over the three tested quantities, one-sided p-values as the share of replications at or below
  zero. Recalibration is logistic, on the mid proxy 24 hours before the release, fitted
  walk-forward and scored by the Brier skill against the raw proxy.

Details and every parameter: [docs/model.md](docs/model.md).

## Limitations

* **Synthetic data only.** Every result above comes from a world that the generator builds, so a
  recovered effect shows that the estimator works in that world, not that Kalshi behaves like it.
  The real-data run has a protocol ([docs/protocol.md](docs/protocol.md)) and has not been done.
* **Trades only.** The public API gives trades, not the order book: the mid is a proxy from the
  last buy and the last sell, stale where trading is thin, and the quoted spread is never seen.
* **The release effect D is not the informed share.** It is a difference of tape markouts. A surge
  of the trade rate, the level of the price and the staleness of the mid proxy change it without a
  change of the informed share (tables M1, M3, M4); the scaled version removes the price level only.
* **The backtest does not trade against anyone.** A quote fills when a print goes strictly through
  it, at the quote; the touch rule, which fills at equality, is the optimistic bound (table B5 in
  `results/backtest/tables.md`). There is no queue, no market impact and no reaction of other
  makers, and the quote parameters were fixed before the study and not tuned. Profit figures are
  therefore about the rule, not about what an account would earn.
* **A quoting comparison follows the premium that is quoted.** A wider quote loses less, because
  every policy loses money in the backtest (table B2). The pooled premium is larger on average than
  the bucket premiums, and the sign of `release_aware` against `pooled_premium` follows that
  difference (table B8, exploratory). `flat_matched` equals the bucket premiums on average over the
  clock and not at the fills, where `release_aware` quotes 28% (null world) and 54% (planted world)
  more premium; at an equal premium at the fills `release_aware` earns less in both worlds (table
  B10, exploratory). Nothing here shows that a release-aware premium earns money, and a supported
  P2 would not show that the premium measures informed trading.
* **P2 is expected to be exploratory on real data.** The protocol labels a test exploratory when
  fewer than 100 releases enter it (P2 counts those after a burn-in of 20), and P2 also when more
  than 5% of the kept trades are off the 0.01 USD tick. The range holds about 100 scheduled
  releases, so P1 is expected at the threshold and P2 below it ([protocol](docs/protocol.md)).
* **Fees.** The maker fee is the schedule of 7 July 2026, an assumption for earlier trades.
* **Release times come from the user.** The loader writes a draft from market expiry times; the
  analysis does not accept it until it is checked against the official calendars.
* **Three tests, one-sided.** The family is small by design; P3 prices are proxies, not tradable.

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
