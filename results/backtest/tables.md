### B1. `release_aware` minus `pooled_premium`, net P&L per event (synthetic data)

Reading: In the null world `release_aware` earned 0.91 USD per event less than `pooled_premium` (mean of 10 datasets, range -1.28 to -0.60; the 95% interval lies below zero in 10 of 10 datasets and above zero in 0). In the planted world `release_aware` earned 1.22 USD per event less than `pooled_premium` (mean of 10 datasets, range -1.63 to -0.37; the 95% interval lies below zero in 9 of 10 datasets and above zero in 0). Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| World | Mean difference, USD per event (MC SE over datasets) | Range over datasets | 95% interval below zero | 95% interval above zero | 95% interval contains zero | Mean interval half-width, USD |
|---|---|---|---|---|---|---|
| null | -0.91 (0.08) | -1.28 to -0.60 | 10 | 0 | 0 | 0.44 |
| planted | -1.22 (0.12) | -1.63 to -0.37 | 9 | 0 | 1 | 0.85 |

Does not show: whether either policy earns or loses money on real Kalshi markets: the tape, the fill rule and the quote parameters are assumptions.

### B2. The four policies, means over the datasets of each world (synthetic data)

Reading: Net P&L per event is negative in 8 of 8 world and policy cells; the highest is `pooled_premium` in the null world (-8.19 USD) and the lowest `always_on` in the planted world (-21.64 USD).

| World | Policy | Net P&L per event, USD (MC SE over datasets) | Fees per event, USD | Fills per event | Contracts per event | Net cents per contract | Marked P&L per event, USD | Share of intervals with a quote |
|---|---|---|---|---|---|---|---|---|
| null | `always_on` | -16.12 (0.51) | 2.93 | 143.4 | 896.7 | -1.80 | -16.19 | 0.937 |
| null | `pull_only` | -15.69 (0.52) | 2.86 | 139.1 | 871.1 | -1.80 | -15.85 | 0.931 |
| null | `pooled_premium` | -8.19 (0.32) | 2.12 | 102.0 | 637.1 | -1.29 | -8.27 | 0.937 |
| null | `release_aware` | -9.10 (0.35) | 2.20 | 104.6 | 657.0 | -1.39 | -9.22 | 0.937 |
| planted | `always_on` | -21.64 (0.54) | 2.76 | 139.8 | 875.4 | -2.47 | -20.81 | 0.917 |
| planted | `pull_only` | -21.13 (0.53) | 2.74 | 138.3 | 866.9 | -2.44 | -20.32 | 0.916 |
| planted | `pooled_premium` | -9.92 (0.26) | 1.78 | 91.0 | 566.9 | -1.75 | -9.23 | 0.917 |
| planted | `release_aware` | -11.14 (0.21) | 2.09 | 103.3 | 649.3 | -1.72 | -10.58 | 0.917 |

Does not show: which part of a loss is adverse selection: the table has no fill-level decomposition.

### B3. Secondary comparisons of net P&L per event (synthetic data)

Reading: The largest of the 10 differences is `pooled_premium` minus `always_on` in the planted world, +11.72 USD per event; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10.

| World | Comparison | Mean difference, USD per event (MC SE over datasets) | Range over datasets | 95% interval below zero | 95% interval above zero | Mean interval half-width, USD |
|---|---|---|---|---|---|---|
| null | `release_aware` minus `always_on` | +7.02 (0.42) | +4.21 to +8.79 | 0 | 10 | 2.22 |
| null | `release_aware` minus `pull_only` | +6.59 (0.48) | +3.09 to +8.46 | 0 | 10 | 2.47 |
| null | `pooled_premium` minus `always_on` | +7.93 (0.43) | +5.29 to +9.48 | 0 | 10 | 2.22 |
| null | `pooled_premium` minus `pull_only` | +7.50 (0.49) | +4.17 to +9.15 | 0 | 10 | 2.47 |
| null | `pull_only` minus `always_on` | +0.43 (0.12) | -0.14 to +1.12 | 0 | 2 | 0.91 |
| planted | `release_aware` minus `always_on` | +10.50 (0.35) | +8.45 to +12.21 | 0 | 10 | 2.21 |
| planted | `release_aware` minus `pull_only` | +9.99 (0.36) | +8.04 to +11.81 | 0 | 10 | 2.28 |
| planted | `pooled_premium` minus `always_on` | +11.72 (0.31) | +10.07 to +13.25 | 0 | 10 | 2.22 |
| planted | `pooled_premium` minus `pull_only` | +11.21 (0.31) | +9.67 to +12.85 | 0 | 10 | 2.29 |
| planted | `pull_only` minus `always_on` | +0.52 (0.08) | +0.09 to +0.90 | 0 | 7 | 0.42 |

Does not show: a family-wise error rate: the cells are not adjusted for multiple comparisons.

### B4. Net P&L per event by planted share and maker fee, dataset seed 1 (synthetic data)

Reading: The 95% interval of the primary difference excludes zero in 6 of 10 cells; the highest net P&L per event of any policy is -5.76 USD (`pooled_premium`, planted share 0.00, maker fee coefficient 0.0000). Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| Planted share pi1 | Maker fee coefficient | `always_on`, USD per event | `pull_only`, USD per event | `pooled_premium`, USD per event | `release_aware`, USD per event | `release_aware` minus `pooled_premium`, USD per event [95% interval] |
|---|---|---|---|---|---|---|
| 0.00 | 0.0175 | -17.11 | -16.70 | -7.83 | -9.11 | -1.28 [-1.66, -0.90] |
| 0.00 | 0.0000 | -14.25 | -13.91 | -5.76 | -6.95 | -1.19 [-1.57, -0.81] |
| 0.05 | 0.0175 | -19.16 | -18.47 | -8.54 | -9.27 | -0.74 [-1.35, -0.11] |
| 0.05 | 0.0000 | -16.30 | -15.68 | -6.54 | -7.11 | -0.57 [-1.19, +0.05] |
| 0.10 | 0.0175 | -21.13 | -20.30 | -9.21 | -10.11 | -0.90 [-1.61, -0.20] |
| 0.10 | 0.0000 | -18.33 | -17.54 | -7.30 | -7.98 | -0.67 [-1.37, +0.03] |
| 0.20 | 0.0175 | -22.27 | -21.73 | -10.22 | -11.37 | -1.15 [-1.96, -0.33] |
| 0.20 | 0.0000 | -19.58 | -19.05 | -8.48 | -9.33 | -0.85 [-1.68, -0.04] |
| 0.30 | 0.0175 | -22.37 | -22.24 | -10.55 | -11.22 | -0.67 [-1.41, +0.10] |
| 0.30 | 0.0000 | -19.77 | -19.64 | -8.87 | -9.24 | -0.37 [-1.11, +0.39] |

Does not show: variation between datasets: each cell is one dataset, so differences of a few USD per event between cells can be sampling noise.

### B5. Through rule against touch rule, planted world, dataset seed 1 (synthetic data)

Reading: Filling prints at the quote raises net P&L per event by 3.43 USD for `release_aware`, the largest change; the primary difference is -0.75 USD under the touch rule against -1.15 USD under the through rule.

| Fill rule | `always_on`, USD per event | `pull_only`, USD per event | `pooled_premium`, USD per event | `release_aware`, USD per event | `release_aware` minus `pooled_premium`, USD per event [95% interval] |
|---|---|---|---|---|---|
| through (primary) | -22.27 | -21.73 | -10.22 | -11.37 | -1.15 [-1.96, -0.33] |
| touch | -19.71 | -19.18 | -7.20 | -7.94 | -0.75 [-1.52, +0.08] |

Does not show: queue position: the touch rule fills every print at the quote, so it is an upper bound on what resting at the quote can earn.

### B6. Half-width of the primary 95% interval against the number of evaluated events (synthetic data)

Reading: With the first 180 evaluated events the mean half-width is 0.85 USD per event and the interval excludes zero in 9 of 10 planted datasets. By the normal approximation, about 89 evaluated events would make the expected half-width equal the pooled mean difference (-1.22 USD per event) and about 182 would give a two-sided 5% test 80% power.

| Evaluated events used | Half-width, dataset seed 1, USD | Half-width over planted datasets, mean (range), USD | Interval excludes zero |
|---|---|---|---|
| 20 | 2.12 | 2.36 (1.30 to 3.36) | 3 of 10 |
| 40 | 1.60 | 1.76 (1.15 to 2.59) | 4 of 10 |
| 60 | 1.43 | 1.44 (1.11 to 1.90) | 5 of 10 |
| 80 | 1.38 | 1.36 (1.10 to 1.80) | 5 of 10 |
| 100 | 1.15 | 1.20 (1.06 to 1.56) | 7 of 10 |
| 120 | 1.02 | 1.08 (0.99 to 1.40) | 6 of 10 |
| 140 | 0.92 | 0.99 (0.91 to 1.26) | 5 of 10 |
| 160 | 0.89 | 0.92 (0.83 to 1.16) | 8 of 10 |
| 180 | 0.82 | 0.85 (0.77 to 1.06) | 9 of 10 |

Does not show: how many real releases are needed: the extrapolation takes the effect and the variance of this synthetic world.

### B7. Estimated premiums against their target and the latent price impact (synthetic data)

Reading: In the planted world the premium estimated at the end is 8.95 cents in the bucket (0,5] minutes and 0.88 cents in the bucket >360 (latent price impact 9.53 and 2.48); after the burn-in it is 9.79 in the first bucket; the largest gap between the estimate at the end and its target is 0.08 cents.

| World | Minutes to release | Estimate at the end, cents | Target: mean over the dataset, cents | Latent price impact, cents | Estimate after burn-in, cents |
|---|---|---|---|---|---|
| null | (0,5] | 2.17 | 2.17 | 2.48 | 2.06 |
| null | (5,15] | 2.42 | 2.41 | 2.47 | 2.20 |
| null | (15,30] | 2.53 | 2.53 | 2.48 | 2.44 |
| null | (30,60] | 2.45 | 2.44 | 2.48 | 2.44 |
| null | (60,180] | 2.07 | 2.07 | 2.48 | 2.11 |
| null | (180,360] | 1.07 | 1.07 | 2.48 | 1.05 |
| null | >360 | 0.88 | 0.87 | 2.48 | 0.89 |
| null | pooled | 1.81 | 1.80 | 2.48 | 1.77 |
| planted | (0,5] | 8.95 | 9.03 | 9.53 | 9.79 |
| planted | (5,15] | 8.81 | 8.81 | 8.45 | 7.78 |
| planted | (15,30] | 7.00 | 7.00 | 6.70 | 7.20 |
| planted | (30,60] | 4.70 | 4.69 | 4.70 | 4.81 |
| planted | (60,180] | 2.49 | 2.49 | 2.92 | 2.54 |
| planted | (180,360] | 1.07 | 1.07 | 2.48 | 1.05 |
| planted | >360 | 0.88 | 0.87 | 2.48 | 0.89 |
| planted | pooled | 2.80 | 2.80 | 3.54 | 2.80 |

Does not show: whether the premium prices the adverse selection of a fill: it is measured per trade, and the fills of a maker are not a random sample of the trades.

### B8. Exploratory: `release_aware` minus `pooled_premium` by hours of trading before each release, planted world (synthetic data)

Reading: Exploratory: added after the first results because B1 and the rehearsal of the protocol analysis disagreed in sign; it was not part of the design fixed beforehand. With 12 hours of trading before each release `release_aware` earned 1.22 USD per event less than `pooled_premium` (mean of 10 datasets; the 95% interval lies below zero in 9 of 10 datasets and above zero in 0). With 30 hours of trading before each release `release_aware` earned 1.37 USD per event more than `pooled_premium` (mean of 10 datasets; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10). The pooled premium estimated at the end is 2.80 cents with 12 hours and 1.77 cents with 30 hours. Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| Hours of trading before each release | Datasets | Mean difference, USD per event (MC SE over datasets) | Range over datasets | 95% interval below zero | 95% interval above zero | 95% interval contains zero | Mean interval half-width, USD | Pooled premium at the end, cents | Premium of the bucket (0,5] at the end, cents | Premium of the bucket >360 at the end, cents |
|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 10 | -1.22 (0.12) | -1.63 to -0.37 | 9 | 0 | 1 | 0.85 | 2.80 | 8.95 | 0.88 |
| 30 | 10 | +1.37 (0.11) | +0.95 to +1.98 | 0 | 10 | 0 | 0.72 | 1.77 | 8.37 | 0.86 |

Does not show: the mechanism: a longer window adds trades before the quoting window to the premium sample and to the history of the mid proxy together, and only the window differs between the rows.
