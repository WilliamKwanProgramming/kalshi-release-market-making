### B1. `release_aware` minus `flat_matched`, net P&L per event (synthetic data)

Reading: In the null world `release_aware` earned 1.22 USD per event more than `flat_matched` (mean of 10 datasets, range +0.88 to +1.44; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10). In the planted world `release_aware` earned 2.83 USD per event more than `flat_matched` (mean of 10 datasets, range +2.05 to +3.80; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10). `flat_matched` quotes one flat premium equal to the average of the bucket premiums over the clock of the quoting window (both premiums are per unit of 4m(1 - m); the quoted spreads in USD also depend on the price, see B2). That is the same premium averaged over the clock and not the same premium where the quotes are hit: B2 gives the premium at the fills of each policy and B10 repeats the comparison at an equal premium at the fills. Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| World | Mean difference, USD per event (MC SE over datasets) | Range over datasets | 95% interval below zero | 95% interval above zero | 95% interval contains zero | Mean interval half-width, USD |
|---|---|---|---|---|---|---|
| null | +1.22 (0.05) | +0.88 to +1.44 | 0 | 10 | 0 | 0.56 |
| planted | +2.83 (0.14) | +2.05 to +3.80 | 0 | 10 | 0 | 0.95 |

Does not show: whether the planted informed share is what makes the bucket premiums useful: the null world's buckets also differ, because of the trade-rate surge and the price level; whether the difference comes from when the premium is applied or from the larger premium that `release_aware` has at its fills (B10); and whether either policy earns or loses money on real Kalshi markets: the tape, the fill rule and the quote parameters are assumptions.

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

### B3. Secondary comparisons of net P&L per event (synthetic data)

Reading: The largest of the 18 differences is `pooled_premium` minus `always_on` in the planted world, +11.72 USD per event; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10.

| World | Comparison | Mean difference, USD per event (MC SE over datasets) | Range over datasets | 95% interval below zero | 95% interval above zero | Mean interval half-width, USD |
|---|---|---|---|---|---|---|
| null | `release_aware` minus `pooled_premium` | -0.91 (0.08) | -1.28 to -0.60 | 10 | 0 | 0.44 |
| null | `flat_matched` minus `pooled_premium` | -2.13 (0.07) | -2.49 to -1.88 | 10 | 0 | 0.43 |
| null | `release_aware` minus `always_on` | +7.02 (0.42) | +4.21 to +8.79 | 0 | 10 | 2.22 |
| null | `release_aware` minus `pull_only` | +6.59 (0.48) | +3.09 to +8.46 | 0 | 10 | 2.47 |
| null | `pooled_premium` minus `always_on` | +7.93 (0.43) | +5.29 to +9.48 | 0 | 10 | 2.22 |
| null | `pooled_premium` minus `pull_only` | +7.50 (0.49) | +4.17 to +9.15 | 0 | 10 | 2.47 |
| null | `flat_matched` minus `always_on` | +5.79 (0.39) | +3.33 to +7.43 | 0 | 10 | 2.15 |
| null | `flat_matched` minus `pull_only` | +5.36 (0.46) | +2.21 to +7.10 | 0 | 9 | 2.42 |
| null | `pull_only` minus `always_on` | +0.43 (0.12) | -0.14 to +1.12 | 0 | 2 | 0.91 |
| planted | `release_aware` minus `pooled_premium` | -1.22 (0.12) | -1.63 to -0.37 | 9 | 0 | 0.85 |
| planted | `flat_matched` minus `pooled_premium` | -4.05 (0.09) | -4.48 to -3.62 | 10 | 0 | 0.63 |
| planted | `release_aware` minus `always_on` | +10.50 (0.35) | +8.45 to +12.21 | 0 | 10 | 2.21 |
| planted | `release_aware` minus `pull_only` | +9.99 (0.36) | +8.04 to +11.81 | 0 | 10 | 2.28 |
| planted | `pooled_premium` minus `always_on` | +11.72 (0.31) | +10.07 to +13.25 | 0 | 10 | 2.22 |
| planted | `pooled_premium` minus `pull_only` | +11.21 (0.31) | +9.67 to +12.85 | 0 | 10 | 2.29 |
| planted | `flat_matched` minus `always_on` | +7.67 (0.31) | +6.09 to +9.21 | 0 | 10 | 2.08 |
| planted | `flat_matched` minus `pull_only` | +7.16 (0.32) | +5.19 to +8.81 | 0 | 10 | 2.16 |
| planted | `pull_only` minus `always_on` | +0.52 (0.08) | +0.09 to +0.90 | 0 | 7 | 0.42 |

Does not show: the effect of the premium alone in the pairs with `always_on` or `pull_only`: `pooled_premium`, `flat_matched` and `release_aware` also add the terminal variance m(1 - m) to the variance of the reservation price, which `always_on` and `pull_only` do not, so their differences from those two policies are the effect of the premium and of a stronger inventory skew together; the pairs with `pooled_premium` have a sign that follows the level of the pooled premium (B8, B9); and a family-wise error rate: the cells are not adjusted for multiple comparisons.

### B4. Net P&L per event by planted share and maker fee, dataset seed 1 (synthetic data)

Reading: The 95% interval of the primary difference excludes zero in 10 of 10 cells; the highest net P&L per event of any policy is -5.76 USD (`pooled_premium`, planted share 0.00, maker fee coefficient 0.0000). Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| Planted share pi1 | Maker fee coefficient | `always_on`, USD per event | `pull_only`, USD per event | `pooled_premium`, USD per event | `flat_matched`, USD per event | `release_aware`, USD per event | `release_aware` minus `flat_matched`, USD per event [95% interval] |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.0175 | -17.11 | -16.70 | -7.83 | -10.32 | -9.11 | +1.21 [+0.71, +1.71] |
| 0.00 | 0.0000 | -14.25 | -13.91 | -5.76 | -7.99 | -6.95 | +1.03 [+0.55, +1.53] |
| 0.05 | 0.0175 | -19.16 | -18.47 | -8.54 | -11.49 | -9.27 | +2.21 [+1.56, +2.89] |
| 0.05 | 0.0000 | -16.30 | -15.68 | -6.54 | -9.15 | -7.11 | +2.04 [+1.38, +2.69] |
| 0.10 | 0.0175 | -21.13 | -20.30 | -9.21 | -13.00 | -10.11 | +2.89 [+2.06, +3.73] |
| 0.10 | 0.0000 | -18.33 | -17.54 | -7.30 | -10.71 | -7.98 | +2.74 [+1.90, +3.57] |
| 0.20 | 0.0175 | -22.27 | -21.73 | -10.22 | -13.84 | -11.37 | +2.47 [+1.60, +3.38] |
| 0.20 | 0.0000 | -19.58 | -19.05 | -8.48 | -11.70 | -9.33 | +2.38 [+1.49, +3.29] |
| 0.30 | 0.0175 | -22.37 | -22.24 | -10.55 | -13.56 | -11.22 | +2.33 [+1.50, +3.19] |
| 0.30 | 0.0000 | -19.77 | -19.64 | -8.87 | -11.54 | -9.24 | +2.30 [+1.48, +3.15] |

Does not show: variation between datasets: each cell is one dataset, so differences of a few USD per event between cells can be sampling noise.

### B5. Through rule against touch rule, planted world, dataset seed 1 (synthetic data)

Reading: Filling prints at the quote raises net P&L per event by 3.43 USD for `release_aware`, the largest change; the primary difference is +3.30 USD under the touch rule against +2.47 USD under the through rule.

| Fill rule | `always_on`, USD per event | `pull_only`, USD per event | `pooled_premium`, USD per event | `flat_matched`, USD per event | `release_aware`, USD per event | `release_aware` minus `flat_matched`, USD per event [95% interval] |
|---|---|---|---|---|---|---|
| through (primary) | -22.27 | -21.73 | -10.22 | -13.84 | -11.37 | +2.47 [+1.60, +3.38] |
| touch | -19.71 | -19.18 | -7.20 | -11.24 | -7.94 | +3.30 [+2.42, +4.26] |

Does not show: queue position: the touch rule fills every print at the quote, so it is an upper bound on what resting at the quote can earn.

### B6. Half-width of the primary 95% interval against the number of evaluated events (synthetic data)

Reading: With the first 180 evaluated events the mean half-width is 0.95 USD per event and the interval excludes zero in 10 of 10 planted datasets. By the normal approximation, about 21 evaluated events would make the expected half-width equal the mean difference over all planted datasets (+2.83 USD per event) and about 42 would give a two-sided 5% test 80% power.

| Evaluated events used | Half-width, dataset seed 1, USD | Half-width over planted datasets, mean (range), USD | Interval excludes zero |
|---|---|---|---|
| 20 | 2.54 | 2.63 (1.69 to 3.47) | 8 of 10 |
| 40 | 1.77 | 1.99 (1.24 to 3.24) | 7 of 10 |
| 60 | 1.46 | 1.66 (1.31 to 2.38) | 9 of 10 |
| 80 | 1.54 | 1.54 (1.14 to 2.19) | 9 of 10 |
| 100 | 1.28 | 1.34 (1.08 to 1.91) | 10 of 10 |
| 120 | 1.09 | 1.19 (0.97 to 1.63) | 10 of 10 |
| 140 | 0.97 | 1.08 (0.86 to 1.48) | 10 of 10 |
| 160 | 0.92 | 1.01 (0.80 to 1.35) | 10 of 10 |
| 180 | 0.90 | 0.95 (0.72 to 1.23) | 10 of 10 |

Does not show: how many real releases are needed: the extrapolation takes the effect and the variance of this synthetic world.

### B7. Estimated premiums against their target and the latent price impact (synthetic data)

Reading: In the planted world the premium estimated at the end is 8.95 cents in the bucket (0,5] minutes and 0.88 cents in the bucket >360 (latent price impact 9.53 and 2.48); after the burn-in it is 9.79 in the first bucket; the largest gap between the estimate at the end and its target is 0.08 cents. The pooled premium estimated at the end is 2.80 cents and the matched premium, the flat premium of `flat_matched` (the bucket premiums averaged over the decisions of the quoting window), is 1.65 cents: a ratio of 1.70 of the pooled premium to the matched premium.

| World | Minutes to release (or pooled, or matched) | Estimate at the end, cents | Target: mean over the dataset, cents | Latent price impact, cents | Estimate after burn-in, cents |
|---|---|---|---|---|---|
| null | (0,5] | 2.17 | 2.17 | 2.48 | 2.06 |
| null | (5,15] | 2.42 | 2.41 | 2.47 | 2.20 |
| null | (15,30] | 2.53 | 2.53 | 2.48 | 2.44 |
| null | (30,60] | 2.45 | 2.44 | 2.48 | 2.44 |
| null | (60,180] | 2.07 | 2.07 | 2.48 | 2.11 |
| null | (180,360] | 1.07 | 1.07 | 2.48 | 1.05 |
| null | >360 | 0.88 | 0.87 | 2.48 | 0.89 |
| null | pooled | 1.81 | 1.80 | 2.48 | 1.77 |
| null | matched | 1.25 | 1.25 | 2.48 | 1.25 |
| planted | (0,5] | 8.95 | 9.03 | 9.53 | 9.79 |
| planted | (5,15] | 8.81 | 8.81 | 8.45 | 7.78 |
| planted | (15,30] | 7.00 | 7.00 | 6.70 | 7.20 |
| planted | (30,60] | 4.70 | 4.69 | 4.70 | 4.81 |
| planted | (60,180] | 2.49 | 2.49 | 2.92 | 2.54 |
| planted | (180,360] | 1.07 | 1.07 | 2.48 | 1.05 |
| planted | >360 | 0.88 | 0.87 | 2.48 | 0.89 |
| planted | pooled | 2.80 | 2.80 | 3.54 | 2.80 |
| planted | matched | 1.65 | 1.65 | 2.86 | 1.66 |

Does not show: whether the premium prices the adverse selection of a fill: it is measured per trade, and the fills of a maker are not a random sample of the trades.

### B8. Exploratory: `release_aware` minus `flat_matched` and `release_aware` minus `pooled_premium` by hours of trading before each release, planted world (synthetic data)

Reading: Exploratory: added after the first results because B1 and the rehearsal of the protocol analysis disagreed in sign for the first primary pair; it was not part of the design fixed beforehand. With 12 hours of trading before each release `release_aware` earned 2.83 USD per event more than `flat_matched` (the primary difference; mean of 10 datasets; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10) and `release_aware` earned 1.22 USD per event less than `pooled_premium` (the secondary difference; the 95% interval lies below zero in 9 of 10 datasets and above zero in 0). With 30 hours of trading before each release `release_aware` earned 1.71 USD per event more than `flat_matched` (the primary difference; mean of 10 datasets; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10) and `release_aware` earned 1.37 USD per event more than `pooled_premium` (the secondary difference; the 95% interval lies below zero in 0 of 10 datasets and above zero in 10). From 12 to 30 hours of trading the premiums estimated at the end move as follows: the pooled premium from 2.80 to 1.77 cents (-37%), the matched premium from 1.65 to 1.64 cents (-1%), the premium of the bucket (0,5] from 8.95 to 8.37 cents (-6%) and the premium of the bucket >360 from 0.88 to 0.86 cents (-2%). Across the windows the primary difference stays positive and the secondary difference does not keep one sign. Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| Hours of trading before each release | Datasets | Primary, `release_aware` minus `flat_matched`: mean difference, USD per event (MC SE over datasets) | Primary: 95% interval below zero | Primary: 95% interval above zero | Secondary, `release_aware` minus `pooled_premium`: mean difference, USD per event (MC SE over datasets) | Secondary: 95% interval below zero | Secondary: 95% interval above zero | Pooled premium at the end, cents | Matched premium at the end, cents | Premium of the bucket (0,5] at the end, cents | Premium of the bucket >360 at the end, cents |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 12 | 10 | +2.83 (0.14) | 0 | 10 | -1.22 (0.12) | 9 | 0 | 2.80 | 1.65 | 8.95 | 0.88 |
| 30 | 10 | +1.71 (0.10) | 0 | 10 | +1.37 (0.11) | 0 | 10 | 1.77 | 1.64 | 8.37 | 0.86 |

Does not show: what either difference is made of: B9 and B10 change the level of the flat premium and describe what happens, they do not test it; and anything about real markets: the hours of trading are those of the synthetic world.

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

### B10. Exploratory: `release_aware` minus `flat_matched` at an equal premium at the fills (synthetic data)

Reading: Exploratory: added after a second independent review showed that `flat_matched` is matched to `release_aware` on the average of the premium over the clock and not where the quotes are hit (B1, B2). In the null world `release_aware` quotes 1.60 cents per unit of 4m(1 - m) at its fills against 1.25 for `flat_matched` (+28%). With the flat premium scaled by F_eq = 1.28 both quote about the same premium at their fills (1.60 and 1.60 cents) and `release_aware` earned 0.17 USD per event less than `flat_matched` (mean of 10 datasets; the 95% interval lies below zero in 3 of 10 datasets and above zero in 0). In the planted world `release_aware` quotes 2.54 cents per unit of 4m(1 - m) at its fills against 1.65 for `flat_matched` (+54%). With the flat premium scaled by F_eq = 1.54 both quote about the same premium at their fills (2.54 and 2.54 cents) and `release_aware` earned 0.38 USD per event less than `flat_matched` (mean of 10 datasets; the 95% interval lies below zero in 1 of 10 datasets and above zero in 0). F_eq is the premium per unit of 4m(1 - m) at the fills of `release_aware` divided by that of `flat_matched`, from one run of each; `flat_matched` is then run again with its flat premium multiplied by F_eq. Intervals are percentile intervals of a cluster bootstrap over events with 10,000 replications.

| World | Scale F_eq of the flat premium, mean (range over datasets) | Premium at the fills of `release_aware`, cents | Premium at the fills of `flat_matched` at F = 1, cents | Premium at the fills of `flat_matched` at F_eq, cents | `release_aware` minus `flat_matched` at F_eq: mean difference, USD per event (MC SE over datasets) | Range over datasets | 95% interval below zero | 95% interval above zero |
|---|---|---|---|---|---|---|---|---|
| null | 1.28 (1.26 to 1.30) | 1.60 | 1.25 | 1.60 | -0.17 (0.08) | -0.59 to +0.06 | 3 | 0 |
| planted | 1.54 (1.50 to 1.59) | 2.54 | 1.65 | 2.54 | -0.38 (0.12) | -1.04 to +0.38 | 1 | 0 |

Does not show: which level is the right one to hold fixed: the average over the clock is fixed in advance and is not what the P&L responds to, and the premium at the fills is what it responds to but is partly an outcome of the schedule, because a policy is filled less where it quotes more premium; that F_eq is known beforehand: it is estimated from the same run as the difference, and the intervals treat it as given; and anything about real markets.
