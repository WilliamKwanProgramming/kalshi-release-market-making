### R1. Sample (synthetic data)

Reading: 200 releases, 200 markets and 100,213 trades enter the analysis.

| Step | Count |
|---|---|
| Events | 200 |
| Releases | 200 |
| Markets | 200 |
| Markets without a settled outcome | 0 |
| Markets with fewer than 50 trades | 0 |
| Markets without a mid proxy in the quoting window | 0 |
| Markets kept | 200 |
| Releases kept | 200 |
| Trades kept | 100,213 |

Does not show: whether the markets that the rules removed differ from the ones kept.

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

### R3. P1: adverse selection by minutes to release (synthetic data)

Reading: mean 5-minute adverse selection (cents per contract, signed for the taker) in each bucket of minutes to the release; the standard error treats releases as clusters.

| Minutes to release | Trades | Releases | Mean (cents) | SE (cents) |
|---|---|---|---|---|
| (0,5] | 3,357 | 200 | 0.52 | 0.19 |
| (5,15] | 6,015 | 200 | 0.73 | 0.15 |
| (15,30] | 7,215 | 200 | 1.24 | 0.16 |
| (30,60] | 10,607 | 200 | 1.04 | 0.10 |
| (60,180] | 16,732 | 200 | 0.79 | 0.06 |
| (180,360] | 8,094 | 200 | 0.41 | 0.03 |
| >360 | 46,818 | 200 | 0.49 | 0.02 |

Does not show: the jump at the release itself: the markout of a trade in the last 5 minutes is cut at the release, so it covers less than 5 minutes.

### R4. P1: reported, not tested (synthetic data)

Reading: D (cents per contract, 95% interval) for each series, at the other horizons, for the settlement markout, for the markout divided by 4 m (1 - m) (m is the mid proxy before the trade, so the unit is cents per contract at a price of 50 cents) and with a near window that never cuts the horizon; none of these is part of the tested family.

| Slice | D [interval] | Near mean | Quiet mean | Trades near / quiet |
|---|---|---|---|---|
| series SYN | 0.42 [0.19, 0.66] | 0.91 | 0.49 | 16,587 / 46,818 |
| horizon 1 minute | 0.58 [0.40, 0.78] | 0.71 | 0.13 | 16,587 / 46,902 |
| horizon 15 minute | -0.05 [-0.31, 0.23] | 0.97 | 1.02 | 16,587 / 46,682 |
| settlement markout | -0.21 [-0.56, 0.15] | 0.73 | 0.94 | 16,587 / 48,168 |
| scaled to a price of 50 cents (mid proxy in [0.03, 0.97]) | 6.21 [5.09, 7.44] | 7.05 | 0.85 | 4,234 / 41,700 |
| near window (5, 30] minutes | 0.51 [0.28, 0.78] | 1.01 | 0.49 | 13,230 / 46,818 |

Does not show: a test: the slices are listed in docs/protocol.md, but only the 5-minute D is tested. A level of the price and a surge of the trade rate both change D without a change of the informed share, which is why the scaled slice is shown.

### R5. P2: the four quoting policies (synthetic data)

Reading: mean over the evaluated releases of the sum over each release's markets; net P&L is settled at the outcome and after maker fees.

| Policy | Net P&L (USD) | Gross (USD) | Fees (USD) | Fills | Contracts | Net cents per contract filled | Share of intervals quoted |
|---|---|---|---|---|---|---|---|
| always_on | -10.855 | -9.368 | 1.487 | 85.8 | 541.0 | -2.01 | 0.71 |
| pull_only | -10.370 | -8.893 | 1.477 | 85.0 | 536.6 | -1.93 | 0.71 |
| pooled_premium | -6.919 | -5.773 | 1.146 | 65.8 | 413.2 | -1.67 | 0.71 |
| release_aware | -5.868 | -4.759 | 1.110 | 61.5 | 388.3 | -1.51 | 0.71 |

Does not show: queue position, market impact and the rest of the book: a fill is a strict trade-through of a tape that does not react to the quotes.

### R6. P2: net P&L difference per release (USD) (synthetic data)

Reading: release_aware minus pooled_premium is 1.051 USD per release [0.423, 1.733] over 180 evaluated releases (cluster bootstrap over releases, 10,000 replications, seed 20261113).

| Difference | Mean (USD) | 95% interval | One-sided p | Releases |
|---|---|---|---|---|
| release_aware minus pooled_premium (primary) | 1.051 | [0.423, 1.733] | 0.0001 (tested) | 180 |
| always_on minus pooled_premium | -3.935 | [-5.387, -2.524] | 1.0000 | 180 |
| always_on minus release_aware | -4.986 | [-6.584, -3.430] | 1.0000 | 180 |
| pull_only minus always_on | 0.485 | [0.185, 0.876] | 0.0001 | 180 |
| pull_only minus pooled_premium | -3.450 | [-4.933, -1.974] | 1.0000 | 180 |
| pull_only minus release_aware | -4.501 | [-6.095, -2.943] | 1.0000 | 180 |
| release_aware minus pooled_premium, touch fill rule | 1.114 | [0.601, 1.669] | 0.0001 | 180 |
| release_aware minus pooled_premium, maker fee coefficient 0 | 1.014 | [0.392, 1.689] | 0.0001 | 180 |
| release_aware minus pooled_premium, first half | 0.706 | [-0.045, 1.513] | 0.0314 | 90 |
| release_aware minus pooled_premium, second half | 1.395 | [0.398, 2.547] | 0.0015 | 90 |
| release_aware minus pooled_premium, series SYN | 1.051 | [0.423, 1.733] | 0.0001 | 180 |

Does not show: which policy earns money: a difference can be positive while both policies lose.

### R7. P3: walk-forward recalibration (synthetic data)

Reading: BSS = -0.0051 [-0.0288, 0.0199] over 148 markets (cluster bootstrap over releases, 10,000 replications, seed 20261113); positive means that the recalibration helps out of sample.

| Quantity | Value |
|---|---|
| BSS | -0.0051 [-0.0288, 0.0199] |
| Brier score, raw / recalibrated | 0.1629 / 0.1637 |
| One-sided p (share of replications with BSS at or below 0) | 0.6544 |
| Holm-adjusted p (family P1, P2, P3) | 0.6544 |
| Verdict | not supported |
| Markets predicted / releases | 148 / 148 |
| Markets without a price at 24 hours before the release | 2 |
| Coefficients (a, b) of the first fit / the last fit | (0.115, 0.921) / (0.129, 1.093) |
| Markets in the first fit / the last fit | 50 / 190 |
| BSS, series SYN | -0.0051 [-0.0288, 0.0199] |

Does not show: whether a trader could have used the recalibration: the prices are mid proxies from prints, not prices that anyone could have traded at.

### R8. The tested family (synthetic data)

Reading: P1, P2 and P3 are one family; a test is supported only if its Holm-adjusted p-value is below 0.05.

| Test | One-sided p | Holm-adjusted p | Verdict |
|---|---|---|---|
| P1 | 0.0001 | 0.0003 | supported |
| P2 | 0.0001 | 0.0003 | supported |
| P3 | 0.6544 | 0.6544 | not supported |

Does not show: the size of any effect: see the intervals in the tables above.
