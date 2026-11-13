# Model

This file defines every object, estimator and rule in the repository. The code is checked against
the closed-form statements here (see `tests/`). Prices are USD per contract, between 0 and 1.
Time $t$ is in hours unless a quantity is called "minutes to release".

## 1. Contracts and the release

A release event $e$ is one announcement (a CPI, payrolls, GDP, unemployment or FOMC release) at
time $T_e$. A market on the event is a binary contract: it pays USD 1 if the statement in its
rules is true when the number is announced and USD 0 otherwise. Write $y \in \{0, 1\}$ for the
outcome and $p_t$ for the price of the YES side. One event has several markets on real Kalshi
(one per strike); the synthetic worlds have one binary market per event. Statistics always treat
the event as the unit, because all markets of one release share the same news.

Minutes to release is $\tau = 60\,(T_e - t)$. $T_e$ comes from a release calendar supplied by the
user, never from the market's close time. Trades at or after $T_e$ are dropped.

## 2. Public belief

Let $x_t = \operatorname{logit}(p_t) = \ln\frac{p_t}{1-p_t}$ be the log-odds of the public belief.
Suppose the public learns about $y$ through a signal $dZ_t = (2y-1)\,\tfrac{\sigma}{2}\,dt + dW_t$,
where $W$ is a Brownian motion. Girsanov's theorem gives the posterior log-odds
$x_t = x_0 + \sigma Z_t$, and in the public filtration

$$dx_t = \tfrac{\sigma^2}{2}\tanh\!\left(\tfrac{x_t}{2}\right)dt + \sigma\,d\hat W_t ,
\qquad \tanh(x/2) = 2p-1 ,$$

with $\hat W$ a Brownian motion. This is the logit diffusion with the drift that makes $p_t$ a
martingale; Ito's lemma on $p = 1/(1+e^{-x})$ gives drift
$p(1-p)\,\mu + \tfrac12 p(1-p)(1-2p)\sigma^2 = 0$ exactly when $\mu = \tfrac{\sigma^2}{2}(2p-1)$.
The idea of a logit-space model with a risk-neutral drift is in Dalen (2025); this repository uses
the simplest case, a diffusion between releases and one resolution at the release.

Two consequences are used by the quoting rules. The local price volatility is
$\sigma\,p(1-p)$ per square-root hour. The variance of the payoff still to be resolved at the
release is $\operatorname{Var}(y \mid p) = p(1-p)$, which is far larger than the diffusive
variance accumulated over a few hours. A market maker holding $q$ contracts through the release
carries variance $q^2 p(1-p)$.

## 3. Order flow and adverse selection

Trades arrive at rate $\lambda_T(\tau)$ per hour. A fraction $\pi(\tau)$ of trades is informed
and knows $y$: an informed taker buys YES if $y = 1$ and sells if $y = 0$. The others buy or sell
with probability 1/2. This is the model of Glosten and Milgrom (1985). Then
$P(\text{buy}\mid y=1) = (1+\pi)/2$ and $P(\text{buy}\mid y=0) = (1-\pi)/2$, so one buy
multiplies the odds of $y=1$ by $(1+\pi)/(1-\pi)$ and one sell divides them by it:

$$x \;\to\; x + s\,\ell(\pi), \qquad \ell(\pi) = \ln\frac{1+\pi}{1-\pi}, \qquad s = +1 \text{ (buy)},\; -1 \text{ (sell)}.$$

The competitive zero-profit quotes are therefore $\operatorname{sigmoid}(x \pm \ell(\pi))$, and the
expected price impact of a trade is about $\ell(\pi)\,p(1-p)$. The tape in the synthetic world
prints at quotes $\operatorname{sigmoid}(x \pm \kappa\,\ell(\pi))$ rounded outward to the 1-cent
tick, with $\kappa \ge 1$ the margin of the other makers. A maker who sells at such an ask earns
in expectation $\text{ask} - \operatorname{sigmoid}(x + \ell)$, which is zero at $\kappa = 1$
without rounding and positive for $\kappa > 1$.

Because trade directions depend only on $y$ (given $\pi$), they are conditionally independent of
the news signal given $y$. The public log-odds are exactly

$$x^{\text{pub}}_t = x_0 + \sigma Z_t + \sum_{i:\,t_i \le t} s_i\,\ell(\pi(\tau_i)),$$

and $p_t = \operatorname{sigmoid}(x^{\text{pub}}_t)$ is still a martingale: the probability that
$y = 1$ given everything public. In particular $y \sim \text{Bernoulli}(p_0)$ and outcomes are
calibrated at every time in the synthetic world.

The release effect is the profile of $\pi(\tau)$. The planted profile is
$\pi(\tau) = \pi_0 + \pi_1 e^{-\tau/\tau_I}$; the null world has $\pi_1 = 0$.

## 4. What the tape shows

The data are trades only: time, YES price, size, and the side of the taker. There are no quotes.
A taker buying YES pays the ask; a taker selling YES (buying NO) receives the bid. So the latest
buy print is a proxy for the ask and the latest sell print a proxy for the bid:

$$\hat m_t = \tfrac12\left(\text{last buy price} + \text{last sell price}\right) \quad\text{using prints before } t,$$

defined only when both sides have printed within the last 60 minutes. The proxy is noisy; the
recovery study in the README compares it with the latent mid of the synthetic world.

## 5. Markouts

For a trade $i$ at time $t_i$ with taker side $s_i$ and price $p_i$, and horizon $h$ (5 minutes
unless stated), the maker is on the other side. All quantities are USD per contract:

- effective half-spread $= s_i\,(p_i - \hat m_{t_i^-})$,
- adverse selection $\mathrm{AS}_i = s_i\,(\hat m_{t_i+h} - \hat m_{t_i^-})$, the move of the mid
  proxy in the taker's direction,
- realized spread $= s_i\,(p_i - \hat m_{t_i+h}) = $ effective half-spread $- \mathrm{AS}_i$, the
  maker's gross profit at horizon $h$,
- settlement markout $= s_i\,(p_i - y)$, the maker's gross profit if the position is held to
  resolution; it is unbiased for the maker's edge but has the variance of a binary payoff.

For trades within $h$ of the release, $\hat m_{t_i+h}$ is the last proxy before $T_e$ (the horizon is
truncated). The buckets in minutes to release are $(0,5]$, $(5,15]$, $(15,30]$, $(30,60]$,
$(60,180]$, $(180,360]$ and $(360,\infty)$. "Near" is $(0,30]$ and "quiet" is $(360,\infty)$.
Bucket means are trade-weighted over all trades in the bucket. The release effect is estimated as
the difference $D = \overline{\mathrm{AS}}_{\text{near}} - \overline{\mathrm{AS}}_{\text{quiet}}$ and, only
when the quiet mean exceeds 0.2 cent, the ratio $R = \overline{\mathrm{AS}}_{\text{near}} /
\overline{\mathrm{AS}}_{\text{quiet}}$.

## 6. Quoting

Quotes are refreshed every 60 seconds and become live 1 second later. A refresh at time $t$ uses
only prints up to $t$. Let $m = \hat m_t$, $q$ the inventory in contracts, $\tau_h = T_e - t$ in
hours. The Avellaneda and Stoikov (2008) reservation price and spread are

$$r = m - q\,\gamma\,v, \qquad \delta = \gamma\,v + \tfrac{2}{\gamma}\ln\!\left(1+\tfrac{\gamma}{k}\right),$$

with bid $= r - \delta/2 - h_{\text{adv}}$ and ask $= r + \delta/2 + h_{\text{adv}}$, rounded outward
to the tick. Here $\gamma$ is the risk aversion per contract and $k$ the decay of the fill
intensity with distance from the mid (both fixed in advance, see `docs/protocol.md`), and $v$ is
the variance still to be borne. The policies differ in $v$ and in the premium $h_{\text{adv}}$:

| policy | variance $v$ | premium $h_{\text{adv}}$ | quotes |
|---|---|---|---|
| `always_on` | $\sigma_p^2\tau_h$ with $\sigma_p = \hat\sigma\,m(1-m)$ | 0 | always |
| `pull_only` | as `always_on` | 0 | none within 5 minutes of the release |
| `pooled_premium` | $\sigma_p^2\tau_h + m(1-m)$ | $\widehat{\mathrm{AS}}$ pooled over all minutes to release | always |
| `release_aware` | $\sigma_p^2\tau_h + m(1-m)$ | $\widehat{\mathrm{AS}}$ of the bucket containing $\tau$ | always |

$\hat\sigma$ is the realized logit volatility of the mid proxy over the previous 2 hours of the
same market, floored at the walk-forward average of earlier events. $\widehat{\mathrm{AS}}$ is the
mean 5-minute adverse selection in earlier events only, floored at zero. Until 20 earlier events
exist, the premium is zero. The primary comparison is `release_aware` against `pooled_premium`,
which differ only in whether the premium depends on minutes to release. The others are secondary.

## 7. Fills, fees and settlement

The backtest uses the tape only. A resting bid is filled by a taker sale printing strictly below
the bid; a resting ask by a taker purchase printing strictly above the ask; the fill is at the
quote, for at most the quote size and the trade size. Prints at the quote are ignored because the
queue position is unknown (the touch rule, filling at equality, is run as a sensitivity). The tape
does not react to the maker's quotes. Quote size is 10 contracts per side and no side is quoted
beyond a net position of 100 contracts. Inventory is held to resolution: at $T_e$ each contract
pays $y$.

Kalshi's schedule of 7 July 2026 charges, per fill of $C$ contracts at price $P$, a taker fee
$\lceil 0.07\,C\,P(1-P)\rceil$ and a maker fee $\lceil 0.0175\,C\,P(1-P)\rceil$, rounded up to
the next 0.0001 USD, with multiplier 1 for the CPI, payrolls, GDP, unemployment, Fed-rate and
Fed-decision series. The maker fee is charged when a resting order trades. The settlement fee is
zero. The code takes the maker coefficient as a parameter: 0.0175 in the primary runs and 0 as
a sensitivity. The per-order rounding accumulator and rebate in Kalshi's documentation can only
lower fees and are not modelled.

Net P&L of an event is $\sum (\text{cash from fills}) + \text{inventory} \times y - \text{fees}$.

## 8. Calibration

The favourite-longshot bias is the pattern that low-priced contracts win less often than their
price and high-priced contracts more often (Snowberg and Wolfers, 2010; Bürgi, Deng and Whelan,
2025, report it on Kalshi). A price $p$ is recalibrated by $\hat p = \operatorname{sigmoid}(a + b\,
\operatorname{logit} p)$ with $(a,b)$ fitted by logistic regression of $y$ on $\operatorname{logit} p$
over earlier events only (expanding window, at least 50 events). The score is the Brier score
(Brier, 1950) $\frac1n\sum(\hat p - y)^2$ and the skill is $\mathrm{BSS} = 1 - \mathrm{Brier}_{\text{recal}} /
\mathrm{Brier}_{\text{raw}}$. Positive values mean that recalibration helps out of sample.

## 9. Inference

All intervals resample whole events with replacement (the cluster bootstrap; Cameron, Gelbach and
Miller, 2008, motivate resampling clusters when observations within a cluster are dependent).
Percentile intervals, 95%, with the number of replications stated next to each result. Paired
comparisons resample the per-event differences. Families of tests use the Holm (1979) step-down
adjustment. A naive trade-level bootstrap is shown alongside in the recovery study to show how
much narrower it is.

## 10. What the synthetic studies can and cannot show

The worlds are built from the model above, so a method that uses the model's structure will do
well in them. The studies show whether the code recovers planted effects, how wide the intervals
are, how many releases are needed, and where the effect disappears (the null world and the
break-even sweep). They show nothing about whether adverse selection near releases exists on
Kalshi. That is a question for real data, with the protocol in `docs/protocol.md`; it has not
been run.

## References

See the README.
