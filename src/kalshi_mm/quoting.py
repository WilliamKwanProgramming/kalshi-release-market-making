"""Quoting rules and the adverse-selection premium (docs/model.md section 6).

A refresh at time ``t`` has the mid proxy ``m``, the inventory ``q`` (contracts), the hours to the
release ``tau_h`` and the logit volatility ``sigma`` of the mid proxy. The reservation price and
the spread are those of Avellaneda and Stoikov (2008), with a variance ``v`` that depends on the
policy and a premium ``h``:

    r     = m - q gamma v
    delta = gamma v + (2 / gamma) ln(1 + gamma / k)
    bid   = r - delta / 2 - h,      ask = r + delta / 2 + h

The bid is rounded down and the ask up to the tick, both are kept in ``[tick, 1 - tick]`` and at
least one tick apart.

====================  ==============================================  =====================
policy                variance ``v``                                  premium ``h``
====================  ==============================================  =====================
``always_on``         ``(sigma m (1 - m))^2 tau_h``                   0
``pull_only``         as ``always_on``, no quote within 5 minutes     0
``pooled_premium``    ``(sigma m (1 - m))^2 tau_h + m (1 - m)``       ``4 m (1 - m) AS``
``release_aware``     as ``pooled_premium``                           ``4 m (1 - m) AS_bucket``
====================  ==============================================  =====================

``AS`` is the mean over the trades of earlier events of the 5-minute adverse selection divided by
``4 m (1 - m)`` (:func:`estimate_premiums`); ``AS_bucket`` is the same mean over the trades whose
minutes to the release are in the bucket of the refresh. ``m`` is the mid proxy at the refresh.
Nothing here reads a column whose name starts with ``latent_``.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from kalshi_mm.logit import logit
from kalshi_mm.markout import BUCKET_EDGES_MIN, BUCKET_LABELS
from kalshi_mm.tape import mid_proxy_asof

__all__ = [
    "POLICIES",
    "Policy",
    "Premiums",
    "QuoteParams",
    "estimate_premiums",
    "quote_prices",
    "realized_logit_vol",
]

Policy = Literal["always_on", "pull_only", "pooled_premium", "release_aware"]
POLICIES: tuple[Policy, ...] = ("always_on", "pull_only", "pooled_premium", "release_aware")

MIN_VARIANCE_SCALE = 0.05  # trades with m (1 - m) below this do not enter the premium (model.md 6)
MIN_VOL_POINTS = 6  # valid grid points needed for a realized volatility
_EPS = 1e-9  # slack for comparisons of prices and of ticks, far below any tick


@dataclass(frozen=True)
class QuoteParams:
    """Every number of the quoting rules and of the fill engine (docs/protocol.md, P2).

    ``gamma`` is the risk aversion per contract per USD, ``k`` the decay of the fill intensity per
    USD of distance from the mid. A refresh happens every ``refresh_s`` seconds and its quotes are
    live ``latency_s`` seconds later (the latency must be shorter than the refresh). Each side
    quotes ``size`` contracts and a side is not quoted when the inventory has reached
    ``max_position`` in its direction. ``pull_only`` does not quote within ``pull_minutes`` of the
    release. No quote is made when the mid proxy is below ``min_mid`` or above ``max_mid``. The
    backtest covers the last ``window_hours`` before the release; the logit volatility looks back
    ``vol_lookback_s``; the mid proxy is stale after ``stale_s``; the premium is measured at the
    markout horizon ``horizon_s``. The premium is zero until ``min_history_events`` earlier events
    exist and ``sigma_default`` is the volatility floor before any earlier event exists.
    ``maker_fee_coeff`` is the coefficient of the maker fee and ``fill_rule`` is ``"through"``
    (a print strictly beyond the quote fills it) or ``"touch"`` (a print at the quote fills it too).
    """

    gamma: float = 0.002
    k: float = 100.0
    size: float = 10.0
    max_position: float = 100.0
    refresh_s: float = 60.0
    latency_s: float = 1.0
    tick: float = 0.01
    pull_minutes: float = 5.0
    min_mid: float = 0.03
    max_mid: float = 0.97
    window_hours: float = 12.0
    vol_lookback_s: float = 7200.0
    stale_s: float = 3600.0
    horizon_s: float = 300.0
    min_history_events: int = 20
    sigma_default: float = 0.1
    maker_fee_coeff: float = 0.0175
    fill_rule: Literal["through", "touch"] = "through"

    def __post_init__(self) -> None:
        for name in ("gamma", "k", "size", "max_position", "refresh_s", "window_hours",
                     "vol_lookback_s", "horizon_s"):
            _require(_positive(getattr(self, name)), f"{name} must be positive and finite")
        for name in ("pull_minutes", "stale_s", "sigma_default", "maker_fee_coeff"):
            _require(_non_negative(getattr(self, name)), f"{name} must be non-negative and finite")
        _require(_non_negative(self.latency_s) and self.latency_s < self.refresh_s,
                 "latency_s must lie in [0, refresh_s): a quote is live before the next refresh")
        _require(_positive(self.tick) and self.tick <= 0.25
                 and abs(1.0 / self.tick - round(1.0 / self.tick)) <= 1e-9 / self.tick,
                 "tick must lie in (0, 0.25] and 1 / tick must be a whole number")
        _require(0.0 <= self.min_mid < self.max_mid <= 1.0, "need 0 <= min_mid < max_mid <= 1")
        _require(isinstance(self.min_history_events, int | np.integer)
                 and self.min_history_events >= 0, "min_history_events must be a non-negative int")
        _require(self.fill_rule in ("through", "touch"), "fill_rule must be 'through' or 'touch'")


@dataclass(frozen=True)
class Premiums:
    """Adverse-selection premiums in USD per contract per unit of ``4 m (1 - m)``.

    ``pooled`` is the mean over all minutes to the release, ``by_bucket`` has one entry per
    :data:`kalshi_mm.markout.BUCKET_LABELS` and ``n_events`` is the number of events the estimate
    comes from. The quote adds ``4 m (1 - m)`` times the premium to each half-spread.
    """

    pooled: float
    by_bucket: tuple[float, ...]
    n_events: int

    def __post_init__(self) -> None:
        _require(len(self.by_bucket) == len(BUCKET_LABELS),
                 f"by_bucket needs one premium per bucket ({len(BUCKET_LABELS)})")
        _require(all(math.isfinite(x) for x in (self.pooled, *self.by_bucket)),
                 "premiums must be finite")


def _positive(x: object) -> bool:
    return isinstance(x, int | float | np.integer | np.floating) and math.isfinite(x) and x > 0.0


def _non_negative(x: object) -> bool:
    return isinstance(x, int | float | np.integer | np.floating) and math.isfinite(x) and x >= 0.0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def estimate_premiums(
    markouts_prior: pd.DataFrame, *, min_events: int, floor: float = 0.0
) -> Premiums:
    """Premiums per unit of ``4 m (1 - m)`` from the trade markouts of earlier events.

    ``markouts_prior`` has the columns of :func:`kalshi_mm.markout.trade_markouts` (``event``,
    ``bucket``, ``mid_before``, ``adverse_selection``) and must hold the trades of earlier events
    only; the caller selects them. With ``s = mid_before (1 - mid_before)`` the normalised
    adverse selection of a trade is ``AS / (4 s)`` (docs/model.md section 6). The pooled premium is
    its mean over the trades with ``s >= 0.05`` and a defined value, and the premium of a bucket is
    the same mean over the trades of the bucket. Each is floored at ``floor``. A bucket without a
    usable trade gets the pooled premium; if no trade is usable at all, every premium is
    ``max(0, floor)``. With fewer than ``min_events`` distinct events in the frame all premiums
    are 0. ``n_events`` of the result is the number of distinct events in the frame.
    """
    n_events = int(markouts_prior["event"].nunique())
    n_buckets = len(BUCKET_LABELS)
    if n_events < min_events:
        return Premiums(0.0, (0.0,) * n_buckets, n_events)
    mid = markouts_prior["mid_before"].to_numpy(dtype=np.float64)
    adverse = markouts_prior["adverse_selection"].to_numpy(dtype=np.float64)
    bucket = markouts_prior["bucket"].to_numpy()
    variance = mid * (1.0 - mid)
    usable = (variance >= MIN_VARIANCE_SCALE) & np.isfinite(adverse) & (bucket >= 0)
    normalised = adverse[usable] / (4.0 * variance[usable])
    in_bucket = bucket[usable]
    no_information = max(0.0, floor)
    if normalised.size == 0:
        return Premiums(no_information, (no_information,) * n_buckets, n_events)
    pooled = max(floor, float(normalised.mean()))
    counts = np.bincount(in_bucket, minlength=n_buckets)
    sums = np.bincount(in_bucket, weights=normalised, minlength=n_buckets)
    by_bucket = tuple(
        max(floor, float(sums[b] / counts[b])) if counts[b] > 0 else pooled
        for b in range(n_buckets)
    )
    return Premiums(pooled, by_bucket, n_events)


def realized_logit_vol(
    ts: ArrayLike,
    side: ArrayLike,
    price: ArrayLike,
    at_ts: ArrayLike,
    *,
    lookback_s: float,
    stale_s: float,
    grid_s: float = 300.0,
) -> float | NDArray[np.float64]:
    """Realized volatility of the log-odds of the mid proxy, per square-root hour (model.md 6).

    The mid proxy (:func:`kalshi_mm.tape.mid_proxy_asof`, prints at or before the time count) is
    read on the grid ``at_ts - i grid_s`` for ``i = 0, ..., floor(lookback_s / grid_s)``. With
    ``x`` the log-odds of the proxy at the grid points that have one, the estimate is

        sqrt( mean over adjacent valid pairs of (x_{i+1} - x_i)^2  /  (grid_s / 3600) ).

    It is NaN unless at least 6 grid points are valid and one pair of adjacent grid points is
    valid; a pair that has an undefined point between its ends is not used. ``ts``, ``side`` and
    ``price`` are the sorted trades of one market. ``at_ts`` may be an array (the result has its
    shape) or a scalar (the result is a float); only prints at or before each ``at_ts`` are read.
    """
    _require(_positive(lookback_s) and _positive(grid_s), "lookback_s and grid_s must be positive")
    at = np.asarray(at_ts, dtype=np.float64)
    n_steps = int(math.floor(lookback_s / grid_s + 1e-9))
    offsets = -grid_s * np.arange(n_steps, -1, -1, dtype=np.float64)  # oldest first, at_ts last
    mid = mid_proxy_asof(ts, side, price, at[..., None] + offsets, stale_s=stale_s)
    x = logit(mid)
    valid = np.isfinite(x)
    pair = valid[..., 1:] & valid[..., :-1]
    with np.errstate(invalid="ignore"):  # an undefined point gives NaN or inf - inf in the step
        step = np.where(pair, x[..., 1:] - x[..., :-1], 0.0)
    n_pairs = pair.sum(axis=-1)
    enough = (valid.sum(axis=-1) >= MIN_VOL_POINTS) & (n_pairs > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance_per_hour = (step**2).sum(axis=-1) / n_pairs / (grid_s / 3600.0)
    vol = np.where(enough, np.sqrt(variance_per_hour), np.nan)
    return float(vol) if vol.ndim == 0 else vol


def _bucket_of(tau_minutes: float) -> int:
    """Index of the right-closed bucket of ``BUCKET_EDGES_MIN`` that holds ``tau_minutes > 0``."""
    return bisect_left(BUCKET_EDGES_MIN, tau_minutes) - 1


def _snap(ticks: float) -> float:
    """``ticks`` moved onto the nearest whole number if it is within float noise of it.

    The tolerance is that of :func:`kalshi_mm.logit.round_to_tick` (1e-9, relative 1e-14), so a
    price of 0.29 is 29 ticks and not 28.999999999999996.
    """
    whole = round(ticks)
    return float(whole) if abs(ticks - whole) <= max(1e-9, 1e-14 * abs(ticks)) else ticks


def quote_prices(
    policy: Policy,
    mid: float,
    inventory: float,
    tau_minutes: float,
    sigma_hat: float,
    premiums: Premiums,
    params: QuoteParams,
) -> tuple[float, float] | None:
    """Bid and ask of ``policy`` at one refresh (docs/model.md section 6), or None for no quote.

    ``mid`` is the mid proxy, ``inventory`` the contracts held, ``tau_minutes`` the minutes to the
    release, ``sigma_hat`` the logit volatility per square-root hour (already floored by the
    caller) and ``premiums`` the estimate of :func:`estimate_premiums`. The result is None when the
    mid is undefined or outside ``[min_mid, max_mid]`` and, for ``pull_only``, when
    ``tau_minutes <= pull_minutes`` (a refresh exactly five minutes before the release is
    inside the pull window, like the bucket "(0,5]"). A side that the inventory has closed
    (``inventory >= max_position`` for the bid, ``<= -max_position`` for the ask) is NaN.

    With ``mm = m (1 - m)``, ``v = (sigma mm)^2 tau_h`` plus ``mm`` for the premium policies,
    ``r = m - q gamma v``, ``delta = gamma v + (2 / gamma) ln(1 + gamma / k)`` and the premium
    ``h = 4 mm * AS`` (0 for the policies without one) the bid is ``r - delta / 2 - h`` rounded
    down and the ask ``r + delta / 2 + h`` rounded up to the tick, then clipped to
    ``[tick, 1 - tick]``; if that leaves them less than a tick apart the ask moves up a tick (the
    bid moves down at the upper edge).
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    if not tau_minutes > 0.0:
        raise ValueError("tau_minutes must be positive")
    if not sigma_hat >= 0.0:
        raise ValueError("sigma_hat must be non-negative and defined")
    if not params.min_mid - _EPS <= mid <= params.max_mid + _EPS:  # a NaN mid is outside
        return None
    if policy == "pull_only" and tau_minutes <= params.pull_minutes:
        return None
    mm = mid * (1.0 - mid)
    sigma_price = sigma_hat * mm
    variance = sigma_price * sigma_price * (tau_minutes / 60.0)
    premium = 0.0
    if policy == "pooled_premium":
        variance += mm
        premium = 4.0 * mm * premiums.pooled
    elif policy == "release_aware":
        variance += mm
        premium = 4.0 * mm * premiums.by_bucket[_bucket_of(tau_minutes)]
    risk = params.gamma * variance
    reservation = mid - inventory * risk
    half = 0.5 * (risk + (2.0 / params.gamma) * math.log1p(params.gamma / params.k)) + premium
    tick = params.tick
    top = round(1.0 / tick) - 1  # the tick index of 1 - tick
    k_bid = min(max(math.floor(_snap((reservation - half) / tick)), 1), top)
    k_ask = min(max(math.ceil(_snap((reservation + half) / tick)), 1), top)
    if k_ask - k_bid < 1:  # clipping made them equal
        if k_bid + 1 <= top:
            k_ask = k_bid + 1
        else:
            k_bid = k_ask - 1
    bid = round(k_bid * tick, 10)
    ask = round(k_ask * tick, 10)
    if inventory >= params.max_position:
        bid = math.nan
    if inventory <= -params.max_position:
        ask = math.nan
    return bid, ask
