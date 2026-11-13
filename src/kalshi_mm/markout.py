"""Markouts of the trades of a tape and the release effect (docs/model.md section 5).

For a trade ``i`` at time ``t`` with taker side ``s`` (+1 the taker bought YES, -1 sold), price
``p``, horizon ``h`` and release time ``T`` (all prices in USD per contract), with the mid proxy
``m`` of :func:`kalshi_mm.tape.mid_proxy_asof`:

    mid_before          = m from the prints strictly before t
    mid_after           = m at min(t + h, T) from the prints at or before that time and strictly
                          before T (the horizon is cut at the release)
    eff_half_spread     = s (p - mid_before)
    adverse_selection   = s (mid_after - mid_before)
    realized_spread     = s (p - mid_after) = eff_half_spread - adverse_selection
    settlement_markout  = s (p - y)           y = 1 if the market settled YES, else 0

A value is NaN where a proxy it needs is undefined or the outcome is missing. Trades at or after
the release are dropped. The buckets are right-closed intervals of the minutes to the release
``tau_min = (T - t) / 60``: (0,5], (5,15], (15,30], (30,60], (60,180], (180,360], (360,inf).

The release effect is ``D = mean(AS | near) - mean(AS | quiet)``, each mean taken over all trades
in its window, with a percentile interval from the cluster bootstrap over events (section 9).
Nothing here reads a column whose name starts with ``latent_``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from kalshi_mm.stats import cluster_weights, p_at_or_below_zero, percentile_interval, ratio_of_sums
from kalshi_mm.tape import mid_proxy_asof

__all__ = [
    "BUCKET_EDGES_MIN",
    "BUCKET_LABELS",
    "bucket_index",
    "bucket_means",
    "per_event_sums",
    "release_effect",
    "trade_markouts",
]

BUCKET_EDGES_MIN = (0.0, 5.0, 15.0, 30.0, 60.0, 180.0, 360.0, np.inf)
BUCKET_LABELS = ("(0,5]", "(5,15]", "(15,30]", "(30,60]", "(60,180]", "(180,360]", ">360")


def bucket_index(tau_min: ArrayLike) -> NDArray[np.intp]:
    """Index into :data:`BUCKET_LABELS` of each time to release in minutes.

    The intervals are right-closed, so 5.0 belongs to "(0,5]" and 5.0000001 to "(5,15]". A time
    of zero or less, and NaN, belong to no bucket and give -1.
    """
    tau = np.asarray(tau_min, dtype=np.float64)
    index = np.searchsorted(np.asarray(BUCKET_EDGES_MIN), tau, side="left") - 1
    return np.where(np.isnan(tau), -1, index)[()]  # NaN sorts last; tau <= 0 is already -1


def _lookup(frame: pd.DataFrame, key: str, value: str, keys: pd.Series, what: str) -> np.ndarray:
    """``frame[value]`` at the rows whose ``frame[key]`` equals each entry of ``keys``, as floats.

    A key that is missing from the frame, or listed twice in it, is an error.
    """
    index = pd.Index(frame[key])
    if index.has_duplicates:
        raise ValueError(f"{what} lists a {key} more than once")
    position = index.get_indexer(keys)
    if np.any(position < 0):
        raise ValueError(f"the tape has trades whose {key} is not in {what}")
    values = pd.to_numeric(frame[value], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    return values[position]


def trade_markouts(
    tape: pd.DataFrame,
    events: pd.DataFrame,
    markets: pd.DataFrame,
    *,
    horizon_s: float = 300.0,
    stale_s: float = 3600.0,
) -> pd.DataFrame:
    """One row per trade before its release, with the markouts of the module docstring.

    ``tape`` follows the data contract and must be sorted by ``(market, ts, seq)``; ``events`` gives
    ``release_ts`` per ``event`` and ``markets`` gives ``outcome`` (0, 1 or missing) per ``market``.
    The result keeps the index and every column of ``tape`` that does not start with ``latent_``
    and adds ``tau_min``, ``bucket`` (the index into :data:`BUCKET_LABELS`), ``mid_before``,
    ``mid_after``, ``eff_half_spread``, ``adverse_selection``, ``realized_spread`` and
    ``settlement_markout``. ``stale_s`` is the staleness limit of the mid proxy.
    """
    if not horizon_s > 0.0:
        raise ValueError("horizon_s must be positive")
    missing = [c for c in ("event", "market", "ts", "seq", "price", "side") if c not in tape]
    if missing:
        raise ValueError(f"the tape lacks the columns {missing}")
    n = len(tape)
    ts = tape["ts"].to_numpy(dtype=np.float64)
    price = tape["price"].to_numpy(dtype=np.float64)
    side = tape["side"].to_numpy()
    seq = tape["seq"].to_numpy()
    codes, _ = pd.factorize(tape["market"], sort=True)
    if not np.array_equal(np.lexsort((seq, ts, codes)), np.arange(n)) or np.any(codes < 0):
        raise ValueError("the tape must be sorted by market, ts and seq, without missing markets")

    release = _lookup(events, "event", "release_ts", tape["event"], "events")
    outcome = _lookup(markets, "market", "outcome", tape["market"], "markets")
    if np.isnan(release).any():
        raise ValueError("an event with trades has no release_ts")

    mid_before = np.full(n, np.nan)
    mid_after = np.full(n, np.nan)
    if n:
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        for a, b in zip(starts, np.r_[starts[1:], n], strict=True):
            T = release[a]
            if not np.all(release[a:b] == T):
                raise ValueError("a market has trades of events with different release times")
            rows = np.arange(a, b)[ts[a:b] < T]  # prints before the release
            if rows.size == 0:
                continue
            t, s, p = ts[rows], side[rows], price[rows]
            mid_before[rows] = mid_proxy_asof(t, s, p, t, stale_s=stale_s, inclusive=False)
            mid_after[rows] = mid_proxy_asof(t, s, p, np.minimum(t + horizon_s, T), stale_s=stale_s)

    tau_min = (release - ts) / 60.0
    keep = tau_min > 0.0
    columns = [c for c in tape.columns if not str(c).startswith("latent_")]
    sgn = side[keep].astype(np.float64)
    before, after = mid_before[keep], mid_after[keep]
    return tape.loc[keep, columns].assign(
        tau_min=tau_min[keep],
        bucket=bucket_index(tau_min[keep]),
        mid_before=before,
        mid_after=after,
        eff_half_spread=sgn * (price[keep] - before),
        adverse_selection=sgn * (after - before),
        realized_spread=sgn * (price[keep] - after),
        settlement_markout=sgn * (price[keep] - outcome[keep]),
    )


def per_event_sums(markouts: pd.DataFrame, mask: ArrayLike, column: str) -> pd.DataFrame:
    """Per event: the sum and the number of non-NaN values of ``column`` over the rows in ``mask``.

    The result has one row for every event that occurs in ``markouts`` (sorted by event id), with
    columns ``event``, ``sum`` and ``n``; an event with no selected value has ``sum = 0`` and
    ``n = 0``. Every call on the same ``markouts`` lists the same events in the same order, so the
    outputs for different masks or columns line up row by row.
    """
    values = markouts[column].to_numpy(dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != values.shape:
        raise ValueError("mask must have one entry per row of markouts")
    codes, events = pd.factorize(markouts["event"], sort=True)
    keep = selected & ~np.isnan(values)
    return pd.DataFrame(
        {
            "event": np.asarray(events),
            "sum": np.bincount(codes[keep], weights=values[keep], minlength=len(events)),
            "n": np.bincount(codes[keep], minlength=len(events)),
        }
    )


def _cluster_se(sums: NDArray[np.float64], counts: NDArray[np.int64], mean: float) -> float:
    """Standard error of ``sum(sums) / sum(counts)`` when the events are the clusters:
    ``sqrt(G / (G - 1) * sum_g (sums_g - mean * counts_g)^2) / sum(counts)`` over all ``G`` events
    (an event with no value in the domain contributes zero but still counts in ``G``)."""
    g = len(sums)
    if g < 2:
        return float("nan")
    residual = sums - mean * counts
    return float(np.sqrt(g / (g - 1.0) * np.sum(residual**2)) / counts.sum())


def bucket_means(markouts: pd.DataFrame, column: str = "adverse_selection") -> pd.DataFrame:
    """Mean of ``column`` in each of the seven buckets, trade-weighted, NaN values dropped.

    One row per bucket, in the order of :data:`BUCKET_LABELS`, with ``bucket`` (the index),
    ``label``, ``n`` (values used), ``n_events`` (events with at least one value), ``mean``,
    ``sd`` (standard deviation over trades) and ``se``. The standard error treats the event as the
    cluster, because the trades of one event are dependent: it is
    ``sqrt(G / (G - 1) * sum_g (A_g - mean * n_g)^2) / n`` with ``A_g`` the sum and ``n_g`` the
    count of event ``g``, over all ``G`` events of ``markouts``. It is larger than ``sd / sqrt(n)``
    when the values of an event are positively correlated. Mean, ``sd`` and ``se`` are NaN for an
    empty bucket, ``sd`` for one value and ``se`` for fewer than two events in ``markouts``.
    """
    values = markouts[column].to_numpy(dtype=np.float64)
    bucket = markouts["bucket"].to_numpy()
    rows = []
    for k, label in enumerate(BUCKET_LABELS):
        in_bucket = bucket == k
        sums = per_event_sums(markouts, in_bucket, column)
        counts = sums["n"].to_numpy()
        n = int(counts.sum())
        mean = sd = se = float("nan")
        if n > 0:
            mean = float(sums["sum"].sum() / n)
            se = _cluster_se(sums["sum"].to_numpy(), counts, mean)
        if n > 1:
            sd = float(np.std(values[in_bucket & ~np.isnan(values)], ddof=1))
        rows.append(
            {"bucket": k, "label": label, "n": n, "n_events": int((counts > 0).sum()),
             "mean": mean, "sd": sd, "se": se}
        )
    return pd.DataFrame(rows)


def release_effect(
    markouts: pd.DataFrame,
    *,
    near: tuple[float, float] = (0.0, 30.0),
    quiet: tuple[float, float] = (360.0, np.inf),
    column: str = "adverse_selection",
    B: int = 10_000,
    seed: int = 20261113,
    level: float = 0.95,
    min_quiet_for_ratio: float = 0.002,
) -> dict:
    """The release effect ``D = mean_near - mean_quiet`` with a cluster-bootstrap interval.

    ``near`` and ``quiet`` are windows ``(lo, hi]`` of ``tau_min``; each mean is over all non-NaN
    values of ``column`` in its window. The bootstrap resamples the events of ``markouts`` (all
    events that have a row there): with ``w = cluster_weights(n_events, B, default_rng(seed))``,
    events in sorted id order, replication ``b`` has the near mean
    ``w[b] @ sum_near / w[b] @ n_near`` and the quiet mean likewise, from the same weights. The
    interval is the percentile interval at ``level`` and ``p_one_sided`` is the share of
    replications with ``D <= 0``, never below 1 / (B + 1) (docs/protocol.md).

    The ratio ``mean_near / mean_quiet`` and its interval are reported only when ``mean_quiet``
    exceeds ``min_quiet_for_ratio``, otherwise both are None. A replication in which a window has
    no value (or the ratio has a zero denominator) is dropped from the intervals and the p-value
    and counted in ``B_undefined``; with many events it is 0.

    Returns ``mean_near``, ``mean_quiet``, ``D``, ``D_ci``, ``ratio``, ``ratio_ci``,
    ``p_one_sided``, ``n_events`` (clusters), ``n_near`` and ``n_quiet`` (values), ``B``,
    ``B_undefined`` and ``seed``.
    """
    if B < 1:
        raise ValueError("B must be at least 1")
    for name, window in (("near", near), ("quiet", quiet)):
        if not window[0] < window[1]:
            raise ValueError(f"{name} must be a window (lo, hi] with lo < hi")
    tau = markouts["tau_min"].to_numpy(dtype=np.float64)
    near_sums = per_event_sums(markouts, (tau > near[0]) & (tau <= near[1]), column)
    quiet_sums = per_event_sums(markouts, (tau > quiet[0]) & (tau <= quiet[1]), column)
    n_near, n_quiet = int(near_sums["n"].sum()), int(quiet_sums["n"].sum())
    if n_near == 0 or n_quiet == 0:
        raise ValueError("the near and the quiet window each need at least one value")
    n_events = len(near_sums)
    mean_near = float(near_sums["sum"].sum() / n_near)
    mean_quiet = float(quiet_sums["sum"].sum() / n_quiet)
    with_ratio = mean_quiet > min_quiet_for_ratio

    weights = cluster_weights(n_events, B, np.random.default_rng(seed))
    near_b = ratio_of_sums(near_sums["sum"].to_numpy(), near_sums["n"].to_numpy(), weights)
    quiet_b = ratio_of_sums(quiet_sums["sum"].to_numpy(), quiet_sums["n"].to_numpy(), weights)
    diff = near_b - quiet_b
    usable = np.isfinite(diff)
    if with_ratio:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_b = near_b / quiet_b
        usable &= np.isfinite(ratio_b)
    if not usable.any():
        raise ValueError("no bootstrap replication has both windows defined; too few events")

    return {
        "mean_near": mean_near,
        "mean_quiet": mean_quiet,
        "D": mean_near - mean_quiet,
        "D_ci": percentile_interval(diff[usable], level),
        "ratio": mean_near / mean_quiet if with_ratio else None,
        "ratio_ci": percentile_interval(ratio_b[usable], level) if with_ratio else None,
        "p_one_sided": p_at_or_below_zero(diff[usable]),
        "n_events": n_events,
        "n_near": n_near,
        "n_quiet": n_quiet,
        "B": int(B),
        "B_undefined": int(B - usable.sum()),
        "seed": int(seed),
    }
