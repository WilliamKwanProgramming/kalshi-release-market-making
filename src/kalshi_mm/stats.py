"""Cluster bootstrap and multiple-testing helpers (docs/model.md section 9).

All intervals resample whole events with replacement. One bootstrap replication is a row of
multinomial counts ``w`` with ``sum(w) = n_clusters``: event ``g`` appears ``w[g]`` times. A
statistic that is a ratio of sums over events (a trade-weighted mean, for example) is then
``(w @ numerator) / (w @ denominator)`` with one numerator and one denominator per event.

* :func:`cluster_weights`: the matrix of resampling counts (Cameron, Gelbach and Miller 2008
  motivate resampling clusters when observations within a cluster are dependent).
* :func:`ratio_of_sums`: the ratio statistic for every replication.
* :func:`percentile_interval`: the percentile interval of the replications.
* :func:`p_at_or_below_zero`: the one-sided p-value used by the protocol, the share of replications
  at or below zero, never below ``1 / (B + 1)``.
* :func:`holm_adjust`: the Holm (1979) step-down adjustment of a family of p-values.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "cluster_weights",
    "holm_adjust",
    "p_at_or_below_zero",
    "percentile_interval",
    "ratio_of_sums",
]


def cluster_weights(n_clusters: int, B: int, rng: np.random.Generator) -> NDArray[np.int64]:
    """Resampling counts for ``B`` cluster-bootstrap replications.

    Returns an integer array of shape ``(B, n_clusters)``. Row ``b`` is one draw from the
    multinomial distribution with ``n_clusters`` trials and equal cell probabilities, so it sums
    to ``n_clusters``: it counts how often each cluster is drawn when ``n_clusters`` clusters are
    drawn with replacement.
    """
    if n_clusters < 1 or B < 1:
        raise ValueError("n_clusters and B must be at least 1")
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=B)


def percentile_interval(samples: ArrayLike, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval ``(lo, hi)`` of bootstrap replications at confidence ``level``.

    The end points are the ``(1 - level) / 2`` and ``(1 + level) / 2`` sample quantiles
    (numpy's default linear interpolation). NaN replications are an error, not ignored.
    """
    x = _replications(samples)
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1")
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(x, [a, 1.0 - a])
    return float(lo), float(hi)


def p_at_or_below_zero(samples: ArrayLike) -> float:
    """One-sided bootstrap p-value: the share of replications that are at or below zero.

    This is the level at which the one-sided percentile interval stops excluding zero. It is
    never below ``1 / (B + 1)`` for ``B`` replications (docs/protocol.md). NaN replications are an
    error.
    """
    x = _replications(samples)
    return float(max(np.mean(x <= 0.0), 1.0 / (x.size + 1.0)))


def holm_adjust(pvalues: ArrayLike) -> NDArray[np.float64]:
    """Holm (1979) step-down adjusted p-values, in the order of the input.

    With ``m`` p-values sorted as ``p_(1) <= ... <= p_(m)`` the adjusted values are
    ``max_{j <= k} min(1, (m - j + 1) p_(j))``: capped at 1 and non-decreasing along the sorted
    order. Rejecting the hypotheses with adjusted p-value below ``a`` controls the family-wise
    error rate at ``a``. Every input must lie in [0, 1]; NaN is an error, because a test that is
    left out of the family would silently make the others less strict.
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p, kind="stable")
    stepped = np.minimum(1.0, (m - np.arange(m)) * p[order])
    adjusted = np.empty(m)
    adjusted[order] = np.maximum.accumulate(stepped)
    return adjusted


def ratio_of_sums(
    num: ArrayLike, den: ArrayLike, weights: ArrayLike
) -> NDArray[np.float64]:
    """``(weights @ num) / (weights @ den)``: the ratio of sums for every bootstrap row.

    ``num`` and ``den`` have one entry per cluster, ``weights`` has shape ``(B, n_clusters)``
    (from :func:`cluster_weights`). The result has length ``B``; a replication whose resampled
    denominator is zero gives NaN.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    w = np.asarray(weights)
    if num.ndim != 1 or num.shape != den.shape:
        raise ValueError("num and den must be one-dimensional and of equal length")
    if w.ndim != 2 or w.shape[1] != num.size:
        raise ValueError("weights must have shape (B, n_clusters)")
    top = w @ num
    bottom = w @ den
    out = np.full(top.shape, np.nan)
    np.divide(top, bottom, out=out, where=bottom != 0.0)
    return out


def _replications(samples: ArrayLike) -> NDArray[np.float64]:
    """Bootstrap replications as a non-empty one-dimensional float array without NaN."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional array")
    if np.isnan(x).any():
        raise ValueError("samples contain NaN")
    return x
