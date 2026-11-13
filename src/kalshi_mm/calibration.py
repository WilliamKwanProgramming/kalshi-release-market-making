"""Recalibration of prices and the Brier skill score (docs/model.md section 8).

A price ``p`` is recalibrated by

    p_hat = sigmoid(a + b * logit(p))

where ``(a, b)`` are the coefficients of an unpenalised logistic regression of the outcome ``y`` on
``logit(p)`` over earlier releases only. The score is the Brier score (Brier 1950),
``mean((p_hat - y)^2)``, and the skill is ``BSS = 1 - Brier(recal) / Brier(raw)``: a positive value
means that recalibrating helps out of sample. The favourite-longshot pattern is ``b > 1`` here
(low prices win less often than their price, high prices more often) when the raw price is the
regressor; a flatter price than the truth gives ``b > 1``, an over-dispersed price ``b < 1``.

* :func:`fit_recalibration`: the coefficients, by Newton's method on the log-likelihood.
* :func:`recalibrate`: ``sigmoid(a + b logit(clip(p)))``.
* :func:`brier` and :func:`brier_skill`.
* :func:`walk_forward_recalibration`: an expanding window in time order, counted in releases.
* :func:`skill_interval`: the skill with a cluster-bootstrap interval (docs/model.md section 9).

Prices are clipped to ``[clip, 1 - clip]`` before the logit, so that a price of 0 or 1 does not give
an infinite regressor. The raw price is not clipped when it is scored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from kalshi_mm.logit import logit, sigmoid
from kalshi_mm.stats import cluster_weights, p_at_or_below_zero, percentile_interval, ratio_of_sums

__all__ = [
    "RecalibrationError",
    "brier",
    "brier_skill",
    "fit_recalibration",
    "recalibrate",
    "skill_interval",
    "walk_forward_recalibration",
]

MIN_OBSERVATIONS = 10  # fewest observations a fit accepts
_HALVINGS = 30  # a Newton step is halved at most this many times
_LL_SLACK = 1e-12  # relative rounding slack when comparing log-likelihoods


class RecalibrationError(ValueError):
    """A recalibration cannot be fitted: too few observations, a single outcome class, equal
    prices, or a logistic regression that does not converge (the prices separate the outcomes)."""


def _pair(a: ArrayLike, outcome: ArrayLike, what: str = "price") -> tuple[NDArray, NDArray]:
    """``a`` as floats in [0, 1] and ``outcome`` in {0, 1}: one-dimensional, same length."""
    p = np.asarray(a, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if p.ndim != 1 or y.shape != p.shape:
        raise ValueError(f"{what} and outcome must be one-dimensional arrays of the same length")
    if not np.all((p >= 0.0) & (p <= 1.0)):  # also false for NaN
        raise ValueError(f"{what} must be finite and lie in [0, 1]")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("outcome must be 0 or 1 (not missing)")
    return p, y


def _check_clip(clip: float) -> None:
    if not 0.0 < clip < 0.5:
        raise ValueError(f"clip must lie in (0, 0.5), got {clip!r}")


def _clipped_logit(price: NDArray, clip: float) -> NDArray:
    return logit(np.clip(price, clip, 1.0 - clip))


def _loglik(a: float, b: float, x: NDArray, y: NDArray) -> float:
    """Log-likelihood ``sum(y z - ln(1 + e^z))`` of a logistic regression with ``z = a + b x``."""
    z = a + b * x
    return float(y @ z - np.logaddexp(0.0, z).sum())


def fit_recalibration(
    price: ArrayLike,
    outcome: ArrayLike,
    *,
    clip: float = 0.01,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[float, float]:
    """``(a, b)`` of the logistic regression of ``outcome`` on ``logit(price clipped)``.

    The coefficients maximise ``sum(y z - ln(1 + e^z))`` with ``z = a + b x``, ``x`` the logit of
    the price clipped to ``[clip, 1 - clip]``; there is no penalty. Newton's method starts at
    ``(0, 0)`` and solves ``H delta = g`` with the score ``g = sum((y - p_hat) (1, x))`` and the
    information ``H = sum(p_hat (1 - p_hat) (1, x)' (1, x))``; a step that lowers the
    log-likelihood by more than rounding error is halved (at most 30 times). It stops when the full
    Newton step is at most ``tol`` in every coordinate.

    Raises :class:`RecalibrationError` (a ``ValueError``) for fewer than 10 observations, for a
    single outcome class, for prices that are all equal after clipping, and when the iteration has
    not converged after ``max_iter`` steps: with (near) separation the maximum likelihood estimate
    is at infinity and Newton's method keeps stepping, so the error is raised rather than the last
    iterate returned.
    """
    _check_clip(clip)
    p, y = _pair(price, outcome)
    n = p.size
    if n < MIN_OBSERVATIONS:
        raise RecalibrationError(f"a fit needs at least {MIN_OBSERVATIONS} observations, got {n}")
    if int(y.sum()) in (0, n):
        raise RecalibrationError("the outcomes are all equal: there is only one outcome class")
    x = _clipped_logit(p, clip)
    if x.max() == x.min():
        raise RecalibrationError("the prices are all equal after clipping: no slope can be fitted")
    not_converged = RecalibrationError(
        f"the logistic regression did not converge in {max_iter} iterations: the prices (nearly) "
        "separate the outcomes, so the coefficients are not finite"
    )

    a = b = 0.0
    ll = _loglik(a, b, x, y)
    for _ in range(max_iter):
        prob = sigmoid(a + b * x)
        resid = y - prob
        w = prob * (1.0 - prob)
        h00 = float(w.sum())
        if not h00 > 0.0:
            raise not_converged
        mu = float(w @ x) / h00  # weighted mean of x
        s2 = float(w @ (x - mu) ** 2)  # h11 - h01^2 / h00 without cancellation
        if not s2 > 0.0:
            raise not_converged
        g0, g1 = float(resid.sum()), float(resid @ x)
        db = (g1 - mu * g0) / s2  # H delta = g with H = [[h00, mu h00], [mu h00, h11]]
        da = g0 / h00 - mu * db
        if max(abs(da), abs(db)) <= tol:
            return a + da, b + db
        step = 1.0
        for _ in range(_HALVINGS):
            ll_new = _loglik(a + step * da, b + step * db, x, y)
            if ll_new >= ll - _LL_SLACK * (1.0 + abs(ll)):
                break
            step *= 0.5
        else:
            raise not_converged
        a, b, ll = a + step * da, b + step * db, ll_new
    raise not_converged


def recalibrate(
    price: ArrayLike, a: float, b: float, *, clip: float = 0.01
) -> NDArray[np.float64]:
    """``sigmoid(a + b * logit(price clipped to [clip, 1 - clip]))`` (docs/model.md section 8).

    ``(a, b) = (0, 1)`` returns the price itself (inside the clip). A scalar price gives a scalar.
    """
    _check_clip(clip)
    p = np.asarray(price, dtype=float)
    if not np.all((p >= 0.0) & (p <= 1.0)):
        raise ValueError("price must be finite and lie in [0, 1]")
    return sigmoid(a + b * _clipped_logit(p, clip))


def brier(prob: ArrayLike, outcome: ArrayLike) -> float:
    """Brier (1950) score ``mean((prob - outcome)^2)`` for outcomes in {0, 1}."""
    p, y = _pair(prob, outcome, "prob")
    if p.size == 0:
        raise ValueError("at least one observation is needed")
    return float(np.mean((p - y) ** 2))


def brier_skill(raw: ArrayLike, recal: ArrayLike, outcome: ArrayLike) -> float:
    """``1 - Brier(recal) / Brier(raw)``: positive when the recalibrated probabilities score
    better than the raw prices (docs/model.md section 8). Undefined (an error) if the raw Brier
    score is zero."""
    brier_raw = brier(raw, outcome)
    if brier_raw == 0.0:
        raise ValueError("the Brier score of the raw prices is zero: the skill is undefined")
    return 1.0 - brier(recal, outcome) / brier_raw


def _cluster_codes(labels: ArrayLike, n: int, what: str = "clusters") -> tuple[NDArray, int]:
    """Integer code of each label in sorted label order, and the number of distinct labels."""
    arr = np.asarray(labels)
    if arr.shape != (n,):
        raise ValueError(f"{what} must have one label per row ({n}), got shape {arr.shape}")
    if pd.isna(arr).any():
        raise ValueError(f"{what} must not contain missing labels")
    codes, uniques = pd.factorize(arr, sort=True)
    return np.asarray(codes), len(uniques)


def walk_forward_recalibration(
    price: ArrayLike,
    outcome: ArrayLike,
    release_ts: ArrayLike,
    *,
    min_events: int = 50,
    refit_every: int = 10,
    cluster: ArrayLike | None = None,
    gap_hours: float = 0.0,
) -> pd.DataFrame:
    """Out-of-sample recalibrated probabilities from an expanding window (docs/model.md section 8).

    The distinct values of ``release_ts`` are the releases (several markets may share one); the
    outcome of a release is taken to be known at its ``release_ts``. Let ``T_1 < T_2 < ...`` be
    them in time order. A release ``T_k`` is scored with the price observed ``gap_hours`` before
    it (the caller has taken that price), so only the outcomes of releases earlier than
    ``T_k - gap_hours`` were known when the price was observed. The coefficients used for release
    ``T_k`` are fitted by :func:`fit_recalibration` on every observation with ``release_ts``
    strictly earlier than ``T_k' - gap_hours``, where ``k'`` is the release at which the last
    refit happened: the fit is refreshed when ``refit_every`` further releases have passed since
    the previous one, and the releases inside the gap are left out although their outcomes are
    known by ``T_k``. The first release that is scored is the first one with at least
    ``min_events`` releases strictly earlier than its price time: the ``(min_events + 1)``-th
    unless releases lie within ``gap_hours`` of each other. All observations of one release share
    the same coefficients, and no observation of the release itself, of a later one or of one
    inside the gap enters the fit, so changing or deleting data from time ``T_k - gap_hours`` on
    cannot change the coefficients used for ``T_k``, and data from ``T_k`` on cannot change a row
    of a release before ``T_k``. A fit that fails (see :class:`RecalibrationError`) stops the run
    with a message that names the release.

    Returns one row per observation of a scored release, in the order of ``release_ts``, then
    ``price``, then ``outcome``, then the cluster label: the order does not depend on the order of
    the input. Columns ``price``, ``recal`` (the recalibrated probability), ``outcome`` (0 or 1),
    ``release_ts``, ``a``, ``b`` (the coefficients used), ``n_fit`` (observations in that fit) and,
    when ``cluster`` (one label per input row) is given, ``cluster``, so that ``frame["cluster"]``
    can be passed to :func:`skill_interval`. The frame is empty when no release has ``min_events``
    releases before its price time.
    """
    for name, value in (("min_events", min_events), ("refit_every", refit_every)):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if not (np.isfinite(gap_hours) and gap_hours >= 0.0):
        raise ValueError(f"gap_hours must be finite and not negative, got {gap_hours!r}")
    p, y = _pair(price, outcome)
    ts = np.asarray(release_ts, dtype=float)
    if ts.shape != p.shape or not np.all(np.isfinite(ts)):
        raise ValueError("release_ts must be finite with one value per observation")
    keys = [y, p, ts]  # np.lexsort sorts by the last key first
    if cluster is not None:
        keys.insert(0, _cluster_codes(cluster, p.size, "cluster")[0])
    order = np.lexsort(keys)
    p, y, ts = p[order], y[order], ts[order]

    releases, first = np.unique(ts, return_index=True)  # first row of each release in time order
    n, k_total = p.size, releases.size
    price_time = releases - 3600.0 * float(gap_hours)
    known = np.searchsorted(releases, price_time, side="left")  # releases before each price time
    eligible = np.flatnonzero(known >= min_events)
    first_scored = int(eligible[0]) if eligible.size else k_total
    columns = {"price": p, "recal": np.empty(n), "outcome": y.astype(np.int64), "release_ts": ts,
               "a": np.empty(n), "b": np.empty(n), "n_fit": np.zeros(n, dtype=np.int64)}
    for k in range(first_scored, k_total, refit_every):
        start = int(first[k])  # rows before it belong to earlier releases
        stop = int(first[k + refit_every]) if k + refit_every < k_total else n
        in_fit = int(np.searchsorted(ts, price_time[k], side="left"))  # rows known at price time
        try:
            a, b = fit_recalibration(p[:in_fit], y[:in_fit])
        except RecalibrationError as err:
            raise RecalibrationError(
                f"the fit before the release at {releases[k]:.0f} failed: {err}"
            ) from err
        columns["recal"][start:stop] = recalibrate(p[start:stop], a, b)
        columns["a"][start:stop], columns["b"][start:stop] = a, b
        columns["n_fit"][start:stop] = in_fit
    frame = pd.DataFrame(columns)
    if cluster is not None:
        frame["cluster"] = np.asarray(cluster)[order]
    scored = int(first[first_scored]) if first_scored < k_total else n
    return frame.iloc[scored:].reset_index(drop=True)


def skill_interval(
    frame: pd.DataFrame,
    clusters: ArrayLike,
    *,
    B: int = 10_000,
    seed: int = 20261113,
    level: float = 0.95,
) -> dict:
    """The Brier skill score of a frame with a cluster-bootstrap interval (docs/model.md section 9).

    ``frame`` has the columns ``price`` (raw), ``recal`` and ``outcome`` (for example the output of
    :func:`walk_forward_recalibration`); ``clusters`` has one label per row (the release or event,
    no label missing). ``bss = 1 - brier_recal / brier_raw`` is computed on all rows.
    With ``G`` clusters, each replication draws a row of counts from
    :func:`kalshi_mm.stats.cluster_weights` (seeded with ``seed``, clusters in sorted label
    order), so that cluster ``g`` is used ``w[g]`` times, and recomputes the skill from the squared
    errors of the clusters drawn. ``ci`` is the percentile interval at ``level`` of the ``B``
    replications and ``p_one_sided`` the share of replications at or below zero, never below
    ``1 / (B + 1)`` (docs/protocol.md). Returns ``bss``, ``ci``, ``brier_raw``, ``brier_recal``,
    ``p_one_sided``, ``n`` (rows), ``n_clusters``, ``B`` and ``seed``.
    """
    price, outcome = _pair(frame["price"], frame["outcome"])
    recal, _ = _pair(frame["recal"], frame["outcome"], "recal")
    codes, n_clusters = _cluster_codes(clusters, price.size)
    if price.size == 0:
        raise ValueError("at least one row is needed")
    sq_raw = (price - outcome) ** 2
    sq_recal = (recal - outcome) ** 2
    brier_raw, brier_recal = float(sq_raw.mean()), float(sq_recal.mean())
    if brier_raw == 0.0:
        raise ValueError("the Brier score of the raw prices is zero: the skill is undefined")

    weights = cluster_weights(n_clusters, B, np.random.default_rng(seed))
    ratio = ratio_of_sums(
        np.bincount(codes, weights=sq_recal, minlength=n_clusters),
        np.bincount(codes, weights=sq_raw, minlength=n_clusters),
        weights,
    )  # Brier_recal / Brier_raw of each replication: the counts of rows cancel
    boot = 1.0 - ratio
    if np.isnan(boot).any():
        raise ValueError("a bootstrap replication has a raw Brier score of zero: too few clusters")
    return {
        "bss": 1.0 - brier_recal / brier_raw,
        "ci": percentile_interval(boot, level),
        "brier_raw": brier_raw,
        "brier_recal": brier_recal,
        "p_one_sided": p_at_or_below_zero(boot),
        "n": int(price.size),
        "n_clusters": int(n_clusters),
        "B": int(B),
        "seed": int(seed),
    }
