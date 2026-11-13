"""Log-odds helpers and tick rounding (docs/model.md, sections 2 and 3).

* ``logit(p) = ln(p / (1 - p))`` and its inverse ``sigmoid(x) = 1 / (1 + exp(-x))``.
* ``information_step(pi) = ln((1 + pi) / (1 - pi))``: the change of the public log-odds caused by
  one trade when a share ``pi`` of trades is informed (section 3; Glosten and Milgrom 1985).
* ``martingale_drift(x, sigma) = sigma^2 / 2 * tanh(x / 2)``: the drift of the log-odds that
  makes the price ``sigmoid(x)`` a martingale (section 2).
* ``local_price_vol(p, sigma) = sigma p (1 - p)``: price volatility per square-root hour.
* ``payoff_variance(p) = p (1 - p)``: variance of the binary payoff still to be resolved.
* ``round_to_tick``: rounding of a price to the tick grid, guarded against floating-point noise.

All functions are vectorised. A scalar argument gives a numpy scalar, an array gives an array.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special

__all__ = [
    "information_step",
    "local_price_vol",
    "logit",
    "martingale_drift",
    "payoff_variance",
    "round_to_tick",
    "sigmoid",
]

_DIRECTIONS = ("nearest", "down", "up")


def _unwrap(a: NDArray[np.float64]) -> Any:
    """A 0-d array becomes a numpy scalar; an array with at least one axis is returned as is."""
    return a[()]


def logit(p: ArrayLike) -> NDArray[np.float64]:
    """Log-odds ``ln(p / (1 - p))``, computed as ``ln p - ln(1 - p)`` with ``log1p``.

    Gives ``-inf`` at 0, ``+inf`` at 1 and NaN outside [0, 1].
    """
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return _unwrap(np.log(p) - np.log1p(-p))


def sigmoid(x: ArrayLike) -> NDArray[np.float64]:
    """Inverse of :func:`logit`, ``1 / (1 + exp(-x))``, stable for large ``|x|`` (no overflow)."""
    return _unwrap(special.expit(np.asarray(x, dtype=float)))


def information_step(pi: ArrayLike) -> NDArray[np.float64]:
    """``ln((1 + pi) / (1 - pi))`` (docs/model.md section 3), as ``log1p(pi) - log1p(-pi)``.

    One buy multiplies the odds of ``y = 1`` by ``(1 + pi) / (1 - pi)``, one sell divides them by
    it. Defined for the share ``pi`` in [0, 1]; ``+inf`` at 1 and NaN outside [0, 1].
    """
    pi = np.asarray(pi, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.log1p(pi) - np.log1p(-pi)
    return _unwrap(np.where((pi >= 0.0) & (pi <= 1.0), step, np.nan))


def martingale_drift(x: ArrayLike, sigma: ArrayLike) -> NDArray[np.float64]:
    """Drift ``sigma^2 / 2 * tanh(x / 2)`` of the log-odds ``x`` under the public filtration.

    With ``tanh(x / 2) = 2 p - 1`` this is ``sigma^2 (2 p - 1) / 2`` and makes
    ``p = sigmoid(x)`` a martingale (docs/model.md section 2).
    """
    return _unwrap(0.5 * np.asarray(sigma, dtype=float) ** 2 * np.tanh(0.5 * np.asarray(x, float)))


def local_price_vol(p: ArrayLike, sigma: ArrayLike) -> NDArray[np.float64]:
    """Local volatility of the price per square-root hour, ``sigma p (1 - p)`` (section 2)."""
    p = np.asarray(p, dtype=float)
    return _unwrap(np.asarray(sigma, dtype=float) * p * (1.0 - p))


def payoff_variance(p: ArrayLike) -> NDArray[np.float64]:
    """Variance ``p (1 - p)`` of the binary payoff given the price ``p`` (section 2)."""
    p = np.asarray(p, dtype=float)
    return _unwrap(p * (1.0 - p))


def round_to_tick(
    price: ArrayLike, tick: float = 0.01, direction: str = "nearest"
) -> NDArray[np.float64]:
    """Round ``price`` to a multiple of ``tick``.

    ``direction`` is ``"nearest"`` (ties go up), ``"down"`` or ``"up"``. The price is first divided
    by the tick; a quotient within 1e-9 (relative to its size, at least 1e-14) of a grid point, or
    of a half-way point for ``"nearest"``, is snapped to it, so that float noise does not move a
    price across the grid (0.29 / 0.01 is 28.999999999999996, and 0.29 still rounds down to 0.29).
    The result is rounded to 10 decimals, so 0.35 stays 0.35 and not 0.35000000000000003.
    """
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    if not (np.isfinite(tick) and tick > 0.0):
        raise ValueError(f"tick must be positive and finite, got {tick!r}")
    with np.errstate(invalid="ignore"):  # an infinite price gives inf - inf in the snap test
        n = np.asarray(price, dtype=float) / tick
        grid = np.round(n * 2.0) / 2.0 if direction == "nearest" else np.round(n)
        tol = np.maximum(1e-9, 1e-14 * np.abs(n))
        n = np.where(np.abs(n - grid) <= tol, grid, n)
        if direction == "down":
            n = np.floor(n)
        elif direction == "up":
            n = np.ceil(n)
        else:
            n = np.floor(n + 0.5)
        return _unwrap(np.round(n * tick, 10))
