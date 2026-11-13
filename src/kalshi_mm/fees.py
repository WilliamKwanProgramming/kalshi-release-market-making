"""Kalshi trading fees (docs/model.md section 7).

For a fill of ``C`` contracts at YES price ``P`` the schedule of 7 July 2026 charges

    fee = ceil_to_0.0001( multiplier * coeff * C * P * (1 - P) )

with ``coeff = 0.07`` for the taker and ``0.0175`` for the maker (resting order), and
``multiplier = 1`` for the CPI, payrolls, GDP, unemployment, Fed-rate and Fed-decision series. The
fee is symmetric in ``P <-> 1 - P`` and zero at 0 and 1. The per-order rounding accumulator and
rebate in Kalshi's documentation can only lower fees and are not modelled.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["MAKER_COEFF", "TAKER_COEFF", "kalshi_fee", "maker_fee", "taker_fee"]

TAKER_COEFF = 0.07
MAKER_COEFF = 0.0175

_UNITS_PER_USD = 1e4  # fees are rounded up to the next 0.0001 USD


def kalshi_fee(
    count: ArrayLike, price: ArrayLike, coeff: float, multiplier: ArrayLike = 1.0
) -> NDArray[np.float64]:
    """Fee in USD for ``count`` contracts at YES price ``price``.

    ``ceil`` to the next 0.0001 USD of ``multiplier * coeff * count * price * (1 - price)``. The raw
    fee is first rounded to 9 decimals, so that float noise (0.07 * 100 * 0.1 * 0.9 evaluates to
    0.6300000000000001) cannot push a fee that is exactly on the 0.0001 grid up by one unit.
    Vectorised; ``count`` must be non-negative and ``price`` must lie in [0, 1].
    """
    count = np.asarray(count, dtype=float)
    price = np.asarray(price, dtype=float)
    if np.any(count < 0.0):
        raise ValueError("count must be non-negative")
    if np.any((price < 0.0) | (price > 1.0)):
        raise ValueError("price must lie in [0, 1]")
    raw = multiplier * coeff * count * price * (1.0 - price)
    units = np.round(raw * _UNITS_PER_USD, 5)  # raw fee in USD rounded to 9 decimals
    return np.ceil(units) / _UNITS_PER_USD


def maker_fee(
    count: ArrayLike, price: ArrayLike, multiplier: ArrayLike = 1.0
) -> NDArray[np.float64]:
    """Fee of a resting order that trades: :func:`kalshi_fee` with ``MAKER_COEFF`` (0.0175)."""
    return kalshi_fee(count, price, MAKER_COEFF, multiplier)


def taker_fee(
    count: ArrayLike, price: ArrayLike, multiplier: ArrayLike = 1.0
) -> NDArray[np.float64]:
    """Fee of an order that takes liquidity: :func:`kalshi_fee` with ``TAKER_COEFF`` (0.07)."""
    return kalshi_fee(count, price, TAKER_COEFF, multiplier)
