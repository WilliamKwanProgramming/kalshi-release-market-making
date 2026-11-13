"""The mid proxy of a trades-only tape (docs/model.md section 4).

The tape has no quotes. A taker who buys YES pays the ask and a taker who sells YES receives the
bid, so the latest buy print stands for the ask and the latest sell print for the bid:

    m_hat(t) = (price of the last buy + price of the last sell) / 2

where "last" is the latest print at or before ``t`` (strictly before ``t`` with
``inclusive=False``). The proxy is undefined (NaN) unless both sides have printed and the older of
the two prints is no more than ``stale_s`` seconds before ``t``.

The value at ``t`` uses prints up to ``t`` only: changing or deleting later prints cannot change it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["mid_proxy_asof"]


def mid_proxy_asof(
    ts: ArrayLike,
    side: ArrayLike,
    price: ArrayLike,
    query_ts: ArrayLike,
    *,
    stale_s: float = 3600.0,
    inclusive: bool = True,
) -> NDArray[np.float64]:
    """Mid proxy ``(last buy price + last sell price) / 2`` at each time in ``query_ts``.

    ``ts``, ``side`` and ``price`` are the arrays of the trades of one market, sorted by time
    (equal times keep their order, the later print counts as the last one); ``side`` is +1 when the
    taker bought and -1 when the taker sold. With ``inclusive=True`` a print at exactly the query
    time counts, with ``inclusive=False`` only earlier prints do. The result has the shape of
    ``query_ts`` and is NaN where a side has no print, where either last print is more than
    ``stale_s`` seconds old (an age of exactly ``stale_s`` is still fresh), or where the query time
    is NaN.
    """
    t = np.asarray(ts, dtype=np.float64)
    s = np.asarray(side)
    p = np.asarray(price, dtype=np.float64)
    q = np.asarray(query_ts, dtype=np.float64)
    if t.ndim != 1 or s.shape != t.shape or p.shape != t.shape:
        raise ValueError("ts, side and price must be one-dimensional arrays of the same length")
    if np.any(np.isnan(t)) or np.any(np.diff(t) < 0.0):
        raise ValueError("ts must be sorted in increasing order and free of NaN")
    if not np.all((s == 1) | (s == -1)):
        raise ValueError("side must be +1 (taker bought) or -1 (taker sold)")
    if not stale_s >= 0.0:
        raise ValueError("stale_s must be non-negative")

    last_price = []
    for sign in (1, -1):
        keep = s == sign
        t_side, p_side = t[keep], p[keep]
        if t_side.size == 0:
            return np.full(q.shape, np.nan)[()]
        k = np.searchsorted(t_side, q, side="right" if inclusive else "left") - 1
        found = k >= 0
        k = np.where(found, k, 0)
        fresh = found & (q - t_side[k] <= stale_s)  # NaN in q compares False: stays undefined
        last_price.append(np.where(fresh, p_side[k], np.nan))
    return (0.5 * (last_price[0] + last_price[1]))[()]
