"""Test configuration and shared helpers.

Puts this checkout's ``src/`` first on ``sys.path`` and drops any ``kalshi_mm`` module imported
earlier, so the tests always exercise the sources next to them (a mutation check runs the suite
on a modified copy and relies on this).

The ``cluster`` fixture gives cluster-robust standard errors for statistics over trades. Trades of
one event are dependent, so every standard error in the tests treats the event as the unit.
"""

import pathlib
import sys

import numpy as np
import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
for _name in [m for m in sys.modules if m == "kalshi_mm" or m.startswith("kalshi_mm.")]:
    del sys.modules[_name]


class ClusterStats:
    """Ratio estimators over rows grouped in clusters, with linearised standard errors.

    For a ratio of sums ``R = sum(num) / sum(den)`` over the rows in ``mask``, with per-cluster
    sums ``A_g`` and ``B_g``, the standard error is
    ``sqrt(G / (G - 1) * sum_g (A_g - R B_g)^2) / sum(B)`` where ``G`` is the number of clusters
    with at least one row. With ``den = 1`` it is the standard error of a mean of rows. ``codes``
    are non-negative integer cluster numbers, one per row.
    """

    @staticmethod
    def _se(influence, n_clusters):
        """``sqrt(G / (G - 1) sum u^2)``; NaN when only one cluster is left."""
        if n_clusters < 2:
            return np.nan
        return np.sqrt(n_clusters / (n_clusters - 1.0) * np.sum(influence**2))

    @staticmethod
    def _sums(num, den, codes, mask):
        n_clusters = int(codes.max()) + 1
        c = codes[mask]
        a = np.bincount(c, weights=np.asarray(num, float)[mask], minlength=n_clusters)
        b = np.bincount(c, weights=np.asarray(den, float)[mask], minlength=n_clusters)
        used = np.bincount(c, minlength=n_clusters) > 0
        return a, b, used

    def ratio_se(self, num, den, codes, mask=None):
        """``(R, se)`` of ``sum(num) / sum(den)`` over the rows in ``mask`` (default: all)."""
        codes = np.asarray(codes)
        mask = np.ones(codes.shape, bool) if mask is None else np.asarray(mask, bool)
        a, b, used = self._sums(num, den, codes, mask)
        ratio = a.sum() / b.sum()
        u = a - ratio * b
        return ratio, self._se(u, int(used.sum())) / b.sum()

    def mean_se(self, values, codes, mask=None):
        """``(mean, se)`` of ``values`` over the rows in ``mask``."""
        return self.ratio_se(values, np.ones(len(values)), codes, mask)

    def diff_se(self, num, den, codes, mask_a, mask_b):
        """``(R_a - R_b, se)``: the two ratios share clusters, so their covariance is included."""
        codes = np.asarray(codes)
        a1, b1, used1 = self._sums(num, den, codes, np.asarray(mask_a, bool))
        a2, b2, used2 = self._sums(num, den, codes, np.asarray(mask_b, bool))
        r1, r2 = a1.sum() / b1.sum(), a2.sum() / b2.sum()
        u = (a1 - r1 * b1) / b1.sum() - (a2 - r2 * b2) / b2.sum()
        return r1 - r2, self._se(u, int((used1 | used2).sum()))


@pytest.fixture(scope="session")
def cluster():
    return ClusterStats()
