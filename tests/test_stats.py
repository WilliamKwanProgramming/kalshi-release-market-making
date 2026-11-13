"""Cluster bootstrap and multiple-testing helpers (docs/model.md section 9)."""

import numpy as np
import pytest

from kalshi_mm.stats import (
    cluster_weights,
    holm_adjust,
    p_at_or_below_zero,
    percentile_interval,
    ratio_of_sums,
)


# ----------------------------------------------------------------------------- cluster weights
def test_cluster_weights_shape_dtype_and_row_sums():
    w = cluster_weights(37, 500, np.random.default_rng(1))
    assert w.shape == (500, 37) and np.issubdtype(w.dtype, np.integer)
    assert np.all(w >= 0) and np.all(w.sum(axis=1) == 37)


def test_cluster_weights_are_reproducible_and_depend_on_the_generator():
    a = cluster_weights(20, 50, np.random.default_rng(5))
    b = cluster_weights(20, 50, np.random.default_rng(5))
    c = cluster_weights(20, 50, np.random.default_rng(6))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cluster_weights_marginal_mean_and_variance():
    """Each count is Binomial(G, 1/G): mean 1, variance 1 - 1/G; tolerance 4 standard errors."""
    G, B = 25, 20000
    w = cluster_weights(G, B, np.random.default_rng(11)).astype(float)
    se_mean = np.sqrt((1 - 1 / G) / B)
    assert np.all(np.abs(w.mean(axis=0) - 1.0) < 4 * se_mean)
    # standard error of a sample variance, from the fourth moment of the draws themselves
    v = w.var(axis=0, ddof=1)
    m4 = ((w - w.mean(axis=0)) ** 4).mean(axis=0)
    se_var = np.sqrt((m4 - v**2) / B)
    assert np.all(np.abs(v - (1 - 1 / G)) < 4 * se_var)


def test_bootstrap_mean_has_the_plug_in_variance():
    """Var*(mean of a resample) = (1/G) * (1/G) sum (x - xbar)^2, exactly, for resampling G of G."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(30) * np.arange(1, 31) ** 0.5
    G, B = x.size, 40000
    w = cluster_weights(G, B, np.random.default_rng(4))
    boot_means = (w @ x) / G
    target = np.var(x) / G  # plug-in variance (ddof = 0)
    se = target * np.sqrt(2.0 / (B - 1))  # bootstrap means are close to normal for G = 30
    assert abs(boot_means.var(ddof=1) - target) < 4 * se
    assert abs(boot_means.mean() - x.mean()) < 4 * np.sqrt(target / B)


def test_cluster_weights_invalid():
    with pytest.raises(ValueError):
        cluster_weights(0, 10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        cluster_weights(10, 0, np.random.default_rng(0))


# --------------------------------------------------------------------------- percentile interval
def test_percentile_interval_by_hand():
    x = np.arange(101.0)  # the q-quantile of 0..100 is 100 q
    assert percentile_interval(x) == pytest.approx((2.5, 97.5), rel=1e-12)
    assert percentile_interval(x, level=0.9) == pytest.approx((5.0, 95.0), rel=1e-12)
    # order does not matter
    assert percentile_interval(x[::-1].copy(), level=0.5) == pytest.approx((25.0, 75.0), rel=1e-12)


def test_percentile_interval_matches_numpy_percentile():
    x = np.random.default_rng(8).standard_normal(1001)
    lo, hi = percentile_interval(x, 0.95)
    np.testing.assert_allclose([lo, hi], np.percentile(x, [2.5, 97.5]))
    assert isinstance(lo, float) and isinstance(hi, float)
    assert lo < np.median(x) < hi


def test_percentile_interval_rejects_bad_input():
    for level in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            percentile_interval([1.0, 2.0, 3.0], level)
    with pytest.raises(ValueError):
        percentile_interval([1.0, np.nan, 3.0])
    with pytest.raises(ValueError):
        percentile_interval([])
    with pytest.raises(ValueError):
        percentile_interval(np.ones((3, 3)))


# ---------------------------------------------------------------------------------- p-value
def test_p_at_or_below_zero_by_hand():
    assert p_at_or_below_zero([-1.0, 0.0, 1.0, 2.0]) == 0.5  # zero counts as "at or below"
    assert p_at_or_below_zero([0.0, 1.0, 2.0, 3.0]) == 0.25
    assert p_at_or_below_zero([-3.0, -1.0, 0.0]) == 1.0
    assert p_at_or_below_zero(np.arange(1.0, 1000.0)) == pytest.approx(1 / 1000)  # floor, B = 999
    assert p_at_or_below_zero(np.arange(1.0, 10001.0)) == pytest.approx(1 / 10001)
    assert p_at_or_below_zero([0.5, 0.5]) == pytest.approx(1 / 3)  # floor 1 / (B + 1) binds
    assert p_at_or_below_zero([-1.0] + [1.0] * 99) == pytest.approx(0.01)  # share above the floor


def test_p_at_or_below_zero_rejects_nan():
    with pytest.raises(ValueError):
        p_at_or_below_zero([1.0, np.nan])


# -------------------------------------------------------------------------------------- Holm
def _holm_by_loop(p):
    """Independent step-down implementation: reject-threshold form turned into adjusted values."""
    m = len(p)
    order = sorted(range(m), key=lambda i: (p[i], i))
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):  # rank 0 is the smallest p-value
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def test_holm_hand_example():
    """The textbook example: p = 0.01, 0.04, 0.03, 0.005 with m = 4."""
    adj = holm_adjust([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02], atol=1e-15)
    # step by step: sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 = 0.02, 0.03, 0.06, 0.04,
    # then the running maximum makes the last one 0.06
    assert list(holm_adjust([0.5])) == [0.5]
    assert holm_adjust([]).size == 0


def test_holm_cap_monotonicity_and_order_restored():
    adj = holm_adjust([0.4, 0.3, 0.9, 0.02])
    np.testing.assert_allclose(adj, [0.9, 0.9, 0.9, 0.08], atol=1e-15)  # capped / carried forward
    assert holm_adjust([0.3, 0.001, 0.2])[1] == pytest.approx(0.003)
    # ties get equal adjusted values
    adj = holm_adjust([0.02, 0.02, 0.5])
    assert adj[0] == adj[1] == pytest.approx(0.06)


def test_holm_equals_independent_loop_and_has_the_expected_properties():
    rng = np.random.default_rng(21)
    for _ in range(300):
        m = int(rng.integers(1, 12))
        p = rng.uniform(0, 1, m) ** 2  # many small values
        adj = holm_adjust(p)
        np.testing.assert_allclose(adj, _holm_by_loop(list(p)), rtol=0, atol=1e-15)
        assert np.all(adj >= p - 1e-15) and np.all(adj <= 1.0)
        assert np.all(adj <= np.minimum(1.0, m * p) + 1e-15)  # never worse than Bonferroni
        order = np.argsort(p, kind="stable")
        assert np.all(np.diff(adj[order]) >= -1e-15)  # monotone along the sorted p-values
        perm = rng.permutation(m)
        np.testing.assert_allclose(holm_adjust(p[perm]), adj[perm], atol=1e-15)  # order restored


def test_holm_controls_the_familywise_error_rate_under_the_global_null():
    """m independent U(0,1) p-values: P(any adjusted p < 0.05) = 1 - (1 - 0.05 / m)^m."""
    m, R = 5, 40000
    p = np.random.default_rng(2).uniform(size=(R, m))
    rejected = np.array([holm_adjust(row).min() < 0.05 for row in p])
    target = 1 - (1 - 0.05 / m) ** m
    se = np.sqrt(target * (1 - target) / R)
    assert abs(rejected.mean() - target) < 4 * se
    assert rejected.mean() < 0.05 + 4 * se


def test_holm_rejects_bad_input():
    for bad in ([0.1, np.nan], [0.1, 1.2], [-0.1, 0.5]):
        with pytest.raises(ValueError):
            holm_adjust(bad)
    with pytest.raises(ValueError):
        holm_adjust(np.ones((2, 2)) * 0.5)


# ------------------------------------------------------------------------------- ratio of sums
def test_ratio_of_sums_equals_an_explicit_resample():
    rng = np.random.default_rng(13)
    G, B = 17, 60
    num = rng.normal(1.0, 2.0, G)
    den = rng.integers(1, 40, G).astype(float)
    w = cluster_weights(G, B, rng)
    fast = ratio_of_sums(num, den, w)
    assert fast.shape == (B,)
    for b in range(B):
        idx = np.repeat(np.arange(G), w[b])  # the resampled list of clusters, with repeats
        assert idx.size == G
        assert fast[b] == pytest.approx(num[idx].sum() / den[idx].sum(), rel=1e-12)


def test_ratio_of_sums_with_identity_weights_is_the_pooled_ratio():
    num, den = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 4.0])
    out = ratio_of_sums(num, den, np.ones((1, 3), dtype=int))
    assert out[0] == pytest.approx(6.0 / 8.0)


def test_ratio_of_sums_zero_denominator_and_shape_errors():
    out = ratio_of_sums([1.0, 2.0], [0.0, 1.0], np.array([[2, 0], [0, 2]]))
    assert np.isnan(out[0]) and out[1] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        ratio_of_sums([1.0, 2.0], [1.0], np.ones((1, 2), dtype=int))
    with pytest.raises(ValueError):
        ratio_of_sums([1.0, 2.0], [1.0, 1.0], np.ones((1, 3), dtype=int))
    with pytest.raises(ValueError):
        ratio_of_sums([1.0, 2.0], [1.0, 1.0], np.ones(2, dtype=int))


# --------------------------------------------------------------------------------- coverage
def _clustered_sample(rng, G, mu=0.3, n_mean=15):
    """Per-cluster sum and count of a trade-weighted mean with a cluster effect; truth is mu."""
    n = 1 + rng.poisson(n_mean, G)
    effect = rng.normal(0.0, 1.0, G)  # shared by all trades of the cluster
    noise = rng.normal(0.0, 1.0, G) * np.sqrt(n)  # sum of n independent trade-level errors
    return n * (mu + effect) + noise, n.astype(float)


def test_cluster_bootstrap_covers_the_known_mean_and_the_naive_interval_does_not():
    """95% percentile intervals of a clustered mean, 500 simulated samples of 200 clusters.

    The coverage of the cluster bootstrap must lie within 4 Monte Carlo standard errors
    (sqrt(0.95 * 0.05 / 500) = 0.0097) of 0.95; the percentile interval is a little short at this
    size (a separate run of 3000 samples gave 0.940), well inside that band. The naive normal
    interval that treats all trades as independent ignores the cluster effect and must miss by
    more than 10 standard errors.
    """
    R, G, B, mu = 500, 200, 400, 0.3
    rng = np.random.default_rng(2026)
    cover_cluster = cover_naive = 0
    for _ in range(R):
        num, den = _clustered_sample(rng, G, mu)
        w = cluster_weights(G, B, rng)
        lo, hi = percentile_interval(ratio_of_sums(num, den, w))
        cover_cluster += lo <= mu <= hi
        n_tot = den.sum()
        est = num.sum() / n_tot
        per_trade_var = np.sum((num - est * den) ** 2 / den) / n_tot  # variance if trades were iid
        cover_naive += abs(est - mu) <= 1.96 * np.sqrt(per_trade_var / n_tot)
    mc_se = np.sqrt(0.95 * 0.05 / R)
    assert abs(cover_cluster / R - 0.95) < 4 * mc_se
    assert cover_naive / R < 0.95 - 10 * mc_se
