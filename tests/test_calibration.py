"""Recalibration, the Brier skill and its cluster interval (kalshi_mm.calibration, model.md 8).

Closed forms used below:

* Two distinct regressor values make the logistic regression saturated: the fitted probabilities
  are the group frequencies, so ``a - b = logit(q0)`` and ``a + b = logit(q1)`` for prices
  ``sigmoid(-1)`` and ``sigmoid(1)``. Every fit in the hand examples is worked out this way.
* If the true probability is ``sigmoid(x)`` and the price is ``sigmoid(beta x)``, the logit of the
  price is ``beta x`` and ``P(y = 1 | price) = sigmoid(logit(price) / beta)``: the population
  coefficients are ``(a, b) = (0, 1 / beta)``, the recalibrated probability is the true one, and
  the expected Brier scores follow by Gauss-Hermite quadrature over ``x ~ N(0, s^2)``.

Statistical tests compare with a standard error computed from the data (the inverse information
for coefficients, the delta method for a skill score) and allow four of them; fixed seeds.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.polynomial.hermite_e import hermegauss
from scipy import optimize

from kalshi_mm.calibration import (
    RecalibrationError,
    brier,
    brier_skill,
    fit_recalibration,
    recalibrate,
    skill_interval,
    walk_forward_recalibration,
)
from kalshi_mm.logit import logit, sigmoid
from kalshi_mm.stats import cluster_weights, percentile_interval

Z = 4.0
LO, HI = sigmoid(-1.0), sigmoid(1.0)  # prices whose logit is -1 and +1
COLUMNS = ["price", "recal", "outcome", "release_ts", "a", "b", "n_fit"]


# --------------------------------------------------------------------------------- Brier score
def test_brier_by_hand():
    assert brier([0.9, 0.2, 0.5], [1, 0, 1]) == pytest.approx((0.01 + 0.04 + 0.25) / 3.0)
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 1]) == 0.25
    assert brier([0.0, 1.0], [1, 0]) == 1.0
    assert brier(np.array([0.3]), np.array([True])) == pytest.approx(0.49)  # booleans are outcomes


def test_brier_rejects_bad_input():
    with pytest.raises(ValueError, match="outcome"):
        brier([0.5, 0.5], [1, 2])
    with pytest.raises(ValueError, match="outcome"):
        brier([0.5, 0.5], [1, np.nan])
    with pytest.raises(ValueError, match="prob"):
        brier([0.5, 1.5], [1, 0])
    with pytest.raises(ValueError, match="prob"):
        brier([0.5, np.nan], [1, 0])
    with pytest.raises(ValueError, match="same length"):
        brier([0.5, 0.5, 0.5], [1, 0])
    with pytest.raises(ValueError, match="one-dimensional"):
        brier(np.full((2, 2), 0.5), np.ones((2, 2)))
    with pytest.raises(ValueError, match="at least one"):
        brier([], [])


def test_brier_skill_by_hand_and_sign():
    raw = [0.8, 0.6, 0.3, 0.5]
    recal = [0.9, 0.7, 0.2, 0.5]
    y = [1, 1, 0, 1]
    assert brier(raw, y) == pytest.approx((0.04 + 0.16 + 0.09 + 0.25) / 4.0)  # 0.135
    assert brier(recal, y) == pytest.approx((0.01 + 0.09 + 0.04 + 0.25) / 4.0)  # 0.0975
    assert brier_skill(raw, recal, y) == pytest.approx(1.0 - 0.0975 / 0.135)  # 0.2778 > 0
    assert brier_skill(raw, raw, y) == 0.0
    assert brier_skill(recal, raw, y) < 0.0  # the worse forecast has negative skill
    assert brier_skill(raw, y, y) == 1.0  # a perfect recalibrated forecast
    with pytest.raises(ValueError, match="zero"):
        brier_skill([1.0, 0.0], [0.5, 0.5], [1, 0])  # a perfect raw price: the skill is undefined


# -------------------------------------------------------------------------------- recalibrate
def test_recalibrate_by_hand():
    p = np.array([0.1, 0.5, 0.8])
    np.testing.assert_allclose(recalibrate(p, 0.0, 1.0), p, rtol=1e-14)  # the identity map
    np.testing.assert_allclose(recalibrate(p, 0.0, -1.0), 1.0 - p, rtol=1e-14)
    np.testing.assert_allclose(recalibrate(p, 0.0, 2.0), p**2 / (p**2 + (1 - p) ** 2), rtol=1e-14)
    np.testing.assert_allclose(recalibrate(p, 1.0, 0.0), np.full(3, sigmoid(1.0)), rtol=1e-14)
    # the intercept shifts the log-odds: odds are multiplied by e^a
    np.testing.assert_allclose(recalibrate(p, np.log(2.0), 1.0), 2 * p / (2 * p + 1 - p),
                               rtol=1e-14)
    scalar = recalibrate(0.3, 0.0, 1.0)
    assert np.ndim(scalar) == 0 and scalar == pytest.approx(0.3)


def test_recalibrate_clips_the_price_before_the_logit():
    edge = logit(0.99)
    got = recalibrate([0.0, 0.004, 0.01, 0.99, 0.999, 1.0], 0.2, 1.5)
    ends = np.array([-edge, -edge, -edge, edge, edge, edge])
    np.testing.assert_allclose(got, sigmoid(0.2 + 1.5 * ends), rtol=1e-14)
    assert np.all(np.isfinite(got))
    other = recalibrate([0.0, 0.05, 1.0], 0.0, 1.0, clip=0.05)  # a wider clip
    np.testing.assert_allclose(other, [0.05, 0.05, 0.95], rtol=1e-14)
    # an unclipped middle price does not depend on the clip
    assert recalibrate(0.4, 0.3, 1.2, clip=0.001) == recalibrate(0.4, 0.3, 1.2, clip=0.2)


def test_recalibrate_rejects_bad_input():
    for bad in ([-0.1], [1.2], [np.nan]):
        with pytest.raises(ValueError, match="price"):
            recalibrate(bad, 0.0, 1.0)
    for clip in (0.0, 0.5, -0.1, np.nan):
        with pytest.raises(ValueError, match="clip"):
            recalibrate([0.3], 0.0, 1.0, clip=clip)


# --------------------------------------------------------------------- the fit: closed forms
def saturated(q0, n0, q1, n1):
    """Prices ``sigmoid(-1)`` (n0 of them, a share q0 of outcomes 1) and ``sigmoid(1)``."""
    k0, k1 = round(q0 * n0), round(q1 * n1)
    price = np.r_[np.full(n0, LO), np.full(n1, HI)]
    outcome = np.r_[np.arange(n0) < k0, np.arange(n1) < k1].astype(int)
    return price, outcome


@pytest.mark.parametrize(
    "q0, n0, q1, n1",
    [(0.3, 10, 0.7, 10), (0.2, 10, 0.5, 10), (0.25, 4, 0.75, 16), (0.1, 50, 0.9, 50),
     (0.02, 50, 0.98, 50)],  # the last two are close to separation: q is 10% and 2% from the edge
)
def test_the_fit_of_two_prices_reproduces_the_group_frequencies(q0, n0, q1, n1):
    """Saturated model: sigmoid(a - b) = q0 and sigmoid(a + b) = q1, so a = (logit q0 + logit q1) /
    2 and b = (logit q1 - logit q0) / 2. The counts per group play no role."""
    a, b = fit_recalibration(*saturated(q0, n0, q1, n1))
    assert a == pytest.approx((logit(q0) + logit(q1)) / 2.0, abs=1e-8)
    assert b == pytest.approx((logit(q1) - logit(q0)) / 2.0, abs=1e-8)


def test_the_fit_of_the_worked_example():
    """3 of 10 outcomes at sigmoid(-1) and 7 of 10 at sigmoid(1): (a, b) = (0, ln(7/3))."""
    a, b = fit_recalibration(*saturated(0.3, 10, 0.7, 10))
    assert a == pytest.approx(0.0, abs=1e-9) and b == pytest.approx(np.log(7.0 / 3.0), abs=1e-9)
    assert isinstance(a, float) and isinstance(b, float)


def score_and_information(coef, price, outcome, clip=0.01):
    """Score ``sum((y - p_hat)(1, x))`` and information ``sum(p_hat (1 - p_hat) (1, x)'(1, x))``
    written out independently of the module (explicit design matrix)."""
    x = logit(np.clip(price, clip, 1 - clip))
    design = np.column_stack([np.ones_like(x), x])
    p_hat = 1.0 / (1.0 + np.exp(-design @ coef))
    return design.T @ (outcome - p_hat), design.T @ (design * (p_hat * (1 - p_hat))[:, None])


def noisy_data(rng, n, a0=-0.3, b0=1.4):
    """Prices from a normal logit and outcomes with a floor and ceiling of 10% so that the maximum
    likelihood estimate is finite and no sample is (nearly) separated."""
    z = rng.normal(0.0, 1.5, n)
    price = sigmoid(z)
    outcome = (rng.random(n) < 0.1 + 0.8 * sigmoid(a0 + b0 * z)).astype(int)
    return price, outcome


@pytest.mark.parametrize("seed, n", [(1, 30), (2, 100), (3, 1000), (4, 12), (5, 5000)])
def test_the_fit_solves_the_score_equations(seed, n):
    price, outcome = noisy_data(np.random.default_rng(seed), n)
    a, b = fit_recalibration(price, outcome)
    score, info = score_and_information(np.array([a, b]), price, outcome)
    assert np.abs(score).max() < 1e-8
    assert np.all(np.linalg.eigvalsh(info) > 0.0)  # a maximum of the concave log-likelihood


@pytest.mark.parametrize("seed", range(6))
def test_the_fit_equals_scipy_bfgs_on_the_negative_log_likelihood(seed):
    """An independent algorithm (quasi-Newton, no Hessian) reaches the same maximiser."""
    rng = np.random.default_rng(100 + seed)
    n, a0, b0 = int(rng.integers(30, 400)), rng.normal(), rng.uniform(0.3, 2.0)
    price, outcome = noisy_data(rng, n, a0=a0, b0=b0)
    x = logit(np.clip(price, 0.01, 0.99))

    def fun(c):
        z = c[0] + c[1] * x
        p_hat = 1.0 / (1.0 + np.exp(-z))
        gradient = np.array([np.sum(outcome - p_hat), np.sum(x * (outcome - p_hat))])
        return -(outcome @ z - np.log1p(np.exp(z)).sum()), -gradient

    ref = optimize.minimize(fun, np.zeros(2), jac=True, method="BFGS", options={"gtol": 1e-10})
    np.testing.assert_allclose(fit_recalibration(price, outcome), ref.x, atol=1e-6)


@pytest.mark.parametrize("beta", [1.0, 0.7, 0.5, 1.5])
def test_the_population_coefficients_are_zero_and_one_over_beta(beta):
    """True probability sigmoid(x), price sigmoid(beta x), x ~ N(0, 1.5^2): the fit recovers
    (0, 1 / beta) within 4 standard errors from the inverse information, and the model is correct
    (the maximum likelihood fit is then consistent). The clip is moved out of the way."""
    rng = np.random.default_rng(7)
    n = 300_000
    x = rng.normal(0.0, 1.5, n)
    outcome = (rng.random(n) < sigmoid(x)).astype(int)
    price = sigmoid(beta * x)
    a, b = fit_recalibration(price, outcome, clip=1e-9)
    _, info = score_and_information(np.array([a, b]), price, outcome, clip=1e-9)
    se = np.sqrt(np.diag(np.linalg.inv(info)))
    assert abs(a - 0.0) < Z * se[0], (a, se[0])
    assert abs(b - 1.0 / beta) < Z * se[1], (b, 1.0 / beta, se[1])
    assert se[1] < 0.01  # the test has power: it separates beta = 1 from beta = 0.9


def test_fit_is_not_changed_by_duplicating_every_row_or_reordering():
    price, outcome = noisy_data(np.random.default_rng(11), 80)
    base = np.array(fit_recalibration(price, outcome))
    np.testing.assert_allclose(fit_recalibration(np.tile(price, 3), np.tile(outcome, 3)), base,
                               atol=1e-9)  # the maximum likelihood estimate ignores replication
    perm = np.random.default_rng(12).permutation(price.size)
    np.testing.assert_allclose(fit_recalibration(price[perm], outcome[perm]), base, atol=1e-10)


def test_flipping_both_sides_flips_the_intercept_and_keeps_the_slope():
    """P(1 - y = 1 | 1 - p) = 1 - sigmoid(a + b x) = sigmoid(-a + b (-x)): (a, b) -> (-a, b)."""
    price, outcome = noisy_data(np.random.default_rng(13), 200, a0=0.8)
    a, b = fit_recalibration(price, outcome)
    a2, b2 = fit_recalibration(1.0 - price, 1 - outcome)
    assert abs(a) > 0.1  # the intercept is not close to zero, so the check has power
    assert a2 == pytest.approx(-a, abs=1e-9) and b2 == pytest.approx(b, abs=1e-9)


def test_the_clip_applies_to_prices_outside_it():
    price, outcome = noisy_data(np.random.default_rng(14), 120)
    price[:6] = [0.0, 0.002, 0.009, 0.991, 0.999, 1.0]  # outside [0.01, 0.99]
    manual = np.clip(price, 0.01, 0.99)
    assert fit_recalibration(price, outcome) == fit_recalibration(manual, outcome)
    assert fit_recalibration(price, outcome) != fit_recalibration(price, outcome, clip=0.05)
    wide = fit_recalibration(price, outcome, clip=0.05)
    assert wide == fit_recalibration(np.clip(price, 0.05, 0.95), outcome)


# ------------------------------------------------------------------ the fit: failures and limits
def test_a_single_outcome_class_is_an_error():
    price = np.linspace(0.1, 0.9, 20)
    for value in (0, 1):
        with pytest.raises(RecalibrationError, match="only one outcome class"):
            fit_recalibration(price, np.full(20, value))


def test_fewer_than_ten_observations_is_an_error():
    price, outcome = noisy_data(np.random.default_rng(15), 60)
    with pytest.raises(RecalibrationError, match="at least 10"):
        fit_recalibration(price[:9], outcome[:9])
    assert len(fit_recalibration(price[:10], outcome[:10])) == 2  # ten is enough, with both classes
    assert 0 < outcome[:10].sum() < 10


def test_equal_prices_are_an_error():
    outcome = np.arange(20) % 2
    with pytest.raises(RecalibrationError, match="equal"):
        fit_recalibration(np.full(20, 0.4), outcome)
    with pytest.raises(RecalibrationError, match="equal"):
        fit_recalibration(np.r_[np.full(10, 0.001), np.full(10, 0.0)], outcome)  # clipped to one


def test_complete_separation_is_an_error_and_not_a_number():
    price = np.r_[np.full(10, 0.2), np.full(10, 0.8)]
    outcome = np.r_[np.zeros(10), np.ones(10)].astype(int)
    with pytest.raises(RecalibrationError, match="did not converge"):
        fit_recalibration(price, outcome)
    ordered = np.sort(np.random.default_rng(16).uniform(0.05, 0.95, 40))
    with pytest.raises(RecalibrationError, match="separate"):
        fit_recalibration(ordered, (ordered > 0.5).astype(int))  # a threshold rule: separated


def test_too_few_iterations_is_an_error_and_enough_is_not():
    price, outcome = noisy_data(np.random.default_rng(17), 200)
    with pytest.raises(RecalibrationError, match="did not converge in 2 iterations"):
        fit_recalibration(price, outcome, max_iter=2)
    a, b = fit_recalibration(price, outcome, max_iter=30)  # Newton needs only a handful
    assert (a, b) == fit_recalibration(price, outcome)


def test_the_errors_are_value_errors():
    assert issubclass(RecalibrationError, ValueError)
    with pytest.raises(ValueError):
        fit_recalibration(np.full(20, 0.4), np.arange(20) % 2)


def test_fit_rejects_bad_input():
    good = np.linspace(0.1, 0.9, 20)
    y = np.arange(20) % 2
    with pytest.raises(ValueError, match="outcome"):
        fit_recalibration(good, np.where(y == 1, 2, 0))
    with pytest.raises(ValueError, match="price"):
        fit_recalibration(np.r_[good[:-1], 1.1], y)
    with pytest.raises(ValueError, match="price"):
        fit_recalibration(np.r_[good[:-1], np.nan], y)
    with pytest.raises(ValueError, match="same length"):
        fit_recalibration(good, y[:-1])
    with pytest.raises(ValueError, match="clip"):
        fit_recalibration(good, y, clip=0.6)


# ------------------------------------------ population Brier scores by quadrature
def population_scores(beta, sd=1.5, nodes=120):
    """Expected Brier scores of the raw price ``sigmoid(beta x)`` and of the recalibrated price
    ``sigmoid(x)`` (the map ``(0, 1 / beta)``) when ``y ~ Bernoulli(sigmoid(x))`` and
    ``x ~ N(0, sd^2)``: ``E[q (1 - p)^2 + (1 - q) p^2]`` with Gauss-Hermite nodes for the weight
    ``exp(-t^2 / 2)``."""
    t, w = hermegauss(nodes)
    w = w / np.sqrt(2.0 * np.pi)
    q = sigmoid(sd * t)
    p = sigmoid(beta * sd * t)
    raw = np.sum(w * (q * (1 - p) ** 2 + (1 - q) * p**2))
    recal = np.sum(w * q * (1 - q))  # the recalibrated probability is q itself
    return raw, recal


def test_the_quadrature_weights_are_a_probability_distribution():
    t, w = hermegauss(120)
    w = w / np.sqrt(2.0 * np.pi)
    assert w.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.sum(w * t**2) == pytest.approx(1.0, abs=1e-12)  # variance one
    assert np.sum(w * sigmoid(1.5 * t)) == pytest.approx(0.5, abs=1e-12)  # the symmetric mean


def test_population_skill_is_zero_at_beta_one_and_positive_below():
    raw, recal = population_scores(1.0)
    assert 1.0 - recal / raw == pytest.approx(0.0, abs=1e-14)
    skill = [1.0 - population_scores(b)[1] / population_scores(b)[0] for b in (0.9, 0.8, 0.7, 0.5)]
    assert all(s > 0.0 for s in skill) and skill == sorted(skill)  # more distortion, more skill
    assert skill[2] < 0.05  # a 30% distortion of the slope is worth only a few percent of Brier


@pytest.mark.parametrize("beta", [0.7, 0.9, 1.4])
def test_population_skill_agrees_with_a_large_sample(beta):
    """Sample Brier scores of the raw price and of the recalibrated one (the map (0, 1/beta), then
    the fitted map) on 2 million events agree with the quadrature within 4 delta-method standard
    errors of the skill score."""
    rng = np.random.default_rng(21)
    n = 2_000_000
    x = rng.normal(0.0, 1.5, n)
    y = (rng.random(n) < sigmoid(x)).astype(int)
    price = sigmoid(beta * x)
    raw_pop, recal_pop = population_scores(beta)
    target = 1.0 - recal_pop / raw_pop
    a, b = fit_recalibration(price, y, clip=1e-9)
    for coef in ((0.0, 1.0 / beta), (a, b)):
        recal = recalibrate(price, *coef, clip=1e-9)
        u, v = (recal - y) ** 2, (price - y) ** 2
        ratio = u.mean() / v.mean()
        # delta method for 1 - mean(u) / mean(v)
        se = np.sqrt(np.var(u - ratio * v, ddof=1) / n) / v.mean()
        assert abs((1.0 - ratio) - target) < Z * se, (beta, coef, 1.0 - ratio, target, se)
        assert abs(u.mean() - recal_pop) < Z * u.std(ddof=1) / np.sqrt(n)
        assert abs(v.mean() - raw_pop) < Z * v.std(ddof=1) / np.sqrt(n)
    assert se < 0.1 * target  # the comparison is tight enough to see a 10% error in the skill


# ------------------------------------------------------------------------ walk-forward: by hand
def hand_data():
    """Eight releases with four markets each: two at price sigmoid(-1) and two at sigmoid(1). The
    outcomes per release are (lo, lo, hi, hi). With ``min_events = 3`` releases and
    ``refit_every = 2`` the fits use releases 0-2 (12 observations), 0-4 (20) and 0-6 (28); the
    price has only two values, so each fit is the saturated one worked out in ``HAND_FITS``."""
    outcomes = [
        (0, 1, 1, 1), (0, 0, 1, 0), (1, 0, 1, 1), (0, 0, 1, 1),
        (1, 1, 0, 1), (0, 1, 1, 1), (0, 0, 1, 0), (1, 1, 1, 1),
    ]
    price = np.tile([LO, LO, HI, HI], 8)
    outcome = np.array(outcomes).ravel()
    release_ts = np.repeat(1000.0 * np.arange(1, 9), 4)
    return price, outcome, release_ts


# (lo frequency, hi frequency) in each fit window: releases 0-2: 2/6 and 5/6; 0-4: 4/10 and 8/10;
# 0-6: 5/14 and 11/14. a = (logit q0 + logit q1) / 2, b = (logit q1 - logit q0) / 2.
HAND_FITS = [
    ((np.log(1 / 2) + np.log(5)) / 2, (np.log(5) - np.log(1 / 2)) / 2, 12),
    ((np.log(2 / 3) + np.log(4)) / 2, (np.log(4) - np.log(2 / 3)) / 2, 20),
    ((np.log(5 / 9) + np.log(11 / 3)) / 2, (np.log(11 / 3) - np.log(5 / 9)) / 2, 28),
]


def test_walk_forward_by_hand_with_several_markets_per_release():
    """The first scored release is the 4th (3 releases, 12 observations, are the history), the fit
    is refreshed after two further releases (releases 3-4 share one fit, 5-6 the next, 7 the last),
    and every market of a release has the same coefficients."""
    price, outcome, release_ts = hand_data()
    got = walk_forward_recalibration(price, outcome, release_ts, min_events=3, refit_every=2)
    assert list(got.columns) == COLUMNS
    assert len(got) == 20  # releases 3..7, four markets each
    assert got["release_ts"].tolist() == [1000.0 * k for k in (4, 5, 6, 7, 8) for _ in range(4)]
    for block, (a, b, n_fit) in zip([range(0, 8), range(8, 16), range(16, 20)], HAND_FITS,
                                    strict=True):
        rows = got.iloc[list(block)]
        assert (rows["n_fit"] == n_fit).all()
        np.testing.assert_allclose(rows["a"], a, atol=1e-9)
        np.testing.assert_allclose(rows["b"], b, atol=1e-9)
        np.testing.assert_allclose(rows["recal"], sigmoid(a + b * logit(rows["price"])), atol=1e-9)
    # within a release the rows are in the order of price, then outcome
    assert got["price"].tolist() == [LO, LO, HI, HI] * 5
    assert got["outcome"].tolist() == [0, 0, 1, 1,  # release 3: lo (0, 0), hi (1, 1)
                                       1, 1, 0, 1,  # release 4: lo (1, 1), hi (1, 0) sorted
                                       0, 1, 1, 1,  # release 5: lo (0, 1), hi (1, 1)
                                       0, 0, 0, 1,  # release 6: lo (0, 0), hi (1, 0) sorted
                                       1, 1, 1, 1]  # release 7
    assert got["recal"].between(0.0, 1.0).all()
    assert got.dtypes["outcome"] == np.int64 and got.dtypes["n_fit"] == np.int64
    assert list(got.index) == list(range(20))


def test_the_release_itself_and_later_releases_never_enter_its_fit():
    """Release 3 is the first scored one. Replace the outcomes and prices of release 3 and all later
    releases: the coefficients of release 3 (fit on releases 0-2) do not change, its own rows do."""
    price, outcome, release_ts = hand_data()
    base = walk_forward_recalibration(price, outcome, release_ts, min_events=3, refit_every=2)
    p2, y2 = price.copy(), outcome.copy()
    p2[12:] = np.tile([0.9, 0.1, 0.6, 0.4], 5)
    y2[12:] = np.tile([1, 0, 0, 1], 5)
    changed = walk_forward_recalibration(p2, y2, release_ts, min_events=3, refit_every=2)
    first = slice(0, 4)  # the four markets of release 3
    for col in ("a", "b", "n_fit"):
        assert changed[col].iloc[first].tolist() == base[col].iloc[first].tolist()
    assert not np.allclose(changed["recal"].iloc[first], base["recal"].iloc[first])
    assert not np.allclose(changed["a"].iloc[8:12], base["a"].iloc[8:12])  # later fits do see them


def test_min_events_counts_releases_and_not_observations():
    price, outcome, release_ts = hand_data()
    by_release = walk_forward_recalibration(price, outcome, release_ts, min_events=3, refit_every=2)
    assert by_release["release_ts"].min() == 4000.0  # not 1000 * (3 / 4 + 1): three observations
    seven = walk_forward_recalibration(price, outcome, release_ts, min_events=7, refit_every=5)
    assert len(seven) == 4 and seven["n_fit"].tolist() == [28] * 4  # only release 7 is scored
    assert len(walk_forward_recalibration(price, outcome, release_ts, min_events=8)) == 0


def test_refit_every_counts_releases_and_not_observations():
    price, outcome, release_ts = hand_data()
    got = walk_forward_recalibration(price, outcome, release_ts, min_events=3, refit_every=1)
    per_release = got.groupby("release_ts")["n_fit"].agg(["min", "max"])
    assert per_release["min"].tolist() == per_release["max"].tolist() == [12, 16, 20, 24, 28]
    one_fit = walk_forward_recalibration(price, outcome, release_ts, min_events=3, refit_every=50)
    assert one_fit["n_fit"].unique().tolist() == [12] and one_fit["a"].nunique() == 1


# --------------------------------------------------------- walk-forward: independent plain loop
def slow_walk_forward(price, outcome, ts, min_events, refit_every, gap_hours=0.0):
    """The rule of the docstring one release at a time with plain loops: walk through the sorted
    distinct release times; a release is scored when at least ``min_events`` releases lie strictly
    before its price time (``gap_hours`` earlier); at the first scored release and whenever
    ``refit_every`` releases have passed since the last fit, fit on all rows strictly before the
    price time of the release."""
    releases = sorted(set(ts.tolist()))
    gap = 3600.0 * gap_hours
    rows, last_fit, coef, n_fit = [], None, None, None
    for k, T in enumerate(releases):
        if sum(1 for r in releases if r < T - gap) < min_events:
            continue
        if last_fit is None or k - last_fit >= refit_every:
            earlier = ts < T - gap
            coef = fit_recalibration(price[earlier], outcome[earlier])
            last_fit, n_fit = k, int(earlier.sum())
        for i in np.flatnonzero(ts == T):
            rows.append((price[i], float(recalibrate(price[i], *coef)), int(outcome[i]), T, *coef,
                         n_fit))
    out = pd.DataFrame(rows, columns=COLUMNS)
    return out.sort_values(["release_ts", "price", "outcome"], kind="stable").reset_index(drop=True)


def random_panel(rng, n_releases, max_markets):
    """Prices and outcomes of several markets per release (an event effect in the price)."""
    counts = rng.integers(1, max_markets + 1, n_releases)
    release_ts = np.repeat(1.7e9 + 3600.0 * 24 * np.arange(n_releases), counts)
    shift = np.repeat(rng.normal(0.0, 1.0, n_releases), counts)
    price = sigmoid(0.8 * (rng.normal(0.0, 1.5, counts.sum()) + shift))
    outcome = (rng.random(counts.sum()) < sigmoid(logit(price) / 0.8)).astype(int)
    return price, outcome, release_ts


@pytest.mark.parametrize("min_events, refit_every, markets", [(20, 7, 1), (15, 10, 4), (12, 1, 3),
                                                             (30, 100, 2), (11, 3, 1)])
def test_walk_forward_equals_a_plain_loop(min_events, refit_every, markets):
    price, outcome, ts = random_panel(np.random.default_rng(31), 60, markets)
    got = walk_forward_recalibration(price, outcome, ts, min_events=min_events,
                                     refit_every=refit_every)
    want = slow_walk_forward(price, outcome, ts, min_events, refit_every)
    assert len(got) == len(want) > 0
    pd.testing.assert_frame_equal(got, want, check_dtype=False, check_exact=False, atol=1e-9,
                                  rtol=0)
    assert got["n_fit"].is_monotonic_increasing  # more history each time
    assert got["release_ts"].is_monotonic_increasing
    # the fit used for a row never contains an observation of its own release or a later one
    assert (got["n_fit"] <= np.searchsorted(np.sort(ts), got["release_ts"], side="left")).all()


@pytest.mark.parametrize("gap_hours, markets", [(12.0, 2), (24.0, 2), (30.0, 3), (80.0, 1)])
def test_walk_forward_with_a_gap_equals_a_plain_loop(gap_hours, markets):
    """Daily releases. A gap of 12 hours changes nothing; 30 hours leaves out the release of the day
    before, 80 hours the three before it; a gap of exactly 24 hours leaves out the release of the
    day before too, because a release at the price time is not before it."""
    price, outcome, ts = random_panel(np.random.default_rng(42), 60, markets)
    got = walk_forward_recalibration(price, outcome, ts, min_events=15, refit_every=4,
                                     gap_hours=gap_hours)
    want = slow_walk_forward(price, outcome, ts, 15, 4, gap_hours)
    assert len(got) == len(want) > 0
    pd.testing.assert_frame_equal(got, want, check_dtype=False, check_exact=False, atol=1e-9,
                                  rtol=0)
    # no fit holds an observation at or after the price time of the first release of its block
    price_time = got["release_ts"] - 3600.0 * gap_hours
    assert (got["n_fit"] <= np.searchsorted(np.sort(ts), price_time, side="left")).all()
    plain = walk_forward_recalibration(price, outcome, ts, min_events=15, refit_every=4)
    if gap_hours < 24.0:
        pd.testing.assert_frame_equal(got, plain)
    else:  # the first scored release comes later
        assert got["release_ts"].min() > plain["release_ts"].min()


def test_walk_forward_by_hand_with_a_gap():
    """Releases 1000 s apart and a gap of half an hour (1800 s), so the fit for the release at T
    holds the releases at least 2000 s before it. With ``min_events = 3`` the first scored release
    is the 5th (the first three lie before its price time, the 4th is inside the gap); the next
    fit is at the 7th release and holds releases 1 to 5."""
    price, outcome, release_ts = hand_data()
    got = walk_forward_recalibration(price, outcome, release_ts, min_events=3, refit_every=2,
                                     gap_hours=0.5)
    assert list(got.columns) == COLUMNS and len(got) == 16
    assert got["release_ts"].tolist() == [1000.0 * k for k in (5, 6, 7, 8) for _ in range(4)]
    for block, (a, b, n_fit) in zip([range(0, 8), range(8, 16)], HAND_FITS[:2], strict=True):
        rows = got.iloc[list(block)]
        assert (rows["n_fit"] == n_fit).all()
        np.testing.assert_allclose(rows["a"], a, atol=1e-9)
        np.testing.assert_allclose(rows["b"], b, atol=1e-9)
    plain = walk_forward_recalibration(price, outcome, release_ts, min_events=3, refit_every=2)
    for release, with_gap, without in ((6000.0, 12, 20), (8000.0, 20, 28)):
        assert got.loc[got["release_ts"] == release, "n_fit"].tolist() == [with_gap] * 4
        assert plain.loc[plain["release_ts"] == release, "n_fit"].tolist() == [without] * 4


def test_no_release_is_scored_when_the_gap_leaves_too_few_releases_before_every_price_time():
    price, outcome, release_ts = hand_data()  # eight releases, 1000 s apart
    for min_events in (1, 3):  # 7200 s: the price time of the 8th release is 800, before them all
        got = walk_forward_recalibration(price, outcome, release_ts, min_events=min_events,
                                         refit_every=2, gap_hours=2.0)
        assert len(got) == 0 and list(got.columns) == COLUMNS
    some = walk_forward_recalibration(price, outcome, release_ts, min_events=3, refit_every=2,
                                      gap_hours=1.0)  # 3600 s: before 7000 - 3600 = 3400 lie three
    assert some["release_ts"].tolist() == [7000.0] * 4 + [8000.0] * 4
    assert some["n_fit"].tolist() == [12] * 8


def test_an_outcome_inside_the_gap_does_not_enter_the_fit():
    """Daily releases and a gap of 30 hours: the outcome of the release of the day before the start
    of a block was not known at the price time of that block. Flip it: the block does not change.
    With a gap of 12 hours the same flip changes the block (the test has power), and a later block
    does see the flip in both runs."""
    price, outcome, ts = random_panel(np.random.default_rng(43), 60, 3)
    releases = np.unique(ts)
    flipped = outcome.copy()
    flipped[ts == releases[24]] = 1 - flipped[ts == releases[24]]

    def block(frame, first, last):
        inside = (frame["release_ts"] >= releases[first]) & (frame["release_ts"] <= releases[last])
        return frame[inside]

    # min_events is 19 with the gap of 30 hours, so that both runs have blocks from release 20
    for gap, min_events, leaks in ((30.0, 19, False), (12.0, 20, True)):
        base = walk_forward_recalibration(price, outcome, ts, min_events=min_events, refit_every=5,
                                          gap_hours=gap)
        changed = walk_forward_recalibration(price, flipped, ts, min_events=min_events,
                                             refit_every=5, gap_hours=gap)
        assert block(base, 20, 24)["a"].nunique() == 1  # the block of releases 20 to 24
        same = block(changed, 25, 29)[["a", "b", "recal", "n_fit"]].reset_index(drop=True)
        was = block(base, 25, 29)[["a", "b", "recal", "n_fit"]].reset_index(drop=True)
        if leaks:
            assert not np.allclose(same["a"], was["a"]) and not np.allclose(same["b"], was["b"])
        else:
            pd.testing.assert_frame_equal(same, was, check_exact=True)
        assert not np.allclose(block(changed, 30, 34)["a"], block(base, 30, 34)["a"])


def test_the_default_schedule_is_fifty_releases_then_a_refit_every_ten():
    price, outcome, ts = random_panel(np.random.default_rng(32), 95, 1)
    got = walk_forward_recalibration(price, outcome, ts)
    n_fit = [50] * 10 + [60] * 10 + [70] * 10 + [80] * 10 + [90] * 5  # releases 51..95
    assert got["n_fit"].tolist() == n_fit
    assert got["a"].nunique() == 5


def test_default_walk_forward_with_one_market_per_release_has_one_row_per_release():
    n = 120
    rng = np.random.default_rng(33)
    x = rng.normal(0.0, 1.5, n)
    y = (rng.random(n) < sigmoid(x)).astype(int)
    ts = 1.7e9 + 604800.0 * np.arange(n)
    got = walk_forward_recalibration(sigmoid(0.8 * x), y, ts)
    assert len(got) == 70 and got["release_ts"].tolist() == ts[50:].tolist()


# ---------------------------------------------------------------- walk-forward: look-ahead guards
def test_changing_later_releases_does_not_change_earlier_rows():
    price, outcome, ts = random_panel(np.random.default_rng(34), 80, 3)
    base = walk_forward_recalibration(price, outcome, ts, min_events=20, refit_every=5)
    cut = np.sort(np.unique(ts))[45]  # alter everything strictly after the 46th release
    later = ts > cut
    rng = np.random.default_rng(35)
    p2, y2 = price.copy(), outcome.copy()
    p2[later] = rng.uniform(0.02, 0.98, later.sum())
    y2[later] = rng.integers(0, 2, later.sum())
    changed = walk_forward_recalibration(p2, y2, ts, min_events=20, refit_every=5)
    early = base["release_ts"] <= cut
    assert early.sum() > 50
    pd.testing.assert_frame_equal(changed[changed["release_ts"] <= cut], base[early])
    assert not np.allclose(changed.loc[~early, "recal"], base.loc[~early, "recal"])  # has power
    assert not np.allclose(changed.loc[~early, "a"], base.loc[~early, "a"])


def test_deleting_later_releases_does_not_change_earlier_rows():
    price, outcome, ts = random_panel(np.random.default_rng(36), 80, 3)
    base = walk_forward_recalibration(price, outcome, ts, min_events=20, refit_every=5)
    cut = np.sort(np.unique(ts))[50]
    keep = ts <= cut
    shorter = walk_forward_recalibration(price[keep], outcome[keep], ts[keep], min_events=20,
                                         refit_every=5)
    # the last scored block of the shorter run may be cut short, but its coefficients are the same
    pd.testing.assert_frame_equal(shorter, base[base["release_ts"] <= cut].reset_index(drop=True))
    assert len(shorter) < len(base)


# ----------------------------------------------------- walk-forward: order, ties, clusters, errors
def test_shuffling_the_input_gives_the_same_frame():
    price, outcome, ts = random_panel(np.random.default_rng(37), 70, 4)
    price[5] = price[4]  # exact ties in release_ts, price and outcome
    outcome[5] = outcome[4]
    ts[5] = ts[4]
    labels = np.array([f"mkt-{i % 3}" for i in range(price.size)])  # tied rows differ in label
    base = walk_forward_recalibration(price, outcome, ts, min_events=20, refit_every=5,
                                      cluster=labels)
    for seed in range(3):
        perm = np.random.default_rng(seed).permutation(price.size)
        shuffled = walk_forward_recalibration(price[perm], outcome[perm], ts[perm], min_events=20,
                                              refit_every=5, cluster=labels[perm])
        pd.testing.assert_frame_equal(shuffled, base, check_exact=True)
    plain = walk_forward_recalibration(price, outcome, ts, min_events=20, refit_every=5)
    pd.testing.assert_frame_equal(base.drop(columns="cluster"), plain)  # a label changes nothing
    key = ["release_ts", "price"]
    assert base.sort_values(key, kind="stable").index.is_monotonic_increasing  # already sorted


def test_cluster_labels_follow_their_rows():
    price, outcome, ts = random_panel(np.random.default_rng(38), 60, 3)
    labels = np.array([f"row{i:04d}" for i in range(price.size)])  # a unique label per input row
    got = walk_forward_recalibration(price, outcome, ts, min_events=20, refit_every=5,
                                     cluster=labels)
    assert list(got.columns) == [*COLUMNS, "cluster"]
    source = got["cluster"].str[3:].astype(int).to_numpy()
    np.testing.assert_array_equal(got["price"], price[source])
    np.testing.assert_array_equal(got["outcome"], outcome[source])
    np.testing.assert_array_equal(got["release_ts"], ts[source])
    ints = walk_forward_recalibration(price, outcome, ts, min_events=20, refit_every=5,
                                      cluster=np.arange(price.size))
    assert ints["cluster"].tolist() == source.tolist() and ints["cluster"].dtype.kind == "i"
    assert "cluster" not in walk_forward_recalibration(price, outcome, ts, min_events=20)


def test_walk_forward_returns_an_empty_frame_without_enough_releases():
    price, outcome, ts = random_panel(np.random.default_rng(39), 25, 1)
    got = walk_forward_recalibration(price, outcome, ts, min_events=25)
    assert len(got) == 0 and list(got.columns) == COLUMNS
    assert len(walk_forward_recalibration(price, outcome, ts, min_events=30)) == 0


def test_walk_forward_refuses_bad_arguments():
    price, outcome, ts = random_panel(np.random.default_rng(40), 30, 1)
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="min_events"):
            walk_forward_recalibration(price, outcome, ts, min_events=bad)
        with pytest.raises(ValueError, match="refit_every"):
            walk_forward_recalibration(price, outcome, ts, refit_every=bad)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="gap_hours"):
            walk_forward_recalibration(price, outcome, ts, min_events=12, gap_hours=bad)
    with pytest.raises(ValueError, match="release_ts"):
        walk_forward_recalibration(price, outcome, ts[:-1], min_events=12)
    with pytest.raises(ValueError, match="release_ts"):
        walk_forward_recalibration(price, outcome, np.where(ts == ts[3], np.nan, ts), min_events=12)
    with pytest.raises(ValueError, match="outcome"):
        walk_forward_recalibration(price, np.where(outcome == 1, 3, 0), ts, min_events=12)
    with pytest.raises(ValueError, match="cluster"):
        walk_forward_recalibration(price, outcome, ts, min_events=12, cluster=np.arange(5))
    with pytest.raises(ValueError, match="cluster"):
        walk_forward_recalibration(price, outcome, ts, min_events=12,
                                   cluster=np.where(np.arange(30) == 2, None, "a"))


def test_a_failed_fit_names_the_release():
    price, outcome, ts = random_panel(np.random.default_rng(41), 40, 1)
    outcome[:15] = 0  # no event in the first fit window
    message = r"release at \d+ failed: .*only one outcome class"
    with pytest.raises(RecalibrationError, match=message):
        walk_forward_recalibration(price, outcome, ts, min_events=12, refit_every=5)
    with pytest.raises(RecalibrationError, match="at least 10"):  # 5 releases, one market each
        walk_forward_recalibration(price, outcome, ts, min_events=5, refit_every=5)


# ------------------------------------------------------------------------------ skill_interval
def skill_frame(price, recal, outcome):
    return pd.DataFrame({"price": price, "recal": recal, "outcome": outcome})


def noisy_prices(rng, n, beta=0.7):
    """Price ``sigmoid(beta x)``, its recalibration by the true map, and the outcome."""
    x = rng.normal(0.0, 1.5, n)
    y = (rng.random(n) < sigmoid(x)).astype(int)
    price = sigmoid(beta * x)
    return price, recalibrate(price, 0.0, 1.0 / beta), y


def slow_skill_bootstrap(frame, labels, B, seed):
    """Explicit resampling: list the rows of each drawn cluster (``w[g]`` times cluster ``g`` in
    sorted label order) and compute the skill of the pooled rows with plain numpy."""
    names = sorted(set(labels))
    weights = cluster_weights(len(names), B, np.random.default_rng(seed))
    out = []
    for w in weights:
        rows = np.concatenate([np.flatnonzero(labels == names[g]) for g in range(len(names))
                               for _ in range(int(w[g]))])
        raw = np.mean((frame["price"].to_numpy()[rows] - frame["outcome"].to_numpy()[rows]) ** 2)
        recal = np.mean((frame["recal"].to_numpy()[rows] - frame["outcome"].to_numpy()[rows]) ** 2)
        out.append(1.0 - recal / raw)
    return np.array(out)


def test_skill_interval_by_hand_and_against_an_explicit_resample():
    frame = skill_frame([0.8, 0.6, 0.3, 0.5, 0.7, 0.2], [0.9, 0.7, 0.2, 0.5, 0.9, 0.1],
                        [1, 1, 0, 1, 1, 0])
    labels = np.array(["b", "a", "c", "a", "b", "c"])  # three clusters of two rows, unsorted
    out = skill_interval(frame, labels, B=400, seed=5)
    raw = np.mean([0.04, 0.16, 0.09, 0.25, 0.09, 0.04])
    recal = np.mean([0.01, 0.09, 0.04, 0.25, 0.01, 0.01])
    assert out["brier_raw"] == pytest.approx(raw) and out["brier_recal"] == pytest.approx(recal)
    assert out["bss"] == pytest.approx(1.0 - recal / raw) == pytest.approx(
        brier_skill(frame["price"], frame["recal"], frame["outcome"]))
    assert (out["n"], out["n_clusters"], out["B"], out["seed"]) == (6, 3, 400, 5)
    assert set(out) == {"bss", "ci", "brier_raw", "brier_recal", "p_one_sided", "n", "n_clusters",
                        "B", "seed"}
    boot = slow_skill_bootstrap(frame, labels, 400, 5)
    assert out["ci"] == pytest.approx(percentile_interval(boot), abs=1e-12)
    assert out["p_one_sided"] == pytest.approx(max(np.mean(boot <= 0.0), 1.0 / 401.0), abs=1e-15)
    narrow = skill_interval(frame, labels, B=400, seed=5, level=0.5)
    boot_narrow = percentile_interval(boot, 0.5)
    assert narrow["ci"] == pytest.approx(boot_narrow, abs=1e-12)  # the level is used


def test_skill_interval_is_reproducible_and_depends_on_the_seed():
    frame = skill_frame(*noisy_prices(np.random.default_rng(51), 120))
    labels = np.arange(120)
    first = skill_interval(frame, labels, B=200, seed=1)
    assert skill_interval(frame, labels, B=200, seed=1) == first
    other = skill_interval(frame, labels, B=200, seed=2)
    assert other["ci"] != first["ci"] and other["bss"] == first["bss"]  # the estimate has no seed
    default = skill_interval(frame, labels)
    assert default["B"] == 10_000 and default["seed"] == 20261113


def same_summary(got, want):
    """Equal up to the rounding of a sum taken in another order (1e-12): the counts and the seed
    exactly, every score, end point and p-value to 1e-12."""
    assert got.keys() == want.keys()
    for key in got:
        if key in ("n", "n_clusters", "B", "seed"):
            assert got[key] == want[key]
        else:
            np.testing.assert_allclose(got[key], want[key], rtol=0.0, atol=1e-12)


def test_row_order_and_label_type_do_not_change_the_interval():
    frame = skill_frame(*noisy_prices(np.random.default_rng(52), 90))
    labels = np.repeat(np.arange(30), 3)
    base = skill_interval(frame, labels, B=300, seed=3)
    perm = np.random.default_rng(1).permutation(90)
    same_summary(skill_interval(frame.iloc[perm].reset_index(drop=True), labels[perm], B=300,
                                seed=3), base)
    as_text = np.array([f"ev{g:03d}" for g in labels])  # the same sorted order as the integers
    same_summary(skill_interval(frame, as_text, B=300, seed=3), base)
    same_summary(skill_interval(frame, pd.Series(labels), B=300, seed=3), base)
    # the draw follows the sorted labels: renaming the clusters in reverse order changes it
    reversed_names = np.array([f"z{29 - g:02d}" for g in labels])
    assert skill_interval(frame, reversed_names, B=300, seed=3)["ci"] != base["ci"]


def test_clustering_widens_the_interval_when_the_rows_of_a_cluster_share_an_outcome():
    """Forty events with six markets each. All markets of an event have the same outcome and the
    same price shock, so the rows are far from independent. Treating every row as its own cluster
    gives an interval that is clearly too narrow: the cluster interval is wider by more than 1.5."""
    rng = np.random.default_rng(53)
    g = 40
    event_x = rng.normal(0.0, 1.5, g)
    event_y = (rng.random(g) < sigmoid(event_x)).astype(int)
    x = np.repeat(event_x, 6) + rng.normal(0.0, 0.2, 6 * g)  # markets of an event move together
    y = np.repeat(event_y, 6)
    price = sigmoid(0.6 * x)
    frame = skill_frame(price, recalibrate(price, 0.0, 1.0 / 0.6), y)
    by_event = skill_interval(frame, np.repeat(np.arange(g), 6), B=2000, seed=7)
    by_row = skill_interval(frame, np.arange(6 * g), B=2000, seed=7)
    width = lambda r: r["ci"][1] - r["ci"][0]  # noqa: E731
    assert by_event["bss"] == by_row["bss"] and by_event["n_clusters"] == g
    assert by_row["n_clusters"] == 6 * g
    assert width(by_event) > 1.5 * width(by_row)


def test_one_sided_p_value_direction():
    rng = np.random.default_rng(54)
    price, recal, y = noisy_prices(rng, 2000, beta=0.4)  # large distortion: positive skill
    good = skill_interval(skill_frame(price, recal, y), np.arange(2000), B=500, seed=1)
    assert good["bss"] > 0.0 and good["ci"][0] > 0.0
    assert good["p_one_sided"] == pytest.approx(1.0 / 501.0)  # none at or below zero: the floor
    worse = skill_interval(skill_frame(recal, price, y), np.arange(2000), B=500, seed=1)
    assert worse["bss"] < 0.0 and worse["ci"][1] < 0.0 and worse["p_one_sided"] == 1.0
    same = skill_interval(skill_frame(price, price, y), np.arange(2000), B=500, seed=1)
    assert same["bss"] == 0.0 and same["ci"] == (0.0, 0.0) and same["p_one_sided"] == 1.0  # at zero


def test_skill_interval_covers_the_population_skill():
    """Check the checker: 300 samples of 400 independent events with the true map applied, B = 300.
    The 95% percentile interval contains the quadrature skill in 95% of samples, within 4 Monte
    Carlo standard errors (the percentile interval of a ratio is a little short at this size)."""
    R, n, B, beta = 300, 400, 300, 0.7
    raw_pop, recal_pop = population_scores(beta)
    target = 1.0 - recal_pop / raw_pop
    rng = np.random.default_rng(55)
    hit = 0
    for r in range(R):
        frame = skill_frame(*noisy_prices(rng, n, beta))
        lo, hi = skill_interval(frame, np.arange(n), B=B, seed=r)["ci"]
        hit += lo <= target <= hi
    mc_se = np.sqrt(0.95 * 0.05 / R)
    assert abs(hit / R - 0.95) < Z * mc_se, hit / R


def test_skill_interval_refuses_bad_input():
    frame = skill_frame([0.8, 0.6, 0.3, 0.5], [0.9, 0.7, 0.2, 0.5], [1, 1, 0, 1])
    labels = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="clusters"):
        skill_interval(frame, labels[:3])
    with pytest.raises(ValueError, match="clusters"):
        skill_interval(frame, np.array([0.0, np.nan, 1.0, 1.0]))
    with pytest.raises(ValueError, match="B"):
        skill_interval(frame, labels, B=0)
    with pytest.raises(ValueError, match="level"):
        skill_interval(frame, labels, level=1.2)
    with pytest.raises(ValueError, match="zero"):
        skill_interval(skill_frame([1.0, 0.0], [0.4, 0.6], [1, 0]), np.array([0, 1]))
    with pytest.raises(ValueError, match="recal"):
        skill_interval(skill_frame([0.8, 0.6], [0.9, 1.7], [1, 1]), np.array([0, 1]))
    with pytest.raises(ValueError, match="at least one"):
        skill_interval(frame.iloc[0:0], np.array([]))
    with pytest.raises(ValueError, match="raw Brier score of zero"):  # a replication draws only the
        skill_interval(skill_frame([1.0, 1.0, 0.5], [0.5, 0.5, 0.5], [1, 1, 1]),  # perfect clusters
                       np.array([0, 1, 2]), B=200, seed=1)
