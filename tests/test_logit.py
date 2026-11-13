"""Log-odds helpers and tick rounding (docs/model.md sections 2 and 3)."""

import numpy as np
import pytest

from kalshi_mm.logit import (
    information_step,
    local_price_vol,
    logit,
    martingale_drift,
    payoff_variance,
    round_to_tick,
    sigmoid,
)

CENTS = np.arange(0, 101)
PRICES = CENTS / 100.0


# ---------------------------------------------------------------------------- logit and sigmoid
def test_logit_known_values():
    assert logit(0.5) == 0.0
    assert logit(0.75) == pytest.approx(np.log(3.0), rel=1e-15)
    assert logit(0.25) == pytest.approx(-np.log(3.0), rel=1e-15)
    assert logit(0.0) == -np.inf and logit(1.0) == np.inf


def test_logit_outside_unit_interval_is_nan():
    assert np.isnan(logit(1.5)) and np.isnan(logit(-0.1)) and np.isnan(logit(np.nan))


def test_logit_is_antisymmetric():
    p = np.linspace(1e-6, 1.0 - 1e-6, 999)
    np.testing.assert_allclose(logit(1.0 - p), -logit(p), rtol=1e-9, atol=1e-9)


def test_round_trips():
    p = np.concatenate([np.linspace(1e-6, 1.0 - 1e-6, 9999), [1e-12, 0.5, 1.0 - 1e-12]])
    np.testing.assert_allclose(sigmoid(logit(p)), p, rtol=1e-12, atol=0)
    x = np.linspace(-12.0, 12.0, 4801)
    np.testing.assert_allclose(logit(sigmoid(x)), x, rtol=0, atol=1e-8)


def test_sigmoid_is_stable_for_large_arguments():
    with np.errstate(all="raise"):  # an overflow in exp(-x) would raise here
        big = sigmoid(np.array([-800.0, -745.0, -40.0, 0.0, 40.0, 800.0]))
    assert big[0] == 0.0 and big[-1] == 1.0
    assert big[1] == pytest.approx(5e-324, abs=5e-324)  # smallest subnormal, e^-745
    assert big[2] == pytest.approx(4.248354255291589e-18, rel=1e-14)  # e^-40 / (1 + e^-40)
    assert big[3] == 0.5 and big[4] == 1.0
    assert sigmoid(-30.0) == pytest.approx(np.exp(-30.0), rel=1e-12)  # tail relative accuracy


def test_sigmoid_symmetry_and_monotonicity():
    x = np.linspace(-30.0, 30.0, 6001)
    np.testing.assert_allclose(sigmoid(x) + sigmoid(-x), 1.0, rtol=0, atol=1e-15)
    assert np.all(np.diff(sigmoid(np.linspace(-20.0, 20.0, 4001))) > 0.0)


def test_scalars_in_scalars_out_and_shapes_kept():
    assert np.ndim(logit(0.3)) == 0 and np.ndim(sigmoid(0.3)) == 0
    assert np.ndim(information_step(0.3)) == 0 and np.ndim(round_to_tick(0.304)) == 0
    a = np.full((3, 4), 0.3)
    for f in (logit, sigmoid, information_step, payoff_variance):
        assert f(a).shape == (3, 4)
    assert round_to_tick(a).shape == (3, 4)


# ------------------------------------------------------------------------------ information step
def test_information_step_value_and_identities():
    pi = np.array([0.0, 0.05, 0.25, 0.5, 0.9])
    expected = np.array([0.0, 0.10008345855698263, np.log(1.25 / 0.75), np.log(3.0), np.log(19.0)])
    np.testing.assert_allclose(information_step(pi), expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(information_step(pi), 2.0 * np.arctanh(pi), rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(np.tanh(information_step(pi) / 2.0), pi, rtol=1e-13, atol=1e-15)


def test_information_step_is_the_glosten_milgrom_posterior():
    """One buy multiplies the odds by (1 + pi) / (1 - pi): the Bayes posterior of section 3."""
    p, pi = 0.3, 0.2
    prior_odds = p / (1.0 - p)
    after_buy = p * (1.0 + pi) / 2.0 / (p * (1.0 + pi) / 2.0 + (1.0 - p) * (1.0 - pi) / 2.0)
    assert sigmoid(logit(p) + information_step(pi)) == pytest.approx(after_buy, rel=1e-14)
    assert after_buy / (1 - after_buy) == pytest.approx(prior_odds * (1 + pi) / (1 - pi), rel=1e-12)
    after_sell = p * (1.0 - pi) / 2.0 / (p * (1.0 - pi) / 2.0 + (1.0 - p) * (1.0 + pi) / 2.0)
    assert sigmoid(logit(p) - information_step(pi)) == pytest.approx(after_sell, rel=1e-14)


def test_information_step_domain():
    assert information_step(1.0) == np.inf
    assert np.isnan(information_step(-0.1)) and np.isnan(information_step(1.1))
    assert np.isnan(information_step(np.nan))
    assert np.all(np.diff(information_step(np.linspace(0.0, 0.99, 100))) > 0.0)


# ----------------------------------------------------------------------- martingale drift and vols
def test_martingale_drift_closed_form():
    x = np.array([-3.0, -1.0, 0.0, 0.7, 4.0])
    p = sigmoid(x)
    np.testing.assert_allclose(martingale_drift(x, 0.2), 0.04 / 2.0 * (2.0 * p - 1.0), rtol=1e-13)
    assert martingale_drift(0.0, 0.5) == 0.0
    assert martingale_drift(2.0, 0.3) > 0.0 > martingale_drift(-2.0, 0.3)  # drifts away from 1/2
    assert martingale_drift(1.0, 0.2) == pytest.approx(0.02 * np.tanh(0.5), rel=1e-15)


def test_ito_identity_for_the_drift_closed_forms():
    """p'(x) mu + p''(x) sigma^2 / 2 = 0 with p' = p (1 - p) and p'' = p (1 - p) (1 - 2 p)."""
    x = np.linspace(-8.0, 8.0, 321)
    p = sigmoid(x)
    for sigma in (0.05, 0.2, 1.3):
        d1, d2 = p * (1 - p), p * (1 - p) * (1 - 2 * p)  # p'(x) and p''(x)
        residual = d1 * martingale_drift(x, sigma) + d2 * sigma**2 / 2
        np.testing.assert_allclose(residual, 0.0, atol=1e-16)


def test_ito_identity_with_finite_differences():
    h = 1e-4
    x = np.linspace(-6.0, 6.0, 49)
    sigma = 0.37
    d1 = (sigmoid(x + h) - sigmoid(x - h)) / (2 * h)
    d2 = (sigmoid(x + h) - 2 * sigmoid(x) + sigmoid(x - h)) / h**2
    drift_of_p = d1 * martingale_drift(x, sigma) + d2 * sigma**2 / 2
    np.testing.assert_allclose(drift_of_p, 0.0, atol=1e-7)
    # and the wrong sign of the drift does not satisfy it
    assert np.max(np.abs(d1 * -martingale_drift(x, sigma) + d2 * sigma**2 / 2)) > 1e-3


def test_local_price_vol_and_payoff_variance():
    assert local_price_vol(0.3, 0.2) == pytest.approx(0.042, rel=1e-14)
    assert payoff_variance(0.3) == pytest.approx(0.21, rel=1e-14)
    assert payoff_variance(0.5) == 0.25 and payoff_variance(0.0) == 0.0 == payoff_variance(1.0)
    p = np.linspace(0.01, 0.99, 99)
    np.testing.assert_allclose(payoff_variance(p), payoff_variance(1 - p), rtol=1e-13)
    # sigma p (1 - p) is the diffusion coefficient of sigmoid(x) when x has volatility sigma
    x = np.linspace(-5.0, 5.0, 41)
    h = 1e-6
    slope = (sigmoid(x + h) - sigmoid(x - h)) / (2 * h)
    np.testing.assert_allclose(0.2 * slope, local_price_vol(sigmoid(x), 0.2), rtol=1e-8)


# -------------------------------------------------------------------------------- tick rounding
@pytest.mark.parametrize("direction", ["nearest", "down", "up"])
def test_every_cent_is_a_fixed_point(direction):
    """Float noise must not move a price that is on the grid (0.07 / 0.01 is 7.000000000000001)."""
    np.testing.assert_array_equal(round_to_tick(PRICES, 0.01, direction), PRICES)
    assert round_to_tick(0.35, 0.01, direction) == 0.35


def test_naive_floor_would_fail_where_the_guard_matters():
    """Check the checker: the cents below really break a plain floor or ceil of price / tick."""
    plain_down = np.floor(PRICES / 0.01) * 0.01
    plain_up = np.ceil(PRICES / 0.01) * 0.01
    assert np.any(np.abs(plain_down - PRICES) > 1e-9)  # e.g. 0.29 floors to 0.28
    assert np.any(np.abs(plain_up - PRICES) > 1e-9)  # e.g. 0.07 ceils to 0.08


def test_between_ticks_and_ties():
    assert round_to_tick(0.351, 0.01, "down") == 0.35
    assert round_to_tick(0.351, 0.01, "up") == 0.36
    assert round_to_tick(0.351, 0.01, "nearest") == 0.35
    assert round_to_tick(0.356, 0.01, "nearest") == 0.36
    assert round_to_tick(0.3449999, 0.01, "nearest") == 0.34
    assert round_to_tick(0.345, 0.01, "nearest") == 0.35  # ties go up, also with float noise
    assert round_to_tick(0.355, 0.01, "nearest") == 0.36
    assert round_to_tick(0.995, 0.01, "up") == 1.0
    assert round_to_tick(0.0049, 0.01, "down") == 0.0


def test_other_ticks():
    assert round_to_tick(0.123, 0.005, "up") == 0.125
    assert round_to_tick(0.123, 0.005, "down") == 0.12
    assert round_to_tick(0.123456789, 1e-9, "down") == pytest.approx(0.123456789, abs=1e-10)
    assert round_to_tick(0.5, 0.25, "nearest") == 0.5 and round_to_tick(0.6, 0.25, "nearest") == 0.5


def test_rounding_properties_on_random_prices():
    p = np.random.default_rng(7).uniform(0.0, 1.0, 20000)
    down, up, near = (round_to_tick(p, 0.01, d) for d in ("down", "up", "nearest"))
    assert np.all(down <= p + 1e-12) and np.all(up >= p - 1e-12)
    assert np.all(p - down < 0.01 + 1e-12) and np.all(up - p < 0.01 + 1e-12)
    assert np.all(np.abs(near - p) <= 0.005 + 1e-12)
    for r in (down, up, near):
        assert np.all(np.abs(r / 0.01 - np.round(r / 0.01)) < 1e-9)  # on the grid
        assert np.all(r == np.round(r, 10))  # rounded to 10 decimals
    # up and down differ by exactly one tick off the grid
    off = np.abs(p / 0.01 - np.round(p / 0.01)) > 1e-6
    np.testing.assert_allclose((up - down)[off], 0.01, atol=1e-12)


def test_rounding_edge_inputs_and_errors():
    out = round_to_tick(np.array([np.nan, np.inf, -np.inf]), 0.01, "up")
    assert np.isnan(out[0]) and out[1] == np.inf and out[2] == -np.inf
    with pytest.raises(ValueError):
        round_to_tick(0.3, 0.0)
    with pytest.raises(ValueError):
        round_to_tick(0.3, -0.01)
    with pytest.raises(ValueError):
        round_to_tick(0.3, 0.01, "sideways")
