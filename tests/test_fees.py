"""Kalshi fee schedule (docs/model.md section 7)."""

from decimal import ROUND_CEILING, Decimal

import numpy as np
import pytest

from kalshi_mm.fees import MAKER_COEFF, TAKER_COEFF, kalshi_fee, maker_fee, taker_fee


def _exact_fee(count: int, cents: int, coeff: str, multiplier: str = "1") -> float:
    """Independent implementation in exact decimal arithmetic: ceiling to 0.0001 USD."""
    price = Decimal(cents) / 100
    raw = Decimal(multiplier) * Decimal(coeff) * count * price * (1 - price)
    return float((raw * 10000).to_integral_value(rounding=ROUND_CEILING) / 10000)


def test_constants():
    assert TAKER_COEFF == 0.07 and MAKER_COEFF == 0.0175


def test_hand_computed_values():
    # 0.07 * 10 * 0.5 * 0.5 = 0.175 exactly
    assert taker_fee(10, 0.50) == pytest.approx(0.1750, abs=1e-12)
    # 0.0175 * 10 * 0.25 = 0.04375 -> rounded up to 0.0438
    assert maker_fee(10, 0.50) == pytest.approx(0.0438, abs=1e-12)
    # 0.07 * 0.37 * 0.63 = 0.016317 -> 0.0164
    assert taker_fee(1, 0.37) == pytest.approx(0.0164, abs=1e-12)
    # 0.0175 * 0.25 = 0.004375 -> 0.0044
    assert maker_fee(1, 0.50) == pytest.approx(0.0044, abs=1e-12)
    # 0.07 * 100 * 0.1 * 0.9 = 0.63 exactly; in floats the product is 0.6300000000000001
    assert taker_fee(100, 0.10) == pytest.approx(0.6300, abs=1e-12)
    # 0.07 * 75 * 0.02 * 0.98 = 0.1029 exactly; a plain ceiling gives 0.1030
    assert taker_fee(75, 0.02) == pytest.approx(0.1029, abs=1e-12)
    assert np.ceil(0.07 * 75 * 0.02 * 0.98 / 1e-4) * 1e-4 > 0.1029 + 5e-5  # the guard matters here


def test_multiplier_scales_before_rounding():
    assert kalshi_fee(10, 0.5, TAKER_COEFF, multiplier=2.0) == pytest.approx(0.35, abs=1e-12)
    assert kalshi_fee(10, 0.5, TAKER_COEFF, multiplier=0.5) == pytest.approx(0.0875, abs=1e-12)
    assert kalshi_fee(10, 0.5, TAKER_COEFF, multiplier=0.0) == 0.0
    # 0.0175 * 3 * 0.25 * 0.5 = 0.0065625 -> 0.0066 (rounding after the multiplier)
    assert kalshi_fee(3, 0.5, MAKER_COEFF, multiplier=0.5) == pytest.approx(0.0066, abs=1e-12)


@pytest.mark.parametrize("coeff", ["0.07", "0.0175"])
def test_equals_exact_decimal_arithmetic_on_every_cent_and_count(coeff):
    counts = np.arange(1, 201)
    for cents in range(0, 101):
        got = kalshi_fee(counts, cents / 100, float(coeff))
        want = np.array([_exact_fee(int(c), cents, coeff) for c in counts])
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_symmetry_and_zero_at_the_ends():
    cents = np.arange(0, 101)
    for coeff in (TAKER_COEFF, MAKER_COEFF):
        for count in (1, 7, 100):
            fee = kalshi_fee(count, cents / 100, coeff)
            np.testing.assert_array_equal(fee, fee[::-1])  # price p and 1 - p pay the same
            assert fee[0] == 0.0 and fee[-1] == 0.0
            assert fee[50] == fee.max()  # largest at 1/2 (rounding can tie neighbouring cents)


def test_monotone_in_count_and_rounded_up():
    counts = np.arange(0, 501)
    for price in (0.03, 0.37, 0.5, 0.81):
        for coeff in (TAKER_COEFF, MAKER_COEFF):
            fee = kalshi_fee(counts, price, coeff)
            raw = coeff * counts * price * (1 - price)
            assert np.all(np.diff(fee) >= 0.0)
            assert np.all(fee >= raw - 1e-12) and np.all(fee < raw + 1e-4 + 1e-12)
            assert fee[0] == 0.0


def test_fee_is_on_the_grid_of_one_hundredth_of_a_cent():
    fee = kalshi_fee(np.arange(1, 300), 0.43, TAKER_COEFF)
    np.testing.assert_allclose(fee * 1e4, np.round(fee * 1e4), rtol=0, atol=1e-9)


def test_maker_is_cheaper_and_zero_coefficient_is_free():
    p = np.linspace(0.01, 0.99, 99)
    assert np.all(maker_fee(10, p) <= taker_fee(10, p))
    assert np.all(kalshi_fee(10, p, 0.0) == 0.0)
    np.testing.assert_array_equal(maker_fee(7, p), kalshi_fee(7, p, MAKER_COEFF))
    np.testing.assert_array_equal(taker_fee(7, p), kalshi_fee(7, p, TAKER_COEFF))


def test_vectorised_and_scalar_forms_agree():
    counts = np.array([[1, 10], [100, 7]])
    prices = np.array([0.5, 0.37])  # broadcast over rows
    fee = taker_fee(counts, prices)
    assert fee.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            assert fee[i, j] == taker_fee(int(counts[i, j]), float(prices[j]))
    assert np.ndim(taker_fee(3, 0.4)) == 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        kalshi_fee(-1, 0.5, TAKER_COEFF)
    with pytest.raises(ValueError):
        kalshi_fee(1, 1.2, TAKER_COEFF)
    with pytest.raises(ValueError):
        kalshi_fee(1, -0.01, MAKER_COEFF)
