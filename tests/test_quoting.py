"""Quoting rules, the premium estimate and the realized volatility (kalshi_mm.quoting)."""

import math

import numpy as np
import pandas as pd
import pytest

from kalshi_mm.logit import logit, round_to_tick
from kalshi_mm.markout import BUCKET_LABELS, bucket_index, trade_markouts
from kalshi_mm.quoting import (
    POLICIES,
    Premiums,
    QuoteParams,
    _bucket_of,
    estimate_premiums,
    quote_prices,
    realized_logit_vol,
)
from kalshi_mm.world import WorldParams, generate_world

NAN = float("nan")
# k = 40 makes the constant part of the half-spread (1 / gamma) ln(1 + gamma / k) = 0.0249994, the
# middle of a tick interval, so the rounded quotes of the hand examples are not on a knife edge
HAND = QuoteParams(gamma=0.002, k=40.0)
PREMIUMS = Premiums(pooled=0.05, by_bucket=(0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01), n_events=40)
ZERO = Premiums(0.0, (0.0,) * 7, 0)


def quote(policy, m, q, tau, sigma, premiums=PREMIUMS, params=HAND):
    return quote_prices(policy, m, q, tau, sigma, premiums, params)


# ------------------------------------------------------------------------------- parameters
def test_the_policies_are_the_four_of_the_model():
    assert POLICIES == ("always_on", "pull_only", "pooled_premium", "release_aware")


def test_default_parameters_are_those_of_the_protocol():
    p = QuoteParams()
    assert (p.gamma, p.k, p.size, p.max_position) == (0.002, 100.0, 10.0, 100.0)
    assert (p.refresh_s, p.latency_s, p.tick, p.pull_minutes) == (60.0, 1.0, 0.01, 5.0)
    assert (p.min_mid, p.max_mid, p.window_hours) == (0.03, 0.97, 12.0)
    assert (p.vol_lookback_s, p.stale_s, p.horizon_s) == (7200.0, 3600.0, 300.0)
    assert (p.min_history_events, p.sigma_default) == (20, 0.1)
    assert (p.maker_fee_coeff, p.fill_rule) == (0.0175, "through")
    with pytest.raises(AttributeError):  # frozen
        p.gamma = 1.0


@pytest.mark.parametrize(
    "change",
    [
        {"gamma": 0.0}, {"k": -1.0}, {"size": 0.0}, {"max_position": float("nan")},
        {"refresh_s": 0.0}, {"latency_s": 60.0}, {"latency_s": -1.0}, {"tick": 0.03},
        {"tick": 0.0}, {"min_mid": 0.5, "max_mid": 0.5}, {"max_mid": 1.2}, {"window_hours": 0.0},
        {"fill_rule": "near"}, {"min_history_events": -1}, {"pull_minutes": -1.0},
        {"maker_fee_coeff": -0.1}, {"horizon_s": 0.0}, {"stale_s": float("inf")},
    ],
)
def test_invalid_parameters_are_refused(change):
    with pytest.raises(ValueError):
        QuoteParams(**change)


def test_premiums_need_one_entry_per_bucket_and_finite_values():
    with pytest.raises(ValueError):
        Premiums(0.01, (0.0,) * 6, 5)
    with pytest.raises(ValueError):
        Premiums(float("nan"), (0.0,) * 7, 5)
    with pytest.raises(ValueError):
        Premiums(0.01, (0.0, 0.0, 0.0, float("inf"), 0.0, 0.0, 0.0), 5)


# ------------------------------------------------------------------- hand-computed quotes
# With gamma = 0.002 and k = 40: (2 / gamma) ln(1 + gamma / k) = 1000 ln(1.00005) = 0.04999875.
# At m = 0.5, m (1 - m) = 0.25 and the volatility term is (sigma * 0.25)^2 tau_h.
def test_always_on_at_one_half():
    # sigma 0.1, tau 1 h: v = 0.025^2 = 6.25e-4, gamma v = 1.25e-6, delta = 0.05 (to 1e-10); the
    # bid 0.475 rounds down to 0.47 and the ask 0.525 up to 0.53
    assert quote("always_on", 0.5, 0.0, 60.0, 0.1) == (0.47, 0.53)


def test_always_on_inventory_moves_the_reservation_price():
    # sigma 1, tau 10 h: v = 0.25^2 * 10 = 0.625, gamma v = 0.00125, delta / 2 = 0.025624375
    # q = 40: r = 0.5 - 40 * 0.00125 = 0.45, bid 0.424375 -> 0.42, ask 0.475624 -> 0.48
    assert quote("always_on", 0.5, 40.0, 600.0, 1.0) == (0.42, 0.48)
    assert quote("always_on", 0.5, -40.0, 600.0, 1.0) == (0.52, 0.58)
    assert quote("always_on", 0.5, 0.0, 600.0, 1.0) == (0.47, 0.53)


def test_bid_rounds_down_and_ask_rounds_up_not_to_the_nearest_tick():
    # m = 0.504: bid 0.479 (the nearest tick is 0.48, the tick below is 0.47), ask 0.529 -> 0.53
    assert quote("always_on", 0.504, 0.0, 60.0, 0.1) == (0.47, 0.53)
    # m = 0.497: bid 0.472 -> 0.47, ask 0.522 (the nearest tick is 0.52) -> 0.53
    assert quote("always_on", 0.497, 0.0, 60.0, 0.1) == (0.47, 0.53)


def test_pull_only_is_always_on_outside_the_pull_window():
    assert quote("pull_only", 0.5, 0.0, 60.0, 0.1) == quote("always_on", 0.5, 0.0, 60.0, 0.1)
    assert quote("pull_only", 0.5, 40.0, 600.0, 1.0) == (0.42, 0.48)
    assert quote("pull_only", 0.5, 0.0, 5.0001, 0.1) == (0.47, 0.53)


def test_pull_window_edge_belongs_to_the_window():
    assert quote("pull_only", 0.5, 0.0, 5.0, 0.1) is None  # exactly five minutes: pulled
    assert quote("pull_only", 0.5, 0.0, 1.0, 0.1) is None
    assert quote("pull_only", 0.5, 0.0, 0.001, 0.1) is None
    assert quote("always_on", 0.5, 0.0, 5.0, 0.1) == (0.47, 0.53)  # the others stay
    assert quote("pooled_premium", 0.5, 0.0, 1.0, 0.1) is not None
    assert quote("release_aware", 0.5, 0.0, 1.0, 0.1) is not None
    wide = QuoteParams(k=40.0, pull_minutes=30.0)
    assert quote("pull_only", 0.5, 0.0, 30.0, 0.1, params=wide) is None
    assert quote("pull_only", 0.5, 0.0, 30.01, 0.1, params=wide) is not None


def test_pooled_premium_adds_the_payoff_variance_and_the_pooled_premium():
    # v = 6.25e-4 + 0.25, gamma v = 0.00050125, delta / 2 = 0.02525, h = 4 * 0.25 * 0.05 = 0.05:
    # bid 0.5 - 0.07525 = 0.42475 -> 0.42, ask 0.57525 -> 0.58
    assert quote("pooled_premium", 0.5, 0.0, 60.0, 0.1) == (0.42, 0.58)
    # q = 60: r = 0.5 - 60 * 0.00050125 = 0.469925, bid 0.394675 -> 0.39, ask 0.545175 -> 0.55
    assert quote("pooled_premium", 0.5, 60.0, 60.0, 0.1) == (0.39, 0.55)


def test_release_aware_uses_the_premium_of_the_bucket_of_tau():
    # bucket (0,5] has 0.10: h = 0.10, bid 0.374751 -> 0.37, ask 0.625249 -> 0.63
    assert quote("release_aware", 0.5, 0.0, 3.0, 0.1) == (0.37, 0.63)
    # bucket (60,180] has 0.03: h = 0.03, bid 0.444750 -> 0.44, ask 0.555250 -> 0.56
    assert quote("release_aware", 0.5, 0.0, 100.0, 0.1) == (0.44, 0.56)
    # bucket >360 has 0.01: h = 0.01, bid 0.464745 -> 0.46, ask 0.535255 -> 0.54
    assert quote("release_aware", 0.5, 0.0, 500.0, 0.1) == (0.46, 0.54)
    # the pooled policy ignores the bucket: pooled 0.05 at tau = 100 gives 0.42 / 0.58
    assert quote("pooled_premium", 0.5, 0.0, 100.0, 0.1) == (0.42, 0.58)


@pytest.mark.parametrize(
    ("tau", "bucket"),
    [(0.5, 0), (5.0, 0), (5.0001, 1), (15.0, 1), (15.0001, 2), (30.0, 2), (30.0001, 3),
     (60.0, 3), (60.0001, 4), (180.0, 4), (180.0001, 5), (360.0, 5), (360.0001, 6), (1e6, 6)],
)
def test_the_bucket_of_tau_is_right_closed_and_equals_the_markout_buckets(tau, bucket):
    assert _bucket_of(tau) == bucket == int(bucket_index(tau))
    premiums = Premiums(0.0, tuple(0.01 * (i + 1) for i in range(7)), 30)  # distinct, one each
    params = QuoteParams(k=40.0)
    got = quote_prices("release_aware", 0.5, 0.0, tau, 0.0, premiums, params)
    h = 4 * 0.25 * 0.01 * (bucket + 1)
    # sigma 0: v = 0.25; delta / 2 = (0.0005 + 0.04999875) / 2 = 0.025249375
    assert got == (
        float(round_to_tick(0.5 - 0.025249375 - h, 0.01, "down")),
        float(round_to_tick(0.5 + 0.025249375 + h, 0.01, "up")),
    )


def test_buckets_have_one_premium_each_in_the_order_of_the_labels():
    assert len(PREMIUMS.by_bucket) == len(BUCKET_LABELS) == 7


def test_release_aware_equals_pooled_when_every_bucket_premium_is_equal():
    rng = np.random.default_rng(11)
    for _ in range(300):
        level = float(rng.uniform(0.0, 0.2))
        same = Premiums(level, (level,) * 7, 50)
        m, q = float(rng.uniform(0.03, 0.97)), float(rng.uniform(-150, 150))
        tau, sigma = float(rng.uniform(0.01, 800.0)), float(rng.uniform(0.0, 3.0))
        assert quote("release_aware", m, q, tau, sigma, same) == quote(
            "pooled_premium", m, q, tau, sigma, same
        )


def test_the_premium_policies_differ_when_the_bucket_premiums_differ():
    assert quote("release_aware", 0.5, 0.0, 3.0, 0.1) != quote("pooled_premium", 0.5, 0.0, 3.0, 0.1)


def test_without_a_premium_the_premium_policies_only_add_the_payoff_variance():
    for m in (0.2, 0.5, 0.8):
        a = quote("always_on", m, 0.0, 120.0, 0.3, ZERO)
        b = quote("pooled_premium", m, 0.0, 120.0, 0.3, ZERO)
        assert b[0] <= a[0] and b[1] >= a[1]  # the larger variance widens the spread
    # at q = 0 only gamma v / 2 differs: 0.5 * 0.002 * 0.25 = 2.5e-4, too small to move a tick here
    assert quote("pooled_premium", 0.5, 0.0, 60.0, 0.1, ZERO) == (0.47, 0.53)


# ----------------------------------------------------------- closed form of the A and S rule
def test_the_spread_and_the_reservation_price_are_those_of_avellaneda_and_stoikov():
    """With a tick of 1e-6 the rounding moves a quote by less than 1e-6, so the unrounded
    formulas r = m - q gamma v and delta = gamma v + (2 / gamma) ln(1 + gamma / k) show."""
    fine = QuoteParams(gamma=0.01, k=25.0, tick=1e-6)
    m, q, tau, sigma = 0.37, 25.0, 90.0, 0.8
    mm = m * (1 - m)
    for policy, extra_variance, premium in (
        ("always_on", 0.0, 0.0),
        ("pooled_premium", mm, 4 * mm * 0.03),
    ):
        v = (sigma * mm) ** 2 * (tau / 60.0) + extra_variance
        r = m - q * 0.01 * v
        delta = 0.01 * v + (2.0 / 0.01) * math.log(1.0 + 0.01 / 25.0)
        bid, ask = quote_prices(
            policy, m, q, tau, sigma, Premiums(0.03, (0.03,) * 7, 30), fine
        )
        assert bid <= r - delta / 2 - premium < bid + 1e-6 + 1e-12
        assert ask - 1e-6 - 1e-12 < r + delta / 2 + premium <= ask
        assert (bid + ask) / 2 == pytest.approx(r, abs=1e-6)
        assert ask - bid == pytest.approx(delta + 2 * premium, abs=2e-6)


def test_the_constant_part_of_the_spread_tends_to_two_over_k():
    """(2 / gamma) ln(1 + gamma / k) -> 2 / k as gamma -> 0 (the fill-intensity part of A and S)."""
    tiny = QuoteParams(gamma=1e-9, k=50.0, tick=1e-6)
    bid, ask = quote_prices("always_on", 0.5, 0.0, 1.0, 0.0, ZERO, tiny)
    assert ask - bid == pytest.approx(2.0 / 50.0, abs=2e-6)


# ------------------------------------------------------------------------------- properties
def test_the_quote_moves_down_with_inventory_and_up_with_the_premium():
    for policy in POLICIES:
        for m in (0.15, 0.5, 0.85):
            quotes = [quote(policy, m, q, 240.0, 0.5) for q in range(-90, 91, 10)]
            bids = [x[0] for x in quotes]
            asks = [x[1] for x in quotes]
            assert all(a >= b for a, b in zip(bids, bids[1:], strict=False))  # q up: bid down
            assert all(a >= b for a, b in zip(asks, asks[1:], strict=False))
    for policy in POLICIES:  # and it does move (large volatility and a long horizon)
        low, high = quote(policy, 0.5, -90.0, 600.0, 2.0), quote(policy, 0.5, 90.0, 600.0, 2.0)
        assert low[0] > high[0] and low[1] > high[1]
    for policy in ("pooled_premium", "release_aware"):
        previous = None
        for level in (0.0, 0.02, 0.05, 0.1, 0.2):
            same = Premiums(level, (level,) * 7, 50)
            bid, ask = quote(policy, 0.5, 0.0, 20.0, 0.3, same)
            if previous is not None:
                assert bid <= previous[0] and ask >= previous[1]
            previous = (bid, ask)
        assert (previous[0], previous[1]) != quote(policy, 0.5, 0.0, 20.0, 0.3, ZERO)


def test_quotes_are_on_the_tick_grid_inside_the_range_and_a_tick_apart():
    rng = np.random.default_rng(5)
    for _ in range(3000):
        policy = POLICIES[int(rng.integers(4))]
        got = quote(
            policy, float(rng.uniform(0.03, 0.97)), float(rng.uniform(-99.0, 99.0)),
            float(rng.uniform(5.01, 900.0)), float(rng.uniform(0.0, 6.0)),
            Premiums(0.0, tuple(rng.uniform(0.0, 0.3, 7)), 30),
        )
        bid, ask = got
        for price in (bid, ask):
            assert 0.01 <= price <= 0.99
            assert price == pytest.approx(round(price / 0.01) * 0.01, abs=1e-9)
        assert ask - bid >= 0.01 - 1e-9


def test_clipping_keeps_a_tick_between_the_quotes_at_the_edges():
    huge = Premiums(100.0, (100.0,) * 7, 50)  # h = 4 * 0.0291 * 100 = 11.6 USD: far outside
    bid, ask = quote("pooled_premium", 0.97, 0.0, 60.0, 0.1, huge)
    assert (bid, ask) == (0.01, 0.99)
    # the reservation price far below zero: both unrounded quotes are below the bottom tick
    bid, ask = quote_prices("always_on", 0.5, 90.0, 6000.0, 5.0, ZERO, HAND)
    assert (bid, ask) == (0.01, 0.02)  # the ask moves up a tick
    # far above one: both above the top tick, the bid moves down a tick
    bid, ask = quote_prices("always_on", 0.5, -90.0, 6000.0, 5.0, ZERO, HAND)
    assert (bid, ask) == (0.98, 0.99)


def test_the_mid_range_is_inclusive_and_a_nan_mid_gives_no_quote():
    assert quote("always_on", 0.03, 0.0, 60.0, 0.1) is not None
    assert quote("always_on", 0.97, 0.0, 60.0, 0.1) is not None
    assert quote("always_on", 0.0299, 0.0, 60.0, 0.1) is None
    assert quote("always_on", 0.9701, 0.0, 60.0, 0.1) is None
    assert quote("always_on", NAN, 0.0, 60.0, 0.1) is None
    assert quote("always_on", 0.0, 0.0, 60.0, 0.1) is None
    # a mid that is 0.03 up to float noise (0.03 - 1e-12) still quotes
    assert quote("always_on", 0.03 - 1e-12, 0.0, 60.0, 0.1) is not None
    assert quote("always_on", 0.97 + 1e-12, 0.0, 60.0, 0.1) is not None
    wide = QuoteParams(min_mid=0.4, max_mid=0.6, k=40.0)
    assert quote("always_on", 0.39, 0.0, 60.0, 0.1, params=wide) is None
    assert quote("always_on", 0.5, 0.0, 60.0, 0.1, params=wide) == (0.47, 0.53)


def test_a_side_is_not_quoted_at_the_position_limit():
    full = quote("always_on", 0.5, 100.0, 60.0, 0.1)
    assert math.isnan(full[0]) and full[1] == 0.53
    empty = quote("always_on", 0.5, -100.0, 60.0, 0.1)
    assert math.isnan(empty[1]) and empty[0] == 0.47
    just_below = quote("always_on", 0.5, 99.9999, 60.0, 0.1)
    assert not any(math.isnan(x) for x in just_below)
    just_above = quote("always_on", 0.5, -99.9999, 60.0, 0.1)
    assert not any(math.isnan(x) for x in just_above)
    beyond = quote("pooled_premium", 0.5, 130.0, 60.0, 0.1)
    assert math.isnan(beyond[0]) and not math.isnan(beyond[1])
    small = QuoteParams(k=40.0, max_position=20.0)
    assert math.isnan(quote("always_on", 0.5, 20.0, 60.0, 0.1, params=small)[0])
    assert not math.isnan(quote("always_on", 0.5, 19.0, 60.0, 0.1, params=small)[0])


def test_the_side_that_remains_at_the_limit_is_the_unconstrained_quote_of_that_side():
    loose = QuoteParams(k=40.0, max_position=1000.0)
    for q in (100.0, -100.0, 250.0):
        bid, ask = quote("pooled_premium", 0.5, q, 60.0, 0.1)
        free_bid, free_ask = quote("pooled_premium", 0.5, q, 60.0, 0.1, params=loose)
        assert (math.isnan(bid), math.isnan(ask)) == (q > 0, q < 0)
        assert (bid == free_bid) or q > 0
        assert (ask == free_ask) or q < 0


def test_a_floored_premium_of_zero_is_the_same_as_no_premium():
    assert quote("pooled_premium", 0.5, 0.0, 60.0, 0.1, ZERO) == quote(
        "release_aware", 0.5, 0.0, 60.0, 0.1, ZERO
    )


def expected_quote(policy, m, q, tau, sigma, premiums, p):
    """The quote from the formulas of model.md 6 with numpy and the foundation's tick rounding."""
    mm = m * (1.0 - m)
    v = (sigma * mm) ** 2 * tau / 60.0
    h = 0.0
    if policy in ("pooled_premium", "release_aware"):
        v += mm
        rate = premiums.pooled if policy == "pooled_premium" else premiums.by_bucket[
            int(bucket_index(tau))
        ]
        h = 4.0 * mm * rate
    r = m - q * p.gamma * v
    delta = p.gamma * v + 2.0 / p.gamma * np.log1p(p.gamma / p.k)
    top = round(1.0 / p.tick) - 1
    low = float(round_to_tick(r - delta / 2 - h, p.tick, "down"))
    high = float(round_to_tick(r + delta / 2 + h, p.tick, "up"))
    kb = int(np.clip(round(low / p.tick), 1, top))
    ka = int(np.clip(round(high / p.tick), 1, top))
    if ka - kb < 1:
        if kb + 1 <= top:
            ka = kb + 1
        else:
            kb = ka - 1
    bid, ask = round(kb * p.tick, 10), round(ka * p.tick, 10)
    return (NAN if q >= p.max_position else bid, NAN if q <= -p.max_position else ask)


@pytest.mark.parametrize("tick", [0.01, 0.005, 0.02])
def test_quote_prices_equals_the_formulas_on_random_inputs(tick):
    """Differential test against numpy and :func:`kalshi_mm.logit.round_to_tick`."""
    rng = np.random.default_rng(17)
    params = QuoteParams(gamma=0.004, k=60.0, tick=tick, max_position=80.0)
    for _ in range(4000):
        policy = POLICIES[int(rng.integers(4))]
        premiums = Premiums(float(rng.uniform(0, 0.2)), tuple(rng.uniform(0.0, 0.3, 7)), 30)
        args = (
            float(rng.uniform(0.03, 0.97)), float(rng.uniform(-120, 120)),
            float(rng.uniform(5.01, 1000.0)), float(rng.uniform(0.0, 4.0)),
        )
        if policy == "pull_only" or rng.random() < 0.3:
            args = (*args[:2], float(rng.uniform(0.01, 8.0)), args[3])
        got = quote_prices(policy, args[0], args[1], args[2], args[3], premiums, params)
        if policy == "pull_only" and args[2] <= params.pull_minutes:
            assert got is None
            continue
        want = expected_quote(policy, *args, premiums, params)
        assert got is not None
        for a, b in zip(got, want, strict=True):
            assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, abs=1e-9)


def test_the_scalar_rounding_agrees_with_the_foundation_rounding_on_awkward_values():
    """0.29 / 0.01 is 28.999999999999996: a quote at 0.29 must stay 0.29, and one tick above
    a grid point stays on the next tick; both the bid and the ask path are checked."""
    params = QuoteParams(gamma=0.002, k=100.0)
    # constant half-spread at q = 0, sigma 0: gamma v = 0 for always_on at tau tiny
    half = 0.5 * (2.0 / 0.002) * math.log1p(0.002 / 100.0)  # 0.0099999 -> ticks just under 1
    for centre in np.arange(0.05, 0.96, 0.01):
        bid, ask = quote_prices("always_on", float(centre), 0.0, 1e-9, 0.0, ZERO, params)
        want_bid = float(round_to_tick(centre - half, 0.01, "down"))
        want_ask = float(round_to_tick(centre + half, 0.01, "up"))
        assert (bid, ask) == (want_bid, want_ask)


def test_arguments_are_checked():
    with pytest.raises(ValueError):
        quote("on_always", 0.5, 0.0, 60.0, 0.1)
    with pytest.raises(ValueError):
        quote("always_on", 0.5, 0.0, 0.0, 0.1)  # at the release the quote has no meaning
    with pytest.raises(ValueError):
        quote("release_aware", 0.5, 0.0, -3.0, 0.1)  # a negative tau would pick the last bucket
    with pytest.raises(ValueError):
        quote("always_on", 0.5, 0.0, 60.0, NAN)
    with pytest.raises(ValueError):
        quote("always_on", 0.5, 0.0, 60.0, -0.1)


# --------------------------------------------------------------------- estimate_premiums
def markout_frame(rows):
    return pd.DataFrame(
        rows, columns=["event", "bucket", "mid_before", "adverse_selection"]
    ).astype({"bucket": "int64", "mid_before": "float64", "adverse_selection": "float64"})


HAND_ROWS = [
    ("E1", 0, 0.50, 0.04),  # normalised 0.04 / (4 * 0.25) = 0.04
    ("E1", 0, 0.50, 0.02),  # 0.02
    ("E2", 0, 0.20, 0.032),  # 0.032 / (4 * 0.16) = 0.05
    ("E2", 3, 0.50, -0.01),  # -0.01
    ("E2", 3, 0.50, 0.03),  # 0.03
    ("E3", 3, 0.95, 0.50),  # m (1 - m) = 0.0475 < 0.05: left out
    ("E3", 5, 0.50, -0.06),  # -0.06
    ("E3", 5, 0.50, NAN),  # no adverse selection: left out
    ("E3", 6, NAN, 0.10),  # no mid: left out
]


def test_premiums_by_hand():
    got = estimate_premiums(markout_frame(HAND_ROWS), min_events=3)
    pooled = (0.04 + 0.02 + 0.05 - 0.01 + 0.03 - 0.06) / 6  # six usable trades
    assert got.n_events == 3
    assert got.pooled == pytest.approx(pooled, abs=1e-15)
    assert got.by_bucket[0] == pytest.approx((0.04 + 0.02 + 0.05) / 3, abs=1e-15)
    assert got.by_bucket[3] == pytest.approx(0.01, abs=1e-15)  # (-0.01 + 0.03) / 2
    assert got.by_bucket[5] == 0.0  # the mean -0.06 is floored at zero
    for empty in (1, 2, 4, 6):  # no usable trade: the pooled premium
        assert got.by_bucket[empty] == pytest.approx(pooled, abs=1e-15)


def test_premiums_use_the_floor():
    got = estimate_premiums(markout_frame(HAND_ROWS), min_events=3, floor=0.02)
    assert got.pooled == 0.02  # the mean 0.0117 is below the floor
    assert got.by_bucket[0] == pytest.approx(0.11 / 3, abs=1e-15)  # above it
    assert got.by_bucket[3] == 0.02 and got.by_bucket[5] == 0.02
    assert got.by_bucket[1] == 0.02  # a bucket with no trade gets the floored pooled premium


def test_the_default_floor_is_zero_and_a_negative_mean_gives_zero():
    rows = [("E1", 2, 0.5, -0.03), ("E2", 2, 0.5, -0.01), ("E3", 2, 0.5, -0.05)]
    got = estimate_premiums(markout_frame(rows), min_events=3)
    assert got.pooled == 0.0 and got.by_bucket == (0.0,) * 7


def test_the_variance_filter_is_at_five_hundredths():
    # 0.05279 * 0.94721 = 0.050004 is kept, 0.05278 * 0.94722 = 0.049993 is not
    rows = [("E1", 0, 0.05279, 0.04), ("E2", 0, 0.05278, 0.40), ("E3", 0, 0.5, 0.0)]
    got = estimate_premiums(markout_frame(rows), min_events=3)
    kept = 0.04 / (4 * 0.05279 * (1 - 0.05279))
    assert got.pooled == pytest.approx((kept + 0.0) / 2, rel=1e-12)


def test_the_premium_is_zero_below_the_minimum_number_of_events():
    got = estimate_premiums(markout_frame(HAND_ROWS), min_events=4)
    assert got == Premiums(0.0, (0.0,) * 7, 3)
    two_events = estimate_premiums(markout_frame(HAND_ROWS[:3]), min_events=3)
    assert two_events.pooled == 0.0 and two_events.n_events == 2  # E1 and E2 only
    empty = estimate_premiums(markout_frame([]), min_events=1)
    assert empty == Premiums(0.0, (0.0,) * 7, 0)


def test_events_are_counted_not_trades():
    rows = [("E1", 0, 0.5, 0.04)] * 50  # one event with fifty usable trades
    assert estimate_premiums(markout_frame(rows), min_events=2).pooled == 0.0
    rows += [("E2", 0, 0.5, 0.04)]
    assert estimate_premiums(markout_frame(rows), min_events=2).pooled == pytest.approx(0.04)


def test_no_usable_trade_at_all_gives_the_floor_or_zero():
    rows = [("E1", 0, 0.99, 0.5), ("E2", 1, NAN, 0.5), ("E3", 2, 0.5, NAN)]
    assert estimate_premiums(markout_frame(rows), min_events=3) == Premiums(0.0, (0.0,) * 7, 3)
    got = estimate_premiums(markout_frame(rows), min_events=3, floor=0.01)
    assert got == Premiums(0.01, (0.01,) * 7, 3)
    assert estimate_premiums(markout_frame([]), min_events=0) == Premiums(0.0, (0.0,) * 7, 0)


def test_premiums_equal_a_plain_loop_on_the_markouts_of_a_world():
    world = generate_world(WorldParams(n_events=24, window_hours=6.0), 3)
    tape = world.tape.drop(columns=[c for c in world.tape.columns if c.startswith("latent_")])
    markouts = trade_markouts(tape, world.events, world.markets)
    got = estimate_premiums(markouts, min_events=10, floor=0.0)
    sums = np.zeros(7)
    counts = np.zeros(7, dtype=int)
    for row in markouts.itertuples():
        m, a = row.mid_before, row.adverse_selection
        if math.isnan(m) or math.isnan(a) or m * (1 - m) < 0.05:
            continue
        sums[row.bucket] += a / (4 * m * (1 - m))
        counts[row.bucket] += 1
    pooled = max(0.0, sums.sum() / counts.sum())
    assert got.n_events == 24
    assert got.pooled == pytest.approx(pooled, abs=1e-12)
    for b in range(7):
        want = max(0.0, sums[b] / counts[b]) if counts[b] else pooled
        assert got.by_bucket[b] == pytest.approx(want, abs=1e-12)


# ---------------------------------------------------------------------- realized_logit_vol
def hand_vol_tape(first_pair_at=-1, alternate=True):
    """Both sides print 1 s before each grid time ``300 k``: the proxy is 0.50 at even k, 0.60 at
    odd k (buy 0.70, sell 0.50), starting with the pair at ``first_pair_at``."""
    ts, side, price = [], [], []
    for k in range(11):
        t = 300.0 * k - 1.0
        if t < first_pair_at:
            continue
        if alternate and k % 2 == 1:
            ts += [t, t + 0.5]
            side += [1, -1]
            price += [0.70, 0.50]
        else:
            ts += [t, t + 0.5]
            side += [1, -1]
            price += [0.55, 0.45]
    return np.array(ts), np.array(side, dtype=np.int8), np.array(price)


def test_the_volatility_of_an_alternating_proxy_has_a_closed_form():
    ts, side, price = hand_vol_tape()
    # the proxy alternates 0.50, 0.60: every step is logit(0.6) in size, the grid step is 1/12 hour
    got = realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=3600.0)
    d = math.log(0.6 / 0.4)
    assert got == pytest.approx(d * math.sqrt(12.0), rel=1e-12)
    assert isinstance(got, float)


def test_a_constant_proxy_has_zero_volatility():
    ts, side, price = hand_vol_tape(alternate=False)
    assert realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=3600.0) == 0.0


def test_the_grid_step_scales_the_estimate():
    ts, side, price = hand_vol_tape()
    d = math.log(0.6 / 0.4)
    # a grid of 600 s reads the proxy at the even points only: it never moves, 0 volatility
    coarse = realized_logit_vol(
        ts, side, price, 3000.0, lookback_s=3000.0, stale_s=3600.0, grid_s=600.0
    )
    assert coarse == 0.0
    # the grid 300 s at a time 300 s earlier covers points 0 ... 9: nine steps of the same size
    earlier = realized_logit_vol(ts, side, price, 2700.0, lookback_s=2700.0, stale_s=3600.0)
    assert earlier == pytest.approx(d * math.sqrt(12.0), rel=1e-12)


def test_the_volatility_needs_six_valid_grid_points():
    # the first pair of prints is 1 s before grid point 5 (time 1500): points 5, ..., 10 are valid
    ts, side, price = hand_vol_tape(first_pair_at=1499.0)
    six = realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=3600.0)
    assert six == pytest.approx(math.log(0.6 / 0.4) * math.sqrt(12.0), rel=1e-12)
    # starting one point later leaves five valid points
    ts, side, price = hand_vol_tape(first_pair_at=1799.0)
    assert np.isnan(realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=3600.0))


def test_the_volatility_needs_a_pair_of_adjacent_valid_points():
    """Six valid points that are never adjacent (the proxy is stale in between) give NaN."""
    ts, side, price = hand_vol_tape()
    keep = np.array([int(round((t + 1.0) / 300.0)) % 2 == 0 for t in np.floor(ts + 1.0)])
    ts, side, price = ts[keep], side[keep], price[keep]  # prints before the even points only
    got = realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=100.0)
    assert np.isnan(got)
    # with the staleness limit of the default the proxy persists and the volatility exists
    assert not np.isnan(
        realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=3600.0)
    )


def test_adjacent_pairs_only_are_used_when_a_point_is_missing():
    """Remove the prints before grid point 4 (time 1200) and make the proxy stale there: points
    3 and 5 are not a pair, so only the steps among the other points count."""
    ts, side, price = hand_vol_tape()
    odd_four = np.abs(ts - 1199.0) > 0.6  # drop both prints at 1199 and 1199.5
    ts, side, price = ts[odd_four], side[odd_four], price[odd_four]
    # stale_s = 400: at 1200 the last prints are those at 899 / 899.5 (age 300) -> still valid
    still_valid = realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=400.0)
    assert not np.isnan(still_valid)
    # stale_s = 250: the prints at 899 are 301 s old at 1200, so point 4 is undefined; the pairs
    # (3, 4) and (4, 5) are lost, the remaining eight steps are all of the same size
    got = realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=250.0)
    assert got == pytest.approx(math.log(0.6 / 0.4) * math.sqrt(12.0), rel=1e-12)


def test_scalar_and_array_queries_agree():
    world = generate_world(WorldParams(n_events=2, window_hours=6.0), 2)
    tape = world.tape[world.tape["market"] == "SYN-000001"]
    release = float(world.events["release_ts"].iloc[1])
    ts, side, price = tape["ts"].to_numpy(), tape["side"].to_numpy(), tape["price"].to_numpy()
    queries = release - 3600.0 * np.array([5.5, 4.0, 3.0, 1.0, 0.2])
    many = realized_logit_vol(ts, side, price, queries, lookback_s=7200.0, stale_s=3600.0)
    assert many.shape == (5,)
    for q, v in zip(queries, many, strict=True):
        one = realized_logit_vol(ts, side, price, q, lookback_s=7200.0, stale_s=3600.0)
        assert (math.isnan(one) and math.isnan(v)) or one == v
    shaped = realized_logit_vol(
        ts, side, price, queries.reshape(5, 1), lookback_s=7200.0, stale_s=3600.0
    )
    assert shaped.shape == (5, 1)


def slow_vol(ts, side, price, at, lookback, stale, grid=300.0):
    """The definition with plain loops: the proxy at each grid point from a scan of the prints."""
    def proxy(t):
        last = {}
        for i in range(len(ts)):
            if ts[i] <= t:
                last[int(side[i])] = (ts[i], price[i])
        if len(last) < 2 or any(t - when > stale for when, _ in last.values()):
            return None
        return (last[1][1] + last[-1][1]) / 2.0

    steps = int(math.floor(lookback / grid + 1e-9))
    points = [proxy(at - grid * i) for i in range(steps, -1, -1)]  # oldest first
    xs = [None if m is None or not 0.0 < m < 1.0 else math.log(m / (1.0 - m)) for m in points]
    squares = [
        (xs[i + 1] - xs[i]) ** 2 for i in range(len(xs) - 1)
        if xs[i] is not None and xs[i + 1] is not None
    ]
    if sum(x is not None for x in xs) < 6 or not squares:
        return NAN
    return math.sqrt(sum(squares) / len(squares) / (grid / 3600.0))


@pytest.mark.parametrize(
    ("lookback", "stale", "grid"), [(7200.0, 3600.0, 300.0), (3000.0, 900.0, 250.0)]
)
def test_the_volatility_equals_a_plain_loop_on_a_world(lookback, stale, grid):
    world = generate_world(WorldParams(n_events=3, window_hours=5.0), 8)
    for event in ("SYN-000000", "SYN-000002"):
        tape = world.tape[world.tape["market"] == event]
        release = float(world.events.set_index("event").loc[event, "release_ts"])
        ts, side, price = tape["ts"].to_numpy(), tape["side"].to_numpy(), tape["price"].to_numpy()
        at = release - 3600.0 * np.linspace(4.8, 0.05, 25)
        got = realized_logit_vol(
            ts, side, price, at, lookback_s=lookback, stale_s=stale, grid_s=grid
        )
        want = np.array([slow_vol(ts, side, price, a, lookback, stale, grid) for a in at])
        assert np.array_equal(np.isnan(got), np.isnan(want))
        np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=1e-12)
        assert (~np.isnan(got)).sum() >= 10  # the comparison has defined values in it


def test_later_prints_do_not_change_the_volatility():
    """Look-ahead guard: changing, deleting or adding prints after ``at`` changes nothing."""
    world = generate_world(WorldParams(n_events=1, window_hours=6.0), 4)
    tape = world.tape
    ts, side, price = tape["ts"].to_numpy(), tape["side"].to_numpy(), tape["price"].to_numpy()
    at = float(ts[len(ts) // 2]) + 17.3
    base = realized_logit_vol(ts, side, price, at, lookback_s=7200.0, stale_s=3600.0)
    assert not math.isnan(base)
    cut = int(np.searchsorted(ts, at, side="right"))
    rng = np.random.default_rng(2)
    altered_price = price.copy()
    altered_price[cut:] = rng.uniform(0.2, 0.8, len(ts) - cut)
    altered_side = side.copy()
    altered_side[cut:] = -altered_side[cut:]
    for t2, s2, p2 in (
        (ts[:cut], side[:cut], price[:cut]),  # deleted
        (ts, altered_side, altered_price),  # altered
        (np.r_[ts, ts[-1] + 5.0], np.r_[side, 1], np.r_[price, 0.99]),  # appended
    ):
        assert realized_logit_vol(t2, s2, p2, at, lookback_s=7200.0, stale_s=3600.0) == base


def test_a_print_at_the_query_time_counts():
    ts, side, price = hand_vol_tape()
    base = realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=3600.0)
    # a buy at exactly 3000 moves the last grid point: the last sell is 0.45, so the proxy becomes
    # (0.90 + 0.45) / 2 = 0.675 instead of 0.50
    t2, s2, p2 = np.r_[ts, 3000.0], np.r_[side, 1], np.r_[price, 0.90]
    moved = realized_logit_vol(t2, s2, p2, 3000.0, lookback_s=3000.0, stale_s=3600.0)
    assert moved != base
    a = math.log(1.5)  # logit(0.6)
    xs = [0.0, a, 0.0, a, 0.0, a, 0.0, a, 0.0, a, math.log(0.675 / 0.325)]  # grid points 0 .. 10
    want = math.sqrt(sum((y - x) ** 2 for x, y in zip(xs, xs[1:], strict=False)) / 10 * 12.0)
    assert moved == pytest.approx(want, rel=1e-12)


def test_invalid_volatility_arguments_are_refused():
    ts, side, price = hand_vol_tape()
    with pytest.raises(ValueError):
        realized_logit_vol(ts, side, price, 3000.0, lookback_s=0.0, stale_s=3600.0)
    with pytest.raises(ValueError):
        realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=3600.0, grid_s=-1.0)


def test_extreme_proxies_do_not_break_the_volatility():
    """A proxy of exactly 0 or 1 has no log-odds: the point is undefined, with no warning."""
    ts = np.arange(0.0, 3300.0, 100.0)
    side = np.where(np.arange(len(ts)) % 2 == 0, 1, -1).astype(np.int8)
    price = np.where(side > 0, 1.0, 0.0)  # every proxy is exactly 0.5; logit finite
    got = realized_logit_vol(ts, side, price, 3000.0, lookback_s=3000.0, stale_s=3600.0)
    assert got == 0.0
    one = np.ones(len(ts))
    edge = realized_logit_vol(ts, side, one, 3000.0, lookback_s=3000.0, stale_s=3600.0)
    assert np.isnan(edge)  # the proxy is 1.0 everywhere
    assert np.isinf(float(logit(1.0)))
