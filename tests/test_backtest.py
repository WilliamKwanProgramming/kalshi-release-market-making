"""The tape-only fill engine and the walk-forward backtest (kalshi_mm.backtest, model.md 6, 7)."""

import math
from dataclasses import asdict, fields, replace

import numpy as np
import pandas as pd
import pytest

from kalshi_mm import backtest as bt
from kalshi_mm.backtest import (
    EventResult,
    backtest_grid,
    backtest_market,
    prepare_market,
    walk_forward,
)
from kalshi_mm.fees import kalshi_fee
from kalshi_mm.markout import bucket_index, trade_markouts
from kalshi_mm.quoting import (
    POLICIES,
    PREMIUM_POLICIES,
    Premiums,
    QuoteParams,
    estimate_premiums,
    realized_logit_vol,
)
from kalshi_mm.world import WorldParams, generate_world

NAN = float("nan")
RELEASE = 10_000.0
# six refresh intervals of 60 s: decisions at 9640, 9700, ..., 9940, live from 1 s later. With
# k = 40 the always-on quotes at a mid of 0.50 are bid 0.47 and ask 0.53 in every interval
HAND = QuoteParams(gamma=0.002, k=40.0, window_hours=0.1)
ZERO = Premiums(0.0, (0.0,) * 7, 0, 0.0)
PREMIUMS = Premiums(
    pooled=0.05, by_bucket=(0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01), n_events=40, matched=0.04
)

BUY, SELL = 1, -1


def tape_of(rows):
    """Arrays ``ts, side, price, size`` from ``(ts, side, price, size)`` rows."""
    ts, side, price, size = zip(*rows, strict=True)
    return (np.array(ts, dtype=float), np.array(side, dtype=np.int8),
            np.array(price, dtype=float), np.array(size, dtype=float))


def run(rows, policy="always_on", *, y=1, premiums=ZERO, floor=0.1, params=HAND, release=RELEASE):
    ts, side, price, size = tape_of(rows)
    return backtest_market(ts, side, price, size, release, y, policy, premiums, floor, params,
                           event="E", market="M")


# the proxy is 0.50 before the window and stays 0.50 after every print below: buys print at
# 0.55 (or at the quote under test, followed by a buy at 0.55) and sells at 0.45
PRE = [(9500.0, BUY, 0.55, 5.0), (9510.0, SELL, 0.45, 5.0)]
S1 = PRE + [
    (9640.5, SELL, 0.45, 5.0),  # after decision 0, before its quotes are live: no fill
    (9641.0, BUY, 0.55, 4.0),  # exactly t_0 + latency: the live interval is open on the left
    (9650.0, SELL, 0.45, 6.0),  # fills 6 at the bid 0.47
    (9660.0, SELL, 0.45, 7.0),  # only 4 left on the bid: partial fill of 4
    (9670.0, SELL, 0.45, 5.0),  # the bid is used up
    (9701.0, BUY, 0.55, 3.0),  # exactly t_1 + latency: still the first interval, fills 3 at 0.53
    (9705.0, SELL, 0.45, 2.0),  # second interval: size restored, fills 2 at 0.47
    (9750.0, BUY, 0.53, 5.0),  # a print at the ask: no fill through, fills 5 under touch
    (9755.0, BUY, 0.55, 20.0),  # through: fills the 10 left (touch: 5)
    (9770.0, SELL, 0.47, 3.0),  # a print at the bid: no fill through, fills 3 under touch
    (9775.0, SELL, 0.45, 1.0),  # fills 1 at 0.47
    (9830.0, BUY, 0.55, 4.0),  # fills 4 at 0.53
    (9999.0, SELL, 0.45, 2.0),  # last interval: fills 2 at 0.47
    (10000.0, BUY, 0.90, 9.0),  # at the release: dropped (it would fill the ask and move the proxy)
    (10005.0, SELL, 0.10, 9.0),  # after the release: dropped
]


def maker_fee(count, price):
    return float(kalshi_fee(count, price, 0.0175))


QUOTE = bt.quote_prices  # the real function, taken before any test replaces it in the module


def record_quotes(monkeypatch):
    """Wrap the quote function of the backtest. The list that is returned gets one row per decision:
    mid proxy, inventory, minutes to the release, volatility, bid and ask (NaN for a side that is
    not quoted, both NaN for no quote)."""
    rows = []

    def wrapper(policy, mid, inventory, tau, sigma, premiums, params):
        quote = QUOTE(policy, mid, inventory, tau, sigma, premiums, params)
        rows.append((mid, inventory, tau, sigma, *(quote if quote is not None else (NAN, NAN))))
        return quote

    monkeypatch.setattr(bt, "quote_prices", wrapper)
    return rows


# ----------------------------------------------------------------------- hand-built tape
def test_through_rule_by_hand():
    r = run(S1)
    # fills (contracts @ price): +6 @ 0.47, +4 @ 0.47, -3 @ 0.53, +2 @ 0.47, -10 @ 0.53,
    # +1 @ 0.47, -4 @ 0.53, +2 @ 0.47: eight fills of 32 contracts
    assert (r.fills, r.contracts) == (8, 32.0)
    assert r.inventory_end == 6 + 4 - 3 + 2 - 10 + 1 - 4 + 2 == -2.0
    cash = -2.82 - 1.88 + 1.59 - 0.94 + 5.30 - 0.47 + 2.12 - 0.94
    assert cash == pytest.approx(1.96)
    # fee of each fill: 0.0175 c * 0.2491 rounded up to 0.0001
    fees = 0.0262 + 0.0175 + 0.0131 + 0.0088 + 0.0436 + 0.0044 + 0.0175 + 0.0088
    assert fees == pytest.approx(0.1399)
    assert r.fees == pytest.approx(fees, abs=1e-12)
    assert r.gross == pytest.approx(cash + (-2.0) * 1.0, abs=1e-12)  # settles at 1
    assert r.pnl == pytest.approx(r.gross - r.fees, abs=1e-12)
    assert (r.gross, r.pnl) == pytest.approx((-0.04, -0.1799), abs=1e-9)
    # the last valid proxy before the release is 0.50: -2 contracts valued at 0.50 USD
    assert r.marked_pnl == pytest.approx(cash - 2 * 0.50 - fees, abs=1e-12)
    assert r.quotes_live_share == 1.0
    assert r.mean_spread == pytest.approx(0.06, abs=1e-12)  # bid 0.47 and ask 0.53 every interval
    assert (r.event, r.market, r.policy) == ("E", "M", "always_on")


def test_the_settled_pnl_depends_on_the_outcome_only_through_the_inventory():
    lost, won = run(S1, y=0), run(S1, y=1)
    assert lost.gross == pytest.approx(1.96, abs=1e-9)  # the short position expires worthless
    assert lost.pnl == pytest.approx(1.96 - 0.1399, abs=1e-9)
    assert won.pnl - lost.pnl == pytest.approx(-2.0, abs=1e-9)  # q (y1 - y0), q = -2
    for field in ("fees", "fills", "contracts", "inventory_end", "marked_pnl", "quotes_live_share",
                  "mean_spread", "fill_premium"):
        assert getattr(lost, field) == getattr(won, field)  # nothing else sees the outcome


def test_touch_rule_by_hand():
    r = run(S1, params=replace(HAND, fill_rule="touch"))
    # as under the through rule, plus the print at the ask at 9750 (5 contracts sold at 0.53) and
    # the print at the bid at 9770 (3 bought at 0.47); the print through the ask at 9755 then
    # finds only 5 contracts left instead of 10: ten fills of 35 contracts, inventory +1
    assert (r.fills, r.contracts, r.inventory_end) == (10, 35.0, 1.0)
    cash = -2.82 - 1.88 + 1.59 - 0.94 + 2.65 + 2.65 - 1.41 - 0.47 + 2.12 - 0.94
    assert cash == pytest.approx(0.55)
    fees = (0.0262 + 0.0175 + 0.0131 + 0.0088 + 0.0218 + 0.0218 + 0.0131 + 0.0044 + 0.0175
            + 0.0088)
    assert fees == pytest.approx(0.1530)
    assert r.fees == pytest.approx(fees, abs=1e-12)
    assert r.gross == pytest.approx(cash + 1.0, abs=1e-12)
    assert r.pnl == pytest.approx(1.55 - 0.1530, abs=1e-9)
    assert r.marked_pnl == pytest.approx(cash + 0.5 - fees, abs=1e-12)


def test_a_print_at_the_quote_fills_only_under_touch():
    rows = PRE + [(9650.0, SELL, 0.47, 4.0), (9660.0, BUY, 0.53, 6.0)]
    through = run(rows)
    touch = run(rows, params=replace(HAND, fill_rule="touch"))
    assert (through.fills, through.contracts) == (0, 0.0)
    assert (touch.fills, touch.contracts, touch.inventory_end) == (2, 10.0, -2.0)
    assert touch.gross == pytest.approx(-4 * 0.47 + 6 * 0.53 - 2.0, abs=1e-12)  # y = 1
    # one cent beyond the quote fills under both rules
    beyond = PRE + [(9650.0, SELL, 0.46, 4.0), (9660.0, BUY, 0.54, 6.0)]
    for rule in ("through", "touch"):
        r = run(beyond, params=replace(HAND, fill_rule=rule))
        assert (r.fills, r.contracts, r.inventory_end) == (2, 10.0, -2.0)


def test_the_fill_is_at_our_price_not_the_print_price():
    rows = PRE + [(9650.0, SELL, 0.20, 3.0), (9660.0, BUY, 0.90, 2.0)]
    r = run(rows, y=0)
    assert (r.fills, r.contracts, r.inventory_end) == (2, 5.0, 1.0)
    assert r.gross == pytest.approx(-3 * 0.47 + 2 * 0.53)  # prints at 0.20 and 0.90, fills at 0.47


def test_inventory_known_at_each_decision(monkeypatch):
    """The inventory of a decision counts the fills at or before its time: the fills of the
    previous interval between the decision and the moment its own quotes are live are left out."""
    seen = record_quotes(monkeypatch)
    run(S1)
    # t_1 = 9700 sees 9650 and 9660 (+10) but not the sale at 9701 (-3); t_2 = 9760 sees the
    # fills of 9701 (-3), 9705 (+2) and 9755 (-10); t_3 = 9820 adds 9775 (+1); t_4 adds 9830 (-4)
    assert [s[1] for s in seen] == [0.0, 10.0, -1.0, 0.0, -4.0, -4.0]
    assert [s[0] for s in seen] == [0.5] * 6
    assert [s[2] for s in seen] == [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]  # minutes to the release
    assert [s[3] for s in seen] == [0.1] * 6  # too little history: the floor
    assert [(s[4], s[5]) for s in seen] == [(0.47, 0.53)] * 6


def test_a_fill_at_the_decision_time_is_known_to_that_decision(monkeypatch):
    """The decision at t_1 = 9700 reads the prints up to and including 9700, and so it knows the
    fill of the sale at 9700 (3 of the 4 contracts left on the bid of the first interval)."""
    seen = record_quotes(monkeypatch)
    rows = PRE + [(9650.0, SELL, 0.45, 6.0), (9700.0, SELL, 0.45, 3.0)]
    assert run(rows).inventory_end == 9.0
    assert [s[1] for s in seen] == [0.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    seen.clear()
    mirrored = [(ts, -s, 1.0 - p, c) for ts, s, p, c in rows]  # the ask side, upside down
    assert run(mirrored).inventory_end == -9.0
    assert [s[1] for s in seen] == [0.0, -9.0, -9.0, -9.0, -9.0, -9.0]


def test_the_known_inventory_catches_up_in_an_interval_without_prints(monkeypatch):
    seen = record_quotes(monkeypatch)
    # the sale at 9700.5 fills the first interval's bid after the decision at 9700; no print
    # follows, and from the decision at 9760 on the fill is known
    r = run(PRE + [(9700.5, SELL, 0.45, 5.0)])
    assert r.inventory_end == 5.0
    assert [s[1] for s in seen] == [0.0, 0.0, 5.0, 5.0, 5.0, 5.0]


def test_the_known_inventory_catches_up_after_a_decision_without_a_quote(monkeypatch):
    seen = record_quotes(monkeypatch)
    rows = PRE + [
        (9690.0, BUY, 0.99, 1.0),  # fills 1 at the ask 0.53 of the first interval: q -1
        (9690.5, SELL, 0.98, 1.0),  # the proxy at 9700 is 0.985: no quote decided at 9700
        (9700.5, SELL, 0.45, 5.0),  # still the first interval's bid: q +4; the proxy is 0.72 again
    ]
    r = run(rows)
    assert (r.fills, r.inventory_end, r.quotes_live_share) == (2, 4.0, pytest.approx(5 / 6))
    assert [s[1] for s in seen] == [0.0, -1.0, 4.0, 4.0, 4.0, 4.0]
    assert [s[0] for s in seen] == pytest.approx([0.5, 0.985, 0.72, 0.72, 0.72, 0.72])
    assert math.isnan(seen[1][4]) and math.isnan(seen[1][5])  # no quote at the second decision


def test_the_volatility_of_a_decision_is_the_larger_of_the_estimate_and_the_floor(monkeypatch):
    seen = record_quotes(monkeypatch)
    # a buy and, a second later, a sale every 300 s from 7000: the proxy alternates between two
    # levels, so the realized volatility is defined and differs from one decision to the next
    rows = [
        (7000.0 + 300.0 * i + lag, side, price, 1.0)
        for i in range(10)
        for lag, side, price in (
            (0.0, BUY, 0.60 + 0.10 * (i % 2)),
            (1.0, SELL, 0.50 + 0.10 * (i % 2)),
        )
    ]
    ts, side, price, size = tape_of(rows)
    decisions = RELEASE - HAND.window_hours * 3600.0 + HAND.refresh_s * np.arange(6)
    vol = realized_logit_vol(ts, side, price, decisions, lookback_s=HAND.vol_lookback_s,
                             stale_s=HAND.stale_s)
    assert np.all(np.isfinite(vol)) and vol.min() < vol.max()
    for floor in (0.0, 0.5 * (vol.min() + vol.max()), 2.0 * vol.max()):
        seen.clear()
        run(rows, floor=floor)
        np.testing.assert_allclose([s[3] for s in seen], np.maximum(vol, floor), rtol=1e-12)
    middle = 0.5 * (vol.min() + vol.max())
    assert (vol < middle).any() and (vol > middle).any()  # the middle floor binds in some decisions


def test_no_fill_before_the_first_quotes_are_live_and_none_after_the_release():
    rows = PRE + [(9640.5, SELL, 0.45, 5.0), (9641.0, BUY, 0.55, 5.0)]
    assert run(rows).fills == 0
    after = PRE + [(10000.0, SELL, 0.45, 5.0), (10001.0, BUY, 0.55, 5.0), (10500.0, BUY, 0.9, 1.0)]
    assert run(after).fills == 0  # the last interval is live until 10001, the release comes first
    inside = PRE + [(9999.9999, SELL, 0.45, 5.0)]
    assert run(inside).fills == 1


def test_size_is_restored_at_each_refresh():
    rows = PRE + [(9650.0, SELL, 0.45, 10.0), (9651.0, SELL, 0.45, 10.0),
                  (9710.0, SELL, 0.45, 10.0), (9711.0, SELL, 0.45, 10.0)]
    r = run(rows)
    assert (r.fills, r.contracts, r.inventory_end) == (2, 20.0, 20.0)
    assert r.quotes_live_share == 1.0


def test_the_two_sides_have_their_own_size():
    rows = PRE + [(9650.0, SELL, 0.45, 10.0), (9651.0, BUY, 0.55, 10.0),
                  (9652.0, SELL, 0.45, 10.0), (9653.0, BUY, 0.55, 10.0)]
    r = run(rows)
    assert (r.fills, r.contracts, r.inventory_end) == (2, 20.0, 0.0)


def test_a_fill_is_never_larger_than_the_trade():
    rows = PRE + [(9650.0, SELL, 0.45, 1.0), (9651.0, SELL, 0.45, 2.5), (9652.0, BUY, 0.55, 0.5)]
    r = run(rows)
    assert (r.fills, r.contracts, r.inventory_end) == (3, 4.0, 3.0)


def test_the_position_limit_closes_the_side_at_the_next_refresh():
    small = replace(HAND, max_position=20.0)
    rows = PRE + [
        (9650.0, SELL, 0.45, 9.0),  # q 9
        (9710.0, SELL, 0.45, 10.0),  # decision at 9700 saw 9: bid open, q 19
        (9770.0, SELL, 0.45, 10.0),  # decision at 9760 saw 19 < 20: bid open, q 29
        (9830.0, SELL, 0.45, 10.0),  # decision at 9820 saw 29 >= 20: no bid
        (9890.0, BUY, 0.55, 4.0),  # the ask is open: q 25
        (9950.0, SELL, 0.45, 10.0),  # decision at 9940 saw 25: no bid
    ]
    r = run(rows, params=small)
    assert (r.fills, r.contracts, r.inventory_end) == (4, 33.0, 25.0)
    assert r.quotes_live_share == 1.0  # one side is always open
    # both sides are quoted at 9640, 9700 and 9760 only (bid 0.47, ask 0.53): the refreshes with
    # one side closed are left out of the mean spread
    assert r.mean_spread == pytest.approx(0.06, abs=1e-12)
    assert 25.0 <= small.max_position + small.size
    mirrored = [(ts, -s, 1.0 - p, c) for ts, s, p, c in rows]  # the same tape upside down
    r = run(mirrored, params=small)
    assert (r.fills, r.contracts, r.inventory_end) == (4, 33.0, -25.0)


def test_mean_spread_is_ask_minus_bid_of_the_quotes_averaged_over_the_refreshes_with_both_sides(
        monkeypatch):
    seen = record_quotes(monkeypatch)
    r = run(S1, "release_aware", premiums=PREMIUMS)
    both = [ask - bid for *_, bid, ask in seen if bid == bid and ask == ask]
    assert len(both) == 6 == len(seen)
    assert r.mean_spread == pytest.approx(sum(both) / 6, abs=1e-12)
    # by hand at a mid of 0.50 with no inventory: the premium is 4 * 0.25 * 0.08 at 6 minutes to
    # the release (bucket (5,15]) and 4 * 0.25 * 0.10 in the five refreshes of the bucket (0,5];
    # the spread is the tick-rounded 2 * (0.0252494 + premium): 0.22 and 0.26
    assert r.mean_spread == pytest.approx((0.22 + 5 * 0.26) / 6, abs=1e-12)
    assert run(S1).mean_spread == pytest.approx(0.06, abs=1e-12)  # always_on, no premium


# ---------------------------------------------------------------------- premium at the fills
# a tape that fills every policy in two refreshes: in each of them a sell at 0.20 and a buy at 0.80
# print through any quote below, and the two keep the proxy at 0.50 for the next decision
FILL_TAPE = PRE + [
    (9650.0, SELL, 0.20, 6.0),  # refresh 0, 6 minutes to the release, bucket (5,15]: 6 on the bid
    (9655.0, BUY, 0.80, 4.0),  # 4 on the ask
    (9710.0, SELL, 0.20, 3.0),  # refresh 1, 5 minutes to the release, bucket (0,5]: 3 on the bid
    (9715.0, BUY, 0.80, 2.0),  # 2 on the ask
]


def test_the_premium_at_the_fills_is_the_premium_of_the_refreshes_hit_weighted_by_contracts():
    params = replace(HAND, flat_scale=1.5)
    got = {policy: run(FILL_TAPE, policy, premiums=PREMIUMS, params=params) for policy in POLICIES}
    aware = got["release_aware"]
    # 10 contracts filled at the premium of the bucket (5,15] (0.08) and 5 at that of (0,5] (0.10)
    assert (aware.fills, aware.contracts) == (4, 15.0)
    assert aware.fill_premium == pytest.approx((10 * 0.08 + 5 * 0.10) / 15, abs=1e-12)
    # not the premium averaged over the six refreshes of the window, which is (0.08 + 5 * 0.10) / 6
    assert aware.fill_premium < (0.08 + 5 * 0.10) / 6 - 0.005
    assert got["pooled_premium"].fill_premium == pytest.approx(0.05, abs=1e-12)
    assert got["flat_matched"].fill_premium == pytest.approx(0.04 * 1.5, abs=1e-12)  # matched * F
    assert got["always_on"].fill_premium == 0.0
    # pull_only is filled in the first refresh only: the others are inside its pull window
    assert (got["pull_only"].contracts, got["pull_only"].fill_premium) == (10.0, 0.0)


def test_a_refresh_without_a_fill_does_not_count_in_the_premium_at_the_fills():
    for tau_minutes, unit in ((6, 0.08), (1, 0.10)):
        ts = RELEASE - 60.0 * tau_minutes + 10.0  # 10 s into the interval of that decision
        rows = PRE + [(ts, SELL, 0.20, 5.0)]
        r = run(rows, "release_aware", premiums=PREMIUMS)
        assert (r.fills, r.contracts) == (1, 5.0)
        assert r.fill_premium == pytest.approx(unit, abs=1e-12)


def test_the_premium_at_the_fills_weights_the_contracts_filled_not_the_size_of_the_prints():
    rows = PRE + [
        (9650.0, SELL, 0.20, 30.0),  # only the 10 contracts of the bid are filled
        (9710.0, SELL, 0.20, 5.0),  # 5 contracts at the premium of the next refresh
    ]
    r = run(rows, "release_aware", premiums=PREMIUMS)
    assert (r.fills, r.contracts) == (2, 15.0)
    assert r.fill_premium == pytest.approx((10 * 0.08 + 5 * 0.10) / 15, abs=1e-12)


def test_the_touch_rule_gives_the_premium_of_the_refresh_to_the_fills_that_it_adds():
    # prints at the quotes of release_aware: 0.39 and 0.61 in refresh 0, 0.37 and 0.63 in refresh 1
    rows = PRE + [
        (9650.0, SELL, 0.39, 4.0), (9655.0, BUY, 0.61, 2.0),
        (9710.0, SELL, 0.37, 3.0), (9715.0, BUY, 0.63, 1.0),
    ]
    through = run(rows, "release_aware", premiums=PREMIUMS)
    touch = run(rows, "release_aware", premiums=PREMIUMS, params=replace(HAND, fill_rule="touch"))
    assert through.fills == 0 and math.isnan(through.fill_premium)
    assert (touch.fills, touch.contracts) == (4, 10.0)
    assert touch.fill_premium == pytest.approx((6 * 0.08 + 4 * 0.10) / 10, abs=1e-12)


def test_mean_fill_premium_weights_the_rows_by_their_contracts_and_skips_those_without_a_fill():
    rows = pd.DataFrame({"contracts": [10.0, 0.0, 30.0], "fill_premium": [0.10, NAN, 0.20]})
    assert bt.mean_fill_premium(rows) == pytest.approx((10 * 0.10 + 30 * 0.20) / 40, abs=1e-15)
    assert bt.mean_fill_premium(rows.iloc[[0]]) == 0.10
    assert math.isnan(bt.mean_fill_premium(rows.iloc[[1]]))
    assert math.isnan(bt.mean_fill_premium(rows.iloc[:0]))


def test_the_latency_can_take_the_inventory_past_the_limit_plus_one_size():
    """A decision does not see the fills in the second after it: a bid that was open for the
    interval before can still fill (up to the size) while the new bid, decided with the old
    inventory, is about to go live. The bound is the limit plus two sizes, not one."""
    small = replace(HAND, max_position=20.0)
    rows = PRE + [
        (9650.0, SELL, 0.45, 10.0),  # q 10
        (9710.0, SELL, 0.45, 9.0),  # q 19 < 20
        (9820.5, SELL, 0.45, 10.0),  # between decision 3 (9820) and its quotes: the old bid, q 29
        (9830.0, SELL, 0.45, 10.0),  # the new bid was decided with q 19: fills, q 39
        (9890.0, SELL, 0.45, 10.0),  # decision 4 sees 39: no bid
    ]
    r = run(rows, params=small)
    assert r.inventory_end == 39.0
    assert r.inventory_end > small.max_position + small.size
    assert r.inventory_end < small.max_position + 2 * small.size
    # without the latency the decision sees every fill and the bound is limit plus one size
    now = run(rows, params=replace(small, latency_s=0.0))
    assert now.inventory_end <= small.max_position + small.size


# the pull window: 12 refreshes of 60 s, the quotes decided 6 minutes before the release are live
# until 9701 = 299 s before it; the decision 5 minutes before the release (9700) is pulled
PULL = replace(HAND, window_hours=0.2)
PULL_PRE = [(9000.0, BUY, 0.55, 5.0), (9010.0, SELL, 0.45, 5.0)]
PULL_ROWS = PULL_PRE + [
    (9700.5, SELL, 0.45, 3.0),  # still the quote decided 6 minutes before the release
    (9701.0, SELL, 0.45, 2.0),  # exactly at the end of that quote: it fills
    (9701.5, SELL, 0.45, 3.0),  # now the decision 5 minutes before the release is live: pulled
    (9990.0, BUY, 0.55, 5.0),  # ten seconds before the release
]


def test_pull_only_has_no_quote_in_the_last_five_minutes():
    always = run(PULL_ROWS, "always_on", params=PULL)
    pulled = run(PULL_ROWS, "pull_only", params=PULL)
    assert (always.fills, always.contracts) == (4, 13.0)
    assert (pulled.fills, pulled.contracts) == (2, 5.0)  # 9700.5 and 9701.0 only
    assert always.quotes_live_share == 1.0
    assert pulled.quotes_live_share == pytest.approx(7 / 12)  # refreshes 12 ... 6 minutes before


def test_a_pulled_market_still_earns_with_the_quote_before_the_window():
    far = PULL_PRE + [(9300.0, SELL, 0.45, 4.0), (9400.0, BUY, 0.55, 4.0)]
    r = run(far, "pull_only", y=0, params=PULL)
    assert (r.fills, r.inventory_end) == (2, 0.0)
    assert r.gross == pytest.approx(-4 * 0.47 + 4 * 0.53)


# premium policies: the proxy is 0.50 with prints at 0.25 and 0.75, one print per interval
WIDE_PRE = [(9500.0, BUY, 0.75, 5.0), (9510.0, SELL, 0.25, 5.0)]
WIDE = WIDE_PRE + [
    (9650.0, SELL, 0.25, 4.0), (9710.0, BUY, 0.75, 4.0), (9770.0, SELL, 0.25, 4.0),
    (9830.0, BUY, 0.75, 4.0), (9890.0, SELL, 0.25, 4.0), (9950.0, BUY, 0.75, 4.0),
]


def test_pooled_premium_quotes_the_pooled_premium_in_every_interval():
    r = run(WIDE, "pooled_premium", premiums=PREMIUMS)
    # bid 0.42 and ask 0.58 (see test_quoting) in all six intervals: three sales to us, three
    # purchases from us, 4 contracts each
    assert (r.fills, r.contracts, r.inventory_end) == (6, 24.0, 0.0)
    assert r.gross == pytest.approx(3 * 4 * (0.58 - 0.42), abs=1e-12)
    assert r.fees == pytest.approx(6 * 0.0171, abs=1e-12)  # 0.0175 * 4 * 0.2436 = 0.017052
    assert r.pnl == pytest.approx(1.92 - 0.1026, abs=1e-12)


def test_release_aware_quotes_the_premium_of_the_bucket_of_each_refresh():
    r = run(WIDE, "release_aware", premiums=PREMIUMS)
    # refreshes 6, 5, ..., 1 minutes before the release: bucket (5,15] with premium 0.08 at 6
    # minutes (quotes 0.39/0.61), bucket (0,5] with 0.10 from 5 minutes on (quotes 0.37/0.63). The
    # sales to us come in the refreshes of 6, 4 and 2 minutes, the purchases from us in those of 5,
    # 3 and 1 minutes, so every ask that fills is 0.63
    bids = [0.39, 0.37, 0.37]
    cash = -4 * sum(bids) + 4 * 3 * 0.63
    assert cash == pytest.approx(3.04)
    assert (r.fills, r.contracts, r.inventory_end) == (6, 24.0, 0.0)
    assert r.gross == pytest.approx(cash, abs=1e-12)
    assert r.fees == pytest.approx(0.0167 + 5 * 0.0164, abs=1e-12)
    assert r.pnl == pytest.approx(3.04 - 0.0987, abs=1e-12)


def test_flat_matched_quotes_the_matched_premium_in_every_interval():
    r = run(WIDE, "flat_matched", premiums=PREMIUMS)
    # matched 0.04: h = 4 * 0.25 * 0.04 = 0.04, bid 0.5 - 0.02525 - 0.04 = 0.43475 -> 0.43 and
    # ask 0.56525 -> 0.57 in all six intervals; the prints at 0.25 and 0.75 are through them
    assert (r.fills, r.contracts, r.inventory_end) == (6, 24.0, 0.0)
    assert r.gross == pytest.approx(3 * 4 * (0.57 - 0.43), abs=1e-12)
    assert r.fees == pytest.approx(6 * 0.0172, abs=1e-12)  # 0.0175 * 4 * 0.2451 = 0.017157
    assert r.pnl == pytest.approx(1.68 - 0.1032, abs=1e-12)
    # flat_scale multiplies the premium: 2 gives h = 0.08, bid 0.39475 -> 0.39, ask 0.60525 -> 0.61
    double = run(WIDE, "flat_matched", premiums=PREMIUMS, params=replace(HAND, flat_scale=2.0))
    assert double.gross == pytest.approx(3 * 4 * (0.61 - 0.39), abs=1e-12)
    assert double.fees == pytest.approx(6 * 0.0167, abs=1e-12)  # 0.0175 * 4 * 0.2379 = 0.016653


def test_flat_matched_needs_the_matched_premium():
    without = Premiums(0.05, (0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01), 40)
    with pytest.raises(ValueError, match="matched"):
        run(WIDE, "flat_matched", premiums=without)


def test_the_three_premium_policies_agree_when_all_their_premiums_are_equal():
    same = Premiums(0.03, (0.03,) * 7, 40, 0.03)
    results = {p: run(WIDE, p, premiums=same) for p in
               ("pooled_premium", "flat_matched", "release_aware")}
    assert len({(r.pnl, r.fills, r.contracts, r.fees) for r in results.values()}) == 1


def test_the_premium_widens_the_quote_until_the_prints_no_longer_reach_it():
    rows = PRE + [(9650.0, SELL, 0.45, 4.0), (9710.0, BUY, 0.55, 4.0)]
    # at a mid of 0.50 the premium of 4 m (1 - m) = 1 times the rate is added to the half-spread
    # 0.02525: the quotes are 0.47/0.53 at rate 0, 0.46/0.54 at 0.01, 0.45/0.55 at 0.02 and
    # 0.43/0.57 at 0.04, and the prints are at 0.45 and 0.55
    for rule, want in (("through", (2, 2, 0, 0)), ("touch", (2, 2, 2, 0))):
        params = replace(HAND, fill_rule=rule)
        got = tuple(
            run(rows, "pooled_premium", premiums=Premiums(r, (r,) * 7, 40), params=params).fills
            for r in (0.0, 0.01, 0.02, 0.04)
        )
        assert got == want


def test_a_market_without_a_valid_proxy_gets_no_fills():
    only_buys = [(9500.0, BUY, 0.55, 5.0), (9650.0, BUY, 0.55, 5.0), (9720.0, BUY, 0.90, 5.0)]
    for policy in POLICIES:
        r = run(only_buys, policy, premiums=PREMIUMS)
        assert (r.fills, r.contracts, r.inventory_end, r.fees) == (0, 0.0, 0.0, 0.0)
        assert (r.pnl, r.gross, r.marked_pnl, r.quotes_live_share) == (0.0, 0.0, 0.0, 0.0)
        assert math.isnan(r.mean_spread)  # no quote, no spread
        assert math.isnan(r.fill_premium)  # no fill, no premium at the fills
    empty = run([(9500.0, BUY, 0.5, 1.0)], "always_on", release=9000.0)  # all after the release
    assert (empty.fills, empty.quotes_live_share) == (0, 0.0)
    assert math.isnan(empty.mean_spread) and math.isnan(empty.fill_premium)


def test_an_empty_tape_gives_an_empty_result():
    r = backtest_market(np.array([]), np.array([], dtype=np.int8), np.array([]), np.array([]),
                        RELEASE, 1, "release_aware", PREMIUMS, 0.1, HAND)
    assert (r.fills, r.contracts, r.pnl, r.gross, r.marked_pnl, r.quotes_live_share) == (
        0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert math.isnan(r.mean_spread) and math.isnan(r.fill_premium)


def test_no_quote_when_the_proxy_is_outside_the_range():
    low = [(9500.0, BUY, 0.03, 5.0), (9510.0, SELL, 0.01, 5.0)]  # proxy 0.02
    assert run(low).quotes_live_share == 0.0
    assert math.isnan(run(low).mean_spread)
    # a buy at 0.99 at 9650 moves the proxy to 0.50, seen from the decision at 9700 on
    moved = low + [(9650.0, BUY, 0.99, 5.0)]
    assert run(moved).quotes_live_share == pytest.approx(5 / 6)
    assert run(moved).mean_spread == pytest.approx(0.06, abs=1e-12)  # the five quoted refreshes


def test_the_range_of_the_proxy_is_inclusive_at_the_edges():
    edge = [(9500.0, BUY, 0.05, 5.0), (9510.0, SELL, 0.01, 5.0)]  # proxy 0.03
    assert run(edge).quotes_live_share == 1.0
    edge = [(9500.0, BUY, 0.99, 5.0), (9510.0, SELL, 0.95, 5.0)]  # proxy 0.97
    assert run(edge).quotes_live_share == 1.0
    below = [(9500.0, BUY, 0.04, 5.0), (9510.0, SELL, 0.01, 5.0)]  # proxy 0.025
    assert run(below).quotes_live_share == 0.0


def test_the_proxy_becomes_valid_in_the_window():
    late = [(9700.0, BUY, 0.55, 5.0), (9705.0, SELL, 0.45, 5.0), (9800.0, SELL, 0.45, 5.0)]
    r = run(late)  # decisions at 9640 and 9700 see no sale (the sell is at 9705): quotes from 9760
    assert r.quotes_live_share == pytest.approx(4 / 6)
    assert r.fills == 1  # the sale at 9800 hits the bid decided at 9760


def test_a_stale_proxy_gives_no_quote():
    rows = PRE + [(9650.0, SELL, 0.45, 5.0)]
    fresh = run(rows)
    stale = run(rows, params=replace(HAND, stale_s=100.0))
    assert fresh.quotes_live_share == 1.0
    # the last buy, at 9500, is 140 s old at the first decision and older at every later one
    assert stale.quotes_live_share == 0.0
    assert stale.fills == 0


def test_zero_maker_fee_coefficient_removes_the_fees():
    r = run(S1, params=replace(HAND, maker_fee_coeff=0.0))
    assert r.fees == 0.0 and r.pnl == r.gross
    assert r.pnl == pytest.approx(-0.04, abs=1e-12)


def test_the_fee_is_charged_per_fill_not_per_contract():
    split = PRE + [(9650.0, SELL, 0.45, 1.0), (9651.0, SELL, 0.45, 1.0), (9652.0, SELL, 0.45, 1.0)]
    r = run(split)
    assert r.fills == 3
    assert r.fees == pytest.approx(3 * maker_fee(1.0, 0.47), abs=1e-12)  # 3 x 0.0044
    assert r.fees > maker_fee(3.0, 0.47)  # 0.0131 in one fill: each fill is rounded up on its own


def test_the_mark_is_the_last_valid_proxy_before_the_release():
    rows = PRE + [(9650.0, SELL, 0.45, 5.0), (9990.0, BUY, 0.85, 1.0)]  # last proxy 0.65
    r = run(rows)
    # the sale at 9650 fills 5 at 0.47; the buy at 0.85 is through the ask, fills 1 at 0.53
    assert r.inventory_end == 4.0
    cash = -5 * 0.47 + 0.53
    assert r.marked_pnl == pytest.approx(cash + 4 * 0.65 - r.fees, abs=1e-12)
    assert r.pnl == pytest.approx(cash + 4 * 1.0 - r.fees, abs=1e-12)


def test_the_mark_falls_back_to_the_last_decision_proxy_when_the_release_proxy_is_stale():
    # with stale_s = 100 the proxy at the release is undefined (the last buy is 360 s old); at the
    # decision at 9700 the last buy (0.55) is 60 s old and the last sale (0.40) 40 s old; at the
    # decision at 9640 there is no sale yet, and at 9760 the buy is 120 s old
    params = replace(HAND, stale_s=100.0)
    rows = [(9640.0, BUY, 0.55, 5.0), (9641.0, SELL, 0.45, 5.0), (9660.0, SELL, 0.40, 5.0)]
    grid = prepare_market(*tape_of(rows), RELEASE, params)
    assert np.isnan(grid.mid).tolist() == [True, False, True, True, True, True]
    assert grid.mid[1] == pytest.approx(0.475)
    assert grid.mark == pytest.approx(0.475)
    nothing = prepare_market(*tape_of([(9640.0, BUY, 0.55, 5.0)]), RELEASE, params)
    assert math.isnan(nothing.mark)


# ---------------------------------------------------------------------- the prepared market
def test_prepared_intervals_partition_the_prints():
    ts, side, price, size = tape_of(S1)
    grid = prepare_market(ts, side, price, size, RELEASE, HAND)
    assert grid.n_intervals == 6
    assert len(grid.side) == len(S1) - 2  # the two prints at or after the release are dropped
    assert np.all(grid.first <= grid.split) and np.all(grid.split <= grid.stop)
    assert np.array_equal(grid.stop[:-1], grid.first[1:])  # consecutive intervals
    assert grid.first[0] == 4 and grid.stop[-1] == len(S1) - 2  # prints 0 to 3 precede the quotes
    # interval 0 holds the prints at 9650, 9660, 9670 and 9701; the last one is after the decision
    # at 9700, so it is the only print of the second part
    assert (grid.first[0], grid.split[0], grid.stop[0]) == (4, 7, 8)
    # the starts and ends of the other intervals, from the live windows (t + 1, t + 61]
    assert grid.first.tolist() == [4, 8, 11, 13, 14, 14]
    assert grid.stop.tolist() == [8, 11, 13, 14, 14, 15]
    assert grid.split.tolist() == [7, 11, 13, 14, 14, 15]
    np.testing.assert_allclose(grid.tau_min, [6, 5, 4, 3, 2, 1])
    np.testing.assert_allclose(grid.mid, 0.5)
    assert grid.mark == pytest.approx(0.5) and grid.release_ts == RELEASE


def test_prepare_market_counts_the_intervals_of_the_window():
    ts, side, price, size = tape_of(PRE)
    for hours, refresh, expected in ((12.0, 60.0, 720), (1.0, 45.0, 80), (0.1, 60.0, 6),
                                     (0.1, 70.0, 6), (0.1, 7.0, 52)):
        params = QuoteParams(window_hours=hours, refresh_s=refresh)
        assert prepare_market(ts, side, price, size, RELEASE, params).n_intervals == expected


def test_prepare_market_refuses_bad_input():
    ts, side, price, size = tape_of(PRE)
    with pytest.raises(ValueError):
        prepare_market(ts[::-1], side, price, size, RELEASE, HAND)  # unsorted
    with pytest.raises(ValueError):
        prepare_market(ts, np.array([1, 0], dtype=np.int8), price, size, RELEASE, HAND)
    with pytest.raises(ValueError):
        prepare_market(ts, side, price, np.array([5.0, 0.0]), RELEASE, HAND)  # size zero
    with pytest.raises(ValueError):
        prepare_market(ts, side, np.array([0.5, NAN]), size, RELEASE, HAND)
    with pytest.raises(ValueError):
        prepare_market(ts, side, price[:1], size, RELEASE, HAND)
    with pytest.raises(ValueError):
        prepare_market(ts, side, price, size, float("inf"), HAND)


def test_backtest_arguments_are_checked():
    args = (*tape_of(PRE), RELEASE)
    with pytest.raises(ValueError):
        backtest_market(*args, 1, "never", ZERO, 0.1, HAND)
    with pytest.raises(ValueError):
        backtest_market(*args, 2, "always_on", ZERO, 0.1, HAND)  # outcome
    with pytest.raises(ValueError):
        backtest_market(*args, NAN, "always_on", ZERO, 0.1, HAND)
    with pytest.raises(ValueError):
        backtest_market(*args, 1, "always_on", ZERO, -0.1, HAND)  # floor


def test_event_result_has_the_fields_of_the_brief():
    names = [f.name for f in fields(EventResult)]
    assert names == ["event", "market", "policy", "pnl", "gross", "fees", "fills", "contracts",
                     "inventory_end", "marked_pnl", "quotes_live_share", "mean_spread",
                     "fill_premium"]


def _same_result(a, b):
    """Every field of the two results is equal, a NaN equalling a NaN: from Python 3.13 on a
    dataclass compares its fields one by one, so two results without a fill are not equal."""
    for f in fields(EventResult):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert x == y or (x != x and y != y), (f.name, x, y)


def test_backtest_grid_is_backtest_market_on_a_prepared_grid():
    ts, side, price, size = tape_of(S1)
    grid = prepare_market(ts, side, price, size, RELEASE, HAND)
    for policy in POLICIES:
        a = backtest_grid(grid, 1, policy, PREMIUMS, 0.3, HAND, event="E", market="M")
        b = backtest_market(ts, side, price, size, RELEASE, 1, policy, PREMIUMS, 0.3, HAND,
                            event="E", market="M")
        _same_result(a, b)


# ------------------------------------------------------------ the slow reference implementation
# Plain loops over the prints, written for clarity: the quote from the formulas of model.md 6, the
# proxy from a scan of the prints, the interval of every print from the live windows.
def ref_proxy(trades, t, stale):
    last = {}
    for ts, side, price, _ in trades:
        if ts <= t:
            last[side] = (ts, price)
    if len(last) < 2 or any(t - when > stale for when, _ in last.values()):
        return NAN
    return (last[1][1] + last[-1][1]) / 2.0


def ref_vol(trades, t, lookback, stale, grid=300.0):
    steps = int(math.floor(lookback / grid + 1e-9))
    points = [ref_proxy(trades, t - grid * i, stale) for i in range(steps, -1, -1)]
    xs = [None if not 0.0 < m < 1.0 else math.log(m / (1.0 - m)) for m in points]
    squares = [(xs[i + 1] - xs[i]) ** 2 for i in range(len(xs) - 1)
               if xs[i] is not None and xs[i + 1] is not None]
    if sum(x is not None for x in xs) < 6 or not squares:
        return NAN
    return math.sqrt(sum(squares) / len(squares) / (grid / 3600.0))


def ref_quote(policy, m, q, tau, sigma, premiums, p):
    if not p.min_mid - 1e-9 <= m <= p.max_mid + 1e-9:
        return None
    if policy == "pull_only" and tau <= p.pull_minutes:
        return None
    mm = m * (1.0 - m)
    v = (sigma * mm) ** 2 * tau / 60.0
    h = 0.0
    if policy in ("pooled_premium", "flat_matched", "release_aware"):
        v += mm
        rate = {
            "pooled_premium": premiums.pooled,
            "flat_matched": None if premiums.matched is None else premiums.matched * p.flat_scale,
            "release_aware": premiums.by_bucket[int(bucket_index(tau))],
        }[policy]
        h = 4.0 * mm * rate
    r = m - q * p.gamma * v
    delta = p.gamma * v + (2.0 / p.gamma) * math.log(1.0 + p.gamma / p.k)
    top = round(1.0 / p.tick) - 1
    k_bid = max(1, min(top, math.floor((r - delta / 2 - h) / p.tick + 1e-9)))
    k_ask = max(1, min(top, math.ceil((r + delta / 2 + h) / p.tick - 1e-9)))
    if k_ask - k_bid < 1:
        if k_bid + 1 <= top:
            k_ask = k_bid + 1
        else:
            k_bid = k_ask - 1
    bid, ask = round(k_bid * p.tick, 10), round(k_ask * p.tick, 10)
    return (NAN if q >= p.max_position else bid, NAN if q <= -p.max_position else ask)


def ref_decisions(trades, release, p):
    """Proxy and volatility at every decision time, from prints at or before it."""
    window = p.window_hours * 3600.0
    n = math.ceil(window / p.refresh_s - 1e-9)
    times = [release - window + j * p.refresh_s for j in range(n)]
    mids = [ref_proxy(trades, t, p.stale_s) for t in times]
    vols = [ref_vol(trades, t, p.vol_lookback_s, p.stale_s) for t in times]
    return mids, vols


def ref_backtest(trades, release, y, policy, premiums, floor, p, decisions):
    mids, vols = decisions
    window = p.window_hours * 3600.0
    fills = []  # (time of the print, signed contracts, price)
    live = 0
    for j, m in enumerate(mids):
        t = release - window + j * p.refresh_s
        inventory = sum(c for when, c, _ in fills if when <= t)
        sigma = floor if math.isnan(vols[j]) else max(vols[j], floor)
        quote = ref_quote(policy, m, inventory, (window - j * p.refresh_s) / 60.0, sigma, premiums,
                          p)
        if quote is None:
            continue
        live += 1
        bid, ask = quote
        bid_left = ask_left = p.size
        for ts, side, price, size in trades:
            if not t + p.latency_s < ts <= t + p.refresh_s + p.latency_s:
                continue
            if side == SELL and bid_left > 0 and not math.isnan(bid):
                if price < bid or (p.fill_rule == "touch" and price == bid):
                    take = min(bid_left, size)
                    bid_left -= take
                    fills.append((ts, take, bid))
            if side == BUY and ask_left > 0 and not math.isnan(ask):
                if price > ask or (p.fill_rule == "touch" and price == ask):
                    take = min(ask_left, size)
                    ask_left -= take
                    fills.append((ts, -take, ask))
    inventory = sum(c for _, c, _ in fills)
    cash = -sum(c * pr for _, c, pr in fills)
    fees = sum(float(kalshi_fee(abs(c), pr, p.maker_fee_coeff)) for _, c, pr in fills)
    at_release = ref_proxy(trades, release, p.stale_s)
    mark = at_release if not math.isnan(at_release) else next(
        (m for m in reversed(mids) if not math.isnan(m)), NAN)
    return {
        "pnl": cash + inventory * y - fees, "gross": cash + inventory * y, "fees": fees,
        "fills": len(fills), "contracts": sum(abs(c) for _, c, _ in fills),
        "inventory_end": inventory,
        "marked_pnl": cash + (inventory * mark if inventory != 0 else 0.0) - fees,
        "quotes_live_share": live / len(mids),
    }


def assert_same(result, want, context=""):
    for key, value in want.items():
        got = getattr(result, key) if hasattr(result, key) else result[key]
        assert got == pytest.approx(value, abs=1e-9), f"{context} {key}: {got} != {value}"


def small_world(seed, *, n_events=6, window_hours=2.0, min_history_events=5, **kwargs):
    """A world with short windows, so that the tests run in a second, and matching parameters."""
    params = WorldParams(n_events=n_events, window_hours=window_hours, **kwargs)
    world = generate_world(params, seed)
    return world, QuoteParams(
        window_hours=window_hours, vol_lookback_s=3600.0, min_history_events=min_history_events
    )


def market_arrays(world, event):
    tape = world.tape[world.tape["market"] == event]
    release = float(world.events.set_index("event").loc[event, "release_ts"])
    y = int(world.markets.set_index("market").loc[event, "outcome"])
    return tape, release, y


@pytest.mark.parametrize("rule", ["through", "touch"])
def test_the_engine_equals_the_slow_reference_on_synthetic_tapes(rule):
    """All five policies, both fill rules, several markets: every field agrees."""
    world, params = small_world(21, n_events=5)
    params = replace(params, fill_rule=rule)
    checked = 0
    for event in world.events["event"]:
        tape, release, y = market_arrays(world, event)
        trades = list(zip(tape["ts"], tape["side"].astype(int), tape["price"], tape["size"],
                          strict=True))
        decisions = ref_decisions(trades, release, params)
        for policy in POLICIES:
            got = backtest_market(
                tape["ts"].to_numpy(), tape["side"].to_numpy(), tape["price"].to_numpy(),
                tape["size"].to_numpy(), release, y, policy, PREMIUMS, 0.4, params,
            )
            want = ref_backtest(trades, release, y, policy, PREMIUMS, 0.4, params, decisions)
            assert_same(got, want, f"{event} {policy} {rule}")
            checked += got.fills
    assert checked > 100  # the comparison has fills in it


def test_the_engine_equals_the_reference_with_a_position_limit_and_no_latency():
    world, params = small_world(33, n_events=4)
    params = replace(params, max_position=15.0, size=5.0, latency_s=0.0, gamma=0.01, k=30.0)
    hits = 0
    for event in world.events["event"]:
        tape, release, y = market_arrays(world, event)
        trades = list(zip(tape["ts"], tape["side"].astype(int), tape["price"], tape["size"],
                          strict=True))
        decisions = ref_decisions(trades, release, params)
        for policy in ("always_on", "release_aware"):
            got = backtest_market(
                tape["ts"].to_numpy(), tape["side"].to_numpy(), tape["price"].to_numpy(),
                tape["size"].to_numpy(), release, y, policy, PREMIUMS, 0.4, params,
            )
            assert_same(got, ref_backtest(trades, release, y, policy, PREMIUMS, 0.4, params,
                                          decisions), f"{event} {policy}")
            assert abs(got.inventory_end) <= params.max_position + params.size  # no latency
            hits += abs(got.inventory_end) >= params.max_position
    assert hits >= 1  # the limit does bind in this comparison


# ------------------------------------------------------------------------- look-ahead guards
def synthetic_market(seed, event="SYN-000001", n_events=2):
    world, params = small_world(seed, n_events=n_events)
    tape, release, y = market_arrays(world, event)
    arrays = {c: tape[c].to_numpy().copy() for c in ("ts", "side", "price", "size")}
    return arrays, release, y, params


def run_arrays(arrays, release, y, params, policy="release_aware", floor=0.01):
    return backtest_market(arrays["ts"], arrays["side"], arrays["price"], arrays["size"], release,
                           y, policy, PREMIUMS, floor, params)


def test_changing_the_prints_after_a_decision_leaves_the_decision_unchanged(monkeypatch):
    arrays, release, y, params = synthetic_market(5)
    cut = release - 3600.0  # half way through the 2 hour window: decisions 0 to 60 are at or before
    late = arrays["ts"] > cut
    assert late.sum() > 20
    rng = np.random.default_rng(11)
    changed = {name: values.copy() for name, values in arrays.items()}
    for name in ("side", "price", "size"):
        changed[name][late] = rng.permutation(arrays[name][late])
    assert not np.array_equal(changed["price"], arrays["price"])

    first = record_quotes(monkeypatch)
    run_arrays(arrays, release, y, params)
    second = record_quotes(monkeypatch)
    run_arrays(changed, release, y, params)
    a, b = np.array(first), np.array(second)
    assert a.shape == b.shape == (120, 6)
    np.testing.assert_array_equal(a[:61], b[:61])  # mid, inventory, volatility, bid and ask
    assert np.any(a[:61, 3] != 0.01)  # the volatility is defined and moves the quotes
    assert np.any(a[:61, 4] > 0.0)  # there are quotes
    assert not np.array_equal(a[61:], b[61:], equal_nan=True)  # the later decisions do change


def test_deleting_the_prints_after_a_decision_leaves_the_decision_unchanged(monkeypatch):
    arrays, release, y, params = synthetic_market(7)
    cut = release - 5400.0  # decisions at or before it: 0 to 30
    kept = {name: values[arrays["ts"] <= cut] for name, values in arrays.items()}
    first = record_quotes(monkeypatch)
    run_arrays(arrays, release, y, params)
    second = record_quotes(monkeypatch)
    run_arrays(kept, release, y, params)
    np.testing.assert_array_equal(np.array(first)[:31], np.array(second)[:31])


def test_the_decisions_do_not_depend_on_the_outcome(monkeypatch):
    arrays, release, _, params = synthetic_market(8)
    lost = record_quotes(monkeypatch)
    a = run_arrays(arrays, release, 0, params)
    won = record_quotes(monkeypatch)
    b = run_arrays(arrays, release, 1, params)
    np.testing.assert_array_equal(np.array(lost), np.array(won))
    assert a.inventory_end == b.inventory_end != 0.0
    assert b.pnl - a.pnl == pytest.approx(a.inventory_end, abs=1e-9)


def test_prints_at_or_after_the_release_change_nothing():
    arrays, release, y, params = synthetic_market(6)
    extra_ts = release + np.array([0.0, 0.5, 1.0, 50.0, 4000.0])
    extended = {
        "ts": np.concatenate([arrays["ts"], extra_ts]),
        "side": np.concatenate([arrays["side"], np.array([1, -1, 1, -1, 1], dtype=np.int8)]),
        "price": np.concatenate([arrays["price"], [0.99, 0.01, 0.99, 0.01, 0.50]]),
        "size": np.concatenate([arrays["size"], [50.0] * 5]),
    }
    for policy in POLICIES:
        assert run_arrays(extended, release, y, params, policy) == run_arrays(
            arrays, release, y, params, policy)


def test_the_prints_of_the_last_interval_cannot_fill_a_pulled_quote():
    """pull_only has no quote live after release - 5 minutes + latency: prints after that time
    change neither the fills nor the inventory nor the share of live quotes."""
    world, params = small_world(3, n_events=8)
    gaps, fills = [], 0
    for event in world.events["event"]:
        tape, release, y = market_arrays(world, event)
        arrays = {c: tape[c].to_numpy() for c in ("ts", "side", "price", "size")}
        until = release - 300.0 + params.latency_s
        early = {name: values[arrays["ts"] <= until] for name, values in arrays.items()}
        full = run_arrays(arrays, release, y, params, "pull_only", 0.1)
        cut = run_arrays(early, release, y, params, "pull_only", 0.1)
        for field in ("pnl", "gross", "fees", "fills", "contracts", "inventory_end",
                      "quotes_live_share"):
            assert getattr(full, field) == getattr(cut, field)
        always = run_arrays(arrays, release, y, params, "always_on", 0.1)
        gaps.append(always.quotes_live_share - full.quotes_live_share)
        fills += full.fills
    # the five decisions at 5 to 1 minutes before the release are the only ones without a quote
    assert all(0.0 <= g <= 5 / 120 + 1e-12 for g in gaps)
    assert max(gaps) == pytest.approx(5 / 120)  # in some market all five have a proxy in range
    assert fills > 50


def test_fills_are_bounded_by_the_prints_and_the_inventory_by_the_limits():
    world, params = small_world(9, n_events=6)
    tight = replace(params, max_position=15.0, size=5.0, latency_s=0.0)
    for event in world.events["event"]:
        tape, release, y = market_arrays(world, event)
        before = tape[tape["ts"] < release]
        for policy in POLICIES:
            r = backtest_market(tape["ts"].to_numpy(), tape["side"].to_numpy(),
                                tape["price"].to_numpy(), tape["size"].to_numpy(), release, y,
                                policy, PREMIUMS, 0.1, tight)
            assert r.fills <= len(before)
            assert r.contracts <= before["size"].sum()
            assert r.contracts <= r.fills * tight.size  # no fill is larger than the quote size
            assert abs(r.inventory_end) <= tight.max_position + tight.size  # no latency
            assert r.gross == pytest.approx(r.pnl + r.fees, abs=1e-12)
            assert 0.0 <= r.quotes_live_share <= 1.0


def test_the_engine_does_not_read_the_latent_columns():
    """Only ts, side, price and size are passed to the engine; walk_forward is run on a tape with
    garbage in, and without, the latent columns."""
    world, params = small_world(10, n_events=12)
    policies = ("always_on", "release_aware")
    base = walk_forward(world.tape, world.markets, world.events, policies, params)
    garbage = world.tape.copy()
    rng = np.random.default_rng(0)
    for column in [c for c in garbage if c.startswith("latent_")]:
        garbage[column] = rng.random(len(garbage)) * 1e6
    events = world.events.assign(latent_x0=1e9)
    stripped = world.tape.drop(columns=[c for c in world.tape if c.startswith("latent_")])
    bare = world.events.drop(columns=["latent_x0"])
    pd.testing.assert_frame_equal(
        base, walk_forward(garbage, world.markets, events, policies, params))
    pd.testing.assert_frame_equal(
        base, walk_forward(stripped, world.markets, bare, policies, params))
    assert base["fills"].sum() > 0


# ------------------------------------------------------------------------------- walk_forward
WF_COLUMNS = ["event", "market", "policy", "pnl", "gross", "fees", "fills", "contracts",
              "inventory_end", "marked_pnl", "quotes_live_share", "mean_spread", "fill_premium",
              "evaluated", "release_ts", "series"]


def window_vols(tape, market, release, params):
    """The defined realized volatilities at the decision times of one market."""
    part = tape[tape["market"] == market]
    window = params.window_hours * 3600.0
    times = release - window + params.refresh_s * np.arange(math.ceil(window / params.refresh_s))
    vol = realized_logit_vol(part["ts"].to_numpy(), part["side"].to_numpy(),
                             part["price"].to_numpy(), times, lookback_s=params.vol_lookback_s,
                             stale_s=params.stale_s)
    return vol[~np.isnan(vol)]


def explicit_rows(world, policies, params, burn_in=None):
    """The walk-forward written out event by event from the pieces that are tested on their own:
    the premiums and the volatility floor of an event come from the events released at or before
    the start of its window, every market is run with backtest_market."""
    burn = params.min_history_events if burn_in is None else burn_in
    markouts = trade_markouts(world.tape, world.events, world.markets, horizon_s=params.horizon_s,
                              stale_s=params.stale_s)
    window = params.window_hours * 3600.0
    # the minutes to the release of each decision of the window, written out without the library
    minutes = [(window - params.refresh_s * j) / 60.0
               for j in range(max(1, math.ceil(window / params.refresh_s - 1e-9)))]
    events = world.events.sort_values(["release_ts", "event"]).reset_index(drop=True)
    release_of = dict(zip(world.events["event"], world.events["release_ts"], strict=True))
    event_of = dict(zip(world.markets["market"], world.markets["event"], strict=True))
    outcome_of = dict(zip(world.markets["market"], world.markets["outcome"], strict=True))
    markets_of = {e: sorted(g["market"]) for e, g in world.markets.groupby("event")}
    rows = []
    for rank, (event, release, series) in enumerate(
            zip(events["event"], events["release_ts"], events["series"], strict=True)):
        earlier = set(events.loc[events["release_ts"] <= release - window, "event"])
        premiums = estimate_premiums(markouts[markouts["event"].isin(earlier)],
                                     min_events=params.min_history_events, minutes=minutes)
        vols = [window_vols(world.tape, m, release_of[event_of[m]], params)
                for m in world.markets["market"] if event_of[m] in earlier]
        pooled = np.concatenate(vols) if vols else np.array([])
        floor = float(pooled.mean()) if pooled.size else params.sigma_default
        for market in markets_of.get(event, []):
            part = world.tape[world.tape["market"] == market]
            for policy in policies:
                r = backtest_market(
                    part["ts"].to_numpy(), part["side"].to_numpy(), part["price"].to_numpy(),
                    part["size"].to_numpy(), release, outcome_of[market], policy, premiums, floor,
                    params, event=event, market=market)
                rows.append({**asdict(r), "evaluated": rank >= burn, "release_ts": release,
                             "series": series})
    return pd.DataFrame(rows, columns=WF_COLUMNS)


def merge_pairs(world):
    """A world in which the market of every odd event belongs to the even event before it, with
    its prints moved in time to the release of that event: events of two markets."""
    ids = list(world.events["event"])
    target = {ids[i + 1]: ids[i] for i in range(0, len(ids) - 1, 2)}
    release = dict(zip(world.events["event"], world.events["release_ts"], strict=True))
    market = world.tape["market"].to_numpy()
    shift = np.array([release[target[m]] - release[m] if m in target else 0.0 for m in market])
    tape = world.tape.assign(
        ts=world.tape["ts"].to_numpy() + shift,
        event=[target.get(m, e) for m, e in zip(market, world.tape["event"], strict=True)],
    )
    markets = world.markets.assign(
        event=[target.get(m, e) for m, e in zip(world.markets["market"], world.markets["event"],
                                                strict=True)]
    )
    events = world.events[~world.events["event"].isin(list(target))].reset_index(drop=True)
    return tape, markets, events


def assert_rows_equal(got, want):
    pd.testing.assert_frame_equal(got.reset_index(drop=True), want.reset_index(drop=True),
                                  check_exact=False, atol=1e-9, rtol=1e-9)


def test_walk_forward_equals_the_event_by_event_computation():
    world, params = small_world(12, n_events=30)
    got = walk_forward(world.tape, world.markets, world.events, POLICIES, params)
    assert_rows_equal(got, explicit_rows(world, POLICIES, params))
    assert len(got) == 30 * len(POLICIES)
    assert got["fills"].sum() > 1000 and (got["fills"] == 0).sum() < 75


def test_walk_forward_layout_and_burn_in():
    world, params = small_world(13, n_events=26)
    policies = ("release_aware", "always_on")
    got = walk_forward(world.tape, world.markets, world.events, policies, params)
    assert list(got.columns) == WF_COLUMNS
    assert got["policy"].tolist() == list(policies) * 26  # the order of the policies is kept
    assert got["event"].tolist() == [e for e in world.events["event"] for _ in policies]
    assert got["evaluated"].tolist() == [i >= 5 for i in range(26) for _ in policies]
    assert got["series"].eq("SYN").all()
    np.testing.assert_array_equal(got["release_ts"].to_numpy()[::2],
                                  world.events["release_ts"].to_numpy())
    for burn, count in ((0, 26 * 2), (3, 23 * 2), (26, 0), (40, 0)):
        again = walk_forward(world.tape, world.markets, world.events, policies, params,
                             burn_in=burn)
        assert int(again["evaluated"].sum()) == count
        assert_rows_equal(again.drop(columns="evaluated"), got.drop(columns="evaluated"))
    assert got["marked_pnl"].notna().all() and got["pnl"].notna().all()


def test_walk_forward_ignores_the_order_of_the_input_rows():
    world, params = small_world(14, n_events=24)
    base = walk_forward(world.tape, world.markets, world.events, POLICIES, params)
    rng = np.random.default_rng(5)
    events = world.events.iloc[rng.permutation(len(world.events))].reset_index(drop=True)
    markets = world.markets.iloc[rng.permutation(len(world.markets))].reset_index(drop=True)
    shuffled = walk_forward(world.tape, markets, events, POLICIES, params)
    pd.testing.assert_frame_equal(base, shuffled)  # exactly the same rows in the same order
    reverse = walk_forward(world.tape, markets.iloc[::-1], events.iloc[::-1], POLICIES, params)
    pd.testing.assert_frame_equal(base, reverse)


def test_walk_forward_accepts_any_iterable_of_policies():
    world, params = small_world(15, n_events=8)
    a = walk_forward(world.tape, world.markets, world.events, iter(["always_on"]), params)
    b = walk_forward(world.tape, world.markets, world.events, ["always_on"], params)
    pd.testing.assert_frame_equal(a, b)


def test_a_premium_policy_gets_its_premiums_whether_it_runs_alone_or_with_the_others():
    world, params = small_world(17, n_events=24)
    together = walk_forward(world.tape, world.markets, world.events, POLICIES, params)
    for policy in ("pooled_premium", "flat_matched", "release_aware"):
        alone = walk_forward(world.tape, world.markets, world.events, (policy,), params)
        assert_rows_equal(alone, together[together["policy"] == policy])
    flat = together[together["policy"] == "flat_matched"]
    assert flat["fills"].sum() > 100
    assert not flat["pnl"].equals(together.loc[together["policy"] == "always_on", "pnl"])


def test_walk_forward_gives_a_premium_at_the_fills_to_the_rows_with_a_fill_and_to_those_only():
    world, params = small_world(13, n_events=26)
    got = walk_forward(world.tape, world.markets, world.events, POLICIES, params)
    filled = got["contracts"] > 0.0
    assert filled.sum() > 100 and (~filled).sum() > 0
    assert got["fill_premium"].isna().eq(~filled).all()
    for policy in POLICIES:
        premium = got.loc[filled & got["policy"].eq(policy), "fill_premium"]
        if policy in PREMIUM_POLICIES:
            assert (premium >= 0.0).all() and premium.max() > 0.0
        else:
            assert (premium == 0.0).all()  # no premium: nothing to weigh
    # the rows without history (the first five events) quote no premium and so carry none
    for policy in PREMIUM_POLICIES:
        first = got[got["policy"].eq(policy) & ~got["evaluated"] & filled]
        assert len(first) > 0 and (first["fill_premium"] == 0.0).all()


def test_events_of_two_markets_run_each_market_on_its_own():
    world, params = small_world(16, n_events=30)
    tape, markets, events = merge_pairs(world)
    assert len(events) == 15 and markets["event"].value_counts().eq(2).all()
    merged = replace(world, tape=tape, markets=markets, events=events)
    got = walk_forward(tape, markets, events, POLICIES, params)
    assert_rows_equal(got, explicit_rows(merged, POLICIES, params))
    assert len(got) == 30 * len(POLICIES)  # one row per market and policy
    per_event = got.groupby(["event", "policy"])["pnl"].sum()
    assert len(per_event) == 15 * len(POLICIES)
    assert got["market"].nunique() == 30 and got["event"].nunique() == 15

    # the second market of an event does not change the first one, and the other way round
    first, second = markets["market"].iloc[0], markets["market"].iloc[1]
    assert markets.set_index("market").loc[[first, second], "event"].nunique() == 1
    changed = tape.copy()
    mask = (changed["market"] == second).to_numpy()
    changed.loc[mask, "price"] = np.random.default_rng(1).permutation(changed.loc[mask, "price"])
    again = walk_forward(changed, markets, events, POLICIES, params)
    same = got["market"] == first
    assert_rows_equal(again[same], got[same])
    assert not got.loc[got["market"] == second, "pnl"].equals(
        again.loc[again["market"] == second, "pnl"])


def test_walk_forward_does_not_use_later_events():
    """Changing, flipping or removing the events after event 20 leaves everything up to it as it was
    (the premiums and the volatility floor of an event come from earlier events only)."""
    world, params = small_world(17, n_events=40)
    base = walk_forward(world.tape, world.markets, world.events, POLICIES, params)
    ids = list(world.events["event"])
    later = set(ids[21:])
    upto = base[~base["event"].isin(later)]
    assert len(upto) == 21 * len(POLICIES)

    rng = np.random.default_rng(2)
    tape = world.tape.copy()
    mask = tape["market"].isin(later).to_numpy()
    for column in ("price", "side", "size"):
        tape.loc[mask, column] = rng.permutation(tape.loc[mask, column].to_numpy())
    markets = world.markets.assign(
        outcome=np.where(world.markets["market"].isin(later), 1 - world.markets["outcome"],
                         world.markets["outcome"]))
    changed = walk_forward(tape, markets, world.events, POLICIES, params)
    assert_rows_equal(changed[~changed["event"].isin(later)], upto)
    assert not changed["pnl"].equals(base["pnl"])  # the later events themselves do change

    kept = world.tape[~world.tape["event"].isin(later)]
    removed = walk_forward(kept, world.markets[~world.markets["event"].isin(later)],
                           world.events[~world.events["event"].isin(later)], POLICIES, params)
    assert_rows_equal(removed, upto)


def test_the_outcomes_of_other_events_do_not_matter():
    world, params = small_world(18, n_events=30)
    base = walk_forward(world.tape, world.markets, world.events, POLICIES, params)
    target = world.events["event"].iloc[25]
    flipped = world.markets.assign(
        outcome=np.where(world.markets["event"] == target, world.markets["outcome"],
                         1 - world.markets["outcome"]))
    got = walk_forward(world.tape, flipped, world.events, POLICIES, params)
    assert_rows_equal(got[got["event"] == target], base[base["event"] == target])
    # every other event keeps its fills and fees, and its settled P&L moves by q (y_new - y_old)
    other = got["event"] != target
    for field in ("fills", "contracts", "fees", "inventory_end", "marked_pnl",
                  "quotes_live_share", "mean_spread", "fill_premium"):
        np.testing.assert_allclose(got.loc[other, field], base.loc[other, field], atol=1e-12)
    sign = 1.0 - 2.0 * world.markets.set_index("market").loc[got.loc[other, "market"],
                                                             "outcome"].to_numpy()
    np.testing.assert_allclose(got.loc[other, "pnl"] - base.loc[other, "pnl"],
                               sign * base.loc[other, "inventory_end"], atol=1e-9)
    assert (base["inventory_end"] != 0.0).sum() > 40


def test_an_event_released_at_the_window_start_is_history_one_second_later_it_is_not():
    hours = 2.0
    got = {}
    for name, spacing in (("at", hours), ("inside", hours - 1.0 / 3600.0)):
        world, params = small_world(19, n_events=8, min_history_events=1,
                                    event_spacing_hours=spacing)
        got[name] = walk_forward(world.tape, world.markets, world.events, ["pooled_premium"],
                                 params, burn_in=0)
        assert_rows_equal(got[name], explicit_rows(world, ["pooled_premium"], params, burn_in=0))
    at, inside = got["at"].drop(columns="release_ts"), got["inside"].drop(columns="release_ts")
    assert_rows_equal(at.iloc[[0]], inside.iloc[[0]])  # no history for the first event either way
    # the second event has the first one as history in one world and not in the other (a premium
    # estimated from one event changes the quotes, so the fills differ)
    assert at["fills"].iloc[1] != inside["fills"].iloc[1]


def test_an_event_released_inside_the_window_is_not_history():
    world, params = small_world(20, n_events=12, min_history_events=2, event_spacing_hours=1.5)
    policies = ["pooled_premium", "release_aware"]
    base = walk_forward(world.tape, world.markets, world.events, policies, params, burn_in=0)
    assert_rows_equal(base, explicit_rows(world, policies, params, burn_in=0))
    ids = list(world.events["event"])
    target = ids[8]  # window start 2 hours before its release: events 0 to 6 are released by then

    def shuffled_prices(event):
        tape = world.tape.copy()
        mask = (tape["event"] == event).to_numpy()
        tape.loc[mask, "price"] = np.random.default_rng(3).permutation(tape.loc[mask, "price"])
        return walk_forward(tape, world.markets, world.events, policies, params, burn_in=0)

    inside = shuffled_prices(ids[7])  # released 1.5 hours before the target: inside its window
    assert_rows_equal(inside[inside["event"] == target], base[base["event"] == target])
    earlier = shuffled_prices(ids[6])  # released 3 hours before the target: history
    assert not earlier.loc[earlier["event"] == target, "pnl"].equals(
        base.loc[base["event"] == target, "pnl"])


# ------------------------------------------------------------------------- economics sanity
def gross_by_event(seeds, n_events, rule, pi0):
    """Gross P&L per event of always_on without fees in a world with ``pi1 = 0`` and informed
    share ``pi0``, the events of every seed pooled; and the fills per event."""
    values, fills = [], 0
    for seed in seeds:
        world = generate_world(WorldParams(n_events=n_events, pi0=pi0, pi1=0.0, kappa=1.5), seed)
        params = QuoteParams(maker_fee_coeff=0.0, fill_rule=rule)
        rows = walk_forward(world.tape, world.markets, world.events, ["always_on"], params,
                            burn_in=0)
        values.append(rows.groupby("event")["gross"].sum().to_numpy())
        fills += int(rows["fills"].sum())
    return np.concatenate(values), fills / (len(seeds) * n_events)


@pytest.fixture(scope="module")
def no_information():
    """Seeds 31 to 33, 300 events each, no informed trading (pi0 = pi1 = 0, kappa = 1.5)."""
    return {rule: gross_by_event((31, 32, 33), 300, rule, 0.0) for rule in ("through", "touch")}


def test_without_informed_trading_always_on_does_not_lose_if_a_print_at_the_quote_fills(
        no_information, cluster):
    """The expected gross P&L is not negative: the mean per event is compared with its standard
    error over the 900 events (each event is a cluster). The others quote at the fair value, so
    the margin is the tick rounding: the maker earns half a tick on the fills at the quote."""
    values, fills = no_information["touch"]
    mean, se = cluster.mean_se(values, np.arange(values.size))
    assert fills > 50
    assert mean > -2.0 * se


def test_without_informed_trading_a_through_fill_earns_nothing_because_there_is_no_spread(
        no_information, cluster):
    """With the others quoting at the fair value a print strictly beyond our quote means the price
    has moved through it: the through rule fills the maker only when the news has gone against the
    quote. Its expected gross P&L cannot be significantly positive, which a fill at the print
    price or a sign error would make it."""
    values, fills = no_information["through"]
    mean, se = cluster.mean_se(values, np.arange(values.size))
    assert fills > 5
    assert mean < 2.0 * se
    touch_mean = no_information["touch"][0].mean()
    assert touch_mean > mean  # the fills at the quote are what the through rule leaves out


def test_with_strongly_informed_flow_a_policy_without_premium_loses(no_information, cluster):
    for rule in ("through", "touch"):
        values, fills = gross_by_event((41, 42), 150, rule, 0.3)
        mean, se = cluster.mean_se(values, np.arange(values.size))
        assert fills > 5
        assert mean < -5.0 * se
        base, base_se = cluster.mean_se(no_information[rule][0],
                                        np.arange(no_information[rule][0].size))
        assert mean - base < -5.0 * math.hypot(se, base_se)  # informed flow costs the maker
