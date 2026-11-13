"""The mid proxy of a trades-only tape (kalshi_mm.tape, docs/model.md section 4)."""

import numpy as np
import pytest

from kalshi_mm.tape import mid_proxy_asof
from kalshi_mm.world import WorldParams, generate_world

NAN = np.nan

# a hand-made tape of one market; seconds, with two prints at t = 40 (a buy, then a sell)
TS = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 40.0])
SIDE = np.array([1, -1, 1, -1, 1, -1], dtype=np.int8)
PRICE = np.array([0.50, 0.46, 0.52, 0.48, 0.54, 0.47])


def same(a, b):
    """Equal arrays with NaN in the same places and values equal to 1e-12."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    assert np.array_equal(np.isnan(a), np.isnan(b)), (a, b)
    ok = ~np.isnan(a)
    assert np.allclose(a[ok], b[ok], rtol=0.0, atol=1e-12), (a, b)


def slow_proxy(ts, side, price, queries, stale_s, inclusive):
    """The definition as plain loops: scan every print, keep the last one of each side."""
    out = []
    for q in queries:
        last = {}
        for i in range(len(ts)):
            if (ts[i] <= q) if inclusive else (ts[i] < q):
                last[int(side[i])] = (ts[i], price[i])  # the later print of the side replaces it
        if len(last) == 2 and all(q - when <= stale_s for when, _ in last.values()):
            out.append((last[1][1] + last[-1][1]) / 2.0)
        else:
            out.append(NAN)
    return np.array(out)


def random_tape(rng, n):
    """Times on a 1-second grid so that prints tie, sides at random, prices on the cent grid."""
    ts = np.sort(rng.integers(0, 4000, n)).astype(float)
    side = rng.choice(np.array([1, -1], dtype=np.int8), n)
    price = rng.integers(1, 100, n) / 100.0
    return ts, side, price


# ------------------------------------------------------------------------------ hand examples
def test_hand_example_inclusive():
    queries = [5.0, 10.0, 15.0, 25.0, 35.0, 40.0, 100.0]
    got = mid_proxy_asof(TS, SIDE, PRICE, queries)
    # 5: no sell yet. 10: buy 0.50 (t=0), sell 0.46 (t=10). 15: same. 25: buy 0.52, sell 0.46.
    # 35: buy 0.52, sell 0.48. 40: buy 0.54, and the later of the two sells at 40, 0.47. 100: same.
    same(got, [NAN, 0.48, 0.48, 0.49, 0.50, 0.505, 0.505])


def test_hand_example_strict():
    queries = [5.0, 10.0, 15.0, 25.0, 35.0, 40.0, 100.0]
    got = mid_proxy_asof(TS, SIDE, PRICE, queries, inclusive=False)
    # at 10 and 40 the print at the query time no longer counts: no sell before 10; at 40 the last
    # buy is 0.52 (t=20) and the last sell 0.48 (t=30)
    same(got, [NAN, NAN, 0.48, 0.49, 0.50, 0.50, 0.505])


def test_a_print_at_the_query_time_counts_only_when_inclusive():
    t, s, p = np.array([5.0, 8.0]), np.array([1, -1], dtype=np.int8), np.array([0.60, 0.50])
    assert mid_proxy_asof(t, s, p, 8.0) == pytest.approx(0.55)
    assert np.isnan(mid_proxy_asof(t, s, p, 8.0, inclusive=False))
    assert mid_proxy_asof(t, s, p, 8.0000001, inclusive=False) == pytest.approx(0.55)


def test_the_last_print_of_equal_times_counts():
    t = np.array([1.0, 2.0, 2.0, 2.0])
    s = np.array([1, 1, -1, 1], dtype=np.int8)
    p = np.array([0.10, 0.20, 0.30, 0.40])  # at t = 2 the last buy is 0.40 (it comes last)
    assert mid_proxy_asof(t, s, p, 2.0) == pytest.approx(0.35)
    assert np.isnan(mid_proxy_asof(t, s, p, 2.0, inclusive=False))  # no sell before 2


def test_output_shape_and_scalar_query():
    assert isinstance(mid_proxy_asof(TS, SIDE, PRICE, 25.0), float)
    grid = np.array([[15.0, 25.0], [35.0, 5.0]])
    same(mid_proxy_asof(TS, SIDE, PRICE, grid), [[0.48, 0.49], [0.50, NAN]])
    assert mid_proxy_asof(TS, SIDE, PRICE, np.array([])).shape == (0,)
    assert np.isnan(mid_proxy_asof(TS, SIDE, PRICE, NAN))  # an undefined time gives no proxy
    assert np.isnan(mid_proxy_asof(TS, SIDE, PRICE, -np.inf))
    assert np.isnan(mid_proxy_asof(TS, SIDE, PRICE, np.inf))  # too long after the last prints


# ------------------------------------------------------------------------------------ staleness
def test_staleness_is_measured_for_each_side():
    # at 25 the last buy is 5 s old (t=20) and the last sell 15 s old (t=10)
    assert mid_proxy_asof(TS, SIDE, PRICE, 25.0, stale_s=15.0) == pytest.approx(0.49)  # not stale
    assert np.isnan(mid_proxy_asof(TS, SIDE, PRICE, 25.0, stale_s=14.999))  # the sell is stale
    # at 36 the last buy is 16 s old (t=20) and the last sell only 6 s old (t=30)
    assert np.isnan(mid_proxy_asof(TS, SIDE, PRICE, 36.0, stale_s=15.0))  # the buy is stale
    assert mid_proxy_asof(TS, SIDE, PRICE, 36.0, stale_s=16.0) == pytest.approx(0.50)


def test_default_staleness_is_one_hour():
    t, s, p = np.array([0.0, 1.0]), np.array([1, -1], dtype=np.int8), np.array([0.6, 0.4])
    assert mid_proxy_asof(t, s, p, 3600.0) == pytest.approx(0.5)  # the buy is exactly 3600 s old
    assert np.isnan(mid_proxy_asof(t, s, p, 3600.5))  # more than 3600 s old
    assert mid_proxy_asof(t, s, p, 3600.5, stale_s=3601.0) == pytest.approx(0.5)


def test_a_one_sided_tape_has_no_proxy():
    queries = np.array([0.0, 10.0, 100.0])
    for sign in (1, -1):
        t = np.array([1.0, 2.0, 3.0])
        s = np.full(3, sign, dtype=np.int8)
        assert np.isnan(mid_proxy_asof(t, s, np.array([0.4, 0.5, 0.6]), queries)).all()
    assert np.isnan(mid_proxy_asof([], np.array([], dtype=np.int8), [], queries)).all()


# ----------------------------------------------------------------------------------- look-ahead
@pytest.mark.parametrize("inclusive", [True, False])
def test_later_prints_do_not_change_the_proxy(inclusive):
    """Deleting, altering or appending prints after the cut changes no proxy at or before it. With
    ``inclusive=False`` a print at exactly the cut does not count either, so the strict proxy at the
    cut is also unchanged when prints from the cut on are deleted."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        t, s, p = random_tape(rng, 400)
        cut = float(rng.integers(500, 3500))
        grid = np.concatenate([rng.uniform(0.0, cut, 150), t[t <= cut], t[t <= cut] + 0.5, [cut]])
        grid = grid[grid <= cut]
        base = mid_proxy_asof(t, s, p, grid, inclusive=inclusive)
        later = t > cut if inclusive else t >= cut
        same(mid_proxy_asof(t[~later], s[~later], p[~later], grid, inclusive=inclusive), base)
        p2, s2 = p.copy(), s.copy()
        p2[later] = rng.integers(1, 100, later.sum()) / 100.0
        s2[later] = rng.choice(np.array([1, -1], dtype=np.int8), later.sum())
        same(mid_proxy_asof(t, s2, p2, grid, inclusive=inclusive), base)
        extra_t = cut + 1.0 + np.sort(rng.integers(0, 100, 30)).astype(float)
        t3 = np.concatenate([t[~later], extra_t])
        s3 = np.concatenate([s[~later], rng.choice(np.array([1, -1], dtype=np.int8), 30)])
        p3 = np.concatenate([p[~later], rng.integers(1, 100, 30) / 100.0])
        same(mid_proxy_asof(t3, s3, p3, grid, inclusive=inclusive), base)


def test_altering_a_print_before_the_query_does_change_the_proxy():
    """Check the check: the guard above would catch a proxy that ignored the prints it needs."""
    p2 = PRICE.copy()
    p2[2] = 0.60  # the last buy before 25
    assert mid_proxy_asof(TS, SIDE, p2, 25.0) == pytest.approx(0.53)
    assert mid_proxy_asof(TS, SIDE, PRICE, 25.0) == pytest.approx(0.49)


# ------------------------------------------------------------------------- slow loop and a world
@pytest.mark.parametrize("inclusive", [True, False])
@pytest.mark.parametrize("stale_s", [0.0, 30.0, 400.0, 3600.0])
def test_equals_a_slow_loop_on_random_tapes(inclusive, stale_s):
    rng = np.random.default_rng(int(stale_s) + 5)
    for _ in range(6):
        t, s, p = random_tape(rng, int(rng.integers(0, 120)))
        queries = np.concatenate(
            [rng.uniform(-100.0, 4200.0, 150), t, t + 1.0, t - 1.0, [NAN, -np.inf, np.inf]]
        )
        got = mid_proxy_asof(t, s, p, queries, stale_s=stale_s, inclusive=inclusive)
        same(got, slow_proxy(t, s, p, queries, stale_s, inclusive))


def test_proxy_on_the_tape_of_a_synthetic_world_equals_a_slow_loop():
    world = generate_world(WorldParams(n_events=2, window_hours=2.0), seed=3)
    for market, rows in world.tape.groupby("market"):
        t, s, p = rows["ts"].to_numpy(), rows["side"].to_numpy(), rows["price"].to_numpy()
        assert len(rows) > 100 and market.startswith("SYN-")
        for inclusive in (True, False):
            got = mid_proxy_asof(t, s, p, t, stale_s=600.0, inclusive=inclusive)
            same(got, slow_proxy(t, s, p, t, 600.0, inclusive))
            assert np.isnan(got[0]) and np.isfinite(got[-1])  # a proxy needs both sides to print
        # the proxy of a market with no latent columns is the same: it reads time, side and price
        no_latent = rows.drop(columns=[c for c in rows if c.startswith("latent_")])
        same(
            mid_proxy_asof(no_latent["ts"], no_latent["side"], no_latent["price"], t),
            mid_proxy_asof(t, s, p, t),
        )


# ------------------------------------------------------------------------------------ validation
def test_invalid_input_is_refused():
    with pytest.raises(ValueError, match="sorted"):
        mid_proxy_asof([2.0, 1.0], np.array([1, -1], dtype=np.int8), [0.5, 0.5], 3.0)
    with pytest.raises(ValueError, match="sorted"):
        mid_proxy_asof([1.0, NAN], np.array([1, -1], dtype=np.int8), [0.5, 0.5], 3.0)
    with pytest.raises(ValueError, match="same length"):
        mid_proxy_asof([1.0, 2.0], np.array([1], dtype=np.int8), [0.5, 0.5], 3.0)
    with pytest.raises(ValueError, match="same length"):
        mid_proxy_asof(np.ones((2, 2)), np.ones((2, 2)), np.ones((2, 2)), 3.0)
    with pytest.raises(ValueError, match="side"):
        mid_proxy_asof([1.0, 2.0], np.array([1, 0]), [0.5, 0.5], 3.0)
    with pytest.raises(ValueError, match="stale_s"):
        mid_proxy_asof(TS, SIDE, PRICE, 25.0, stale_s=-1.0)
    with pytest.raises(ValueError, match="stale_s"):
        mid_proxy_asof(TS, SIDE, PRICE, 25.0, stale_s=NAN)
