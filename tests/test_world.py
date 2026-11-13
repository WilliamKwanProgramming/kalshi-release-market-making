"""The synthetic world (docs/model.md sections 2 and 3): schema, determinism and closed forms.

Statistical tests compare an estimate with a closed-form target within ``Z`` standard errors (at
most 4). Every standard error treats the event as the cluster (the ``cluster`` fixture of
conftest.py), because trades of one event are dependent. The two 2000-event worlds are built once
per module with the default parameters (``planted``) and with ``pi1 = 0`` (``null``).
"""

import dataclasses
import pathlib
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import integrate
from scipy.special import lambertw

from kalshi_mm.logit import information_step, logit, sigmoid
from kalshi_mm.world import (
    World,
    WorldParams,
    _cum_rate,
    _invert_cum_rate,
    _quotes,
    generate_world,
    informed_share,
    trade_rate,
)

SEED = 20261113
Z = 4.0
TAPE_COLUMNS = ["event", "market", "ts", "seq", "price", "size", "side", "latent_mid_pre",
                "latent_mid_post", "latent_pi", "latent_informed"]
BUCKETS = [(0.0, 5.0), (5.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 180.0), (180.0, 360.0),
           (360.0, np.inf)]  # minutes to release, right-closed


def within(estimate, target, se, what=""):
    """Assert ``|estimate - target| <= Z se`` with a readable message."""
    assert abs(estimate - target) <= Z * se, (
        f"{what}: estimate {estimate:.6g}, target {target:.6g}, se {se:.3g}, "
        f"z = {(estimate - target) / se:.2f}"
    )


def trades_of(world: World) -> SimpleNamespace:
    """The tape of a world as plain arrays plus the quantities the tests need."""
    tape, events, markets = world.tape, world.events, world.markets
    codes = pd.Index(events["event"]).get_indexer(tape["event"])  # event number of each trade
    release = events["release_ts"].to_numpy()[codes]
    tau_h = (release - tape["ts"].to_numpy()) / 3600.0
    return SimpleNamespace(
        world=world, params=world.params, codes=codes, n_events=len(events), tau_h=tau_h,
        tau_min=60.0 * tau_h, y=markets["outcome"].to_numpy()[codes].astype(float),
        side=tape["side"].to_numpy().astype(float), ts=tape["ts"].to_numpy(),
        price=tape["price"].to_numpy(), size=tape["size"].to_numpy(),
        pre=tape["latent_mid_pre"].to_numpy(), post=tape["latent_mid_post"].to_numpy(),
        pi=tape["latent_pi"].to_numpy(), informed=tape["latent_informed"].to_numpy(),
        ones=np.ones(len(tape)),
    )


def in_bucket(t, lo, hi):
    return (t.tau_min > lo) & (t.tau_min <= hi)


@pytest.fixture(scope="module")
def planted():
    return trades_of(generate_world(WorldParams(n_events=2000), SEED))


@pytest.fixture(scope="module")
def null():
    return trades_of(generate_world(WorldParams(n_events=2000, pi1=0.0), SEED))


# ------------------------------------------------------------------- tests of the test helpers
def test_cluster_helper_hand_examples(cluster):
    values, codes = np.array([1.0, 3.0, 5.0]), np.array([0, 0, 1])  # clusters {1, 3} and {5}
    mean, se = cluster.mean_se(values, codes)
    assert mean == pytest.approx(3.0) and se == pytest.approx(4.0 / 3.0)  # sqrt(2 * (4 + 4)) / 3
    ratio, se = cluster.ratio_se(values, np.array([2.0, 2.0, 1.0]), codes)
    assert ratio == pytest.approx(1.8) and se == pytest.approx(1.28)  # sqrt(2 * 20.48) / 5
    diff, se = cluster.diff_se(values, np.ones(3), codes, np.array([1, 0, 1], bool),
                               np.array([0, 1, 1], bool))
    assert diff == pytest.approx(-1.0) and se == pytest.approx(1.0)  # 3 - 4; sqrt(2 * 0.5)
    mean, se = cluster.mean_se(values, codes, np.array([1, 1, 0], bool))  # a mask drops rows
    assert mean == 2.0 and np.isnan(se)  # one cluster left: no standard error


def test_cluster_helper_covers_the_truth(cluster):
    """Check the checker: 95% intervals built with the helper cover a known mean and a known
    difference, 1500 samples of 150 clusters each, within 4 Monte Carlo standard errors of 0.95."""
    rng = np.random.default_rng(31)
    R, G, mu, delta = 1500, 150, 0.3, 0.2
    cover_mean = cover_diff = 0
    for _ in range(R):
        n = 1 + rng.poisson(8, G)
        codes = np.repeat(np.arange(G), n)
        group = rng.random(codes.size) < 0.5
        values = mu + delta * group + rng.normal(0, 1, G)[codes] + rng.normal(0, 1, codes.size)
        m, se = cluster.mean_se(values, codes, ~group)
        cover_mean += abs(m - mu) <= 1.96 * se
        d, se = cluster.diff_se(values, np.ones(codes.size), codes, group, ~group)
        cover_diff += abs(d - delta) <= 1.96 * se
    mc_se = np.sqrt(0.95 * 0.05 / R)
    assert abs(cover_mean / R - 0.95) < Z * mc_se
    assert abs(cover_diff / R - 0.95) < Z * mc_se


# ------------------------------------------------------------------------------- parameters
def test_default_parameters_are_the_specified_ones():
    p = WorldParams()
    assert (p.n_events, p.window_hours, p.sigma, p.x0_sd) == (200, 12.0, 0.2, 1.0)
    assert (p.base_rate, p.surge, p.surge_hours) == (10.0, 20.0, 1.0)
    assert (p.pi0, p.pi1, p.pi_hours, p.kappa, p.tick) == (0.05, 0.20, 0.5, 1.5, 0.01)
    assert (p.mean_size, p.event_spacing_hours, p.t0) == (10.0, 168.0, 1_700_000_000.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.sigma = 1.0


def test_documented_parameter_table_matches_the_defaults():
    """Section 11 of docs/model.md lists every field of WorldParams, once, with its default."""
    doc = pathlib.Path(__file__).resolve().parents[1] / "docs" / "model.md"
    if not doc.exists():
        pytest.skip("docs/model.md is not part of this checkout")
    section = doc.read_text().split("## 11. Parameters of the synthetic world", 1)[1]
    section = section.split("\n## ", 1)[0]
    rows = re.findall(r"^\| `(\w+)` \| ([-+0-9.eE]+) \|", section, flags=re.MULTILINE)
    defaults = {k: float(v) for k, v in dataclasses.asdict(WorldParams()).items()}
    assert [name for name, _ in rows] == list(defaults)
    assert {name: float(value) for name, value in rows} == defaults


@pytest.mark.parametrize("bad", [
    {"n_events": 0}, {"n_events": 1_000_000}, {"n_events": 2.5}, {"window_hours": 0.0},
    {"sigma": -0.1}, {"x0_sd": -1.0}, {"base_rate": 0.0}, {"surge": -1.0},
    {"surge_hours": 0.0}, {"pi0": -0.01}, {"pi1": -0.01}, {"pi0": 0.9, "pi1": 0.1},
    {"pi_hours": 0.0}, {"kappa": 0.0}, {"tick": 0.0}, {"tick": 0.3}, {"tick": 0.03},
    {"mean_size": 0.5}, {"event_spacing_hours": 0.0}, {"base_rate": float("nan")},
    {"sigma": float("inf")}, {"t0": float("nan")}, {"window_hours": "12"},
])
def test_invalid_parameters_raise(bad):
    with pytest.raises(ValueError):
        WorldParams(**bad)


@pytest.mark.parametrize("ok", [
    {"pi1": 0.0}, {"sigma": 0.0}, {"surge": 0.0}, {"tick": 1e-9}, {"tick": 0.005},
    {"x0_sd": 0.0}, {"kappa": 1.0}, {"mean_size": 1.0}, {"pi0": 0.0, "pi1": 0.0},
])
def test_boundary_parameters_are_allowed(ok):
    generate_world(dataclasses.replace(WorldParams(n_events=3), **ok), SEED)


def test_invalid_seeds_raise():
    p = WorldParams(n_events=2)
    for bad in (-1, 1.5):
        with pytest.raises(ValueError):
            generate_world(p, bad)


# ---------------------------------------------------------------------- schema and sorting
def test_schema_dtypes_and_sorting(planted):
    w = planted.world
    tape, events, markets = w.tape, w.events, w.markets
    assert list(tape.columns) == TAPE_COLUMNS
    assert list(events.columns) == ["event", "release_ts", "series", "latent_x0"]
    assert list(markets.columns) == ["market", "event", "outcome", "open_ts", "close_ts"]
    for col in ("event", "market"):
        assert pd.api.types.is_string_dtype(tape[col])
    assert tape["ts"].dtype == np.float64 and tape["seq"].dtype == np.int64
    assert tape["price"].dtype == np.float64 and tape["size"].dtype == np.float64
    assert tape["side"].dtype == np.int8 and tape["latent_informed"].dtype == bool
    for col in ("latent_mid_pre", "latent_mid_post", "latent_pi"):
        assert tape[col].dtype == np.float64
    assert not tape.isna().to_numpy().any()
    assert set(np.unique(tape["side"])) == {-1, 1}
    assert (tape["market"] == tape["event"]).all()
    # sorted by (market, ts, seq); seq runs 0, 1, 2, ... within a market
    order = np.lexsort((tape["seq"].to_numpy(), tape["ts"].to_numpy(),
                        pd.factorize(tape["market"], sort=True)[0]))
    np.testing.assert_array_equal(order, np.arange(len(tape)))
    np.testing.assert_array_equal(tape["seq"].to_numpy(), tape.groupby("market").cumcount())
    assert tape["ts"].groupby(tape["market"]).apply(lambda s: s.is_monotonic_increasing).all()


def test_events_and_markets_frames(planted):
    w = planted.world
    p = w.params
    events, markets = w.events, w.markets
    assert events["event"].is_unique and len(events) == p.n_events
    assert events["event"].iloc[0] == "SYN-000000" and events["event"].iloc[-1] == "SYN-001999"
    assert (events["series"] == "SYN").all() and np.isfinite(events["latent_x0"]).all()
    expected = p.t0 + np.arange(p.n_events) * p.event_spacing_hours * 3600.0
    np.testing.assert_array_equal(events["release_ts"].to_numpy(), expected)
    assert (markets["market"] == events["event"]).all()
    assert (markets["event"] == events["event"]).all()
    assert markets["outcome"].isin([0, 1]).all()
    np.testing.assert_array_equal(markets["close_ts"], events["release_ts"])
    np.testing.assert_allclose(markets["open_ts"], events["release_ts"] - 3600 * p.window_hours)
    assert set(w.tape["event"]) == set(events["event"])  # every event has trades
    assert 0.3 < markets["outcome"].mean() < 0.7  # coarse sanity bound; calibration: see below


def test_trades_are_inside_the_window_and_before_the_release(planted):
    p, w = planted.params, planted.world
    assert planted.tau_h.min() > 0.0  # strictly before the release
    assert planted.tau_h.max() <= p.window_hours + 1e-9
    open_ts = w.markets["open_ts"].to_numpy()[planted.codes]
    assert (planted.ts >= open_ts - 1e-6).all()
    np.testing.assert_allclose(planted.pi, informed_share(planted.tau_h, p), rtol=0, atol=1e-9)


def test_prices_are_on_the_tick_grid_inside_the_bounds_and_below_the_ask(planted):
    p = planted.params
    price = planted.price
    assert price.min() >= p.tick - 1e-12 and price.max() <= 1.0 - p.tick + 1e-12
    assert np.all(np.abs(price / p.tick - np.round(price / p.tick)) < 1e-9)
    # the quotes behind the prints, recomputed from the latent columns
    bid, ask = _quotes(logit(planted.pre), information_step(planted.pi), p.kappa, p.tick)
    assert np.all(np.round((ask - bid) / p.tick) >= 1.0)  # bid < ask: at least one tick apart
    np.testing.assert_array_equal(price, np.where(planted.side > 0, ask, bid))
    assert np.all(bid >= p.tick) and np.all(ask <= 1 - p.tick)


@pytest.mark.parametrize("tick", [0.01, 0.005, 0.05, 0.25])
@pytest.mark.parametrize("kappa", [1.0, 1.5, 3.0])
def test_quotes_stay_ordered_on_the_grid_and_outside_the_fair_value(tick, kappa):
    x = np.linspace(-40.0, 40.0, 8001)
    for ell in (0.001, 0.2, 0.5, 2.0):
        ell_arr = np.full_like(x, ell)
        bid, ask = _quotes(x, ell_arr, kappa, tick)
        k_bid, k_ask = np.round(bid / tick), np.round(ask / tick)
        top = round(1 / tick) - 1
        assert np.all(k_ask - k_bid >= 1) and np.all(k_bid >= 1) and np.all(k_ask <= top)
        assert np.allclose(bid / tick, k_bid, atol=1e-9)
        assert np.allclose(ask / tick, k_ask, atol=1e-9)
        # rounded outward from the fair value, except where clipped at the edges of the grid
        assert np.all((ask >= sigmoid(x + kappa * ell_arr) - 1e-12) | (ask >= 1 - tick - 1e-12))
        assert np.all((bid <= sigmoid(x - kappa * ell_arr) + 1e-12) | (bid <= tick + 1e-12)
                      | (bid >= 1 - 2 * tick - 1e-12))
        assert np.all(np.diff(ask) >= -1e-12) and np.all(np.diff(bid) >= -1e-12)  # monotone in x


def test_quotes_at_the_edges_of_the_grid_by_hand():
    bid, ask = _quotes(np.array([-40.0, 0.0, 40.0]), np.full(3, 0.5), 1.5, 0.01)
    np.testing.assert_allclose(bid, [0.01, 0.32, 0.98])  # sigmoid(+-0.75) = 0.679 and 0.321
    np.testing.assert_allclose(ask, [0.02, 0.68, 0.99])  # lower edge: ask moved up one tick;
    # upper edge: both quotes clip to 0.99, so the bid moves down one tick


def test_sizes_are_geometric_with_the_requested_mean(planted, cluster):
    p, s = planted.params, planted.size
    assert s.min() >= 1.0 and np.all(s == np.round(s))
    m, se = cluster.mean_se(s, planted.codes)
    within(m, p.mean_size, se, "mean size")
    q = 1.0 / p.mean_size
    for k in (1, 6, 21):  # P(size >= k) = (1 - q)^(k - 1)
        share, se = cluster.mean_se((s >= k).astype(float), planted.codes)
        within(share, (1 - q) ** (k - 1), se, f"P(size >= {k})")


# ---------------------------------------------------------------------- determinism
def test_same_seed_gives_identical_frames_and_another_seed_does_not():
    p = WorldParams(n_events=12)
    a, b, c = generate_world(p, 3), generate_world(p, 3), generate_world(p, 4)
    for name in ("events", "markets", "tape"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name), check_exact=True)
    assert not a.tape.equals(c.tape) and not a.events["latent_x0"].equals(c.events["latent_x0"])


def test_events_share_no_random_state():
    """Raising n_events leaves the first events unchanged; no two events repeat each other."""
    small = generate_world(WorldParams(n_events=12), 3)
    big = generate_world(WorldParams(n_events=20), 3)
    pd.testing.assert_frame_equal(big.events.iloc[:12].reset_index(drop=True), small.events,
                                  check_exact=True)
    pd.testing.assert_frame_equal(big.markets.iloc[:12].reset_index(drop=True), small.markets,
                                  check_exact=True)
    first = big.tape[big.tape["event"].isin(small.events["event"])].reset_index(drop=True)
    pd.testing.assert_frame_equal(first, small.tape, check_exact=True)
    assert big.events["latent_x0"].nunique() == 20
    counts = big.tape.groupby("event").size()
    assert counts.nunique() > 10  # Poisson counts of 20 events do not coincide


def test_a_world_without_trades_still_has_its_frames():
    p = WorldParams(n_events=4, base_rate=1e-6, surge=0.0, window_hours=1.0)
    w = generate_world(p, 1)
    assert len(w.tape) == 0 and list(w.tape.columns) == TAPE_COLUMNS
    assert w.tape["side"].dtype == np.int8 and w.tape["ts"].dtype == np.float64
    assert len(w.events) == 4 and len(w.markets) == 4
    p = WorldParams(n_events=30, base_rate=0.7, surge=0.0, window_hours=1.0)
    w = generate_world(p, 1)  # Poisson(0.7): about half of the events have no trade at all
    n = w.tape.groupby("event").size().reindex(w.events["event"], fill_value=0)
    assert (n == 0).any() and (n > 0).any() and len(w.events) == 30


# ------------------------------------------------------------- the trade process (section 3)
def test_rate_and_share_functions_by_hand():
    p = WorldParams()
    assert trade_rate(0.0, p) == 210.0 and trade_rate(1.0, p) == pytest.approx(10 * (1 + 20 / np.e))
    assert trade_rate(50.0, p) == pytest.approx(10.0)
    assert informed_share(0.0, p) == pytest.approx(0.25)
    assert informed_share(0.5, p) == pytest.approx(0.05 + 0.2 / np.e)
    assert informed_share(100.0, p) == pytest.approx(0.05)
    tau = np.array([0.0, 0.3, 2.0])
    assert trade_rate(tau, p).shape == (3,) and informed_share(tau, p).shape == (3,)
    flat = WorldParams(pi1=0.0)
    np.testing.assert_array_equal(informed_share(tau, flat), np.full(3, 0.05))


@pytest.mark.parametrize("changes", [
    {}, {"surge": 0.0}, {"surge_hours": 0.3}, {"surge_hours": 2.5, "surge": 5.0},
    {"base_rate": 3.0, "surge": 40.0, "surge_hours": 0.25},
])
def test_cumulative_rate_and_its_inverse(changes):
    p = dataclasses.replace(WorldParams(), **changes)
    W = p.window_hours
    tau = np.linspace(0.0, W, 25)
    by_quadrature = np.array([integrate.quad(lambda s: float(trade_rate(s, p)), 0.0, t)[0]
                              for t in tau])
    np.testing.assert_allclose(_cum_rate(tau, p), by_quadrature, rtol=1e-12, atol=1e-12)
    # the derivative of the cumulative rate is the rate
    h = 1e-6
    slope = (_cum_rate(tau[1:] + h, p) - _cum_rate(tau[1:] - h, p)) / (2 * h)
    np.testing.assert_allclose(slope, trade_rate(tau[1:], p), rtol=1e-6)
    # the inverse: forward residual, endpoints and monotonicity
    total = float(_cum_rate(W, p))
    y = np.concatenate([[1e-15, 1e-9], np.linspace(0.01, 1.0, 400) * total])
    back = _invert_cum_rate(y, p)
    np.testing.assert_allclose(_cum_rate(back, p), y, rtol=1e-12, atol=1e-11)
    assert back[0] > 0.0 and back[-1] == pytest.approx(W, abs=1e-9) and np.all(np.diff(back) > 0.0)
    assert _invert_cum_rate(np.array([]), p).size == 0
    if p.surge > 0.0:  # an independent closed form through the Lambert W function
        a, s = p.surge, p.surge_hours
        w_arg = y / (p.base_rate * s)
        u = w_arg - a + np.real(lambertw(a * np.exp(a - w_arg)))
        np.testing.assert_allclose(back, s * u, rtol=1e-9, atol=1e-12)
    else:  # constant rate: Lambda is linear
        np.testing.assert_allclose(back, y / p.base_rate, rtol=1e-12)


def test_trade_counts_follow_the_poisson_process(planted):
    p = planted.params
    expected = integrate.quad(lambda s: float(trade_rate(s, p)), 0.0, p.window_hours)[0]
    counts = np.bincount(planted.codes, minlength=planted.n_events).astype(float)
    G = counts.size
    within(counts.mean(), expected, counts.std(ddof=1) / np.sqrt(G), "mean trades per event")
    # Poisson: variance equals the mean; standard error of a sample variance from its 4th moment
    s2 = counts.var(ddof=1)
    m4 = np.mean((counts - counts.mean()) ** 4)
    within(s2, expected, np.sqrt((m4 - s2**2) / G), "variance of trades per event")


@pytest.mark.parametrize("which", ["planted", "null"])
def test_rate_by_minutes_to_release_bucket(which, request, cluster):
    t = request.getfixturevalue(which)
    p = t.params
    for lo, hi in BUCKETS:
        mask = in_bucket(t, lo, hi)
        expected = integrate.quad(lambda s: float(trade_rate(s, p)), lo / 60.0,
                                  min(hi, 60.0 * p.window_hours) / 60.0)[0]
        per_event = np.bincount(t.codes[mask], minlength=t.n_events).astype(float)
        within(per_event.mean(), expected, per_event.std(ddof=1) / np.sqrt(t.n_events),
               f"trades per event in ({lo}, {hi}]")


@pytest.mark.parametrize("which", ["planted", "null"])
def test_informed_share_by_bucket(which, request, cluster):
    t = request.getfixturevalue(which)
    p = t.params
    informed = t.informed.astype(float)
    for lo, hi in BUCKETS:
        a, b = lo / 60.0, min(hi, 60.0 * p.window_hours) / 60.0
        weighted = integrate.quad(lambda s: float(trade_rate(s, p) * informed_share(s, p)), a, b)[0]
        total = integrate.quad(lambda s: float(trade_rate(s, p)), a, b)[0]
        share, se = cluster.mean_se(informed, t.codes, in_bucket(t, lo, hi))
        within(share, weighted / total, se, f"informed share in ({lo}, {hi}]")
    near, quiet = in_bucket(t, 0.0, 30.0), in_bucket(t, 360.0, np.inf)
    d, se = cluster.diff_se(informed, t.ones, t.codes, near, quiet)
    if which == "planted":  # power: the planted profile is far from flat
        assert d > 10 * se
    else:  # the null world has a flat share, exactly pi0 on every trade
        within(d, 0.0, se, "near minus quiet informed share")
        assert np.all(t.pi == t.params.pi0)


def test_informed_takers_trade_the_outcome_and_the_others_toss_a_coin(planted, cluster):
    t = planted
    assert np.all(t.side[t.informed] == 2 * t.y[t.informed] - 1)  # informed: buy iff y = 1
    coin, se = cluster.mean_se((t.side[~t.informed] > 0).astype(float), t.codes[~t.informed])
    within(coin, 0.5, se, "P(buy | not informed)")
    # P(buy | y, pi) = (1 + pi (2 y - 1)) / 2 for every trade (model.md section 3)
    resid = (t.side > 0) - 0.5 * (1.0 + t.pi * (2 * t.y - 1))
    m, se = cluster.mean_se(resid, t.codes)
    within(m, 0.0, se, "buy probability given the outcome")
    for y in (0.0, 1.0):
        m, se = cluster.mean_se(resid, t.codes, t.y == y)
        within(m, 0.0, se, f"buy probability given y = {y:g}")


# -------------------------------------------------------- the public belief (section 3)
def test_initial_belief_is_calibrated(planted):
    """Outcomes are Bernoulli(p0): the mean outcome in deciles of p0 equals the mean p0."""
    w = planted.world
    p0 = sigmoid(w.events["latent_x0"].to_numpy())
    y = w.markets["outcome"].to_numpy().astype(float)
    edges = np.quantile(p0, np.linspace(0.0, 1.0, 11))
    bins = np.clip(np.searchsorted(edges, p0, side="right") - 1, 0, 9)
    for b in range(10):
        d = (y - p0)[bins == b]
        within(d.mean(), 0.0, d.std(ddof=1) / np.sqrt(d.size), f"decile {b} of p0")
    within((y - p0).mean(), 0.0, (y - p0).std(ddof=1) / np.sqrt(y.size), "all events")
    assert np.std(p0) > 0.1  # the bins are not all the same


@pytest.mark.parametrize("which", ["planted", "null"])
def test_public_belief_is_a_martingale(which, request, cluster):
    """E[y - mid] = 0 over all trades and within each bucket, before and after the trade."""
    t = request.getfixturevalue(which)
    for name in ("pre", "post"):
        resid = t.y - getattr(t, name)
        m, se = cluster.mean_se(resid, t.codes)
        within(m, 0.0, se, f"mean(y - mid_{name}) over all trades")
        for lo, hi in BUCKETS:
            m, se = cluster.mean_se(resid, t.codes, in_bucket(t, lo, hi))
            within(m, 0.0, se, f"mean(y - mid_{name}) in ({lo}, {hi}]")


def calibration_scores(p, y):
    """Score statistics of a logistic recalibration ``y ~ Bernoulli(sigmoid(a + b logit p))`` at
    ``(a, b) = (0, 1)``, standardised by their exact null variance. One observation per event, and
    given the public information ``y`` is Bernoulli(p), so the variance is known (no sample
    variance, which collapses in a tail bin without a single surprise). Under calibration both
    are about N(0, 1): ``level`` detects a shifted belief, ``slope`` a belief that is too extreme
    (negative) or not extreme enough (positive)."""
    x = logit(p)
    v = p * (1.0 - p)
    return np.sum(y - p) / np.sqrt(np.sum(v)), np.sum((y - p) * x) / np.sqrt(np.sum(v * x**2))


@pytest.mark.parametrize("which", ["planted", "null"])
@pytest.mark.parametrize("hours_before", [6.0, 1.0, 10.0 / 60.0, 1.0 / 60.0])
def test_public_belief_is_calibrated_at_fixed_times(which, hours_before, request):
    """The mid after the last trade at least ``hours_before`` before the release, one per event
    (which trade is chosen depends on the trade times only), is the probability that y = 1."""
    t = request.getfixturevalue(which)
    eligible = t.tau_h >= hours_before
    next_is_eligible = np.r_[eligible[1:] & (t.codes[1:] == t.codes[:-1]), False]
    last = np.flatnonzero(eligible & ~next_is_eligible)
    assert last.size == t.n_events
    level, slope = calibration_scores(t.post[last], t.y[last])
    assert abs(level) <= Z and abs(slope) <= Z, f"level z = {level:.2f}, slope z = {slope:.2f}"
    if hours_before >= 1.0:
        # power: a belief that is 50% too extreme, or only half as extreme, is far from calibrated.
        # (Within 10 minutes of the release the planted world's beliefs are mostly saturated, few
        # events carry information about the slope, and the same stretch is only 2.8 sigma away.)
        x = logit(t.post[last])
        for stretch in (1.5, 0.5):
            _, off = calibration_scores(sigmoid(stretch * x), t.y[last])
            assert abs(off) > 2 * Z


def test_every_trade_moves_the_mid_in_the_direction_of_its_side(planted):
    move = planted.side * (planted.post - planted.pre)
    assert np.all(move >= 0.0) and np.mean(move > 0.0) > 0.99  # equal only where mids round to 1


@pytest.mark.parametrize("sigma", [0.0, 0.2])
def test_kappa_one_prints_the_zero_profit_price(sigma, cluster):
    """Glosten and Milgrom: with kappa = 1 and a vanishing tick the print is sigmoid(x_post) and the
    maker's gross edge mean(side * (price - y)) is zero. The news (sigma) does not matter."""
    p = WorldParams(n_events=300, sigma=sigma, kappa=1.0, tick=1e-9, pi0=0.1, pi1=0.0)
    t = trades_of(generate_world(p, SEED))
    np.testing.assert_allclose(t.price, t.post, rtol=0, atol=1e-8)
    edge = t.side * (t.price - t.y)
    m, se = cluster.mean_se(edge, t.codes)
    within(m, 0.0, se, "maker gross edge at kappa = 1")
    assert np.mean(edge**2) > 0.1 * m**2 + 1e-3  # a non-degenerate payoff, not a vanishing one


def test_kappa_above_one_leaves_the_maker_the_closed_form_margin(cluster):
    p = WorldParams(n_events=300, sigma=0.0, kappa=1.5, tick=1e-9, pi0=0.1, pi1=0.0)
    t = trades_of(generate_world(p, SEED))
    x_pre, ell = logit(t.pre), information_step(t.pi)
    closed = t.side * (sigmoid(x_pre + t.side * p.kappa * ell) - sigmoid(x_pre + t.side * ell))
    margin = t.side * (t.price - t.post)
    np.testing.assert_allclose(margin, closed, rtol=0, atol=1e-8)  # trade by trade
    assert margin.mean() > 0.0 and np.all(margin >= -1e-8)
    assert margin.mean() == pytest.approx(closed.mean(), abs=1e-8)
    # the realised gross edge side * (price - y) has the closed-form mean, within its standard error
    edge = t.side * (t.price - t.y)
    m, se = cluster.mean_se(edge - closed, t.codes)
    within(m, 0.0, se, "maker gross edge minus closed-form margin")
    assert closed.mean() > Z * se  # power: the margin is several standard errors from zero


def test_rounding_outward_adds_to_the_makers_edge(cluster):
    """With the 1 cent tick and kappa = 1 the maker earns the rounding: positive on average."""
    t = trades_of(generate_world(WorldParams(n_events=600, kappa=1.0, pi1=0.0), SEED))
    m, se = cluster.mean_se(t.side * (t.price - t.post), t.codes)
    assert m > Z * se


# ---------------------------------------------------------------------- price impact
def conditional_expected_impact(t):
    """E[side (post - pre)] given the pre-trade state (x_pre, pi): the buy probability is
    q = (1 + pi (2 p - 1)) / 2 and the mid moves to sigmoid(x +- ell)."""
    x, ell = logit(t.pre), information_step(t.pi)
    q = 0.5 * (1.0 + t.pi * (2.0 * t.pre - 1.0))
    return q * (sigmoid(x + ell) - t.pre) + (1.0 - q) * (t.pre - sigmoid(x - ell))


@pytest.mark.parametrize("which", ["planted", "null"])
def test_price_impact_has_its_closed_form_conditional_mean(which, request, cluster):
    t = request.getfixturevalue(which)
    impact = t.side * (t.post - t.pre)
    expected = conditional_expected_impact(t)
    for lo, hi in BUCKETS:
        m, se = cluster.mean_se(impact - expected, t.codes, in_bucket(t, lo, hi))
        within(m, 0.0, se, f"impact minus closed form in ({lo}, {hi}]")
    weight = t.pre * (1.0 - t.pre)  # and the same per unit of p (1 - p)
    for lo, hi in BUCKETS:
        mask = in_bucket(t, lo, hi)
        r, se = cluster.ratio_se(impact - expected, weight, t.codes, mask)
        within(r, 0.0, se, f"adjusted impact minus closed form in ({lo}, {hi}]")


def test_price_impact_per_unit_of_variance_is_larger_near_the_release_when_planted(
        planted, cluster):
    """Impact is about ell(pi) p (1 - p) (section 3). p (1 - p) shrinks as the order flow reveals
    the outcome, so the impact in USD is not larger near the release at the default parameters
    (this is tested); per unit of p (1 - p) it is, bucket by bucket, and it tracks ell(pi)."""
    t = planted
    impact, weight = t.side * (t.post - t.pre), t.pre * (1.0 - t.pre)
    near, quiet = in_bucket(t, 0.0, 30.0), in_bucket(t, 360.0, np.inf)
    d, se = cluster.diff_se(impact, weight, t.codes, near, quiet)
    assert d > 0.0 and d > 10 * se  # power
    adjusted = []
    for lo, hi in BUCKETS:
        mask = in_bucket(t, lo, hi)
        r, se = cluster.ratio_se(impact, weight, t.codes, mask)
        assert r > 0.0
        adjusted.append(r)
        ell_bar = information_step(t.pi[mask]).mean()
        assert abs(r - ell_bar) < 0.05 * ell_bar  # about ell(pi): second-order terms are small
    assert all(a > b for a, b in zip(adjusted[:4], adjusted[1:5], strict=True))
    raw_near, _ = cluster.mean_se(impact, t.codes, in_bucket(t, 0.0, 5.0))
    raw_quiet, _ = cluster.mean_se(impact, t.codes, quiet)
    assert raw_near < raw_quiet  # saturation: the USD impact falls near the release


def test_price_impact_per_unit_of_variance_is_equal_across_buckets_when_null(null, cluster):
    t = null
    impact, weight = t.side * (t.post - t.pre), t.pre * (1.0 - t.pre)
    quiet = in_bucket(t, 360.0, np.inf)
    ell = float(information_step(t.params.pi0))
    r_quiet, _ = cluster.ratio_se(impact, weight, t.codes, quiet)
    assert r_quiet == pytest.approx(ell, rel=1e-2)  # about ell(pi0) = 0.1001
    for lo, hi in BUCKETS[:-1]:
        d, se = cluster.diff_se(impact, weight, t.codes, in_bucket(t, lo, hi), quiet)
        within(d, 0.0, se, f"adjusted impact, ({lo}, {hi}] minus quiet")
    # without the adjustment the buckets differ, because p (1 - p) varies across them
    raw = [cluster.mean_se(impact, t.codes, in_bucket(t, lo, hi))[0] for lo, hi in BUCKETS]
    assert max(raw) > 1.5 * min(raw)


# ---------------------------------------------------------------------------- the news
def test_news_increments_between_trades(planted, cluster):
    """Between trades, logit(pre[i + 1]) - logit(post[i]) is the news increment: normal with mean
    (2 y - 1) sigma^2 dt / 2 and variance sigma^2 dt, whatever the sides of the trades. Pairs are
    kept when |x_post[i]| < 10: that conditions on the past only (the logit of a mid within 1e-4 of
    0 or 1 is not accurate), and the increment is independent of the past given y."""
    t, p, w = planted, planted.params, planted.world
    x_pre, x_post = logit(t.pre), logit(t.post)
    dt = t.tau_h[:-1] - t.tau_h[1:]
    keep = (t.codes[1:] == t.codes[:-1]) & (np.abs(x_post[:-1]) < 10.0) & (dt > 0.0)
    inc = (x_pre[1:] - x_post[:-1])[keep]
    dt, y, codes = dt[keep], t.y[:-1][keep], t.codes[:-1][keep]
    sides_before, sides_after = t.side[:-1][keep], t.side[1:][keep]
    assert keep.mean() > 0.95
    drift = (2 * y - 1) * p.sigma**2 / 2 * dt
    z = (inc - drift) / (p.sigma * np.sqrt(dt))
    for outcome in (1.0, 0.0):
        mask = y == outcome
        target = (2 * outcome - 1) * p.sigma**2 / 2
        rate, se = cluster.ratio_se(inc, dt, codes, mask)  # drift per hour
        within(rate, target, se, f"news drift per hour, y = {outcome:g}")
        assert abs(rate) > 5 * se  # power: far from "no drift", and the sign is right
        var, se = cluster.mean_se((inc - drift) ** 2 / dt, codes, mask)  # variance per hour
        within(var, p.sigma**2, se, f"news variance per hour, y = {outcome:g}")
    for name, sides in (("previous", sides_before), ("next", sides_after)):
        d, se = cluster.diff_se(z, np.ones(z.size), codes, sides > 0, sides < 0)
        within(d, 0.0, se, f"standardised increment, buy minus sell {name} trade")
    # the first increment, from the start of the window to the first trade, starts at x0
    first = np.flatnonzero(np.r_[True, t.codes[1:] != t.codes[:-1]])
    x0 = w.events["latent_x0"].to_numpy()[t.codes[first]]
    dt1 = (t.ts[first] - w.markets["open_ts"].to_numpy()[t.codes[first]]) / 3600.0
    z1 = (logit(t.pre[first]) - x0 - (2 * t.y[first] - 1) * p.sigma**2 / 2 * dt1) / (
        p.sigma * np.sqrt(dt1))
    G = z1.size
    within(z1.mean(), 0.0, z1.std(ddof=1) / np.sqrt(G), "first increment, mean")
    s2, m4 = z1.var(ddof=1), np.mean((z1 - z1.mean()) ** 4)
    within(s2, 1.0, np.sqrt((m4 - s2**2) / G), "first increment, variance")
