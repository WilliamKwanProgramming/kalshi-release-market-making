"""Markouts, buckets and the release effect (kalshi_mm.markout, docs/model.md section 5).

A hand-built tape fixes every column of ``trade_markouts`` to numbers worked out on paper; an
independent plain-loop implementation is compared with it on small synthetic worlds; look-ahead
guards change the data after a time and check that earlier rows stay the same; and the release
effect is compared with an explicit resampling loop for the same weight matrix.

Statistical tests on synthetic worlds use 200 events and the seed 20261113. The standard error that
a difference is compared with is the width of its 95% bootstrap interval divided by 3.92 (that is,
2 x 1.96), or, in the paired comparison, the standard deviation of the bootstrap replications.
"""

import numpy as np
import pandas as pd
import pytest

from kalshi_mm.markout import (
    BUCKET_EDGES_MIN,
    BUCKET_LABELS,
    bucket_index,
    bucket_means,
    per_event_sums,
    release_effect,
    scaled_adverse_selection,
    trade_markouts,
)
from kalshi_mm.stats import cluster_weights, percentile_interval, ratio_of_sums
from kalshi_mm.world import WorldParams, generate_world

NAN = np.nan
SEED = 20261113
NEW_COLUMNS = ["tau_min", "bucket", "mid_before", "mid_after", "eff_half_spread",
               "adverse_selection", "realized_spread", "settlement_markout"]


# ---------------------------------------------------------------------------------- buckets
def test_bucket_edges_are_right_closed():
    edges = np.array([5.0, 15.0, 30.0, 60.0, 180.0, 360.0])
    assert bucket_index(edges).tolist() == [0, 1, 2, 3, 4, 5]  # an edge belongs to the lower bucket
    just_above = np.nextafter(edges, np.inf)
    assert bucket_index(just_above).tolist() == [1, 2, 3, 4, 5, 6]
    assert bucket_index(np.array([1e-9, 2.5, 5.0, 5.0000001, 400.0, 1e9, np.inf])).tolist() == [
        0, 0, 0, 1, 6, 6, 6]
    assert bucket_index(np.array([0.0, -3.0, np.nan])).tolist() == [-1, -1, -1]  # no bucket
    assert bucket_index(5.0) == 0 and np.ndim(bucket_index(5.0)) == 0
    assert bucket_index(np.array([[1.0, 20.0], [100.0, 500.0]])).tolist() == [[0, 2], [4, 6]]


def test_bucket_labels_follow_the_edges():
    assert len(BUCKET_LABELS) == len(BUCKET_EDGES_MIN) - 1 == 7
    inner = [f"({lo:g},{hi:g}]" for lo, hi in zip(BUCKET_EDGES_MIN[:-2], BUCKET_EDGES_MIN[1:-1],
                                                    strict=True)]
    assert list(BUCKET_LABELS[:-1]) == inner and BUCKET_LABELS[-1] == ">360"
    assert BUCKET_EDGES_MIN[-1] == np.inf and BUCKET_EDGES_MIN[-2] == 360.0


# --------------------------------------------------------------------------- a hand-built tape
def hand_frames(outcome=None):
    """Three markets in two events. Market A (event EV-1, release 10000, outcome 1) has five prints
    before the release, one exactly at it and one after; B (EV-2, release 20000, outcome missing)
    has three sell-heavy prints; C (EV-1, outcome 0) has three prints."""
    rows = [
        ("EV-1", "MKT-A", 9000.0, 0, 0.55, 10.0, 1),
        ("EV-1", "MKT-A", 9100.0, 1, 0.45, 10.0, -1),
        ("EV-1", "MKT-A", 9200.0, 2, 0.60, 10.0, 1),
        ("EV-1", "MKT-A", 9650.0, 3, 0.50, 10.0, -1),
        ("EV-1", "MKT-A", 9800.0, 4, 0.58, 10.0, 1),
        ("EV-1", "MKT-A", 10000.0, 5, 0.99, 10.0, 1),  # at the release: dropped, used by no proxy
        ("EV-1", "MKT-A", 10100.0, 6, 0.01, 10.0, -1),  # after the release: dropped
        ("EV-2", "MKT-B", 19000.0, 0, 0.30, 10.0, -1),
        ("EV-2", "MKT-B", 19500.0, 1, 0.28, 10.0, -1),
        ("EV-2", "MKT-B", 19990.0, 2, 0.35, 10.0, 1),
        ("EV-1", "MKT-C", 9100.0, 0, 0.20, 10.0, 1),
        ("EV-1", "MKT-C", 9200.0, 1, 0.10, 10.0, -1),
        ("EV-1", "MKT-C", 9300.0, 2, 0.30, 10.0, 1),
    ]
    tape = pd.DataFrame(rows, columns=["event", "market", "ts", "seq", "price", "size", "side"])
    tape = tape.astype({"ts": "float64", "seq": "int64", "price": "float64", "size": "float64",
                        "side": "int8"})
    tape["latent_mid_pre"] = 0.5  # a latent column: it must be dropped and never read
    events = pd.DataFrame({"event": ["EV-1", "EV-2"], "release_ts": [10000.0, 20000.0],
                           "series": "TEST"})
    if outcome is None:
        outcome = pd.array([1, pd.NA, 0], dtype="Int64")
    markets = pd.DataFrame({"market": ["MKT-A", "MKT-B", "MKT-C"],
                            "event": ["EV-1", "EV-2", "EV-1"], "outcome": outcome,
                            "open_ts": [0.0, 0.0, 0.0], "close_ts": [10000.0, 20000.0, 10000.0]})
    return tape, events, markets


HAND_INDEX = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12]  # rows of the tape kept: before their release


def hand_expected():
    """Worked out by hand for horizon 300 s and staleness 3600 s. Rows 0-4 are market A, 7-9
    market B, 10-12 market C. Examples: row 2 (buy 0.60 at 9200) has mid_before = (0.55 + 0.45) / 2
    = 0.50 from the prints before it and mid_after = (0.60 + 0.45) / 2 = 0.525 at 9500; row 4 (buy
    0.58 at 9800) has its horizon cut at the release 10000: mid_after = (0.58 + 0.50) / 2 = 0.54,
    and the print 0.99 at exactly 10000 is not used."""
    return pd.DataFrame(
        {
            "tau_min": np.array([1000, 900, 800, 350, 200, 1000, 500, 10, 900, 800, 700]) / 60.0,
            "bucket": [2, 1, 1, 1, 0, 2, 1, 0, 1, 1, 1],
            "mid_before": [NAN, NAN, 0.50, 0.525, 0.55, NAN, NAN, NAN, NAN, NAN, 0.15],
            "mid_after": [0.525, 0.525, 0.525, 0.54, 0.54, NAN, NAN, 0.315, 0.20, 0.20, 0.20],
            "eff_half_spread": [NAN, NAN, 0.10, 0.025, 0.03, NAN, NAN, NAN, NAN, NAN, 0.15],
            "adverse_selection": [NAN, NAN, 0.025, -0.015, -0.01, NAN, NAN, NAN, NAN, NAN, 0.05],
            "realized_spread": [0.025, 0.075, 0.075, 0.04, 0.04, NAN, NAN, 0.035, 0.0, 0.10, 0.10],
            "settlement_markout": [-0.45, 0.55, -0.40, 0.50, -0.42, NAN, NAN, NAN, 0.20, -0.10,
                                   0.30],
        },
        index=HAND_INDEX,
    )


def assert_same(got, expected):
    """Equal frames (or series) with NaN in the same places and values equal to 1e-12."""
    check = (pd.testing.assert_series_equal if isinstance(got, pd.Series)
             else pd.testing.assert_frame_equal)
    check(got, expected, check_dtype=False, check_exact=False, check_names=False,
          atol=1e-12, rtol=0.0)


def test_hand_tape_every_column():
    tape, events, markets = hand_frames()
    got = trade_markouts(tape, events, markets)
    assert list(got.index) == HAND_INDEX  # the tape's index labels are kept; two rows are dropped
    assert list(got.columns) == ["event", "market", "ts", "seq", "price", "size", "side",
                                 *NEW_COLUMNS]
    assert_same(got[NEW_COLUMNS], hand_expected())
    pd.testing.assert_frame_equal(got[tape.columns[:7]], tape.loc[HAND_INDEX, tape.columns[:7]])


def test_hand_tape_identity_and_sign_conventions():
    got = trade_markouts(*hand_frames())
    parts = ["eff_half_spread", "adverse_selection", "realized_spread"]
    both = got[got[parts].notna().all(axis=1)]
    assert both.index.tolist() == [2, 3, 4, 12]  # the rows that have both proxies
    assert np.allclose(both["eff_half_spread"], both["realized_spread"] + both["adverse_selection"],
                       rtol=0.0, atol=1e-15)
    # row 2 is a taker buy and the mid rose afterwards (adverse selection > 0: the maker who sold
    # lost); row 3 is a taker sell and the mid rose afterwards (< 0: the maker who bought gained)
    assert got.loc[2, "adverse_selection"] > 0 and got.loc[3, "adverse_selection"] < 0


def test_hand_tape_with_a_shorter_horizon():
    got = trade_markouts(*hand_frames(), horizon_s=60.0)
    # rows 0, 8 have no sell print yet at t + 60; row 1: (0.55 + 0.45) / 2; row 3: (0.60 + 0.50) / 2
    # at 9710; row 11: (0.20 + 0.10) / 2 at 9260; rows 2, 4, 9, 10, 12 are as with 300 s
    assert_same(got["mid_after"], pd.Series(
        [NAN, 0.50, 0.525, 0.55, 0.54, NAN, NAN, 0.315, NAN, 0.15, 0.20], index=HAND_INDEX))
    assert_same(got["realized_spread"], pd.Series(
        [NAN, 0.05, 0.075, 0.05, 0.04, NAN, NAN, 0.035, NAN, 0.05, 0.10], index=HAND_INDEX))
    assert got.loc[3, "adverse_selection"] == pytest.approx(-0.025)
    assert_same(got["eff_half_spread"], hand_expected()["eff_half_spread"])  # no horizon in it


def test_hand_tape_with_a_shorter_staleness_limit():
    got = trade_markouts(*hand_frames(), stale_s=400.0)
    # row 3: the last buy before 9650 is 450 s old, row 4: 600 s old; row 2: the sell is exactly
    # 400 s old at 9500, which is still fresh; row 9: the sell at 19500 is 500 s old at the release
    assert np.isnan(got.loc[[3, 4], "mid_before"]).all()
    assert np.isnan(got.loc[[3, 4], "adverse_selection"]).all()
    assert got.loc[2, "mid_after"] == pytest.approx(0.525)
    assert got.loc[2, "adverse_selection"] == pytest.approx(0.025)
    assert np.isnan(got.loc[9, "mid_after"]) and np.isnan(got.loc[9, "realized_spread"])
    assert got.loc[4, "realized_spread"] == pytest.approx(0.04)  # mid_after is fresh


def test_the_outcome_may_be_missing_in_any_form():
    for outcome in ([1.0, np.nan, 0.0], [1, None, 0], pd.array([1, pd.NA, 0], dtype="Int64")):
        got = trade_markouts(*hand_frames(outcome=outcome))
        assert_same(got["settlement_markout"], hand_expected()["settlement_markout"])
    got = trade_markouts(*hand_frames(outcome=[0, 1, 1]))  # settlement is side (price - outcome)
    assert got.loc[0, "settlement_markout"] == pytest.approx(0.55)
    assert got.loc[7, "settlement_markout"] == pytest.approx(0.70)  # a sell at 0.30, outcome 1
    assert got.loc[12, "settlement_markout"] == pytest.approx(-0.70)  # a buy at 0.30, outcome 1


def test_prints_at_or_after_the_release_change_nothing_before_it():
    tape, events, markets = hand_frames()
    base = trade_markouts(tape, events, markets)
    altered = tape.copy()
    altered.loc[[5, 6], "price"] = [0.50, 0.50]
    altered.loc[[5, 6], "side"] = np.array([-1, 1], dtype=np.int8)
    altered.loc[6, "ts"] = 50000.0
    pd.testing.assert_frame_equal(trade_markouts(altered, events, markets), base)
    assert_same(base[NEW_COLUMNS], hand_expected())
    assert not (base["tau_min"] <= 0.0).any()  # nothing at or after the release is kept


def test_an_empty_tape_gives_an_empty_frame():
    tape, events, markets = hand_frames()
    got = trade_markouts(tape.iloc[0:0], events, markets)
    assert len(got) == 0
    assert list(got.columns) == ["event", "market", "ts", "seq", "price", "size", "side",
                                 *NEW_COLUMNS]
    assert len(bucket_means(got)) == 7


def test_invalid_input_is_refused():
    tape, events, markets = hand_frames()
    with pytest.raises(ValueError, match="horizon_s"):
        trade_markouts(tape, events, markets, horizon_s=0.0)
    with pytest.raises(ValueError, match="lacks"):
        trade_markouts(tape.drop(columns="seq"), events, markets)
    swapped = tape.iloc[[1, 0, *range(2, len(tape))]]  # two prints of market A out of order
    with pytest.raises(ValueError, match="sorted"):
        trade_markouts(swapped, events, markets)
    interleaved = tape.iloc[[0, 7, 1, *range(2, 7), *range(8, len(tape))]]  # B inside A
    with pytest.raises(ValueError, match="sorted"):
        trade_markouts(interleaved, events, markets)
    with pytest.raises(ValueError, match="not in events"):
        trade_markouts(tape, events.iloc[:1], markets)
    with pytest.raises(ValueError, match="not in markets"):
        trade_markouts(tape, events, markets.iloc[:2])
    with pytest.raises(ValueError, match="more than once"):
        trade_markouts(tape, pd.concat([events, events.iloc[:1]]), markets)
    with pytest.raises(ValueError, match="more than once"):
        trade_markouts(tape, events, pd.concat([markets, markets.iloc[:1]]))
    no_release = events.assign(release_ts=[10000.0, np.nan])
    with pytest.raises(ValueError, match="no release_ts"):
        trade_markouts(tape, no_release, markets)
    mixed = tape.copy()
    mixed.loc[2, "event"] = "EV-2"  # one market with two release times
    with pytest.raises(ValueError, match="different release times"):
        trade_markouts(mixed, events, markets)


# --------------------------------------------------------------- a plain-loop implementation
def slow_proxy(prints, accept, now, stale_s):
    """Mid proxy by scanning the prints: the last accepted print of each side, both fresh."""
    last = {}
    for ts, side, price in prints:
        if accept(ts):
            last[side] = (ts, price)
    if len(last) < 2 or any(now - ts > stale_s for ts, _ in last.values()):
        return NAN
    return (last[1][1] + last[-1][1]) / 2.0


def slow_markouts(tape, events, markets, horizon_s, stale_s):
    """The definitions of docs/model.md section 5 for one trade at a time."""
    release = dict(zip(events["event"], events["release_ts"], strict=True))
    outcome = {m: (NAN if pd.isna(y) else float(y))
               for m, y in zip(markets["market"], markets["outcome"], strict=True)}
    prints = {
        m: list(zip(g["ts"], g["side"].astype(int), g["price"], strict=True))
        for m, g in tape.groupby("market")
    }
    rows = {}
    for idx, ev, mk, ts, price, side in zip(tape.index, tape["event"], tape["market"], tape["ts"],
                                            tape["price"], tape["side"].astype(int), strict=True):
        T = release[ev]
        if ts >= T:
            continue
        before = slow_proxy(prints[mk], lambda t, ts=ts: t < ts, ts, stale_s)
        q = min(ts + horizon_s, T)
        after = slow_proxy(prints[mk], lambda t, q=q, T=T: t <= q and t < T, q, stale_s)
        rows[idx] = {
            "tau_min": (T - ts) / 60.0,
            "mid_before": before,
            "mid_after": after,
            "eff_half_spread": side * (price - before),
            "adverse_selection": side * (after - before),
            "realized_spread": side * (price - after),
            "settlement_markout": side * (price - outcome[mk]),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


@pytest.fixture(scope="module")
def small():
    return generate_world(WorldParams(n_events=6, window_hours=3.0), seed=5)


@pytest.mark.parametrize("horizon_s, stale_s", [(300.0, 3600.0), (60.0, 600.0), (900.0, 3600.0)])
def test_equals_a_plain_loop_on_a_synthetic_world(small, horizon_s, stale_s):
    w = small
    got = trade_markouts(w.tape, w.events, w.markets, horizon_s=horizon_s, stale_s=stale_s)
    slow = slow_markouts(w.tape, w.events, w.markets, horizon_s, stale_s)
    assert len(got) == len(w.tape) > 500  # all prints of a synthetic world are before the release
    assert list(got.index) == list(slow.index)
    columns = [c for c in NEW_COLUMNS if c != "bucket"]
    assert_same(got[columns], slow[columns])
    assert got[columns].isna().to_numpy().any()  # the comparison covers undefined proxies too
    assert got["adverse_selection"].notna().sum() > 300


def test_the_slow_implementation_matches_the_hand_tape():
    """Check the check: the plain loop reproduces the worked numbers of the hand tape."""
    tape, events, markets = hand_frames()
    slow = slow_markouts(tape, events, markets, 300.0, 3600.0)
    expected = hand_expected()
    columns = [c for c in NEW_COLUMNS if c != "bucket"]
    assert_same(slow[columns], expected[columns])


def test_identity_and_truncation_on_the_planted_world():
    world = generate_world(WorldParams(n_events=60), seed=SEED)
    m = trade_markouts(world.tape, world.events, world.markets)
    full = m[["eff_half_spread", "realized_spread", "adverse_selection"]].dropna()
    assert len(full) > 10000
    resid = full["eff_half_spread"] - full["realized_spread"] - full["adverse_selection"]
    assert np.abs(resid).max() < 1e-12
    # within the horizon of the release mid_after is the proxy at the release itself
    release = world.events.set_index("event")["release_ts"]
    at_release = m["ts"] + 300.0 >= release.reindex(m["event"]).to_numpy()
    assert at_release.sum() > 500
    last_proxy = {}
    for market, g in world.tape.groupby("market"):
        side, price = g["side"].to_numpy(), g["price"].to_numpy()
        last_proxy[market] = (price[side > 0][-1] + price[side < 0][-1]) / 2.0
    expected = m.loc[at_release, "market"].map(last_proxy)  # every print is within an hour
    pd.testing.assert_series_equal(m.loc[at_release, "mid_after"], expected,
                                   check_names=False, check_dtype=False, atol=1e-12, rtol=0.0)


# ----------------------------------------------------------------------- look-ahead guards
def altered_after(tape, market, cut, rng, strict):
    """Copy of the tape with random prices and sides for the prints of ``market`` after ``cut`` (at
    or after it when ``strict``), the times and order unchanged."""
    later = (tape["ts"] >= cut) if strict else (tape["ts"] > cut)
    later = later & (tape["market"] == market)
    out = tape.copy()
    k = int(later.sum())
    out.loc[later, "price"] = rng.integers(1, 100, k) / 100.0
    out.loc[later, "side"] = rng.choice(np.array([1, -1], dtype=np.int8), k)
    return out


def test_prints_after_t_plus_the_horizon_do_not_change_a_row(small):
    w, h = small, 300.0
    base = trade_markouts(w.tape, w.events, w.markets, horizon_s=h)
    rng = np.random.default_rng(1)
    for market, g in w.tape.groupby("market"):
        cut = float(g["ts"].quantile(0.6))
        altered = trade_markouts(altered_after(w.tape, market, cut, rng, strict=False), w.events,
                                 w.markets, horizon_s=h)
        safe = (base["market"] == market) & (base["ts"] + h <= cut)
        assert safe.sum() > 20
        assert_same(altered.loc[safe, NEW_COLUMNS], base.loc[safe, NEW_COLUMNS])
        later = (base["market"] == market) & (base["ts"] > cut)
        changed = ~np.isclose(altered.loc[later, "mid_after"], base.loc[later, "mid_after"],
                              equal_nan=True, atol=1e-12, rtol=0.0)
        assert changed.any()  # the guard has power: later prints do reach later rows


def test_prints_from_t_on_do_not_change_mid_before(small):
    w = small
    base = trade_markouts(w.tape, w.events, w.markets)
    rng = np.random.default_rng(2)
    for market, g in w.tape.groupby("market"):
        cut = float(g["ts"].quantile(0.5))
        altered = trade_markouts(altered_after(w.tape, market, cut, rng, strict=True), w.events,
                                 w.markets)
        safe = (base["market"] == market) & (base["ts"] <= cut)  # a print at the cut is not used
        assert safe.sum() > 20
        assert_same(altered.loc[safe, ["mid_before"]], base.loc[safe, ["mid_before"]])


def test_deleting_later_prints_does_not_change_earlier_rows(small):
    w, h = small, 300.0
    base = trade_markouts(w.tape, w.events, w.markets, horizon_s=h)
    for market, g in w.tape.groupby("market"):
        cut = float(g["ts"].quantile(0.5))
        kept = w.tape[~((w.tape["market"] == market) & (w.tape["ts"] > cut))]
        cut_rows = trade_markouts(kept, w.events, w.markets, horizon_s=h)
        assert len(cut_rows) == len(kept)
        safe = (base["market"] == market) & (base["ts"] + h <= cut)
        assert safe.sum() > 20
        assert_same(cut_rows.loc[base.index[safe], NEW_COLUMNS], base.loc[safe, NEW_COLUMNS])


def test_latent_columns_are_dropped_and_never_read(small):
    w = small
    base = trade_markouts(w.tape, w.events, w.markets)
    assert not any(c.startswith("latent_") for c in base.columns)
    latent = [c for c in w.tape.columns if c.startswith("latent_")]
    assert len(latent) == 4
    plain = trade_markouts(w.tape.drop(columns=latent), w.events, w.markets)
    pd.testing.assert_frame_equal(plain, base)
    rng = np.random.default_rng(3)
    scrambled = w.tape.copy()
    for c in latent:
        scrambled[c] = rng.permutation(scrambled[c].to_numpy())
    pd.testing.assert_frame_equal(trade_markouts(scrambled, w.events, w.markets), base)
    # the same holds for the latent column of the events frame
    no_x0 = w.events.drop(columns="latent_x0")
    pd.testing.assert_frame_equal(trade_markouts(w.tape, no_x0, w.markets), base)


# ---------------------------------------------------------------------------- per-event sums
def marks(rows):
    """A markouts-like frame from (event, tau_min, value) triples."""
    df = pd.DataFrame(rows, columns=["event", "tau_min", "adverse_selection"])
    df["bucket"] = bucket_index(df["tau_min"].to_numpy())
    return df


def test_per_event_sums_by_hand():
    df = marks([("e2", 10.0, 0.5), ("e1", 10.0, 0.1), ("e1", 20.0, 0.3), ("e1", 400.0, NAN),
                ("e3", 400.0, 0.7), ("e2", 400.0, 0.2)])
    got = per_event_sums(df, df["tau_min"].to_numpy() <= 30.0, "adverse_selection")
    assert list(got.columns) == ["event", "sum", "n"]
    assert got["event"].tolist() == ["e1", "e2", "e3"]  # sorted ids, an event without rows is kept
    assert got["sum"].tolist() == pytest.approx([0.4, 0.5, 0.0])
    assert got["n"].tolist() == [2, 1, 0]
    quiet = per_event_sums(df, df["tau_min"].to_numpy() > 360.0, "adverse_selection")
    assert quiet["event"].tolist() == ["e1", "e2", "e3"]  # the same events in the same order
    assert quiet["sum"].tolist() == pytest.approx([0.0, 0.2, 0.7])
    assert quiet["n"].tolist() == [0, 1, 1]  # the NaN of e1 is not counted
    with pytest.raises(ValueError, match="mask"):
        per_event_sums(df, np.ones(3, bool), "adverse_selection")


def test_per_event_sums_agree_with_a_loop_on_a_world(small):
    m = trade_markouts(small.tape, small.events, small.markets)
    mask = (m["tau_min"] > 10.0).to_numpy()
    got = per_event_sums(m, mask, "adverse_selection")
    for event, total, n in zip(got["event"], got["sum"], got["n"], strict=True):
        values = [v for e, v, keep in zip(m["event"], m["adverse_selection"], mask, strict=True)
                  if e == event and keep and not np.isnan(v)]
        assert n == len(values) and total == pytest.approx(sum(values), abs=1e-12)


# ---------------------------------------------------------------------------- bucket means
def test_bucket_means_by_hand():
    df = marks([
        ("e1", 10.0, 0.1), ("e1", 12.0, 0.5), ("e2", 14.0, 0.2),  # bucket (5,15]: n = 3
        ("e3", 100.0, 0.4),  # bucket (60,180]: one value, so no sd and no se from two clusters
        ("e1", 20.0, NAN),  # bucket (15,30]: only a NaN
        ("e2", 1.0, 0.3), ("e3", 2.0, 0.7),  # bucket (0,5]: two events, one value each
    ])
    out = bucket_means(df)
    assert list(out.columns) == ["bucket", "label", "n", "n_events", "mean", "sd", "se"]
    assert out["bucket"].tolist() == list(range(7)) and out["label"].tolist() == list(BUCKET_LABELS)
    assert out["n"].tolist() == [2, 3, 0, 0, 1, 0, 0]
    assert out["n_events"].tolist() == [2, 2, 0, 0, 1, 0, 0]
    b1 = out.loc[1]
    mean = (0.1 + 0.5 + 0.2) / 3.0
    assert b1["mean"] == pytest.approx(mean)
    assert b1["sd"] == pytest.approx(np.sqrt(((0.1 - mean) ** 2 + (0.5 - mean) ** 2
                                               + (0.2 - mean) ** 2) / 2.0))
    # events are the clusters: sums 0.6, 0.2, 0 (e3 has no value here but counts in G = 3)
    resid = [0.6 - 2 * mean, 0.2 - mean, 0.0]
    assert b1["se"] == pytest.approx(np.sqrt(3.0 / 2.0 * sum(r * r for r in resid)) / 3.0)
    b0 = out.loc[0]
    assert b0["mean"] == pytest.approx(0.5) and b0["sd"] == pytest.approx(np.sqrt(0.08))
    assert b0["se"] == pytest.approx(np.sqrt(3.0 / 2.0 * (0.2**2 + 0.2**2)) / 2.0)  # 0, -0.2, 0.2
    assert np.isnan(out.loc[4, "sd"]) and out.loc[4, "mean"] == pytest.approx(0.4)
    assert np.isnan(out.loc[2, "mean"]) and np.isnan(out.loc[2, "se"])  # a bucket with only NaN


def test_bucket_means_with_one_event_has_no_standard_error():
    df = marks([("only", 10.0, 0.1), ("only", 11.0, 0.3)])
    row = bucket_means(df).loc[1]
    assert row["mean"] == pytest.approx(0.2) and row["n"] == 2 and np.isnan(row["se"])


def test_bucket_means_on_a_world_use_the_trades_of_each_bucket(small):
    m = trade_markouts(small.tape, small.events, small.markets)
    out = bucket_means(m)
    for k in range(7):
        values = m.loc[m["bucket"] == k, "adverse_selection"].dropna().to_numpy()
        assert out.loc[k, "n"] == values.size
        if values.size:
            assert out.loc[k, "mean"] == pytest.approx(values.mean(), abs=1e-12)
    assert out.loc[out["n"] > 0, "n"].sum() == m["adverse_selection"].notna().sum()
    settlement = bucket_means(m, "settlement_markout")
    assert settlement["n"].sum() == m["settlement_markout"].notna().sum()


def test_bucket_se_covers_when_trades_of_an_event_are_dependent():
    """Check the checker: clustered values (event effect and trade noise, equal variances) in one
    bucket. The 95% interval with the clustered standard error covers the true mean in about 95% of
    800 samples of 100 events, within 4 Monte Carlo standard errors; the interval with sd / sqrt(n)
    does not (the design effect is about 5)."""
    rng = np.random.default_rng(23)
    R, G, mu = 800, 100, 0.3
    hit = hit_naive = 0
    for _ in range(R):
        n = 1 + rng.poisson(8.0, G)
        event = np.repeat(np.arange(G), n)
        values = mu + rng.normal(0.0, 1.0, G)[event] + rng.normal(0.0, 1.0, event.size)
        df = pd.DataFrame({"event": event.astype(str), "tau_min": 10.0,
                           "adverse_selection": values, "bucket": 1})
        row = bucket_means(df).loc[1]
        hit += abs(row["mean"] - mu) <= 1.96 * row["se"]
        hit_naive += abs(row["mean"] - mu) <= 1.96 * row["sd"] / np.sqrt(row["n"])
    mc_se = np.sqrt(0.95 * 0.05 / R)
    assert abs(hit / R - 0.95) < 4 * mc_se
    assert hit_naive / R < 0.7


# ---------------------------------------------------------------------------- release effect
def effect_frame():
    """Four events. Near (0,30]: e1 0.10 and 0.30, e2 0.20 (tau 5.0), e3 a NaN, e4 0.40 (tau 30.0);
    a value at tau 30.0000001 is not near. Quiet (360,inf): e1 0.01, e2 0.03 and 0.05; the value
    0.02 of e3 at tau 360.0 is not quiet; e3 has 0.50 at tau 100, in neither window."""
    return marks([
        ("e1", 10.0, 0.10), ("e1", 20.0, 0.30), ("e1", 400.0, 0.01),
        ("e2", 5.0, 0.20), ("e2", 361.0, 0.03), ("e2", 1000.0, 0.05),
        ("e3", 15.0, NAN), ("e3", 360.0, 0.02), ("e3", 100.0, 0.50),
        ("e4", 30.0, 0.40), ("e4", 30.0000001, 0.90),
    ])


def slow_release_effect(df, weights, near, quiet, min_quiet, level=0.95):
    """Explicit resampling: the events of each weight row are listed (``w[g]`` times event ``g``),
    their trades pooled and averaged, with plain loops over rows. Returns the usable differences
    and ratios, the number of undefined replications and the two quantile levels."""
    events = sorted(set(df["event"]))
    rows = list(zip(df["event"], df["tau_min"], df["adverse_selection"], strict=True))

    def pooled_mean(drawn, window):
        values = []
        for e in drawn:
            values += [x for ev, tau, x in rows
                       if ev == e and window[0] < tau <= window[1] and not np.isnan(x)]
        return sum(values) / len(values) if values else NAN

    d, ratio = [], []
    for w in weights:
        drawn = [events[g] for g in range(len(events)) for _ in range(int(w[g]))]
        a, b = pooled_mean(drawn, near), pooled_mean(drawn, quiet)
        d.append(a - b)
        ratio.append(a / b if b != 0 and not np.isnan(b) else NAN)
    d, ratio = np.array(d), np.array(ratio)
    with_ratio = pooled_mean(events, quiet) > min_quiet
    use = np.isfinite(d) & (np.isfinite(ratio) if with_ratio else True)
    return d[use], ratio[use], int((~use).sum()), ((1 - level) / 2, 1 - (1 - level) / 2)


def test_release_effect_by_hand():
    df = effect_frame()
    out = release_effect(df, B=200, seed=1)
    assert out["n_events"] == 4 and out["n_near"] == 4 and out["n_quiet"] == 3
    assert out["mean_near"] == pytest.approx((0.10 + 0.30 + 0.20 + 0.40) / 4.0)  # 0.25
    assert out["mean_quiet"] == pytest.approx((0.01 + 0.03 + 0.05) / 3.0)  # 0.03
    assert out["D"] == pytest.approx(0.25 - 0.03)
    assert out["ratio"] == pytest.approx(0.25 / 0.03)
    assert out["B"] == 200 and out["seed"] == 1
    assert set(out) == {"mean_near", "mean_quiet", "D", "D_ci", "ratio", "ratio_ci", "p_one_sided",
                        "n_events", "n_near", "n_quiet", "B", "B_undefined", "seed"}


def test_release_effect_equals_an_explicit_resampling_loop():
    df = effect_frame()
    B, seed = 400, 3
    out = release_effect(df, B=B, seed=seed)
    weights = cluster_weights(4, B, np.random.default_rng(seed))  # the documented draw
    d, ratio, undefined, (lo, hi) = slow_release_effect(df, weights, (0.0, 30.0), (360.0, np.inf),
                                                        0.002)
    assert out["B_undefined"] == undefined and d.size == B - undefined
    assert out["D_ci"] == pytest.approx(tuple(np.quantile(d, [lo, hi])), abs=1e-12)
    assert out["ratio_ci"] == pytest.approx(tuple(np.quantile(ratio, [lo, hi])), abs=1e-12)
    assert out["p_one_sided"] == pytest.approx(max(np.mean(d <= 0.0), 1.0 / (d.size + 1.0)))
    assert release_effect(df, B=B, seed=seed + 1)["D"] == out["D"]  # the point estimate has no seed


def test_release_effect_with_other_windows_and_columns():
    df = effect_frame().assign(other=lambda x: 2.0 * x["adverse_selection"] + 1.0)
    out = release_effect(df, near=(4.0, 20.0), quiet=(90.0, 500.0), column="other", B=100, seed=5)
    near = [2 * v + 1 for v in (0.10, 0.30, 0.20)]  # taus 10, 20, 5; NaN of e3 dropped
    quiet = [2 * v + 1 for v in (0.01, 0.50, 0.03)]  # taus 400, 100, 361 (e3's 360 is inside, too)
    quiet.append(2 * 0.02 + 1)
    assert out["mean_near"] == pytest.approx(np.mean(near)) and out["n_near"] == 3
    assert out["mean_quiet"] == pytest.approx(np.mean(quiet)) and out["n_quiet"] == 4


def test_ratio_is_reported_only_above_the_threshold():
    df = effect_frame()  # quiet mean 0.03
    below = release_effect(df, B=100, seed=1, min_quiet_for_ratio=0.0301)
    assert below["ratio"] is None and below["ratio_ci"] is None
    above = release_effect(df, B=100, seed=1, min_quiet_for_ratio=0.0299)
    assert above["ratio"] == pytest.approx(0.25 / 0.03) and len(above["ratio_ci"]) == 2
    assert above["D_ci"] == below["D_ci"]  # the ratio is finite whenever the quiet mean is defined
    negative = df.assign(adverse_selection=-df["adverse_selection"])  # quiet mean -0.03
    assert release_effect(negative, B=100, seed=1)["ratio"] is None


def test_p_value_is_never_zero_and_counts_replications_at_or_below_zero():
    big = release_effect(effect_frame().assign(adverse_selection=lambda x: x["adverse_selection"]
                                               .where(x["tau_min"] > 360.0, 5.0)), B=50, seed=1)
    usable = 50 - big["B_undefined"]
    assert big["p_one_sided"] == pytest.approx(1.0 / (usable + 1.0))  # none is at or below zero
    flipped = release_effect(effect_frame().assign(adverse_selection=lambda x: -x[
        "adverse_selection"]), B=50, seed=1)
    assert flipped["D"] < 0 and flipped["p_one_sided"] > 0.9


def test_undefined_replications_are_dropped_and_counted():
    """Three events; only e1 has quiet values. A replication that never draws e1 has no quiet mean:
    that happens with probability (2/3)^3 = 8/27."""
    df = marks([("e1", 10.0, 0.2), ("e1", 400.0, 0.1), ("e2", 10.0, 0.4), ("e3", 10.0, 0.1)])
    B = 2000
    out = release_effect(df, B=B, seed=9)
    p = 8.0 / 27.0
    assert abs(out["B_undefined"] - B * p) < 4 * np.sqrt(B * p * (1 - p))
    weights = cluster_weights(3, B, np.random.default_rng(9))
    assert out["B_undefined"] == int((weights[:, 0] == 0).sum())  # e1 is event 0
    d, _, undefined, (lo, hi) = slow_release_effect(
        df, weights, (0.0, 30.0), (360.0, np.inf), 0.002)
    assert undefined == out["B_undefined"]
    assert out["D_ci"] == pytest.approx(tuple(np.quantile(d, [lo, hi])), abs=1e-12)


def test_every_replication_undefined_is_an_error():
    df = marks([("e1", 10.0, 0.2), ("e2", 400.0, 0.1)])  # near only in e1, quiet only in e2
    # a seed whose single replication draws one of the two events twice and the other not at all
    seed = next(k for k in range(200)
                if cluster_weights(2, 1, np.random.default_rng(k)).max() == 2)
    with pytest.raises(ValueError, match="no bootstrap replication"):
        release_effect(df, B=1, seed=seed)


def test_release_effect_refuses_bad_arguments():
    df = effect_frame()
    with pytest.raises(ValueError, match="at least one value"):
        release_effect(df, near=(500.0, 600.0))
    with pytest.raises(ValueError, match="at least one value"):
        release_effect(df, quiet=(5000.0, np.inf))
    with pytest.raises(ValueError, match="B must"):
        release_effect(df, B=0)
    with pytest.raises(ValueError, match="near"):
        release_effect(df, near=(30.0, 0.0))
    with pytest.raises(ValueError, match="quiet"):
        release_effect(df, quiet=(360.0, 360.0))
    with pytest.raises(ValueError, match="level"):
        release_effect(df, level=1.5)


# ------------------------------------------------------------- planted and null worlds
def interval_se(out):
    """Standard error of D: the width of its 95% bootstrap interval divided by 2 x 1.96."""
    lo, hi = out["D_ci"]
    return (hi - lo) / (2.0 * 1.959964)


@pytest.fixture(scope="module")
def worlds():
    made = {}
    for name, extra in (("planted", {}), ("spec_null", {"pi1": 0.0}),
                        ("flat", {"pi1": 0.0, "surge": 0.0})):
        w = generate_world(WorldParams(n_events=200, **extra), SEED)
        made[name] = trade_markouts(w.tape, w.events, w.markets)
    return made


def test_release_effect_is_positive_in_the_planted_world(worlds):
    """Default parameters (informed share 0.25 at the release against 0.05 far from it), 200
    events: D is positive, its 95% interval excludes zero and the one-sided p-value is at its
    floor 1 / (B + 1) for B = 1000."""
    out = release_effect(worlds["planted"], B=1000, seed=7)
    assert out["D"] > 0 and out["D_ci"][0] > 0
    assert out["p_one_sided"] <= 0.01
    assert out["ratio"] is not None and out["ratio"] > 1.0 and out["ratio_ci"][0] > 1.0
    assert out["n_events"] == 200 and out["B_undefined"] == 0


def test_the_trade_rate_surge_alone_makes_the_difference_positive(worlds):
    """The null world of docs/model.md has no extra informed share (pi1 = 0) but keeps the surge
    of the trade rate at the release. A 5-minute markout counts every trade in the next 5 minutes,
    and there are about 20 times more of them near the release, so D is positive there as well:
    its interval excludes zero. The difference of D to the planted world is still positive, paired
    by event (the two worlds have the same trade times): that is what the planted profile adds."""
    null = release_effect(worlds["spec_null"], B=1000, seed=7)
    assert null["D_ci"][0] > 0  # D is not zero in this null world
    planted, spec_null = worlds["planted"], worlds["spec_null"]

    def sums(m):
        tau = m["tau_min"].to_numpy()
        return (per_event_sums(m, (tau > 0) & (tau <= 30), "adverse_selection"),
                per_event_sums(m, tau > 360, "adverse_selection"))

    w = cluster_weights(200, 1000, np.random.default_rng(7))
    d = []
    for m in (planted, spec_null):
        (near, quiet) = sums(m)
        d.append(ratio_of_sums(near["sum"], near["n"], w) - ratio_of_sums(quiet["sum"],
                                                                          quiet["n"], w))
    paired = d[0] - d[1]
    lo, hi = percentile_interval(paired)
    assert lo > 0.0 and np.mean(paired) > 2.0 * paired.std(ddof=1)  # more than 2 bootstrap sd


def test_a_world_with_no_release_profile_has_no_effect_once_the_price_level_is_removed(worlds):
    """No surge and a constant informed share (pi1 = 0, surge = 0). The raw difference is not
    zero here, because a markout in USD scales with p (1 - p) and that falls toward the release as
    the prices move away from one half. Dividing by mid_before (1 - mid_before), and keeping the
    near window (5, 30] away from the cut horizon, D is zero within its interval: 0 lies inside the
    95% interval and |D| is below 1.96 interval standard errors."""
    m = worlds["flat"]
    scaled = m.assign(scaled=m["adverse_selection"] / (m["mid_before"] * (1.0 - m["mid_before"])))
    out = release_effect(scaled, near=(5.0, 30.0), column="scaled", B=1000, seed=7)
    assert out["D_ci"][0] <= 0.0 <= out["D_ci"][1]
    assert abs(out["D"]) < 2.0 * interval_se(out)
    assert out["n_near"] > 500 and out["n_events"] == 200


# ------------------------------------------------------------------- scaled adverse selection
def test_scaled_adverse_selection_by_hand():
    """Divided by 4 m (1 - m) where mid_before is in [0.03, 0.97] (both ends included), else NaN;
    the index and the other columns are untouched."""
    frame = pd.DataFrame(
        {
            "mid_before": [0.5, 0.25, 0.03, 0.97, 0.0299, 0.9701, NAN, 0.9],
            "adverse_selection": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, NAN],
        },
        index=[10, 11, 12, 13, 14, 15, 16, 17],
    )
    out = scaled_adverse_selection(frame)
    assert list(out.index) == [10, 11, 12, 13, 14, 15, 16, 17]
    assert out.name == "adverse_selection_scaled"
    expected = [0.01, 0.01 / 0.75, 0.01 / (4 * 0.03 * 0.97), 0.01 / (4 * 0.03 * 0.97),
                NAN, NAN, NAN, NAN]
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=1e-12, equal_nan=True)
    assert list(frame.columns) == ["mid_before", "adverse_selection"]  # not modified


def test_scaled_adverse_selection_range_and_column_arguments():
    frame = pd.DataFrame({"mid_before": [0.1, 0.5, 0.9], "eff_half_spread": [0.04, 0.02, 0.04]})
    wide = scaled_adverse_selection(frame, min_mid=0.0, max_mid=1.0, column="eff_half_spread")
    np.testing.assert_allclose(wide.to_numpy(), [0.04 / 0.36, 0.02, 0.04 / 0.36], rtol=1e-12)
    narrow = scaled_adverse_selection(frame, min_mid=0.2, max_mid=0.8, column="eff_half_spread")
    assert narrow.isna().tolist() == [True, False, True] and narrow.iloc[1] == pytest.approx(0.02)
    for lo, hi in ((0.5, 0.5), (0.6, 0.4), (-0.1, 0.9), (0.1, 1.1)):
        with pytest.raises(ValueError, match="min_mid"):
            scaled_adverse_selection(frame, min_mid=lo, max_mid=hi, column="eff_half_spread")


def test_scaled_adverse_selection_on_a_world_undoes_the_scale(small):
    m = trade_markouts(small.tape, small.events, small.markets)
    scaled = scaled_adverse_selection(m)
    ok = scaled.notna()
    assert ok.sum() > 0.5 * len(m) and (scaled[~ok].isna()).all()
    back = scaled[ok] * 4.0 * m.loc[ok, "mid_before"] * (1.0 - m.loc[ok, "mid_before"])
    np.testing.assert_allclose(back, m.loc[ok, "adverse_selection"], rtol=1e-12, atol=1e-15)
    # NaN exactly where the markout is undefined or the price is outside the range
    outside = ~m["mid_before"].between(0.03, 0.97) | m["adverse_selection"].isna()
    assert (~ok).to_numpy().tolist() == outside.to_numpy().tolist()
