"""Dataset preparation and settings of kalshi_mm.protocol (docs/protocol.md, Data)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from kalshi_mm.protocol import (
    DEFAULT,
    Dataset,
    Settings,
    data_label,
    deviations,
    family,
    merge_releases,
    p3_recalibration,
    paired_mean,
    prepare,
    with_overrides,
)
from kalshi_mm.quoting import QuoteParams
from kalshi_mm.world import WorldParams, generate_world

RELEASE = 1_000_000.0
HOUR = 3600.0


def _tape(market: str, event: str, ts: list[float], side: list[int], release: float = RELEASE):
    """Trades of one market at the given times; the price is 0.55 for a buy and 0.45 for a sell."""
    side_arr = np.array(side, dtype=np.int8)
    return pd.DataFrame(
        {
            "event": event,
            "market": market,
            "ts": np.array(ts, dtype=float),
            "seq": np.arange(len(ts), dtype=np.int64),
            "price": np.where(side_arr > 0, 0.55, 0.45),
            "size": 5.0,
            "side": side_arr,
        }
    )


def _alternating(n: int, start: float = RELEASE - 6 * HOUR, step: float = 60.0):
    ts = [start + step * i for i in range(n)]
    return ts, [1 if i % 2 == 0 else -1 for i in range(n)]


def _frames(specs: list[dict]):
    """Tape, markets and events from specs ``{event, market, outcome, ts, side, release}``."""
    tapes, markets, events = [], [], {}
    for s in specs:
        release = s.get("release", RELEASE)
        tapes.append(_tape(s["market"], s["event"], s["ts"], s["side"], release))
        markets.append(
            {"market": s["market"], "event": s["event"], "outcome": s["outcome"],
             "open_ts": release - 13 * HOUR, "close_ts": release}
        )
        events[s["event"]] = {"event": s["event"], "release_ts": release,
                              "series": s.get("series", "SER")}
    return pd.concat(tapes, ignore_index=True), pd.DataFrame(markets), pd.DataFrame(events.values())


def test_merge_releases_joins_events_with_the_same_release_time():
    specs = [
        {"event": "KXCPI-A", "market": "m1", "outcome": 1, "series": "KXCPI",
         "ts": [RELEASE - 60.0], "side": [1]},
        {"event": "KXCPIYOY-A", "market": "m2", "outcome": 0, "series": "KXCPIYOY",
         "ts": [RELEASE - 30.0], "side": [-1]},
        {"event": "KXU3-B", "market": "m3", "outcome": 1, "series": "KXU3", "release": RELEASE + 99,
         "ts": [RELEASE + 10.0], "side": [1]},
    ]
    tape, markets, events = _frames(specs)
    t, m, e, series = merge_releases(tape, markets, events)
    assert list(e["event"]) == ["KXCPI-A+KXCPIYOY-A", "KXU3-B"]
    assert list(e["series"]) == ["KXCPI+KXCPIYOY", "KXU3"]
    assert list(e["release_ts"]) == [RELEASE, RELEASE + 99]
    assert list(m["event"]) == ["KXCPI-A+KXCPIYOY-A", "KXCPI-A+KXCPIYOY-A", "KXU3-B"]
    assert list(t["event"]) == ["KXCPI-A+KXCPIYOY-A", "KXCPI-A+KXCPIYOY-A", "KXU3-B"]
    assert series.to_dict() == {"m1": "KXCPI", "m2": "KXCPIYOY", "m3": "KXU3"}
    # the inputs are not modified
    assert list(events["event"]) == ["KXCPI-A", "KXCPIYOY-A", "KXU3-B"]
    assert list(tape["event"]) == ["KXCPI-A", "KXCPIYOY-A", "KXU3-B"]


def test_merge_releases_names_a_release_by_its_sorted_events_and_series():
    specs = [
        {"event": "B-2", "market": "m1", "outcome": 1, "series": "KXU3",
         "ts": [RELEASE - 60.0], "side": [1]},
        {"event": "A-1", "market": "m2", "outcome": 0, "series": "KXCPI",
         "ts": [RELEASE - 30.0], "side": [-1]},
        {"event": "C-3", "market": "m3", "outcome": 1, "series": "KXCPI",
         "ts": [RELEASE - 20.0], "side": [1]},
    ]
    tape, markets, events = _frames(specs)
    _, m, e, series = merge_releases(tape, markets, events)
    assert list(events["event"]) == ["B-2", "A-1", "C-3"]  # not in order on the way in
    assert list(e["event"]) == ["A-1+B-2+C-3"]
    assert list(e["series"]) == ["KXCPI+KXU3"]  # sorted, and a series named once
    assert list(m["market"]) == ["m1", "m2", "m3"] and set(m["event"]) == {"A-1+B-2+C-3"}
    assert series.to_dict() == {"m1": "KXU3", "m2": "KXCPI", "m3": "KXCPI"}


def test_merge_releases_leaves_distinct_release_times_alone():
    w = generate_world(WorldParams(n_events=5), seed=1)
    t, m, e, series = merge_releases(w.tape, w.markets, w.events)
    assert list(e["event"]) == list(w.events["event"])
    assert t["event"].equals(w.tape["event"]) and m["event"].equals(w.markets["event"])
    assert set(series) == {"SYN"}


def test_merge_releases_rejects_bad_input():
    tape, markets, events = _frames(
        [{"event": "A", "market": "m1", "outcome": 1, "ts": [RELEASE - 1.0], "side": [1]}]
    )
    with pytest.raises(ValueError, match="unique"):
        merge_releases(tape, markets, pd.concat([events, events], ignore_index=True))
    with pytest.raises(ValueError, match="release time"):
        merge_releases(tape, markets, events.assign(release_ts=np.nan))
    with pytest.raises(ValueError, match="not in the events frame"):
        merge_releases(tape, markets.assign(event="Z"), events)


def _kept(counts: dict[str, int], name: str) -> int:
    return counts[name]


def test_prepare_applies_each_rule_at_its_boundary():
    ts50, side50 = _alternating(50)
    ts49, side49 = _alternating(49)
    specs = [
        {"event": "E1", "market": "keep50", "outcome": 1, "ts": ts50, "side": side50},
        {"event": "E1", "market": "only49", "outcome": 1, "ts": ts49, "side": side49},
        {"event": "E1", "market": "no_outcome", "outcome": np.nan, "ts": ts50, "side": side50},
        {"event": "E1", "market": "only_buys", "outcome": 0, "ts": ts50, "side": [1] * 50},
        # 50 prints, but the 50th is at the release: only 49 are before it
        {"event": "E2", "market": "at_release", "outcome": 1, "release": RELEASE + 1000.0,
         "ts": [RELEASE + 1000.0 - 60.0 * (49 - i) for i in range(49)] + [RELEASE + 1000.0],
         "side": side49 + [1]},
        # the release has no market left after the rules
        {"event": "E3", "market": "lonely", "outcome": 0, "release": RELEASE + 2000.0,
         "ts": ts49, "side": side49},
    ]
    tape, markets, events = _frames(specs)
    ds = prepare(tape, markets, events)
    assert list(ds.markets["market"]) == ["keep50"]
    assert list(ds.events["event"]) == ["E1"]
    assert ds.counts == {
        "events": 3,
        "releases": 3,
        "markets": 6,
        "markets without a settled outcome": 1,
        "markets with fewer than 50 trades": 3,
        "markets without a mid proxy in the quoting window": 1,
        "markets kept": 1,
        "releases kept": 1,
        "trades kept": 50,
    }
    assert set(ds.tape["market"]) == {"keep50"} and len(ds.tape) == 50
    assert list(ds.market_series.index) == ["keep50"]


def test_prepare_drops_trades_at_or_after_the_release():
    ts, side = _alternating(50)
    late = _tape("keep", "E1", ts + [RELEASE, RELEASE + 5.0], side + [1, -1])
    markets = pd.DataFrame(
        {"market": ["keep"], "event": ["E1"], "outcome": [1], "open_ts": [0.0],
         "close_ts": [RELEASE]}
    )
    events = pd.DataFrame({"event": ["E1"], "release_ts": [RELEASE], "series": ["S"]})
    ds = prepare(late, markets, events)
    assert len(ds.tape) == 50 and ds.tape["ts"].max() < RELEASE


def test_prepare_uses_the_window_of_the_quote_parameters():
    # both sides print 20 hours before the release and one more buy 6 hours before it: inside a
    # 12-hour window the proxy is never defined (the old prints are stale), inside a 24-hour
    # window it is defined at the last trade of the early block
    early_ts = [RELEASE - 20 * HOUR + 10.0 * i for i in range(50)]
    early_side = [1 if i % 2 == 0 else -1 for i in range(50)]
    tape, markets, events = _frames(
        [{"event": "E1", "market": "m", "outcome": 1, "ts": early_ts + [RELEASE - 6 * HOUR],
          "side": early_side + [1]}]
    )
    assert len(prepare(tape, markets, events).markets) == 0
    wide = Settings(quote=QuoteParams(window_hours=24.0))
    assert len(prepare(tape, markets, events, wide).markets) == 1


def test_prepare_uses_the_minimum_number_of_trades_of_the_settings():
    ts, side = _alternating(30)
    tape, markets, events = _frames(
        [{"event": "E1", "market": "m", "outcome": 1, "ts": ts, "side": side}]
    )
    assert len(prepare(tape, markets, events).markets) == 0
    assert len(prepare(tape, markets, events, Settings(min_trades=30)).markets) == 1


def test_prepare_does_not_depend_on_the_row_order():
    w = generate_world(WorldParams(n_events=12), seed=2)
    ref = prepare(w.tape, w.markets, w.events)
    shuffled = prepare(
        w.tape.sample(frac=1.0, random_state=1).reset_index(drop=True),
        w.markets.sample(frac=1.0, random_state=2).reset_index(drop=True),
        w.events.sample(frac=1.0, random_state=3).reset_index(drop=True),
    )
    assert ref.counts == shuffled.counts
    assert ref.tape.equals(shuffled.tape)
    assert ref.markets.equals(shuffled.markets) and ref.events.equals(shuffled.events)
    assert ref.market_series.equals(shuffled.market_series)


def test_prepare_never_reads_the_latent_columns():
    w = generate_world(WorldParams(n_events=8), seed=4)
    latent = [c for c in w.tape.columns if c.startswith("latent_")]
    assert latent
    full = prepare(w.tape, w.markets, w.events)
    stripped = prepare(w.tape.drop(columns=latent), w.markets, w.events)
    assert full.counts == stripped.counts
    assert full.tape.drop(columns=latent).equals(stripped.tape)


def test_deviations_lists_every_changed_setting():
    assert deviations(DEFAULT) == []
    changed = Settings(n_boot=500, quote=QuoteParams(gamma=0.01), near=(0.0, 15.0))
    assert deviations(changed) == [
        "n_boot: 500 (protocol: 10000)",
        "near: (0.0, 15.0) (protocol: (0.0, 30.0))",
        "quote.gamma: 0.01 (protocol: 0.002)",
    ]


def test_the_default_settings_are_those_of_the_protocol():
    assert (DEFAULT.n_boot, DEFAULT.seed, DEFAULT.level) == (10_000, 20261113, 0.95)
    assert (DEFAULT.near, DEFAULT.quiet) == ((0.0, 30.0), (360.0, np.inf))
    assert (DEFAULT.min_trades, DEFAULT.min_releases) == (50, 100)
    q = DEFAULT.quote
    assert (q.gamma, q.k, q.size, q.max_position) == (0.002, 100.0, 10.0, 100.0)
    assert (q.refresh_s, q.latency_s) == (60.0, 1.0)
    assert (q.maker_fee_coeff, q.fill_rule) == (0.0175, "through")
    assert (q.min_mid, q.max_mid, q.window_hours, q.min_history_events) == (0.03, 0.97, 12.0, 20)
    assert (DEFAULT.recal_hours, DEFAULT.recal_min_releases) == (24.0, 50)
    assert DEFAULT.recal_refit_every == 10


def test_data_label_names_the_source_and_refuses_unknown_ones():
    assert data_label("synthetic") == "synthetic data"
    assert data_label("kalshi-public-api") == "real data"
    with pytest.raises(ValueError, match="unknown data source"):
        data_label("csv from a friend")


def test_with_overrides_changes_the_named_settings_and_nothing_else():
    changed = with_overrides(DEFAULT, ["n_boot=500", "quote.gamma=0.01", "recal_hours=12",
                                       "quote.fill_rule=touch"])
    assert (changed.n_boot, changed.recal_hours) == (500, 12.0)
    assert (changed.quote.gamma, changed.quote.fill_rule) == (0.01, "touch")
    assert isinstance(changed.n_boot, int) and isinstance(changed.recal_hours, float)
    assert deviations(changed) == [  # in the order of the fields
        "n_boot: 500 (protocol: 10000)",
        "quote.gamma: 0.01 (protocol: 0.002)",
        "quote.fill_rule: touch (protocol: through)",
        "recal_hours: 12.0 (protocol: 24.0)",
    ]
    assert with_overrides(DEFAULT, []) == DEFAULT and DEFAULT.n_boot == 10_000  # not modified


def test_with_overrides_refuses_what_it_cannot_set():
    for bad, message in (("n_boot", "name=value"), ("nonsense=1", "unknown setting"),
                         ("quote.nonsense=1", "unknown setting"), ("quote=1", "unknown setting"),
                         ("near=1", "cannot be set"), ("n_boot=lots", "bad value"),
                         ("n_boot=1.5", "bad value")):
        with pytest.raises(ValueError, match=message):
            with_overrides(DEFAULT, [bad])


def _separated_dataset(n_releases: int) -> Dataset:
    """One market per release; the proxy 1 hour before the release is 0.9 if the market settles
    YES and 0.1 if not, so the prices separate the outcomes perfectly."""
    rows, markets, events = [], [], []
    for i in range(n_releases):
        release = 10_000_000.0 + i * 7 * 24 * HOUR
        outcome = i % 2
        price = 0.9 if outcome else 0.1
        events.append({"event": f"E{i:03d}", "release_ts": release, "series": "S"})
        markets.append({"market": f"M{i:03d}", "event": f"E{i:03d}", "outcome": outcome,
                        "open_ts": 0.0, "close_ts": release})
        for seq, (offset, side) in enumerate(((2.0, 1), (1.9, -1))):
            rows.append({"event": f"E{i:03d}", "market": f"M{i:03d}", "ts": release - offset * HOUR,
                         "seq": seq, "price": price, "size": 1.0, "side": side})
    tape, mk, ev = pd.DataFrame(rows), pd.DataFrame(markets), pd.DataFrame(events)
    series = pd.Series("S", index=mk["market"], name="series")
    return Dataset(tape, mk, ev, series, {})


def test_p3_that_cannot_be_fitted_is_reported_not_raised():
    """A recalibration whose logistic fit does not converge (the prices separate the outcomes)
    leaves P3 without a value and with the reason; it does not stop the other analyses."""
    settings = with_overrides(DEFAULT, ["recal_hours=1", "recal_min_releases=12",
                                        "recal_refit_every=4", "n_boot=100"])
    out = p3_recalibration(_separated_dataset(20), settings)
    assert out["primary"] is None and out["walk"] is None and out["by_series"] == {}
    assert "did not converge" in out["error"]
    assert out["n_markets"] == 20 and out["n_without_price"] == 0


def test_p3_with_too_few_releases_has_no_value_and_no_error():
    settings = with_overrides(DEFAULT, ["recal_hours=1", "recal_min_releases=30", "n_boot=100"])
    out = p3_recalibration(_separated_dataset(20), settings)
    assert out["primary"] is None and out["error"] is None
    assert out["n_releases_with_price"] == 20


def test_p3_needs_more_releases_than_the_minimum_for_a_first_fit():
    data = _separated_dataset(20)
    exactly = with_overrides(DEFAULT, ["recal_hours=1", "recal_min_releases=20", "n_boot=100"])
    out = p3_recalibration(data, exactly)
    assert out["primary"] is None and out["walk"] is None and out["error"] is None
    one_more = with_overrides(DEFAULT, ["recal_hours=1", "recal_min_releases=19", "n_boot=100"])
    assert "did not converge" in p3_recalibration(data, one_more)["error"]


def _tests(p1, p2, p3):
    """The three ``primary`` entries of P1 to P3 with the given one-sided p-values (None: none)."""

    def wrap(p):
        return {"primary": None if p is None else {"p_one_sided": p}}

    return wrap(p1), wrap(p2), wrap(p3)


def test_family_adjusts_by_holm_and_calls_a_test_supported_below_alpha():
    out = family(*_tests(0.001, 0.02, 0.2), 150, DEFAULT)
    # Holm: 3 * 0.001, 2 * 0.02, 1 * 0.2
    assert [out[k]["p_holm"] for k in ("P1", "P2", "P3")] == pytest.approx([0.003, 0.04, 0.2])
    assert [out[k]["verdict"] for k in ("P1", "P2", "P3")] == [
        "supported", "supported", "not supported"]
    assert [out[k]["p_one_sided"] for k in ("P1", "P2", "P3")] == [0.001, 0.02, 0.2]


def test_family_needs_the_adjusted_p_value_strictly_below_alpha():
    strict = dataclasses.replace(DEFAULT, alpha=0.375)  # 3 * 0.125 = 0.375 exactly
    out = family(*_tests(0.125, 0.5, 0.5), 150, strict)
    assert out["P1"]["p_holm"] == 0.375 and out["P1"]["verdict"] == "not supported"
    looser = dataclasses.replace(DEFAULT, alpha=0.376)
    assert family(*_tests(0.125, 0.5, 0.5), 150, looser)["P1"]["verdict"] == "supported"


def test_family_labels_p1_and_p2_exploratory_below_the_minimum_number_of_releases():
    below = family(*_tests(0.001, 0.001, 0.001), DEFAULT.min_releases - 1, DEFAULT)
    assert [below[k]["verdict"] for k in ("P1", "P2", "P3")] == [
        "exploratory", "exploratory", "supported"]
    enough = family(*_tests(0.001, 0.001, 0.001), DEFAULT.min_releases, DEFAULT)
    assert [enough[k]["verdict"] for k in ("P1", "P2", "P3")] == ["supported"] * 3
    fewer = dataclasses.replace(DEFAULT, min_releases=20)
    assert family(*_tests(0.001, 0.001, 0.001), 20, fewer)["P1"]["verdict"] == "supported"
    assert family(*_tests(0.001, 0.001, 0.001), 19, fewer)["P1"]["verdict"] == "exploratory"


def test_family_counts_a_test_that_is_not_computed_as_p_equal_to_one():
    out = family(*_tests(0.01, None, 0.02), 150, DEFAULT)
    assert out["P2"] == {"p_one_sided": None, "p_holm": 1.0, "verdict": "not computed"}
    # Holm over (0.01, 0.02, 1): 3 * 0.01, 2 * 0.02
    assert out["P1"]["p_holm"] == pytest.approx(0.03) and out["P3"]["p_holm"] == pytest.approx(0.04)
    assert out["P1"]["verdict"] == out["P3"]["verdict"] == "supported"
    none = family(*_tests(0.5, 0.5, None), 10, DEFAULT)
    assert none["P3"]["verdict"] == "not computed" and none["P1"]["verdict"] == "exploratory"


def test_paired_mean_of_a_constant_difference_is_that_constant_with_the_smallest_p():
    settings = dataclasses.replace(DEFAULT, n_boot=500, seed=7)
    out = paired_mean(np.array([2.0, 2.0, 2.0]), settings)
    assert out["mean"] == 2.0 and out["ci"] == (2.0, 2.0)
    assert out["p_one_sided"] == pytest.approx(1.0 / 501.0)
    assert (out["n_releases"], out["B"], out["seed"]) == (3, 500, 7)


def test_paired_mean_counts_a_replication_equal_to_zero_as_at_or_below_zero():
    settings = dataclasses.replace(DEFAULT, n_boot=500, seed=7)
    assert paired_mean(np.zeros(4), settings)["p_one_sided"] == 1.0
    assert paired_mean(-np.ones(4), settings)["p_one_sided"] == 1.0
    assert paired_mean(np.array([]), settings) is None


def test_paired_mean_resamples_releases_with_the_seed_of_the_settings():
    diff = np.array([-3.0, 1.0, 4.0, 5.0, -1.0, 2.0, 7.0, 0.5])
    a = dataclasses.replace(DEFAULT, n_boot=2000, seed=11)
    again = paired_mean(diff, a)
    assert again == paired_mean(diff, a)
    other = paired_mean(diff, dataclasses.replace(a, seed=12))
    assert other["mean"] == again["mean"] and other["ci"] != again["ci"]
    lo, hi = again["ci"]
    assert lo < diff.mean() < hi
    wide = paired_mean(diff, dataclasses.replace(a, level=0.99))["ci"]
    assert wide[0] < lo and wide[1] > hi


def _recalibration_world():
    """Thirteen releases, one market each, a day and a bit apart.

    Market ``i`` has a buy and a sell print a little before 24 hours ahead of its release, around
    ``PRICES[i]``, and a pair of prints about 33 minutes ahead of the release, around ``0.9 -
    PRICES[i]`` (the proxy there, if the price is taken too late). The last market has the second
    pair only, so it has no proxy 24 hours ahead.
    """
    prices = [0.30, 0.50, 0.40, 0.60, 0.35, 0.55, 0.45, 0.65, 0.42, 0.52, 0.48, 0.58]
    outcomes = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1]
    tapes, markets, events = [], [], []
    for i, outcome in enumerate(outcomes):
        release = RELEASE + i * 100_000.0
        late = 0.9 - prices[i] if i < len(prices) else 0.70
        rows = [(release - 100.0, 1, 0.90), (release - 50.0, -1, 0.80),
                (release - 2000.0, 1, late + 0.02), (release - 1900.0, -1, late - 0.02)]
        if i < len(prices):
            rows += [(release - 24 * HOUR - 1800.0, 1, prices[i] + 0.02),
                     (release - 24 * HOUR - 900.0, -1, prices[i] - 0.02)]
        rows.sort()
        tapes.append(pd.DataFrame(
            {"event": f"E{i}", "market": f"M{i}", "ts": [r[0] for r in rows],
             "seq": np.arange(len(rows)), "price": [r[2] for r in rows], "size": 5.0,
             "side": np.array([r[1] for r in rows], dtype=np.int8)}))
        markets.append({"market": f"M{i}", "event": f"E{i}", "outcome": outcome,
                        "open_ts": release - 40 * HOUR, "close_ts": release})
        events.append({"event": f"E{i}", "release_ts": release, "series": "SER"})
    return pd.concat(tapes, ignore_index=True), pd.DataFrame(markets), pd.DataFrame(events)


def test_p3_prices_each_market_at_the_proxy_recal_hours_before_its_release():
    settings = dataclasses.replace(
        DEFAULT, min_trades=4, recal_min_releases=10, recal_refit_every=1, n_boot=50)
    ds = prepare(*_recalibration_world(), settings)
    assert ds.counts["releases kept"] == 13
    out = p3_recalibration(ds, settings)
    assert out["error"] is None and out["walk"] is not None
    assert (out["n_markets"], out["n_without_price"], out["n_releases_with_price"]) == (13, 1, 12)
    walk = out["walk"].sort_values("release_ts")
    assert walk["price"].to_numpy() == pytest.approx([0.48, 0.58])
    assert walk["outcome"].tolist() == [0, 1]
    assert walk["n_fit"].tolist() == [10, 11]
    shorter = dataclasses.replace(settings, recal_hours=0.5)
    other = p3_recalibration(prepare(*_recalibration_world(), shorter), shorter)
    assert other["error"] is None and other["n_without_price"] == 0
    late = other["walk"].sort_values("release_ts")
    assert late["price"].to_numpy() == pytest.approx([0.42, 0.32, 0.70])
    assert late["n_fit"].tolist() == [10, 11, 12]
