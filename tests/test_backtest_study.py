"""The backtest study: its statistics against hand and independent computations, and its files."""

import importlib.util
import itertools
import json
import math
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from kalshi_mm import protocol
from kalshi_mm.backtest import walk_forward
from kalshi_mm.experiments import backtest_study as bs
from kalshi_mm.experiments.common import REPO, Manifest
from kalshi_mm.markout import BUCKET_LABELS, trade_markouts
from kalshi_mm.quoting import POLICIES, QuoteParams, decision_minutes, estimate_premiums
from kalshi_mm.stats import cluster_weights
from kalshi_mm.world import WorldParams, generate_world

NAN = float("nan")
NULL, PLANTED = bs.NULL_WORLD, bs.PLANTED_WORLD


def load_make_tables():
    """The script that pastes tables into README.md, loaded by path (it is not a package)."""
    path = REPO / "scripts" / "make_tables.py"
    spec = importlib.util.spec_from_file_location("make_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_events(pnl, evaluated=None):
    """An event table (the output of ``event_table``) from ``{policy: [pnl per event]}``."""
    n = len(next(iter(pnl.values())))
    evaluated = [True] * n if evaluated is None else evaluated
    frames = []
    for policy, values in pnl.items():
        frame = pd.DataFrame({
            "event": [f"E{i:03d}" for i in range(n)],
            "policy": policy,
            "pnl": values,
            "gross": np.asarray(values) + 1.0,
            "fees": 1.0,
            "fills": 10,
            "contracts": 100.0,
            "inventory_end": 0.0,
            "marked_pnl": values,
            "quotes_live_share": 0.5,
            "evaluated": evaluated,
            "release_ts": 1000.0 + 100.0 * np.arange(n),
        })
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------------ the design
def test_the_default_design_is_the_preregistered_one():
    config = bs.StudyConfig()
    assert config.seeds == tuple(range(1, 11))
    assert config.world == WorldParams() and config.world.n_events == 200
    assert config.quote == QuoteParams() and config.quote.min_history_events == 20
    assert config.worlds() == ((NULL, 0.0), (PLANTED, 0.20))
    assert config.boot_reps == 10_000 and bs.LEVEL == 0.95 and bs.SEED == 20261113
    assert config.sweep_pi1 == (0.0, 0.05, 0.10, 0.20, 0.30)
    assert config.sweep_fees == (0.0175, 0.0) and config.sweep_seed == 1
    assert max(config.event_counts) == config.world.n_events - config.quote.min_history_events
    assert bs.PRIMARY == ("release_aware", "flat_matched") == bs.PAIRS[0] == protocol.PRIMARY_PAIR
    assert bs.SECONDARY == ("release_aware", "pooled_premium") == bs.PAIRS[1]


def test_the_ten_pairs_are_a_minus_b_with_the_more_elaborate_policy_first():
    assert bs.PAIRS == (
        ("release_aware", "flat_matched"), ("release_aware", "pooled_premium"),
        ("flat_matched", "pooled_premium"), ("release_aware", "always_on"),
        ("release_aware", "pull_only"), ("pooled_premium", "always_on"),
        ("pooled_premium", "pull_only"), ("flat_matched", "always_on"),
        ("flat_matched", "pull_only"), ("pull_only", "always_on"))
    every = {frozenset(pair) for pair in itertools.combinations(POLICIES, 2)}
    assert len(POLICIES) == 5 and len(every) == 10
    assert len(bs.PAIRS) == 10 and {frozenset(pair) for pair in bs.PAIRS} == every  # each once
    assert {frozenset(pair) for pair in protocol.PAIRS} == every  # the pairs of the protocol


def test_the_exploratory_window_of_b8_is_30_hours_on_the_same_seeds():
    config = bs.StudyConfig()
    assert config.extra_windows == (30.0,) and config.world.window_hours == 12.0
    assert 30.0 != config.world.window_hours and config.planted_pi1 == 0.20
    kinds = (bs.KIND_MAIN, bs.KIND_SWEEP, bs.KIND_TOUCH, bs.KIND_EVENTS, bs.KIND_WINDOW,
             bs.KIND_LEVEL)
    assert kinds == (1, 2, 3, 4, 5, 6) and bs.SECONDARY_KEY == 1
    assert bs.WINDOW_POLICIES == ("release_aware", "flat_matched", "pooled_premium")
    assert bs.WINDOW_PREMIUMS == ("premium_pooled", "premium_matched", "premium_near",
                                  "premium_far")


def test_the_level_sweep_of_b9_is_the_grid_fixed_before_any_result():
    config = bs.StudyConfig()
    assert config.sweep_levels == (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
    assert config.quote.flat_scale == 1.0  # the main grid runs flat_matched at F = 1
    assert config.world.window_hours == 12.0 and config.planted_pi1 == 0.20


def test_main_runs_the_default_design(monkeypatch, tmp_path):
    seen = {}

    def fake(config, out, manifest):
        seen.update(config=config, out=out, manifest=manifest)

    monkeypatch.setattr(bs, "run_study", fake)
    bs.main(["--out", str(tmp_path / "somewhere")])
    assert seen["config"] == bs.StudyConfig()
    assert seen["out"] == tmp_path / "somewhere"
    assert seen["manifest"].command().startswith("python -m kalshi_mm.experiments.backtest_study")
    assert "20261113" in seen["manifest"].seeds


def test_the_world_names_survive_a_csv_round_trip(tmp_path):
    frame = pd.DataFrame({"world": [NULL, PLANTED, "pi1=0"]})
    frame.to_csv(tmp_path / "w.csv", index=False)
    assert pd.read_csv(tmp_path / "w.csv")["world"].tolist() == [NULL, PLANTED, "pi1=0"]


# --------------------------------------------------------------------------- the statistics
def test_event_table_sums_the_markets_of_an_event_and_averages_the_live_share():
    rows = pd.DataFrame({
        "event": ["A", "A", "B"], "market": ["A1", "A2", "B1"], "policy": ["always_on"] * 3,
        "pnl": [1.0, 2.0, 4.0], "gross": [1.5, 2.5, 4.5], "fees": [0.5, 0.5, 0.5],
        "fills": [3, 4, 5], "contracts": [30.0, 40.0, 50.0], "inventory_end": [5.0, -2.0, 1.0],
        "marked_pnl": [0.0, 1.0, 2.0], "quotes_live_share": [0.5, 1.0, 0.25],
        "evaluated": [True, True, False], "release_ts": [10.0, 10.0, 20.0], "series": "S",
    })
    out = bs.event_table(rows).set_index("event")
    assert out.loc["A", ["pnl", "gross", "fees", "fills", "contracts", "inventory_end",
                         "marked_pnl"]].tolist() == [3.0, 4.0, 1.0, 7, 70.0, 3.0, 1.0]
    assert out.loc["A", "quotes_live_share"] == 0.75
    assert out.loc["B", "pnl"] == 4.0 and not out.loc["B", "evaluated"]
    assert out.loc["A", "release_ts"] == 10.0


def test_evaluated_pnl_is_in_release_order_without_the_burn_in():
    events = make_events({"always_on": [1.0, 2.0, 3.0, 4.0], "pull_only": [5.0, 6.0, 7.0, 8.0]},
                         evaluated=[False, True, True, True])
    events["release_ts"] = events["release_ts"].map({1000.0: 1300.0, 1100.0: 1200.0,
                                                     1200.0: 1100.0, 1300.0: 1000.0})
    wide = bs.evaluated_pnl(events)
    assert wide.index.tolist() == ["E003", "E002", "E001"]  # E000 is burn-in, the rest reversed
    assert wide["always_on"].tolist() == [4.0, 3.0, 2.0]
    assert wide["pull_only"].tolist() == [8.0, 7.0, 6.0]


def test_mean_difference_by_hand():
    d = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array([[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 4, 0], [0, 0, 0, 4], [1, 1, 1, 1]])
    out = bs.mean_difference(d, weights)
    # the replications are 1, 2, 3, 4 and 2.5: sorted 1, 2, 2.5, 3, 4; the 2.5% quantile is at
    # position 0.1 (1.1) and the 97.5% quantile at position 3.9 (3.9)
    assert out == pytest.approx({"mean": 2.5, "lo": 1.1, "hi": 3.9})


def test_policy_summary_by_hand():
    events = make_events({"always_on": [100.0, 2.0, -4.0], "pull_only": [0.0, 0.0, 0.0]},
                         evaluated=[False, True, True])
    events.loc[events["policy"] == "always_on", "contracts"] = [1.0, 50.0, 150.0]
    events.loc[events["policy"] == "always_on", "fills"] = [1, 4, 8]
    events.loc[events["policy"] == "always_on", "quotes_live_share"] = [1.0, 0.5, 0.25]
    s = bs.policy_summary(events, "always_on")
    assert s["n_events"] == 2 and s["pnl"] == -1.0 and s["fees"] == 1.0 and s["gross"] == 0.0
    assert s["fills"] == 6.0 and s["contracts"] == 100.0 and s["marked_pnl"] == -1.0
    assert s["cents_per_contract"] == pytest.approx(100.0 * (2.0 - 4.0) / 200.0)  # -1 cent
    assert s["live_share"] == 0.375
    none = events.assign(contracts=0.0)
    assert math.isnan(bs.policy_summary(none, "always_on")["cents_per_contract"])


def test_analyse_gives_each_pair_as_a_minus_b_with_the_documented_bootstrap():
    rng = np.random.default_rng(7)
    base = rng.normal(size=14)
    pnl = {"release_aware": base + 4.0 + rng.normal(0.0, 0.1, 14),  # not the order of POLICIES
           "flat_matched": base + 3.2 + rng.normal(0.0, 0.1, 14),
           "pooled_premium": base + 3.0 + rng.normal(0.0, 0.1, 14),
           "pull_only": base + 0.5, "always_on": base}
    events = make_events(pnl, evaluated=[False] * 4 + [True] * 10)
    meta = {"sweep": "main", "world": NULL, "seed": 3}
    policies, pairs = bs.analyse(events, 200, (1, 0, 3), meta)
    assert [p["policy"] for p in policies] == list(POLICIES)
    assert all(p["world"] == NULL and p["n_events"] == 10 for p in policies)
    assert [(q["a"], q["b"]) for q in pairs] == list(bs.PAIRS)

    counts = cluster_weights(10, 200, np.random.default_rng([bs.SEED, 1, 0, 3]))
    for q in pairs:
        d = (np.asarray(pnl[q["a"]]) - np.asarray(pnl[q["b"]]))[4:]
        boot = [sum(c * x for c, x in zip(row, d, strict=True)) / 10 for row in counts]
        assert q["mean"] == pytest.approx(d.mean())
        assert (q["lo"], q["hi"]) == pytest.approx(tuple(np.quantile(boot, [0.025, 0.975])))
        assert q["n_events"] == 10 and q["sweep"] == "main"
    primary = pairs[0]
    assert (primary["a"], primary["b"]) == ("release_aware", "flat_matched")
    assert 0.5 < primary["mean"] < 1.1 and primary["lo"] > 0.0  # release_aware is ahead by 0.8
    assert (pairs[1]["a"], pairs[1]["b"]) == ("release_aware", "pooled_premium")
    assert 0.8 < pairs[1]["mean"] < 1.2  # and by 1.0 of the pooled premium
    assert (pairs[-1]["a"], pairs[-1]["b"]) == ("pull_only", "always_on")
    assert pairs[-1]["mean"] == pytest.approx(0.5, abs=1e-12)  # a minus b, and a is 0.5 ahead


def test_across_datasets_by_hand():
    pairs = pd.DataFrame({"mean": [-1.0, -2.0, 0.5, 1.5], "lo": [-2.0, -3.0, -0.5, 0.5],
                          "hi": [-0.5, -1.0, 1.0, 2.5]})
    s = bs.across_datasets(pairs)
    sd = np.std([-1.0, -2.0, 0.5, 1.5], ddof=1)
    assert s["n_datasets"] == 4 and s["mean"] == -0.25
    assert s["mc_se"] == pytest.approx(sd / 2.0)
    assert (s["min"], s["max"]) == (-2.0, 1.5)
    assert (s["below_zero"], s["above_zero"]) == (2, 1)  # the third interval contains zero
    assert s["half_width"] == pytest.approx(np.mean([0.75, 1.0, 0.75, 1.0]))
    one = bs.across_datasets(pairs.iloc[:1])
    assert math.isnan(one["mc_se"]) and one["mean"] == -1.0


def test_an_interval_that_ends_exactly_at_zero_does_not_exclude_it():
    pairs = pd.DataFrame({"mean": [-1.0, 1.0], "lo": [-2.0, 0.0], "hi": [0.0, 2.0]})
    s = bs.across_datasets(pairs)
    assert (s["below_zero"], s["above_zero"]) == (0, 0)


def primary_frame(per_dataset):
    """A per-event table of planted datasets whose primary differences are ``per_dataset``."""
    frames = []
    for seed, diffs in enumerate(per_dataset, start=1):
        diffs = np.asarray(diffs, dtype=float)
        events = make_events({"release_aware": diffs, "flat_matched": np.zeros(diffs.size)})
        frames.append(events.assign(world=PLANTED, seed=seed))
    other = make_events({"release_aware": [99.0], "flat_matched": [-99.0]})
    frames.append(other.assign(world=NULL, seed=1))  # the null world is not used
    return pd.concat(frames, ignore_index=True)


def test_sample_size_by_hand():
    frame = primary_frame([[1.0, 3.0], [2.0, 4.0]])
    s = bs.sample_size(frame)
    sd = np.std([1.0, 3.0, 2.0, 4.0], ddof=1)
    z_alpha, z_power = norm.ppf(0.975), norm.ppf(0.80)
    assert s["mean"] == 2.5 and s["sd"] == pytest.approx(sd)
    assert s["events_half_width"] == pytest.approx((z_alpha * sd / 2.5) ** 2, rel=1e-9)
    assert s["events_power_80"] == pytest.approx(((z_alpha + z_power) * sd / 2.5) ** 2, rel=1e-9)
    zero = bs.sample_size(primary_frame([[1.0, -1.0]]))
    assert math.isinf(zero["events_half_width"]) and math.isinf(zero["events_power_80"])


def test_events_needed_uses_the_first_n_events_only():
    d = [1.0] * 20 + [100.0] * 20  # no spread among the first 20, a jump after
    config = bs.StudyConfig(boot_reps=300, event_counts=(10, 20, 30, 40, 50))
    out = bs.events_needed(config, primary_frame([d]))
    assert out["n"].tolist() == [10, 20, 30, 40]  # 50 evaluated events are not there
    assert out.loc[out["n"] <= 20, "halfwidth"].tolist() == [0.0, 0.0]
    assert (out.loc[out["n"] > 20, "halfwidth"] > 1.0).all()
    assert out["mean"].tolist() == pytest.approx([1.0, 1.0, (20 + 10 * 100.0) / 30.0, 50.5])
    only_planted = primary_frame([d, d])
    assert set(bs.events_needed(config, only_planted)["seed"]) == {1, 2}
    with pytest.raises(ValueError):
        bs.events_needed(replace(config, event_counts=(50, 60)), primary_frame([d]))


def test_clean_makes_json():
    obj = {"a": np.float64(1.5), "b": [np.int64(2), float("nan")], 3: (float("inf"), "x")}
    assert bs._clean(obj) == {"a": 1.5, "b": [2, None], "3": [None, "x"]}
    json.dumps(bs._clean(obj), allow_nan=False)


# ----------------------------------------------------------------------------- the wording
def fake_summary(means, below=0, above=0, n=10):
    pairs = {}
    for world, mean in means.items():
        key = f"{world}:{bs.PRIMARY[0]}-{bs.PRIMARY[1]}"
        pairs[key] = {"n_datasets": n, "mean": mean, "mc_se": 0.1, "min": mean - 1.0,
                      "max": mean + 1.0, "below_zero": below, "above_zero": above,
                      "half_width": 0.5}
    return {"pairs": pairs}


def test_the_reading_of_the_primary_table_follows_the_sign():
    config = bs.StudyConfig()
    block = bs.table_primary(fake_summary({NULL: -1.5, PLANTED: 2.25}, below=7, above=1), config)
    reading = next(line for line in block.splitlines() if line.startswith("Reading:"))
    assert ("In the null world `release_aware` earned 1.50 USD per event less than `flat_matched` "
            "(mean of 10 datasets, range -2.50 to -0.50; the 95% interval lies below zero in 7 "
            "of 10 datasets and above zero in 1).") in reading
    assert ("In the planted world `release_aware` earned 2.25 USD per event more than "
            "`flat_matched` (mean of 10 datasets, range +1.25 to +3.25;") in reading
    assert "-1.50 (0.10)" in block and "+2.25 (0.10)" in block
    rows = [line for line in block.splitlines() if line.startswith(("| null", "| planted"))]
    assert row_cells(rows[0]) == ["null", "-1.50 (0.10)", "-2.50 to -0.50", "7", "1", "2", "0.50"]
    assert row_cells(rows[1]) == ["planted", "+2.25 (0.10)", "+1.25 to +3.25", "7", "1", "2",
                                  "0.50"]  # 10 datasets: 7 below zero, 1 above, 2 with zero
    flat = bs.table_primary(fake_summary({NULL: 0.0, PLANTED: 0.0}), config)
    assert "null world `release_aware` and `flat_matched` earned the same per event" in flat


def test_the_primary_table_says_what_flat_matched_is_and_what_it_does_not_show():
    block = bs.table_primary(fake_summary({NULL: -1.5, PLANTED: 2.25}), bs.StudyConfig())
    heading, reading, _, not_shown = block.strip().split("\n\n")
    assert heading == ("### B1. `release_aware` minus `flat_matched`, net P&L per event "
                       "(synthetic data)")
    assert ("`flat_matched` quotes one flat premium equal to the average of the bucket premiums "
            "over the quoting window, so the difference is the effect of when a premium is "
            "applied and not of its average size.") in reading
    assert "pooled" not in block  # the first primary pair is B3's and B8's
    assert not_shown.startswith(
        "Does not show: whether the planted informed share is what makes the bucket premiums "
        "useful: the null world's buckets also differ, because of the trade-rate surge and the "
        "price level; and whether either policy earns or loses money on real Kalshi markets")
    assert "\n" not in reading and "\n" not in not_shown


def test_numbers_are_formatted_with_a_sign_and_n_a():
    assert bs._num(1.234) == "1.23" and bs._num(1.234, 1) == "1.2"
    assert bs._num(1.234, signed=True) == "+1.23" and bs._num(-0.004, signed=True) == "-0.00"
    assert bs._num(NAN) == "n/a" and bs._num(None) == "n/a" and bs._num(math.inf) == "n/a"
    assert bs._label(NULL) == "null" and bs._label(PLANTED) == "planted"


# ----------------------------------------------------------------- the study on a small design
SMALL = bs.StudyConfig(
    world=WorldParams(n_events=30, window_hours=2.0),
    quote=QuoteParams(window_hours=2.0, vol_lookback_s=3600.0, min_history_events=5),
    seeds=(1, 2, 3),
    boot_reps=300,
    sweep_pi1=(0.0, 0.20),
    event_counts=(10, 25),
    extra_windows=(8.0,),  # 6 hours of trading before the quoting window: trades in the last bucket
    sweep_levels=(0.5, 1.0, 2.0),
)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    out = tmp_path_factory.mktemp("study")
    manifest = Manifest("kalshi_mm.experiments.backtest_study", "test", ["--out", "results/small"])
    summary = bs.run_study(SMALL, out, manifest)
    return {"out": out, "summary": summary}


def read(small, name):
    return pd.read_csv(small["out"] / name)


def test_the_files_are_written(small):
    names = sorted(p.name for p in small["out"].iterdir())
    assert names == ["backtest_study.manifest.txt", "events_needed.csv", "levels.csv",
                     "pairs.csv", "per_event.csv", "per_event_windows.csv", "policies.csv",
                     "premium_check.csv", "summary.json", "tables.md", "windows.csv"]
    manifest = (small["out"] / "backtest_study.manifest.txt").read_text()
    assert "command: python -m kalshi_mm.experiments.backtest_study --out results/small" in manifest
    assert "seeds: test" in manifest and "bootstrap: 300 replications" in manifest
    assert "dataset seeds 1 to 3" in manifest and str(REPO) not in manifest
    assert "B8, added after the first results (exploratory): the planted world with 8 hours " \
           "of trading before each release instead of 2" in manifest
    assert "only the policies release_aware, flat_matched, pooled_premium" in manifest
    assert "default_rng([20261113, 5, window index, seed])" in manifest  # the primary difference
    assert "default_rng([20261113, 5, window index, seed, 1])" in manifest  # the secondary one
    assert "B9, added after an independent review of the first results (exploratory): the " \
           "planted world of the main grid" in manifest
    assert "release_aware once and flat_matched with the flat premium multiplied by F " \
           "(QuoteParams.flat_scale) in {0.5, 1, 2}, no other policy" in manifest
    assert "default_rng([20261113, 6, index of F, seed])" in manifest
    assert json.loads((small["out"] / "summary.json").read_text()) == small["summary"]


def test_every_dataset_has_the_evaluated_events_and_all_policies(small):
    per_event = read(small, "per_event.csv")
    assert per_event.groupby(["world", "seed"]).size().eq(30 * 5).all()
    assert sorted(per_event["world"].unique()) == [NULL, PLANTED]
    evaluated = per_event[per_event["evaluated"]]
    assert evaluated.groupby(["world", "seed", "policy"]).size().eq(25).all()  # 30 - 5 burn-in
    first = per_event[per_event["event"] == "SYN-000000"]
    assert not first["evaluated"].any()
    assert per_event["pnl"].notna().all()


def test_the_pairs_file_equals_a_recomputation_from_the_per_event_file(small):
    per_event, pairs = read(small, "per_event.csv"), read(small, "pairs.csv")
    main = pairs[pairs["sweep"] == "main"]
    assert len(main) == 2 * 3 * 10
    for row in main.itertuples():
        part = per_event[(per_event["world"] == row.world) & (per_event["seed"] == row.seed)
                         & per_event["evaluated"]]
        a = part[part["policy"] == row.a].set_index("event")["pnl"].sort_index()
        b = part[part["policy"] == row.b].set_index("event")["pnl"].sort_index()
        d = (a - b).to_numpy()
        assert row.n_events == d.size == 25 and row.mean == pytest.approx(d.mean(), rel=1e-8)
        key = (bs.KIND_MAIN, 0 if row.world == NULL else 1, row.seed)
        counts = np.random.default_rng([bs.SEED, *key]).multinomial(25, np.full(25, 1 / 25), 300)
        boot = (counts * d).sum(axis=1) / 25
        lo, hi = np.quantile(boot, [0.025, 0.975])
        assert (row.lo, row.hi) == pytest.approx((lo, hi), rel=1e-7, abs=1e-9)
        assert row.lo <= row.mean <= row.hi


def test_the_policies_file_equals_a_recomputation_from_the_per_event_file(small):
    per_event, policies = read(small, "per_event.csv"), read(small, "policies.csv")
    main = policies[policies["sweep"] == "main"]
    assert len(main) == 2 * 3 * 5
    for row in main.itertuples():
        part = per_event[(per_event["world"] == row.world) & (per_event["seed"] == row.seed)
                         & (per_event["policy"] == row.policy) & per_event["evaluated"]]
        assert row.pnl == pytest.approx(part["pnl"].mean(), rel=1e-8)
        assert row.fills == pytest.approx(part["fills"].mean(), rel=1e-8)
        assert row.gross == pytest.approx(part["pnl"].mean() + part["fees"].mean(), rel=1e-8)
        assert row.cents_per_contract == pytest.approx(
            100.0 * part["pnl"].sum() / part["contracts"].sum(), rel=1e-8)
        assert row.live_share == pytest.approx(part["quotes_live_share"].mean(), rel=1e-8)


def test_the_sweep_cells_agree_with_the_main_grid_and_with_each_other(small):
    policies = read(small, "policies.csv")
    columns = ["n_events", "pnl", "gross", "fees", "fills", "contracts", "marked_pnl",
               "live_share"]
    cells = policies[policies["sweep"] == "pi1_fee"]
    assert len(cells) == 2 * 2 * 5  # two planted shares, two fees, five policies
    for pi1, world in ((0.0, NULL), (0.2, PLANTED)):
        main = policies[(policies["sweep"] == "main") & (policies["world"] == world)
                        & (policies["seed"] == 1)].set_index("policy")
        cell = cells[(cells["pi1"] == pi1) & (cells["fee"] == 0.0175)].set_index("policy")
        free = cells[(cells["pi1"] == pi1) & (cells["fee"] == 0.0)].set_index("policy")
        # the same world, the same parameters: the same numbers, by another code path
        pd.testing.assert_frame_equal(cell[columns], main.loc[cell.index, columns],
                                      check_exact=False, rtol=1e-9)
        # the fee does not change the quotes, only the accounting
        assert (free["fees"] == 0.0).all() and (free["pnl"] == free["gross"]).all()
        pd.testing.assert_frame_equal(free[["fills", "contracts", "gross"]],
                                      cell[["fills", "contracts", "gross"]],
                                      check_exact=False, rtol=1e-9)
    touch = policies[policies["sweep"] == "touch"].set_index("policy")
    through = cells[(cells["pi1"] == 0.2) & (cells["fee"] == 0.0175)].set_index("policy")
    assert (touch["rule"] == "touch").all() and (through["rule"] == "through").all()
    assert (touch["fills"] >= through["fills"]).all()  # a print at the quote fills as well


def test_tables_follow_the_block_format(small):
    text = (small["out"] / "tables.md").read_text()
    headings = re.findall(r"(?m)^### .*$", text)
    assert [re.match(r"### (B\d+)\. ", h).group(1) for h in headings] == [
        "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
    assert all(h.endswith(" (synthetic data)") for h in headings)
    heading = load_make_tables().HEADING  # the pattern that finds the tables for the README
    assert [m.group(1) for m in heading.finditer(text)] == [f"B{i}" for i in range(1, 10)]
    blocks = re.split(r"(?m)^(?=### )", text.strip() + "\n")
    blocks = [b.strip() for b in blocks if b.strip()]
    assert len(blocks) == 9
    for block in blocks:
        paragraphs = block.split("\n\n")
        assert len(paragraphs) == 4, block[:80]
        title, reading, table, not_shown = paragraphs
        assert title.startswith("### ") and "\n" not in title
        assert reading.startswith("Reading: ") and "\n" not in reading
        assert not_shown.startswith("Does not show: ") and "\n" not in not_shown
        lines = table.splitlines()
        assert len(lines) >= 4
        assert all(line.startswith("|") and line.endswith("|") for line in lines)
        widths = {line.count("|") for line in lines}
        assert len(widths) == 1  # every row has the same number of cells
    assert not re.search(r"\b(nan|none|todo)\b", text.lower())
    assert "$" not in text  # a dollar sign would start math in a README


def test_the_tables_quote_the_numbers_of_the_summary(small):
    text = (small["out"] / "tables.md").read_text()
    s = small["summary"]
    primary = s["pairs"][f"{PLANTED}:release_aware-flat_matched"]
    assert f"{primary['mean']:+.2f} ({primary['mc_se']:.2f})" in text
    second = s["pairs"][f"{PLANTED}:release_aware-pooled_premium"]  # B3 and the secondary of B8
    assert f"{second['mean']:+.2f} ({second['mc_se']:.2f})" in text
    for entry in s["levels"]:
        assert f"{entry['mean']:+.2f} ({entry['mc_se']:.2f})" in text
    cell = s["sweeps"]["pi1_fee"][0]
    assert (f"{cell['primary']['mean']:+.2f} [{cell['primary']['lo']:+.2f}, "
            f"{cell['primary']['hi']:+.2f}]") in text
    assert f"{s['policies'][f'{NULL}:always_on']['pnl']:.2f}" in text
    assert s["design"]["bootstrap_replications"] == 300 and s["design"]["burn_in_events"] == 5
    assert len(s["events_needed"]) == 2 and len(s["sweeps"]["pi1_fee"]) == 4
    assert len(s["sweeps"]["touch"]) == 1 and s["sweeps"]["touch"][0]["rule"] == "touch"
    assert s["design"]["primary"] == ["release_aware", "flat_matched"]
    assert s["design"]["secondary"] == ["release_aware", "pooled_premium"]
    assert s["design"]["sweep_levels"] == [0.5, 1.0, 2.0]


def test_a_second_run_gives_the_same_files(small, tmp_path):
    manifest = Manifest("kalshi_mm.experiments.backtest_study", "test", ["--out", "x"])
    bs.run_study(SMALL, tmp_path, manifest)
    for name in ("per_event.csv", "policies.csv", "pairs.csv", "events_needed.csv",
                 "premium_check.csv", "windows.csv", "per_event_windows.csv", "levels.csv",
                 "summary.json", "tables.md"):
        assert (tmp_path / name).read_bytes() == (small["out"] / name).read_bytes(), name


def test_the_backtest_gets_no_latent_columns(monkeypatch):
    seen = []
    real = bs.walk_forward

    def spy(tape, markets, events, policies, params, **kwargs):
        seen.append(list(tape.columns))
        return real(tape, markets, events, policies, params, **kwargs)

    monkeypatch.setattr(bs, "walk_forward", spy)
    world, rows = bs.run_dataset(WorldParams(n_events=4, window_hours=2.0), 5, SMALL.quote)
    assert any(c.startswith("latent_") for c in world.tape.columns)  # the world has them
    assert len(seen) == 1 and not any(c.startswith("latent_") for c in seen[0])
    assert len(rows) == 4 * len(POLICIES)


UPPER_EDGES = (5.0, 15.0, 30.0, 60.0, 180.0, 360.0)  # of the buckets, in minutes to the release


def decision_weights(window_hours, refresh_s=60.0):
    """How many decisions of a quoting window fall in each of the seven buckets: decision j is
    taken ``window - j * refresh`` seconds before the release and a bucket includes its upper
    edge."""
    window = window_hours * 3600.0
    counts = [0] * 7
    for j in range(math.ceil(window / refresh_s)):
        counts[sum(edge < (window - refresh_s * j) / 60.0 for edge in UPPER_EDGES)] += 1
    return counts


def matched_by_hand(values, weights):
    """The premium of the bucket of each decision, averaged over the decisions; a bucket without
    a decision does not count, whatever its value."""
    used = [(w, v) for w, v in zip(weights, values, strict=True) if w > 0]
    return sum(w * v for w, v in used) / sum(weights)


def test_the_decision_weights_of_the_helper_are_those_of_the_windows_of_the_tests():
    assert decision_weights(12.0) == [5, 10, 15, 30, 120, 180, 360]  # of 720 decisions
    assert decision_weights(2.0) == [5, 10, 15, 30, 60, 0, 0]  # of 120
    assert decision_weights(8.0, 600.0) == [0, 1, 2, 3, 12, 18, 12]  # 48: 10, 20, ..., 480


def test_the_premium_check_equals_loops_over_the_markouts(small):
    check = read(small, "premium_check.csv")
    quote = SMALL.quote
    for name, pi1 in ((NULL, 0.0), (PLANTED, 0.2)):
        seed = 2
        world = generate_world(replace(SMALL.world, pi1=pi1), seed)
        stripped = world.tape.drop(columns=[c for c in world.tape if c.startswith("latent_")])
        markouts = trade_markouts(stripped, world.events, world.markets,
                                  horizon_s=quote.horizon_s, stale_s=quote.stale_s)
        total, latent, count = np.zeros(7), np.zeros(7), np.zeros(7)
        for row in markouts.itertuples():
            m = row.mid_before
            if not m == m or m * (1.0 - m) < 0.05:
                continue
            scale = 4.0 * m * (1.0 - m)
            truth = world.tape.loc[row.Index]
            b = int(row.bucket)
            total[b] += row.adverse_selection / scale
            latent[b] += row.side * (truth["latent_mid_post"] - truth["latent_mid_pre"]) / scale
            count[b] += 1
        got = check[(check["world"] == name) & (check["seed"] == seed)].set_index("bucket")
        assert got.index.tolist() == [*BUCKET_LABELS, "pooled", "matched"]
        with np.errstate(invalid="ignore", divide="ignore"):
            target, impact = total / count, latent / count
        np.testing.assert_allclose(got["target"].to_numpy()[:7], target, rtol=1e-7)
        np.testing.assert_allclose(got["latent"].to_numpy()[:7], impact, rtol=1e-7)
        assert got.loc["pooled", "target"] == pytest.approx(total.sum() / count.sum(), rel=1e-7)
        assert got.loc["pooled", "latent"] == pytest.approx(latent.sum() / count.sum(), rel=1e-7)
        last = world.events["event"].iloc[-1]
        end = estimate_premiums(markouts[markouts["event"] != last], min_events=5)
        np.testing.assert_allclose(got["estimate_end"].to_numpy()[:7], end.by_bucket, rtol=1e-7)
        assert got.loc["pooled", "estimate_end"] == pytest.approx(end.pooled, rel=1e-7)
        first = estimate_premiums(markouts[markouts["event"].isin(world.events["event"][:5])],
                                  min_events=5)
        np.testing.assert_allclose(got["estimate_burn_in"].to_numpy()[:7], first.by_bucket,
                                   rtol=1e-7)
        # the matched row: the buckets averaged over the 120 decisions of the 2 hour window
        weights = decision_weights(quote.window_hours)
        for column, values in (("target", target), ("latent", impact),
                               ("estimate_end", end.by_bucket),
                               ("estimate_burn_in", first.by_bucket)):
            assert got.loc["matched", column] == pytest.approx(
                matched_by_hand(values, weights), rel=1e-7), column
    assert check["estimate_end"].notna().all() and (check["estimate_end"] >= 0.0).all()
    assert check["target"].isna().sum() > 0  # the 2 hour windows leave the far buckets empty


def test_the_matched_row_averages_the_buckets_over_the_720_decisions_of_the_default_window():
    quote = QuoteParams(min_history_events=3)  # the default 12 hour window, a refresh every minute
    world = generate_world(WorldParams(n_events=10, window_hours=12.0, pi1=0.2), 4)
    got = bs.premium_check(world, quote, 4, PLANTED).set_index("bucket")
    weights = [5, 10, 15, 30, 120, 180, 360]  # the decisions 720, 719, ... 1 minutes before
    assert weights == decision_weights(12.0) and sum(weights) == len(decision_minutes(quote)) == 720
    assert got.index.tolist() == [*BUCKET_LABELS, "pooled", "matched"]
    for column in ("target", "latent", "estimate_end", "estimate_burn_in"):
        buckets = got.loc[list(BUCKET_LABELS), column].to_numpy()
        assert np.isfinite(buckets).all(), column  # every bucket has trades in this world
        by_hand = sum(w * v for w, v in zip(weights, buckets, strict=True)) / 720.0
        assert got.loc["matched", column] == pytest.approx(by_hand, rel=1e-12), column
        assert got.loc["matched", column] != pytest.approx(got.loc["pooled", column], rel=1e-6)


def test_a_window_of_a_whole_number_of_refreshes_has_no_decision_too_many():
    # 0.55 hours with a refresh every minute are 33 decisions, but 0.55 * 3600 / 60 is
    # 33.00000000000001 in floating point
    minutes = decision_minutes(QuoteParams(window_hours=0.55, refresh_s=60.0))
    assert len(minutes) == 33
    assert minutes[0] == pytest.approx(33.0) and minutes[-1] == pytest.approx(1.0)


def test_the_premiums_known_at_an_event_use_only_events_released_before_its_window():
    quote = SMALL.quote
    world = generate_world(replace(SMALL.world, pi1=0.2), 2)
    stripped = world.tape.drop(columns=[c for c in world.tape if c.startswith("latent_")])
    markouts = trade_markouts(stripped, world.events, world.markets, horizon_s=quote.horizon_s,
                              stale_s=quote.stale_s)
    events = world.events.sort_values(["release_ts", "event"]).reset_index(drop=True)
    rank = 15
    known = bs._premiums_known_at(markouts, events, rank, quote)
    assert known.n_events == rank  # the events 0 to 14 were released before the window of event 15
    assert known.matched == pytest.approx(
        matched_by_hand(known.by_bucket, decision_weights(quote.window_hours)), rel=1e-12)
    start = events["release_ts"].iloc[rank] - quote.window_hours * 3600.0
    later = events.loc[events["release_ts"] > start, "event"]
    assert rank in later.index and len(later) == len(events) - rank
    scrambled = markouts.copy()
    hit = scrambled["event"].isin(later)
    scrambled.loc[hit, "adverse_selection"] = 50.0 + 3.0 * scrambled.loc[hit, "adverse_selection"]
    assert bs._premiums_known_at(scrambled, events, rank, quote) == known  # no look-ahead
    changed = markouts.copy()
    hit = changed["event"] == events["event"].iloc[rank - 1]
    changed.loc[hit, "adverse_selection"] += 0.5  # an earlier event does matter
    after = bs._premiums_known_at(changed, events, rank, quote)
    assert after.pooled != known.pooled and after.matched != known.matched


def premium_rows(world, end, target=None, latent=None, burn_in=None):
    """A premium check of one dataset: the rows of the seven buckets, pooled and matched."""
    assert len(end) == len(BUCKET_LABELS) + 2
    return pd.DataFrame({
        "world": world, "seed": 1, "bucket": [*BUCKET_LABELS, "pooled", "matched"],
        "estimate_end": end, "estimate_burn_in": burn_in or end, "target": target or end,
        "latent": latent or end})


def test_the_reading_of_the_premium_table_gives_the_pooled_and_matched_premium_and_their_ratio():
    end = [0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.05, 0.03]
    target = [0.10, 0.08, 0.06, 0.042, 0.03, 0.02, 0.01, 0.05, 0.0295]  # gaps 0.20 and 0.05 cents
    latent = [0.09, 0.07, 0.06, 0.04, 0.03, 0.02, 0.015, 0.05, 0.03]
    burn_in = [0.11, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.05, 0.03]
    check = pd.concat([premium_rows(NULL, [2.0 * v for v in end]),
                       premium_rows(PLANTED, end, target, latent, burn_in)], ignore_index=True)
    heading, reading, table, not_shown = bs.table_premium_check(check).strip().split("\n\n")
    assert heading == ("### B7. Estimated premiums against their target and the latent price "
                       "impact (synthetic data)")
    assert reading == (
        "Reading: In the planted world the premium estimated at the end is 10.00 cents in the "
        "bucket (0,5] minutes and 1.00 cents in the bucket >360 (latent price impact 9.00 and "
        "1.50); after the burn-in it is 11.00 in the first bucket; the largest gap between the "
        "estimate at the end and its target is 0.20 cents. The pooled premium estimated at the "
        "end is 5.00 cents and the matched premium, the flat premium of `flat_matched` (the "
        "bucket premiums averaged over the decisions of the quoting window), is 3.00 cents: a "
        "ratio of 1.67 of the pooled premium to the matched premium.")
    rows = table.splitlines()
    assert row_cells(rows[0]) == [
        "World", "Minutes to release (or pooled, or matched)", "Estimate at the end, cents",
        "Target: mean over the dataset, cents", "Latent price impact, cents",
        "Estimate after burn-in, cents"]
    assert len(rows) == 2 + 2 * 9
    assert row_cells(rows[2 + 9 + 0]) == ["planted", "(0,5]", "10.00", "10.00", "9.00", "11.00"]
    assert row_cells(rows[2 + 9 + 7]) == ["planted", "pooled", "5.00", "5.00", "5.00", "5.00"]
    assert row_cells(rows[2 + 9 + 8]) == ["planted", "matched", "3.00", "2.95", "3.00", "3.00"]
    assert row_cells(rows[2 + 8]) == ["null", "matched", "6.00", "6.00", "6.00", "6.00"]
    assert not_shown == ("Does not show: whether the premium prices the adverse selection of a "
                         "fill: it is measured per trade, and the fills of a maker are not a "
                         "random sample of the trades.")


def test_the_helpers_of_the_exploratory_tables():
    assert bs._ratio(0.03, 0.02) == "1.50" and bs._ratio(1.0, 3.0) == "0.33"
    assert bs._ratio(1.0, 0.0) == "n/a" and bs._ratio(NAN, 1.0) == "n/a"
    assert bs._ratio(1.0, NAN) == "n/a" and bs._ratio(math.inf, 1.0) == "n/a"
    assert bs._change(2.0, 1.0) == "-50%" and bs._change(1.0, 1.256) == "+26%"
    assert bs._change(0.0, 1.0) == "n/a" and bs._change(NAN, 1.0) == "n/a"
    assert bs._and(["a"]) == "a" and bs._and(["a", "b"]) == "a and b"
    assert bs._and(["a", "b", "c"]) == "a, b and c"
    assert bs._sign_pattern([1.0, 2.0]) == "stays positive"
    assert bs._sign_pattern([-1.0, -2.0]) == "stays negative"
    assert bs._sign_pattern([1.0, -2.0]) == "does not keep one sign"
    assert bs._sign_pattern([0.0, 1.0]) == "does not keep one sign"


def fake_sweeps(through, touch, planted=0.2, fee=0.0175):
    def cell(pi1, rule, pnl):
        primary = {"mean": 0.5, "lo": -0.5, "hi": 1.5}
        return {"pi1": pi1, "fee": fee, "rule": rule, "primary": primary, "pnl": pnl}

    cells = [cell(0.0, "through", through), cell(planted, "through", through)]
    return {"sweeps": {"pi1_fee": cells, "touch": [cell(planted, "touch", touch)]}}


def test_the_reading_of_the_touch_table_names_the_largest_change_and_its_direction():
    config = bs.StudyConfig()
    # always_on, pull_only, pooled_premium, flat_matched, release_aware
    through = dict(zip(POLICIES, [-10.0, -9.0, -5.0, -5.5, -6.0], strict=True))
    touch = dict(zip(POLICIES, [-9.0, -8.5, -8.0, -5.0, -5.5], strict=True))  # pooled: -3.0
    reading = bs.table_touch(fake_sweeps(through, touch), config)
    assert "lowers net P&L per event by 3.00 USD for `pooled_premium`, the largest" in reading
    up = dict(zip(POLICIES, [-4.0, -9.0, -5.0, -5.5, -6.0], strict=True))  # always_on: +6.0
    assert "raises net P&L per event by 6.00 USD for `always_on`" in bs.table_touch(
        fake_sweeps(through, up), config)
    flat = dict(zip(POLICIES, [-10.0, -9.0, -5.0, -9.5, -6.0], strict=True))  # flat_matched: -4.0
    assert "lowers net P&L per event by 4.00 USD for `flat_matched`" in bs.table_touch(
        fake_sweeps(through, flat), config)


def sweep_table(block):
    """The lines of the table of a sweep block."""
    return next(part for part in block.split("\n\n") if part.startswith("| ")).splitlines()


def row_cells(line):
    """The cells of a table row."""
    return line.strip("| ").split(" | ")


def test_the_sweep_tables_have_one_column_per_policy_then_the_primary_difference():
    config = bs.StudyConfig()
    through = dict(zip(POLICIES, [-10.0, -9.0, -5.0, -5.5, -6.0], strict=True))
    touch = dict(zip(POLICIES, [-9.0, -8.5, -8.0, -5.0, -5.5], strict=True))
    last = "`release_aware` minus `flat_matched`, USD per event [95% interval]"
    policies = ["`always_on`, USD per event", "`pull_only`, USD per event",
                "`pooled_premium`, USD per event", "`flat_matched`, USD per event",
                "`release_aware`, USD per event"]
    b4 = sweep_table(bs.table_sweep(fake_sweeps(through, touch), config))
    assert row_cells(b4[0]) == ["Planted share pi1", "Maker fee coefficient", *policies, last]
    assert row_cells(b4[2]) == ["0.00", "0.0175", "-10.00", "-9.00", "-5.00", "-5.50", "-6.00",
                                "+0.50 [-0.50, +1.50]"]
    b5 = sweep_table(bs.table_touch(fake_sweeps(through, touch), config))
    assert row_cells(b5[0]) == ["Fill rule", *policies, last]
    assert row_cells(b5[2])[:2] == ["through (primary)", "-10.00"]
    assert row_cells(b5[3]) == ["touch", "-9.00", "-8.50", "-8.00", "-5.00", "-5.50",
                                "+0.50 [-0.50, +1.50]"]


def test_the_reading_of_the_policy_table_counts_the_losing_cells():
    config = bs.StudyConfig()
    values = {f"{w}:{p}": {"pnl": v, "pnl_mc_se": 0.1, "fees": 1.0, "fills": 10.0,
                           "contracts": 50.0, "cents_per_contract": 0.5, "marked_pnl": v,
                           "live_share": 0.9}
              for w, vs in ((NULL, [-3.0, -2.0, 1.0, 2.0, 4.0]),
                            (PLANTED, [-8.0, -1.0, -2.0, 0.5, -0.5]))
              for p, v in zip(POLICIES, vs, strict=True)}
    block = bs.table_policies({"policies": values}, config)
    assert block.startswith("### B2. The five policies, means over the datasets of each world ")
    assert "negative in 6 of 10 world and policy cells" in block
    assert "the highest is `release_aware` in the null world (+4.00 USD)" in block
    assert "the lowest `always_on` in the planted world (-8.00 USD)" in block
    rows = [line for line in block.splitlines() if line.startswith(("| null", "| planted"))]
    assert [row.split(" | ")[1] for row in rows] == [f"`{p}`" for p in POLICIES] * 2


def fake_pairs(means):
    """A summary with the pairs of B3 in both worlds; ``means`` gives ``{(world, a, b): mean}``,
    every other pair has the mean 0.1."""
    pairs = {}
    for world in (NULL, PLANTED):
        for a, b in bs.PAIRS:
            mean = means.get((world, a, b), 0.1)
            pairs[f"{world}:{a}-{b}"] = {
                "n_datasets": 10, "mean": mean, "mc_se": 0.1, "min": mean - 1.0,
                "max": mean + 1.0, "below_zero": 0, "above_zero": 4, "half_width": 0.5}
    return {"pairs": pairs}


def test_the_pairs_table_has_the_nine_other_pairs_and_the_caveats_of_the_review():
    config = bs.StudyConfig()
    block = bs.table_pairs(fake_pairs({(PLANTED, "pooled_premium", "always_on"): -7.5}), config)
    heading, reading, table, not_shown = block.strip().split("\n\n")
    assert heading == "### B3. Secondary comparisons of net P&L per event (synthetic data)"
    assert reading == (
        "Reading: The largest of the 18 differences is `pooled_premium` minus `always_on` in the "
        "planted world, -7.50 USD per event; the 95% interval lies below zero in 0 of 10 "
        "datasets and above zero in 4.")
    rows = table.splitlines()[2:]
    assert len(rows) == 18
    assert rows[0].startswith("| null | `release_aware` minus `pooled_premium` | +0.10 (0.10) |")
    assert rows[1].startswith("| null | `flat_matched` minus `pooled_premium` |")
    assert rows[8].startswith("| null | `pull_only` minus `always_on` |")
    assert rows[9].startswith("| planted | `release_aware` minus `pooled_premium` |")
    assert not any("`release_aware` minus `flat_matched`" in row for row in rows)  # B1's pair
    assert ("`pooled_premium`, `flat_matched` and `release_aware` also add the terminal variance "
            "m(1 - m) to the variance of the reservation price, which `always_on` and "
            "`pull_only` do not, so their differences from those two policies are the effect of "
            "the premium and of a stronger inventory skew together;") in not_shown
    assert ("the pairs with `pooled_premium` have a sign that follows the level of the pooled "
            "premium (B8, B9);") in not_shown
    assert "family-wise error rate" in not_shown and "\n" not in not_shown


def test_the_reading_of_the_events_needed_table_handles_a_zero_effect():
    config = bs.StudyConfig()
    rows = [{"n": 10, "half_width_first_seed": 2.0, "half_width_mean": 2.5, "half_width_min": 2.0,
             "half_width_max": 3.0, "excludes_zero": 1, "n_datasets": 10}]
    size = {"mean": 0.0, "sd": 3.0, "events_half_width": None, "events_power_80": None}
    block = bs.table_events_needed({"events_needed": rows, "sample_size": size}, config)
    assert ("The mean difference over all planted datasets is zero, so no number of events is "
            "given.") in block
    size = {"mean": -1.0, "sd": 3.0, "events_half_width": 34.57, "events_power_80": 70.6}
    block = bs.table_events_needed({"events_needed": rows, "sample_size": size}, config)
    assert "about 35 evaluated events" in block and "about 71 would give" in block
    assert "(-1.00 USD per event)" in block
    assert "pooled" not in block.lower()  # the table follows the primary pair


# ------------------------------------------------------------ B8: hours of trading (exploratory)
def check_frame(rows):
    """A premium check (one row per world, seed and bucket) from ``{(world, seed): [premium of each
    bucket, then the pooled and the matched premium]}``; only ``estimate_end`` is filled with the
    given numbers."""
    frames = []
    for (world, seed), values in rows.items():
        assert len(values) == len(BUCKET_LABELS) + 2
        frames.append(pd.DataFrame({"world": world, "seed": seed,
                                    "bucket": [*BUCKET_LABELS, "pooled", "matched"],
                                    "estimate_end": values}))
    return pd.concat(frames, ignore_index=True)


def test_window_premiums_are_the_pooled_the_matched_and_the_first_and_last_bucket():
    values = [0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01, 0.05, 0.045]
    got = bs.window_premiums(check_frame({(PLANTED, 1): values}))
    assert got == {"premium_pooled": 0.05, "premium_matched": 0.045, "premium_near": 0.10,
                   "premium_far": 0.01}


def test_the_default_window_row_holds_the_planted_primary_and_secondary_rows_of_the_main_grid():
    def pair(sweep, world, seed, a, b, mean):
        return {"sweep": sweep, "world": world, "seed": seed, "a": a, "b": b, "n_events": 180,
                "mean": mean, "lo": mean - 1.0, "hi": mean + 1.0}

    primary, secondary = bs.PRIMARY, bs.SECONDARY
    pairs = pd.DataFrame([
        pair("main", PLANTED, 1, *primary, -1.0), pair("main", PLANTED, 2, *primary, -2.0),
        pair("main", PLANTED, 1, *secondary, 5.0), pair("main", PLANTED, 2, *secondary, 6.0),
        pair("main", NULL, 1, *primary, 50.0), pair("main", NULL, 1, *secondary, 51.0),  # other
        pair("main", PLANTED, 1, "release_aware", "always_on", 60.0),  # another pair
        pair("touch", PLANTED, 1, *primary, 70.0), pair("touch", PLANTED, 1, *secondary, 71.0),
    ])
    check = check_frame({
        (PLANTED, 1): [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        (PLANTED, 2): [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        (NULL, 1): [9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9],
    })
    rows = bs.default_window_records(bs.StudyConfig(), pairs, check)
    assert rows[["window_hours", "seed", "n_events"]].values.tolist() == [[12.0, 1, 180],
                                                                           [12.0, 2, 180]]
    assert rows[["mean", "lo", "hi"]].values.tolist() == [[-1.0, -2.0, 0.0], [-2.0, -3.0, -1.0]]
    assert rows[list(bs.SECONDARY_COLUMNS)].values.tolist() == [[5.0, 4.0, 6.0], [6.0, 5.0, 7.0]]
    # the pooled and the matched premium are the last two rows of a check, the buckets the first
    # seven
    np.testing.assert_allclose(rows[list(bs.WINDOW_PREMIUMS)].to_numpy(),
                               [[0.8, 0.9, 0.1, 0.7], [1.8, 1.9, 1.1, 1.7]], rtol=1e-12)


def test_window_summary_by_hand():
    windows = pd.DataFrame({
        "window_hours": [30.0, 30.0, 12.0, 12.0, 12.0], "seed": [1, 2, 1, 2, 3], "n_events": 180,
        "mean": [1.0, 3.0, -1.0, -3.0, -2.0], "lo": [0.5, -1.0, -2.0, -4.0, -2.5],
        "hi": [1.5, 7.0, -0.5, -2.0, 0.5],
        "secondary_mean": [2.0, 4.0, 5.0, 7.0, 9.0], "secondary_lo": [1.0, 3.0, 4.0, -1.0, 8.0],
        "secondary_hi": [3.0, 5.0, 6.0, 15.0, 10.0],
        "premium_pooled": [0.02, 0.04, 0.03, 0.05, 0.10],
        "premium_matched": [0.01, 0.03, 0.02, 0.04, 0.12],
        "premium_near": [0.1, 0.3, 0.2, 0.4, 0.9],
        "premium_far": [0.0, 0.01, 0.02, 0.03, 0.10],
    })
    short, long = bs.window_summary(windows)  # increasing hours of trading
    assert (short["window_hours"], long["window_hours"]) == (12.0, 30.0)
    assert (short["n_datasets"], long["n_datasets"]) == (3, 2) and short["mean"] == -2.0
    assert short["mc_se"] == pytest.approx(1.0 / math.sqrt(3))  # sd 1 of -1, -3, -2
    assert (short["min"], short["max"]) == (-3.0, -1.0)
    assert (short["below_zero"], short["above_zero"]) == (2, 0)  # the third interval has zero in it
    assert short["half_width"] == pytest.approx(np.mean([0.75, 1.0, 1.5]))
    assert (long["below_zero"], long["above_zero"]) == (0, 1)  # the second interval contains zero
    # the secondary difference has its own columns: its numbers are not those of the primary
    two, one = short["secondary"], long["secondary"]
    assert (two["n_datasets"], one["n_datasets"]) == (3, 2) and two["mean"] == 7.0
    assert two["mc_se"] == pytest.approx(2.0 / math.sqrt(3))  # sd 2 of 5, 7, 9
    assert (two["min"], two["max"]) == (5.0, 9.0) and one["mean"] == 3.0
    assert (two["below_zero"], two["above_zero"]) == (0, 2)  # the second interval has zero in it
    assert (one["below_zero"], one["above_zero"]) == (0, 2)
    assert two["half_width"] == pytest.approx(np.mean([1.0, 8.0, 1.0]))
    # the premiums are means over the datasets (the medians would be 0.05, 0.04, 0.4 and 0.03)
    assert [short[k] for k in bs.WINDOW_PREMIUMS] == pytest.approx([0.06, 0.06, 0.5, 0.05])
    assert [long[k] for k in bs.WINDOW_PREMIUMS] == pytest.approx([0.03, 0.02, 0.2, 0.005])


def fake_windows(entries):
    """A summary with the B8 entries ``(hours, mean, below, above, secondary mean, below, above,
    pooled, matched, near, far)``: the primary and the secondary difference, then the premiums at
    the end (USD per contract per unit of 4 m (1 - m))."""
    def difference(mean, below, above, se):
        return {"n_datasets": 10, "mean": mean, "mc_se": se, "min": mean - 1.0,
                "max": mean + 1.0, "below_zero": below, "above_zero": above, "half_width": 0.5}

    return {"windows": [
        {"window_hours": h, **difference(m, b, a, 0.1), "secondary": difference(sm, sb, sa, 0.2),
         "premium_pooled": pooled, "premium_matched": matched, "premium_near": near,
         "premium_far": far}
        for h, m, b, a, sm, sb, sa, pooled, matched, near, far in entries]}


def test_the_reading_of_the_window_table_follows_the_sign_and_says_it_is_exploratory():
    config = bs.StudyConfig()
    summary = fake_windows([(12.0, -0.5, 6, 0, -1.5, 9, 0, 0.028, 0.0165, 0.0895, 0.0088),
                            (30.0, 0.75, 0, 6, -0.25, 0, 3, 0.0195, 0.015, 0.09, 0.0075)])
    heading, reading, table, not_shown = bs.table_windows(summary, config).strip().split("\n\n")
    assert heading == ("### B8. Exploratory: `release_aware` minus `flat_matched` and "
                       "`release_aware` minus `pooled_premium` by hours of trading before each "
                       "release, planted world (synthetic data)")
    assert reading.startswith(
        "Reading: Exploratory: added after the first results because B1 and the rehearsal of the "
        "protocol analysis disagreed in sign for the first primary pair; it was not part of the "
        "design fixed beforehand. ")
    assert ("With 12 hours of trading before each release `release_aware` earned 0.50 USD per "
            "event less than `flat_matched` (the primary difference; mean of 10 datasets; the "
            "95% interval lies below zero in 6 of 10 datasets and above zero in 0) and "
            "`release_aware` earned 1.50 USD per event less than `pooled_premium` (the "
            "secondary difference; the 95% interval lies below zero in 9 of 10 datasets and "
            "above zero in 0).") in reading
    assert ("With 30 hours of trading before each release `release_aware` earned 0.75 USD per "
            "event more than `flat_matched` (the primary difference; mean of 10 datasets; the "
            "95% interval lies below zero in 0 of 10 datasets and above zero in 6) and "
            "`release_aware` earned 0.25 USD per event less than `pooled_premium` (the "
            "secondary difference; the 95% interval lies below zero in 0 of 10 datasets and "
            "above zero in 3).") in reading
    assert ("From 12 to 30 hours of trading the premiums estimated at the end move as follows: "
            "the pooled premium from 2.80 to 1.95 cents (-30%), the matched premium from 1.65 to "
            "1.50 cents (-9%), the premium of the bucket (0,5] from 8.95 to 9.00 cents (+1%) and "
            "the premium of the bucket >360 from 0.88 to 0.75 cents (-15%). Across the windows "
            "the primary difference does not keep one sign and the secondary difference stays "
            "negative.") in reading
    assert "10,000 replications" in reading and "\n" not in reading
    rows = table.splitlines()
    assert row_cells(rows[0]) == [
        "Hours of trading before each release", "Datasets",
        "Primary, `release_aware` minus `flat_matched`: mean difference, USD per event "
        "(MC SE over datasets)", "Primary: 95% interval below zero",
        "Primary: 95% interval above zero",
        "Secondary, `release_aware` minus `pooled_premium`: mean difference, USD per event "
        "(MC SE over datasets)", "Secondary: 95% interval below zero",
        "Secondary: 95% interval above zero", "Pooled premium at the end, cents",
        "Matched premium at the end, cents", "Premium of the bucket (0,5] at the end, cents",
        "Premium of the bucket >360 at the end, cents"]
    assert rows[2] == "| 12 | 10 | -0.50 (0.10) | 6 | 0 | -1.50 (0.20) | 9 | 0 | 2.80 | 1.65 " \
                      "| 8.95 | 0.88 |"
    assert rows[3] == "| 30 | 10 | +0.75 (0.10) | 0 | 6 | -0.25 (0.20) | 0 | 3 | 1.95 | 1.50 " \
                      "| 9.00 | 0.75 |"
    assert not_shown == (
        "Does not show: whether either difference follows the level of the premium only (B9 is "
        "the test of that), and anything about real markets: the hours of trading are those of "
        "the synthetic world.")


def test_the_reading_of_the_window_table_handles_no_difference_and_three_windows():
    config = bs.StudyConfig(boot_reps=500)
    block = bs.table_windows(fake_windows([
        (6.0, 0.0, 0, 0, 0.5, 0, 2, 0.01, 0.01, 0.05, 0.0),
        (12.0, -0.5, 1, 0, 0.0, 0, 0, 0.02, 0.015, 0.06, 0.01),
        (30.0, 2.0, 0, 3, 1.5, 0, 4, 0.03, 0.02, 0.07, 0.02)]), config)
    reading = next(line for line in block.splitlines() if line.startswith("Reading:"))
    assert ("With 6 hours of trading before each release `release_aware` and `flat_matched` "
            "earned the same per event (the primary difference; mean of 10 datasets") in reading
    assert ("and `release_aware` earned 0.50 USD per event more than `pooled_premium` (the "
            "secondary difference; the 95% interval lies below zero in 0 of 10 datasets and "
            "above zero in 2).") in reading
    assert ("With 12 hours of trading before each release `release_aware` earned 0.50 USD per "
            "event less than `flat_matched` (the primary difference; mean of 10 datasets; the "
            "95% interval lies below zero in 1 of 10 datasets and above zero in 0) and "
            "`release_aware` and `pooled_premium` earned the same per event (the secondary "
            "difference;") in reading
    assert ("the pooled premium from 1.00 to 3.00 cents (+200%), the matched premium from 1.00 "
            "to 2.00 cents (+100%), the premium of the bucket (0,5] from 5.00 to 7.00 cents "
            "(+40%) and the premium of the bucket >360 from 0.00 to 2.00 cents (n/a).") in reading
    assert ("Across the windows the primary difference does not keep one sign and the secondary "
            "difference does not keep one sign.") in reading  # a mean of 0.00 is not positive
    assert "500 replications" in reading
    one = bs.table_windows(fake_windows([(12.0, 1.0, 0, 1, 0.5, 0, 1, 0.02, 0.01, 0.06, 0.01)]),
                           config)
    assert ("The premiums estimated at the end: the pooled premium is 2.00 cents, the matched "
            "premium is 1.00 cents, the premium of the bucket (0,5] is 6.00 cents and the "
            "premium of the bucket >360 is 1.00 cents.") in one
    assert "From " not in one and "Across the windows" not in one
    positive = bs.table_windows(fake_windows([
        (12.0, 1.0, 0, 1, 0.5, 0, 1, 0.02, 0.01, 0.06, 0.01),
        (30.0, 2.0, 0, 3, 1.5, 0, 4, 0.03, 0.02, 0.07, 0.02)]), config)
    assert ("Across the windows the primary difference stays positive and the secondary "
            "difference stays positive.") in positive


def test_the_extra_windows_must_exist_and_differ_from_the_world_of_the_main_grid():
    with pytest.raises(ValueError, match="empty"):
        bs.window_runs(replace(SMALL, extra_windows=()))
    with pytest.raises(ValueError, match="differ"):
        bs.window_runs(replace(SMALL, extra_windows=(SMALL.world.window_hours,)))


def test_two_policies_give_the_rows_of_the_five_policy_run():
    params = replace(SMALL.world, pi1=0.2)
    _, two = bs.run_dataset(params, 4, SMALL.quote, bs.PRIMARY)
    _, five = bs.run_dataset(params, 4, SMALL.quote)
    assert sorted(two["policy"].unique()) == sorted(bs.PRIMARY) and len(two) == len(five) // 5 * 2
    key = ["event", "policy"]
    kept = five[five["policy"].isin(bs.PRIMARY)].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(two.sort_values(key).reset_index(drop=True), kept)
    three = bs.run_dataset(params, 4, SMALL.quote, bs.WINDOW_POLICIES)[1]
    assert len(three) == len(five) // 5 * 3
    kept = five[five["policy"].isin(bs.WINDOW_POLICIES)].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(three.sort_values(key).reset_index(drop=True), kept)


def test_the_runs_of_the_extra_windows_get_no_latent_columns_and_only_three_policies(
        monkeypatch):
    seen = []
    real = bs.walk_forward

    def spy(tape, markets, events, policies, params, **kwargs):
        seen.append((list(tape.columns), tuple(policies)))
        return real(tape, markets, events, policies, params, **kwargs)

    monkeypatch.setattr(bs, "walk_forward", spy)
    config = replace(SMALL, world=WorldParams(n_events=8, window_hours=2.0), seeds=(1, 2),
                     extra_windows=(3.0,), boot_reps=50)
    records, per_event = bs.window_runs(config)
    assert len(seen) == 2
    for columns, policies in seen:
        assert not any(c.startswith("latent_") for c in columns)
        assert policies == ("release_aware", "flat_matched", "pooled_premium")
    assert records[["window_hours", "seed"]].values.tolist() == [[3.0, 1], [3.0, 2]]
    assert set(per_event["policy"]) == set(bs.WINDOW_POLICIES)
    assert len(per_event) == 2 * 8 * 3


def test_the_window_records_equal_a_recomputation_from_the_per_event_windows_file(small):
    windows, per_event = read(small, "windows.csv"), read(small, "per_event_windows.csv")
    extra = windows[windows["window_hours"] == SMALL.extra_windows[0]]
    assert extra["seed"].tolist() == [1, 2, 3] and len(windows) == 2 * 3
    assert per_event.groupby(["window_hours", "seed", "policy"]).size().eq(30).all()
    assert sorted(per_event["policy"].unique()) == sorted(bs.WINDOW_POLICIES)
    assert per_event.groupby(["window_hours", "seed"])["evaluated"].sum().eq(3 * 25).all()
    secondary = ("secondary_mean", "secondary_lo", "secondary_hi")
    for row in extra.itertuples():
        part = per_event[(per_event["seed"] == row.seed) & per_event["evaluated"]]
        pnl = {p: part[part["policy"] == p].set_index("event")["pnl"].sort_index()
               for p in bs.WINDOW_POLICIES}
        # the primary difference with the key of the window and the secondary with one more
        for pair, key, columns in (
                (bs.PRIMARY, (bs.KIND_WINDOW, 0, row.seed), ("mean", "lo", "hi")),
                (bs.SECONDARY, (bs.KIND_WINDOW, 0, row.seed, 1), secondary)):
            d = (pnl[pair[0]] - pnl[pair[1]]).to_numpy()
            assert row.n_events == d.size == 25
            rng = np.random.default_rng([bs.SEED, *key])
            counts = rng.multinomial(25, np.full(25, 1 / 25), 300)
            lo, hi = np.quantile((counts * d).sum(axis=1) / 25, [0.025, 0.975])
            mean, got_lo, got_hi = (getattr(row, c) for c in columns)
            assert mean == pytest.approx(d.mean(), rel=1e-8)
            assert (got_lo, got_hi) == pytest.approx((lo, hi), rel=1e-7, abs=1e-9)
        assert row.mean != row.secondary_mean and row.lo != row.secondary_lo


def test_the_default_window_rows_are_the_planted_rows_of_the_pairs_file(small):
    windows, pairs = read(small, "windows.csv"), read(small, "pairs.csv")
    check = read(small, "premium_check.csv")
    main = windows[windows["window_hours"] == SMALL.world.window_hours].set_index("seed")
    planted = pairs[(pairs["sweep"] == "main") & (pairs["world"] == PLANTED)]
    assert main.index.tolist() == [1, 2, 3]
    fields = ("n_events", "mean", "lo", "hi")
    for pair, columns in ((bs.PRIMARY, fields),
                          (bs.SECONDARY, ("n_events", *bs.SECONDARY_COLUMNS))):
        rows = planted[(planted["a"] == pair[0]) & (planted["b"] == pair[1])].set_index("seed")
        for column, source in zip(columns, fields, strict=True):
            assert main[column].tolist() == rows.loc[main.index, source].tolist()
    for seed in main.index:
        end = check[(check["world"] == PLANTED) & (check["seed"] == seed)].set_index("bucket")
        buckets = ("pooled", "matched", BUCKET_LABELS[0], BUCKET_LABELS[-1])
        for column, bucket in zip(bs.WINDOW_PREMIUMS, buckets, strict=True):
            assert main.loc[seed, column] == pytest.approx(end.loc[bucket, "estimate_end"])


def test_the_premiums_of_an_extra_window_are_those_of_its_markouts(small):
    windows = read(small, "windows.csv")
    quote, hours, seed = SMALL.quote, SMALL.extra_windows[0], 2
    world = generate_world(replace(SMALL.world, pi1=SMALL.planted_pi1, window_hours=hours), seed)
    stripped = world.tape.drop(columns=[c for c in world.tape if c.startswith("latent_")])
    markouts = trade_markouts(stripped, world.events, world.markets, horizon_s=quote.horizon_s,
                              stale_s=quote.stale_s)
    last = world.events["event"].iloc[-1]
    end = estimate_premiums(markouts[markouts["event"] != last],
                            min_events=quote.min_history_events,
                            minutes=decision_minutes(quote))
    row = windows[(windows["window_hours"] == hours) & (windows["seed"] == seed)].iloc[0]
    assert row["premium_pooled"] == pytest.approx(end.pooled, rel=1e-7)
    assert row["premium_matched"] == pytest.approx(end.matched, rel=1e-7)
    # the matched premium is over the 120 decisions of the 2 hour quoting window, not the world's
    # 8 hours of trading
    by_hand = matched_by_hand(end.by_bucket, decision_weights(quote.window_hours))
    assert row["premium_matched"] == pytest.approx(by_hand, rel=1e-7)
    assert row["premium_near"] == pytest.approx(end.by_bucket[0], rel=1e-7)
    assert row["premium_far"] == pytest.approx(end.by_bucket[-1], rel=1e-7)
    assert (markouts["tau_min"] > 360.0).any()  # the world has trades in the last bucket
    assert end.by_bucket[-1] != end.pooled  # so that bucket has a premium of its own
    assert row["premium_matched"] != row["premium_pooled"]


def test_the_summary_and_the_table_carry_the_window_numbers(small):
    s = small["summary"]
    assert s["design"]["extra_windows"] == [8.0]
    short, long = s["windows"]
    assert (short["window_hours"], long["window_hours"]) == (2.0, 8.0)
    assert short["n_datasets"] == long["n_datasets"] == 3
    first = s["pairs"][f"{PLANTED}:release_aware-flat_matched"]  # B1's planted row is the 2 h row
    second = s["pairs"][f"{PLANTED}:release_aware-pooled_premium"]  # B3's is the secondary one
    for key in ("mean", "mc_se", "min", "max", "below_zero", "above_zero", "half_width"):
        assert short[key] == first[key]
        assert short["secondary"][key] == second[key]
    text = (small["out"] / "tables.md").read_text()
    for entry in (short, long):
        t = entry["secondary"]
        premiums = " | ".join(f"{100.0 * entry[k]:.2f}" for k in bs.WINDOW_PREMIUMS)
        assert (f"| {entry['window_hours']:g} | 3 | {entry['mean']:+.2f} ({entry['mc_se']:.2f}) | "
                f"{entry['below_zero']} | {entry['above_zero']} | {t['mean']:+.2f} "
                f"({t['mc_se']:.2f}) | {t['below_zero']} | {t['above_zero']} | {premiums} |"
                ) in text


def test_the_readme_parser_reads_the_nine_tables_of_the_study(small, tmp_path):
    (tmp_path / "backtest").mkdir()
    (tmp_path / "backtest" / "tables.md").write_text((small["out"] / "tables.md").read_text())
    tables = load_make_tables().read_tables(tmp_path)
    assert list(tables) == [f"B{i}" for i in range(1, 10)]
    parts = tables["B8"].split("\n\n")  # heading, Reading, table, Does not show
    assert len(parts) == 4 and parts[0].startswith("### B8. Exploratory: ")
    assert parts[0].endswith("(synthetic data)") and parts[1].startswith("Reading: Exploratory: ")
    assert len(parts[2].splitlines()) == 2 + 2 and parts[3].startswith("Does not show: ")
    levels = tables["B9"].split("\n\n")
    assert len(levels) == 4 and levels[0].startswith("### B9. Exploratory: ")
    assert levels[0].endswith("(synthetic data)") and levels[1].startswith("Reading: Exploratory: ")
    assert len(levels[2].splitlines()) == 2 + 3 and levels[3].startswith("Does not show: ")


# ------------------------------------------------------------ B9: the level sweep (exploratory)
def test_sign_changes_interpolate_linearly_between_the_two_grid_points_of_the_means():
    # the line through (1, 1) and (1.5, -3) reaches zero at 1 + 0.5 * 1 / (1 + 3) = 1.125
    assert bs.sign_changes([0.5, 1.0, 1.5], [2.0, 1.0, -3.0]) == [(pytest.approx(1.125), 1.0, 1.5)]
    # upwards: the line through (1, -1) and (2, 3) reaches zero at 1 + 1 * (-1) / (-1 - 3) = 1.25
    assert bs.sign_changes([1.0, 2.0], [-1.0, 3.0]) == [(pytest.approx(1.25), 1.0, 2.0)]
    assert bs.sign_changes([1.0, 2.0], [1.0, -1.0]) == [(pytest.approx(1.5), 1.0, 2.0)]
    assert bs.sign_changes([1.0, 2.0, 3.0], [1.0, -1.0, 1.0]) == [
        (pytest.approx(1.5), 1.0, 2.0), (pytest.approx(2.5), 2.0, 3.0)]


def test_sign_changes_finds_nothing_when_the_means_keep_their_sign():
    assert bs.sign_changes([0.5, 1.0, 1.5], [1.0, 2.0, 3.0]) == []
    assert bs.sign_changes([0.5, 1.0, 1.5], [-1.0, -2.0, -3.0]) == []
    assert bs.sign_changes([1.0], [5.0]) == [] and bs.sign_changes([], []) == []


def test_sign_changes_at_a_grid_point_with_a_mean_of_exactly_zero():
    # the mean is zero at F = 1 and has opposite signs on the two sides: a change at F = 1
    assert bs.sign_changes([0.5, 1.0, 1.5], [2.0, 0.0, -2.0]) == [(1.0, 0.5, 1.0)]
    assert bs.sign_changes([0.5, 1.0, 1.5], [-2.0, 0.0, 2.0]) == [(1.0, 0.5, 1.0)]
    # a run of zeros: one change, at the first grid point with a zero mean
    assert bs.sign_changes([1.0, 2.0, 3.0, 4.0], [1.0, 0.0, 0.0, -1.0]) == [(2.0, 1.0, 2.0)]
    # a mean that only touches zero is not a change, nor is a zero at either end of the grid
    assert bs.sign_changes([0.5, 1.0, 1.5], [1.0, 0.0, 1.0]) == []
    assert bs.sign_changes([0.5, 1.0, 1.5], [-1.0, 0.0, -1.0]) == []
    assert bs.sign_changes([0.5, 1.0], [1.0, 0.0]) == []
    assert bs.sign_changes([0.5, 1.0], [0.0, 1.0]) == []


def test_level_summary_by_hand():
    levels = pd.DataFrame({
        "flat_scale": [2.0, 2.0, 1.0, 1.0, 1.0], "seed": [1, 2, 1, 2, 3], "n_events": 180,
        "mean": [-1.0, -3.0, 1.0, 3.0, 2.0], "lo": [-2.0, -4.0, 0.5, -1.0, 1.5],
        "hi": [-0.5, -2.0, 1.5, 7.0, 2.5],
        "premium_flat": [0.08, 0.10, 0.04, 0.05, 0.09],
        "premium_pooled": [0.03, 0.05, 0.03, 0.05, 0.10],
    })
    one, two = bs.level_summary(levels)  # increasing F, whatever the order of the rows
    assert (one["flat_scale"], two["flat_scale"]) == (1.0, 2.0)
    assert (one["n_datasets"], two["n_datasets"]) == (3, 2) and one["mean"] == 2.0
    assert one["mc_se"] == pytest.approx(1.0 / math.sqrt(3))  # sd 1 of 1, 3, 2
    assert (one["min"], one["max"]) == (1.0, 3.0)
    assert (one["below_zero"], one["above_zero"]) == (0, 2)  # the second interval has zero in it
    assert one["half_width"] == pytest.approx(np.mean([0.5, 4.0, 0.5]))
    assert two["mean"] == -2.0 and two["mc_se"] == pytest.approx(1.0)  # sd sqrt(2) of -1, -3
    assert (two["below_zero"], two["above_zero"]) == (2, 0)
    assert two["half_width"] == pytest.approx(0.875)
    # means over the datasets (the medians would be 0.05 and 0.05 for F = 1)
    assert (one["premium_flat"], one["premium_pooled"]) == pytest.approx((0.06, 0.06))
    assert (two["premium_flat"], two["premium_pooled"]) == pytest.approx((0.09, 0.04))


def fake_levels(means, losing=5, pooled=0.03, matched=0.02):
    """A summary with the B9 entries of ``{F: (mean, intervals below zero, intervals above)}``,
    whose flat premium quoted is F times ``matched``, and the net P&L per event of the five
    policies in the planted world, ``losing`` of them negative."""
    levels = [{"flat_scale": f, "n_datasets": 10, "mean": m, "mc_se": 0.2, "min": m - 1.0,
               "max": m + 1.0, "below_zero": b, "above_zero": a, "half_width": 0.5,
               "premium_flat": f * matched, "premium_pooled": pooled}
              for f, (m, b, a) in means.items()]
    pnl = [-1.0] * losing + [1.0] * (len(POLICIES) - losing)
    return {"levels": levels,
            "policies": {f"{PLANTED}:{p}": {"pnl": v} for p, v in zip(POLICIES, pnl, strict=True)}}


def test_the_reading_of_the_level_table_gives_the_ends_the_sign_change_and_the_premium_ratio():
    config = bs.StudyConfig()
    grid = {0.5: (4.0, 0, 10), 0.75: (2.5, 0, 9), 1.0: (1.0, 0, 3), 1.25: (-0.5, 1, 0),
            1.5: (-2.0, 6, 0), 1.75: (-3.0, 9, 0), 2.0: (-3.5, 10, 0)}
    heading, reading, table, not_shown = bs.table_levels(
        fake_levels(grid), config).strip().split("\n\n")
    assert heading == ("### B9. Exploratory: `release_aware` minus `flat_matched` by the scale of "
                       "the flat premium, planted world (synthetic data)")
    assert reading == (
        "Reading: Exploratory: added after an independent review showed that the first primary "
        "pair followed the level of the pooled premium. The flat premium of `flat_matched` is "
        "multiplied by F and F = 1 is the primary comparison. At F = 1 `release_aware` earned "
        "1.00 USD per event more than `flat_matched` (mean of 10 datasets; the 95% interval "
        "lies below zero in 0 of 10 datasets and above zero in 3). At the largest F = 2 "
        "`release_aware` earned 3.50 USD per event less than `flat_matched` (mean of 10 "
        "datasets; the 95% interval lies below zero in 10 of 10 datasets and above zero in 0). "
        "The mean difference changes sign at F = 1.17, by linear interpolation between the means "
        "at F = 1 and F = 1.25; this describes the grid and is not an estimate with an interval. "
        "The mean pooled premium at the end is 3.00 cents and the mean matched premium 2.00 "
        "cents, a ratio of 1.50, so at the end of a dataset `pooled_premium` quotes the premium "
        "that `flat_matched` quotes at F = 1.50. Every policy loses money in the planted world "
        "(B2), so a wider quote loses less. Intervals are percentile intervals of a cluster "
        "bootstrap over events with 10,000 replications.")
    rows = table.splitlines()
    assert row_cells(rows[0]) == [
        "Scale F of the flat premium", "Flat premium quoted at the end, cents",
        "`release_aware` minus `flat_matched`: mean difference, USD per event "
        "(MC SE over datasets)", "Range over datasets", "95% interval below zero",
        "95% interval above zero"]
    assert len(rows) == 2 + 7
    assert row_cells(rows[2]) == ["0.5", "1.00", "+4.00 (0.20)", "+3.00 to +5.00", "0", "10"]
    assert row_cells(rows[4]) == ["1", "2.00", "+1.00 (0.20)", "+0.00 to +2.00", "0", "3"]
    assert row_cells(rows[8]) == ["2", "4.00", "-3.50 (0.20)", "-4.50 to -2.50", "10", "0"]
    assert not_shown == (
        "Does not show: that either policy earns money on real markets, and that the planted "
        "informed share is what makes the bucket premiums useful at F = 1: the null world has "
        "buckets that differ too (see the null row of B1).")


def test_the_reading_of_the_level_table_without_a_sign_change_or_a_level_of_one():
    config = bs.StudyConfig(boot_reps=500)
    block = bs.table_levels(fake_levels({1.5: (2.0, 0, 8), 2.0: (3.0, 0, 9)}, losing=4), config)
    reading = next(line for line in block.splitlines() if line.startswith("Reading:"))
    assert "At F = 1 " not in reading  # the grid has no F = 1
    assert ("At the largest F = 2 `release_aware` earned 3.00 USD per event more than "
            "`flat_matched` (mean of 10 datasets; the 95% interval lies below zero in 0 of 10 "
            "datasets and above zero in 9).") in reading
    assert "The mean difference does not change sign on the grid of F from 1.5 to 2." in reading
    assert "so at the end of a dataset `pooled_premium` quotes the premium that `flat_matched` " \
           "quotes at F = 1.50." in reading
    assert "Net P&L per event is negative for 4 of 5 policies in the planted world (B2)." in reading
    assert "so a wider quote loses less" not in reading and "500 replications" in reading
    only = bs.table_levels(fake_levels({0.5: (2.0, 0, 5), 1.0: (1.0, 0, 3)}), config)
    assert "At F = 1 `release_aware` earned 1.00 USD per event more" in only
    assert "At the largest" not in only  # F = 1 is the largest F


def test_the_reading_of_the_level_table_names_the_first_of_several_sign_changes_and_a_zero_mean():
    config = bs.StudyConfig()
    wavy = bs.table_levels(fake_levels({0.5: (1.0, 0, 3), 1.0: (-1.0, 3, 0), 1.5: (1.0, 0, 3)}),
                           config)
    assert ("The mean difference changes sign at F = 0.75 (the first of 2 sign changes), by "
            "linear interpolation between the means at F = 0.5 and F = 1;") in wavy
    zero = bs.table_levels(fake_levels({0.5: (1.0, 0, 3), 1.0: (0.0, 0, 0), 1.5: (-1.0, 3, 0)}),
                           config)
    assert "At F = 1 `release_aware` and `flat_matched` earned the same per event" in zero
    assert "The mean difference changes sign at F = 1.00, by linear interpolation" in zero


def independent_pnl(world, policy, quote):
    """Net P&L per evaluated event of one policy in release order, from the rows of the
    backtest: the sum over the markets of an event."""
    tape = world.tape.drop(columns=[c for c in world.tape if c.startswith("latent_")])
    rows = walk_forward(tape, world.markets, world.events, (policy,), quote)
    done = rows[rows["evaluated"]]
    per_event = done.groupby("event").agg(pnl=("pnl", "sum"), when=("release_ts", "first"))
    return per_event.sort_values("when")["pnl"]


def test_the_levels_file_has_one_row_per_scale_and_dataset(small):
    levels = read(small, "levels.csv")
    assert levels.columns.tolist() == ["flat_scale", "seed", "n_events", "mean", "lo", "hi",
                                       "premium_flat", "premium_pooled"]
    assert levels[["flat_scale", "seed"]].values.tolist() == [
        [f, seed] for seed in (1, 2, 3) for f in (0.5, 1.0, 2.0)]
    assert (levels["n_events"] == 25).all()
    assert ((levels["lo"] <= levels["mean"]) & (levels["mean"] <= levels["hi"])).all()
    assert (levels["premium_flat"] > 0.0).all() and (levels["premium_pooled"] > 0.0).all()


def test_the_levels_file_equals_a_recomputation_from_independent_runs(small):
    levels, check = read(small, "levels.csv"), read(small, "premium_check.csv")
    params = replace(SMALL.world, pi1=SMALL.planted_pi1)
    worlds = {seed: generate_world(params, seed) for seed in (1, 2, 3)}
    for row in levels.itertuples():
        world = worlds[row.seed]
        aware = independent_pnl(world, "release_aware", SMALL.quote)
        flat = independent_pnl(world, "flat_matched",
                               replace(SMALL.quote, flat_scale=row.flat_scale))
        d = (aware - flat).to_numpy()  # a minus b
        assert row.n_events == d.size == 25 and row.mean == pytest.approx(d.mean(), rel=1e-8)
        index = [0.5, 1.0, 2.0].index(row.flat_scale)  # the generator is keyed by the index of F
        counts = np.random.default_rng([bs.SEED, bs.KIND_LEVEL, index, row.seed]).multinomial(
            25, np.full(25, 1 / 25), 300)
        lo, hi = np.quantile((counts * d).sum(axis=1) / 25, [0.025, 0.975])
        assert (row.lo, row.hi) == pytest.approx((lo, hi), rel=1e-7, abs=1e-9)
        end = check[(check["world"] == PLANTED) & (check["seed"] == row.seed)].set_index("bucket")
        # the flat premium quoted is F times the matched premium at the end, in USD per contract
        # per unit of 4 m (1 - m); the pooled premium is there for reference
        assert row.premium_flat == pytest.approx(
            row.flat_scale * end.loc["matched", "estimate_end"], rel=1e-8)
        assert row.premium_pooled == pytest.approx(end.loc["pooled", "estimate_end"], rel=1e-8)


def test_the_level_sweep_at_one_is_the_planted_row_of_the_main_grid(small):
    levels, pairs = read(small, "levels.csv"), read(small, "pairs.csv")
    one = levels[levels["flat_scale"] == 1.0].set_index("seed")
    main = pairs[(pairs["sweep"] == "main") & (pairs["world"] == PLANTED)
                 & (pairs["a"] == "release_aware") & (pairs["b"] == "flat_matched")]
    main = main.set_index("seed")
    assert one.index.tolist() == main.index.tolist() == [1, 2, 3]
    # the same events and the same mean: one run of flat_matched, one of release_aware
    assert one["n_events"].tolist() == main["n_events"].tolist()
    np.testing.assert_allclose(one["mean"], main["mean"], rtol=1e-9)
    # the bootstrap counts come from another generator key, so the interval ends agree only
    # up to the Monte Carlo error of the 300 replications
    width = main["hi"] - main["lo"]
    assert ((one["lo"] - main["lo"]).abs() < 0.25 * width).all()
    assert ((one["hi"] - main["hi"]).abs() < 0.25 * width).all()
    assert (one["lo"] != main["lo"]).any()


def test_a_larger_flat_scale_quotes_a_larger_premium_and_fills_less(small):
    levels = read(small, "levels.csv")
    for _, part in levels.groupby("seed"):
        part = part.sort_values("flat_scale")
        assert part["premium_flat"].is_monotonic_increasing and part["premium_flat"].is_unique
        np.testing.assert_allclose(part["premium_flat"] / part["flat_scale"],
                                   part["premium_flat"].iloc[0] / 0.5, rtol=1e-8)
    world = generate_world(replace(SMALL.world, pi1=SMALL.planted_pi1), 2)
    fills = {f: float(bs.walk_world(world, ("flat_matched",),
                                    replace(SMALL.quote, flat_scale=f))["fills"].sum())
             for f in (0.5, 1.0, 2.0)}
    assert fills[0.5] > fills[1.0] > fills[2.0]  # a wider quote is hit less often


def test_the_level_runs_walk_release_aware_once_and_flat_matched_once_per_scale(monkeypatch):
    seen = []
    real = bs.walk_forward

    def spy(tape, markets, events, policies, params, **kwargs):
        seen.append((list(tape.columns), tuple(policies), params.flat_scale))
        return real(tape, markets, events, policies, params, **kwargs)

    monkeypatch.setattr(bs, "walk_forward", spy)
    config = replace(SMALL, world=WorldParams(n_events=8, window_hours=2.0), seeds=(1, 2),
                     sweep_levels=(0.5, 2.0), boot_reps=50)
    levels = bs.level_runs(config)
    assert [(policies, scale) for _, policies, scale in seen] == [
        (("release_aware",), 1.0), (("flat_matched",), 0.5), (("flat_matched",), 2.0)] * 2
    assert not any(c.startswith("latent_") for columns, _, _ in seen for c in columns)
    assert levels[["flat_scale", "seed"]].values.tolist() == [[0.5, 1], [2.0, 1], [0.5, 2],
                                                              [2.0, 2]]
    assert (levels["n_events"] == 3).all()  # 8 events, 5 of them burn-in


def test_the_levels_must_be_a_grid_without_repeats():
    with pytest.raises(ValueError, match="empty"):
        bs.level_runs(replace(SMALL, sweep_levels=()))
    with pytest.raises(ValueError, match="repeat"):
        bs.level_runs(replace(SMALL, sweep_levels=(1.0, 2.0, 1.0)))


def test_the_summary_of_the_levels_equals_the_levels_file(small):
    levels, entries = read(small, "levels.csv"), small["summary"]["levels"]
    assert [e["flat_scale"] for e in entries] == [0.5, 1.0, 2.0]
    for entry in entries:
        part = levels[levels["flat_scale"] == entry["flat_scale"]]
        assert entry["n_datasets"] == 3 and entry["mean"] == pytest.approx(part["mean"].mean())
        assert entry["mc_se"] == pytest.approx(part["mean"].std(ddof=1) / math.sqrt(3))
        assert entry["below_zero"] == int((part["hi"] < 0.0).sum())
        assert entry["above_zero"] == int((part["lo"] > 0.0).sum())
        assert entry["premium_flat"] == pytest.approx(part["premium_flat"].mean(), rel=1e-8)
        assert entry["premium_pooled"] == pytest.approx(part["premium_pooled"].mean(), rel=1e-8)
