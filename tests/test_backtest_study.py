"""The backtest study: its statistics against hand and independent computations, and its files."""

import json
import math
import re
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from kalshi_mm.experiments import backtest_study as bs
from kalshi_mm.experiments.common import REPO, Manifest
from kalshi_mm.markout import BUCKET_LABELS, trade_markouts
from kalshi_mm.quoting import POLICIES, QuoteParams, estimate_premiums
from kalshi_mm.stats import cluster_weights
from kalshi_mm.world import WorldParams, generate_world

NAN = float("nan")
NULL, PLANTED = bs.NULL_WORLD, bs.PLANTED_WORLD


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
    assert bs.PRIMARY == ("release_aware", "pooled_premium") == bs.PAIRS[0]
    assert len(set(bs.PAIRS)) == 6 == len({frozenset(p) for p in bs.PAIRS})
    assert {p for pair in bs.PAIRS for p in pair} == set(POLICIES)


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
    pnl = {"always_on": base, "pull_only": base + 0.5,
           "pooled_premium": base + 3.0 + rng.normal(0.0, 0.1, 14),
           "release_aware": base + 4.0 + rng.normal(0.0, 0.1, 14)}
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
    assert 0.5 < primary["mean"] < 1.5 and primary["lo"] > 0.0  # release_aware is ahead by 1
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
        events = make_events({"release_aware": diffs, "pooled_premium": np.zeros(diffs.size)})
        frames.append(events.assign(world=PLANTED, seed=seed))
    other = make_events({"release_aware": [99.0], "pooled_premium": [-99.0]})
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
    assert "In the null world `release_aware` earned 1.50 USD per event less than " in reading
    assert "In the planted world `release_aware` earned 2.25 USD per event more than " in reading
    assert "below zero in 7 of 10 datasets and above zero in 1" in reading
    assert "-1.50 (0.10)" in block and "+2.25 (0.10)" in block
    flat = bs.table_primary(fake_summary({NULL: 0.0, PLANTED: 0.0}), config)
    assert "null world `release_aware` and `pooled_premium` earned the same per event" in flat


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
    assert names == ["backtest_study.manifest.txt", "events_needed.csv", "pairs.csv",
                     "per_event.csv", "policies.csv", "premium_check.csv", "summary.json",
                     "tables.md"]
    manifest = (small["out"] / "backtest_study.manifest.txt").read_text()
    assert "command: python -m kalshi_mm.experiments.backtest_study --out results/small" in manifest
    assert "seeds: test" in manifest and "bootstrap: 300 replications" in manifest
    assert "dataset seeds 1 to 3" in manifest and str(REPO) not in manifest
    assert json.loads((small["out"] / "summary.json").read_text()) == small["summary"]


def test_every_dataset_has_the_evaluated_events_and_all_policies(small):
    per_event = read(small, "per_event.csv")
    assert per_event.groupby(["world", "seed"]).size().eq(30 * 4).all()
    assert sorted(per_event["world"].unique()) == [NULL, PLANTED]
    evaluated = per_event[per_event["evaluated"]]
    assert evaluated.groupby(["world", "seed", "policy"]).size().eq(25).all()  # 30 - 5 burn-in
    first = per_event[per_event["event"] == "SYN-000000"]
    assert not first["evaluated"].any()
    assert per_event["pnl"].notna().all()


def test_the_pairs_file_equals_a_recomputation_from_the_per_event_file(small):
    per_event, pairs = read(small, "per_event.csv"), read(small, "pairs.csv")
    main = pairs[pairs["sweep"] == "main"]
    assert len(main) == 2 * 3 * 6
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
    assert len(main) == 2 * 3 * 4
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
    assert len(cells) == 2 * 2 * 4  # two planted shares, two fees, four policies
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
        "B1", "B2", "B3", "B4", "B5", "B6", "B7"]
    assert all(h.endswith(" (synthetic data)") for h in headings)
    blocks = re.split(r"(?m)^(?=### )", text.strip() + "\n")
    blocks = [b.strip() for b in blocks if b.strip()]
    assert len(blocks) == 7
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
    primary = s["pairs"][f"{PLANTED}:release_aware-pooled_premium"]
    assert f"{primary['mean']:+.2f} ({primary['mc_se']:.2f})" in text
    cell = s["sweeps"]["pi1_fee"][0]
    assert (f"{cell['primary']['mean']:+.2f} [{cell['primary']['lo']:+.2f}, "
            f"{cell['primary']['hi']:+.2f}]") in text
    assert f"{s['policies'][f'{NULL}:always_on']['pnl']:.2f}" in text
    assert s["design"]["bootstrap_replications"] == 300 and s["design"]["burn_in_events"] == 5
    assert len(s["events_needed"]) == 2 and len(s["sweeps"]["pi1_fee"]) == 4
    assert len(s["sweeps"]["touch"]) == 1 and s["sweeps"]["touch"][0]["rule"] == "touch"


def test_a_second_run_gives_the_same_files(small, tmp_path):
    manifest = Manifest("kalshi_mm.experiments.backtest_study", "test", ["--out", "x"])
    bs.run_study(SMALL, tmp_path, manifest)
    for name in ("per_event.csv", "policies.csv", "pairs.csv", "events_needed.csv",
                 "premium_check.csv", "summary.json", "tables.md"):
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
        assert got.index.tolist() == [*BUCKET_LABELS, "pooled"]
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
    assert check["estimate_end"].notna().all() and (check["estimate_end"] >= 0.0).all()
    assert check["target"].isna().sum() > 0  # the 2 hour windows leave the far buckets empty


def fake_sweeps(through, touch, planted=0.2, fee=0.0175):
    def cell(pi1, rule, pnl):
        primary = {"mean": 0.5, "lo": -0.5, "hi": 1.5}
        return {"pi1": pi1, "fee": fee, "rule": rule, "primary": primary, "pnl": pnl}

    cells = [cell(0.0, "through", through), cell(planted, "through", through)]
    return {"sweeps": {"pi1_fee": cells, "touch": [cell(planted, "touch", touch)]}}


def test_the_reading_of_the_touch_table_names_the_largest_change_and_its_direction():
    config = bs.StudyConfig()
    through = dict(zip(POLICIES, [-10.0, -9.0, -5.0, -6.0], strict=True))
    touch = dict(zip(POLICIES, [-9.0, -8.5, -8.0, -5.5], strict=True))  # pooled_premium: -3.0
    reading = bs.table_touch(fake_sweeps(through, touch), config)
    assert "lowers net P&L per event by 3.00 USD for `pooled_premium`, the largest" in reading
    up = dict(zip(POLICIES, [-4.0, -9.0, -5.0, -6.0], strict=True))  # always_on: +6.0
    assert "raises net P&L per event by 6.00 USD for `always_on`" in bs.table_touch(
        fake_sweeps(through, up), config)


def test_the_reading_of_the_policy_table_counts_the_losing_cells():
    config = bs.StudyConfig()
    values = {f"{w}:{p}": {"pnl": v, "pnl_mc_se": 0.1, "fees": 1.0, "fills": 10.0,
                           "contracts": 50.0, "cents_per_contract": 0.5, "marked_pnl": v,
                           "live_share": 0.9}
              for w, vs in ((NULL, [-3.0, -2.0, 1.0, 4.0]), (PLANTED, [-8.0, -1.0, -2.0, 0.5]))
              for p, v in zip(POLICIES, vs, strict=True)}
    block = bs.table_policies({"policies": values}, config)
    assert "negative in 5 of 8 world and policy cells" in block
    assert "the highest is `release_aware` in the null world (+4.00 USD)" in block
    assert "the lowest `always_on` in the planted world (-8.00 USD)" in block


def test_the_reading_of_the_events_needed_table_handles_a_zero_effect():
    config = bs.StudyConfig()
    rows = [{"n": 10, "half_width_first_seed": 2.0, "half_width_mean": 2.5, "half_width_min": 2.0,
             "half_width_max": 3.0, "excludes_zero": 1, "n_datasets": 10}]
    size = {"mean": 0.0, "sd": 3.0, "events_half_width": None, "events_power_80": None}
    block = bs.table_events_needed({"events_needed": rows, "sample_size": size}, config)
    assert "so no number of events is given" in block
    size = {"mean": -1.0, "sd": 3.0, "events_half_width": 34.57, "events_power_80": 70.6}
    block = bs.table_events_needed({"events_needed": rows, "sample_size": size}, config)
    assert "about 35 evaluated events" in block and "about 71 would give" in block
    assert "(-1.00 USD per event)" in block
