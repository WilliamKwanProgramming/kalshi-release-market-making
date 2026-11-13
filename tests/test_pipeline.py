"""The whole pipeline on a small synthetic world: database, protocol, tables, runners.

docs/protocol.md asks that this file passes on the machine that runs the analysis. The world is
small (26 events of 8 hours) so that the file runs in well under a minute; the settings below
shorten the burn-in, the recalibration price time and the bootstrap, and every one of them is a
setting the protocol fixes to another value, so it is listed as a deviation. What is checked:

* the path through SQLite gives the same numbers as the frames themselves;
* each tested quantity equals a recomputation from the library functions;
* shuffling the rows of the input changes nothing;
* a small sample is labelled exploratory;
* tables carry the letter and label of their data, and synthetic results cannot land in a
  directory named ``real``;
* ``run_real`` refuses a synthetic database, reads a database read-only and writes its
  departures from the protocol.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalshi_mm import run_real
from kalshi_mm.backtest import walk_forward
from kalshi_mm.calibration import skill_interval, walk_forward_recalibration
from kalshi_mm.markout import release_effect, scaled_adverse_selection, trade_markouts
from kalshi_mm.panel import create_schema, get_source, read_dataset, write_dataset
from kalshi_mm.protocol import (
    DEFAULT,
    PAIRS,
    PRIMARY_PAIR,
    Results,
    p2_pricing,
    paired_mean,
    prepare,
    run_protocol,
    with_overrides,
)
from kalshi_mm.protocol_report import tables_markdown, write_results
from kalshi_mm.quoting import POLICIES
from kalshi_mm.world import WorldParams, generate_world

SMALL = [
    "n_boot=200",
    "recal_min_releases=12",
    "recal_refit_every=4",
    "recal_hours=2",
    "quote.min_history_events=5",
]


@pytest.fixture(scope="module")
def world():
    return generate_world(WorldParams(n_events=26, window_hours=8.0), seed=3)


def _database(path: Path, world, source: str) -> Path:
    conn = sqlite3.connect(path)
    try:
        create_schema(conn)
        write_dataset(conn, world.tape, world.markets, world.events, source=source)
        conn.commit()
    finally:
        conn.close()
    return path


@pytest.fixture(scope="module")
def run(world, tmp_path_factory):
    """The protocol on the frames read back from a database, with the shortened settings."""
    path = _database(tmp_path_factory.mktemp("db") / "synthetic.sqlite", world, "synthetic")
    conn = sqlite3.connect(path)
    try:
        source = get_source(conn)
        tape, markets, events = read_dataset(conn)
    finally:
        conn.close()
    settings = with_overrides(DEFAULT, SMALL)
    results = run_protocol(tape, markets, events, source, settings)
    return results, (tape, markets, events), settings


def test_the_database_path_gives_the_numbers_of_the_frames(world, run):
    results, _, settings = run
    direct = run_protocol(world.tape, world.markets, world.events, "synthetic", settings)
    assert results.dataset.counts == direct.dataset.counts
    assert results.p1["primary"] == direct.p1["primary"]
    assert results.p2["primary"] == direct.p2["primary"]
    assert results.p3["primary"] == direct.p3["primary"]
    assert results.family == direct.family


def test_the_sample_counts_are_consistent(run):
    results, (tape, markets, events), _ = run
    counts = results.dataset.counts
    assert counts["events"] == len(events) == 26 and counts["markets"] == len(markets)
    assert counts["markets kept"] == len(results.dataset.markets)
    assert counts["trades kept"] == len(results.dataset.tape) <= len(tape)
    removed = sum(v for k, v in counts.items() if "without" in k or "fewer" in k)
    assert counts["markets"] - removed == counts["markets kept"]


def test_p1_is_the_release_effect_of_the_markouts(run):
    results, _, settings = run
    ds = results.dataset
    markouts = trade_markouts(ds.tape, ds.events, ds.markets, horizon_s=settings.horizon_s,
                              stale_s=settings.quote.stale_s)
    expected = release_effect(markouts, near=settings.near, quiet=settings.quiet,
                              B=settings.n_boot, seed=settings.seed, level=settings.level)
    assert results.p1["primary"] == expected
    assert results.p1["n_trades"] == len(markouts)
    assert results.p1["buckets"]["n"].sum() <= len(markouts)


def _effect_again(markouts, settings, column="adverse_selection", near=None):
    return release_effect(markouts, near=near or settings.near, quiet=settings.quiet,
                          column=column, B=settings.n_boot, seed=settings.seed,
                          level=settings.level)


def test_the_reported_effects_of_p1_are_recomputed_from_the_markouts(run):
    results, _, settings = run
    ds, q = results.dataset, settings.quote
    markouts = trade_markouts(ds.tape, ds.events, ds.markets, horizon_s=settings.horizon_s,
                              stale_s=q.stale_s)
    p1 = results.p1
    assert p1["settlement"] == _effect_again(markouts, settings, "settlement_markout")
    scaled = markouts.assign(adverse_selection_scaled=scaled_adverse_selection(
        markouts, min_mid=q.min_mid, max_mid=q.max_mid))
    assert p1["scaled"] == _effect_again(scaled, settings, "adverse_selection_scaled")
    uncut_near = (settings.horizon_s / 60.0, settings.near[1])
    assert p1["uncut"] == _effect_again(markouts, settings, near=uncut_near)
    assert set(p1["horizons"]) == set(settings.extra_horizons_s)
    for h, effect in p1["horizons"].items():
        other = trade_markouts(ds.tape, ds.events, ds.markets, horizon_s=h, stale_s=q.stale_s)
        assert effect == _effect_again(other, settings)
    series = markouts["market"].map(ds.market_series)
    assert set(p1["by_series"]) == set(series)
    for name, effect in p1["by_series"].items():
        assert effect == _effect_again(markouts[series == name], settings)


def test_the_policies_and_the_primary_pair_are_those_of_the_protocol():
    assert PRIMARY_PAIR == ("release_aware", "pooled_premium")
    assert PAIRS[0] == PRIMARY_PAIR and len(set(PAIRS)) == len(PAIRS) == 6
    assert set(POLICIES) == {"always_on", "pull_only", "pooled_premium", "release_aware"}
    assert all(a in POLICIES and b in POLICIES and a != b for a, b in PAIRS)


def test_p2_is_the_paired_difference_of_the_policy_pnl(run):
    results, _, settings = run
    ds = results.dataset
    frame = walk_forward(ds.tape, ds.markets, ds.events, POLICIES, settings.quote)
    assert set(frame["policy"]) == set(POLICIES)
    evaluated = frame[frame["evaluated"]]
    by_release = evaluated.groupby(["event", "policy"])["pnl"].sum().unstack("policy")
    diff = by_release[PRIMARY_PAIR[0]] - by_release[PRIMARY_PAIR[1]]
    assert results.p2["primary"] == paired_mean(diff.to_numpy(), settings)
    assert results.p2["n_releases"] == len(diff) > 0
    # the releases of the burn-in are run but not compared
    assert len(diff) == ds.counts["releases kept"] - settings.quote.min_history_events


def _per_release(frame, column):
    evaluated = frame[frame["evaluated"]]
    return evaluated.groupby(["event", "policy"])[column].sum().unstack("policy").sort_index()


def test_the_reported_parts_of_p2_are_recomputed_from_the_walk_forward_frame(run):
    results, _, settings = run
    ds, p2 = results.dataset, results.p2
    frame = p2["frame"]
    pnl = _per_release(frame, "pnl")
    for a, b in PAIRS:
        assert p2["pairs"][f"{a} - {b}"] == paired_mean((pnl[a] - pnl[b]).to_numpy(), settings)
    columns = ("pnl", "gross", "fees", "fills", "contracts", "marked_pnl")
    for policy in POLICIES:
        for column in columns:
            expected = _per_release(frame, column)[policy].mean()
            assert p2["totals"][policy][column] == pytest.approx(expected)
        live = frame.loc[frame["evaluated"] & (frame["policy"] == policy), "quotes_live_share"]
        assert p2["totals"][policy]["quotes_live_share"] == pytest.approx(live.mean())
    series = frame["market"].map(ds.market_series)
    assert set(p2["by_series"]) == set(series)
    for name, effect in p2["by_series"].items():
        part = _per_release(frame[series == name], "pnl")
        diff = part[PRIMARY_PAIR[0]] - part[PRIMARY_PAIR[1]]
        assert effect == paired_mean(diff.to_numpy(), settings)


def test_the_halves_of_p2_split_the_releases_in_time_at_the_median(run):
    results, _, settings = run
    ds, p2 = results.dataset, results.p2
    pnl = _per_release(p2["frame"], "pnl")
    diff = pnl[PRIMARY_PAIR[0]] - pnl[PRIMARY_PAIR[1]]
    when = ds.events.set_index("event")["release_ts"].loc[diff.index].sort_values(kind="mergesort")
    ordered = diff.loc[when.index].to_numpy()
    n = len(ordered)
    cut = (n + 1) // 2  # the release at the median time belongs to the first half
    assert p2["halves"]["first half"] == paired_mean(ordered[:cut], settings)
    assert p2["halves"]["second half"] == paired_mean(ordered[cut:], settings)


def test_the_sensitivity_runs_of_p2_change_one_setting_of_the_quote_parameters(run):
    results, _, settings = run
    ds, p2 = results.dataset, results.p2
    variants = {
        "touch fill rule": dataclasses.replace(settings.quote, fill_rule="touch"),
        "maker fee coefficient 0": dataclasses.replace(settings.quote, maker_fee_coeff=0.0),
    }
    assert set(p2["sensitivity"]) == set(variants)
    for name, params in variants.items():
        frame = walk_forward(ds.tape, ds.markets, ds.events, PRIMARY_PAIR, params)
        pnl = _per_release(frame, "pnl")
        diff = pnl[PRIMARY_PAIR[0]] - pnl[PRIMARY_PAIR[1]]
        assert p2["sensitivity"][name] == paired_mean(diff.to_numpy(), settings)


def _paired(world):
    """The world with its events in pairs: the second event of a pair is moved to the release
    time of the first, so that the two merge into one release with two markets."""
    tape, markets, events = world.tape.copy(), world.markets.copy(), world.events.copy()
    release = events.set_index("event")["release_ts"].to_dict()
    ids = list(events["event"])
    for first, second in zip(ids[0::2], ids[1::2], strict=False):
        shift = release[first] - release[second]
        tape.loc[tape["event"] == second, "ts"] += shift
        markets.loc[markets["event"] == second, ["open_ts", "close_ts"]] += shift
        events.loc[events["event"] == second, "release_ts"] = release[first]
    return tape, markets, events


def test_a_release_with_two_markets_adds_up_the_profit_of_its_markets(world):
    settings = with_overrides(DEFAULT, SMALL)
    ds = prepare(*_paired(world), settings)
    assert ds.counts["events"] == 26 and ds.counts["releases"] == 13
    p2 = p2_pricing(ds, settings)
    frame = p2["frame"]
    assert (frame.groupby(["event", "policy"]).size() == 2).all()
    evaluated = frame[frame["evaluated"]]
    total, average = {}, {}
    for (event, policy), part in evaluated.groupby(["event", "policy"]):
        total[event, policy] = sum(part["pnl"])
        average[event, policy] = sum(part["pnl"]) / len(part)
    events = sorted({event for event, _ in total})
    a, b = PRIMARY_PAIR
    diff = np.array([total[e, a] - total[e, b] for e in events])
    assert len(events) == 13 - settings.quote.min_history_events
    assert p2["primary"] == paired_mean(diff, settings)
    assert abs(diff.mean() - np.mean([average[e, a] - average[e, b] for e in events])) > 1e-9


def test_p3_is_the_skill_of_the_walk_forward_recalibration(run):
    results, _, settings = run
    p3 = results.p3
    assert p3["primary"] is not None and p3["walk"] is not None
    walk = p3["walk"]
    assert p3["primary"]["n"] == len(walk) > 0
    again = skill_interval(walk, walk["cluster"], B=settings.n_boot, seed=settings.seed,
                           level=settings.level)
    assert again == p3["primary"]
    assert p3["n_markets"] >= len(walk)
    assert walk_forward_recalibration is not None  # the function the walk comes from


def test_shuffling_the_rows_changes_nothing(world, run):
    results, _, settings = run
    shuffled = run_protocol(
        world.tape.sample(frac=1.0, random_state=1).reset_index(drop=True),
        world.markets.sample(frac=1.0, random_state=2).reset_index(drop=True),
        world.events.sample(frac=1.0, random_state=3).reset_index(drop=True),
        "synthetic",
        settings,
    )
    assert shuffled.dataset.counts == results.dataset.counts
    assert shuffled.p1["primary"] == results.p1["primary"]
    assert shuffled.p2["primary"] == results.p2["primary"]
    assert shuffled.p3["primary"] == results.p3["primary"]


def test_the_latent_columns_are_never_read(world, run):
    results, _, settings = run
    latent = [c for c in world.tape.columns if c.startswith("latent_")]
    assert latent
    stripped = run_protocol(world.tape.drop(columns=latent), world.markets, world.events,
                            "synthetic", settings)
    assert stripped.p1["primary"] == results.p1["primary"]
    assert stripped.p2["primary"] == results.p2["primary"]


def test_a_small_sample_is_labelled_exploratory(run):
    results, _, settings = run
    assert results.dataset.counts["releases kept"] < DEFAULT.min_releases
    assert results.family["P1"]["verdict"] == "exploratory"
    assert results.family["P2"]["verdict"] == "exploratory"
    assert results.family["P3"]["verdict"] in {"supported", "not supported"}
    relaxed = with_overrides(settings, ["min_releases=10"])
    from kalshi_mm.protocol import family

    labels = family(results.p1, results.p2, results.p3, 26, relaxed)
    assert all(v["verdict"] in {"supported", "not supported", "not computed"}
               for v in labels.values())


def test_tables_carry_the_letter_and_the_label_of_their_data(run, tmp_path):
    results, _, _ = run
    out = tmp_path / "rehearsal"
    names = write_results(results, out)
    assert {"summary.json", "p1_buckets.csv", "p2_releases.csv", "p3_fits.csv",
            "tables.md"} <= set(names)
    fits = pd.read_csv(out / "p3_fits.csv")
    assert list(fits.columns) == ["a", "b", "n_fit", "first_release_ts", "markets_scored"]
    assert fits["first_release_ts"].is_monotonic_increasing and (fits["markets_scored"] > 0).all()
    text = (out / "tables.md").read_text()
    headings = re.findall(r"^### ([A-Z]\d+)\. (.*)$", text, flags=re.MULTILINE)
    assert [h[0] for h in headings] == [f"R{i}" for i in range(1, 9)]
    assert all(h[1].endswith("(synthetic data)") for h in headings)
    assert text.count("\nReading: ") == 8 and text.count("\nDoes not show: ") == 8
    assert "DEVIATIONS.md" in names  # the settings above differ from the protocol
    assert "n_boot: 200 (protocol: 10000)" in (out / "DEVIATIONS.md").read_text()
    csv = pd.read_csv(out / "p2_releases.csv")
    assert set(csv["policy"]) == set(POLICIES) and "latent" not in " ".join(csv.columns)


def test_synthetic_results_are_refused_a_directory_named_real(run, tmp_path):
    results, _, _ = run
    with pytest.raises(ValueError, match="named real"):
        write_results(results, tmp_path / "real")
    assert not (tmp_path / "real").exists()


def test_a_p3_without_prices_says_why_it_was_not_computed(world):
    settings = with_overrides(DEFAULT, [*SMALL, "recal_hours=24"])  # the world is 8 hours long
    results = run_protocol(world.tape, world.markets, world.events, "synthetic", settings)
    assert results.p3["primary"] is None and results.p3["error"] is None
    assert (results.p3["n_markets"], results.p3["n_without_price"]) == (26, 26)
    assert results.family["P3"]["verdict"] == "not computed"
    text = tables_markdown(results)
    assert ("the recalibration was not computed: it needs more than 12 releases with a price "
            "24 hours before the release, and 0 have one.") in text


def test_real_results_are_refused_a_directory_named_rehearsal(world, tmp_path):
    from kalshi_mm.protocol import Results

    results = run_protocol(world.tape, world.markets, world.events, "kalshi-public-api",
                           with_overrides(DEFAULT, SMALL))
    assert isinstance(results, Results)
    with pytest.raises(ValueError, match="named rehearsal"):
        write_results(results, tmp_path / "rehearsal")
    write_results(results, tmp_path / "out")
    text = (tmp_path / "out" / "tables.md").read_text()
    assert re.findall(r"^### ([A-Z]\d+)\.", text, flags=re.MULTILINE) == [
        f"K{i}" for i in range(1, 9)]
    assert text.count("(real data)") == 8


def test_run_real_refuses_a_synthetic_database(world, tmp_path, capsys):
    db = _database(tmp_path / "synthetic.sqlite", world, "synthetic")
    code = run_real.main(["--db", str(db), "--out", str(tmp_path / "out")])
    assert code == 2 and not (tmp_path / "out").exists()
    assert "synthetic data, not real data" in capsys.readouterr().out
    assert run_real.main(["--db", str(tmp_path / "missing.sqlite")]) == 2


def test_run_real_wants_the_tag_and_then_reads_the_database_read_only(
    world, tmp_path, capsys, monkeypatch
):
    db = _database(tmp_path / "kalshi.sqlite", world, "kalshi-public-api")
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    out = tmp_path / "out"
    args = ["--db", str(db), "--out", str(out)] + [f"--set={a}" for a in SMALL]

    monkeypatch.setattr(run_real, "tag_problem", lambda: "the tag real-run does not exist")
    assert run_real.main(args) == 2 and not out.exists()
    assert "git tag real-run" in capsys.readouterr().out

    assert run_real.main([*args, "--no-tag-check"]) == 0
    printed = capsys.readouterr().out
    assert "Counts by series:" in printed
    assert "Events to compare with the Kalshi website:" in printed
    assert "Tests of the family" in printed
    deviations = (out / "DEVIATIONS.md").read_text()
    assert "run without a clean tag: the tag real-run does not exist" in deviations
    assert "n_boot: 200 (protocol: 10000)" in deviations
    manifest = (out / "run_real.manifest.txt").read_text()
    assert f"sha256 {before}" in manifest and "source: kalshi-public-api" in manifest
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before  # opened read-only
    assert not np.isnan(float(re.search(r"count, trades kept: (\d+)", manifest).group(1)))

    monkeypatch.setattr(run_real, "tag_problem", lambda: None)
    clean = tmp_path / "clean"
    assert run_real.main(["--db", str(db), "--out", str(clean), "--set=n_boot=200",
                          *[f"--set={a}" for a in SMALL[1:]]]) == 0
    assert "n_boot" in (clean / "DEVIATIONS.md").read_text()  # still a changed setting
    assert "run without" not in (clean / "DEVIATIONS.md").read_text()


class _StopError(Exception):
    """Raised by the probe below to end ``run_real.main`` once the connection has been checked."""


def test_run_real_reads_the_database_through_a_connection_that_cannot_write(
    world, tmp_path, monkeypatch
):
    db = _database(tmp_path / "kalshi.sqlite", world, "kalshi-public-api")
    real_get_source = run_real.get_source

    def probe(conn):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("create table probe (x integer)")
        real_get_source(conn)
        raise _StopError

    monkeypatch.setattr(run_real, "get_source", probe)
    with pytest.raises(_StopError):
        run_real.main(["--db", str(db), "--out", str(tmp_path / "out")])


def test_prepare_on_the_database_frames_is_the_dataset_of_the_run(run):
    results, (tape, markets, events), settings = run
    again = prepare(tape, markets, events, settings)
    assert again.counts == results.dataset.counts
    assert again.tape.equals(results.dataset.tape)
    assert isinstance(results, Results)
