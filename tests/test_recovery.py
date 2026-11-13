"""The recovery study (kalshi_mm.experiments.recovery): oracle, population values, replications.

* The oracle adverse selection and the price impact are checked by hand on a tape of seven trades,
  against a plain-loop reference on a simulated world, and by the identity that makes the oracle
  worth having: the public belief is a martingale and the arrival times do not depend on the
  outcome, so the oracle and the price impact have the same expectation in every bucket.
* The bucket table, the saturation share, the surge share, the naive interval and the coverage and
  power tables are checked on hand-made inputs whose answers are worked out in the comments.
* The replications are checked against a by-hand run on a standalone world of each size (the
  worlds are nested), and must not read a latent column.
* The tables go through the parser that pastes them into the README.
"""

import dataclasses
import importlib.util
import json
import pathlib
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest

from kalshi_mm.experiments import recovery as rc
from kalshi_mm.markout import (
    bucket_index,
    bucket_means,
    release_effect,
    scaled_adverse_selection,
    trade_markouts,
)
from kalshi_mm.stats import cluster_weights, percentile_interval
from kalshi_mm.world import World, WorldParams, generate_world, informed_share

Z = 4.0
ROOT = pathlib.Path(__file__).resolve().parents[1]
BASE = 20261113
QUIET = (360.0, np.inf)
Z95 = NormalDist().inv_cdf(0.975)


def load_make_tables():
    """The script that pastes tables into README.md, loaded by path (it is not a package)."""
    path = ROOT / "scripts" / "make_tables.py"
    spec = importlib.util.spec_from_file_location("make_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_world(world, n_events, seed):
    return generate_world(rc.world_params(world, n_events), seed)


# ------------------------------------------------------------------------- worlds and estimands
def test_worlds_differ_from_the_defaults_only_where_they_say():
    base = WorldParams()
    assert rc.world_params("planted", 7) == dataclasses.replace(base, n_events=7)
    assert rc.world_params("spec_null", 7) == dataclasses.replace(base, n_events=7, pi1=0.0)
    assert rc.world_params("flat_null", 9) == dataclasses.replace(
        base, n_events=9, pi1=0.0, surge=0.0)
    assert base.pi1 == 0.20 and base.surge == 20.0  # what "planted" means
    assert list(rc.WORLDS) == ["planted", "spec_null", "flat_null"]


def test_the_four_estimands_are_the_documented_ones():
    spec = [(e.key, e.column, e.near) for e in rc.ESTIMANDS]
    assert spec == [
        ("raw_0_30", "adverse_selection", (0.0, 30.0)),
        ("scaled_0_30", "adverse_selection_scaled", (0.0, 30.0)),
        ("raw_5_30", "adverse_selection", (5.0, 30.0)),
        ("scaled_5_30", "adverse_selection_scaled", (5.0, 30.0)),
    ]
    assert [e.label[:3] for e in rc.ESTIMANDS] == ["(a)", "(b)", "(c)", "(d)"]
    assert rc.QUIET == (360.0, np.inf) and rc.HORIZON_S == 300.0


def test_the_design_constants_are_the_ones_the_docstring_states():
    assert (rc.REPS_DESIGN, rc.REPS, rc.BOOTSTRAP_B) == (200, 100, 1000)
    assert (rc.POP_EVENTS, rc.POP_SEED, rc.SIZES) == (3000, 0, (25, 50, 100, 200, 400))
    assert rc.ALPHA == 0.05 and rc.POWER_TARGET == 0.8 and rc.BASE_SEED == BASE
    doc = " ".join(rc.__doc__.split())
    assert "3,000 events with seed 0" in doc and "the first ``n`` events form the world" in doc
    assert "``REPS`` is 100" in doc


def test_public_markouts_come_from_the_public_columns_and_carry_the_event_position():
    world = small_world("planted", 8, 3)
    garbage = world.tape.assign(latent_mid_pre=np.nan, latent_mid_post=-1.0, latent_pi=7.0,
                                latent_informed=True)
    corrupted = World(world.params, world.events, world.markets, garbage)
    m = rc.public_markouts(world)
    pd.testing.assert_frame_equal(m, rc.public_markouts(corrupted))
    assert not [c for c in m.columns if str(c).startswith("latent_")]
    np.testing.assert_array_equal(
        m[rc.SCALED].to_numpy(), scaled_adverse_selection(m).to_numpy())  # NaN-aware equality
    assert m["event_no"].min() == 0 and m["event_no"].max() == 7
    by_hand = [int(e.split("-")[1]) for e in m["event"]]  # SYN-000005 is event 5
    assert list(m["event_no"]) == by_hand


# ------------------------------------------------------------------------------------- oracle
def tiny_tape():
    """Market A: five trades, market B: two. Times in seconds, mids are latent beliefs."""
    return pd.DataFrame({
        "event": ["A"] * 5 + ["B"] * 2,
        "market": ["A"] * 5 + ["B"] * 2,
        "ts": [0.0, 100.0, 250.0, 400.0, 700.0, 0.0, 50.0],
        "seq": [0, 1, 2, 3, 4, 0, 1],
        "side": np.array([1, -1, 1, 1, -1, 1, 1], dtype=np.int8),
        "latent_mid_pre": [0.50, 0.52, 0.55, 0.54, 0.60, 0.30, 0.33],
        "latent_mid_post": [0.52, 0.51, 0.56, 0.58, 0.55, 0.32, 0.36],
    })


def test_oracle_by_hand_with_the_default_horizon():
    """Horizon 300 s. Trade 0 (t = 0): the first trade at or after 300 is trade 3 (t = 400), so the
    latent mid is pre[3] = 0.54 and the value is 0.54 - 0.50. Trade 1 (t = 100): the first trade at
    or after 400 is trade 3 itself (at, not after). Trade 2 (t = 250): first at or after 550 is
    trade 4. Trade 3 (t = 400): first at or after 700 is trade 4 (equal time). Trade 4 is the last:
    nothing is left, so the latent mid after it, post[4] = 0.55, is used. In market B no trade is
    at or after t + 300, so both use post of the last trade, 0.36 (the markets do not mix)."""
    oracle = rc.oracle_adverse_selection(tiny_tape())
    expected = [1 * (0.54 - 0.50), -1 * (0.54 - 0.52), 1 * (0.60 - 0.55), 1 * (0.60 - 0.54),
                -1 * (0.55 - 0.60), 1 * (0.36 - 0.30), 1 * (0.36 - 0.33)]
    np.testing.assert_allclose(oracle, expected, rtol=0, atol=1e-15)
    assert rc.HORIZON_S == 300.0


def test_oracle_by_hand_with_a_shorter_horizon():
    """Horizon 120 s: trade 0 looks at t = 120, first trade at or after is trade 2 (pre 0.55);
    trade 1 at 220, trade 2 again; trade 2 at 370, trade 3 (pre 0.54); trade 3 at 520, trade 4
    (pre 0.60); trade 4 has nothing left (post 0.55)."""
    oracle = rc.oracle_adverse_selection(tiny_tape(), horizon_s=120.0)
    expected = [1 * (0.55 - 0.50), -1 * (0.55 - 0.52), 1 * (0.54 - 0.55), 1 * (0.60 - 0.54),
                -1 * (0.55 - 0.60), 1 * (0.36 - 0.30), 1 * (0.36 - 0.33)]
    np.testing.assert_allclose(oracle[:5], expected[:5], rtol=0, atol=1e-15)
    np.testing.assert_allclose(oracle[5:], expected[5:], rtol=0, atol=1e-15)


def test_price_impact_by_hand_and_empty_tape():
    impact = rc.price_impact(tiny_tape())
    np.testing.assert_allclose(impact, [0.02, 0.01, 0.01, 0.04, 0.05, 0.02, 0.03], atol=1e-15)
    empty = tiny_tape().iloc[:0]
    assert rc.oracle_adverse_selection(empty).shape == (0,)


def reference_oracle(tape, horizon):
    """The same definition as a loop over trades: scan forward for the first trade at or after
    t + horizon in the same market, else take the latent mid after the market's last trade."""
    market, ts, side = tape["market"].to_numpy(), tape["ts"].to_numpy(), tape["side"].to_numpy()
    pre, post = tape["latent_mid_pre"].to_numpy(), tape["latent_mid_post"].to_numpy()
    out = []
    for i in range(len(tape)):
        same = [j for j in range(len(tape)) if market[j] == market[i]]
        later = [j for j in same if j >= i and ts[j] >= ts[i] + horizon]
        last = same[-1]
        after = pre[later[0]] if later else post[last]
        out.append(side[i] * (after - pre[i]))
    return np.array(out)


@pytest.mark.parametrize("horizon", [300.0, 60.0, 4000.0])
def test_oracle_equals_a_plain_loop_on_a_simulated_world(horizon):
    tape = small_world("planted", 3, 4).tape
    np.testing.assert_allclose(
        rc.oracle_adverse_selection(tape, horizon_s=horizon), reference_oracle(tape, horizon),
        rtol=0, atol=1e-14)


@pytest.fixture(scope="module")
def planted_600():
    world = small_world("planted", 600, 11)
    return world, rc.public_markouts(world)


def test_oracle_and_price_impact_have_the_same_expectation_in_every_bucket(planted_600, cluster):
    """The public belief is a martingale and the arrival times do not depend on the outcome, so
    E[side (m_after - m_after_trade_i)] = 0 for any later latent mid, and the oracle has the
    expectation of the impact. The check is a difference of per-trade values with the event as
    cluster, four standard errors, in each of the seven buckets (this fails if, when no trade is
    left, the latent mid before the last trade is used instead of the one after it)."""
    world, m = planted_600
    oracle = pd.Series(rc.oracle_adverse_selection(world.tape), index=world.tape.index)
    impact = pd.Series(rc.price_impact(world.tape), index=world.tape.index)
    diff = (oracle - impact).loc[m.index].to_numpy()
    for k in range(7):
        mean, se = cluster.mean_se(diff, m["event_no"].to_numpy(), m["bucket"].to_numpy() == k)
        assert abs(mean) < Z * se, f"bucket {k}: mean {mean:.3g}, se {se:.3g}"
    pooled, se = cluster.mean_se(diff, m["event_no"].to_numpy())
    assert abs(pooled) < Z * se
    assert impact.mean() > 0.005  # and the quantity is far from zero, so the check has power


def test_saturation_share_counts_events_by_their_last_latent_mid():
    rows = [("E1", 0.999), ("E1", 0.5), ("E2", 0.3), ("E2", 0.995), ("E3", 0.5),
            ("E3", 0.9995), ("E3", 0.004), ("E4", 0.5), ("E4", 0.99), ("E5", 0.2)]
    tape = pd.DataFrame(rows, columns=["event", "latent_mid_post"])
    # last values 0.5, 0.995, 0.004, 0.99, 0.2: outside [0.01, 0.99] are 0.995 and 0.004 (0.99 is
    # not above 0.99): 2 of 5. First values would give 1 of 5 and maxima 3 of 5.
    assert rc.saturation_share(tape) == pytest.approx(0.4)
    # another band, [0.3, 0.6]: only the last value 0.5 is inside, 4 of 5 are outside
    assert rc.saturation_share(tape, lo=0.3, hi=0.6) == pytest.approx(0.8)


# ------------------------------------------------------------------------------ bucket table
def hand_markouts():
    tau = np.array([3.0, 10.0, 20.0, 40.0, 100.0, 200.0, 400.0, 500.0])
    return pd.DataFrame({
        "tau_min": tau,
        "bucket": bucket_index(tau),
        "adverse_selection": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, np.nan],
        rc.SCALED: [0.10, 0.20, np.nan, 0.40, 0.50, 0.60, 0.70, np.nan],
    })


def test_bucket_table_by_hand():
    """One trade per bucket except the last bucket, which has a trade without a proxy (left out of
    every column), and the (15, 30] bucket, whose scaled value is missing. Cents are 100 times the
    USD values; the bias is tape minus oracle; D is the near row minus the quiet row."""
    oracle = np.array([0.011, 0.022, 0.033, 0.044, 0.055, 0.066, 0.077, 0.088])
    impact = np.array([0.012, 0.024, 0.036, 0.048, 0.060, 0.072, 0.084, 0.096])
    params = WorldParams()
    t = rc.bucket_table(hand_markouts(), oracle, impact, params)
    assert list(t["label"]) == ["(0,5]", "(5,15]", "(15,30]", "(30,60]", "(60,180]", "(180,360]",
                                ">360", "near (0,30]", "D = near - quiet"]
    assert list(t["n_trades"]) == [1, 1, 1, 1, 1, 1, 1, 3, 0]
    np.testing.assert_allclose(t["tape_as_cents"], [1, 2, 3, 4, 5, 6, 7, 2.0, -5.0], atol=1e-12)
    np.testing.assert_allclose(t["oracle_as_cents"],
                               [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 2.2, -5.5], atol=1e-12)
    np.testing.assert_allclose(t["impact_cents"],
                               [1.2, 2.4, 3.6, 4.8, 6.0, 7.2, 8.4, 2.4, -6.0], atol=1e-12)
    np.testing.assert_allclose(t["bias_cents"],
                               [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.2, 0.5], atol=1e-12)
    scaled = t["scaled_as_cents"].to_numpy()
    np.testing.assert_allclose(scaled[[0, 1, 3, 4, 5, 6, 7, 8]],
                               [10, 20, 40, 50, 60, 70, 15, 15 - 70], atol=1e-12)
    assert np.isnan(scaled[2])  # no trade of (15, 30] has a scaled value
    tau_mean = [3, 10, 20, 40, 100, 200, 400, (3 + 10 + 20) / 3]
    np.testing.assert_allclose(t["tau_mean_min"].iloc[:8], tau_mean, atol=1e-12)
    assert np.isnan(t["tau_mean_min"].iloc[8])
    pi = informed_share(np.array(tau_mean) / 60.0, params)
    np.testing.assert_allclose(t["pi_at_mean_tau"].iloc[:8], pi, atol=1e-15)
    assert t["pi_at_mean_tau"].iloc[8] == pytest.approx(pi[7] - pi[6], abs=1e-15)


def test_bucket_table_tape_column_is_the_foundations_bucket_mean(planted_600):
    world, m = planted_600
    oracle = rc.oracle_adverse_selection(world.tape)
    impact = rc.price_impact(world.tape)
    t = rc.bucket_table(m, oracle[m.index.to_numpy()], impact[m.index.to_numpy()], world.params)
    ref = bucket_means(m, "adverse_selection")
    np.testing.assert_allclose(t["tape_as_cents"].iloc[:7], 100 * ref["mean"], rtol=1e-12)
    np.testing.assert_array_equal(t["n_trades"].iloc[:7], ref["n"])
    ref_scaled = bucket_means(m, rc.SCALED)
    np.testing.assert_allclose(t["scaled_as_cents"].iloc[:7], 100 * ref_scaled["mean"], rtol=1e-12)
    # the near row pools the first three buckets, trade-weighted
    n = ref["n"].to_numpy()[:3]
    assert t["tape_as_cents"].iloc[7] == pytest.approx(
        100 * np.sum(ref["mean"].to_numpy()[:3] * n) / n.sum(), rel=1e-12)


# ------------------------------------------------------------------------------- populations
def stub_population(world, effects, sums=None, saturation=0.5):
    return rc.Population(world, WorldParams(), 4, 100, pd.DataFrame(), effects, saturation,
                         sums or {})


def stub_effect(D, lo, hi, near=0.02, quiet=0.01):
    return {"D": D, "D_ci": (lo, hi), "mean_near": near, "mean_quiet": quiet, "n_near": 50,
            "n_quiet": 40, "n_events": 4, "p_one_sided": 0.01}


def test_interval_sign():
    assert rc.interval_sign(0.001, 0.02) == "positive"
    assert rc.interval_sign(-0.02, -0.001) == "negative"
    assert rc.interval_sign(-0.01, 0.01) == "zero"
    assert rc.interval_sign(0.0, 0.01) == "zero" and rc.interval_sign(-0.01, 0.0) == "zero"


def test_effects_frame_converts_to_cents_and_labels_the_sign():
    effects = {
        "raw_0_30": stub_effect(0.0091, 0.0084, 0.0099),
        "scaled_0_30": stub_effect(-0.0016, -0.0019, -0.0013),
        "raw_5_30": stub_effect(0.0003, -0.0001, 0.0008),
        "scaled_5_30": stub_effect(0.05, 0.04, 0.06, near=0.07, quiet=0.02),
    }
    frame = rc.effects_frame({"planted": stub_population("planted", effects)})
    assert list(frame["estimand"]) == ["raw_0_30", "scaled_0_30", "raw_5_30", "scaled_5_30"]
    assert list(frame["sign"]) == ["positive", "negative", "zero", "positive"]
    first = frame.iloc[0]
    assert (first["D_cents"], first["lo_cents"], first["hi_cents"]) == pytest.approx(
        (0.91, 0.84, 0.99))
    assert (first["mean_near_cents"], first["mean_quiet_cents"]) == pytest.approx((2.0, 1.0))
    assert (first["n_near"], first["n_quiet"], first["n_events"]) == (50, 40, 4)
    assert frame.iloc[3]["mean_near_cents"] == pytest.approx(7.0)


def test_surge_share_point_estimate_and_paired_bootstrap_against_an_explicit_resample():
    rng = np.random.default_rng(3)
    G, B = 12, 80
    n_near, n_quiet = rng.integers(3, 9, G), rng.integers(2, 7, G)

    def sums(scale):
        return {"near_n": n_near, "quiet_n": n_quiet,
                "near_sum": scale * n_near * rng.uniform(1.0, 3.0, G),
                "quiet_sum": n_quiet * rng.uniform(0.2, 0.8, G)}

    a, b = sums(1.0), sums(0.5)

    def effect(s):
        return s["near_sum"].sum() / s["near_n"].sum() - s["quiet_sum"].sum() / s["quiet_n"].sum()

    planted = stub_population("planted", {"raw_0_30": stub_effect(effect(a), 0, 0)}, a)
    spec = stub_population("spec_null", {"raw_0_30": stub_effect(effect(b), 0, 0)}, b)
    out = rc.surge_share(planted, spec, B, seed=9)
    assert out["share"] == pytest.approx(effect(b) / effect(a), rel=1e-14)
    assert (out["n_events"], out["B"], out["B_undefined"], out["seed"]) == (G, B, 0, 9)
    def resampled(s, idx):
        return (s["near_sum"][idx].sum() / s["near_n"][idx].sum()
                - s["quiet_sum"][idx].sum() / s["quiet_n"][idx].sum())

    w = cluster_weights(G, B, np.random.default_rng(9))  # the same weights for both worlds
    shares = []
    for row in w:
        idx = np.repeat(np.arange(G), row)  # the resampled list of events, with repeats
        shares.append(resampled(b, idx) / resampled(a, idx))
    assert out["ci"] == pytest.approx(percentile_interval(np.array(shares)), rel=1e-12)
    assert out["ci"][0] < out["share"] < out["ci"][1]
    with pytest.raises(ValueError, match="same events"):
        rc.surge_share(planted, stub_population("spec_null", {}, {k: v[:5] for k, v in b.items()}),
                       B)


@pytest.fixture(scope="module")
def population_40():
    return {w: rc.population_world(w, 40, 3, 100) for w in rc.WORLDS}


def test_population_world_is_consistent_with_its_parts(population_40):
    pop = population_40["planted"]
    sim = small_world("planted", 40, 3)
    m = rc.public_markouts(sim)
    assert (pop.n_events, pop.n_trades, pop.world) == (40, len(sim.tape), "planted")
    for e in rc.ESTIMANDS:  # release_effect on the same markouts, same seed and B
        direct = release_effect(m, near=e.near, quiet=QUIET, column=e.column, B=100, seed=BASE)
        assert pop.effects[e.key] == direct
    last = [g["latent_mid_post"].iloc[-1] for _, g in sim.tape.groupby("event")]
    assert pop.saturation == pytest.approx(np.mean([(x < 0.01) or (x > 0.99) for x in last]))
    near, quiet = pop.sums["near_sum"].sum(), pop.sums["quiet_sum"].sum()
    assert near / pop.sums["near_n"].sum() == pytest.approx(
        pop.effects["raw_0_30"]["mean_near"], rel=1e-12)
    assert quiet / pop.sums["quiet_n"].sum() == pytest.approx(
        pop.effects["raw_0_30"]["mean_quiet"], rel=1e-12)
    assert len(pop.sums["near_n"]) == 40
    assert list(pop.buckets["label"])[-2:] == ["near (0,30]", "D = near - quiet"]
    index = m.index.to_numpy()
    direct_oracle = 100 * rc.oracle_adverse_selection(sim.tape)[index]
    use = (m["bucket"].to_numpy() == 0) & m["adverse_selection"].notna().to_numpy()
    assert pop.buckets["oracle_as_cents"].iloc[0] == pytest.approx(direct_oracle[use].mean())


def test_without_an_informed_share_profile_the_planted_share_is_flat(population_40):
    for world in ("spec_null", "flat_null"):
        share = population_40[world].buckets["pi_at_mean_tau"].iloc[:8]
        np.testing.assert_allclose(share, 0.05, atol=1e-15)
    planted = population_40["planted"].buckets["pi_at_mean_tau"].iloc[:7]
    assert planted.is_monotonic_decreasing and planted.iloc[0] > planted.iloc[-1]


def test_the_notes_about_the_surge_and_the_level_hold_in_the_worlds():
    """The statements the tables make: with no informed-share profile raw D is positive when the
    surge is kept and negative without it, and scaled D near (5, 30] is near zero without the
    surge (800 events per world; the intervals are cluster bootstrap)."""
    pops = {w: rc.population_world(w, 800, 0, 300) for w in rc.WORLDS}
    effects = rc.effects_frame(pops).set_index(["world", "estimand"])
    assert effects.loc[("spec_null", "raw_0_30"), "sign"] == "positive"
    assert effects.loc[("flat_null", "raw_0_30"), "sign"] == "negative"
    assert effects.loc[("flat_null", "scaled_5_30"), "sign"] == "zero"
    assert effects.loc[("planted", "scaled_5_30"), "sign"] == "positive"
    assert pops["planted"].saturation > 0.5 > pops["flat_null"].saturation
    share = rc.surge_share(pops["planted"], pops["spec_null"], 300)
    assert 0.2 < share["share"] < 0.8 and share["ci"][0] < share["share"] < share["ci"][1]


# --------------------------------------------------------------------------- naive interval
def naive_frame():
    return pd.DataFrame({
        "tau_min": [10.0, 20.0, 25.0, 40.0, 400.0, 500.0, 360.0, 5.0],
        "value": [2.0, 4.0, 6.0, 99.0, 1.0, 3.0, 77.0, 55.0],
    })


def test_naive_interval_by_hand():
    """Near (0, 30]: values 2, 4, 6 (40 is outside; 5 is inside only for a window starting below
    5). Quiet (360, inf): 1 and 3 (360 itself is excluded). Means 4 and 2, sample variances 4 and
    2, so D = 2 and the standard error is sqrt(4/3 + 2/2) = sqrt(7/3)."""
    frame = naive_frame()
    frame = frame[frame["tau_min"] != 5.0]
    d, lo, hi = rc.naive_interval(frame, "value", (0.0, 30.0))
    half = Z95 * np.sqrt(7.0 / 3.0)
    assert (d, lo, hi) == pytest.approx((2.0, 2.0 - half, 2.0 + half), rel=1e-12)
    assert half == pytest.approx(2.9939, abs=1e-3)
    # the window (5, 30] drops the trade at 5; the one at 5 minutes belongs to (0, 5]
    both = rc.naive_interval(naive_frame(), "value", (0.0, 30.0))[0]
    uncut = rc.naive_interval(naive_frame(), "value", (5.0, 30.0))[0]
    assert uncut == pytest.approx(2.0)  # 2, 4, 6 against 1, 3
    assert both == pytest.approx((2 + 4 + 6 + 55) / 4 - 2.0)  # the value 55 at 5 minutes joins
    assert rc.naive_interval(frame, "value", (0.0, 30.0), level=0.9)[2] == pytest.approx(
        2.0 + NormalDist().inv_cdf(0.95) * np.sqrt(7.0 / 3.0), rel=1e-12)


def test_naive_interval_ignores_nan_and_handles_thin_windows():
    frame = naive_frame()
    frame = frame[frame["tau_min"] != 5.0].copy()
    frame.loc[frame["tau_min"] == 40.0, "value"] = np.nan
    frame.loc[frame["tau_min"] == 25.0, "value"] = np.nan  # near keeps 2 and 4: mean 3, var 2
    d, lo, hi = rc.naive_interval(frame, "value", (0.0, 30.0))
    half = Z95 * np.sqrt(2.0 / 2 + 2.0 / 2)
    assert (d, lo, hi) == pytest.approx((1.0, 1.0 - half, 1.0 + half), rel=1e-12)
    one = frame[frame["tau_min"] != 20.0]  # a single near value: D is defined, the interval is not
    d, lo, hi = rc.naive_interval(one, "value", (0.0, 30.0))
    assert d == pytest.approx(0.0) and np.isnan(lo) and np.isnan(hi)
    none = frame[frame["tau_min"] > 100]  # no near value at all
    assert all(np.isnan(v) for v in rc.naive_interval(none, "value", (0.0, 30.0)))


def clustered_frame(rng, G=30, trades=40, effect_sd=1.0):
    """G events, each with `trades` near and `trades` quiet trades. The near values share an event
    effect with sd `effect_sd` that the quiet values lack; the population D is zero."""
    event = np.repeat(np.arange(G), 2 * trades)
    near = np.tile(np.r_[np.ones(trades), np.zeros(trades)], G).astype(bool)
    effect = rng.normal(0.0, effect_sd, G)[event]
    value = rng.normal(0.0, 1.0, event.size) + np.where(near, effect, 0.0)
    return pd.DataFrame({"event": event, "tau_min": np.where(near, 10.0, 500.0), "value": value})


def test_naive_interval_covers_independent_trades_and_misses_clustered_ones():
    """D = 0 by construction. With independent trades the naive 95% interval covers (the Monte
    Carlo standard error of a coverage near 0.95 with 300 draws is 0.0126, so 4 of them are
    allowed); with an event effect in the near values it covers far too rarely, while the cluster
    bootstrap of release_effect (B = 300) stays near 0.95 (it is a little short with 30 events)."""
    rng = np.random.default_rng(31)
    R = 300
    cover = {"naive_iid": 0, "naive_clustered": 0, "cluster": 0}
    for _ in range(R):
        iid = clustered_frame(rng, effect_sd=0.0)
        _, lo, hi = rc.naive_interval(iid, "value", (0.0, 30.0))
        cover["naive_iid"] += lo <= 0.0 <= hi
        frame = clustered_frame(rng)
        _, lo, hi = rc.naive_interval(frame, "value", (0.0, 30.0))
        cover["naive_clustered"] += lo <= 0.0 <= hi
        res = release_effect(frame, column="value", B=300, seed=int(rng.integers(1 << 30)))
        cover["cluster"] += res["D_ci"][0] <= 0.0 <= res["D_ci"][1]
    se = np.sqrt(0.95 * 0.05 / R)
    assert abs(cover["naive_iid"] / R - 0.95) < Z * se
    assert cover["naive_clustered"] / R < 0.6  # about 0.41 by the variance ratio
    assert abs(cover["cluster"] / R - 0.95) < 0.05


def test_naive_d_equals_release_effect_d_for_every_estimand(planted_600):
    _, m = planted_600
    first = m[m["event_no"] < 30]
    for e in rc.ESTIMANDS:
        d, lo, hi = rc.naive_interval(first, e.column, e.near)
        ref = release_effect(first, near=e.near, quiet=QUIET, column=e.column, B=50, seed=1)
        assert d == pytest.approx(ref["D"], rel=1e-12)
        assert lo < d < hi


# -------------------------------------------------------------------------------- replications
def test_bootstrap_seed_follows_the_documented_formula():
    assert rc.bootstrap_seed(3) == int(np.random.SeedSequence([BASE, 2, 3]).generate_state(1)[0])
    seeds = [rc.bootstrap_seed(r) for r in range(500)]
    assert len(set(seeds)) == 500 and all(0 <= s < 2**32 for s in seeds)


def test_replication_equals_a_by_hand_run_on_a_standalone_world_of_each_size():
    """Rows of ``run_replication`` against ``release_effect`` and a hand-written naive interval on
    a world of exactly ``n`` events (same seed): the worlds are nested, so the prefix of the larger
    world is that world, and integer event labels must not change a result."""
    world, rep, B = "spec_null", 2, 60
    out = rc.run_replication(world, rep, (6, 12), B)
    assert len(out) == 8 and set(out["n_events"]) == {6, 12}
    assert list(out["estimand"][:4]) == [e.key for e in rc.ESTIMANDS]
    assert set(out["world"]) == {world} and set(out["rep"]) == {rep}
    seed = int(np.random.SeedSequence([BASE, 2, rep]).generate_state(1)[0])
    for n in (6, 12):
        sim = generate_world(dataclasses.replace(WorldParams(), n_events=n, pi1=0.0), BASE + rep)
        public = sim.tape[[c for c in sim.tape.columns if not c.startswith("latent_")]]
        m = trade_markouts(public, sim.events, sim.markets)
        m["scaled"] = scaled_adverse_selection(m)
        for e in rc.ESTIMANDS:
            column = "scaled" if e.column == rc.SCALED else e.column
            ref = release_effect(m, near=e.near, column=column, B=B, seed=seed)
            row = out[(out["n_events"] == n) & (out["estimand"] == e.key)].iloc[0]
            assert (row["D"], row["lo"], row["hi"], row["p"]) == (
                ref["D"], ref["D_ci"][0], ref["D_ci"][1], ref["p_one_sided"])
            tau, v = m["tau_min"].to_numpy(), m[column].to_numpy()
            near = v[~np.isnan(v) & (tau > e.near[0]) & (tau <= e.near[1])]
            quiet = v[~np.isnan(v) & (tau > 360.0)]
            se = np.sqrt(near.var(ddof=1) / near.size + quiet.var(ddof=1) / quiet.size)
            assert row["naive_lo"] == pytest.approx(ref["D"] - Z95 * se, rel=1e-10)
            assert row["naive_hi"] == pytest.approx(ref["D"] + Z95 * se, rel=1e-10)
    alone = rc.run_replication(world, rep, (6,), B)
    pd.testing.assert_frame_equal(alone, out[out["n_events"] == 6].reset_index(drop=True))


def test_replications_are_deterministic_and_differ_between_replications_and_worlds():
    a = rc.run_replication("planted", 0, (8,), 40)
    pd.testing.assert_frame_equal(a, rc.run_replication("planted", 0, (8,), 40))
    assert not a["D"].equals(rc.run_replication("planted", 1, (8,), 40)["D"])
    assert not a["D"].equals(rc.run_replication("flat_null", 0, (8,), 40)["D"])


def test_replications_do_not_read_a_latent_column(monkeypatch):
    plain = rc.run_replication("planted", 1, (6, 10), 40)

    def corrupted(params, seed):
        world = generate_world(params, seed)
        tape = world.tape.assign(latent_mid_pre=np.nan, latent_mid_post=np.nan, latent_pi=np.nan,
                                 latent_informed=False)
        return World(world.params, world.events, world.markets, tape)

    monkeypatch.setattr(rc, "generate_world", corrupted)
    pd.testing.assert_frame_equal(plain, rc.run_replication("planted", 1, (6, 10), 40))


# ------------------------------------------------------------------------------- aggregation
def hand_replications():
    """Four replications of one cell (world planted, estimand raw_0_30, n = 50); USD per contract.
    The population value is 0.010: replications 0 to 2 have a cluster interval around it (the
    third ends exactly at it) and the fourth misses; only the fourth naive interval covers."""
    return pd.DataFrame({
        "world": "planted", "rep": [0, 1, 2, 3], "n_events": 50, "estimand": "raw_0_30",
        "D": [0.012, 0.008, 0.004, 0.020],
        "lo": [0.005, 0.002, -0.002, 0.012], "hi": [0.018, 0.014, 0.010, 0.026],
        "p": [0.001, 0.03, 0.2, 0.05],
        "naive_lo": [0.011, 0.009, 0.005, 0.0095], "naive_hi": [0.013, 0.0095, 0.006, 0.030],
    })


def test_coverage_table_by_hand():
    out = rc.coverage_table(hand_replications(), {("planted", "raw_0_30"): 0.010})
    assert len(out) == 1
    r = out.iloc[0]
    assert (r["world"], r["estimand"], r["n_events"], r["reps"]) == ("planted", "raw_0_30", 50, 4)
    assert r["pop_D_cents"] == pytest.approx(1.0)
    assert r["cover_cluster"] == 0.75  # the end point 0.010 counts as covered
    assert r["cover_naive"] == 0.25  # only replication 3: 0.0095 <= 0.010 <= 0.030
    assert r["cover_cluster_se"] == pytest.approx(np.sqrt(0.75 * 0.25 / 4), rel=1e-14)
    assert r["cover_naive_se"] == pytest.approx(np.sqrt(0.25 * 0.75 / 4), rel=1e-14)
    assert r["width_cluster_cents"] == pytest.approx(100 * np.mean([0.013, 0.012, 0.012, 0.014]))
    assert r["width_naive_cents"] == pytest.approx(
        100 * np.mean([0.002, 0.0005, 0.001, 0.0205]), rel=1e-12)
    assert r["mean_D_cents"] == pytest.approx(100 * 0.011)
    assert r["sd_D_cents"] == pytest.approx(100 * np.std([0.012, 0.008, 0.004, 0.020], ddof=1))


def test_coverage_table_keeps_the_order_of_worlds_estimands_and_sizes():
    base = hand_replications()
    frames = []
    for world in ("flat_null", "planted"):
        for estimand in ("scaled_5_30", "raw_0_30"):
            for n in (100, 50):
                frames.append(base.assign(world=world, estimand=estimand, n_events=n))
    truth = {(w, e.key): 0.01 for w in rc.WORLDS for e in rc.ESTIMANDS}
    out = rc.coverage_table(pd.concat(frames, ignore_index=True), truth)
    assert list(zip(out["world"], out["estimand"], out["n_events"], strict=True)) == [
        ("planted", "raw_0_30", 50), ("planted", "raw_0_30", 100),
        ("planted", "scaled_5_30", 50), ("planted", "scaled_5_30", 100),
        ("flat_null", "raw_0_30", 50), ("flat_null", "raw_0_30", 100),
        ("flat_null", "scaled_5_30", 50), ("flat_null", "scaled_5_30", 100)]
    assert set(out["reps"]) == {4}


def effects_for_power():
    rows = [{"world": w, "estimand": e.key, "D_cents": 1.0 if w == "planted" else -0.1,
             "sign": "positive" if w == "planted" else "negative"}
            for w in rc.WORLDS for e in rc.ESTIMANDS]
    return pd.DataFrame(rows)


def test_power_table_by_hand():
    """p-values 0.001, 0.03, 0.2, 0.05: the one at exactly the level does not reject, so 2 of 4."""
    out = rc.power_table(hand_replications(), effects_for_power())
    r = out.iloc[0]
    assert (r["reject"], r["reps"]) == (0.5, 4)
    assert r["reject_se"] == pytest.approx(0.25, rel=1e-14)
    assert (r["pop_D_cents"], r["pop_sign"]) == (1.0, "positive")
    null = rc.power_table(hand_replications().assign(world="flat_null"), effects_for_power())
    assert (null.iloc[0]["pop_D_cents"], null.iloc[0]["pop_sign"]) == (-0.1, "negative")


def test_sample_size_is_the_smallest_size_strictly_above_the_target():
    sizes = [25, 50, 100, 200]
    rates = {
        "raw_0_30": [0.3, 0.8, 0.85, 0.95],      # exactly 0.8 does not count: 100
        "scaled_0_30": [0.2, 0.4, 0.6, 0.79],    # never above 0.8
        "raw_5_30": [0.9, 0.5, 0.85, 0.95],      # not monotone: the first size above, 25
        "scaled_5_30": [0.0, 0.0, 0.81, 1.0],
    }
    rows = [{"world": "planted", "estimand": k, "n_events": n, "reject": r}
            for k, v in rates.items() for n, r in zip(sizes, v, strict=True)]
    rows += [{"world": "spec_null", "estimand": "raw_0_30", "n_events": 25, "reject": 1.0}]
    power = pd.DataFrame(rows)
    assert rc.sample_size(power) == {"raw_0_30": 100, "scaled_0_30": None, "raw_5_30": 25,
                                     "scaled_5_30": 100}
    assert rc.sample_size(power, world="spec_null")["raw_0_30"] == 25
    assert rc.sample_size(power, level=0.9)["raw_0_30"] == 200


# ------------------------------------------------------------------------------------ tables
@pytest.fixture(scope="module")
def study():
    return rc.run_study(reps=2, B=40, pop_events=40, sizes=(6, 12))


def test_run_study_shapes_and_consistency(study):
    res = study
    assert list(res.populations) == list(rc.WORLDS)
    assert len(res.buckets) == 27 and len(res.effects) == 12
    assert len(res.coverage) == len(res.power) == 3 * 4 * 2
    assert res.design == {"reps": 2, "reps_design": 200, "B": 40, "pop_events": 40,
                          "sizes": [6, 12]}
    assert set(res.coverage["reps"]) == set(res.power["reps"]) == {2}
    truth = res.effects.set_index(["world", "estimand"])["D_cents"]
    for r in res.coverage.itertuples():
        assert r.pop_D_cents == pytest.approx(truth[(r.world, r.estimand)], rel=1e-12)
    assert set(res.sample) == {e.key for e in rc.ESTIMANDS}
    assert res.surge["n_events"] == 40 and res.surge["B"] == 40
    assert res.surge["ci"][0] <= res.surge["ci"][1]


def test_tables_are_eight_blocks_the_readme_parser_reads(study, tmp_path):
    text = rc.tables_markdown(study)
    (tmp_path / "markouts").mkdir()
    (tmp_path / "markouts" / "tables.md").write_text(text)
    tables = load_make_tables().read_tables(tmp_path)
    assert list(tables) == [f"M{i}" for i in range(1, 9)]
    assert len([ln for ln in text.splitlines() if ln.startswith("### ")]) == 8
    assert "$" not in text and text.endswith("\n") and "\n\n\n" not in text
    rows = {"M1": 9, "M2": 18, "M3": 12, "M4": 3, "M5": 12, "M6": 12, "M7": 12, "M8": 4}
    for table_id, block in tables.items():
        parts = block.split("\n\n")  # heading, Reading, table, Does not show
        assert len(parts) == 4, table_id
        assert parts[0].startswith(f"### {table_id}. ") and parts[0].endswith("(synthetic data)")
        assert parts[1].startswith("Reading: ") and "\n" not in parts[1]
        assert parts[2].startswith("| ") and len(parts[2].splitlines()) == 2 + rows[table_id]
        assert parts[3].startswith("Does not show: ") and "\n" not in parts[3]


def with_effects(res, spec, flat, clean):
    """``res`` with the raw D of spec_null and flat_null and the scaled (5, 30] D of flat_null
    replaced by ``(D in cents, sign)`` pairs."""
    effects = res.effects.copy()
    new = {("spec_null", "raw_0_30"): spec, ("flat_null", "raw_0_30"): flat,
           ("flat_null", "scaled_5_30"): clean}
    for (world, key), (d, sign) in new.items():
        row = (effects["world"] == world) & (effects["estimand"] == key)
        effects.loc[row, "D_cents"] = d
        effects.loc[row, "sign"] = sign
    return res._replace(effects=effects)


def reading_of(text, table_id):
    block = text.split(f"### {table_id}. ")[1].split("\n### ")[0]
    return block.split("\n\n")[1]


def test_the_statements_about_the_surge_and_the_level_follow_the_numbers(study):
    both = rc.tables_markdown(with_effects(study, (0.41, "positive"), (-0.16, "negative"),
                                           (0.03, "zero")))
    reading = reading_of(both, "M3")
    assert ("raw D is positive with the surge alone (spec_null, +0.41 cents) and negative with "
            "neither (flat_null, -0.16 cents), so D is a per-contract cost measured on the tape, "
            "not an estimate of the informed share; scaled D near (5, 30] is the cleaner read: "
            "in flat_null it is +0.03 cents, with an interval that contains zero") in reading
    flat = with_effects(study, (0.01, "zero"), (-0.01, "zero"), (0.12, "positive"))
    reading = reading_of(rc.tables_markdown(flat), "M3")
    assert ("raw D is zero with the surge alone (spec_null, +0.01 cents) and zero with neither"
            in reading)
    assert "per-contract cost" not in reading and "not an estimate" not in reading
    assert "scaled D near (5, 30] is still positive in flat_null (+0.12 cents)" in reading
    assert "cleaner read" not in reading


def test_saturation_surge_share_and_cut_note_come_from_the_results(study):
    populations = dict(study.populations)
    populations["planted"] = populations["planted"]._replace(saturation=0.837)
    surge = {"share": 0.45, "ci": (0.41, 0.48), "n_events": 3000, "B": 1000, "B_undefined": 0,
             "seed": BASE}
    res = study._replace(populations=populations, surge=surge)
    text = rc.tables_markdown(res)
    assert "(84% of its events end outside [0.01, 0.99])" in text
    assert "is 45% (95% interval 41% to 48%, paired cluster bootstrap, B = 1000)" in reading_of(
        text, "M4")
    assert "from 3000 events per world" in reading_of(text, "M4")
    assert "| saturated events, % | 83.7 |" in text
    cut = " The design asked for 200 replications per world; 2 were run to stay within the "
    for table_id in ("M5", "M7", "M8"):
        assert cut + "6-minute budget." in reading_of(text, table_id)
    full = rc.tables_markdown(res._replace(design={**res.design, "reps": 200}))
    assert "design asked for" not in full and "6-minute" not in full


def test_coverage_reading_gives_the_ranges_and_the_error_of_the_population_value(study):
    """Coverage rows for the five sizes of the real run are made from the two sizes of the small
    study; the n = 200 rows get known values so that the ranges in the Reading can be checked."""
    sizes = [25, 50, 100, 200, 400]

    def at_sizes(frame):
        return pd.concat([frame[frame["n_events"] == 6].assign(n_events=n) for n in sizes],
                         ignore_index=True)

    cov = at_sizes(study.coverage)
    at = (cov["n_events"] == 200).to_numpy()
    cov.loc[at, "cover_cluster"] = np.linspace(0.90, 0.96, at.sum())
    cov.loc[at, "cover_naive"] = np.linspace(0.20, 0.71, at.sum())
    power = at_sizes(study.power)
    res = study._replace(coverage=cov, power=power, sample=rc.sample_size(power),
                         design={**study.design, "pop_events": 3000, "sizes": sizes})
    text = rc.tables_markdown(res)
    reading = reading_of(text, "M5")
    assert ("at n = 200 the cluster interval covers 90% to 96% and the naive one 20% to 71%"
            in reading)
    assert "Monte Carlo standard error at most 35.4 points" in reading  # sqrt(0.25 / 2) = 0.354
    # the population value comes from 3000 events: the variance of D - D_pop is (1 + 400 / 3000)
    # times that of D, and P(|N(0, 1.1333)| < 1.96) = 2 Phi(1.96 / sqrt(1.1333)) - 1 = 0.934
    lost = 2 * NormalDist().cdf(1.96 / np.sqrt(1 + 400 / 3000)) - 1
    assert round(100 * lost, 1) == 93.4
    assert "lowers the nominal 95% to about 93.4% at n = 400." in text


def test_power_and_sample_size_tables_show_the_rates_and_the_sample_size(study):
    power = study.power.copy()
    planted = (power["world"] == "planted") & (power["estimand"] == "raw_0_30")
    power.loc[planted & (power["n_events"] == 6), "reject"] = 0.5
    power.loc[planted & (power["n_events"] == 12), "reject"] = 1.0
    res = study._replace(power=power, sample=rc.sample_size(power))
    text = rc.tables_markdown(res)
    assert res.sample["raw_0_30"] == 12
    assert "| (a) raw D, near (0, 30] | 12 | 100% | 50% at n = 6 |" in text
    first = power.assign(reject=0.9)
    text = rc.tables_markdown(study._replace(power=first, sample=rc.sample_size(first)))
    assert "| (a) raw D, near (0, 30] | 6 | 90% | no smaller size tried |" in text
    never = power.assign(reject=0.0)
    text = rc.tables_markdown(study._replace(power=never, sample=rc.sample_size(never)))
    assert "| (a) raw D, near (0, 30] | not reached (largest n tried 12) | 0% at n = 12 | |" in text


def test_the_bucket_tables_report_the_bias_of_the_proxy(study):
    text = rc.tables_markdown(study)
    diff = study.buckets[(study.buckets["world"] == "planted")
                         & (study.buckets["label"] == "D = near - quiet")].iloc[0]
    reading = reading_of(text, "M1")
    assert f"so D is {diff['tape_as_cents']:+.2f} on the tape, {diff['oracle_as_cents']:+.2f} by " \
           f"the oracle and {diff['impact_cents']:+.2f} by the price impact" in reading
    near = study.buckets[(study.buckets["world"] == "planted")
                         & (study.buckets["label"] == "near (0,30]")].iloc[0]
    assert f"the oracle is {near['bias_cents']:+.2f} near (0, 30]" in reading


# ----------------------------------------------------------------------------------- the run
def test_main_writes_every_file_and_is_reproducible(tmp_path):
    args = ["--reps", "2", "--B", "40", "--pop-events", "40", "--sizes", "6,12"]
    first, second = tmp_path / "first", tmp_path / "second"
    for out in (first, second):
        rc.main(["--out", str(out), *args])
    assert sorted(p.name for p in first.iterdir()) == [
        "coverage.csv", "population.csv", "power.csv", "recovery.manifest.txt", "summary.json",
        "tables.md"]
    assert len(pd.read_csv(first / "population.csv")) == 27
    coverage = pd.read_csv(first / "coverage.csv")
    assert len(coverage) == 24 and set(coverage["n_events"]) == {6, 12}
    assert len(pd.read_csv(first / "power.csv")) == 24
    for name in ("population.csv", "coverage.csv", "power.csv", "summary.json", "tables.md"):
        assert (first / name).read_bytes() == (second / name).read_bytes()

    summary = json.loads((first / "summary.json").read_text())
    assert summary["experiment"] == "recovery" and summary["data"] == "synthetic"
    assert summary["design"] == {"reps": 2, "reps_design": 200, "B": 40, "pop_events": 40,
                                 "sizes": [6, 12]}
    assert list(summary["population"]) == ["planted", "spec_null", "flat_null"]
    planted = summary["population"]["planted"]
    assert planted["n_events"] == 40 and len(planted["effects"]) == 4
    assert {"D_cents", "lo_cents", "hi_cents", "sign"} <= set(planted["effects"][0])
    assert set(summary["surge_share"]) >= {"share", "ci", "n_events", "B"}
    assert set(summary["sample_size_planted"]) == {e.key for e in rc.ESTIMANDS}
    assert list(summary["estimands"]) == [e.key for e in rc.ESTIMANDS]

    manifest = (first / "recovery.manifest.txt").read_text()
    assert manifest.startswith("experiment: recovery\ncommand: python -m "
                               "kalshi_mm.experiments.recovery --out .../first --reps 2 --B 40 ")
    assert "seeds: population: world seed 0, 40 events, bootstrap seed 20261113;" in manifest
    assert ("design: worlds ['planted', 'spec_null', 'flat_null'], sizes [6, 12], 2 replications "
            "per world (design 200), cluster bootstrap B = 40, population 40 events per world\n"
            ) in manifest
    assert "outputs: population.csv, coverage.csv, power.csv, summary.json, tables.md\n" in manifest
    assert "notes: synthetic data only" in manifest and str(tmp_path) not in manifest
