"""The calibration study on synthetic data (kalshi_mm.experiments.calibration_study).

* The world is checked against its definition: moments of x and of the price noise, the outcome
  probability as a function of x (and not of the price), release times one week apart.
* The population values are checked three ways: the noise-free case against a one-dimensional
  closed form (the recalibrated probability is the true one, so the coefficients are (0, 1 / beta));
  a fine-grid integral that solves the same score equations itself, with the clip active; and large
  simulated samples within four standard errors.
* The replication loop is checked by re-running the documented recipe by hand; the summary row by
  hand-computed means; the tables by the parser that pastes them into the README.
"""

import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, optimize
from scipy.special import expit

from kalshi_mm.calibration import (
    RecalibrationError,
    fit_recalibration,
    skill_interval,
    walk_forward_recalibration,
)
from kalshi_mm.experiments import calibration_study as cs
from kalshi_mm.experiments.calibration_study import (
    BETAS,
    EVENTS,
    calibration_world,
    population_calibration,
    run_cell,
    summarise_cell,
    tables_markdown,
)
from kalshi_mm.logit import logit, sigmoid

Z = 4.0
WEEK = 7 * 24 * 3600.0
ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_make_tables():
    """The script that pastes tables into README.md, loaded by path (it is not a package)."""
    path = ROOT / "scripts" / "make_tables.py"
    spec = importlib.util.spec_from_file_location("make_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------- the world
def test_world_shapes_types_and_weekly_release_times():
    w = calibration_world(250, 0.8, np.random.default_rng(1))
    for column in (w.x, w.price, w.outcome, w.release_ts):
        assert column.shape == (250,)
    assert w.outcome.dtype == np.int64 and set(np.unique(w.outcome)) == {0, 1}
    assert np.all((w.price > 0.0) & (w.price < 1.0))
    np.testing.assert_array_equal(np.diff(w.release_ts), np.full(249, WEEK))


def test_world_is_reproducible_and_depends_on_the_generator():
    a = calibration_world(50, 0.9, np.random.default_rng(3))
    b = calibration_world(50, 0.9, np.random.default_rng(3))
    c = calibration_world(50, 0.9, np.random.default_rng(4))
    for left, right in zip(a, b, strict=True):
        np.testing.assert_array_equal(left, right)
    assert not np.array_equal(a.x, c.x) and not np.array_equal(a.price, c.price)


def test_without_noise_the_price_is_the_sigmoid_of_beta_x():
    w = calibration_world(400, 0.7, np.random.default_rng(2), noise_sd=0.0)
    np.testing.assert_allclose(w.price, expit(0.7 * w.x), rtol=1e-14, atol=0.0)


@pytest.mark.parametrize(
    "kwargs, x_sd, noise_sd",
    [({}, 1.5, 0.1), ({"x_sd": 0.7, "noise_sd": 0.3}, 0.7, 0.3)],
)
def test_world_moments_and_the_outcome_depends_on_x_only(kwargs, x_sd, noise_sd):
    """The defaults are x ~ N(0, 1.5^2) and price noise sd 0.1. A statistic is compared with its
    standard error: sd / sqrt(n) for a mean, sd / sqrt(2 n) for a standard deviation, 1 / sqrt(n)
    for the correlation of independent variables, sqrt(sum q (1 - q) w^2) for sum (y - q) w."""
    beta, n = 0.8, 400_000
    w = calibration_world(n, beta, np.random.default_rng(7), **kwargs)
    noise = logit(w.price) - beta * w.x
    assert abs(w.x.mean()) < Z * x_sd / np.sqrt(n)
    assert abs(w.x.std() - x_sd) < Z * x_sd / np.sqrt(2 * n)
    assert abs(noise.mean()) < Z * noise_sd / np.sqrt(n)
    assert abs(noise.std() - noise_sd) < Z * noise_sd / np.sqrt(2 * n)
    assert abs(np.corrcoef(noise, w.x)[0, 1]) < Z / np.sqrt(n)
    q = expit(w.x)
    resid = w.outcome - q
    for weight in (np.ones(n), w.x):  # the second one would see the slope if y used beta x
        se = np.sqrt(np.sum(q * (1.0 - q) * weight**2))
        assert abs(np.sum(resid * weight)) < Z * se
    assert abs(np.corrcoef(resid, noise)[0, 1]) < Z / np.sqrt(n)


# ----------------------------------------------------------------------------- the population
def noise_free_briers(beta, x_sd=1.5):
    """``E[q (1 - p)^2 + (1 - q) p^2]`` and ``E[q (1 - q)]`` for ``q = sigmoid(x)`` and
    ``p = sigmoid(beta x)``, by adaptive quadrature against the normal density."""
    def density(x):
        return np.exp(-x * x / (2.0 * x_sd**2)) / (x_sd * np.sqrt(2.0 * np.pi))

    def expectation(f):
        return integrate.quad(lambda x: density(x) * f(x), -14 * x_sd, 14 * x_sd,
                              epsabs=1e-14, epsrel=1e-13, limit=400)[0]

    raw = expectation(lambda x: expit(x) * (1 - expit(beta * x)) ** 2
                      + (1 - expit(x)) * expit(beta * x) ** 2)
    recal = expectation(lambda x: expit(x) * (1 - expit(x)))
    return raw, recal


@pytest.mark.parametrize("beta", [1.0, 0.9, 0.7])
def test_population_without_noise_matches_the_one_dimensional_closed_form(beta):
    pop = population_calibration(beta, noise_sd=0.0, clip=1e-12)
    raw, recal = noise_free_briers(beta)
    assert pop["a"] == pytest.approx(0.0, abs=1e-12)
    assert pop["b"] == pytest.approx(1.0 / beta, rel=1e-10)
    assert pop["brier_raw"] == pytest.approx(raw, rel=1e-11)
    assert pop["brier_recal"] == pytest.approx(recal, rel=1e-11)
    assert pop["bss"] == pytest.approx(1.0 - recal / raw, abs=1e-11)


def test_a_calibrated_price_has_no_population_skill():
    pop = population_calibration(1.0, noise_sd=0.0, clip=1e-12)
    assert pop["b"] == pytest.approx(1.0, rel=1e-10) and abs(pop["bss"]) < 1e-11
    assert pop["brier_recal"] == pytest.approx(pop["brier_raw"], rel=1e-11)


def test_noise_attenuates_the_population_slope_a_little_and_never_raises_it():
    for beta in BETAS:
        pop = population_calibration(beta)
        assert 0.98 / beta < pop["b"] < 1.0 / beta
        assert abs(pop["a"]) < 1e-12  # the world is symmetric in x
        assert pop["bss"] >= -1e-6  # the recalibrated score is at least as good as the raw one
    skill = [population_calibration(beta)["bss"] for beta in BETAS]
    assert skill == sorted(skill) and skill[-1] > 0.015  # more slope, more to gain


def grid_population(beta, x_sd, noise_sd, clip, half_width=9.0, nx=1601, ne=801):
    """The population coefficients and Brier scores by a Riemann sum on a fine grid over (x, e),
    solving the score equations with its own root finder (no Gauss-Hermite rule involved)."""
    x = np.linspace(-half_width * x_sd, half_width * x_sd, nx)
    e = np.linspace(-half_width * noise_sd, half_width * noise_sd, ne)
    wx = np.exp(-x**2 / (2 * x_sd**2)) / (x_sd * np.sqrt(2 * np.pi)) * (x[1] - x[0])
    we = np.exp(-e**2 / (2 * noise_sd**2)) / (noise_sd * np.sqrt(2 * np.pi)) * (e[1] - e[0])
    gx, ge = np.meshgrid(x, e, indexing="ij")
    weight = np.outer(wx, we)
    z = beta * gx + ge
    zc = np.clip(z, np.log(clip / (1 - clip)), np.log((1 - clip) / clip))
    q, p = expit(gx), expit(z)

    def score(c):
        r = expit(c[0] + c[1] * zc)
        return [np.sum(weight * (q - r)), np.sum(weight * (q - r) * zc)]

    root = optimize.root(score, [0.0, 1.0 / beta], method="hybr", tol=1e-12)
    a, b = root.x
    r = expit(a + b * zc)
    return {"a": a, "b": b, "brier_raw": np.sum(weight * (q * (1 - p) ** 2 + (1 - q) * p**2)),
            "brier_recal": np.sum(weight * (q * (1 - r) ** 2 + (1 - q) * r**2))}


def test_population_agrees_with_a_fine_grid_integral_for_the_study_parameters():
    pop = population_calibration(0.8)
    grid = grid_population(0.8, 1.5, 0.1, 0.01)
    assert pop["b"] == pytest.approx(grid["b"], abs=1e-6)
    assert pop["a"] == pytest.approx(grid["a"], abs=1e-9)
    assert pop["brier_raw"] == pytest.approx(grid["brier_raw"], abs=1e-9)
    assert pop["brier_recal"] == pytest.approx(grid["brier_recal"], abs=1e-9)


def test_population_with_an_active_clip_agrees_with_the_grid_and_differs_from_no_clip():
    """Noise sd 0.5 and a clip at 0.1 put 4% of the prices outside the clip. The clip moves the
    slope by 0.036 (the check below needs at least 0.02), the two integrals agree to 1e-3."""
    pop = population_calibration(0.8, noise_sd=0.5, clip=0.1)
    grid = grid_population(0.8, 1.5, 0.5, 0.1)
    assert pop["b"] == pytest.approx(grid["b"], abs=1e-3)
    assert pop["brier_raw"] == pytest.approx(grid["brier_raw"], abs=1e-9)
    assert pop["brier_recal"] == pytest.approx(grid["brier_recal"], abs=2e-5)
    unclipped = population_calibration(0.8, noise_sd=0.5, clip=1e-12)
    assert abs(unclipped["b"] - pop["b"]) > 0.02
    assert abs(unclipped["brier_recal"] - pop["brier_recal"]) > 1e-4


def test_population_node_counts_are_converged():
    fine = population_calibration(0.7, nodes_x=240, nodes_e=60)
    default = population_calibration(0.7)
    for key in default:
        assert default[key] == pytest.approx(fine[key], abs=1e-8)


@pytest.mark.parametrize("beta", [0.7, 1.0])
def test_a_large_sample_reproduces_the_population_values(beta):
    """200,000 events: the fitted coefficients lie within four standard errors (from the inverse
    information of the fit) of the population ones, the mean squared errors within four standard
    errors of the population Brier scores."""
    n = 200_000
    pop = population_calibration(beta)
    w = calibration_world(n, beta, np.random.default_rng([11, round(100 * beta)]))
    a, b = fit_recalibration(w.price, w.outcome)
    zc = np.clip(logit(w.price), logit(0.01), logit(0.99))
    r = sigmoid(a + b * zc)
    h = r * (1.0 - r)
    info = np.array([[h.sum(), (h * zc).sum()], [(h * zc).sum(), (h * zc**2).sum()]])
    se_a, se_b = np.sqrt(np.diag(np.linalg.inv(info)))
    assert abs(a - pop["a"]) < Z * se_a
    assert abs(b - pop["b"]) < Z * se_b
    for squares, target in (((w.price - w.outcome) ** 2, pop["brier_raw"]),
                            ((r - w.outcome) ** 2, pop["brier_recal"])):
        assert abs(squares.mean() - target) < Z * squares.std(ddof=1) / np.sqrt(n)


def test_walk_forward_skill_approaches_the_population_skill_in_a_long_history():
    """40,000 events, refit every 400. The fits start from 50 events, so the skill is below the
    population value by the cost of estimating two coefficients, about 2 ln(n) / n = 0.05 points of
    the Brier score; the interval width gives the sampling error (four standard errors plus 0.1
    points are allowed), and a skill of 1.7% is supported at the 5% level."""
    n = 40_000
    pop = population_calibration(0.7)
    w = calibration_world(n, 0.7, np.random.default_rng(12))
    frame = walk_forward_recalibration(w.price, w.outcome, w.release_ts, min_events=50,
                                       refit_every=400, cluster=np.arange(n))
    res = skill_interval(frame, frame["cluster"], B=200, seed=1)
    se = (res["ci"][1] - res["ci"][0]) / (2.0 * 1.96)
    assert abs(res["bss"] - pop["bss"]) < Z * se + 0.001
    assert res["p_one_sided"] < 0.05 and res["n"] == n - 50


# ----------------------------------------------------------------------------------- the cells
def test_seeds_follow_the_documented_formula_and_differ_between_replications():
    key, boot = cs._seeds(0.8, 200, 7)
    assert key == [20261113, 1, 80, 200, 7]
    assert boot == int(np.random.SeedSequence([20261113, 2, 80, 200, 7]).generate_state(1)[0])
    grid = [(beta, n, rep) for beta in BETAS for n in EVENTS for rep in range(30)]
    keys = {tuple(cs._seeds(*cell)[0]) for cell in grid}
    boots = {cs._seeds(*cell)[1] for cell in grid}
    assert len(keys) == len(boots) == len(grid)
    assert all(0 <= s < 2**32 for s in boots)


def test_run_cell_follows_the_documented_recipe():
    """The world, the walk-forward fit and the bootstrap of each replication, redone by hand with
    the seeds of the module docstring. The events are the clusters: the release time (one event a
    week) is a valid label for them."""
    beta, n, B = 0.8, 120, 60
    out = run_cell(beta, n, 3, B)
    assert list(out["rep"]) == [0, 1, 2] and not out["failed"].any()
    for rep in range(3):
        rng = np.random.default_rng([20261113, 1, 80, n, rep])
        x = rng.normal(0.0, 1.5, n)
        y = (rng.random(n) < expit(x)).astype(int)
        p = expit(beta * x + rng.normal(0.0, 0.1, n))
        frame = walk_forward_recalibration(p, y, 1_700_000_000.0 + WEEK * np.arange(n),
                                           min_events=50, refit_every=10)
        seed = int(np.random.SeedSequence([20261113, 2, 80, n, rep]).generate_state(1)[0])
        res = skill_interval(frame, frame["release_ts"], B=B, seed=seed)
        row = out.loc[rep]
        assert row["bss"] == res["bss"]
        assert (row["ci_lo"], row["ci_hi"]) == res["ci"]
        assert row["p_one_sided"] == res["p_one_sided"]
        assert (row["brier_raw"], row["brier_recal"]) == (res["brier_raw"], res["brier_recal"])
        assert row["slope"] == pytest.approx(frame["b"].mean(), rel=1e-12)
        assert row["intercept"] == pytest.approx(frame["a"].mean(), abs=1e-12)
        assert row["n_scored"] == res["n"] == n - 50 == len(frame)


def test_run_cell_is_deterministic_and_replications_are_nested_and_distinct():
    a = run_cell(0.9, 110, 4, 40)
    pd.testing.assert_frame_equal(a, run_cell(0.9, 110, 4, 40))
    pd.testing.assert_frame_equal(a.iloc[:2], run_cell(0.9, 110, 2, 40))  # the first two of four
    assert a["bss"].nunique() == 4 and a["slope"].nunique() == 4
    assert not run_cell(0.8, 110, 4, 40)["bss"].equals(a["bss"])  # another cell, other worlds


def test_a_failed_fit_is_kept_as_a_failed_row_and_left_out_of_the_summary(monkeypatch):
    real = cs.walk_forward_recalibration
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RecalibrationError("separation")
        return real(*args, **kwargs)

    monkeypatch.setattr(cs, "walk_forward_recalibration", flaky)
    out = run_cell(1.0, 100, 3, 40)
    assert list(out["failed"]) == [False, True, False]
    assert out.loc[1, ["bss", "slope", "ci_lo", "p_one_sided"]].isna().all()
    assert out.loc[[0, 2], ["bss", "slope", "ci_lo", "p_one_sided"]].notna().all().all()
    row = summarise_cell(out, 1.0, 100, 40, population_calibration(1.0))
    assert (row["reps"], row["n_failed"]) == (3, 1)
    assert row["bss_mean"] == pytest.approx(out.loc[[0, 2], "bss"].mean(), rel=1e-12)


def test_the_planted_slope_is_recovered_and_skill_is_supported_only_where_it_exists():
    """The mean fitted slope of 20 replications of 200 events is near 1 / beta (the small-history
    bias is at most a few hundredths), the mean skill at 1000 events is higher for beta = 0.7 than
    for the calibrated price, the calibrated price is almost never 'supported' and beta = 0.7
    is."""
    pops = {beta: population_calibration(beta) for beta in (1.0, 0.7)}
    slopes = {beta: summarise_cell(run_cell(beta, 200, 20, 40), beta, 200, 40, pops[beta])
              for beta in (1.0, 0.7)}
    assert abs(slopes[1.0]["slope_mean"] - 1.0) < 0.1
    assert abs(slopes[0.7]["slope_mean"] - 1.0 / 0.7) < 0.15
    assert slopes[0.7]["slope_mean"] - slopes[1.0]["slope_mean"] > 0.3
    long = {beta: summarise_cell(run_cell(beta, 1000, 40, 200), beta, 1000, 200, pops[beta])
            for beta in (1.0, 0.7)}
    gap = long[0.7]["bss_mean"] - long[1.0]["bss_mean"]
    assert gap > Z * np.hypot(long[0.7]["bss_se"], long[1.0]["bss_se"])
    assert long[1.0]["supported"] <= 0.05
    assert long[0.7]["supported"] > 0.05  # 24% in the full study: at least 3 of 40 replications


# ---------------------------------------------------------------------------------- the summary
def hand_replications():
    """Five replications, the third failed; the values are chosen so that every statistic of
    ``summarise_cell`` can be computed by hand. The fourth p-value is exactly the level."""
    nan = np.nan
    return pd.DataFrame({
        "rep": [0, 1, 2, 3, 4],
        "failed": [False, False, True, False, False],
        "bss": [0.01, 0.03, nan, -0.01, 0.05],
        "ci_lo": [-0.01, 0.00, nan, -0.04, 0.02],
        "ci_hi": [0.03, 0.06, nan, 0.02, 0.08],
        "p_one_sided": [0.20, 0.01, nan, 0.05, 0.049],
        "brier_raw": [0.18, 0.18, nan, 0.18, 0.18],
        "brier_recal": [0.17, 0.19, nan, 0.18, 0.16],
        "slope": [1.2, 1.4, nan, 1.3, 1.5],
        "intercept": [0.0, 0.1, nan, -0.1, 0.0],
        "n_scored": [150, 150, nan, 150, 150],
    })


def test_summarise_cell_by_hand():
    pop = {"a": 0.0, "b": 1.41, "bss": 0.017, "brier_raw": 0.18, "brier_recal": 0.177}
    row = summarise_cell(hand_replications(), 0.7, 200, 500, pop)
    # four usable replications; bss 0.01, 0.03, -0.01, 0.05: mean 0.02, sum of squares about the
    # mean 2e-3, sample variance 2e-3 / 3, standard error sqrt(2e-3 / 3 / 4)
    assert (row["beta"], row["n_events"], row["n_scored"], row["B"]) == (0.7, 200, 150, 500)
    assert (row["reps"], row["n_failed"]) == (5, 1)
    assert row["bss_mean"] == pytest.approx(0.02, abs=1e-15)
    assert row["bss_se"] == pytest.approx(np.sqrt(2e-3 / 3.0 / 4.0), rel=1e-12)
    # widths 0.04, 0.06, 0.06, 0.06: mean 0.055, deviations -0.015 and 0.005 (three times)
    assert row["width_mean"] == pytest.approx(0.055, abs=1e-15)
    assert row["width_se"] == pytest.approx(np.sqrt(3e-4 / 3.0 / 4.0), rel=1e-12)
    # supported: p strictly below 0.05, so 0.01 and 0.049 count and 0.05 does not: 2 of 4
    assert row["supported"] == 0.5 and row["supported_se"] == pytest.approx(0.25, rel=1e-14)
    assert row["slope_mean"] == pytest.approx(1.35, abs=1e-15)
    assert row["slope_se"] == pytest.approx(np.sqrt((0.0225 + 0.0025 + 0.0025 + 0.0225) / 3 / 4),
                                            rel=1e-12)
    assert row["intercept_mean"] == pytest.approx(0.0, abs=1e-15)
    assert row["brier_raw_mean"] == pytest.approx(0.18, rel=1e-14)
    assert row["brier_recal_mean"] == pytest.approx(0.175, rel=1e-14)  # (0.17+0.19+0.18+0.16) / 4
    assert (row["bss_pop"], row["slope_pop"], row["intercept_pop"]) == (0.017, 1.41, 0.0)
    assert row["slope_inverse_beta"] == pytest.approx(1.0 / 0.7, rel=1e-15)


def test_summarise_cell_with_no_usable_replication_is_an_error():
    reps = hand_replications().iloc[[2]]
    with pytest.raises(RuntimeError, match="failed"):
        summarise_cell(reps, 0.7, 200, 500, {"a": 0.0, "b": 1.4, "bss": 0.0})


# ---------------------------------------------------------------------------------- the tables
def fake_cells(failed_in_first_cell=0, betas=BETAS):
    """A cells frame with the columns that ``main`` writes, from seeded fake replications."""
    rng = np.random.default_rng(5)
    rows = []
    for beta in betas:
        for n_events in EVENTS:
            reps = 8
            bss = rng.normal(0.01 * (1 - beta) * 10 - 0.02, 0.02, reps)
            half = rng.uniform(0.01, 0.03, reps)
            frame = pd.DataFrame({
                "rep": np.arange(reps), "failed": False, "bss": bss, "ci_lo": bss - half,
                "ci_hi": bss + half, "p_one_sided": rng.uniform(0.0, 1.0, reps),
                "brier_raw": 0.18, "brier_recal": 0.18, "slope": rng.normal(1.3, 0.1, reps),
                "intercept": rng.normal(0.0, 0.02, reps), "n_scored": n_events - 50,
            })
            if failed_in_first_cell and not rows:
                frame.loc[:failed_in_first_cell - 1, "failed"] = True
                frame.loc[:failed_in_first_cell - 1, ["bss", "ci_lo", "ci_hi", "p_one_sided",
                                                       "slope", "intercept"]] = np.nan
            rows.append(summarise_cell(frame, beta, n_events, 500,
                                       {"a": 0.0, "b": 1.0 / beta, "bss": 0.01}))
    return pd.DataFrame(rows)


def test_tables_are_two_blocks_the_readme_parser_reads(tmp_path):
    text = tables_markdown(fake_cells(), 500)
    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "tables.md").write_text(text)
    tables = load_make_tables().read_tables(tmp_path)
    assert list(tables) == ["C1", "C2"]
    assert [ln for ln in text.splitlines() if ln.startswith("### ")] == [
        "### C1. Walk-forward Brier skill score by price slope and number of events "
        "(synthetic data)",
        "### C2. Fitted recalibration slope against its population value (synthetic data)",
    ]
    assert "$" not in text and text.endswith("\n") and "\n\n\n" not in text
    for table_id, block in tables.items():
        parts = block.split("\n\n")  # heading, Reading, table, Does not show
        assert len(parts) == 4, table_id
        assert parts[0].endswith("(synthetic data)") and parts[0].startswith(f"### {table_id}. ")
        assert parts[1].startswith("Reading: ") and "\n" not in parts[1]
        assert parts[2].startswith("| ") and len(parts[2].splitlines()) == 2 + 16
        assert parts[3].startswith("Does not show: ") and "\n" not in parts[3]


def test_skill_table_rows_and_reading_are_computed_from_the_cells():
    cells = fake_cells()
    cells.loc[:, "supported"] = 0.0
    cells.loc[cells["beta"] == 1.0, "supported"] = [0.0, 0.0, 0.005, 0.01]
    cells.loc[cells["beta"] == 1.0, "bss_mean"] = [-0.0276, -0.0189, -0.0103, -0.0065]
    cells.loc[(cells["beta"] == 0.7) & (cells["n_events"] == 500),
              ["bss_mean", "bss_se", "bss_pop", "supported", "supported_se", "width_mean"]] = [
        0.0078770322, 0.0010602719, 0.017118507, 0.08, 0.019183326, 0.057487273]
    lines = tables_markdown(cells, 500).splitlines()
    reading = next(ln for ln in lines if ln.startswith("Reading: mean walk-forward BSS"))
    for text in ("share of 8 replications", "p below 0.05", "B = 500",
                 "at least 50, refit every 10"):
        assert text in reading
    assert '"supported" is 0.0% to 1.0% and the mean BSS is -2.76% to -0.65%.' in reading
    assert "failed" not in reading
    assert "| 0.7 | 500 | 450 | 0.79 (0.11) | 1.71 | 8.0 (1.9) | 5.75 |" in lines
    assert sum(ln.startswith("| 1.0 |") for ln in lines) == 4 + 4  # the rows of C1 and of C2
    assert any(ln.startswith("Does not show:") and "1.5^2" in ln and "0.1 on" in ln for ln in lines)


def test_slope_table_row_is_formatted_from_the_cells():
    cells = fake_cells()
    mask = (cells["beta"] == 0.7) & (cells["n_events"] == 500)
    cells.loc[mask, ["slope_mean", "slope_se", "slope_pop", "intercept_mean", "intercept_se"]] = [
        1.4398466, 0.014121234, 1.4100463, -0.012742513, 0.010241248]
    lines = tables_markdown(cells, 500).splitlines()
    assert "| 0.7 | 500 | 1.440 (0.014) | 1.429 | 1.410 | -0.013 (0.010) |" in lines


def test_failed_replications_are_reported_and_a_study_without_null_rows_still_has_tables():
    text = tables_markdown(fake_cells(failed_in_first_cell=3), 500)
    reading = next(ln for ln in text.splitlines() if ln.startswith("Reading: mean walk-forward"))
    assert reading.endswith(" 3 replications failed to fit and are left out.")
    text = tables_markdown(fake_cells(betas=(0.9, 0.7)), 500)
    reading = next(ln for ln in text.splitlines() if ln.startswith("Reading: mean walk-forward"))
    assert "null rows" not in reading and "| 1.0 |" not in text


# ------------------------------------------------------------------------------------- the run
def test_main_passes_its_arguments_to_every_cell_and_reads_the_null_rows(tmp_path, monkeypatch):
    """With a stand-in for ``run_cell`` that records its arguments and calls every beta = 1 cell
    unsupported (p = 0.9) and every other cell supported (p = 0.01), the cells of the grid are
    visited in order with the requested replications and B, and the false-positive rates are those
    of the beta = 1 rows."""
    calls = []

    def stand_in(beta, n_events, reps, B):
        calls.append((beta, n_events, reps, B))
        frame = hand_replications()
        frame["p_one_sided"] = np.where(frame["failed"], np.nan, 0.9 if beta == 1.0 else 0.01)
        return frame

    monkeypatch.setattr(cs, "run_cell", stand_in)
    cs.main(["--out", str(tmp_path), "--reps", "7", "--B", "33"])
    assert calls == [(beta, n, 7, 33) for beta in BETAS for n in EVENTS]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["false_positive_rate"] == {str(n): 0.0 for n in EVENTS}
    assert (summary["reps"], summary["B"]) == (7, 33)
    cells = pd.read_csv(tmp_path / "cells.csv")
    assert set(cells.loc[cells["beta"] == 1.0, "supported"]) == {0.0}
    assert set(cells.loc[cells["beta"] < 1.0, "supported"]) == {1.0}
    assert set(cells["B"]) == {33}


def test_the_design_constants_are_the_ones_the_docstring_states():
    assert BETAS == (1.0, 0.9, 0.8, 0.7) and EVENTS == (100, 200, 500, 1000)
    assert (cs.REPS, cs.MIN_EVENTS, cs.REFIT_EVERY, cs.BOOTSTRAP_B) == (200, 50, 10, 500)
    assert cs.CLIP == 0.01 and cs.ALPHA == 0.05 and (cs.X_SD, cs.NOISE_SD) == (1.5, 0.1)
    doc = " ".join(cs.__doc__.split())
    assert "``beta`` in {1.0, 0.9, 0.8, 0.7}" in doc and "{100, 200, 500, 1000}" in doc
    assert "200 replications per cell" in doc and "at least 50, refit every 10" in doc
    assert "``x ~ N(0, 1.5^2)``" in doc and "``e ~ N(0, 0.1^2)``" in doc


def test_main_writes_every_file_and_is_reproducible(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for out in (first, second):
        cs.main(["--out", str(out), "--reps", "2", "--B", "50"])
    assert sorted(p.name for p in first.iterdir()) == [
        "calibration_study.manifest.txt", "cells.csv", "summary.json", "tables.md"]
    cells = pd.read_csv(first / "cells.csv")
    assert len(cells) == len(BETAS) * len(EVENTS)
    assert list(cells["beta"].unique()) == list(BETAS) and list(cells["n_events"][:4]) == [
        100, 200, 500, 1000]
    assert set(cells["reps"]) == {2} and set(cells["B"]) == {50} and cells["n_failed"].sum() == 0
    for name in ("cells.csv", "tables.md", "summary.json"):  # the same command, the same bytes
        assert (first / name).read_bytes() == (second / name).read_bytes()

    summary = json.loads((first / "summary.json").read_text())
    assert summary["experiment"] == "calibration_study" and summary["data"] == "synthetic"
    assert (summary["reps"], summary["B"], summary["alpha"]) == (2, 50, 0.05)
    assert summary["walk_forward"] == {"min_events": 50, "refit_every": 10}
    assert len(summary["cells"]) == 16
    assert list(summary["population"]) == ["1.0", "0.9", "0.8", "0.7"]
    for beta in BETAS:  # every cell carries the population values of its own beta
        pop = population_calibration(beta)
        assert summary["population"][f"{beta:.1f}"]["b"] == pytest.approx(pop["b"])
        rows = cells[cells["beta"] == beta]
        assert rows["bss_pop"].to_numpy() == pytest.approx(pop["bss"], rel=1e-6)
        assert rows["slope_pop"].to_numpy() == pytest.approx(pop["b"], rel=1e-6)
    null = cells[cells["beta"] == 1.0]
    assert summary["false_positive_rate"] == {
        str(n): s for n, s in zip(null["n_events"], null["supported"], strict=True)}

    manifest = (first / "calibration_study.manifest.txt").read_text()
    assert manifest.startswith("experiment: calibration_study\ncommand: python -m "
                               "kalshi_mm.experiments.calibration_study --out .../first")
    assert "--reps 2 --B 50\n" in manifest and "seeds: base 20261113;" in manifest
    assert ("design: beta in [1.0, 0.9, 0.8, 0.7] by n_events in [100, 200, 500, 1000], "
            "2 replications per cell, cluster bootstrap B = 50\n") in manifest
    assert "outputs: cells.csv, summary.json, tables.md\n" in manifest
    assert "notes: synthetic data only" in manifest and str(tmp_path) not in manifest
    assert "population BSS" in (first / "tables.md").read_text()
