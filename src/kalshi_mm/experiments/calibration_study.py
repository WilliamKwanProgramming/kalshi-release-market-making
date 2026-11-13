"""Calibration study on synthetic data: does walk-forward recalibration pay in a favourite-longshot
world, and how many events does it take to see it (docs/model.md sections 8 and 9)?

The calibration world is its own world (it is not the order-flow world of ``kalshi_mm.world``).
Each event has a true log-odds ``x ~ N(0, 1.5^2)``, the outcome is ``y ~ Bernoulli(sigmoid(x))``
and the price at the horizon is ``sigmoid(beta * x + e)`` with ``e ~ N(0, 0.1^2)``. Events are one
week apart, one market each. ``beta = 1`` is a calibrated price up to the noise (the null cells);
``beta < 1`` is the favourite-longshot pattern: prices are too close to one half, low prices are
too high and high prices too low, and the logistic recalibration has the right form (slope about
``1 / beta``).

Grid (fixed in advance, no cell is added after a result): ``beta`` in {1.0, 0.9, 0.8, 0.7} by number
of events in {100, 200, 500, 1000}, 200 replications per cell. A replication draws the world,
runs :func:`kalshi_mm.calibration.walk_forward_recalibration` (all earlier events, at least 50,
refit every 10) and :func:`kalshi_mm.calibration.skill_interval` (cluster bootstrap with ``B``
resamples; the clusters are the events). Per cell the study reports the mean skill and the skill of
the population, the share of replications whose one-sided bootstrap p-value is below 0.05
("supported"), the mean fitted slope against its population value and the mean interval width.

The population values are computed by two-dimensional Gauss-Hermite quadrature over ``(x, e)``: the
population logistic regression of ``y`` on the clipped logit of the price (its coefficients are not
exactly ``(0, 1 / beta)``, because the noise ``e`` attenuates the slope a little), and the expected
Brier scores of the raw price and of the recalibrated one. The walk-forward skill is lower than the
population skill by the cost of estimating two coefficients from a short history.

Usage::

    python -m kalshi_mm.experiments.calibration_study --out results/calibration

Writes ``cells.csv`` (one row per cell), ``summary.json``, ``tables.md`` and
``calibration_study.manifest.txt``. Nothing here is real Kalshi data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy import optimize

from kalshi_mm.calibration import (
    RecalibrationError,
    skill_interval,
    walk_forward_recalibration,
)
from kalshi_mm.experiments.common import Manifest, mc_mean_se
from kalshi_mm.logit import logit, sigmoid

__all__ = [
    "BETAS",
    "CalibrationWorld",
    "EVENTS",
    "calibration_world",
    "population_calibration",
    "run_cell",
    "summarise_cell",
    "tables_markdown",
]

BASE_SEED = 20261113
BETAS = (1.0, 0.9, 0.8, 0.7)
EVENTS = (100, 200, 500, 1000)
REPS = 200
BOOTSTRAP_B = 500
X_SD = 1.5  # sd of the true log-odds
NOISE_SD = 0.1  # sd of the noise added to the log-odds of the price
SPACING_S = 168.0 * 3600.0  # one release a week
T0 = 1_700_000_000.0
CLIP = 0.01  # the clip of the regressor, the default of kalshi_mm.calibration
MIN_EVENTS = 50
REFIT_EVERY = 10
ALPHA = 0.05  # "supported" means a one-sided p-value below this
SEEDS = (
    f"base {BASE_SEED}; replication r of the cell (beta, n_events) draws its world from "
    f"default_rng([{BASE_SEED}, 1, round(100 beta), n_events, r]) and its bootstrap from "
    f"default_rng(s) with s = SeedSequence([{BASE_SEED}, 2, round(100 beta), n_events, r])"
    ".generate_state(1)[0]"
)


class CalibrationWorld(NamedTuple):
    """The events of one replication: true log-odds ``x``, ``price``, ``outcome``, release time."""

    x: np.ndarray
    price: np.ndarray
    outcome: np.ndarray
    release_ts: np.ndarray


def calibration_world(
    n_events: int,
    beta: float,
    rng: np.random.Generator,
    *,
    x_sd: float = X_SD,
    noise_sd: float = NOISE_SD,
) -> CalibrationWorld:
    """``n_events`` events, one market each: ``x ~ N(0, x_sd^2)``, ``y ~ Bernoulli(sigmoid(x))``,
    price ``sigmoid(beta x + e)`` with ``e ~ N(0, noise_sd^2)``, released one week apart."""
    x = rng.normal(0.0, x_sd, n_events)
    outcome = (rng.random(n_events) < sigmoid(x)).astype(np.int64)
    price = sigmoid(beta * x + rng.normal(0.0, noise_sd, n_events))
    return CalibrationWorld(x, price, outcome, T0 + SPACING_S * np.arange(n_events))


def population_calibration(
    beta: float,
    *,
    x_sd: float = X_SD,
    noise_sd: float = NOISE_SD,
    clip: float = CLIP,
    nodes_x: int = 120,
    nodes_e: int = 24,
) -> dict[str, float]:
    """Population coefficients and Brier scores of the calibration world, by quadrature.

    With ``z = beta x + e`` the logit of the price, ``p = sigmoid(z)``, ``q = sigmoid(x)`` the
    probability of the outcome and ``z_c`` the clipped logit, the population logistic regression
    solves ``E[(q - sigmoid(a + b z_c)) (1, z_c)] = 0`` (the expected score; ``y`` depends on ``x``
    only, so ``E[y | x, e] = q``). The expected Brier scores are ``E[q (1 - p)^2 + (1 - q) p^2]``
    for the raw price and the same with ``r = sigmoid(a + b z_c)`` for the recalibrated one; the
    skill is ``1 - recal / raw``. Expectations over ``(x, e)`` use the product rule with
    ``nodes_x`` by ``nodes_e`` nodes for the weight ``exp(-t^2 / 2)`` (numpy's rule overflows from
    about 400 nodes). The clip makes the integrand non-smooth at ``|z| = logit(1 - clip)``; for
    the default clip and the four betas of the study the default rule agrees with a 300 by 60 rule
    to ``3e-6`` in ``b`` and ``1e-9`` in the skill. Returns ``a``, ``b``, ``brier_raw``,
    ``brier_recal`` and ``bss``.
    """
    tx, wx = hermegauss(nodes_x)
    te, we = hermegauss(nodes_e)
    grid_x, grid_e = np.meshgrid(tx, te, indexing="ij")
    weight = np.outer(wx, we) / (2.0 * np.pi)  # the weights of E over (x, e), summing to one
    x = x_sd * grid_x
    z = beta * x + noise_sd * grid_e
    q = sigmoid(x)
    p = sigmoid(z)
    zc = np.clip(z, logit(clip), logit(1.0 - clip))
    design = np.stack([np.ones_like(zc), zc])

    def score(coef: np.ndarray) -> np.ndarray:
        r = sigmoid(coef[0] + coef[1] * zc)
        return np.einsum("ij,kij->k", weight * (q - r), design)

    def information(coef: np.ndarray) -> np.ndarray:
        r = sigmoid(coef[0] + coef[1] * zc)
        return np.einsum("ij,kij,lij->kl", weight * r * (1.0 - r), design, design)

    root = optimize.root(score, [0.0, 1.0 / beta], jac=information, method="hybr", tol=1e-13)
    if not root.success:
        raise RuntimeError(f"the population regression did not converge: {root.message}")
    a, b = (float(v) for v in root.x)
    r = sigmoid(a + b * zc)
    brier_raw = float(np.sum(weight * (q * (1.0 - p) ** 2 + (1.0 - q) * p**2)))
    brier_recal = float(np.sum(weight * (q * (1.0 - r) ** 2 + (1.0 - q) * r**2)))
    return {"a": a, "b": b, "brier_raw": brier_raw, "brier_recal": brier_recal,
            "bss": 1.0 - brier_recal / brier_raw}


def _seeds(beta: float, n_events: int, rep: int) -> tuple[list[int], int]:
    """The world stream and the bootstrap seed of replication ``rep`` of one cell (see SEEDS)."""
    key = [BASE_SEED, 1, round(100 * beta), n_events, rep]
    boot = np.random.SeedSequence([BASE_SEED, 2, *key[2:]]).generate_state(1)[0]
    return key, int(boot)


def run_cell(beta: float, n_events: int, reps: int, B: int) -> pd.DataFrame:
    """``reps`` replications of one cell: one row each with the walk-forward skill, its interval and
    p-value, the mean fitted coefficients and the number of scored events. A replication whose fit
    fails (:class:`RecalibrationError`) is kept as a row with ``failed`` true and NaN results."""
    rows = []
    for rep in range(reps):
        key, boot_seed = _seeds(beta, n_events, rep)
        world = calibration_world(n_events, beta, np.random.default_rng(key))
        row: dict[str, float | int | bool] = {"rep": rep, "failed": False}
        try:
            frame = walk_forward_recalibration(
                world.price, world.outcome, world.release_ts, min_events=MIN_EVENTS,
                refit_every=REFIT_EVERY, cluster=np.arange(n_events),
            )
            res = skill_interval(frame, frame["cluster"], B=B, seed=boot_seed)
        except RecalibrationError:
            rows.append({**row, "failed": True})
            continue
        rows.append({
            **row, "bss": res["bss"], "ci_lo": res["ci"][0], "ci_hi": res["ci"][1],
            "p_one_sided": res["p_one_sided"], "brier_raw": res["brier_raw"],
            "brier_recal": res["brier_recal"], "slope": float(frame["b"].mean()),
            "intercept": float(frame["a"].mean()), "n_scored": res["n"],
        })
    return pd.DataFrame(rows)


def summarise_cell(replications: pd.DataFrame, beta: float, n_events: int, B: int,
                   population: dict[str, float]) -> dict[str, float | int]:
    """One row of ``cells.csv``: means over the replications that did not fail, each with its Monte
    Carlo standard error ``sd / sqrt(n)`` (a share has ``sqrt(p (1 - p) / n)``)."""
    ok = replications[~replications["failed"].astype(bool)]
    n = len(ok)
    if n == 0:
        raise RuntimeError(f"every replication of the cell beta={beta}, n_events={n_events} failed")
    out: dict[str, float | int] = {
        "beta": beta, "n_events": n_events, "n_scored": n_events - MIN_EVENTS, "B": B,
        "reps": len(replications), "n_failed": len(replications) - n,
    }
    for name, column in (("bss", "bss"), ("slope", "slope"), ("intercept", "intercept")):
        out[f"{name}_mean"], out[f"{name}_se"] = mc_mean_se(ok[column])
    out["width_mean"], out["width_se"] = mc_mean_se(ok["ci_hi"] - ok["ci_lo"])
    supported = float((ok["p_one_sided"] < ALPHA).mean())
    out["supported"] = supported
    out["supported_se"] = float(np.sqrt(supported * (1.0 - supported) / n))
    out["brier_raw_mean"] = float(ok["brier_raw"].mean())
    out["brier_recal_mean"] = float(ok["brier_recal"].mean())
    out["bss_pop"], out["slope_pop"], out["intercept_pop"] = (
        population["bss"], population["b"], population["a"])
    out["slope_inverse_beta"] = 1.0 / beta
    return out


def _pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}"


def _table_skill(cells: pd.DataFrame, B: int) -> list[str]:
    """Table C1: skill, population skill, share supported and interval width of every cell."""
    reps = int(cells["reps"].iloc[0])
    null = cells[cells["beta"] == 1.0]
    null_text = ""
    if len(null):
        null_text = (
            f" In the null rows (beta = 1) \"supported\" is {_pct(null['supported'].min())}% to "
            f"{_pct(null['supported'].max())}% and the mean BSS is "
            f"{_pct(null['bss_mean'].min(), 2)}% to {_pct(null['bss_mean'].max(), 2)}%."
        )
    failed = int(cells["n_failed"].sum())
    failed_text = f" {failed} replications failed to fit and are left out." if failed else ""
    lines = [
        "### C1. Walk-forward Brier skill score by price slope and number of events "
        "(synthetic data)",
        "",
        f"Reading: mean walk-forward BSS in percent (Monte Carlo standard error in brackets) and "
        f"the share of {reps} replications with one-sided cluster-bootstrap p below {ALPHA} "
        f"(\"supported\", B = {B}); the fits use all earlier events, at least {MIN_EVENTS}, refit "
        f"every {REFIT_EVERY}.{null_text}{failed_text}",
        "",
        "| beta | events | scored | mean BSS, % | population BSS, % | supported, % | "
        "mean interval width, points |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cells.itertuples():
        lines.append(
            f"| {c.beta:.1f} | {c.n_events} | {c.n_scored} | {_pct(c.bss_mean, 2)} "
            f"({_pct(c.bss_se, 2)}) | {_pct(c.bss_pop, 2)} | {_pct(c.supported)} "
            f"({_pct(c.supported_se)}) | {_pct(c.width_mean, 2)} |"
        )
    lines += [
        "",
        "Does not show: whether real Kalshi prices have a favourite-longshot slope, or one that "
        f"the logistic form fits; both are planted here (x ~ N(0, {X_SD}^2), price noise "
        f"{NOISE_SD} on the log-odds scale).",
    ]
    return lines


def _table_slope(cells: pd.DataFrame) -> list[str]:
    """Table C2: the mean fitted slope and intercept against their population values."""
    reps = int(cells["reps"].iloc[0])
    lines = [
        "### C2. Fitted recalibration slope against its population value (synthetic data)",
        "",
        "Reading: coefficients of the walk-forward fits, averaged over each replication's scored "
        f"events and then over {reps} replications (Monte Carlo standard error in brackets); the "
        "population slope includes the noise e and the clip, 1 / beta is its noise-free value, "
        "and the population intercept is 0.",
        "",
        "| beta | events | mean fitted b | 1 / beta | population b | mean fitted a |",
        "|---|---|---|---|---|---|",
    ]
    for c in cells.itertuples():
        lines.append(
            f"| {c.beta:.1f} | {c.n_events} | {c.slope_mean:.3f} ({c.slope_se:.3f}) | "
            f"{c.slope_inverse_beta:.3f} | {c.slope_pop:.3f} | "
            f"{c.intercept_mean:.3f} ({c.intercept_se:.3f}) |"
        )
    lines += [
        "",
        "Does not show: whether the slope changes over time; in this world it is the same for "
        "every event.",
    ]
    return lines


def tables_markdown(cells: pd.DataFrame, B: int) -> str:
    """The text of ``tables.md``: a provenance comment, then table C1 (skill) and table C2 (slope),
    each a heading that says synthetic data, a ``Reading:`` paragraph, the Markdown table and a
    ``Does not show:`` paragraph. The numbers in the paragraphs are computed from ``cells``."""
    header = ["<!-- Generated by kalshi_mm.experiments.calibration_study; do not edit by hand. -->",
              ""]
    return "\n".join([*header, *_table_skill(cells, B), "", *_table_slope(cells)]) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Run the study of the module docstring and write its files to ``--out``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/calibration"))
    ap.add_argument("--reps", type=int, default=REPS, help="replications per cell")
    ap.add_argument("--B", type=int, default=BOOTSTRAP_B, help="bootstrap resamples per run")
    args = ap.parse_args(argv)
    manifest = Manifest("kalshi_mm.experiments.calibration_study", seeds=SEEDS, args=argv)

    population = {beta: population_calibration(beta) for beta in BETAS}
    rows = []
    for beta in BETAS:
        for n_events in EVENTS:
            reps = run_cell(beta, n_events, args.reps, args.B)
            rows.append(summarise_cell(reps, beta, n_events, args.B, population[beta]))
    cells = pd.DataFrame(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out / "cells.csv", index=False, float_format="%.8g")
    summary = {
        "experiment": "calibration_study",
        "data": "synthetic",
        "world": {"x_sd": X_SD, "noise_sd": NOISE_SD, "spacing_hours": SPACING_S / 3600.0,
                  "clip": CLIP},
        "walk_forward": {"min_events": MIN_EVENTS, "refit_every": REFIT_EVERY},
        "betas": list(BETAS), "events": list(EVENTS), "reps": args.reps, "B": args.B,
        "alpha": ALPHA, "population": {f"{beta:.1f}": v for beta, v in population.items()},
        "false_positive_rate": {
            str(int(r.n_events)): float(r.supported) for r in cells.itertuples() if r.beta == 1.0
        },
        "cells": json.loads(cells.to_json(orient="records")),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "tables.md").write_text(tables_markdown(cells, args.B))
    manifest.extra.append(
        f"design: beta in {list(BETAS)} by n_events in {list(EVENTS)}, {args.reps} replications "
        f"per cell, cluster bootstrap B = {args.B}"
    )
    manifest.write(args.out, "calibration_study", ["cells.csv", "summary.json", "tables.md"],
                   notes="synthetic data only; nothing here is real Kalshi data")


if __name__ == "__main__":
    main()
