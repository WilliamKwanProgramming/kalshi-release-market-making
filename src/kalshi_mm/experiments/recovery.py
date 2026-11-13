"""Recovery of the release effect in the synthetic world (docs/model.md sections 5 and 10, P1).

The question: does ``markout.release_effect`` recover the planted release effect, how far is the
tape proxy from the latent mid, do its intervals cover, and how many events does a test need? The
answer is for the world of ``kalshi_mm.world`` only; nothing here is real Kalshi data.

Worlds (the default parameters unless stated)::

    planted    pi1 = 0.20, surge = 20                 informed share and trade-rate surge
    spec_null  pi1 = 0                                the surge alone, no informed-share profile
    flat_null  pi1 = 0 and surge = 0                  neither

Estimands, each ``D = mean(near) - mean(quiet)`` with quiet = (360, inf) minutes to release::

    (a) raw D, near (0, 30]          the quantity of protocol P1
    (b) scaled D, near (0, 30]       adverse selection / (4 m (1 - m)), cents at a price of 50 cents
    (c) raw D, near (5, 30]          no trade whose horizon is cut by the release
    (d) scaled D, near (5, 30]       the cleaner read; added to the three that were asked for

**Population values** come from 3,000 events with seed 0 in each world: the bucket means of the
tape adverse selection, its scaled version, the oracle adverse selection and the analytic price
impact, and ``release_effect`` for each estimand. The oracle of trade ``i`` is ``side * (latent mid
before the first trade at or after t_i + h, minus latent_mid_pre_i)``; when no trade is left at or
after ``t_i + h`` before the release, the latent mid after the last trade (``latent_mid_post``) is
used. That is the reading under which the oracle has the same expectation as the price impact:
the public belief is a martingale and the arrival times do not depend on the outcome, so the
expected change of the latent mid from just after trade ``i`` to any later trade, or to the end,
is zero. Only the oracle part of this module reads columns that start with ``latent_``; the
replications use the public tape.

**Replications** simulate 400 events once per world and replication; the first ``n`` events form
the world of ``n`` events (every event has its own random stream), so one simulation serves every
size. Coverage and power come from the same replications. Per size and estimand: the cluster
bootstrap interval and one-sided p-value of ``release_effect`` (``B`` resamples inside the
replication) and a naive interval that treats trades as independent. Coverage is that of the
population value of the same world and estimand, which is itself estimated from 3,000 events.

The design asks for 200 replications. On a shared machine with two CPUs and a load of 2 to 3, 200
replications took 302 s of CPU time and 438 s of wall time and 100 took 152 s and 195 s, against a
budget of 360 s, so ``REPS`` is 100 (the rule was: cut to 100, never below, and say so in the table
notes). ``--reps 200`` restores the design; the first 100 replications are the same in both runs.

Usage::

    python -m kalshi_mm.experiments.recovery --out results/markouts

Writes ``population.csv``, ``coverage.csv``, ``power.csv``, ``summary.json``, ``tables.md`` and
``recovery.manifest.txt``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import NormalDist
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from kalshi_mm.experiments.common import Manifest
from kalshi_mm.markout import (
    BUCKET_LABELS,
    per_event_sums,
    release_effect,
    scaled_adverse_selection,
    trade_markouts,
)
from kalshi_mm.stats import cluster_weights, percentile_interval, ratio_of_sums
from kalshi_mm.world import World, WorldParams, generate_world, informed_share

__all__ = [
    "ESTIMANDS",
    "WORLDS",
    "Estimand",
    "Population",
    "RecoveryResults",
    "bootstrap_seed",
    "bucket_table",
    "coverage_table",
    "effects_frame",
    "interval_sign",
    "naive_interval",
    "oracle_adverse_selection",
    "population_world",
    "power_table",
    "price_impact",
    "public_markouts",
    "run_replication",
    "run_study",
    "saturation_share",
    "sample_size",
    "seeds_text",
    "surge_share",
    "tables_markdown",
    "world_params",
]

BASE_SEED = 20261113
POP_SEED = 0
POP_EVENTS = 3000
REPS_DESIGN = 200
REPS = 100
SIZES = (25, 50, 100, 200, 400)
BOOTSTRAP_B = 1000
ALPHA = 0.05
POWER_TARGET = 0.8
HORIZON_S = 300.0
QUIET = (360.0, np.inf)
SCALED = "adverse_selection_scaled"
CENTS = 100.0

WORLDS: dict[str, dict[str, float]] = {
    "planted": {},
    "spec_null": {"pi1": 0.0},
    "flat_null": {"pi1": 0.0, "surge": 0.0},
}


class Estimand(NamedTuple):
    """One release effect: the column whose means are compared and the near window ``(lo, hi]``."""

    key: str
    label: str
    column: str
    near: tuple[float, float]


ESTIMANDS = (
    Estimand("raw_0_30", "(a) raw D, near (0, 30]", "adverse_selection", (0.0, 30.0)),
    Estimand("scaled_0_30", "(b) scaled D, near (0, 30]", SCALED, (0.0, 30.0)),
    Estimand("raw_5_30", "(c) raw D, near (5, 30]", "adverse_selection", (5.0, 30.0)),
    Estimand("scaled_5_30", "(d) scaled D, near (5, 30]", SCALED, (5.0, 30.0)),
)
LABELS = {e.key: e.label for e in ESTIMANDS}


def world_params(world: str, n_events: int) -> WorldParams:
    """The parameters of one of the three worlds with ``n_events`` events."""
    return dataclasses.replace(WorldParams(), n_events=n_events, **WORLDS[world])


def public_markouts(world: World) -> pd.DataFrame:
    """Markouts of the tape of ``world`` computed from its public columns only, with the scaled
    adverse selection (:data:`SCALED`) and ``event_no``, the position of the trade's event in the
    world (0 for the first event)."""
    tape = world.tape[[c for c in world.tape.columns if not str(c).startswith("latent_")]]
    markouts = trade_markouts(tape, world.events, world.markets, horizon_s=HORIZON_S)
    return markouts.assign(
        **{SCALED: scaled_adverse_selection(markouts)},
        event_no=pd.Index(world.events["event"]).get_indexer(markouts["event"]),
    )


# ----------------------------------------------------------------------------- oracle part
# Everything from here to the end of the section reads the latent columns of the tape.
def oracle_adverse_selection(tape: pd.DataFrame, *, horizon_s: float = HORIZON_S) -> np.ndarray:
    """Oracle adverse selection of every trade of a synthetic tape, in USD per contract.

    For trade ``i``: ``side * (m_after - latent_mid_pre_i)`` with ``m_after`` the ``latent_mid_pre``
    of the first trade of the market at or after ``ts_i + horizon_s`` and, when there is none,
    the ``latent_mid_post`` of the last trade of the market (the horizon is truncated at the
    release). ``tape`` must be sorted by ``(market, ts, seq)``; the result has one value per row.
    """
    ts = tape["ts"].to_numpy(dtype=np.float64)
    side = tape["side"].to_numpy(dtype=np.float64)
    pre = tape["latent_mid_pre"].to_numpy(dtype=np.float64)
    post = tape["latent_mid_post"].to_numpy(dtype=np.float64)
    out = np.empty(len(tape))
    if len(tape) == 0:
        return out
    codes, _ = pd.factorize(tape["market"], sort=True)
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(tape)], strict=True):
        first = np.searchsorted(ts[a:b], ts[a:b] + horizon_s, side="left")
        later = first < b - a
        after = np.where(later, pre[a:b][np.minimum(first, b - a - 1)], post[b - 1])
        out[a:b] = side[a:b] * (after - pre[a:b])
    return out


def price_impact(tape: pd.DataFrame) -> np.ndarray:
    """Analytic price impact of every trade, ``side * (latent_mid_post - latent_mid_pre)``."""
    side = tape["side"].to_numpy(dtype=np.float64)
    return side * (tape["latent_mid_post"].to_numpy() - tape["latent_mid_pre"].to_numpy())


def saturation_share(tape: pd.DataFrame, lo: float = 0.01, hi: float = 0.99) -> float:
    """Share of events whose latent mid after their last trade lies outside ``[lo, hi]``."""
    last = tape.groupby("event", sort=False)["latent_mid_post"].last()
    return float(((last < lo) | (last > hi)).mean())


def bucket_table(
    markouts: pd.DataFrame, oracle: np.ndarray, impact: np.ndarray, params: WorldParams
) -> pd.DataFrame:
    """Population means per bucket, over the trades whose tape adverse selection is defined.

    ``oracle`` and ``impact`` have one value per row of ``markouts``. One row per bucket of
    :data:`BUCKET_LABELS`, then ``near (0,30]`` (all trades of the first three buckets) and ``D``
    (that row minus the last bucket, the quiet window). Columns: ``n_trades``, ``tau_mean_min``
    (mean minutes to release), ``tape_as_cents``, ``oracle_as_cents``, ``bias_cents`` (tape minus
    oracle), ``impact_cents``, ``scaled_as_cents`` (tape adverse selection divided by 4 m (1 - m),
    over the trades whose mid is in [0.03, 0.97]), ``pi_at_mean_tau`` (the planted informed share
    at the mean time to release). Means are over trades, in cents per contract.
    """
    tau = markouts["tau_min"].to_numpy(dtype=np.float64)
    bucket = markouts["bucket"].to_numpy()
    tape = markouts["adverse_selection"].to_numpy(dtype=np.float64)
    scaled = markouts[SCALED].to_numpy(dtype=np.float64)
    defined = ~np.isnan(tape)
    windows = [(label, bucket == k) for k, label in enumerate(BUCKET_LABELS)]
    windows.append(("near (0,30]", (tau > 0.0) & (tau <= 30.0)))
    rows = []
    for label, in_window in windows:
        use = in_window & defined
        mean_tau = float(tau[use].mean()) if use.any() else np.nan
        has_scaled = in_window & ~np.isnan(scaled)
        row = {
            "label": label,
            "n_trades": int(use.sum()),
            "tau_mean_min": mean_tau,
            "tape_as_cents": CENTS * float(tape[use].mean()) if use.any() else np.nan,
            "oracle_as_cents": CENTS * float(oracle[use].mean()) if use.any() else np.nan,
            "impact_cents": CENTS * float(impact[use].mean()) if use.any() else np.nan,
            "scaled_as_cents": CENTS * float(scaled[has_scaled].mean()) if has_scaled.any()
            else np.nan,
            "pi_at_mean_tau": float(informed_share(mean_tau / 60.0, params)),
        }
        row["bias_cents"] = row["tape_as_cents"] - row["oracle_as_cents"]
        rows.append(row)
    near, quiet = rows[len(BUCKET_LABELS)], rows[len(BUCKET_LABELS) - 1]
    rows.append({
        key: near[key] - quiet[key] if key not in ("label", "n_trades", "tau_mean_min") else None
        for key in near
    } | {"label": "D = near - quiet", "n_trades": 0, "tau_mean_min": np.nan})
    return pd.DataFrame(rows)[
        ["label", "n_trades", "tau_mean_min", "tape_as_cents", "oracle_as_cents", "bias_cents",
         "impact_cents", "scaled_as_cents", "pi_at_mean_tau"]
    ]


class Population(NamedTuple):
    """The large simulation of one world: bucket table, ``release_effect`` of each estimand, the
    saturation share and the per-event sums of the raw adverse selection (for the surge share)."""

    world: str
    params: WorldParams
    n_events: int
    n_trades: int
    buckets: pd.DataFrame
    effects: dict[str, dict[str, Any]]
    saturation: float
    sums: dict[str, np.ndarray]


def population_world(world: str, n_events: int, seed: int, B: int) -> Population:
    """Simulate ``n_events`` events of ``world`` and compute every population quantity."""
    params = world_params(world, n_events)
    sim = generate_world(params, seed)
    markouts = public_markouts(sim)
    effects = {
        e.key: release_effect(markouts, near=e.near, quiet=QUIET, column=e.column, B=B,
                              seed=BASE_SEED)
        for e in ESTIMANDS
    }
    tau = markouts["tau_min"].to_numpy(dtype=np.float64)
    near = per_event_sums(markouts, (tau > 0.0) & (tau <= 30.0), "adverse_selection")
    quiet = per_event_sums(markouts, tau > QUIET[0], "adverse_selection")
    index = markouts.index
    buckets = bucket_table(
        markouts,
        pd.Series(oracle_adverse_selection(sim.tape), index=sim.tape.index).loc[index].to_numpy(),
        pd.Series(price_impact(sim.tape), index=sim.tape.index).loc[index].to_numpy(),
        params,
    )
    return Population(
        world, params, n_events, len(sim.tape), buckets, effects, saturation_share(sim.tape),
        {"near_sum": near["sum"].to_numpy(), "near_n": near["n"].to_numpy(),
         "quiet_sum": quiet["sum"].to_numpy(), "quiet_n": quiet["n"].to_numpy()},
    )


def interval_sign(lo: float, hi: float) -> str:
    """"positive" if the interval lies above zero, "negative" if below, else "zero"."""
    return "positive" if lo > 0.0 else "negative" if hi < 0.0 else "zero"


def effects_frame(populations: Mapping[str, Population]) -> pd.DataFrame:
    """One row per world and estimand: ``D`` with its cluster-bootstrap interval, the near and
    quiet means (cents per contract), the number of values and the sign of the interval."""
    rows = []
    for world, pop in populations.items():
        for e in ESTIMANDS:
            r = pop.effects[e.key]
            lo, hi = CENTS * r["D_ci"][0], CENTS * r["D_ci"][1]
            rows.append({
                "world": world, "estimand": e.key, "D_cents": CENTS * r["D"], "lo_cents": lo,
                "hi_cents": hi, "mean_near_cents": CENTS * r["mean_near"],
                "mean_quiet_cents": CENTS * r["mean_quiet"], "n_near": r["n_near"],
                "n_quiet": r["n_quiet"], "n_events": r["n_events"], "sign": interval_sign(lo, hi),
            })
    return pd.DataFrame(rows)


def surge_share(planted: Population, spec_null: Population, B: int, seed: int = BASE_SEED) -> dict:
    """Share of the planted raw ``D`` near (0, 30] that the surge alone gives:
    ``D(spec_null) / D(planted)`` from the large simulations, with a percentile interval from a
    paired cluster bootstrap (the same event weights for both worlds, which share the seed, so an
    event has the same outcome and trade times in both)."""
    a, b = planted.sums, spec_null.sums
    if a["near_n"].shape != b["near_n"].shape:
        raise ValueError("the two populations must have the same events")
    point = spec_null.effects["raw_0_30"]["D"] / planted.effects["raw_0_30"]["D"]
    weights = cluster_weights(len(a["near_n"]), B, np.random.default_rng(seed))

    def effect(s: dict[str, np.ndarray]) -> np.ndarray:
        return (ratio_of_sums(s["near_sum"], s["near_n"], weights)
                - ratio_of_sums(s["quiet_sum"], s["quiet_n"], weights))

    with np.errstate(divide="ignore", invalid="ignore"):
        share = effect(b) / effect(a)
    usable = np.isfinite(share)
    return {"share": float(point), "ci": percentile_interval(share[usable]),
            "n_events": int(len(a["near_n"])), "B": int(B), "B_undefined": int(B - usable.sum()),
            "seed": int(seed)}


# ------------------------------------------------------------------------------- replications
def naive_interval(
    markouts: pd.DataFrame, column: str, near: tuple[float, float],
    quiet: tuple[float, float] = QUIET, level: float = 0.95,
) -> tuple[float, float, float]:
    """``(D, lo, hi)`` treating every trade as an independent draw: the difference of the near and
    quiet means of ``column`` (non-NaN values, windows ``(lo, hi]`` of ``tau_min``) with the normal
    interval ``D +- z sqrt(s_near^2 / n_near + s_quiet^2 / n_quiet)`` from the trade-level sample
    variances. NaN end points when a window has fewer than two values."""
    tau = markouts["tau_min"].to_numpy(dtype=np.float64)
    values = markouts[column].to_numpy(dtype=np.float64)
    present = ~np.isnan(values)
    a = values[present & (tau > near[0]) & (tau <= near[1])]
    b = values[present & (tau > quiet[0]) & (tau <= quiet[1])]
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    d = float(a.mean() - b.mean())
    if a.size < 2 or b.size < 2:
        return d, float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    se = float(np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size))
    return d, d - z * se, d + z * se


def seeds_text(pop_events: int) -> str:
    """The seeds of a run, for the manifest."""
    return (
        f"population: world seed {POP_SEED}, {pop_events} events, bootstrap seed {BASE_SEED}; "
        f"replication r: world seed {BASE_SEED} + r (event e draws from default_rng([seed, e])), "
        f"bootstrap seed SeedSequence([{BASE_SEED}, 2, r]).generate_state(1)[0] in every call of "
        f"release_effect of that replication; surge share: default_rng({BASE_SEED})"
    )


def bootstrap_seed(rep: int) -> int:
    """The seed of every ``release_effect`` call of replication ``rep``."""
    return int(np.random.SeedSequence([BASE_SEED, 2, rep]).generate_state(1)[0])


def run_replication(world: str, rep: int, sizes: Sequence[int], B: int) -> pd.DataFrame:
    """One replication of one world: simulate ``max(sizes)`` events with seed ``BASE_SEED + rep``
    and, for each size ``n`` and estimand, apply ``release_effect`` and :func:`naive_interval` to
    the markouts of the first ``n`` events. One row per size and estimand with ``D``, ``lo`` and
    ``hi`` (the cluster interval), ``p`` (one-sided), ``naive_lo`` and ``naive_hi``, in USD."""
    sim = generate_world(world_params(world, max(sizes)), BASE_SEED + rep)
    markouts = public_markouts(sim)
    # integer event labels in world order: the same sort order as the ids, much faster to group
    slim = markouts[["event_no", "tau_min", "adverse_selection", SCALED]].rename(
        columns={"event_no": "event"})
    seed = bootstrap_seed(rep)
    rows = []
    for n in sizes:
        first = slim[slim["event"] < n]
        for e in ESTIMANDS:
            r = release_effect(first, near=e.near, quiet=QUIET, column=e.column, B=B, seed=seed)
            _, naive_lo, naive_hi = naive_interval(first, e.column, e.near)
            rows.append({
                "world": world, "rep": rep, "n_events": n, "estimand": e.key, "D": r["D"],
                "lo": r["D_ci"][0], "hi": r["D_ci"][1], "p": r["p_one_sided"],
                "naive_lo": naive_lo, "naive_hi": naive_hi,
            })
    return pd.DataFrame(rows)


def _cells(replications: pd.DataFrame):
    """``(world, estimand, n_events, rows)`` for every cell that has replications, in the order of
    WORLDS, ESTIMANDS and increasing size."""
    for world in WORLDS:
        for e in ESTIMANDS:
            for n in sorted(replications.loc[replications["world"] == world, "n_events"].unique()):
                mask = ((replications["world"] == world) & (replications["estimand"] == e.key)
                        & (replications["n_events"] == n))
                if mask.any():
                    yield world, e.key, int(n), replications[mask]


def _share_se(share: float, n: int) -> float:
    return float(np.sqrt(share * (1.0 - share) / n))


def coverage_table(replications: pd.DataFrame, population: Mapping[tuple[str, str], float]
                   ) -> pd.DataFrame:
    """Coverage of the population value (USD per contract, keyed by ``(world, estimand)``) by the
    cluster-bootstrap and the naive intervals, per world, estimand and size: the shares, their
    Monte Carlo standard errors ``sqrt(c (1 - c) / reps)``, the mean widths and the mean and
    standard deviation of ``D`` over the replications (cents per contract)."""
    rows = []
    for world, key, n, cell in _cells(replications):
        truth = population[(world, key)]
        n_reps = len(cell)
        cluster = float(((cell["lo"] <= truth) & (truth <= cell["hi"])).mean())
        naive = float(((cell["naive_lo"] <= truth) & (truth <= cell["naive_hi"])).mean())
        rows.append({
            "world": world, "estimand": key, "n_events": n, "reps": n_reps,
            "pop_D_cents": CENTS * truth, "mean_D_cents": CENTS * float(cell["D"].mean()),
            "sd_D_cents": CENTS * float(cell["D"].std(ddof=1)) if n_reps > 1 else np.nan,
            "cover_cluster": cluster, "cover_cluster_se": _share_se(cluster, n_reps),
            "cover_naive": naive, "cover_naive_se": _share_se(naive, n_reps),
            "width_cluster_cents": CENTS * float((cell["hi"] - cell["lo"]).mean()),
            "width_naive_cents": CENTS * float((cell["naive_hi"] - cell["naive_lo"]).mean()),
        })
    return pd.DataFrame(rows)


def power_table(replications: pd.DataFrame, effects: pd.DataFrame) -> pd.DataFrame:
    """Share of replications with ``p`` below :data:`ALPHA` per world, estimand and size, with its
    Monte Carlo standard error, next to the population ``D`` of that world and estimand (cents) and
    the sign of its interval, from ``effects`` (:func:`effects_frame`)."""
    truth = effects.set_index(["world", "estimand"])
    rows = []
    for world, key, n, cell in _cells(replications):
        reject = float((cell["p"] < ALPHA).mean())
        rows.append({
            "world": world, "estimand": key, "n_events": n, "reps": len(cell),
            "pop_D_cents": float(truth.loc[(world, key), "D_cents"]),
            "pop_sign": str(truth.loc[(world, key), "sign"]),
            "reject": reject, "reject_se": _share_se(reject, len(cell)),
        })
    return pd.DataFrame(rows)


def sample_size(power: pd.DataFrame, world: str = "planted", level: float = POWER_TARGET
                ) -> dict[str, int | None]:
    """Per estimand, the smallest ``n_events`` at which the rejection rate in ``world`` exceeds
    ``level`` (strictly), or None when no size tried reaches it."""
    out: dict[str, int | None] = {}
    for e in ESTIMANDS:
        rows = power[(power["world"] == world) & (power["estimand"] == e.key)]
        enough = rows.loc[rows["reject"] > level, "n_events"]
        out[e.key] = int(enough.min()) if len(enough) else None
    return out


# ----------------------------------------------------------------------------------- the study
class RecoveryResults(NamedTuple):
    """Everything the tables and files are written from."""

    populations: dict[str, Population]
    buckets: pd.DataFrame  # the bucket tables of the worlds, with a ``world`` column
    effects: pd.DataFrame
    surge: dict[str, Any]
    coverage: pd.DataFrame
    power: pd.DataFrame
    sample: dict[str, int | None]
    design: dict[str, Any]


def run_study(reps: int, B: int, pop_events: int, sizes: Sequence[int]) -> RecoveryResults:
    """Population values of the three worlds, then ``reps`` replications of each."""
    populations = {w: population_world(w, pop_events, POP_SEED, B) for w in WORLDS}
    effects = effects_frame(populations)
    surge = surge_share(populations["planted"], populations["spec_null"], B)
    replications = pd.concat(
        [run_replication(w, r, sizes, B) for w in WORLDS for r in range(reps)], ignore_index=True)
    truth = {(w, e.key): p.effects[e.key]["D"] for w, p in populations.items() for e in ESTIMANDS}
    power = power_table(replications, effects)
    buckets = pd.concat(
        [p.buckets.assign(world=w) for w, p in populations.items()], ignore_index=True)
    design = {"reps": reps, "reps_design": REPS_DESIGN, "B": B, "pop_events": pop_events,
              "sizes": [int(s) for s in sizes]}
    return RecoveryResults(populations, buckets, effects, surge,
                           coverage_table(replications, truth), power, sample_size(power), design)


# ---------------------------------------------------------------------------------- the tables
def _signed(x: float, digits: int = 2) -> str:
    return f"{x:+.{digits}f}"


def _share(x: float, digits: int = 0) -> str:
    return f"{100.0 * x:.{digits}f}"


def _effect(res: RecoveryResults, world: str, key: str) -> Mapping[str, Any]:
    rows = res.effects[(res.effects["world"] == world) & (res.effects["estimand"] == key)]
    return rows.iloc[0]


def _bucket_lines(res: RecoveryResults, worlds: Sequence[str]) -> list[str]:
    head = ["minutes to release", "trades", "tape AS", "oracle AS", "tape - oracle", "impact",
            "scaled tape AS", "informed share"]
    if len(worlds) > 1:
        head.insert(0, "world")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for world in worlds:
        for r in res.buckets[res.buckets["world"] == world].itertuples():
            cells = [
                r.label, "" if r.n_trades == 0 else f"{r.n_trades}", _signed(r.tape_as_cents),
                _signed(r.oracle_as_cents), _signed(r.bias_cents), _signed(r.impact_cents),
                _signed(r.scaled_as_cents), f"{r.pi_at_mean_tau:.3f}",
            ]
            lines.append("| " + " | ".join(([world] if len(worlds) > 1 else []) + cells) + " |")
    return lines


def _bucket_row(res: RecoveryResults, world: str, label: str) -> Mapping[str, Any]:
    rows = res.buckets[(res.buckets["world"] == world) & (res.buckets["label"] == label)]
    return rows.iloc[0]


def _table_planted(res: RecoveryResults) -> list[str]:
    near, quiet = (_bucket_row(res, "planted", "near (0,30]"), _bucket_row(res, "planted", ">360"))
    diff = _bucket_row(res, "planted", "D = near - quiet")
    n = res.populations["planted"].n_events
    return [
        "### M1. Adverse selection by minutes to release, planted world: tape proxy, oracle and "
        "price impact (synthetic data)",
        "",
        f"Reading: cents per contract over {n} events, informed share at the mean time to release "
        f"of the bucket; the tape proxy minus the oracle is {_signed(near.bias_cents)} near "
        f"(0, 30] and {_signed(quiet.bias_cents)} in the quiet window, so D is "
        f"{_signed(diff.tape_as_cents)} on the tape, {_signed(diff.oracle_as_cents)} by the "
        f"oracle and {_signed(diff.impact_cents)} by the price impact.",
        "",
        *_bucket_lines(res, ["planted"]),
        "",
        "Does not show: how the proxy behaves on a real tape (stale prints, ticks, thin books); "
        "the oracle is the latent mid before the first trade at or after the horizon, and after "
        "the last trade when none is left.",
    ]


def _table_nulls(res: RecoveryResults) -> list[str]:
    parts = []
    for world in ("spec_null", "flat_null"):
        diff = _bucket_row(res, world, "D = near - quiet")
        parts.append(f"{_signed(diff.tape_as_cents)} on the tape, "
                     f"{_signed(diff.oracle_as_cents)} by the oracle and "
                     f"{_signed(diff.impact_cents)} by the impact in {world}")
    return [
        "### M2. Adverse selection by minutes to release, the two worlds without an informed-share "
        "profile (synthetic data)",
        "",
        "Reading: the same quantities with pi1 = 0, where spec_null keeps the trade-rate surge and "
        f"flat_null has neither; D in cents is {'; '.join(parts)}.",
        "",
        *_bucket_lines(res, ["spec_null", "flat_null"]),
        "",
        "Does not show: an informed-share profile of any kind; the informed share is constant at "
        "the planted pi0 in both worlds.",
    ]


def _effect_sentence(res: RecoveryResults) -> str:
    spec, flat = _effect(res, "spec_null", "raw_0_30"), _effect(res, "flat_null", "raw_0_30")
    text = (f"raw D is {spec['sign']} with the surge alone (spec_null, {_signed(spec['D_cents'])} "
            f"cents) and {flat['sign']} with neither (flat_null, {_signed(flat['D_cents'])} cents)")
    if spec["sign"] != "zero" or flat["sign"] != "zero":
        text += (", so D is a per-contract cost measured on the tape, not an estimate of the "
                 "informed share")
    clean = _effect(res, "flat_null", "scaled_5_30")
    if clean["sign"] == "zero":
        text += (f"; scaled D near (5, 30] is the cleaner read: in flat_null it is "
                 f"{_signed(clean['D_cents'])} cents, with an interval that contains zero")
    else:
        text += (f"; scaled D near (5, 30] is still {clean['sign']} in flat_null "
                 f"({_signed(clean['D_cents'])} cents)")
    return text


def _table_effects(res: RecoveryResults) -> list[str]:
    n = res.design["pop_events"]
    saturated = res.populations["planted"].saturation
    lines = [
        "### M3. Population release effect by world and estimand (synthetic data)",
        "",
        f"Reading: D = near minus quiet in cents per contract over {n} events per world, with its "
        f"95% cluster-bootstrap interval; {_effect_sentence(res)}.",
        "",
        "| world | estimand | D | 95% interval | near mean | quiet mean |",
        "|---|---|---|---|---|---|",
    ]
    for r in res.effects.itertuples():
        lines.append(
            f"| {r.world} | {LABELS[r.estimand]} | {_signed(r.D_cents)} | "
            f"{_signed(r.lo_cents)} to {_signed(r.hi_cents)} | {_signed(r.mean_near_cents)} | "
            f"{_signed(r.mean_quiet_cents)} |")
    lines += [
        "",
        "Does not show: whether D means the same on real tapes; the planted default saturates "
        f"prices near the end ({_share(saturated)}% of its events end outside [0.01, 0.99]), which "
        "shrinks p(1 - p) toward the release.",
    ]
    return lines


def _table_surge(res: RecoveryResults) -> list[str]:
    s = res.surge
    lines = [
        "### M4. Share of the planted raw D that the trade-rate surge gives, and price saturation "
        "(synthetic data)",
        "",
        f"Reading: D(spec_null) / D(planted) for estimand (a), from {s['n_events']} events per "
        f"world, is {_share(s['share'])}% (95% interval {_share(s['ci'][0])}% to "
        f"{_share(s['ci'][1])}%, paired cluster bootstrap, B = {s['B']}); saturation is the share "
        "of events whose last latent mid is outside [0.01, 0.99].",
        "",
        "| quantity | planted | spec_null | flat_null |",
        "|---|---|---|---|",
        "| events | " + " | ".join(str(res.populations[w].n_events) for w in WORLDS) + " |",
        "| saturated events, % | "
        + " | ".join(_share(res.populations[w].saturation, 1) for w in WORLDS) + " |",
        "| raw D (a), cents | " + " | ".join(
            _signed(_effect(res, w, "raw_0_30")["D_cents"]) for w in WORLDS) + " |",
    ]
    lines += [
        "",
        "Does not show: how much of D on real data is surge, price level or informed trading; "
        "the split holds for this world's parameters.",
    ]
    return lines


def _cut_note(res: RecoveryResults) -> str:
    """A closing sentence when fewer replications were run than the design asked for."""
    d = res.design
    if d["reps"] >= d["reps_design"]:
        return ""
    return (f" The design asked for {d['reps_design']} replications per world; {d['reps']} were "
            "run to stay within the 6-minute budget.")


def _grid_lines(res: RecoveryResults, cell) -> list[str]:
    sizes = res.design["sizes"]
    lines = ["| world | estimand | " + " | ".join(f"n = {n}" for n in sizes) + " |",
             "|---|---|" + "---|" * len(sizes)]
    for world in WORLDS:
        for e in ESTIMANDS:
            cells = [cell(world, e.key, n) for n in sizes]
            lines.append(f"| {world} | {e.label} | " + " | ".join(cells) + " |")
    return lines


def _lookup(frame: pd.DataFrame, world: str, key: str, n: int) -> Mapping[str, Any]:
    rows = frame[(frame["world"] == world) & (frame["estimand"] == key)
                 & (frame["n_events"] == n)]
    return rows.iloc[0]


def _table_coverage(res: RecoveryResults) -> list[str]:
    cov, d = res.coverage, res.design
    ref = 200 if 200 in d["sizes"] else max(d["sizes"])
    at = cov[cov["n_events"] == ref]
    big = max(d["sizes"])
    lost = 2.0 * NormalDist().cdf(1.96 / np.sqrt(1.0 + big / d["pop_events"])) - 1.0
    reps = d["reps"]
    se = 100.0 * np.sqrt(0.25 / reps)

    def cell(world: str, key: str, n: int) -> str:
        r = _lookup(cov, world, key, n)
        return f"{_share(r.cover_cluster)} / {_share(r.cover_naive)}"

    return [
        "### M5. Coverage of the population D by the cluster-bootstrap and the naive interval "
        "(synthetic data)",
        "",
        f"Reading: cluster bootstrap / naive trade-level interval, percent of {reps} replications "
        f"whose 95% interval contains the population value of its own world and estimand (B = "
        f"{d['B']}); at n = {ref} the cluster interval covers "
        f"{_share(at['cover_cluster'].min())}% to {_share(at['cover_cluster'].max())}% and the "
        f"naive one {_share(at['cover_naive'].min())}% to {_share(at['cover_naive'].max())}% "
        f"(Monte Carlo standard error at most {se:.1f} points).{_cut_note(res)}",
        "",
        *_grid_lines(res, cell),
        "",
        f"Does not show: coverage of the true D; the population value comes from "
        f"{d['pop_events']} events and its own error lowers the nominal 95% to about "
        f"{_share(lost, 1)}% at n = {big}.",
    ]


def _table_width(res: RecoveryResults) -> list[str]:
    cov = res.coverage

    def cell(world: str, key: str, n: int) -> str:
        r = _lookup(cov, world, key, n)
        return f"{r.width_cluster_cents:.2f} / {r.width_naive_cents:.2f}"

    return [
        "### M6. Mean width of the 95% interval, cluster bootstrap and naive (synthetic data)",
        "",
        "Reading: cents per contract, cluster bootstrap / naive trade-level interval, same "
        "replications as M5; the width that matches the spread of D over the replications is "
        "3.92 times its standard deviation (sd_D_cents in coverage.csv).",
        "",
        *_grid_lines(res, cell),
        "",
        "Does not show: the width on real data, where the trades of a release are more dependent "
        "and have heavier tails than here.",
    ]


def _table_power(res: RecoveryResults) -> list[str]:
    power, d = res.power, res.design
    truth = res.effects.set_index(["world", "estimand"])
    sizes = d["sizes"]
    lines = ["| world | estimand | population D | " + " | ".join(f"n = {n}" for n in sizes) + " |",
             "|---|---|---|" + "---|" * len(sizes)]
    for world in WORLDS:
        for e in ESTIMANDS:
            t = truth.loc[(world, e.key)]
            cells = []
            for n in sizes:
                r = _lookup(power, world, e.key, n)
                cells.append(f"{_share(r.reject)} ({_share(r.reject_se)})")
            lines.append(f"| {world} | {e.label} | {_signed(t['D_cents'])} ({t['sign']}) | "
                         + " | ".join(cells) + " |")
    return [
        "### M7. Share of replications with one-sided p below 0.05, by world and number of events "
        "(synthetic data)",
        "",
        f"Reading: percent of {d['reps']} replications with the one-sided cluster-bootstrap p "
        f"below 0.05 (B = {d['B']}; Monte Carlo standard error in brackets); the test asks "
        "whether D > 0, so the rate is power where the population D is positive, the size where "
        f"it is zero and below the size where it is negative.{_cut_note(res)}",
        "",
        *lines,
        "",
        "Does not show: power for an effect of another size, series mix or number of markets per "
        "release; the worlds have one market per event.",
    ]


def _table_sample(res: RecoveryResults) -> list[str]:
    power, d = res.power, res.design
    largest = max(d["sizes"])
    lines = ["| estimand | smallest n with power above 80% | power at that n | power at the next "
             "smaller n |", "|---|---|---|---|"]
    for e in ESTIMANDS:
        n = res.sample[e.key]
        if n is None:
            top = _lookup(power, "planted", e.key, largest)
            lines.append(f"| {e.label} | not reached (largest n tried {largest}) | "
                         f"{_share(top.reject)}% at n = {largest} | |")
            continue
        here = _lookup(power, "planted", e.key, n)
        smaller = sorted(s for s in d["sizes"] if s < n)
        below = ("no smaller size tried" if not smaller else
                 f"{_share(_lookup(power, 'planted', e.key, smaller[-1]).reject)}% at "
                 f"n = {smaller[-1]}")
        lines.append(f"| {e.label} | {n} | {_share(here.reject)}% | {below} |")
    return [
        "### M8. Number of events for 80% power in the planted world (synthetic data)",
        "",
        "Reading: smallest size tried with a rejection rate above 80% in the planted world, each "
        f"rate from {d['reps']} replications (Monte Carlo standard error up to "
        f"{100.0 * np.sqrt(0.25 / d['reps']):.1f} points); it applies to an effect like the "
        f"planted one.{_cut_note(res)}",
        "",
        *lines,
        "",
        "Does not show: the number of releases a real series has or needs; the worlds have one "
        "market per event and planted effects, so a real effect can be smaller.",
    ]


def tables_markdown(res: RecoveryResults) -> str:
    """The text of ``tables.md``: a provenance comment, then tables M1 to M8, each a heading that
    says synthetic data, a ``Reading:`` paragraph, a Markdown table and a ``Does not show:``
    paragraph. Every number in the paragraphs is computed from ``res``."""
    blocks = [_table_planted(res), _table_nulls(res), _table_effects(res), _table_surge(res),
              _table_coverage(res), _table_width(res), _table_power(res), _table_sample(res)]
    header = ["<!-- Generated by kalshi_mm.experiments.recovery; do not edit by hand. -->", ""]
    body: list[str] = []
    for block in blocks:
        body += [*block, ""]
    return "\n".join([*header, *body]).rstrip("\n") + "\n"


# ------------------------------------------------------------------------------------- the run
def _parse_sizes(text: str) -> tuple[int, ...]:
    return tuple(int(s) for s in text.split(","))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/markouts"))
    ap.add_argument("--reps", type=int, default=REPS, help="replications per world")
    ap.add_argument("--B", type=int, default=BOOTSTRAP_B, help="bootstrap resamples per interval")
    ap.add_argument("--pop-events", type=int, default=POP_EVENTS,
                    help="events per world in the population simulation")
    ap.add_argument("--sizes", type=_parse_sizes, default=SIZES, help="comma-separated n_events")
    args = ap.parse_args(argv)
    manifest = Manifest("kalshi_mm.experiments.recovery", seeds=seeds_text(args.pop_events),
                        args=argv)

    res = run_study(args.reps, args.B, args.pop_events, args.sizes)
    args.out.mkdir(parents=True, exist_ok=True)
    res.buckets.to_csv(args.out / "population.csv", index=False, float_format="%.8g")
    res.coverage.to_csv(args.out / "coverage.csv", index=False, float_format="%.8g")
    res.power.to_csv(args.out / "power.csv", index=False, float_format="%.8g")
    summary = {
        "experiment": "recovery", "data": "synthetic", "design": res.design,
        "worlds": {w: {"overrides": o} for w, o in WORLDS.items()},
        "estimands": {e.key: {"label": e.label, "column": e.column, "near": list(e.near)}
                      for e in ESTIMANDS},
        "population": {
            w: {"n_events": p.n_events, "n_trades": p.n_trades, "saturation_share": p.saturation,
                "effects": json.loads(res.effects[res.effects["world"] == w]
                                      .drop(columns="world").to_json(orient="records"))}
            for w, p in res.populations.items()},
        "surge_share": res.surge, "sample_size_planted": res.sample,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "tables.md").write_text(tables_markdown(res))
    manifest.extra.append(
        f"design: worlds {list(WORLDS)}, sizes {list(args.sizes)}, {args.reps} replications per "
        f"world (design {REPS_DESIGN}), cluster bootstrap B = {args.B}, population "
        f"{args.pop_events} events per world")
    manifest.write(args.out, "recovery",
                   ["population.csv", "coverage.csv", "power.csv", "summary.json", "tables.md"],
                   notes="synthetic data only; nothing here is real Kalshi data")


if __name__ == "__main__":
    main()
