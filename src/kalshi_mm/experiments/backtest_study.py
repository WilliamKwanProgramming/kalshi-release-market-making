"""Experiment: the four quoting policies on the synthetic worlds (docs/model.md sections 6 to 9).

Usage::

    python -m kalshi_mm.experiments.backtest_study --out results/backtest

The design was fixed before any result was looked at:

* two worlds, the null (``pi1 = 0``) and the planted (``pi1 = 0.20``), every other ``WorldParams``
  field at its default, 200 events each, dataset seeds 1 to 10;
* the four policies of model.md section 6 with the default ``QuoteParams``, run walk-forward; the
  first 20 events are burn-in, the other 180 are evaluated;
* primary quantity: the settled net P&L per event of ``release_aware`` minus ``pooled_premium``,
  paired by event, averaged over the evaluated events of a dataset, with the 95% percentile
  interval of a cluster bootstrap over events (10,000 replications); across the ten datasets the
  mean, the range and the number of intervals that exclude zero;
* secondary: the five other pairs of policies, fills, contracts, net cents per contract, fees,
  marked P&L and the share of refresh intervals with a live quote;
* sweeps on dataset seed 1: the planted share ``pi1`` in {0, 0.05, 0.10, 0.20, 0.30} crossed with
  the maker fee coefficient in {0.0175, 0}; the touch fill rule at ``pi1 = 0.20``; the half-width
  of the primary interval against the number of evaluated events (the first n of the 180);
* a check of the estimated premiums against their target and against the latent price impact.

Table B8 is not part of that design. It was added after the first results, because B1 (12 hours
of trading before each release) and the rehearsal of the protocol analysis (30 hours) disagreed
in the sign of the primary difference. It repeats the primary comparison in the planted world
with 30 hours of trading before each release (``WorldParams.window_hours``), on the same dataset
seeds, and is exploratory.

Writes ``per_event.csv``, ``policies.csv``, ``pairs.csv``, ``events_needed.csv``,
``premium_check.csv``, ``windows.csv``, ``per_event_windows.csv``, ``summary.json``,
``tables.md`` and ``backtest_study.manifest.txt``. The latent columns of the world are read only
by the premium check; the backtest gets a tape without them.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from kalshi_mm.backtest import walk_forward
from kalshi_mm.experiments.common import Manifest
from kalshi_mm.markout import BUCKET_LABELS, trade_markouts
from kalshi_mm.quoting import POLICIES, Premiums, QuoteParams, estimate_premiums
from kalshi_mm.stats import cluster_weights, percentile_interval, ratio_of_sums
from kalshi_mm.world import World, WorldParams, generate_world

__all__ = ["PAIRS", "PRIMARY", "StudyConfig", "main", "run_study"]

MODULE = "kalshi_mm.experiments.backtest_study"
SEED = 20261113  # the seed of docs/protocol.md; the bootstrap generators are keyed by it
PRIMARY = ("release_aware", "pooled_premium")
PAIRS = (
    PRIMARY,
    ("release_aware", "always_on"),
    ("release_aware", "pull_only"),
    ("pooled_premium", "always_on"),
    ("pooled_premium", "pull_only"),
    ("pull_only", "always_on"),
)
NULL_WORLD, PLANTED_WORLD = "null_world", "planted_world"
LEVEL = 0.95
Z_95 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
MIN_SCALE = 0.05  # trades with m (1 - m) below this are left out of the premium (model.md 6)
WINDOW_PREMIUMS = ("premium_pooled", "premium_near", "premium_far")  # the premium columns of B8

# the first key of a bootstrap generator, default_rng([SEED, KIND, ...]), names the analysis
KIND_MAIN, KIND_SWEEP, KIND_TOUCH, KIND_EVENTS, KIND_WINDOW = 1, 2, 3, 4, 5


@dataclass(frozen=True)
class StudyConfig:
    """The design of the study. The defaults are the pre-specified design."""

    world: WorldParams = field(default_factory=WorldParams)
    quote: QuoteParams = field(default_factory=QuoteParams)
    seeds: tuple[int, ...] = tuple(range(1, 11))
    planted_pi1: float = 0.20
    boot_reps: int = 10_000
    sweep_pi1: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.30)
    sweep_fees: tuple[float, ...] = (0.0175, 0.0)
    sweep_seed: int = 1
    event_counts: tuple[int, ...] = (20, 40, 60, 80, 100, 120, 140, 160, 180)
    # hours of trading before each release in the exploratory runs of B8, besides the
    # ``world.window_hours`` of the main grid
    extra_windows: tuple[float, ...] = (30.0,)

    def worlds(self) -> tuple[tuple[str, float], ...]:
        """The two worlds of the main grid as ``(name, pi1)``. The names are not ``null`` and
        ``planted`` because pandas reads the string ``null`` back as a missing value."""
        return ((NULL_WORLD, 0.0), (PLANTED_WORLD, self.planted_pi1))


# -------------------------------------------------------------------------------- one dataset
def boot_rng(*key: int) -> np.random.Generator:
    """The bootstrap generator of one analysis: ``default_rng([SEED, *key])``."""
    return np.random.default_rng([SEED, *key])


def run_dataset(
    world_params: WorldParams, seed: int, quote: QuoteParams, policies: Sequence[str] = POLICIES
) -> tuple[World, pd.DataFrame]:
    """Generate a world and run the walk-forward of ``policies`` (all four by default) on it.

    The tape that the backtest sees has no ``latent_`` column.
    """
    world = generate_world(world_params, seed)
    tape = world.tape.drop(columns=[c for c in world.tape if str(c).startswith("latent_")])
    return world, walk_forward(tape, world.markets, world.events, policies, quote)


def event_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per event and policy: the sums over the event's markets (the share of live
    intervals is their mean)."""
    return (
        rows.groupby(["event", "policy"], sort=False)
        .agg(
            pnl=("pnl", "sum"),
            gross=("gross", "sum"),
            fees=("fees", "sum"),
            fills=("fills", "sum"),
            contracts=("contracts", "sum"),
            inventory_end=("inventory_end", "sum"),
            marked_pnl=("marked_pnl", "sum"),
            quotes_live_share=("quotes_live_share", "mean"),
            evaluated=("evaluated", "first"),
            release_ts=("release_ts", "first"),
        )
        .reset_index()
    )


def evaluated_pnl(events: pd.DataFrame) -> pd.DataFrame:
    """Net P&L per evaluated event (rows, in release order) and policy (columns)."""
    done = events[events["evaluated"]]
    order = done.drop_duplicates("event").sort_values(["release_ts", "event"])["event"]
    wide = done.pivot(index="event", columns="policy", values="pnl")
    return wide.loc[order.to_numpy()]


def primary_differences(events: pd.DataFrame) -> NDArray[np.float64]:
    """Per-event difference of the primary pair over the evaluated events, in release order."""
    wide = evaluated_pnl(events)
    return (wide[PRIMARY[0]] - wide[PRIMARY[1]]).to_numpy(dtype=float)


def mean_difference(d: NDArray[np.float64], weights: NDArray[np.int64]) -> dict[str, float]:
    """Mean of the per-event differences ``d`` and the percentile interval of the cluster
    bootstrap whose resampling counts are ``weights`` (shape ``(B, len(d))``)."""
    boot = ratio_of_sums(d, np.ones(d.size), weights)
    lo, hi = percentile_interval(boot, LEVEL)
    return {"mean": float(d.mean()), "lo": lo, "hi": hi}


def policy_summary(events: pd.DataFrame, policy: str) -> dict[str, float]:
    """Means per evaluated event of one policy; ``cents_per_contract`` is the net P&L of all
    evaluated events over all contracts filled, in cents."""
    done = events[(events["policy"] == policy) & events["evaluated"]]
    contracts = float(done["contracts"].sum())
    cents = 100.0 * float(done["pnl"].sum()) / contracts if contracts > 0.0 else math.nan
    return {
        "n_events": int(len(done)),
        "pnl": float(done["pnl"].mean()),
        "gross": float(done["gross"].mean()),
        "fees": float(done["fees"].mean()),
        "fills": float(done["fills"].mean()),
        "contracts": float(done["contracts"].mean()),
        "cents_per_contract": cents,
        "marked_pnl": float(done["marked_pnl"].mean()),
        "live_share": float(done["quotes_live_share"].mean()),
    }


def analyse(
    events: pd.DataFrame, boot_reps: int, key: tuple[int, ...], meta: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Policy summaries and pair differences of one dataset; ``meta`` labels every record.

    The six pairs share the resampling counts of ``default_rng([SEED, *key])``.
    """
    wide = evaluated_pnl(events)
    weights = cluster_weights(len(wide), boot_reps, boot_rng(*key))
    policies = [{**meta, "policy": p, **policy_summary(events, p)} for p in POLICIES]
    pairs = []
    for a, b in PAIRS:
        d = (wide[a] - wide[b]).to_numpy(dtype=float)
        pairs.append({**meta, "a": a, "b": b, "n_events": int(d.size),
                      **mean_difference(d, weights)})
    return policies, pairs


# ---------------------------------------------------------------------------------- the parts
def main_grid(
    config: StudyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The two worlds times the dataset seeds: per-event rows, policy summaries, pair differences
    and the premium check of every dataset."""
    per_event, policies, pairs, checks = [], [], [], []
    for w, (name, pi1) in enumerate(config.worlds()):
        params = replace(config.world, pi1=pi1)
        for seed in config.seeds:
            world, rows = run_dataset(params, seed, config.quote)
            checks.append(premium_check(world, config.quote, seed, name))
            events = event_table(rows)
            meta = {"sweep": "main", "world": name, "pi1": pi1, "seed": seed,
                    "fee": config.quote.maker_fee_coeff, "rule": config.quote.fill_rule}
            p, q = analyse(events, config.boot_reps, (KIND_MAIN, w, seed), meta)
            policies += p
            pairs += q
            per_event.append(events.assign(world=name, seed=seed))
    return (pd.concat(per_event, ignore_index=True), pd.DataFrame(policies),
            pd.DataFrame(pairs), pd.concat(checks, ignore_index=True))


def sweeps(config: StudyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The planted share crossed with the maker fee, and the touch rule at the planted share, all
    on one dataset seed. Returns policy summaries and pair differences."""
    policies, pairs = [], []
    for i, pi1 in enumerate(config.sweep_pi1):
        params = replace(config.world, pi1=pi1)
        for j, fee in enumerate(config.sweep_fees):
            quote = replace(config.quote, maker_fee_coeff=fee, fill_rule="through")
            _, rows = run_dataset(params, config.sweep_seed, quote)
            meta = {"sweep": "pi1_fee", "world": f"pi1={pi1:g}", "pi1": pi1,
                    "seed": config.sweep_seed, "fee": fee, "rule": "through"}
            p, q = analyse(event_table(rows), config.boot_reps, (KIND_SWEEP, i, j), meta)
            policies += p
            pairs += q
    quote = replace(config.quote, fill_rule="touch")
    _, rows = run_dataset(replace(config.world, pi1=config.planted_pi1), config.sweep_seed, quote)
    meta = {"sweep": "touch", "world": PLANTED_WORLD, "pi1": config.planted_pi1,
            "seed": config.sweep_seed, "fee": quote.maker_fee_coeff, "rule": "touch"}
    p, q = analyse(event_table(rows), config.boot_reps, (KIND_TOUCH,), meta)
    return pd.DataFrame(policies + p), pd.DataFrame(pairs + q)


def events_needed(config: StudyConfig, per_event: pd.DataFrame) -> pd.DataFrame:
    """Half-width of the primary interval when only the first n evaluated events are used, in
    every planted dataset of the main grid."""
    rows = []
    planted = per_event[per_event["world"] == PLANTED_WORLD]
    for seed, part in planted.groupby("seed", sort=True):
        d = primary_differences(part)
        for n in config.event_counts:
            if n > d.size:
                continue
            weights = cluster_weights(n, config.boot_reps, boot_rng(KIND_EVENTS, int(seed), n))
            stats = mean_difference(d[:n], weights)
            rows.append({"seed": int(seed), "n": n, **stats,
                         "halfwidth": (stats["hi"] - stats["lo"]) / 2.0})
    if not rows:
        raise ValueError("no event count is at most the number of evaluated events")
    return pd.DataFrame(rows)


def _premiums_known_at(
    markouts: pd.DataFrame, events: pd.DataFrame, rank: int, quote: QuoteParams
) -> Premiums:
    """The premiums the walk-forward uses for the event of release rank ``rank``: the markouts
    of the events released at or before the start of its window."""
    release = events["release_ts"].to_numpy(dtype=float)
    start = release[rank] - quote.window_hours * 3600.0
    known = events["event"].iloc[: int(np.searchsorted(release, start, side="right"))]
    return estimate_premiums(
        markouts[markouts["event"].isin(known)], min_events=quote.min_history_events
    )


def premium_check(world: World, quote: QuoteParams, seed: int, name: str) -> pd.DataFrame:
    """Per markout bucket and pooled, in USD per contract per unit of ``4 m (1 - m)``:

    * ``estimate_end``: the premium :func:`estimate_premiums` gives for the last event;
    * ``estimate_burn_in``: the same for the first event after the burn-in;
    * ``target``: the mean of ``adverse_selection / (4 m (1 - m))`` over all trades of the
      dataset with ``m (1 - m) >= 0.05``, not floored (the quantity the estimator estimates);
    * ``latent``: the mean over the same trades of the latent price impact of the trade (its
      side times the change of the latent mid it causes) divided the same way.
    """
    tape = world.tape.drop(columns=[c for c in world.tape if str(c).startswith("latent_")])
    markouts = trade_markouts(
        tape, world.events, world.markets, horizon_s=quote.horizon_s, stale_s=quote.stale_s
    )
    latent = world.tape.loc[markouts.index]
    impact = markouts["side"].to_numpy(dtype=float) * (
        latent["latent_mid_post"].to_numpy() - latent["latent_mid_pre"].to_numpy()
    )
    m = markouts["mid_before"].to_numpy(dtype=float)
    scale = 4.0 * m * (1.0 - m)
    use = scale >= 4.0 * MIN_SCALE
    frame = pd.DataFrame(
        {
            "bucket": markouts["bucket"].to_numpy()[use],
            "target": markouts["adverse_selection"].to_numpy()[use] / scale[use],
            "latent": impact[use] / scale[use],
        }
    )
    by_bucket = frame.groupby("bucket")[["target", "latent"]].mean()
    by_bucket = by_bucket.reindex(range(len(BUCKET_LABELS)))
    pooled = frame[["target", "latent"]].mean()

    events = world.events.sort_values(["release_ts", "event"]).reset_index(drop=True)
    end = _premiums_known_at(markouts, events, len(events) - 1, quote)
    first = _premiums_known_at(markouts, events, min(quote.min_history_events, len(events) - 1),
                               quote)
    return pd.DataFrame(
        {
            "world": name,
            "seed": seed,
            "bucket": [*BUCKET_LABELS, "pooled"],
            "estimate_end": [*end.by_bucket, end.pooled],
            "estimate_burn_in": [*first.by_bucket, first.pooled],
            "target": [*by_bucket["target"], pooled["target"]],
            "latent": [*by_bucket["latent"], pooled["latent"]],
        }
    )


def window_premiums(check: pd.DataFrame) -> dict[str, float]:
    """The premiums at the end of a dataset that B8 shows, from its premium check (one dataset):
    the pooled one and those of the first and of the last bucket, in USD per contract per unit
    of ``4 m (1 - m)``."""
    end = check.set_index("bucket")["estimate_end"]
    values = (end["pooled"], end[BUCKET_LABELS[0]], end[BUCKET_LABELS[-1]])
    return {name: float(v) for name, v in zip(WINDOW_PREMIUMS, values, strict=True)}


def default_window_records(
    config: StudyConfig, pairs: pd.DataFrame, check: pd.DataFrame
) -> pd.DataFrame:
    """The row of B8 for the hours of trading of the main grid: the primary difference of the
    planted datasets of the main grid (the planted row of B1) and their premiums at the end."""
    primary = pairs[
        (pairs["sweep"] == "main") & (pairs["world"] == PLANTED_WORLD)
        & (pairs["a"] == PRIMARY[0]) & (pairs["b"] == PRIMARY[1])
    ]
    rows = []
    for r in primary.itertuples():
        planted = check[(check["world"] == PLANTED_WORLD) & (check["seed"] == r.seed)]
        rows.append({
            "window_hours": config.world.window_hours, "seed": int(r.seed),
            "n_events": int(r.n_events), "mean": r.mean, "lo": r.lo, "hi": r.hi,
            **window_premiums(planted),
        })
    return pd.DataFrame(rows)


def window_runs(config: StudyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The exploratory runs of B8: the planted world with each of ``config.extra_windows`` hours
    of trading before the release, on every dataset seed; only the two policies of the primary
    pair are run (their rows do not depend on the other policies). Returns one record per window
    and dataset (the primary difference with its interval and the premiums at the end) and the
    net P&L per event of the two policies. The bootstrap counts come from
    ``default_rng([SEED, KIND_WINDOW, index of the window in extra_windows, seed])``."""
    if not config.extra_windows:
        raise ValueError("extra_windows is empty: B8 needs a window to compare with")
    if config.world.window_hours in config.extra_windows:
        raise ValueError("extra_windows must differ from the hours of trading of the world")
    records, per_event = [], []
    for i, hours in enumerate(config.extra_windows):
        params = replace(config.world, pi1=config.planted_pi1, window_hours=hours)
        for seed in config.seeds:
            world, rows = run_dataset(params, seed, config.quote, PRIMARY)
            events = event_table(rows)
            d = primary_differences(events)
            weights = cluster_weights(d.size, config.boot_reps, boot_rng(KIND_WINDOW, i, seed))
            premiums = window_premiums(premium_check(world, config.quote, seed, PLANTED_WORLD))
            records.append({
                "window_hours": hours, "seed": seed, "n_events": int(d.size),
                **mean_difference(d, weights), **premiums,
            })
            kept = events[["event", "policy", "pnl", "evaluated"]]
            per_event.append(kept.assign(window_hours=hours, seed=seed))
    return pd.DataFrame(records), pd.concat(per_event, ignore_index=True)


# ---------------------------------------------------------------------------------- summaries
def _clean(obj: Any) -> Any:
    """JSON-ready copy: numpy scalars as Python numbers, NaN and infinity as null."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def across_datasets(pairs: pd.DataFrame) -> dict[str, float]:
    """Mean of the dataset means with its Monte Carlo standard error (sd over the square root of
    the number of datasets), their range, the number of 95% intervals that lie below and above
    zero and the mean half-width, for the rows of ``pairs`` (one row per dataset)."""
    mean = pairs["mean"].to_numpy(dtype=float)
    n = mean.size
    return {
        "n_datasets": int(n),
        "mean": float(mean.mean()),
        "mc_se": float(mean.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan,
        "min": float(mean.min()),
        "max": float(mean.max()),
        "below_zero": int((pairs["hi"] < 0.0).sum()),
        "above_zero": int((pairs["lo"] > 0.0).sum()),
        "half_width": float(((pairs["hi"] - pairs["lo"]) / 2.0).mean()),
    }


def window_summary(windows: pd.DataFrame) -> list[dict[str, float]]:
    """One entry per hours of trading, in increasing order: :func:`across_datasets` of the primary
    difference and the mean over the datasets of the three premiums at the end."""
    out = []
    for hours, part in windows.groupby("window_hours", sort=True):
        means = {k: float(part[k].mean()) for k in WINDOW_PREMIUMS}
        out.append({"window_hours": float(hours), **across_datasets(part), **means})
    return out


def sample_size(per_event: pd.DataFrame) -> dict[str, float]:
    """Numbers of evaluated events from the normal approximation, with the mean ``mu`` and the
    standard deviation ``sd`` (``ddof = 1``) of the per-event differences of the primary pair
    pooled over the planted datasets: ``events_half_width = (1.96 sd / mu)^2``, where the expected
    half-width of the 95% interval equals ``|mu|``, and ``events_power_80 = ((1.96 + 0.84) sd /
    mu)^2``, where a two-sided 5% test of zero has 80% power."""
    planted = per_event[per_event["world"] == PLANTED_WORLD]
    pooled = np.concatenate([primary_differences(part) for _, part in planted.groupby("seed")])
    mu, sd = float(pooled.mean()), float(pooled.std(ddof=1))
    if mu == 0.0:
        return {"mean": mu, "sd": sd, "events_half_width": math.inf, "events_power_80": math.inf}
    return {
        "mean": mu,
        "sd": sd,
        "events_half_width": (Z_95 * sd / mu) ** 2,
        "events_power_80": ((Z_95 + Z_POWER_80) * sd / mu) ** 2,
    }


def _policy_means(policies: pd.DataFrame, world: str) -> dict[str, dict[str, float]]:
    """Mean over the datasets of a world of every policy summary column, by policy."""
    columns = ("pnl", "gross", "fees", "fills", "contracts", "cents_per_contract", "marked_pnl",
               "live_share")
    out = {}
    main = policies[(policies["sweep"] == "main") & (policies["world"] == world)]
    for policy in POLICIES:
        part = main[main["policy"] == policy]
        out[policy] = {c: float(part[c].mean()) for c in columns}
        out[policy]["pnl_mc_se"] = (
            float(part["pnl"].std(ddof=1) / math.sqrt(len(part))) if len(part) > 1 else math.nan
        )
    return out


def _sweep_cells(
    policies: pd.DataFrame, pairs: pd.DataFrame, sweep: str
) -> list[dict[str, Any]]:
    """One entry per cell of a sweep: its settings, the primary difference and the net P&L per
    event of every policy."""
    prim = pairs[(pairs["sweep"] == sweep) & (pairs["a"] == PRIMARY[0])
                 & (pairs["b"] == PRIMARY[1])]
    cells = []
    for _, row in prim.iterrows():
        cell = policies[(policies["sweep"] == sweep) & (policies["pi1"] == row["pi1"])
                        & (policies["fee"] == row["fee"])]
        cells.append({
            "pi1": row["pi1"], "fee": row["fee"], "rule": row["rule"],
            "primary": {k: row[k] for k in ("mean", "lo", "hi")},
            "pnl": dict(zip(cell["policy"], cell["pnl"], strict=True)),
            "fills": dict(zip(cell["policy"], cell["fills"], strict=True)),
        })
    return cells


def build_summary(
    config: StudyConfig,
    policies: pd.DataFrame,
    pairs: pd.DataFrame,
    needed: pd.DataFrame,
    size: dict[str, float],
    check: pd.DataFrame,
    windows: pd.DataFrame,
) -> dict[str, Any]:
    """Everything the tables quote, by name (``policies`` and ``pairs`` hold the main grid and
    the sweeps; ``windows`` the records of B8)."""
    main = pairs[pairs["sweep"] == "main"]
    summary: dict[str, Any] = {
        "design": {
            "n_events": config.world.n_events,
            "seeds": list(config.seeds),
            "planted_pi1": config.planted_pi1,
            "bootstrap_replications": config.boot_reps,
            "bootstrap_level": LEVEL,
            "burn_in_events": config.quote.min_history_events,
            "extra_windows": list(config.extra_windows),
            "quote_params": asdict(config.quote),
            "primary": list(PRIMARY),
        },
        "pairs": {},
        "policies": {},
        "sweeps": {s: _sweep_cells(policies, pairs, s) for s in ("pi1_fee", "touch")},
        "events_needed": [],
        "sample_size": size,
        "premium_check": {},
        "windows": window_summary(windows),
    }
    for world, _ in config.worlds():
        for a, b in PAIRS:
            part = main[(main["world"] == world) & (main["a"] == a) & (main["b"] == b)]
            summary["pairs"][f"{world}:{a}-{b}"] = across_datasets(part)
        for policy, values in _policy_means(policies, world).items():
            summary["policies"][f"{world}:{policy}"] = values
    first_seed = config.seeds[0]
    for n, part in needed.groupby("n", sort=True):
        summary["events_needed"].append({
            "n": int(n),
            "half_width_mean": float(part["halfwidth"].mean()),
            "half_width_min": float(part["halfwidth"].min()),
            "half_width_max": float(part["halfwidth"].max()),
            "half_width_first_seed": float(
                part.loc[part["seed"] == first_seed, "halfwidth"].iloc[0]
            ),
            "excludes_zero": int(((part["hi"] < 0.0) | (part["lo"] > 0.0)).sum()),
            "n_datasets": int(len(part)),
        })
    columns = ["estimate_end", "estimate_burn_in", "target", "latent"]
    for (world, bucket), part in check.groupby(["world", "bucket"], sort=False):
        summary["premium_check"][f"{world}:{bucket}"] = {c: float(part[c].mean()) for c in columns}
    return _clean(summary)


# ------------------------------------------------------------------------------------- tables
def _num(x: float | None, digits: int = 2, signed: bool = False) -> str:
    if x is None or not math.isfinite(x):
        return "n/a"
    return f"{x:+.{digits}f}" if signed else f"{x:.{digits}f}"


def _markdown(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _block(table_id: str, title: str, reading: str, table: str, not_shown: str) -> str:
    return (f"### {table_id}. {title} (synthetic data)\n\nReading: {reading}\n\n{table}\n\n"
            f"Does not show: {not_shown}\n")


def _excluded(below: int, above: int, n: int) -> str:
    """Words for the number of datasets whose 95% interval lies below or above zero."""
    return (f"the 95% interval lies below zero in {below} of {n} datasets and above zero in "
            f"{above}")


def _label(world: str) -> str:
    """``null`` for ``null_world``: how a world is named in the tables."""
    return world.removesuffix("_world")


def _more_or_less(x: float) -> str:
    return "more" if x > 0.0 else "less"


def _pair(a: str, b: str) -> str:
    return f"`{a}` minus `{b}`"


def _earned(mean: float) -> str:
    """Words for the primary difference ``mean``: what ``release_aware`` earned per event."""
    if mean == 0.0:
        return f"`{PRIMARY[0]}` and `{PRIMARY[1]}` earned the same per event"
    return (f"`{PRIMARY[0]}` earned {_num(abs(mean))} USD per event {_more_or_less(mean)} than "
            f"`{PRIMARY[1]}`")


def _interval(cell: dict[str, Any]) -> str:
    p = cell["primary"]
    lo, hi = _num(p["lo"], signed=True), _num(p["hi"], signed=True)
    return f"{_num(p['mean'], signed=True)} [{lo}, {hi}]"


def table_primary(summary: dict[str, Any], config: StudyConfig) -> str:
    """B1: the primary comparison in both worlds."""
    body, sentences = [], []
    for world, _ in config.worlds():
        s = summary["pairs"][f"{world}:{PRIMARY[0]}-{PRIMARY[1]}"]
        n = s["n_datasets"]
        spread = f"{_num(s['min'], signed=True)} to {_num(s['max'], signed=True)}"
        body.append([
            _label(world), f"{_num(s['mean'], signed=True)} ({_num(s['mc_se'])})", spread,
            str(s["below_zero"]), str(s["above_zero"]), str(n - s["below_zero"] - s["above_zero"]),
            _num(s["half_width"]),
        ])
        sentences.append(
            f"In the {_label(world)} world {_earned(s['mean'])} (mean of {n} datasets, range "
            f"{spread}; {_excluded(s['below_zero'], s['above_zero'], n)})."
        )
    table = _markdown(
        ["World", "Mean difference, USD per event (MC SE over datasets)", "Range over datasets",
         "95% interval below zero", "95% interval above zero", "95% interval contains zero",
         "Mean interval half-width, USD"],
        body,
    )
    return _block(
        "B1", f"{_pair(*PRIMARY)}, net P&L per event",
        " ".join(sentences) + f" Intervals are percentile intervals of a cluster bootstrap over "
        f"events with {config.boot_reps:,} replications.", table,
        "whether either policy earns or loses money on real Kalshi markets: the tape, the fill "
        "rule and the quote parameters are assumptions.",
    )


def table_policies(summary: dict[str, Any], config: StudyConfig) -> str:
    """B2: the four policies in both worlds."""
    body, cells = [], []
    for world, _ in config.worlds():
        for policy in POLICIES:
            s = summary["policies"][f"{world}:{policy}"]
            cells.append((s["pnl"], world, policy))
            body.append([
                _label(world), f"`{policy}`", f"{_num(s['pnl'])} ({_num(s['pnl_mc_se'])})",
                _num(s["fees"]), _num(s["fills"], 1), _num(s["contracts"], 1),
                _num(s["cents_per_contract"]), _num(s["marked_pnl"]), _num(s["live_share"], 3),
            ])
    losing = sum(1 for value, _, _ in cells if value < 0.0)
    best, worst = max(cells), min(cells)
    reading = (
        f"Net P&L per event is negative in {losing} of {len(cells)} world and policy cells; the "
        f"highest is `{best[2]}` in the {_label(best[1])} world ({_num(best[0], signed=True)} "
        f"USD) and the lowest `{worst[2]}` in the {_label(worst[1])} world "
        f"({_num(worst[0], signed=True)} USD)."
    )
    table = _markdown(
        ["World", "Policy", "Net P&L per event, USD (MC SE over datasets)", "Fees per event, USD",
         "Fills per event", "Contracts per event", "Net cents per contract",
         "Marked P&L per event, USD", "Share of intervals with a quote"],
        body,
    )
    return _block(
        "B2", "The four policies, means over the datasets of each world", reading, table,
        "which part of a loss is adverse selection: the table has no fill-level decomposition.",
    )


def table_pairs(summary: dict[str, Any], config: StudyConfig) -> str:
    """B3: the five other pairs."""
    body, found = [], []
    for world, _ in config.worlds():
        for a, b in PAIRS[1:]:
            s = summary["pairs"][f"{world}:{a}-{b}"]
            found.append((abs(s["mean"]), world, a, b, s))
            body.append([
                _label(world), _pair(a, b), f"{_num(s['mean'], signed=True)} ({_num(s['mc_se'])})",
                f"{_num(s['min'], signed=True)} to {_num(s['max'], signed=True)}",
                str(s["below_zero"]), str(s["above_zero"]), _num(s["half_width"]),
            ])
    _, world, a, b, s = max(found, key=lambda f: f[0])
    reading = (
        f"The largest of the {len(found)} differences is {_pair(a, b)} in the {_label(world)} "
        f"world, {_num(s['mean'], signed=True)} USD per event; "
        f"{_excluded(s['below_zero'], s['above_zero'], s['n_datasets'])}."
    )
    table = _markdown(
        ["World", "Comparison", "Mean difference, USD per event (MC SE over datasets)",
         "Range over datasets", "95% interval below zero", "95% interval above zero",
         "Mean interval half-width, USD"],
        body,
    )
    return _block(
        "B3", "Secondary comparisons of net P&L per event", reading, table,
        "a family-wise error rate: the cells are not adjusted for multiple comparisons.",
    )


def _sweep_row(cell: dict[str, Any]) -> list[str]:
    return [_num(cell["pnl"][p]) for p in POLICIES] + [_interval(cell)]


def _sweep_header() -> list[str]:
    return [f"`{p}`, USD per event" for p in POLICIES] + [
        f"{_pair(*PRIMARY)}, USD per event [95% interval]"
    ]


def table_sweep(summary: dict[str, Any], config: StudyConfig) -> str:
    """B4: the planted share crossed with the maker fee."""
    cells = summary["sweeps"]["pi1_fee"]
    body = [[_num(c["pi1"]), _num(c["fee"], 4), *_sweep_row(c)] for c in cells]
    excluded = [c for c in cells if c["primary"]["hi"] < 0.0 or c["primary"]["lo"] > 0.0]
    best = max(cells, key=lambda c: max(c["pnl"].values()))
    top = max(best["pnl"], key=lambda p: best["pnl"][p])
    reading = (
        f"The 95% interval of the primary difference excludes zero in {len(excluded)} of "
        f"{len(cells)} cells; the highest net P&L per event of any policy is "
        f"{_num(best['pnl'][top], signed=True)} USD (`{top}`, planted share {_num(best['pi1'])}, "
        f"maker fee coefficient {_num(best['fee'], 4)})."
    )
    table = _markdown(["Planted share pi1", "Maker fee coefficient", *_sweep_header()], body)
    return _block(
        "B4", f"Net P&L per event by planted share and maker fee, dataset seed "
        f"{config.sweep_seed}", reading + f" Intervals are percentile intervals of a cluster "
        f"bootstrap over events with {config.boot_reps:,} replications.", table,
        "variation between datasets: each cell is one dataset, so differences of a few USD per "
        "event between cells can be sampling noise.",
    )


def table_touch(summary: dict[str, Any], config: StudyConfig) -> str:
    """B5: the touch rule against the through rule at the planted share."""
    touch = summary["sweeps"]["touch"][0]
    through = next(c for c in summary["sweeps"]["pi1_fee"]
                   if c["pi1"] == config.planted_pi1 and c["fee"] == config.quote.maker_fee_coeff)
    change = {p: touch["pnl"][p] - through["pnl"][p] for p in POLICIES}
    top = max(change, key=lambda p: abs(change[p]))
    verb = "raises" if change[top] > 0.0 else "lowers"
    reading = (
        f"Filling prints at the quote {verb} net P&L per event by {_num(abs(change[top]))} USD "
        f"for `{top}`, the largest change; the primary difference is "
        f"{_num(touch['primary']['mean'], signed=True)} USD under the touch rule against "
        f"{_num(through['primary']['mean'], signed=True)} USD under the through rule."
    )
    table = _markdown(
        ["Fill rule", *_sweep_header()],
        [["through (primary)", *_sweep_row(through)], ["touch", *_sweep_row(touch)]],
    )
    return _block(
        "B5", f"Through rule against touch rule, planted world, dataset seed {config.sweep_seed}",
        reading, table,
        "queue position: the touch rule fills every print at the quote, so it is an upper bound "
        "on what resting at the quote can earn.",
    )


def table_events_needed(summary: dict[str, Any], config: StudyConfig) -> str:
    """B6: half-width of the primary interval against the number of evaluated events."""
    rows = summary["events_needed"]
    body = [[
        str(r["n"]), _num(r["half_width_first_seed"]),
        f"{_num(r['half_width_mean'])} ({_num(r['half_width_min'])} to "
        f"{_num(r['half_width_max'])})",
        f"{r['excludes_zero']} of {r['n_datasets']}",
    ] for r in rows]
    size = summary["sample_size"]
    last = rows[-1]
    if size["events_half_width"] is not None:
        extrapolation = (
            f" By the normal approximation, about {size['events_half_width']:,.0f} evaluated "
            f"events would make the expected half-width equal the pooled mean difference "
            f"({_num(size['mean'], signed=True)} USD per event) and about "
            f"{size['events_power_80']:,.0f} would give a two-sided 5% test 80% power."
        )
    else:
        extrapolation = " The pooled mean difference is zero, so no number of events is given."
    reading = (
        f"With the first {last['n']} evaluated events the mean half-width is "
        f"{_num(last['half_width_mean'])} USD per event and the interval excludes zero in "
        f"{last['excludes_zero']} of {last['n_datasets']} planted datasets." + extrapolation
    )
    table = _markdown(
        ["Evaluated events used", f"Half-width, dataset seed {config.seeds[0]}, USD",
         "Half-width over planted datasets, mean (range), USD", "Interval excludes zero"],
        body,
    )
    return _block(
        "B6", "Half-width of the primary 95% interval against the number of evaluated events",
        reading, table,
        "how many real releases are needed: the extrapolation takes the effect and the variance "
        "of this synthetic world.",
    )


def table_premium_check(check: pd.DataFrame) -> str:
    """B7: estimated premiums against their target and the latent price impact."""
    columns = ["estimate_end", "target", "latent", "estimate_burn_in"]
    mean = check.groupby(["world", "bucket"], sort=False)[columns].mean()
    body = [[_label(world), bucket, *[_num(100.0 * mean.loc[(world, bucket), c]) for c in columns]]
            for world, bucket in mean.index]
    near = mean.loc[(PLANTED_WORLD, BUCKET_LABELS[0])]
    far = mean.loc[(PLANTED_WORLD, BUCKET_LABELS[-1])]
    gap = float((100.0 * (mean["estimate_end"] - mean["target"]).abs()).max())
    reading = (
        f"In the planted world the premium estimated at the end is "
        f"{_num(100 * near['estimate_end'])} cents in the bucket {BUCKET_LABELS[0]} minutes and "
        f"{_num(100 * far['estimate_end'])} cents in the bucket {BUCKET_LABELS[-1]} (latent price "
        f"impact {_num(100 * near['latent'])} and {_num(100 * far['latent'])}); after the burn-in "
        f"it is {_num(100 * near['estimate_burn_in'])} in the first bucket; the largest gap "
        f"between the estimate at the end and its target is {_num(gap)} cents."
    )
    table = _markdown(
        ["World", "Minutes to release", "Estimate at the end, cents",
         "Target: mean over the dataset, cents", "Latent price impact, cents",
         "Estimate after burn-in, cents"],
        body,
    )
    return _block(
        "B7", "Estimated premiums against their target and the latent price impact", reading,
        table,
        "whether the premium prices the adverse selection of a fill: it is measured per trade, "
        "and the fills of a maker are not a random sample of the trades.",
    )


def table_windows(summary: dict[str, Any], config: StudyConfig) -> str:
    """B8, exploratory: the primary difference in the planted world by hours of trading before
    each release, with the premiums estimated at the end of each dataset."""
    rows = summary["windows"]
    body, sentences, premiums = [], [], []
    for s in rows:
        n, hours = s["n_datasets"], f"{s['window_hours']:g}"
        body.append([
            hours, str(n), f"{_num(s['mean'], signed=True)} ({_num(s['mc_se'])})",
            f"{_num(s['min'], signed=True)} to {_num(s['max'], signed=True)}",
            str(s["below_zero"]), str(s["above_zero"]), str(n - s["below_zero"] - s["above_zero"]),
            _num(s["half_width"]), _num(100.0 * s["premium_pooled"]),
            _num(100.0 * s["premium_near"]), _num(100.0 * s["premium_far"]),
        ])
        sentences.append(
            f"With {hours} hours of trading before each release {_earned(s['mean'])} (mean of {n} "
            f"datasets; {_excluded(s['below_zero'], s['above_zero'], n)})."
        )
        premiums.append(f"{_num(100.0 * s['premium_pooled'])} cents with {hours} hours")
    joined = ", ".join(premiums[:-1]) + (" and " if len(premiums) > 1 else "") + premiums[-1]
    reading = (
        "Exploratory: added after the first results because B1 and the rehearsal of the protocol "
        "analysis disagreed in sign; it was not part of the design fixed beforehand. "
        + " ".join(sentences) + f" The pooled premium estimated at the end is {joined}. Intervals "
        f"are percentile intervals of a cluster bootstrap over events with {config.boot_reps:,} "
        f"replications."
    )
    near, far = BUCKET_LABELS[0], BUCKET_LABELS[-1]
    table = _markdown(
        ["Hours of trading before each release", "Datasets",
         "Mean difference, USD per event (MC SE over datasets)", "Range over datasets",
         "95% interval below zero", "95% interval above zero", "95% interval contains zero",
         "Mean interval half-width, USD", "Pooled premium at the end, cents",
         f"Premium of the bucket {near} at the end, cents",
         f"Premium of the bucket {far} at the end, cents"],
        body,
    )
    return _block(
        "B8", f"Exploratory: {_pair(*PRIMARY)} by hours of trading before each release, planted "
        "world", reading, table,
        "the mechanism: a longer window adds trades before the quoting window to the premium "
        "sample and to the history of the mid proxy together, and only the window differs "
        "between the rows.",
    )


def write_tables(summary: dict[str, Any], config: StudyConfig, check: pd.DataFrame) -> str:
    """The text of ``tables.md``."""
    return "\n".join([
        table_primary(summary, config), table_policies(summary, config),
        table_pairs(summary, config), table_sweep(summary, config), table_touch(summary, config),
        table_events_needed(summary, config), table_premium_check(check),
        table_windows(summary, config),
    ])


# ------------------------------------------------------------------------------------ the run
def run_study(config: StudyConfig, out: Path, manifest: Manifest) -> dict[str, Any]:
    """Run every part of the design, write the files to ``out`` and return the summary."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per_event, policies, pairs, check = main_grid(config)
    sweep_policies, sweep_pairs = sweeps(config)
    needed = events_needed(config, per_event)
    extra, extra_events = window_runs(config)
    windows = pd.concat([default_window_records(config, pairs, check), extra], ignore_index=True)
    policies = pd.concat([policies, sweep_policies], ignore_index=True)
    pairs = pd.concat([pairs, sweep_pairs], ignore_index=True)
    summary = build_summary(
        config, policies, pairs, needed, sample_size(per_event), check, windows
    )

    csv = {"index": False, "float_format": "%.10g"}
    columns = ["world", "seed", "event", "policy", "pnl", "gross", "fees", "fills", "contracts",
               "inventory_end", "marked_pnl", "quotes_live_share", "evaluated"]
    per_event[columns].to_csv(out / "per_event.csv", **csv)
    policies.to_csv(out / "policies.csv", **csv)
    pairs.to_csv(out / "pairs.csv", **csv)
    needed.to_csv(out / "events_needed.csv", **csv)
    check.to_csv(out / "premium_check.csv", **csv)
    windows.to_csv(out / "windows.csv", **csv)
    extra_events[["window_hours", "seed", "event", "policy", "pnl", "evaluated"]].to_csv(
        out / "per_event_windows.csv", **csv
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (out / "tables.md").write_text(write_tables(summary, config, check))
    manifest.extra += [
        f"worlds: null (pi1 = 0) and planted (pi1 = {config.planted_pi1}), "
        f"{config.world.n_events} events each; dataset seeds {config.seeds[0]} to "
        f"{config.seeds[-1]} (event e of a dataset uses default_rng([seed, e]))",
        f"bootstrap: {config.boot_reps} replications over events, 95% percentile intervals, "
        f"generators default_rng([{SEED}, analysis, ...])",
        f"quote parameters: {config.quote}",
        f"B8, added after the first results (exploratory): the planted world with "
        f"{', '.join(f'{h:g}' for h in config.extra_windows)} hours of trading before each "
        f"release instead of {config.world.window_hours:g}, the same dataset seeds, only the "
        f"policies {PRIMARY[0]} and {PRIMARY[1]}; bootstrap generators "
        f"default_rng([{SEED}, {KIND_WINDOW}, window index, seed])",
    ]
    manifest.write(
        out, "backtest_study",
        ["per_event.csv", "policies.csv", "pairs.csv", "events_needed.csv", "premium_check.csv",
         "windows.csv", "per_event_windows.csv", "summary.json", "tables.md"],
        notes=f"{config.quote.min_history_events} burn-in events per dataset; sweeps on dataset "
              f"seed {config.sweep_seed}",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("results/backtest"))
    args = ap.parse_args(argv)
    config = StudyConfig()
    manifest = Manifest(
        MODULE, f"{SEED} (bootstrap); datasets {config.seeds[0]} to {config.seeds[-1]}", argv
    )
    run_study(config, args.out, manifest)


if __name__ == "__main__":
    main()
