"""Experiment: the five quoting policies on the synthetic worlds (docs/model.md sections 6 to 9).

Usage::

    python -m kalshi_mm.experiments.backtest_study --out results/backtest

The first version of the design (four policies, ``release_aware`` minus ``pooled_premium`` as the
primary quantity) was fixed before any result was looked at. This is the second version. B8 is not
part of the design fixed first (added after the first results). B9 was added after an independent
review showed that the first primary pair followed the level of the pooled premium and not the time
conditioning: in the planted world the pooled premium, a mean over every earlier trade, is larger
than the bucket premiums averaged over the clock time of the quoting window (B7 shows both), and
every policy loses money (B2), so the policy that quotes wider loses less. The primary pair was then
changed to the matched comparison before the new results were generated. The definition of the
matched premium (the policy ``flat_matched``) and the grid of B9 were fixed before any result of
them was generated. The design:

* two worlds, the null (``pi1 = 0``) and the planted (``pi1 = 0.20``), every other ``WorldParams``
  field at its default, 200 events each, dataset seeds 1 to 10;
* the five policies of model.md section 6 with the default ``QuoteParams``, run walk-forward; the
  first 20 events are burn-in, the other 180 are evaluated;
* primary quantity: the settled net P&L per event of ``release_aware`` minus ``flat_matched``
  (a flat premium of the same average size as the bucket premiums), paired by event, averaged over
  the evaluated events of a dataset, with the 95% percentile interval of a cluster bootstrap over
  events (10,000 replications); across the ten datasets the mean, the range and the number of
  intervals that exclude zero;
* secondary: the nine other pairs of policies (among them ``release_aware`` minus
  ``pooled_premium``), fills, contracts, net cents per contract, fees, marked P&L and the share of
  refresh intervals with a live quote;
* sweeps on dataset seed 1: the planted share ``pi1`` in {0, 0.05, 0.10, 0.20, 0.30} crossed with
  the maker fee coefficient in {0.0175, 0}; the touch fill rule at ``pi1 = 0.20``; the half-width
  of the primary interval against the number of evaluated events (the first n of the 180);
* a check of the estimated premiums (pooled, matched and by bucket) against their target and
  against the latent price impact.

Table B8 (exploratory) repeats the comparison in the planted world with 30 hours of trading before
each release (``WorldParams.window_hours``) on the same dataset seeds, because B1 (12 hours) and the
rehearsal of the protocol analysis (30 hours) disagreed in sign for the first primary pair. It
shows the primary and the secondary difference (``release_aware`` minus ``pooled_premium``) and the
premiums at the end of each dataset. Table B9 (exploratory) is the level sweep: in the planted world
of the main grid, ``release_aware`` minus ``flat_matched`` when the flat premium is multiplied by a
scale F in {0.5, 0.75, 1, 1.25, 1.5, 1.75, 2} (``QuoteParams.flat_scale``); F = 1 is the primary
comparison. Table B10 (exploratory) was added after a second independent review showed that
``flat_matched`` is matched to ``release_aware`` on the average of the premium over the clock and
not at the fills, where ``release_aware`` quotes more premium: B2 reports the premium at the fills
of every policy, and B10 repeats the primary comparison in both worlds with the flat premium scaled
by ``F_eq`` so that the two quote the same premium at their fills.

Writes ``per_event.csv``, ``policies.csv``, ``pairs.csv``, ``events_needed.csv``,
``premium_check.csv``, ``windows.csv``, ``per_event_windows.csv``, ``levels.csv``,
``fill_matched.csv``, ``summary.json``, ``tables.md`` and ``backtest_study.manifest.txt``. The
latent columns of the world are read only by the premium check; the backtest gets a tape without
them.
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

from kalshi_mm.backtest import mean_fill_premium, walk_forward
from kalshi_mm.experiments.common import Manifest
from kalshi_mm.markout import BUCKET_LABELS, trade_markouts
from kalshi_mm.quoting import (
    POLICIES,
    Premiums,
    QuoteParams,
    decision_minutes,
    estimate_premiums,
    matched_premium,
)
from kalshi_mm.stats import cluster_weights, percentile_interval, ratio_of_sums
from kalshi_mm.world import World, WorldParams, generate_world

__all__ = ["PAIRS", "PRIMARY", "SECONDARY", "StudyConfig", "main", "run_study"]

MODULE = "kalshi_mm.experiments.backtest_study"
SEED = 20261113  # the seed of docs/protocol.md; the bootstrap generators are keyed by it
# "a minus b", the more elaborate policy first. The primary pair is the primary pair of the
# protocol analysis (kalshi_mm.protocol.PRIMARY_PAIR); the secondary pair is the primary pair of
# the first version of the design, kept because its sign follows the level of the pooled premium
# (B8, B9).
PRIMARY = ("release_aware", "flat_matched")
SECONDARY = ("release_aware", "pooled_premium")
PAIRS = (
    PRIMARY,
    SECONDARY,
    ("flat_matched", "pooled_premium"),
    ("release_aware", "always_on"),
    ("release_aware", "pull_only"),
    ("pooled_premium", "always_on"),
    ("pooled_premium", "pull_only"),
    ("flat_matched", "always_on"),
    ("flat_matched", "pull_only"),
    ("pull_only", "always_on"),
)
WINDOW_POLICIES = ("release_aware", "flat_matched", "pooled_premium")  # the runs of B8
NULL_WORLD, PLANTED_WORLD = "null_world", "planted_world"
LEVEL = 0.95
Z_95 = 1.959963984540054
Z_POWER_80 = 0.8416212335729143
MIN_SCALE = 0.05  # trades with m (1 - m) below this are left out of the premium (model.md 6)
# the premium columns of B8, at the end of each dataset: the pooled premium, the matched premium
# and those of the first and of the last bucket
WINDOW_PREMIUMS = ("premium_pooled", "premium_matched", "premium_near", "premium_far")
SECONDARY_COLUMNS = ("secondary_mean", "secondary_lo", "secondary_hi")  # the secondary in B8

# the first key of a bootstrap generator, default_rng([SEED, KIND, ...]), names the analysis
KIND_MAIN, KIND_SWEEP, KIND_TOUCH, KIND_EVENTS, KIND_WINDOW, KIND_LEVEL, KIND_FILLS = (
    1, 2, 3, 4, 5, 6, 7
)
# the extra element of the generator of the secondary difference of B8
SECONDARY_KEY = 1


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
    # the scales F of the flat premium of ``flat_matched`` in the exploratory level sweep of B9
    sweep_levels: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)

    def worlds(self) -> tuple[tuple[str, float], ...]:
        """The two worlds of the main grid as ``(name, pi1)``. The names are not ``null`` and
        ``planted`` because pandas reads the string ``null`` back as a missing value."""
        return ((NULL_WORLD, 0.0), (PLANTED_WORLD, self.planted_pi1))


# -------------------------------------------------------------------------------- one dataset
def boot_rng(*key: int) -> np.random.Generator:
    """The bootstrap generator of one analysis: ``default_rng([SEED, *key])``."""
    return np.random.default_rng([SEED, *key])


def walk_world(world: World, policies: Sequence[str], quote: QuoteParams) -> pd.DataFrame:
    """The walk-forward of ``policies`` on a world. The tape that the backtest sees has no
    ``latent_`` column."""
    tape = world.tape.drop(columns=[c for c in world.tape if str(c).startswith("latent_")])
    return walk_forward(tape, world.markets, world.events, policies, quote)


def run_dataset(
    world_params: WorldParams, seed: int, quote: QuoteParams, policies: Sequence[str] = POLICIES
) -> tuple[World, pd.DataFrame]:
    """Generate a world and run the walk-forward of ``policies`` (all five by default) on it.

    The tape that the backtest sees has no ``latent_`` column.
    """
    world = generate_world(world_params, seed)
    return world, walk_world(world, policies, quote)


def event_table(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per event and policy: the sums over the event's markets (the share of live
    intervals and the mean quoted spread are their means; a market without a quote on both sides
    has no spread and is left out of that mean). ``fill_premium`` is the mean over the contracts
    filled in the event of the premium per unit of ``4 m (1 - m)`` of the filled quotes (NaN when
    nothing was filled)."""
    weighted = rows.assign(premium_filled=rows["fill_premium"].fillna(0.0) * rows["contracts"])
    table = (
        weighted.groupby(["event", "policy"], sort=False)
        .agg(
            pnl=("pnl", "sum"),
            gross=("gross", "sum"),
            fees=("fees", "sum"),
            fills=("fills", "sum"),
            contracts=("contracts", "sum"),
            inventory_end=("inventory_end", "sum"),
            marked_pnl=("marked_pnl", "sum"),
            quotes_live_share=("quotes_live_share", "mean"),
            mean_spread=("mean_spread", "mean"),
            premium_filled=("premium_filled", "sum"),
            evaluated=("evaluated", "first"),
            release_ts=("release_ts", "first"),
        )
        .reset_index()
    )
    contracts = table["contracts"].to_numpy(dtype=float)
    filled = contracts > 0.0
    fill_premium = np.full(len(table), math.nan)
    fill_premium[filled] = table["premium_filled"].to_numpy(dtype=float)[filled] / contracts[filled]
    table.insert(table.columns.get_loc("mean_spread") + 1, "fill_premium", fill_premium)
    return table.drop(columns="premium_filled")


def evaluated_pnl(events: pd.DataFrame) -> pd.DataFrame:
    """Net P&L per evaluated event (rows, in release order) and policy (columns)."""
    done = events[events["evaluated"]]
    order = done.drop_duplicates("event").sort_values(["release_ts", "event"])["event"]
    wide = done.pivot(index="event", columns="policy", values="pnl")
    return wide.loc[order.to_numpy()]


def pair_differences(events: pd.DataFrame, pair: tuple[str, str]) -> NDArray[np.float64]:
    """Per-event difference ``a - b`` of the net P&L of ``pair = (a, b)`` over the evaluated
    events, in release order."""
    wide = evaluated_pnl(events)
    return (wide[pair[0]] - wide[pair[1]]).to_numpy(dtype=float)


def primary_differences(events: pd.DataFrame) -> NDArray[np.float64]:
    """Per-event difference of the primary pair over the evaluated events, in release order."""
    return pair_differences(events, PRIMARY)


def mean_difference(d: NDArray[np.float64], weights: NDArray[np.int64]) -> dict[str, float]:
    """Mean of the per-event differences ``d`` and the percentile interval of the cluster
    bootstrap whose resampling counts are ``weights`` (shape ``(B, len(d))``)."""
    boot = ratio_of_sums(d, np.ones(d.size), weights)
    lo, hi = percentile_interval(boot, LEVEL)
    return {"mean": float(d.mean()), "lo": lo, "hi": hi}


def policy_summary(events: pd.DataFrame, policy: str) -> dict[str, float]:
    """Means per evaluated event of one policy; ``cents_per_contract`` is the net P&L of all
    evaluated events over all contracts filled, in cents, ``spread`` the mean quoted spread
    (ask minus bid, USD per contract) of the events that had a quote on both sides and
    ``fill_premium`` the premium per unit of ``4 m (1 - m)`` at the fills (USD per contract) over
    all the contracts the policy filled in the evaluated events
    (:func:`kalshi_mm.backtest.mean_fill_premium`)."""
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
        "spread": float(done["mean_spread"].mean()),
        "fill_premium": mean_fill_premium(done),
    }


def analyse(
    events: pd.DataFrame, boot_reps: int, key: tuple[int, ...], meta: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Policy summaries and pair differences of one dataset; ``meta`` labels every record.

    The ten pairs share the resampling counts of ``default_rng([SEED, *key])``.
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
    of the events released at or before the start of its window, with the matched premium over the
    decisions of the quoting window."""
    release = events["release_ts"].to_numpy(dtype=float)
    start = release[rank] - quote.window_hours * 3600.0
    known = events["event"].iloc[: int(np.searchsorted(release, start, side="right"))]
    return estimate_premiums(
        markouts[markouts["event"].isin(known)], min_events=quote.min_history_events,
        minutes=decision_minutes(quote),
    )


def premium_check(world: World, quote: QuoteParams, seed: int, name: str) -> pd.DataFrame:
    """Per markout bucket, pooled and matched, in USD per contract per unit of ``4 m (1 - m)``:

    * ``estimate_end``: the premium :func:`estimate_premiums` gives for the last event;
    * ``estimate_burn_in``: the same for the first event after the burn-in;
    * ``target``: the mean of ``adverse_selection / (4 m (1 - m))`` over all trades of the
      dataset with ``m (1 - m) >= 0.05``, not floored (the quantity the estimator estimates);
    * ``latent``: the mean over the same trades of the latent price impact of the trade (its
      side times the change of the latent mid it causes) divided the same way.

    The ``matched`` row is the flat premium of ``flat_matched``: ``estimate_end`` and
    ``estimate_burn_in`` are ``Premiums.matched`` of the two estimates, and ``target`` and
    ``latent`` are :func:`kalshi_mm.quoting.matched_premium` of the seven bucket targets (latent
    means) over the decisions of the quoting window, so they are undefined (NaN) when a bucket of
    the window has no usable trade.
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
    minutes = decision_minutes(quote)
    matched = {c: matched_premium(by_bucket[c].to_numpy(), minutes) for c in ("target", "latent")}

    events = world.events.sort_values(["release_ts", "event"]).reset_index(drop=True)
    end = _premiums_known_at(markouts, events, len(events) - 1, quote)
    first = _premiums_known_at(markouts, events, min(quote.min_history_events, len(events) - 1),
                               quote)
    return pd.DataFrame(
        {
            "world": name,
            "seed": seed,
            "bucket": [*BUCKET_LABELS, "pooled", "matched"],
            "estimate_end": [*end.by_bucket, end.pooled, end.matched],
            "estimate_burn_in": [*first.by_bucket, first.pooled, first.matched],
            "target": [*by_bucket["target"], pooled["target"], matched["target"]],
            "latent": [*by_bucket["latent"], pooled["latent"], matched["latent"]],
        }
    )


def window_premiums(check: pd.DataFrame) -> dict[str, float]:
    """The premiums at the end of a dataset that B8 shows, from its premium check (one dataset):
    the pooled one, the matched one and those of the first and of the last bucket, in USD per
    contract per unit of ``4 m (1 - m)``."""
    end = check.set_index("bucket")["estimate_end"]
    values = (end["pooled"], end["matched"], end[BUCKET_LABELS[0]], end[BUCKET_LABELS[-1]])
    return {name: float(v) for name, v in zip(WINDOW_PREMIUMS, values, strict=True)}


def default_window_records(
    config: StudyConfig, pairs: pd.DataFrame, check: pd.DataFrame
) -> pd.DataFrame:
    """The row of B8 for the hours of trading of the main grid: the primary and the secondary
    difference of the planted datasets of the main grid (the planted rows of B1 and B3) and their
    premiums at the end."""
    planted = pairs[(pairs["sweep"] == "main") & (pairs["world"] == PLANTED_WORLD)]
    primary = planted[(planted["a"] == PRIMARY[0]) & (planted["b"] == PRIMARY[1])]
    secondary = planted[(planted["a"] == SECONDARY[0]) & (planted["b"] == SECONDARY[1])]
    second = secondary.set_index("seed")
    rows = []
    for r in primary.itertuples():
        end = check[(check["world"] == PLANTED_WORLD) & (check["seed"] == r.seed)]
        t = second.loc[r.seed]
        rows.append({
            "window_hours": config.world.window_hours, "seed": int(r.seed),
            "n_events": int(r.n_events), "mean": r.mean, "lo": r.lo, "hi": r.hi,
            "secondary_mean": t["mean"], "secondary_lo": t["lo"], "secondary_hi": t["hi"],
            **window_premiums(end),
        })
    return pd.DataFrame(rows)


def window_runs(config: StudyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The exploratory runs of B8: the planted world with each of ``config.extra_windows`` hours
    of trading before the release, on every dataset seed; only the three policies of
    :data:`WINDOW_POLICIES` are run (their rows do not depend on the other policies). Returns one
    record per window and dataset (the primary difference and the secondary difference, each with
    its interval, and the premiums at the end) and the net P&L per event of the three policies.
    The bootstrap counts of the primary difference come from
    ``default_rng([SEED, KIND_WINDOW, index of the window in extra_windows, seed])`` and those of
    the secondary difference from the same key followed by ``SECONDARY_KEY``."""
    if not config.extra_windows:
        raise ValueError("extra_windows is empty: B8 needs a window to compare with")
    if config.world.window_hours in config.extra_windows:
        raise ValueError("extra_windows must differ from the hours of trading of the world")
    records, per_event = [], []
    for i, hours in enumerate(config.extra_windows):
        params = replace(config.world, pi1=config.planted_pi1, window_hours=hours)
        for seed in config.seeds:
            world, rows = run_dataset(params, seed, config.quote, WINDOW_POLICIES)
            events = event_table(rows)
            d = primary_differences(events)
            weights = cluster_weights(d.size, config.boot_reps, boot_rng(KIND_WINDOW, i, seed))
            e = pair_differences(events, SECONDARY)
            counts = cluster_weights(
                e.size, config.boot_reps, boot_rng(KIND_WINDOW, i, seed, SECONDARY_KEY)
            )
            second = {f"secondary_{k}": v for k, v in mean_difference(e, counts).items()}
            premiums = window_premiums(premium_check(world, config.quote, seed, PLANTED_WORLD))
            records.append({
                "window_hours": hours, "seed": seed, "n_events": int(d.size),
                **mean_difference(d, weights), **second, **premiums,
            })
            kept = events[["event", "policy", "pnl", "evaluated"]]
            per_event.append(kept.assign(window_hours=hours, seed=seed))
    return pd.DataFrame(records), pd.concat(per_event, ignore_index=True)


def level_runs(config: StudyConfig) -> pd.DataFrame:
    """The exploratory level sweep of B9: the planted world of the main grid on every dataset
    seed, ``release_aware`` once and ``flat_matched`` once for each scale ``F`` of
    ``config.sweep_levels`` (``QuoteParams.flat_scale``); no other policy is run. Returns one
    record per ``F`` and dataset: the mean of the per-event difference ``release_aware`` minus
    ``flat_matched`` over the evaluated events with the percentile interval of a cluster
    bootstrap (counts of ``default_rng([SEED, KIND_LEVEL, index of F in sweep_levels, seed])``),
    ``premium_flat``, the flat premium that ``flat_matched`` quotes on average (F times the matched
    premium at the end of the dataset) and ``premium_pooled``, the pooled premium at the end, both
    in USD per contract per unit of ``4 m (1 - m)``. ``F = 1`` is the planted row of the main grid:
    the same events and the same mean, with other bootstrap counts."""
    levels = config.sweep_levels
    if not levels:
        raise ValueError("sweep_levels is empty: B9 needs a level to compare with")
    if len(set(levels)) != len(levels):
        raise ValueError("sweep_levels must not repeat a level")
    params = replace(config.world, pi1=config.planted_pi1)
    aware, flat = PRIMARY  # flat_matched is the only policy that reads QuoteParams.flat_scale
    records = []
    for seed in config.seeds:
        world = generate_world(params, seed)
        end = premium_check(world, config.quote, seed, PLANTED_WORLD).set_index("bucket")
        aware_events = event_table(walk_world(world, (aware,), config.quote))
        for i, scale in enumerate(levels):
            quote = replace(config.quote, flat_scale=scale)
            flat_events = event_table(walk_world(world, (flat,), quote))
            d = primary_differences(pd.concat([aware_events, flat_events], ignore_index=True))
            weights = cluster_weights(d.size, config.boot_reps, boot_rng(KIND_LEVEL, i, seed))
            records.append({
                "flat_scale": scale, "seed": seed, "n_events": int(d.size),
                **mean_difference(d, weights),
                "premium_flat": scale * float(end.loc["matched", "estimate_end"]),
                "premium_pooled": float(end.loc["pooled", "estimate_end"]),
            })
    return pd.DataFrame(records)


def fill_matched_runs(config: StudyConfig) -> pd.DataFrame:
    """The exploratory comparison at an equal premium at the fills (B10): both worlds of the main
    grid, every dataset seed.

    ``release_aware`` and ``flat_matched`` are run once. ``F_eq`` is the premium per unit of
    ``4 m (1 - m)`` at the fills of ``release_aware`` divided by that of ``flat_matched``
    (:func:`kalshi_mm.backtest.mean_fill_premium` over the evaluated events), and ``flat_matched``
    is run again with its flat premium multiplied by ``F_eq`` (``QuoteParams.flat_scale``, which is
    ``F_eq`` times the scale of ``config.quote``). ``F_eq`` is estimated from the same run, so the
    interval below treats it as given. Returns one record per world and dataset: ``flat_scale`` (the
    scale of the second run), the mean of the per-event difference ``release_aware`` minus
    ``flat_matched`` at that scale over the evaluated events with the percentile interval of a
    cluster bootstrap (counts of ``default_rng([SEED, KIND_FILLS, index of the world, seed])``), and
    the premiums at the fills, in USD per contract per unit of ``4 m (1 - m)``, of ``release_aware``
    (``fill_aware``), of ``flat_matched`` in the first run (``fill_flat_base``) and of
    ``flat_matched`` in the second (``fill_flat``). A dataset in which either policy has no premium
    at its fills has nothing to match and raises ``ValueError``."""
    aware, flat = PRIMARY
    records = []
    for w, (name, pi1) in enumerate(config.worlds()):
        params = replace(config.world, pi1=pi1)
        for seed in config.seeds:
            world = generate_world(params, seed)
            first = event_table(walk_world(world, (aware, flat), config.quote))
            done = first[first["evaluated"]]
            fill_aware = mean_fill_premium(done[done["policy"] == aware])
            fill_base = mean_fill_premium(done[done["policy"] == flat])
            if not (fill_aware > 0.0 and fill_base > 0.0):  # NaN fails both
                raise ValueError(f"no premium at the fills to match in the {name} dataset {seed}")
            scale = config.quote.flat_scale * fill_aware / fill_base
            quote = replace(config.quote, flat_scale=scale)
            second = event_table(walk_world(world, (flat,), quote))
            d = primary_differences(pd.concat([first[first["policy"] == aware], second],
                                              ignore_index=True))
            weights = cluster_weights(d.size, config.boot_reps, boot_rng(KIND_FILLS, w, seed))
            records.append({
                "world": name, "seed": seed, "n_events": int(d.size), "flat_scale": scale,
                **mean_difference(d, weights),
                "fill_aware": fill_aware, "fill_flat_base": fill_base,
                "fill_flat": mean_fill_premium(second[second["evaluated"]]),
            })
    return pd.DataFrame(records)


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


def window_summary(windows: pd.DataFrame) -> list[dict[str, Any]]:
    """One entry per hours of trading, in increasing order: :func:`across_datasets` of the primary
    difference under the keys of that function, :func:`across_datasets` of the secondary
    difference under ``secondary`` and the mean over the datasets of the premiums at the end
    (:data:`WINDOW_PREMIUMS`)."""
    out = []
    for hours, part in windows.groupby("window_hours", sort=True):
        means = {k: float(part[k].mean()) for k in WINDOW_PREMIUMS}
        second = part[list(SECONDARY_COLUMNS)].rename(
            columns=lambda c: c.removeprefix("secondary_"))
        out.append({"window_hours": float(hours), **across_datasets(part),
                    "secondary": across_datasets(second), **means})
    return out


def level_summary(levels: pd.DataFrame) -> list[dict[str, float]]:
    """One entry per scale F of the flat premium, in increasing order: :func:`across_datasets` of
    ``release_aware`` minus ``flat_matched`` and the mean over the datasets of the flat premium
    that was quoted and of the pooled premium at the end (USD per contract per unit of
    ``4 m (1 - m)``)."""
    out = []
    for scale, part in levels.groupby("flat_scale", sort=True):
        out.append({
            "flat_scale": float(scale), **across_datasets(part),
            "premium_flat": float(part["premium_flat"].mean()),
            "premium_pooled": float(part["premium_pooled"].mean()),
        })
    return out


def fill_matched_summary(fills: pd.DataFrame) -> dict[str, dict[str, float]]:
    """One entry per world of :func:`fill_matched_runs`, in the order of the rows:
    :func:`across_datasets` of ``release_aware`` minus ``flat_matched`` at ``F_eq``, the mean and
    the range over the datasets of the scale ``F_eq`` (``flat_scale``, ``flat_scale_min``,
    ``flat_scale_max``) and the mean over the datasets of the three premiums at the fills
    (``fill_aware``, ``fill_flat_base``, ``fill_flat``)."""
    out = {}
    for world, part in fills.groupby("world", sort=False):
        out[str(world)] = {
            **across_datasets(part),
            "flat_scale": float(part["flat_scale"].mean()),
            "flat_scale_min": float(part["flat_scale"].min()),
            "flat_scale_max": float(part["flat_scale"].max()),
            "fill_aware": float(part["fill_aware"].mean()),
            "fill_flat_base": float(part["fill_flat_base"].mean()),
            "fill_flat": float(part["fill_flat"].mean()),
        }
    return out


def sign_changes(
    scales: Sequence[float], means: Sequence[float]
) -> list[tuple[float, float, float]]:
    """Where the means change sign, by linear interpolation between neighbouring grid points.

    ``scales`` is increasing and ``means[j]`` belongs to ``scales[j]``. The result has one
    ``(F, F_a, F_b)`` for each pair of neighbours ``(F_a, d_a)``, ``(F_b, d_b)`` whose means lie
    strictly on opposite sides of zero, with ``F = F_a + (F_b - F_a) d_a / (d_a - d_b)``, the point
    where the line through the two means reaches zero; and one ``(F_b, F_a, F_b)`` for a mean
    ``d_b`` that is exactly zero at a grid point when the mean before it and the first non-zero
    mean after it have opposite signs (a mean that only touches zero does not count). The list is
    empty when the means keep one sign. It describes the grid; it is not an estimate with an
    interval."""
    found = []
    for a in range(len(means) - 1):
        d_a, d_b = means[a], means[a + 1]
        if d_a == 0.0:
            continue
        f_a, f_b = scales[a], scales[a + 1]
        if d_b != 0.0:
            if (d_a > 0.0) != (d_b > 0.0):
                found.append((f_a + (f_b - f_a) * d_a / (d_a - d_b), f_a, f_b))
            continue
        after = next((d for d in means[a + 2:] if d != 0.0), 0.0)
        if after != 0.0 and (after > 0.0) != (d_a > 0.0):
            found.append((f_b, f_a, f_b))
    return found


def sample_size(per_event: pd.DataFrame) -> dict[str, float]:
    """Numbers of evaluated events from the normal approximation, with the mean ``mu`` and the
    standard deviation ``sd`` (``ddof = 1``) of the per-event differences of the primary pair over
    all the evaluated events of the planted datasets: ``events_half_width = (1.96 sd / mu)^2``,
    where the expected half-width of the 95% interval equals ``|mu|``, and ``events_power_80 =
    ((1.96 + 0.84) sd / mu)^2``, where a two-sided 5% test of zero has 80% power."""
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
               "live_share", "spread", "fill_premium")
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
    levels: pd.DataFrame,
    fills: pd.DataFrame,
) -> dict[str, Any]:
    """Everything the tables quote, by name (``policies`` and ``pairs`` hold the main grid and
    the sweeps; ``windows`` the records of B8, ``levels`` those of B9 and ``fills`` those of
    B10)."""
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
            "sweep_levels": list(config.sweep_levels),
            "quote_params": asdict(config.quote),
            "primary": list(PRIMARY),
            "secondary": list(SECONDARY),
        },
        "pairs": {},
        "policies": {},
        "sweeps": {s: _sweep_cells(policies, pairs, s) for s in ("pi1_fee", "touch")},
        "events_needed": [],
        "sample_size": size,
        "premium_check": {},
        "windows": window_summary(windows),
        "levels": level_summary(levels),
        "fill_matched": fill_matched_summary(fills),
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


def _earned(mean: float, pair: tuple[str, str] = PRIMARY) -> str:
    """Words for the difference ``mean`` of ``pair`` (the primary pair unless given): what its
    first policy earned per event."""
    if mean == 0.0:
        return f"`{pair[0]}` and `{pair[1]}` earned the same per event"
    return (f"`{pair[0]}` earned {_num(abs(mean))} USD per event {_more_or_less(mean)} than "
            f"`{pair[1]}`")


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
        " ".join(sentences) + f" `{PRIMARY[1]}` quotes one flat premium equal to the average of "
        f"the bucket premiums over the clock of the quoting window (both premiums are per unit of "
        f"4m(1 - m); the quoted spreads in USD also depend on the price, see B2). That is the "
        f"same premium averaged over the clock and not the same premium where the quotes are "
        f"hit: B2 gives the premium at the fills of each policy and B10 repeats the comparison at "
        f"an equal premium at the fills. Intervals are percentile intervals of a cluster "
        f"bootstrap over events with {config.boot_reps:,} replications.", table,
        "whether the planted informed share is what makes the bucket premiums useful: the null "
        "world's buckets also differ, because of the trade-rate surge and the price level; "
        "whether the difference comes from when the premium is applied or from the larger "
        "premium that `release_aware` has at its fills (B10); and whether either policy earns or "
        "loses money on real Kalshi markets: the tape, the fill rule and the quote parameters "
        "are assumptions.",
    )


def _policy_sentences(summary: dict[str, Any], config: StudyConfig, key: str, what: str) -> str:
    """The value ``key`` of the policy summaries, in cents, of the three premium policies in each
    world, as one sentence per world that starts ``In the <world> world <what>, in cents, is``."""
    sentences = []
    for world, _ in config.worlds():
        values = {
            policy: 100.0 * summary["policies"][f"{world}:{policy}"][key]
            for policy in ("release_aware", "flat_matched", "pooled_premium")
        }
        parts = ", ".join(f"`{policy}` {_num(value)}" for policy, value in values.items())
        sentences.append(f"In the {_label(world)} world {what}, in cents, is {parts}.")
    return " ".join(sentences)


def table_policies(summary: dict[str, Any], config: StudyConfig) -> str:
    """B2: the five policies in both worlds."""
    body, cells = [], []
    for world, _ in config.worlds():
        for policy in POLICIES:
            s = summary["policies"][f"{world}:{policy}"]
            cells.append((s["pnl"], world, policy))
            body.append([
                _label(world), f"`{policy}`", f"{_num(s['pnl'])} ({_num(s['pnl_mc_se'])})",
                _num(s["fees"]), _num(s["fills"], 1), _num(s["contracts"], 1),
                _num(s["cents_per_contract"]), _num(s["marked_pnl"]), _num(s["live_share"], 3),
                _num(100.0 * s["spread"]), _num(100.0 * s["fill_premium"]),
            ])
    losing = sum(1 for value, _, _ in cells if value < 0.0)
    best, worst = max(cells), min(cells)
    reading = (
        f"Net P&L per event is negative in {losing} of {len(cells)} world and policy cells; the "
        f"highest is `{best[2]}` in the {_label(best[1])} world ({_num(best[0], signed=True)} "
        f"USD) and the lowest `{worst[2]}` in the {_label(worst[1])} world "
        f"({_num(worst[0], signed=True)} USD). "
        + _policy_sentences(summary, config, "spread", "the mean quoted spread") + " "
        + _policy_sentences(
            summary, config, "fill_premium",
            "the premium per unit of 4m(1 - m) at the fills, over the contracts filled")
    )
    table = _markdown(
        ["World", "Policy", "Net P&L per event, USD (MC SE over datasets)", "Fees per event, USD",
         "Fills per event", "Contracts per event", "Net cents per contract",
         "Marked P&L per event, USD", "Share of intervals with a quote",
         "Mean quoted spread, cents", "Premium at the fills, cents per unit of 4m(1 - m)"],
        body,
    )
    return _block(
        "B2", "The five policies, means over the datasets of each world", reading, table,
        "which part of a loss is adverse selection: the table has no decomposition of the "
        "losses; the spread is that of the quotes posted, ask minus bid at the refreshes with "
        "both sides quoted, not the width at which fills happened; and the premium at the fills "
        "is per unit of 4m(1 - m), so in USD it also depends on the price.",
    )


def table_pairs(summary: dict[str, Any], config: StudyConfig) -> str:
    """B3: the nine other pairs."""
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
        "the effect of the premium alone in the pairs with `always_on` or `pull_only`: "
        "`pooled_premium`, `flat_matched` and `release_aware` also add the terminal variance "
        "m(1 - m) to the variance of the reservation price, which `always_on` and `pull_only` do "
        "not, so their differences from those two policies are the effect of the premium and of a "
        "stronger inventory skew together; the pairs with `pooled_premium` have a sign that "
        "follows the level of the pooled premium (B8, B9); and a family-wise error rate: the "
        "cells are not adjusted for multiple comparisons.",
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
            f"events would make the expected half-width equal the mean difference over all "
            f"planted datasets ({_num(size['mean'], signed=True)} USD per event) and about "
            f"{size['events_power_80']:,.0f} would give a two-sided 5% test 80% power."
        )
    else:
        extrapolation = (" The mean difference over all planted datasets is zero, so no number "
                         "of events is given.")
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


def _ratio(numerator: float, denominator: float) -> str:
    """``numerator / denominator`` to two decimals, or ``n/a`` when it is not defined."""
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or denominator == 0.0:
        return "n/a"
    return f"{numerator / denominator:.2f}"


def _change(first: float, last: float) -> str:
    """Relative change from ``first`` to ``last`` in percent with a sign, or ``n/a``."""
    if not (math.isfinite(first) and math.isfinite(last)) or first == 0.0:
        return "n/a"
    return f"{100.0 * (last / first - 1.0):+.0f}%"


def _and(items: list[str]) -> str:
    """The items as a list in words: ``a``, ``a and b`` or ``a, b and c``."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _sign_pattern(means: list[float]) -> str:
    """Words for what a difference does across the rows of a table, from its means."""
    if all(m > 0.0 for m in means):
        return "stays positive"
    if all(m < 0.0 for m in means):
        return "stays negative"
    return "does not keep one sign"


def table_premium_check(check: pd.DataFrame) -> str:
    """B7: estimated premiums (by bucket, pooled and matched) against their target and the latent
    price impact."""
    columns = ["estimate_end", "target", "latent", "estimate_burn_in"]
    mean = check.groupby(["world", "bucket"], sort=False)[columns].mean()
    body = [[_label(world), bucket, *[_num(100.0 * mean.loc[(world, bucket), c]) for c in columns]]
            for world, bucket in mean.index]
    near = mean.loc[(PLANTED_WORLD, BUCKET_LABELS[0])]
    far = mean.loc[(PLANTED_WORLD, BUCKET_LABELS[-1])]
    pooled = float(mean.loc[(PLANTED_WORLD, "pooled"), "estimate_end"])
    matched = float(mean.loc[(PLANTED_WORLD, "matched"), "estimate_end"])
    gap = float((100.0 * (mean["estimate_end"] - mean["target"]).abs()).max())
    reading = (
        f"In the planted world the premium estimated at the end is "
        f"{_num(100 * near['estimate_end'])} cents in the bucket {BUCKET_LABELS[0]} minutes and "
        f"{_num(100 * far['estimate_end'])} cents in the bucket {BUCKET_LABELS[-1]} (latent price "
        f"impact {_num(100 * near['latent'])} and {_num(100 * far['latent'])}); after the burn-in "
        f"it is {_num(100 * near['estimate_burn_in'])} in the first bucket; the largest gap "
        f"between the estimate at the end and its target is {_num(gap)} cents. The pooled premium "
        f"estimated at the end is {_num(100 * pooled)} cents and the matched premium, the flat "
        f"premium of `flat_matched` (the bucket premiums averaged over the decisions of the "
        f"quoting window), is {_num(100 * matched)} cents: a ratio of {_ratio(pooled, matched)} "
        f"of the pooled premium to the matched premium."
    )
    table = _markdown(
        ["World", "Minutes to release (or pooled, or matched)", "Estimate at the end, cents",
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
    """B8, exploratory: the primary and the secondary difference in the planted world by hours of
    trading before each release, with the premiums estimated at the end of each dataset."""
    rows = summary["windows"]
    near, far = BUCKET_LABELS[0], BUCKET_LABELS[-1]
    body, sentences = [], []
    for s in rows:
        n, hours, t = s["n_datasets"], f"{s['window_hours']:g}", s["secondary"]
        body.append([
            hours, str(n),
            f"{_num(s['mean'], signed=True)} ({_num(s['mc_se'])})",
            str(s["below_zero"]), str(s["above_zero"]),
            f"{_num(t['mean'], signed=True)} ({_num(t['mc_se'])})",
            str(t["below_zero"]), str(t["above_zero"]),
            _num(100.0 * s["premium_pooled"]), _num(100.0 * s["premium_matched"]),
            _num(100.0 * s["premium_near"]), _num(100.0 * s["premium_far"]),
        ])
        sentences.append(
            f"With {hours} hours of trading before each release {_earned(s['mean'])} (the primary "
            f"difference; mean of {n} datasets; {_excluded(s['below_zero'], s['above_zero'], n)}) "
            f"and {_earned(t['mean'], SECONDARY)} (the secondary difference; "
            f"{_excluded(t['below_zero'], t['above_zero'], t['n_datasets'])})."
        )
    labels = (("premium_pooled", "the pooled premium"), ("premium_matched", "the matched premium"),
              ("premium_near", f"the premium of the bucket {near}"),
              ("premium_far", f"the premium of the bucket {far}"))
    first, last = rows[0], rows[-1]
    if len(rows) > 1:
        moves = _and([f"{label} from {_num(100.0 * first[k])} to {_num(100.0 * last[k])} cents "
                       f"({_change(first[k], last[k])})" for k, label in labels])
        movement = (
            f" From {first['window_hours']:g} to {last['window_hours']:g} hours of trading the "
            f"premiums estimated at the end move as follows: {moves}. Across the windows the "
            f"primary difference {_sign_pattern([s['mean'] for s in rows])} and the secondary "
            f"difference {_sign_pattern([s['secondary']['mean'] for s in rows])}."
        )
    else:
        values = _and([f"{label} is {_num(100.0 * first[k])} cents" for k, label in labels])
        movement = f" The premiums estimated at the end: {values}."
    reading = (
        "Exploratory: added after the first results because B1 and the rehearsal of the protocol "
        "analysis disagreed in sign for the first primary pair; it was not part of the design "
        "fixed beforehand. " + " ".join(sentences) + movement + f" Intervals are percentile "
        f"intervals of a cluster bootstrap over events with {config.boot_reps:,} replications."
    )
    table = _markdown(
        ["Hours of trading before each release", "Datasets",
         f"Primary, {_pair(*PRIMARY)}: mean difference, USD per event (MC SE over datasets)",
         "Primary: 95% interval below zero", "Primary: 95% interval above zero",
         f"Secondary, {_pair(*SECONDARY)}: mean difference, USD per event (MC SE over datasets)",
         "Secondary: 95% interval below zero", "Secondary: 95% interval above zero",
         "Pooled premium at the end, cents", "Matched premium at the end, cents",
         f"Premium of the bucket {near} at the end, cents",
         f"Premium of the bucket {far} at the end, cents"],
        body,
    )
    return _block(
        "B8", f"Exploratory: {_pair(*PRIMARY)} and {_pair(*SECONDARY)} by hours of trading before "
        "each release, planted world", reading, table,
        "what either difference is made of: B9 and B10 change the level of the flat premium and "
        "describe what happens, they do not test it; and anything about real markets: the hours "
        "of trading are those of the synthetic world.",
    )


def table_levels(summary: dict[str, Any], config: StudyConfig) -> str:
    """B9, exploratory: the primary difference in the planted world by the scale F of the flat
    premium of ``flat_matched``."""
    rows = summary["levels"]
    body = [[
        f"{s['flat_scale']:g}", _num(100.0 * s["premium_flat"]),
        f"{_num(s['mean'], signed=True)} ({_num(s['mc_se'])})",
        f"{_num(s['min'], signed=True)} to {_num(s['max'], signed=True)}",
        str(s["below_zero"]), str(s["above_zero"]),
    ] for s in rows]
    sentences = [
        "Exploratory: added after an independent review showed that the first primary pair "
        "followed the level of the pooled premium. The flat premium of "
        f"`{PRIMARY[1]}` is multiplied by F and F = 1 is the primary comparison."
    ]
    at_one = next((s for s in rows if s["flat_scale"] == 1.0), None)
    top = max(rows, key=lambda s: s["flat_scale"])

    def at(label: str, s: dict[str, Any]) -> str:
        return (f"At {label} {_earned(s['mean'])} (mean of {s['n_datasets']} datasets; "
                f"{_excluded(s['below_zero'], s['above_zero'], s['n_datasets'])}).")

    if at_one is not None:
        sentences.append(at("F = 1", at_one))
    if top is not at_one:
        sentences.append(at(f"the largest F = {top['flat_scale']:g}", top))
    changes = sign_changes([s["flat_scale"] for s in rows], [s["mean"] for s in rows])
    if not changes:
        sentences.append(
            f"The mean difference does not change sign on the grid of F from "
            f"{min(s['flat_scale'] for s in rows):g} to {top['flat_scale']:g}."
        )
    else:
        where, f_a, f_b = changes[0]
        many = f" (the first of {len(changes)} sign changes)" if len(changes) > 1 else ""
        sentences.append(
            f"The mean difference changes sign at F = {where:.2f}{many}, by linear interpolation "
            f"between the means at F = {f_a:g} and F = {f_b:g}; this describes the grid and is "
            f"not an estimate with an interval."
        )
    first = rows[0]
    pooled = first["premium_pooled"]
    matched = first["premium_flat"] / first["flat_scale"]
    sentences.append(
        f"The mean pooled premium at the end is {_num(100.0 * pooled)} cents and the mean matched "
        f"premium {_num(100.0 * matched)} cents, a ratio of {_ratio(pooled, matched)}, so at the "
        f"end of a dataset `pooled_premium` quotes the premium that `{PRIMARY[1]}` quotes at "
        f"F = {_ratio(pooled, matched)}."
    )
    losing = sum(summary["policies"][f"{PLANTED_WORLD}:{p}"]["pnl"] < 0.0 for p in POLICIES)
    if losing == len(POLICIES):
        sentences.append("Every policy loses money in the planted world (B2), so a wider quote "
                         "loses less.")
    else:
        sentences.append(f"Net P&L per event is negative for {losing} of {len(POLICIES)} policies "
                         f"in the planted world (B2).")
    sentences.append("B10 gives the scale at which the two policies quote the same premium at "
                     "their fills.")
    sentences.append(f"Intervals are percentile intervals of a cluster bootstrap over events with "
                     f"{config.boot_reps:,} replications.")
    table = _markdown(
        ["Scale F of the flat premium", "Flat premium quoted at the end, cents",
         f"{_pair(*PRIMARY)}: mean difference, USD per event (MC SE over datasets)",
         "Range over datasets", "95% interval below zero", "95% interval above zero"],
        body,
    )
    return _block(
        "B9", f"Exploratory: {_pair(*PRIMARY)} by the scale of the flat premium, planted world",
        " ".join(sentences), table,
        "that either policy earns money on real markets, and that the planted informed share is "
        "what makes the bucket premiums useful at F = 1: the null world has buckets that differ "
        "too (see the null row of B1).",
    )


def table_fill_matched(summary: dict[str, Any], config: StudyConfig) -> str:
    """B10, exploratory: the primary difference in both worlds when ``flat_matched`` quotes, at its
    fills, the same premium as ``release_aware`` (the flat premium scaled by ``F_eq``)."""
    body, sentences = [], []
    for world, _ in config.worlds():
        s = summary["fill_matched"][world]
        n = s["n_datasets"]
        aware, base, flat = (100.0 * s[k] for k in ("fill_aware", "fill_flat_base", "fill_flat"))
        body.append([
            _label(world),
            f"{_num(s['flat_scale'])} ({_num(s['flat_scale_min'])} to {_num(s['flat_scale_max'])})",
            _num(aware), _num(base), _num(flat),
            f"{_num(s['mean'], signed=True)} ({_num(s['mc_se'])})",
            f"{_num(s['min'], signed=True)} to {_num(s['max'], signed=True)}",
            str(s["below_zero"]), str(s["above_zero"]),
        ])
        sentences.append(
            f"In the {_label(world)} world `{PRIMARY[0]}` quotes {_num(aware)} cents per unit of "
            f"4m(1 - m) at its fills against {_num(base)} for `{PRIMARY[1]}` "
            f"({_change(base, aware)}). With the flat premium scaled by F_eq = "
            f"{_num(s['flat_scale'])} both quote about the same premium at their fills "
            f"({_num(aware)} and {_num(flat)} cents) and {_earned(s['mean'])} (mean of {n} "
            f"datasets; {_excluded(s['below_zero'], s['above_zero'], n)})."
        )
    table = _markdown(
        ["World", "Scale F_eq of the flat premium, mean (range over datasets)",
         f"Premium at the fills of `{PRIMARY[0]}`, cents", f"Premium at the fills of "
         f"`{PRIMARY[1]}` at F = 1, cents",
         f"Premium at the fills of `{PRIMARY[1]}` at F_eq, cents",
         f"{_pair(*PRIMARY)} at F_eq: mean difference, USD per event (MC SE over datasets)",
         "Range over datasets", "95% interval below zero", "95% interval above zero"],
        body,
    )
    return _block(
        "B10", f"Exploratory: {_pair(*PRIMARY)} at an equal premium at the fills",
        "Exploratory: added after a second independent review showed that `" + PRIMARY[1]
        + "` is matched to `" + PRIMARY[0] + "` on the average of the premium over the clock "
        "and not where the quotes are hit (B1, B2). " + " ".join(sentences) + " F_eq is the "
        f"premium per unit of 4m(1 - m) at the fills of `{PRIMARY[0]}` divided by that of "
        f"`{PRIMARY[1]}`, from one run of each; `{PRIMARY[1]}` is then run again with its flat "
        f"premium multiplied by F_eq. Intervals are percentile intervals of a cluster bootstrap "
        f"over events with {config.boot_reps:,} replications.", table,
        "which level is the right one to hold fixed: the average over the clock is fixed in "
        "advance and is not what the P&L responds to, and the premium at the fills is what it "
        "responds to but is partly an outcome of the schedule, because a policy is filled less "
        "where it quotes more premium; that F_eq is known beforehand: it is estimated from the "
        "same run as the difference, and the intervals treat it as given; and anything about "
        "real markets.",
    )


def write_tables(summary: dict[str, Any], config: StudyConfig, check: pd.DataFrame) -> str:
    """The text of ``tables.md``."""
    return "\n".join([
        table_primary(summary, config), table_policies(summary, config),
        table_pairs(summary, config), table_sweep(summary, config), table_touch(summary, config),
        table_events_needed(summary, config), table_premium_check(check),
        table_windows(summary, config), table_levels(summary, config),
        table_fill_matched(summary, config),
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
    levels = level_runs(config)
    fills = fill_matched_runs(config)
    policies = pd.concat([policies, sweep_policies], ignore_index=True)
    pairs = pd.concat([pairs, sweep_pairs], ignore_index=True)
    summary = build_summary(
        config, policies, pairs, needed, sample_size(per_event), check, windows, levels, fills
    )

    csv = {"index": False, "float_format": "%.10g"}
    columns = ["world", "seed", "event", "policy", "pnl", "gross", "fees", "fills", "contracts",
               "inventory_end", "marked_pnl", "quotes_live_share", "mean_spread", "fill_premium",
               "evaluated"]
    per_event[columns].to_csv(out / "per_event.csv", **csv)
    policies.to_csv(out / "policies.csv", **csv)
    pairs.to_csv(out / "pairs.csv", **csv)
    needed.to_csv(out / "events_needed.csv", **csv)
    check.to_csv(out / "premium_check.csv", **csv)
    windows.to_csv(out / "windows.csv", **csv)
    levels.to_csv(out / "levels.csv", **csv)
    fills.to_csv(out / "fill_matched.csv", **csv)
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
        f"policies {', '.join(WINDOW_POLICIES)}; bootstrap generators for the primary difference "
        f"{PRIMARY[0]} minus {PRIMARY[1]}: default_rng([{SEED}, {KIND_WINDOW}, window index, "
        f"seed]) and for the secondary difference {SECONDARY[0]} minus {SECONDARY[1]}: "
        f"default_rng([{SEED}, {KIND_WINDOW}, window index, seed, {SECONDARY_KEY}])",
        f"B9, added after an independent review of the first results (exploratory): the planted "
        f"world of the main grid, the same dataset seeds, {PRIMARY[0]} once and {PRIMARY[1]} with "
        f"the flat premium multiplied by F (QuoteParams.flat_scale) in "
        f"{{{', '.join(f'{f:g}' for f in config.sweep_levels)}}}, no other policy; the difference "
        f"{PRIMARY[0]} minus {PRIMARY[1]} per evaluated event with bootstrap generators "
        f"default_rng([{SEED}, {KIND_LEVEL}, index of F, seed])",
        f"B10, added after a second independent review (exploratory): both worlds of the main "
        f"grid, the same dataset seeds, {PRIMARY[0]} and {PRIMARY[1]} once, then {PRIMARY[1]} "
        f"again with the flat premium multiplied by F_eq, the premium per unit of 4m(1 - m) at "
        f"the fills of {PRIMARY[0]} over that of {PRIMARY[1]} in the first run; bootstrap "
        f"generators default_rng([{SEED}, {KIND_FILLS}, index of the world, seed])",
    ]
    manifest.write(
        out, "backtest_study",
        ["per_event.csv", "policies.csv", "pairs.csv", "events_needed.csv", "premium_check.csv",
         "windows.csv", "per_event_windows.csv", "levels.csv", "fill_matched.csv", "summary.json",
         "tables.md"],
        notes=f"{config.quote.min_history_events} burn-in events per dataset; sweeps on dataset "
              f"seed {config.sweep_seed}",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    """Run the study of the module docstring and write its files to ``--out``."""
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
