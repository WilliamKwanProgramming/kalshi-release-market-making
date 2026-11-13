"""The analysis fixed in docs/protocol.md, run on one dataset (data contract of docs/model.md).

:func:`prepare` turns the three frames of a dataset (tape, markets, events) into the sample the
protocol analyses: events with the same release time are merged into one release (they share the
news, so the bootstrap resamples releases), markets are dropped by the rules of the protocol, and
the number removed by each rule is recorded. :class:`Settings` holds every value the protocol fixes;
a run with any other value is a deviation and says so (:func:`deviations`).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from kalshi_mm.backtest import walk_forward
from kalshi_mm.calibration import skill_interval, walk_forward_recalibration
from kalshi_mm.markout import (
    bucket_means,
    release_effect,
    scaled_adverse_selection,
    trade_markouts,
)
from kalshi_mm.quoting import POLICIES, QuoteParams
from kalshi_mm.stats import (
    cluster_weights,
    holm_adjust,
    p_at_or_below_zero,
    percentile_interval,
    ratio_of_sums,
)
from kalshi_mm.tape import mid_proxy_asof

__all__ = [
    "DEFAULT",
    "PAIRS",
    "PRIMARY_PAIR",
    "Dataset",
    "Results",
    "Settings",
    "data_label",
    "deviations",
    "family",
    "merge_releases",
    "p1_adverse_selection",
    "p2_pricing",
    "p3_recalibration",
    "paired_mean",
    "prepare",
    "run_protocol",
    "with_overrides",
]

SOURCE_LABELS = {"synthetic": "synthetic data", "kalshi-public-api": "real data"}


@dataclass(frozen=True)
class Settings:
    """Every value docs/protocol.md fixes. Quote parameters are those of ``QuoteParams``."""

    n_boot: int = 10_000
    seed: int = 20261113
    level: float = 0.95
    alpha: float = 0.05
    near: tuple[float, float] = (0.0, 30.0)
    quiet: tuple[float, float] = (360.0, math.inf)
    horizon_s: float = 300.0
    extra_horizons_s: tuple[float, ...] = (60.0, 900.0)
    min_trades: int = 50
    quote: QuoteParams = field(default_factory=QuoteParams)
    recal_hours: float = 24.0
    recal_min_releases: int = 50
    recal_refit_every: int = 10
    min_releases: int = 100


DEFAULT = Settings()


def deviations(settings: Settings) -> list[str]:
    """``"name: value (protocol: value)"`` for every setting that differs from the protocol."""
    out: list[str] = []

    def walk(prefix: str, now: Any, ref: Any) -> None:
        if dataclasses.is_dataclass(now):
            for f in dataclasses.fields(now):
                walk(f"{prefix}{f.name}.", getattr(now, f.name), getattr(ref, f.name))
        elif now != ref:
            out.append(f"{prefix[:-1]}: {now} (protocol: {ref})")

    walk("", settings, DEFAULT)
    return out


def with_overrides(settings: Settings, assignments: Sequence[str]) -> Settings:
    """``settings`` with ``name=value`` assignments applied (``quote.<name>`` for quote parameters).

    A value is converted to the type of the current one (int, float or str). The windows and
    horizons (tuples) and flags cannot be set this way. An unknown name, a missing ``=`` or a value
    of the wrong kind raises ``ValueError``. Whatever differs from the protocol afterwards is listed
    by :func:`deviations`.
    """
    out = settings
    for item in assignments:
        name, equals, text = item.partition("=")
        if not equals:
            raise ValueError(f"expected name=value, got {item!r}")
        nested = name.startswith("quote.")
        owner = out.quote if nested else out
        key = name[len("quote."):] if nested else name
        if key == "quote" or key not in {f.name for f in dataclasses.fields(owner)}:
            raise ValueError(f"unknown setting {name!r}")
        current = getattr(owner, key)
        if isinstance(current, bool) or not isinstance(current, int | float | str):
            raise ValueError(f"{name} cannot be set from the command line")
        try:
            value = type(current)(text)
        except ValueError as err:
            raise ValueError(f"bad value for {name}: {text!r}") from err
        owner = dataclasses.replace(owner, **{key: value})
        out = dataclasses.replace(out, quote=owner) if nested else owner
    return out


def data_label(source: str) -> str:
    """How tables name their data: ``"synthetic data"`` or ``"real data"``; other sources raise."""
    if source not in SOURCE_LABELS:
        raise ValueError(f"unknown data source {source!r}; expected one of {sorted(SOURCE_LABELS)}")
    return SOURCE_LABELS[source]


@dataclass(frozen=True, eq=False)
class Dataset:
    """The sample after :func:`prepare`: three frames, the series of each market, the counts."""

    tape: pd.DataFrame
    markets: pd.DataFrame
    events: pd.DataFrame
    market_series: pd.Series
    counts: dict[str, int]


def merge_releases(
    tape: pd.DataFrame, markets: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Merge events that share a release time into one release (docs/protocol.md, Data).

    The release id is the event ids joined by ``+`` in sorted order (a release with one event keeps
    its id) and its series is the sorted series joined the same way. Returns the three frames with
    ``event`` replaced by the release id, and a Series from market to the series of its original
    event, which the per-series tables use.
    """
    if events["event"].duplicated().any():
        raise ValueError("events must have unique ids")
    if events["release_ts"].isna().any():
        raise ValueError("every event needs a release time")
    release_of: dict[str, str] = {}
    rows = []
    for release_ts, group in events.groupby("release_ts", sort=True):
        ids = sorted(group["event"])
        release_id = "+".join(ids)
        release_of.update(dict.fromkeys(ids, release_id))
        rows.append((release_id, float(release_ts), "+".join(sorted(set(group["series"])))))
    unknown = sorted(set(markets["event"]) - set(release_of))
    if unknown:
        raise ValueError(f"markets refer to events that are not in the events frame: {unknown[:3]}")
    series_of_event = events.set_index("event")["series"]
    market_series = pd.Series(
        markets["event"].map(series_of_event).to_numpy(),
        index=markets["market"].to_numpy(),
        name="series",
    )
    merged_events = pd.DataFrame(rows, columns=["event", "release_ts", "series"])
    merged_markets = markets.assign(event=markets["event"].map(release_of))
    merged_tape = tape.assign(event=tape["event"].map(release_of))
    return merged_tape, merged_markets, merged_events, market_series


def prepare(
    tape: pd.DataFrame, markets: pd.DataFrame, events: pd.DataFrame, settings: Settings = DEFAULT
) -> Dataset:
    """The sample of the protocol and the number of markets removed by each rule.

    Order of the steps: merge events with the same release time; drop markets without a settled
    outcome; drop markets with fewer than ``settings.min_trades`` trades before the release; drop
    markets whose mid proxy (docs/model.md section 4) is never defined at the time of a trade inside
    the last ``settings.quote.window_hours`` hours before the release; drop releases left without a
    market. The tape is sorted by (market, ts, seq) and the markets by id, so the result does not
    depend on the row order of the input.
    """
    counts = {"events": int(len(events))}
    tape, markets, events, market_series = merge_releases(tape, markets, events)
    markets = markets.sort_values("market", kind="mergesort").reset_index(drop=True)
    counts["releases"] = int(len(events))
    counts["markets"] = int(len(markets))
    release_ts = events.set_index("event")["release_ts"]

    tape = tape.sort_values(["market", "ts", "seq"], kind="mergesort").reset_index(drop=True)
    tape = tape[tape["ts"] < tape["event"].map(release_ts)]
    n_trades = tape.groupby("market").size()

    keep = markets["outcome"].notna()
    counts["markets without a settled outcome"] = int((~keep).sum())
    enough = markets["market"].map(n_trades).fillna(0) >= settings.min_trades
    counts[f"markets with fewer than {settings.min_trades} trades"] = int((keep & ~enough).sum())
    keep &= enough

    window_s = 3600.0 * settings.quote.window_hours
    quotable = set()
    for market, trades in tape[tape["market"].isin(markets.loc[keep, "market"])].groupby("market"):
        start = float(release_ts[trades["event"].iloc[0]]) - window_s
        ts = trades["ts"].to_numpy()
        inside = ts >= start
        proxy = mid_proxy_asof(
            ts, trades["side"].to_numpy(), trades["price"].to_numpy(), ts[inside],
            stale_s=settings.quote.stale_s,
        )
        if np.any(~np.isnan(proxy)):
            quotable.add(market)
    quotable_mask = markets["market"].isin(quotable)
    counts["markets without a mid proxy in the quoting window"] = int((keep & ~quotable_mask).sum())
    keep &= quotable_mask

    markets = markets[keep].reset_index(drop=True)
    tape = tape[tape["market"].isin(markets["market"])].reset_index(drop=True)
    events = events[events["event"].isin(markets["event"])].reset_index(drop=True)
    counts["markets kept"] = int(len(markets))
    counts["releases kept"] = int(len(events))
    counts["trades kept"] = int(len(tape))
    return Dataset(tape, markets, events, market_series.loc[markets["market"]], counts)


def _effect(
    markouts: pd.DataFrame,
    settings: Settings,
    column: str = "adverse_selection",
    near: tuple[float, float] | None = None,
) -> Any:
    """``release_effect`` with the protocol's windows and bootstrap, or why it is undefined.

    ``near`` replaces the near window of the settings when it is given.
    """
    try:
        return release_effect(
            markouts, near=settings.near if near is None else near, quiet=settings.quiet,
            column=column, B=settings.n_boot, seed=settings.seed, level=settings.level,
        )
    except ValueError as err:
        return str(err)


def p1_adverse_selection(ds: Dataset, settings: Settings = DEFAULT) -> dict[str, Any]:
    """P1 of docs/protocol.md: the release effect ``D`` of the 5-minute adverse selection.

    ``primary`` is the tested quantity (``markout.release_effect`` on the 5-minute markouts of
    every trade). The rest is reported, not tested: ``buckets`` (mean per bucket), ``by_series`` (D
    for the trades of each series), ``horizons`` (D at the extra horizons), ``settlement`` (D of
    the settlement markout), ``scaled`` (D of the markout divided by 4 m (1 - m), for trades whose
    mid proxy m is inside the quoting range: a level of the price and a surge of the trade rate
    both change D without a change of the informed share, see docs/model.md section 5) and
    ``uncut`` (D with the near window starting at the horizon, so that no markout is cut at the
    release). An effect that cannot be estimated, for example in a series with no trade in the
    near or the quiet window, is the error message instead of a dict.
    """
    q = settings.quote
    markouts = trade_markouts(
        ds.tape, ds.events, ds.markets, horizon_s=settings.horizon_s, stale_s=q.stale_s
    )
    series = markouts["market"].map(ds.market_series)
    scaled = markouts.assign(
        adverse_selection_scaled=scaled_adverse_selection(
            markouts, min_mid=q.min_mid, max_mid=q.max_mid
        )
    )
    horizons = {}
    for h in settings.extra_horizons_s:
        other = trade_markouts(ds.tape, ds.events, ds.markets, horizon_s=h, stale_s=q.stale_s)
        horizons[h] = _effect(other, settings)
    return {
        "primary": release_effect(
            markouts, near=settings.near, quiet=settings.quiet, B=settings.n_boot,
            seed=settings.seed, level=settings.level,
        ),
        "buckets": bucket_means(markouts),
        "by_series": {s: _effect(markouts[series == s], settings) for s in sorted(series.unique())},
        "horizons": horizons,
        "settlement": _effect(markouts, settings, "settlement_markout"),
        "scaled": _effect(scaled, settings, "adverse_selection_scaled"),
        "uncut": _effect(markouts, settings, near=(settings.horizon_s / 60.0, settings.near[1])),
        "n_trades": int(len(markouts)),
    }


PRIMARY_PAIR = ("release_aware", "flat_matched")
PAIRS = (
    PRIMARY_PAIR,
    ("release_aware", "pooled_premium"),
    ("flat_matched", "pooled_premium"),
    ("always_on", "pooled_premium"),
    ("always_on", "flat_matched"),
    ("always_on", "release_aware"),
    ("pull_only", "always_on"),
    ("pull_only", "pooled_premium"),
    ("pull_only", "flat_matched"),
    ("pull_only", "release_aware"),
)


def paired_mean(diff: np.ndarray, settings: Settings) -> dict[str, Any] | None:
    """Mean of per-release differences with the cluster bootstrap over releases and its p-value.

    ``p_one_sided`` is the share of replications at or below zero (docs/protocol.md), never below
    ``1 / (B + 1)``. An empty ``diff`` has no mean: the result is None.
    """
    diff = np.asarray(diff, dtype=float)
    if diff.size == 0:
        return None
    weights = cluster_weights(diff.size, settings.n_boot, np.random.default_rng(settings.seed))
    reps = ratio_of_sums(diff, np.ones(diff.size), weights)
    return {
        "mean": float(diff.mean()),
        "ci": percentile_interval(reps, settings.level),
        "p_one_sided": p_at_or_below_zero(reps),
        "n_releases": int(diff.size),
        "B": int(settings.n_boot),
        "seed": int(settings.seed),
    }


def _per_release(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Sum of ``column`` over the markets of each release: one row per release, one column per
    policy, for the evaluated releases only."""
    evaluated = frame[frame["evaluated"]]
    return evaluated.groupby(["event", "policy"])[column].sum().unstack("policy").sort_index()


def _primary_difference(frame: pd.DataFrame, settings: Settings) -> dict[str, Any] | None:
    by_release = _per_release(frame, "pnl")
    diff = by_release[PRIMARY_PAIR[0]] - by_release[PRIMARY_PAIR[1]]
    return paired_mean(diff.to_numpy(), settings)


def p2_pricing(ds: Dataset, settings: Settings = DEFAULT) -> dict[str, Any]:
    """P2 of docs/protocol.md: does a premium by minutes to release beat a flat premium of the
    same average size?

    The walk-forward backtest (docs/model.md sections 6 and 7) runs the five policies. The tested
    quantity is the net settled P&L per release (sum over its markets) of ``release_aware`` minus
    ``flat_matched``, paired by release, over the releases after the burn-in. ``flat_matched``
    quotes the average of the bucket premiums over the quoting window, so the two policies differ
    in when a premium is applied and not in how large it is on average. Reported, not tested: every
    other pair of policies (among them ``release_aware`` minus ``pooled_premium``, the comparison
    of the first version of the protocol, whose sign follows the level of the pooled premium), each
    policy's totals, the primary difference by series and in the two halves of the sample split at
    the median release time, and the primary difference with the touch fill rule and with maker fee
    coefficient 0.
    """
    frame = walk_forward(ds.tape, ds.markets, ds.events, POLICIES, settings.quote)
    frame = frame.assign(original_series=frame["market"].map(ds.market_series))
    pnl = _per_release(frame, "pnl")
    pairs = {f"{a} - {b}": paired_mean((pnl[a] - pnl[b]).to_numpy(), settings) for a, b in PAIRS}
    totals = {
        policy: {
            column: float(_per_release(frame, column)[policy].mean())
            for column in ("pnl", "gross", "fees", "fills", "contracts", "marked_pnl")
        }
        | {"quotes_live_share": float(frame.loc[frame["evaluated"] & (frame["policy"] == policy),
                                                 "quotes_live_share"].mean())}
        for policy in POLICIES
    }
    by_series = {
        s: _primary_difference(frame[frame["original_series"] == s], settings)
        for s in sorted(frame["original_series"].unique())
    }
    release_ts = ds.events.set_index("event")["release_ts"]
    diff = pnl[PRIMARY_PAIR[0]] - pnl[PRIMARY_PAIR[1]]
    when = release_ts.loc[diff.index]
    early = when <= when.median()
    halves = {
        "first half": paired_mean(diff[early].to_numpy(), settings),
        "second half": paired_mean(diff[~early].to_numpy(), settings),
    }
    sensitivity = {}
    for name, params in (
        ("touch fill rule", dataclasses.replace(settings.quote, fill_rule="touch")),
        ("maker fee coefficient 0", dataclasses.replace(settings.quote, maker_fee_coeff=0.0)),
    ):
        variant = walk_forward(ds.tape, ds.markets, ds.events, PRIMARY_PAIR, params)
        sensitivity[name] = _primary_difference(variant, settings)
    return {
        "frame": frame,
        "primary": _primary_difference(frame, settings),
        "pairs": pairs,
        "totals": totals,
        "by_series": by_series,
        "halves": halves,
        "sensitivity": sensitivity,
        "n_releases": int(len(diff)),
    }


def p3_recalibration(ds: Dataset, settings: Settings = DEFAULT) -> dict[str, Any]:
    """P3 of docs/protocol.md: the skill of the walk-forward recalibration of the mid proxy.

    The price of a market is the proxy at ``settings.recal_hours`` before its release; a market
    with no proxy then is left out and counted in ``n_without_price`` (``n_releases_with_price``
    is the number of releases that still have a market). The skill is
    ``calibration.skill_interval`` of the walk-forward recalibration (``recal_min_releases``
    releases before the first prediction, refit every ``recal_refit_every`` releases) with
    releases as clusters. Reported, not tested: the fitted ``(a, b)`` over time (``walk``, one
    row per scored market, with the coefficients that scored it) and the skill by series (the
    error message where a series has no defined skill). ``primary`` is None, with the reason in
    ``error``, when a fit or the skill fails, and None without a reason when there are too few
    releases for a first fit.
    """
    release_ts = ds.events.set_index("event")["release_ts"]
    rows = []
    for market, trades in ds.tape.groupby("market", sort=True):
        event = trades["event"].iloc[0]
        query = float(release_ts[event]) - 3600.0 * settings.recal_hours
        price = mid_proxy_asof(
            trades["ts"].to_numpy(), trades["side"].to_numpy(), trades["price"].to_numpy(),
            np.array([query]), stale_s=settings.quote.stale_s,
        )[0]
        rows.append((market, event, float(release_ts[event]), float(price)))
    prices = pd.DataFrame(rows, columns=["market", "event", "release_ts", "price"])
    prices["outcome"] = prices["market"].map(ds.markets.set_index("market")["outcome"])
    prices["series"] = prices["market"].map(ds.market_series)
    usable = prices.dropna(subset=["price"])
    out: dict[str, Any] = {
        "n_markets": int(len(prices)),
        "n_without_price": int(len(prices) - len(usable)),
        "n_releases_with_price": int(usable["release_ts"].nunique()),
        "primary": None,
        "walk": None,
        "by_series": {},
        "error": None,
    }
    if usable["release_ts"].nunique() <= settings.recal_min_releases:
        return out
    try:
        walk = walk_forward_recalibration(
            usable["price"].to_numpy(), usable["outcome"].to_numpy(),
            usable["release_ts"].to_numpy(), min_events=settings.recal_min_releases,
            refit_every=settings.recal_refit_every, cluster=usable["event"].to_numpy(),
        )
        primary = skill_interval(
            walk, walk["cluster"], B=settings.n_boot, seed=settings.seed, level=settings.level
        )
    except ValueError as err:  # a fit that fails, or a skill that is undefined: P3 is not computed
        out["error"] = str(err)
        return out
    out["walk"], out["primary"] = walk, primary
    series_of = usable.set_index("event")["series"]
    series = walk["cluster"].map(series_of)
    for s in sorted(series.unique()):
        part = walk[series == s]
        try:
            out["by_series"][s] = skill_interval(
                part, part["cluster"], B=settings.n_boot, seed=settings.seed, level=settings.level
            )
        except ValueError as err:
            out["by_series"][s] = str(err)
    return out


def family(p1: dict[str, Any], p2: dict[str, Any], p3: dict[str, Any], n_releases: int,
           settings: Settings = DEFAULT) -> dict[str, Any]:
    """The three tests as one family: raw p-values, Holm-adjusted p-values and the verdicts.

    A test that cannot be computed counts as p = 1 in the family. ``supported`` needs a
    Holm-adjusted p-value below ``settings.alpha``. P1 and P2 are labelled exploratory instead when
    fewer than ``settings.min_releases`` releases enter the test (docs/protocol.md): P1 uses the
    ``n_releases`` releases kept, P2 the ``p2["n_releases"]`` releases after the burn-in.
    """
    raw = {
        "P1": p1["primary"]["p_one_sided"],
        "P2": None if p2["primary"] is None else p2["primary"]["p_one_sided"],
        "P3": None if p3["primary"] is None else p3["primary"]["p_one_sided"],
    }
    adjusted = holm_adjust([1.0 if v is None else v for v in raw.values()])
    used = {"P1": n_releases, "P2": p2["n_releases"]}  # P3 is never labelled exploratory
    out = {}
    for (name, p), adj in zip(raw.items(), adjusted, strict=True):
        label = "not computed" if p is None else (
            "exploratory" if name in used and used[name] < settings.min_releases
            else ("supported" if adj < settings.alpha else "not supported")
        )
        out[name] = {"p_one_sided": p, "p_holm": float(adj), "verdict": label}
    return out


@dataclass(frozen=True, eq=False)
class Results:
    """Everything the protocol asks for, from one run of :func:`run_protocol`."""

    source: str
    settings: Settings
    dataset: Dataset
    p1: dict[str, Any]
    p2: dict[str, Any]
    p3: dict[str, Any]
    family: dict[str, Any]
    deviations: list[str]


def run_protocol(
    tape: pd.DataFrame,
    markets: pd.DataFrame,
    events: pd.DataFrame,
    source: str,
    settings: Settings = DEFAULT,
) -> Results:
    """Prepare the dataset and compute P1, P2, P3 and the Holm-adjusted family (docs/protocol.md).

    ``source`` is the label stored with the dataset (``"synthetic"`` or ``"kalshi-public-api"``);
    an unknown source raises. No column starting with ``latent_`` is read.
    """
    data_label(source)
    ds = prepare(tape, markets, events, settings)
    if ds.counts["releases kept"] == 0:
        raise ValueError(f"no release is left after the sample rules: {ds.counts}")
    p1 = p1_adverse_selection(ds, settings)
    p2 = p2_pricing(ds, settings)
    p3 = p3_recalibration(ds, settings)
    return Results(
        source, settings, ds, p1, p2, p3,
        family(p1, p2, p3, ds.counts["releases kept"], settings), deviations(settings),
    )
