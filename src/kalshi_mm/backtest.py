"""Tape-only fill engine and the walk-forward backtest (docs/model.md sections 6 and 7).

The data are trades only, so a fill is decided from the prints alone. Decisions are taken at
``t_j = T - window + j * refresh`` for ``j = 0, 1, ...`` (``T`` is the release). The decision at
``t_j`` uses the prints at or before ``t_j``: the mid proxy ``m`` and the volatility, and the
inventory after every fill at or before ``t_j``. Its quotes are live on
``(t_j + latency, t_{j+1} + latency]`` and do not change in that interval; each side holds ``size``
contracts that fills consume and the next refresh restores.

* A taker sale (side -1) fills the bid if its price is strictly below the bid; a taker purchase
  (side +1) fills the ask if its price is strictly above the ask. With ``fill_rule = "touch"`` a
  print at the quote fills as well. Prints are compared with a tolerance of 1e-9 USD, so float
  noise cannot turn a print at the quote into a print through it.
* The fill is at the maker's own price for ``min(contracts left on that side, trade size)``, and
  the maker fee is ``fees.kalshi_fee(contracts, price, maker_fee_coeff)`` per fill.
* The tape does not react to the quotes. Prints at or after the release are ignored.
* ``cash`` is the money received on asks minus the money paid on bids, ``q`` the net inventory:
  ``gross = cash + q y``, ``pnl = gross - fees``, and ``marked_pnl = cash + q m_last - fees`` with
  ``m_last`` the last valid mid proxy before the release (the proxy at the release, from the prints
  before it, or the latest decision-time proxy if that one is stale).

Everything that does not depend on the maker's own fills (the mid proxy and the volatility at each
decision, the trades of each interval) is computed once per market by :func:`prepare_market` with
numpy; :func:`backtest_grid` then follows the inventory with scalar arithmetic. Nothing here reads
a column whose name starts with ``latent_``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from kalshi_mm.fees import kalshi_fee
from kalshi_mm.markout import BUCKET_LABELS, trade_markouts
from kalshi_mm.quoting import (
    POLICIES,
    PREMIUM_POLICIES,
    Policy,
    Premiums,
    QuoteParams,
    decision_minutes,
    estimate_premiums,
    premium_per_unit,
    quote_prices,
    realized_logit_vol,
)
from kalshi_mm.tape import mid_proxy_asof

__all__ = [
    "EventResult",
    "MarketGrid",
    "backtest_grid",
    "backtest_market",
    "mean_fill_premium",
    "prepare_market",
    "walk_forward",
]

_PRICE_TOL = 1e-9  # prices closer than this are the same price


@dataclass(frozen=True)
class EventResult:
    """Result of one policy on one market. USD, contracts.

    ``pnl`` is the settled net P&L ``cash + q y - fees``, ``gross`` is ``pnl + fees``, ``fills`` is
    the number of fills, ``contracts`` the contracts filled on both sides, ``inventory_end`` the
    net position at the release, ``marked_pnl`` the P&L with the final inventory valued at the last
    valid mid proxy instead of the outcome, ``quotes_live_share`` the share of refresh
    intervals in which at least one side was quoted, and ``mean_spread`` the mean of ask minus
    bid, in USD per contract, over the refresh intervals in which both sides were quoted (NaN when
    there was none). ``fill_premium`` is the premium that the fills carried: the mean over the
    contracts filled of :func:`kalshi_mm.quoting.premium_per_unit` of the refresh whose quote was
    hit, in USD per contract per unit of ``4 m (1 - m)`` (NaN when nothing was filled).
    """

    event: str
    market: str
    policy: str
    pnl: float
    gross: float
    fees: float
    fills: int
    contracts: float
    inventory_end: float
    marked_pnl: float
    quotes_live_share: float
    mean_spread: float
    fill_premium: float


def mean_fill_premium(rows: pd.DataFrame) -> float:
    """Mean of ``fill_premium`` over the contracts filled in ``rows``, or NaN with none.

    ``rows`` has the columns ``contracts`` and ``fill_premium`` of :func:`walk_forward` (markets) or
    of a table of events: the result is ``sum(fill_premium * contracts) / sum(contracts)`` over the
    rows with ``contracts > 0``, in USD per contract per unit of ``4 m (1 - m)``. A row without a
    fill has no ``fill_premium`` (NaN) and no weight.
    """
    contracts = rows["contracts"].to_numpy(dtype=np.float64)
    premium = rows["fill_premium"].to_numpy(dtype=np.float64)
    filled = contracts > 0.0
    total = float(contracts[filled].sum())
    if total <= 0.0:
        return math.nan
    return float((premium[filled] * contracts[filled]).sum() / total)


@dataclass(frozen=True)
class MarketGrid:
    """Everything of one market's backtest that does not depend on the maker's own fills.

    ``side``, ``price`` and ``size`` are the prints before the release. For each of the
    ``n_intervals`` decisions ``j``: ``mid[j]`` is the mid proxy and ``vol[j]`` the realized logit
    volatility from the prints at or before ``t_j`` (NaN where undefined), ``tau_min[j]`` the
    minutes to the release, and the prints ``first[j] : stop[j]`` are those in the live interval
    ``(t_j + latency, t_{j+1} + latency]``, of which ``first[j] : split[j]`` are at or before the
    next decision time ``t_{j+1}``. ``mark`` is the last valid mid proxy before the release.
    """

    release_ts: float
    side: NDArray[np.int8]
    price: NDArray[np.float64]
    size: NDArray[np.float64]
    mid: NDArray[np.float64]
    vol: NDArray[np.float64]
    tau_min: NDArray[np.float64]
    first: NDArray[np.int64]
    split: NDArray[np.int64]
    stop: NDArray[np.int64]
    mark: float

    @property
    def n_intervals(self) -> int:
        """Number of refresh intervals in the window."""
        return int(self.mid.size)


def prepare_market(
    ts: ArrayLike,
    side: ArrayLike,
    price: ArrayLike,
    size: ArrayLike,
    release_ts: float,
    params: QuoteParams,
) -> MarketGrid:
    """The fill-independent part of a market's backtest (see :class:`MarketGrid`).

    ``ts``, ``side``, ``price`` and ``size`` are the prints of one market sorted by time (equal
    times keep their order); prints at or after ``release_ts`` are dropped. The window has
    ``ceil(window_hours * 3600 / refresh_s)`` refresh intervals.
    """
    t = np.asarray(ts, dtype=np.float64)
    s = np.asarray(side)
    p = np.asarray(price, dtype=np.float64)
    c = np.asarray(size, dtype=np.float64)
    if t.ndim != 1 or not (t.shape == s.shape == p.shape == c.shape):
        raise ValueError("ts, side, price and size must be one-dimensional and of equal length")
    if np.any(np.isnan(t)) or np.any(np.diff(t) < 0.0):
        raise ValueError("ts must be sorted in increasing order and free of NaN")
    if not np.all((s == 1) | (s == -1)):
        raise ValueError("side must be +1 (taker bought) or -1 (taker sold)")
    if not np.all(np.isfinite(p)) or not (np.all(np.isfinite(c)) and np.all(c > 0.0)):
        raise ValueError("price must be finite and size positive and finite")
    release = float(release_ts)
    if not math.isfinite(release):
        raise ValueError("release_ts must be finite")
    n = int(np.searchsorted(t, release, side="left"))  # prints strictly before the release
    t, s, p, c = t[:n], s[:n].astype(np.int8), p[:n], c[:n]

    window = params.window_hours * 3600.0
    refresh = params.refresh_s
    n_intervals = max(1, math.ceil(window / refresh - 1e-9))
    steps = np.arange(n_intervals + 1, dtype=np.float64)
    decision_ts = (release - window) + refresh * steps  # t_0, ..., t_J
    mid = mid_proxy_asof(t, s, p, decision_ts[:-1], stale_s=params.stale_s)
    vol = realized_logit_vol(
        t, s, p, decision_ts[:-1], lookback_s=params.vol_lookback_s, stale_s=params.stale_s
    )
    live_edge = np.searchsorted(t, decision_ts + params.latency_s, side="right")
    known_edge = np.searchsorted(t, decision_ts, side="right")

    last_valid = float(mid_proxy_asof(t, s, p, release, stale_s=params.stale_s))
    if math.isnan(last_valid):  # stale at the release: the latest decision with a proxy
        defined = np.flatnonzero(~np.isnan(mid))
        last_valid = float(mid[defined[-1]]) if defined.size else math.nan
    return MarketGrid(
        release_ts=release,
        side=s,
        price=p,
        size=c,
        mid=np.asarray(mid, dtype=np.float64),
        vol=np.asarray(vol, dtype=np.float64),
        tau_min=decision_minutes(params),
        first=live_edge[:-1],
        split=known_edge[1:],
        stop=live_edge[1:],
        mark=last_valid,
    )


def backtest_grid(
    grid: MarketGrid,
    outcome: float,
    policy: Policy,
    premiums: Premiums,
    sigma_floor: float,
    params: QuoteParams,
    *,
    event: str = "",
    market: str = "",
) -> EventResult:
    """Run one policy over a prepared market (see the module docstring for the rules).

    The volatility of a decision is ``max(realized, sigma_floor)``, or ``sigma_floor`` when the
    realized volatility is undefined. The loop over the decisions only follows the inventory:
    ``q_known`` is the inventory after every fill at or before the decision time, which leaves out
    the fills of the previous interval that happen between the decision and the moment its own
    quotes go live.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    y = float(outcome)
    if y not in (0.0, 1.0):
        raise ValueError("outcome must be 0 or 1")
    if not sigma_floor >= 0.0:
        raise ValueError("sigma_floor must be non-negative")

    shift = _PRICE_TOL if params.fill_rule == "touch" else -_PRICE_TOL
    quote_size = params.size
    mids = grid.mid.tolist()
    taus = grid.tau_min.tolist()
    sigmas = [sigma_floor if v != v else max(v, sigma_floor) for v in grid.vol.tolist()]
    firsts, splits, stops = grid.first.tolist(), grid.split.tolist(), grid.stop.tolist()
    sides, prices, sizes = grid.side.tolist(), grid.price.tolist(), grid.size.tolist()

    inventory = 0.0
    cash = 0.0
    known_inventory = 0.0
    live = 0
    spread_sum = 0.0
    spread_n = 0
    fill_contracts: list[float] = []
    fill_prices: list[float] = []
    premium_filled = 0.0  # sum over the fills of contracts * premium per unit of 4 m (1 - m)
    for j in range(grid.n_intervals):
        quote = quote_prices(policy, mids[j], known_inventory, taus[j], sigmas[j], premiums, params)
        if quote is None:
            known_inventory = inventory
            continue
        bid, ask = quote
        live += 1
        unit = premium_per_unit(policy, taus[j], premiums, params)
        if bid == bid and ask == ask:  # both sides are quoted (a side at the limit is NaN)
            spread_sum += ask - bid
            spread_n += 1
        first, split, stop = firsts[j], splits[j], stops[j]
        if first == stop:
            known_inventory = inventory
            continue
        bid_left = ask_left = quote_size
        bid_below = bid + shift if bid == bid else -math.inf  # a print below this fills the bid
        ask_above = ask - shift if ask == ask else math.inf  # a print above this fills the ask
        for part, (lo, hi) in enumerate(((first, split), (split, stop))):
            for i in range(lo, hi):
                if sides[i] < 0:
                    if prices[i] < bid_below and bid_left > 0.0:
                        take = min(bid_left, sizes[i])
                        bid_left -= take
                        inventory += take
                        cash -= take * bid
                        fill_contracts.append(take)
                        fill_prices.append(bid)
                        premium_filled += take * unit
                elif prices[i] > ask_above and ask_left > 0.0:
                    take = min(ask_left, sizes[i])
                    ask_left -= take
                    inventory -= take
                    cash += take * ask
                    fill_contracts.append(take)
                    fill_prices.append(ask)
                    premium_filled += take * unit
            if part == 0:
                known_inventory = inventory

    fees = 0.0
    if fill_contracts:
        fees = float(
            kalshi_fee(np.asarray(fill_contracts), np.asarray(fill_prices), params.maker_fee_coeff)
            .sum()
        )
    gross = cash + inventory * y
    held_value = inventory * grid.mark if inventory != 0.0 else 0.0
    contracts = float(sum(fill_contracts))
    return EventResult(
        event=event,
        market=market,
        policy=policy,
        pnl=gross - fees,
        gross=gross,
        fees=fees,
        fills=len(fill_contracts),
        contracts=contracts,
        inventory_end=inventory,
        marked_pnl=cash + held_value - fees,
        quotes_live_share=live / grid.n_intervals,
        mean_spread=spread_sum / spread_n if spread_n else math.nan,
        fill_premium=premium_filled / contracts if contracts > 0.0 else math.nan,
    )


def backtest_market(
    ts: ArrayLike,
    side: ArrayLike,
    price: ArrayLike,
    size: ArrayLike,
    release_ts: float,
    outcome: float,
    policy: Policy,
    premiums: Premiums,
    sigma_floor: float,
    params: QuoteParams,
    *,
    event: str = "",
    market: str = "",
) -> EventResult:
    """One policy on one market: :func:`prepare_market` followed by :func:`backtest_grid`.

    ``ts``, ``side``, ``price`` and ``size`` are the prints of the market sorted by time,
    ``release_ts`` its release, ``outcome`` 0 or 1, ``premiums`` the premiums known at the start
    of the window and ``sigma_floor`` the floor of the logit volatility (model.md section 6).
    ``event`` and ``market`` only label the result.
    """
    grid = prepare_market(ts, side, price, size, release_ts, params)
    return backtest_grid(
        grid, outcome, policy, premiums, sigma_floor, params, event=event, market=market
    )


def _check_frames(tape: pd.DataFrame, markets: pd.DataFrame, events: pd.DataFrame) -> None:
    """The columns, the unique ids, the settled outcomes and the event of every market."""
    for name, frame, columns in (
        ("tape", tape, ("event", "market", "ts", "seq", "price", "size", "side")),
        ("markets", markets, ("market", "event", "outcome")),
        ("events", events, ("event", "release_ts", "series")),
    ):
        missing = [c for c in columns if c not in frame]
        if missing:
            raise ValueError(f"{name} lacks the columns {missing}")
    if events["event"].duplicated().any() or markets["market"].duplicated().any():
        raise ValueError("event ids and market ids must be unique")
    outcome = pd.to_numeric(markets["outcome"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all((outcome == 0.0) | (outcome == 1.0)):
        raise ValueError("every market needs a settled outcome of 0 or 1")
    if not markets["event"].isin(events["event"]).all():
        raise ValueError("a market belongs to an event that is not in events")
    release = pd.to_numeric(events["release_ts"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(release)):
        raise ValueError("every event needs a finite release_ts")
    if len(tape):
        expected = pd.Series(markets["event"].to_numpy(), index=pd.Index(markets["market"]))
        event_of_market = expected.reindex(pd.Index(tape["market"])).to_numpy()
        if pd.isna(event_of_market).any():
            raise ValueError("the tape has trades of a market that is not in markets")
        if not (event_of_market == tape["event"].to_numpy()).all():
            raise ValueError("the event of a trade differs from the event of its market")


def walk_forward(
    tape: pd.DataFrame,
    markets: pd.DataFrame,
    events: pd.DataFrame,
    policies: Iterable[Policy],
    params: QuoteParams,
    *,
    burn_in: int | None = None,
) -> pd.DataFrame:
    """Backtest every market of every event, in release order, with premiums known at the time.

    ``tape``, ``markets`` and ``events`` follow the data contract (``tape`` sorted by market, ts
    and seq; every market has an outcome of 0 or 1; one ``event`` id per release). For event ``e``:

    * the premiums are :func:`kalshi_mm.quoting.estimate_premiums` of the trade markouts
      (:func:`kalshi_mm.markout.trade_markouts`, horizon ``params.horizon_s``) of the events whose
      ``release_ts`` is at or before the start of ``e``'s window, ``release_ts - window``; the
      matched premium of ``flat_matched`` averages the bucket premiums over the decisions of the
      window (:func:`kalshi_mm.quoting.decision_minutes`);
    * ``sigma_floor`` is the mean of the defined :func:`kalshi_mm.quoting.realized_logit_vol` over
      the decision times of the windows of those same events (all their markets pooled), or
      ``sigma_default`` when there is no such value;
    * the premiums and the floor are the same for every market of the event and every policy.

    The events are ordered by ``(release_ts, event)``; the first ``burn_in`` of them (default
    ``params.min_history_events``) are run but have ``evaluated = False``. The result has one row
    per market and policy (policies in the order given) with the fields of :class:`EventResult`
    and ``evaluated``, ``release_ts`` and ``series``; the order of the rows does not depend on the
    order of the input frames. A market without a valid mid proxy in its window gets no fills.
    The net P&L of an event is the sum of ``pnl`` over its markets.
    """
    policies = tuple(policies)
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("policies must be a non-empty list without repeats")
    for policy in policies:
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    burn = params.min_history_events if burn_in is None else int(burn_in)
    if burn < 0:
        raise ValueError("burn_in must be non-negative")
    _check_frames(tape, markets, events)

    markouts = trade_markouts(
        tape, events, markets, horizon_s=params.horizon_s, stale_s=params.stale_s
    )
    ordered = (
        events.loc[events["event"].isin(markets["event"]), ["event", "release_ts", "series"]]
        .sort_values(["release_ts", "event"], kind="stable")
        .reset_index(drop=True)
    )
    n_events = len(ordered)
    release = ordered["release_ts"].to_numpy(dtype=np.float64)
    rank_of = pd.Series(np.arange(n_events), index=pd.Index(ordered["event"]))
    window_start = release - params.window_hours * 3600.0
    n_prior = np.searchsorted(release, window_start, side="right")  # events at or before it

    # the trades of each market (the tape is sorted by market, ts, seq)
    codes, labels = pd.factorize(tape["market"], sort=True)
    segment: dict[str, tuple[int, int]] = {}
    if len(tape):
        starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
        for a, b in zip(starts, np.r_[starts[1:], len(tape)], strict=True):
            segment[str(labels[codes[a]])] = (int(a), int(b))
    ts_all = tape["ts"].to_numpy(dtype=np.float64)
    side_all = tape["side"].to_numpy()
    price_all = tape["price"].to_numpy(dtype=np.float64)
    size_all = tape["size"].to_numpy(dtype=np.float64)

    market_rank = rank_of.reindex(pd.Index(markets["event"])).to_numpy().astype(np.int64)
    by_rank = markets.assign(rank=market_rank).sort_values(["rank", "market"], kind="stable")
    grids: list[tuple[int, str, MarketGrid, float]] = []
    vol_sum = np.zeros(n_events)
    vol_count = np.zeros(n_events, dtype=np.int64)
    empty = (0, 0)
    for rank, market, outcome in zip(
        by_rank["rank"].to_numpy(), by_rank["market"], by_rank["outcome"].to_numpy(), strict=True
    ):
        a, b = segment.get(str(market), empty)
        grid = prepare_market(
            ts_all[a:b], side_all[a:b], price_all[a:b], size_all[a:b], release[rank], params
        )
        defined = grid.vol[~np.isnan(grid.vol)]
        vol_sum[rank] += defined.sum()
        vol_count[rank] += defined.size
        grids.append((int(rank), str(market), grid, float(outcome)))
    vol_sum_before = np.concatenate(([0.0], np.cumsum(vol_sum)))  # over the first r events
    vol_count_before = np.concatenate(([0], np.cumsum(vol_count)))

    # markout rows grouped by the release rank of their event
    row_rank = rank_of.reindex(pd.Index(markouts["event"])).to_numpy()
    if np.isnan(row_rank).any():
        raise ValueError("a trade belongs to an event that has no market")
    row_rank = row_rank.astype(np.int64)
    needed = ["event", "bucket", "mid_before", "adverse_selection"]
    sorted_markouts = markouts[needed].iloc[np.argsort(row_rank, kind="stable")]
    rows_before = np.concatenate(([0], np.cumsum(np.bincount(row_rank, minlength=n_events))))

    needs_premiums = any(policy in PREMIUM_POLICIES for policy in policies)
    minutes = decision_minutes(params)
    no_premiums = Premiums(0.0, (0.0,) * len(BUCKET_LABELS), 0, 0.0)
    rows: list[dict] = []
    current = -1
    premiums, sigma_floor = no_premiums, params.sigma_default
    for rank, market, grid, outcome in grids:
        if rank != current:  # first market of the next event
            current = rank
            earlier = int(n_prior[rank])
            if needs_premiums:
                premiums = estimate_premiums(
                    sorted_markouts.iloc[: rows_before[earlier]],
                    min_events=params.min_history_events,
                    minutes=minutes,
                )
            has_vol = vol_count_before[earlier] > 0
            sigma_floor = (
                float(vol_sum_before[earlier] / vol_count_before[earlier])
                if has_vol else params.sigma_default
            )
        event = str(ordered["event"].iat[rank])
        for policy in policies:
            result = backtest_grid(
                grid, outcome, policy, premiums, sigma_floor, params, event=event, market=market
            )
            rows.append(
                {
                    **asdict(result),
                    "evaluated": rank >= burn,
                    "release_ts": float(release[rank]),
                    "series": ordered["series"].iat[rank],
                }
            )
    columns = [
        "event", "market", "policy", "pnl", "gross", "fees", "fills", "contracts",
        "inventory_end", "marked_pnl", "quotes_live_share", "mean_spread", "fill_premium",
        "evaluated", "release_ts", "series",
    ]
    return pd.DataFrame(rows, columns=columns)
