"""The synthetic world: one binary market per release (docs/model.md sections 2 and 3).

For event ``e`` the world draws, with its own generator ``default_rng([seed, e])`` (so events share
no random state and raising ``n_events`` leaves the first events unchanged), in this order:

1. ``x0 ~ N(0, x0_sd^2)``, ``p0 = sigmoid(x0)``, the outcome ``y ~ Bernoulli(p0)``. The release is
   at ``T = t0 + e * event_spacing_hours * 3600`` seconds.
2. ``N ~ Poisson(Lambda(W))`` trades in the window of ``W`` hours before ``T``, where
   ``Lambda(tau) = integral_0^tau trade_rate(s) ds`` and ``trade_rate(tau) = base_rate (1 + surge
   exp(-tau / surge_hours))``. Their times to release are ``Lambda^{-1}`` of uniform draws on
   ``(0, Lambda(W)]`` (the inverse-cdf method), sorted so that the earliest trade comes first.
3. ``pi_i = informed_share(tau_i) = pi0 + pi1 exp(-tau_i / pi_hours)``, ``ell_i = ln((1 + pi_i) /
   (1 - pi_i))``, ``informed_i ~ Bernoulli(pi_i)``, taker side ``s_i = 2 y - 1`` if informed and a
   fair coin of +1 or -1 otherwise (Glosten and Milgrom 1985).
4. News: the public log-odds from the signal alone are ``x0`` plus the sum of independent
   increments over the intervals between trades, each ``N((2y - 1) sigma^2 dt / 2, sigma^2 dt)``
   with ``dt`` in hours (the first interval starts at the beginning of the window).
5. Order flow: ``x_pre_i = x_news_i + sum_{j < i} s_j ell_j`` and
   ``x_post_i = x_pre_i + s_i ell_i``.
   Because trade directions depend on ``y`` only, ``sigmoid(x_post_i)`` is the probability that
   ``y = 1`` given everything public: a martingale, and calibrated at every time.
6. The other makers quote ``ask = sigmoid(x_pre + kappa ell)`` rounded up and ``bid = sigmoid(x_pre
   - kappa ell)`` rounded down to the tick, both kept inside ``[tick, 1 - tick]`` and at least one
   tick apart. The print is at the ask for a buy and at the bid for a sell; sizes are geometric.

``kappa = 1`` without rounding is the zero-profit quote of Glosten and Milgrom: the print equals
``sigmoid(x_post)`` and the maker earns nothing in expectation. ``kappa > 1`` leaves the maker a
margin. The null world has ``pi1 = 0``.

The tape has the columns of the shared data contract plus the latent columns ``latent_mid_pre``,
``latent_mid_post``, ``latent_pi`` and ``latent_informed``; the events frame also carries
``latent_x0`` and the markets frame carries the outcome. Estimators, policies and the backtest must
not read columns that start with ``latent_``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from kalshi_mm.logit import information_step, round_to_tick, sigmoid

__all__ = ["World", "WorldParams", "generate_world", "informed_share", "trade_rate"]

SERIES = "SYN"
MAX_EVENTS = 999_999  # event ids are zero-padded to six digits so that string order is time order
_NEWTON_STEPS = 200
_NEWTON_TOL_HOURS = 1e-12


@dataclass(frozen=True)
class WorldParams:
    """Parameters of the synthetic world (docs/model.md section 11 explains each field)."""

    n_events: int = 200
    window_hours: float = 12.0  # trading window before the release
    sigma: float = 0.2  # logit news volatility per sqrt(hour)
    x0_sd: float = 1.0  # initial log-odds ~ N(0, x0_sd^2)
    base_rate: float = 10.0  # trades per hour far from the release
    surge: float = 20.0  # extra multiple of base_rate at the release
    surge_hours: float = 1.0  # decay time of the surge
    pi0: float = 0.05  # informed share far from the release
    pi1: float = 0.20  # extra informed share at the release (0 = null world)
    pi_hours: float = 0.5  # decay time of the extra informed share
    kappa: float = 1.5  # margin of the other makers on the log-odds spread
    tick: float = 0.01
    mean_size: float = 10.0  # mean contracts per trade (geometric, at least 1)
    event_spacing_hours: float = 168.0
    t0: float = 1_700_000_000.0  # release of event 0 is t0, event e at t0 + e * spacing

    def __post_init__(self) -> None:
        for name in ("window_hours", "base_rate", "surge_hours", "pi_hours", "kappa",
                     "event_spacing_hours"):
            _require(_finite(getattr(self, name)) and getattr(self, name) > 0.0,
                     f"{name} must be positive and finite")
        for name in ("sigma", "x0_sd", "surge", "pi0", "pi1"):
            _require(_finite(getattr(self, name)) and getattr(self, name) >= 0.0,
                     f"{name} must be non-negative and finite")
        _require(_finite(self.t0), "t0 must be finite")
        _require(isinstance(self.n_events, int | np.integer) and 1 <= self.n_events <= MAX_EVENTS,
                 f"n_events must be an integer between 1 and {MAX_EVENTS}")
        _require(self.pi0 + self.pi1 < 1.0, "pi0 + pi1 must be below 1, it is a share")
        _require(_finite(self.mean_size) and self.mean_size >= 1.0, "mean_size must be at least 1")
        _require(_finite(self.tick) and 0.0 < self.tick <= 0.25, "tick must lie in (0, 0.25]")
        per_dollar = 1.0 / self.tick
        _require(abs(per_dollar - round(per_dollar)) <= 1e-9 * per_dollar,
                 "1 / tick must be a whole number, so that the grid contains 1")


@dataclass(frozen=True, eq=False)
class World:
    """A generated world: its parameters and the three frames of the data contract."""

    params: WorldParams
    events: pd.DataFrame  # event, release_ts, series, latent_x0
    markets: pd.DataFrame  # market, event, outcome, open_ts, close_ts
    tape: pd.DataFrame  # event, market, ts, seq, price, size, side, latent_*


def _finite(x: Any) -> bool:
    return isinstance(x, int | float | np.integer | np.floating) and math.isfinite(x)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def informed_share(tau_hours: ArrayLike, params: WorldParams) -> NDArray[np.float64]:
    """Share of informed trades ``pi(tau) = pi0 + pi1 exp(-tau / pi_hours)`` at ``tau`` hours to
    the release (docs/model.md section 3)."""
    tau = np.asarray(tau_hours, dtype=float)
    return params.pi0 + params.pi1 * np.exp(-tau / params.pi_hours)


def trade_rate(tau_hours: ArrayLike, params: WorldParams) -> NDArray[np.float64]:
    """Trades per hour ``base_rate * (1 + surge * exp(-tau / surge_hours))`` at ``tau`` hours to
    the release."""
    tau = np.asarray(tau_hours, dtype=float)
    return params.base_rate * (1.0 + params.surge * np.exp(-tau / params.surge_hours))


def _cum_rate(tau_hours: ArrayLike, params: WorldParams) -> NDArray[np.float64]:
    """``Lambda(tau)``: the integral of :func:`trade_rate` from 0 to ``tau``, in closed form,
    ``base_rate (tau + surge * surge_hours * (1 - exp(-tau / surge_hours)))``."""
    tau = np.asarray(tau_hours, dtype=float)
    decay = -np.expm1(-tau / params.surge_hours)  # 1 - exp(-tau / surge_hours), accurate near 0
    return params.base_rate * (tau + params.surge * params.surge_hours * decay)


def _invert_cum_rate(y: NDArray[np.float64], params: WorldParams) -> NDArray[np.float64]:
    """Solve ``Lambda(tau) = y`` for ``tau >= 0``, elementwise, by Newton's method from ``tau = 0``.

    ``Lambda`` is increasing and concave (``trade_rate`` is positive and decreasing). For a concave
    increasing function the tangent lies above the graph, so a Newton step from a point left of the
    root lands left of the root or on it: the iterates increase monotonically to the root, and the
    method cannot diverge. It stops when every step is below 1e-12 hours.
    """
    tau = np.zeros_like(y)
    for _ in range(_NEWTON_STEPS):
        step = (y - _cum_rate(tau, params)) / trade_rate(tau, params)
        tau = tau + step
        if step.size == 0 or np.max(np.abs(step)) <= _NEWTON_TOL_HOURS:
            return tau
    raise RuntimeError("the inversion of the cumulative trade rate did not converge")


def _quotes(
    x_pre: NDArray[np.float64], ell: NDArray[np.float64], kappa: float, tick: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bid and ask of the other makers (step 6 of the module docstring).

    ``ask = sigmoid(x_pre + kappa ell)`` rounded up and ``bid = sigmoid(x_pre - kappa ell)``
    rounded down to the tick, both clipped to ``[tick, 1 - tick]``. If clipping leaves them less
    than a tick apart (they are then equal), the ask is moved up one tick or, at the upper edge, the
    bid down one tick. The comparison is done in whole ticks, so float noise cannot trigger it.
    """
    ask = np.clip(round_to_tick(sigmoid(x_pre + kappa * ell), tick, "up"), tick, 1.0 - tick)
    bid = np.clip(round_to_tick(sigmoid(x_pre - kappa * ell), tick, "down"), tick, 1.0 - tick)
    k_ask = np.round(ask / tick).astype(np.int64)
    k_bid = np.round(bid / tick).astype(np.int64)
    k_top = round(1.0 / tick) - 1  # the tick index of 1 - tick
    crossed = k_ask - k_bid < 1
    room_above = k_bid + 1 <= k_top
    k_ask = np.where(crossed & room_above, k_bid + 1, k_ask)
    k_bid = np.where(crossed & ~room_above, k_ask - 1, k_bid)
    return np.round(k_bid * tick, 10), np.round(k_ask * tick, 10)


def _release_ts(e: int, params: WorldParams) -> float:
    return params.t0 + e * params.event_spacing_hours * 3600.0


def _simulate_event(e: int, params: WorldParams, seed: int) -> dict[str, Any]:
    """All trades of event ``e`` as arrays, plus ``x0`` and the outcome ``y`` (module docstring)."""
    p = params
    rng = np.random.default_rng([seed, e])
    release = _release_ts(e, p)
    window = p.window_hours

    x0 = float(rng.normal(0.0, p.x0_sd))
    y = int(rng.random() < sigmoid(x0))

    total = float(_cum_rate(window, p))
    n = int(rng.poisson(total))
    u = 1.0 - rng.random(n)  # uniform on (0, 1]
    tau = np.minimum(_invert_cum_rate(u * total, p), window)  # hours to release, in (0, W]
    tau = np.sort(tau)[::-1].copy()  # earliest trade first
    # the float64 spacing of a Unix time is 2.4e-7 s: keep a trade a hair before the release
    ts = np.minimum(release - 3600.0 * tau, np.nextafter(release, -np.inf))

    pi = informed_share(tau, p)
    ell = information_step(pi)
    informed = rng.random(n) < pi
    coin = np.where(rng.random(n) < 0.5, 1, -1)
    side = np.where(informed, 2 * y - 1, coin).astype(np.int8)

    dt = -np.diff(np.concatenate(([window], tau)))  # hours since the previous trade or window start
    news = (2 * y - 1) * 0.5 * p.sigma**2 * dt + p.sigma * np.sqrt(dt) * rng.standard_normal(n)
    x_news = x0 + np.cumsum(news)
    flow = side * ell
    earlier_flow = np.concatenate(([0.0], np.cumsum(flow)))[:-1]  # sum over j < i
    x_pre = x_news + earlier_flow
    x_post = x_pre + flow

    bid, ask = _quotes(x_pre, ell, p.kappa, p.tick)
    return {
        "x0": x0,
        "y": y,
        "ts": ts,
        "price": np.where(side > 0, ask, bid),
        "size": rng.geometric(1.0 / p.mean_size, n).astype(float),
        "side": side,
        "latent_mid_pre": sigmoid(x_pre),
        "latent_mid_post": sigmoid(x_post),
        "latent_pi": pi,
        "latent_informed": informed,
    }


def generate_world(params: WorldParams, seed: int) -> World:
    """Generate ``params.n_events`` independent events (see the module docstring).

    The same ``(params, seed)`` gives identical frames; a different seed gives different ones.
    Event ids are ``SYN-000000``, ``SYN-000001``, ...; the market of an event has the same id.
    """
    _require(int(seed) == seed and seed >= 0, "seed must be a non-negative integer")
    seed = int(seed)
    n_events = int(params.n_events)
    pieces = [_simulate_event(e, params, seed) for e in range(n_events)]
    ids = np.array([f"{SERIES}-{e:06d}" for e in range(n_events)])
    counts = np.array([len(piece["ts"]) for piece in pieces], dtype=np.int64)

    def stack(key: str) -> NDArray[Any]:
        return np.concatenate([piece[key] for piece in pieces])

    tape = pd.DataFrame(
        {
            "event": np.repeat(ids, counts),
            "market": np.repeat(ids, counts),
            "ts": stack("ts").astype(np.float64),
            "seq": np.concatenate([np.arange(c, dtype=np.int64) for c in counts]),
            "price": stack("price").astype(np.float64),
            "size": stack("size").astype(np.float64),
            "side": stack("side").astype(np.int8),
            "latent_mid_pre": stack("latent_mid_pre").astype(np.float64),
            "latent_mid_post": stack("latent_mid_post").astype(np.float64),
            "latent_pi": stack("latent_pi").astype(np.float64),
            "latent_informed": stack("latent_informed").astype(bool),
        }
    )
    release = np.array([_release_ts(e, params) for e in range(n_events)], dtype=np.float64)
    events = pd.DataFrame(
        {
            "event": ids,
            "release_ts": release,
            "series": SERIES,
            "latent_x0": np.array([piece["x0"] for piece in pieces], dtype=np.float64),
        }
    )
    markets = pd.DataFrame(
        {
            "market": ids,
            "event": ids,
            "outcome": np.array([piece["y"] for piece in pieces], dtype=np.int64),
            "open_ts": release - 3600.0 * params.window_hours,
            "close_ts": release,
        }
    )
    return World(params=params, events=events, markets=markets, tape=tape)
