#!/usr/bin/env python3
"""Mutation check: does the test suite notice deliberate bugs?

Copies src/, tests/, scripts/, docs/ and pyproject.toml into a temporary directory, then applies
ONE deliberate bug ("mutant") at a time to the copy and runs pytest there. A mutant is "caught" if
at least one test fails. The working tree is never modified. The mutants are bugs that would
silently produce plausible numbers: a sign, an off-by-one, a wrong tick direction, look-ahead, a
wrong horizon or a unit error. To save time a mutant runs only the test files that exercise the
changed module (listed with it); the baseline runs the whole suite. Exits with status 1 if a
mutant survives or the baseline fails.

Usage: python3 tests/mutation_check.py [--out results/mutation_check.txt]
"""

from __future__ import annotations

import argparse
import datetime
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COPY = ["src", "tests", "pyproject.toml", "docs", "scripts"]
S = "src/kalshi_mm/"

# (name, test files to run, [(file, text that must occur exactly once, replacement), ...])
MUTANTS: list[tuple[str, list[str], list[tuple[str, str, str]]]] = []


def mutant(name: str, tests: str, path: str, old: str, new: str) -> None:
    """Register a mutant that replaces ``old`` by ``new`` in ``src/kalshi_mm/<path>``."""
    MUTANTS.append((name, tests.split(), [(S + path, old, new)]))


def script_mutant(name: str, tests: str, old: str, new: str) -> None:
    """Register a mutant that replaces ``old`` by ``new`` in scripts/fetch_kalshi.py."""
    MUTANTS.append((name, tests.split(), [("scripts/fetch_kalshi.py", old, new)]))


def file_mutant(name: str, tests: str, path: str, old: str, new: str) -> None:
    """Register a mutant that replaces ``old`` by ``new`` in the repository file ``path``."""
    MUTANTS.append((name, tests.split(), [(path, old, new)]))


# ---- mid proxy of the tape (tape.py)
mutant("tape: inclusive and exclusive query swapped", "tests/test_tape.py", "tape.py",
       'side="right" if inclusive else "left"', 'side="left" if inclusive else "right"')
mutant("tape: a print exactly stale_s old counts as stale", "tests/test_tape.py", "tape.py",
       "q - t_side[k] <= stale_s", "q - t_side[k] < stale_s")
mutant("tape: no staleness limit", "tests/test_tape.py", "tape.py",
       "fresh = found & (q - t_side[k] <= stale_s)", "fresh = found")

# ---- markouts and the release effect (markout.py)
mutant("markout: the trade sees its own print in mid_before", "tests/test_markout.py",
       "markout.py", "p, t, stale_s=stale_s, inclusive=False)",
       "p, t, stale_s=stale_s, inclusive=True)")
mutant("markout: horizon not cut at the release", "tests/test_markout.py", "markout.py",
       "np.minimum(t + horizon_s, T)", "t + horizon_s")
mutant("markout: a print at the release is used", "tests/test_markout.py", "markout.py",
       "rows = np.arange(a, b)[ts[a:b] < T]", "rows = np.arange(a, b)[ts[a:b] <= T]")
mutant("markout: a trade at the release is kept", "tests/test_markout.py", "markout.py",
       "keep = tau_min > 0.0", "keep = tau_min >= 0.0")
mutant("markout: adverse selection has the wrong sign", "tests/test_markout.py", "markout.py",
       "adverse_selection=sgn * (after - before),", "adverse_selection=sgn * (before - after),")
mutant("markout: realized spread uses mid_before", "tests/test_markout.py", "markout.py",
       "realized_spread=sgn * (price[keep] - after),",
       "realized_spread=sgn * (price[keep] - before),")
mutant("markout: minutes to release computed as hours", "tests/test_markout.py", "markout.py",
       "tau_min = (release - ts) / 60.0", "tau_min = (release - ts) / 3600.0")
mutant("markout: bucket edges closed on the left", "tests/test_markout.py", "markout.py",
       'tau, side="left") - 1', 'tau, side="right") - 1')
mutant("markout: the quiet window includes its lower edge", "tests/test_markout.py", "markout.py",
       "(tau > quiet[0]) & (tau <= quiet[1])", "(tau >= quiet[0]) & (tau <= quiet[1])")
mutant("markout: D is quiet minus near", "tests/test_markout.py", "markout.py",
       "diff = near_b - quiet_b", "diff = quiet_b - near_b")
mutant("markout: bucket standard error without the G/(G-1) factor", "tests/test_markout.py",
       "markout.py", "np.sqrt(g / (g - 1.0) * np.sum(residual**2))", "np.sqrt(np.sum(residual**2))")
mutant("markout: scaled markout without the factor 4", "tests/test_markout.py", "markout.py",
       "np.divide(values, 4.0 * m * (1.0 - m), out=out, where=inside)",
       "np.divide(values, m * (1.0 - m), out=out, where=inside)")

# ---- synthetic world (world.py)
mutant("world: the price before a trade already includes the trade", "tests/test_world.py",
       "world.py", "x_pre = x_news + earlier_flow\n", "x_pre = x_news + earlier_flow + flow\n")
mutant("world: news drift without the factor one half", "tests/test_world.py", "world.py",
       "(2 * y - 1) * 0.5 * p.sigma**2 * dt", "(2 * y - 1) * p.sigma**2 * dt")
mutant("world: price step equals the informed share", "tests/test_world.py", "world.py",
       "ell = information_step(pi)", "ell = pi")
mutant("world: ask rounded to the nearest tick", "tests/test_world.py", "world.py",
       'sigmoid(x_pre + kappa * ell), tick, "up")',
       'sigmoid(x_pre + kappa * ell), tick, "nearest")')
mutant("world: hours to release used as minutes", "tests/test_world.py", "world.py",
       "release - 3600.0 * tau", "release - 60.0 * tau")
mutant("world: outcome not calibrated to the initial belief", "tests/test_world.py", "world.py",
       "y = int(rng.random() < sigmoid(x0))", "y = int(rng.random() < 0.5)")
mutant("world: events share one random stream", "tests/test_world.py", "world.py",
       "rng = np.random.default_rng([seed, e])", "rng = np.random.default_rng(seed)")

# ---- log-odds helpers, fees, statistics, manifests
mutant("logit: rounding up uses floor", "tests/test_logit.py", "logit.py",
       "n = np.ceil(n)", "n = np.floor(n)")
mutant("logit: no snapping to the grid before rounding", "tests/test_logit.py", "logit.py",
       "n = np.where(np.abs(n - grid) <= tol, grid, n)", "n = n")
mutant("logit: martingale drift without the factor one half", "tests/test_logit.py", "logit.py",
       "0.5 * np.asarray(sigma, dtype=float) ** 2 * np.tanh(",
       "np.asarray(sigma, dtype=float) ** 2 * np.tanh(")
mutant("fees: no rounding to 9 decimals before the ceiling", "tests/test_fees.py", "fees.py",
       "units = np.round(raw * _UNITS_PER_USD, 5)", "units = raw * _UNITS_PER_USD")
mutant("fees: floor instead of ceiling", "tests/test_fees.py", "fees.py",
       "return np.ceil(units) / _UNITS_PER_USD", "return np.floor(units) / _UNITS_PER_USD")
mutant("fees: the maker pays the taker coefficient", "tests/test_fees.py", "fees.py",
       "return kalshi_fee(count, price, MAKER_COEFF, multiplier)",
       "return kalshi_fee(count, price, TAKER_COEFF, multiplier)")
mutant("stats: Holm multiplier off by one", "tests/test_stats.py", "stats.py",
       "(m - np.arange(m)) * p[order]", "(m - np.arange(m) + 1) * p[order]")
mutant("stats: p-value counts strictly below zero", "tests/test_stats.py", "stats.py",
       "np.mean(x <= 0.0)", "np.mean(x < 0.0)")
mutant("stats: p-value without the floor 1/(B+1)", "tests/test_stats.py", "stats.py",
       "float(max(np.mean(x <= 0.0), 1.0 / (x.size + 1.0)))", "float(np.mean(x <= 0.0))")
mutant("stats: ratio of sums with a mean denominator", "tests/test_stats.py", "stats.py",
       "bottom = w @ den", "bottom = w.sum(axis=1) * den.mean()")
mutant("manifest: dirty and clean swapped", "tests/test_common.py", "experiments/common.py",
       'return commit, "dirty" if changed else "clean"',
       'return commit, "clean" if changed else "dirty"')

# ---- recalibration and Brier skill (calibration.py)
CAL = "tests/test_calibration.py"
mutant("calibration: sign of the intercept step", CAL, "calibration.py",
       "da = g0 / h00 - mu * db", "da = g0 / h00 + mu * db")
mutant("calibration: direction of the slope step", CAL, "calibration.py",
       "db = (g1 - mu * g0) / s2", "db = -(g1 - mu * g0) / s2")
mutant("calibration: no clip at the top of the price", CAL, "calibration.py",
       "return logit(np.clip(price, clip, 1.0 - clip))",
       "return logit(np.clip(price, clip, 1.0))")
mutant("calibration: the fit uses the unclipped price", CAL, "calibration.py",
       "x = _clipped_logit(p, clip)", "x = logit(p)")
mutant("calibration: expanding window starts one release late", CAL, "calibration.py",
       "for k in range(min_events, k_total, refit_every):",
       "for k in range(min_events + 1, k_total, refit_every):")
mutant("calibration: the release itself is in its fit", CAL, "calibration.py",
       "a, b = fit_recalibration(p[:start], y[:start])",
       "a, b = fit_recalibration(p[:stop], y[:stop])")
mutant("calibration: one row of the release in its fit", CAL, "calibration.py",
       "a, b = fit_recalibration(p[:start], y[:start])",
       "a, b = fit_recalibration(p[: start + 1], y[: start + 1])")
mutant("calibration: refit block one release short", CAL, "calibration.py",
       "stop = int(first[k + refit_every]) if k + refit_every < k_total else n",
       "stop = int(first[k + refit_every - 1]) if k + refit_every < k_total else n")
mutant("calibration: observations are sorted by price before time", CAL, "calibration.py",
       "keys = [y, p, ts]", "keys = [y, ts, p]")
mutant("calibration: the bootstrap resamples rows, not clusters", CAL, "calibration.py",
       "codes, n_clusters = _cluster_codes(clusters, price.size)",
       "codes, n_clusters = np.arange(price.size), price.size")
mutant("calibration: one-sided p in the wrong direction", CAL, "calibration.py",
       '"p_one_sided": p_at_or_below_zero(boot),', '"p_one_sided": p_at_or_below_zero(-boot),')
mutant("calibration: skill defined the other way round", CAL, "calibration.py",
       '"bss": 1.0 - brier_recal / brier_raw,', '"bss": brier_raw / brier_recal - 1.0,')
mutant("calibration: slope sign in the recalibration", CAL, "calibration.py",
       "return sigmoid(a + b * _clipped_logit(p, clip))",
       "return sigmoid(a - b * _clipped_logit(p, clip))")
mutant("calibration: a fit that does not converge returns its last iterate", CAL,
       "calibration.py", "ll_new\n    raise not_converged\n", "ll_new\n    return a, b\n")
mutant("calibration: cluster labels are not sorted", CAL, "calibration.py",
       "codes, uniques = pd.factorize(arr, sort=True)",
       "codes, uniques = pd.factorize(arr, sort=False)")


# ---- the protocol analysis, its report and the runner (protocol.py and the files that use it)
PROTO, PIPE = "tests/test_protocol.py", "tests/test_pipeline.py"
RUN = "tests/test_run_real.py tests/test_pipeline.py"
mutant("protocol: events with one release time are joined in input order", PROTO, "protocol.py",
       'ids = sorted(group["event"])', 'ids = list(group["event"])')
mutant("protocol: the series of a merged release is repeated", PROTO, "protocol.py",
       '"+".join(sorted(set(group["series"])))', '"+".join(sorted(group["series"]))')
mutant("protocol: trades at the release are kept", PROTO, "protocol.py",
       'tape = tape[tape["ts"] < tape["event"].map(release_ts)]',
       'tape = tape[tape["ts"] <= tape["event"].map(release_ts)]')
mutant("protocol: a market with exactly the minimum number of trades is dropped", PROTO,
       "protocol.py", 'enough = markets["market"].map(n_trades).fillna(0) >= settings.min_trades',
       'enough = markets["market"].map(n_trades).fillna(0) > settings.min_trades')
mutant("protocol: markets without an outcome are kept", PROTO, "protocol.py",
       'keep = markets["outcome"].notna()', 'keep = pd.Series(True, index=markets.index)')
mutant("protocol: the quoting window is in minutes", PROTO, "protocol.py",
       "window_s = 3600.0 * settings.quote.window_hours",
       "window_s = 60.0 * settings.quote.window_hours")
mutant("protocol: a market with no proxy in the quoting window is kept", PROTO, "protocol.py",
       "if np.any(~np.isnan(proxy)):", "if True:")
mutant("protocol: the recalibration price is taken minutes, not hours, ahead", PROTO,
       "protocol.py", "query = float(release_ts[event]) - 3600.0 * settings.recal_hours",
       "query = float(release_ts[event]) - 60.0 * settings.recal_hours")
mutant("protocol: P3 runs with exactly the minimum number of releases", PROTO, "protocol.py",
       'if usable["release_ts"].nunique() <= settings.recal_min_releases:',
       'if usable["release_ts"].nunique() < settings.recal_min_releases:')
mutant("protocol: the bootstrap seed is ignored", PROTO, "protocol.py",
       "np.random.default_rng(settings.seed)", "np.random.default_rng(0)")
mutant("protocol: exploratory only below the minimum minus one", PROTO, "protocol.py",
       "exploratory = n_releases < settings.min_releases",
       "exploratory = n_releases <= settings.min_releases")
mutant("protocol: supported at an adjusted p equal to alpha", PROTO, "protocol.py",
       'else ("supported" if adj < settings.alpha else "not supported")',
       'else ("supported" if adj <= settings.alpha else "not supported")')
mutant("protocol: P3 is labelled exploratory too", PROTO, "protocol.py",
       'exploratory and name != "P3"', "exploratory")
mutant("protocol: a test that is not computed counts as p = 0", PROTO, "protocol.py",
       "[1.0 if v is None else v for v in raw.values()]",
       "[0.0 if v is None else v for v in raw.values()]")
mutant("protocol: a setting from the command line becomes a float", PROTO, "protocol.py",
       "value = type(current)(text)", "value = float(text)")
mutant("protocol: the labels of the two sources are swapped", PROTO, "protocol.py",
       'SOURCE_LABELS = {"synthetic": "synthetic data", "kalshi-public-api": "real data"}',
       'SOURCE_LABELS = {"synthetic": "real data", "kalshi-public-api": "synthetic data"}')
mutant("protocol: the releases of the burn-in enter the comparison", PIPE, "protocol.py",
       'evaluated = frame[frame["evaluated"]]', "evaluated = frame")
mutant("protocol: the P&L of a release is the mean over its markets", PIPE, "protocol.py",
       '.groupby(["event", "policy"])[column].sum()',
       '.groupby(["event", "policy"])[column].mean()')
mutant("protocol: the primary pair is reversed", PIPE, "protocol.py",
       'PRIMARY_PAIR = ("release_aware", "pooled_premium")',
       'PRIMARY_PAIR = ("pooled_premium", "release_aware")')
mutant("protocol: the halves are split at the first release", PIPE, "protocol.py",
       "early = when <= when.median()", "early = when < when.median()")
mutant("protocol: the uncut window starts at the horizon in hours", PIPE, "protocol.py",
       "near=(settings.horizon_s / 60.0, settings.near[1])",
       "near=(settings.horizon_s / 3600.0, settings.near[1])")
mutant("protocol: the scaled effect uses every mid proxy", PIPE, "protocol.py",
       "markouts, min_mid=q.min_mid, max_mid=q.max_mid", "markouts, min_mid=0.0, max_mid=1.0")
mutant("report: the table letters are swapped", PIPE, "protocol_report.py",
       'TABLE_LETTER = {"synthetic data": "R", "real data": "K"}',
       'TABLE_LETTER = {"synthetic data": "K", "real data": "R"}')
mutant("report: synthetic results may go to a directory named real", PIPE, "protocol_report.py",
       'if label == "synthetic data" and "real" in folders:', "if False:")
mutant("report: real results may go to a directory named rehearsal", PIPE, "protocol_report.py",
       'if label == "real data" and "rehearsal" in folders:', "if False:")
mutant("report: departures from the protocol are not written", PIPE, "protocol_report.py",
       '    if deviations:\n        (out_dir / "DEVIATIONS.md")',
       '    if False:\n        (out_dir / "DEVIATIONS.md")')
mutant("run_real: the database is opened for writing", PIPE, "run_real.py",
       "?mode=ro", "?mode=rw")
mutant("run_real: a synthetic database is accepted", PIPE, "run_real.py",
       'if data_label(source) != "real data":', "if False:")
mutant("run_real: the tag is not checked", PIPE, "run_real.py",
       "if not args.no_tag_check:", "if False:")
mutant("run_real: files that are not tracked are ignored by the tag check", RUN, "run_real.py",
       "files = sorted(set(changed) | set(untracked))", "files = sorted(set(changed))")


# ---- the markout panel computed in SQL (panel.py)
PANEL = "tests/test_panel.py"
mutant("panel: effective half-spread without the side sign", PANEL, "panel.py",
       "side * (price - mid_before) AS eff_half_spread", "(price - mid_before) AS eff_half_spread")
mutant("panel: adverse selection has the wrong sign", PANEL, "panel.py",
       "side * (mid_after - mid_before) AS adverse_selection",
       "side * (mid_before - mid_after) AS adverse_selection")
mutant("panel: realized spread uses mid_before", PANEL, "panel.py",
       "side * (price - mid_after) AS realized_spread",
       "side * (price - mid_before) AS realized_spread")
mutant("panel: settlement markout has the wrong sign", PANEL, "panel.py",
       "side * (price - outcome) AS settlement_markout",
       "side * (outcome - price) AS settlement_markout")
mutant("panel: mid_before sees a buy print at the trade's own time", PANEL, "panel.py",
       "s.side = 1 AND s.ts < x.ts", "s.side = 1 AND s.ts <= x.ts")
mutant("panel: mid_before sees a sell print at the trade's own time", PANEL, "panel.py",
       "s.side = -1 AND s.ts < x.ts", "s.side = -1 AND s.ts <= x.ts")
mutant("panel: a buy print exactly at after_ts is not used", PANEL, "panel.py",
       "s.side = 1\n               AND s.ts <= x.after_ts",
       "s.side = 1\n               AND s.ts < x.after_ts")
mutant("panel: a sell print exactly at after_ts is not used", PANEL, "panel.py",
       "s.side = -1\n               AND s.ts <= x.after_ts",
       "s.side = -1\n               AND s.ts < x.after_ts")
mutant("panel: mid_after may use a buy print at the release", PANEL, "panel.py",
       "AND s.ts <= x.after_ts AND s.ts < x.release_ts\n"
       "             ORDER BY s.ts DESC, s.seq DESC LIMIT 1) AS buy_after",
       "AND s.ts <= x.after_ts AND s.ts <= x.release_ts\n"
       "             ORDER BY s.ts DESC, s.seq DESC LIMIT 1) AS buy_after")
mutant("panel: mid_after may use a sell print at the release", PANEL, "panel.py",
       "AND s.ts <= x.after_ts AND s.ts < x.release_ts\n"
       "             ORDER BY s.ts DESC, s.seq DESC LIMIT 1) AS sell_after",
       "AND s.ts <= x.after_ts AND s.ts <= x.release_ts\n"
       "             ORDER BY s.ts DESC, s.seq DESC LIMIT 1) AS sell_after")
mutant("panel: a buy print exactly stale_s old is stale at the trade", PANEL, "panel.py",
       "CASE WHEN x.ts - s.ts <= :stale_s THEN s.price END\n"
       "              FROM trades AS s\n"
       "             WHERE s.market = x.market AND s.side = 1 AND",
       "CASE WHEN x.ts - s.ts < :stale_s THEN s.price END\n"
       "              FROM trades AS s\n"
       "             WHERE s.market = x.market AND s.side = 1 AND")
mutant("panel: a sell print exactly stale_s old is stale at after_ts", PANEL, "panel.py",
       "CASE WHEN x.after_ts - s.ts <= :stale_s THEN s.price END\n"
       "              FROM trades AS s\n"
       "             WHERE s.market = x.market AND s.side = -1\n",
       "CASE WHEN x.after_ts - s.ts < :stale_s THEN s.price END\n"
       "              FROM trades AS s\n"
       "             WHERE s.market = x.market AND s.side = -1\n")
mutant("panel: the horizon is not cut at the release", PANEL, "panel.py",
       "MIN(ts + :horizon_s, release_ts)", "(ts + :horizon_s)")
mutant("panel: a trade at the release is kept", PANEL, "panel.py",
       "WHERE ts < release_ts", "WHERE ts <= release_ts")
mutant("panel: minutes to the release computed as hours", PANEL, "panel.py",
       "(release_ts - ts) / 60.0 AS tau_min", "(release_ts - ts) / 3600.0 AS tau_min")
mutant("panel: the 30 minute bucket edge is open", PANEL, "panel.py",
       "WHEN tau_min <= 30 THEN 2", "WHEN tau_min < 30 THEN 2")
mutant("panel: the earlier of two prints at one time wins", PANEL, "panel.py",
       "ORDER BY s.ts DESC, s.seq DESC LIMIT 1) AS buy_before",
       "ORDER BY s.ts DESC, s.seq ASC LIMIT 1) AS buy_before")
mutant("panel: the mid is the last buy print alone", PANEL, "panel.py",
       "0.5 * (buy_before + sell_before)", "buy_before")
mutant("panel: the index is not the row of the whole tape", PANEL, "panel.py",
       "ROW_NUMBER() OVER (ORDER BY t.market, t.ts, t.seq) - 1",
       "ROW_NUMBER() OVER (ORDER BY t.market, t.ts, t.seq)")

# ---- records from the API, the protocol filters and the fetch loop (ingest.py)
ING = "tests/test_ingest.py"
mutant("ingest: the taker side is inverted", ING, "ingest.py",
       '    if taker == "yes":\n        side = 1\n    elif taker == "no":\n        side = -1\n',
       '    if taker == "yes":\n        side = -1\n    elif taker == "no":\n        side = 1\n')
mutant("ingest: block trades are kept", ING, "ingest.py",
       '        if trade["is_block_trade"]:', "        if False:")
mutant("ingest: a trade at the release is kept", ING, "ingest.py",
       'elif trade["ts"] >= release_ts:', 'elif trade["ts"] > release_ts:')
mutant("ingest: a market with exactly the minimum number of trades is dropped", ING, "ingest.py",
       "if len(trades) < min_trades:", "if len(trades) <= min_trades:")
mutant("ingest: a repeated trade_id is kept", ING, "ingest.py",
       '            if trade["trade_id"] in seen:', "            if False:")
mutant("ingest: a price of exactly 1 is accepted", ING, "ingest.py",
       "if not 0.0 < price < 1.0:", "if not 0.0 < price <= 1.0:")
mutant("ingest: yes and no results are swapped", ING, "ingest.py",
       '"outcome": {"yes": 1, "no": 0}.get(result),', '"outcome": {"yes": 0, "no": 1}.get(result),')
mutant("ingest: a release at the start of the window is outside it", ING, "ingest.py",
       "if not window_start <= release < window_end:",
       "if not window_start < release < window_end:")
mutant("ingest: a release at the end of the window is inside it", ING, "ingest.py",
       "if not window_start <= release < window_end:",
       "if not window_start <= release <= window_end:")
mutant("ingest: a release exactly at the cutoff is read from both endpoints", ING, "ingest.py",
       "if release_ts <= cutoff_ts:", "if release_ts < cutoff_ts:")
mutant("ingest: the release bound of a query is rounded down", ING, "ingest.py",
       "last = math.ceil(release_ts)", "last = math.floor(release_ts)")
mutant("ingest: the historical range ends at the rounded-down cutoff", ING, "ingest.py",
       'TradeQuery("historical", None, math.ceil(cutoff_ts)),',
       'TradeQuery("historical", None, math.floor(cutoff_ts)),')
mutant("ingest: the live range starts at the cutoff, not a second before", ING, "ingest.py",
       'TradeQuery("live", math.floor(cutoff_ts) - 1, last),',
       'TradeQuery("live", math.floor(cutoff_ts), last),')
mutant("ingest: the archived record wins whenever it is settled", ING, "ingest.py",
       "elif ticker not in chosen or (_is_settled(record) and not _is_settled(chosen[ticker])):",
       "elif ticker not in chosen or _is_settled(record):")
mutant("ingest: a stored market is fetched again", ING, "ingest.py",
       "if isinstance(ticker, str) and ticker in loaded:", "if False:")
mutant("ingest: a calendar that moved a stored event is accepted", ING, "ingest.py",
       "release_of[e] != ts)", "release_of[e] == ts)")
mutant("ingest: the cutoff is not read again at the end of a run", ING, "ingest.py",
       "cutoff_after=client.historical_cutoff(),", "cutoff_after=cutoff,")
mutant("ingest: a market with too few trades is written", ING, "ingest.py",
       "        if len(markets):", "        if True:")

# ---- the HTTP client (kalshi_api.py)
API = "tests/test_kalshi_api.py"
mutant("api: an empty cursor does not end the paging", API, "kalshi_api.py",
       "if not cursor:", "if cursor is None:")
mutant("api: a repeated cursor is followed", API, "kalshi_api.py",
       "if cursor in seen:", "if False:")
mutant("api: HTTP 429 is not retried", API, "kalshi_api.py",
       "if status != 429 and status < 500:", "if status < 500:")
mutant("api: Retry-After is ignored", API, "kalshi_api.py",
       "return backoff if retry_after is None else max(backoff, retry_after)", "return backoff")
mutant("api: one retry too few", API, "kalshi_api.py",
       "for attempt in range(self.max_retries + 1):", "for attempt in range(self.max_retries):")
mutant("api: the backoff ceiling is a floor", API, "kalshi_api.py",
       "ceiling = min(self.max_delay_s, self.base_delay_s * 2.0**attempt)",
       "ceiling = max(self.max_delay_s, self.base_delay_s * 2.0**attempt)")

# ---- the command line (scripts/fetch_kalshi.py), which the copy of the repository includes
script_mutant("fetch_kalshi: an API failure exits with status 0", ING,
              'again to resume.")\n        return 1\n', 'again to resume.")\n        return 0\n')
script_mutant("fetch_kalshi: a calendar error exits with status 1", ING,
              'out(f"calendar: {error}")\n            return 2',
              'out(f"calendar: {error}")\n            return 1')
script_mutant("fetch_kalshi: the agreement notice is not printed", ING,
              "    out(NOTICE)\n", "    pass\n")
script_mutant("fetch_kalshi: a dry run fetches", ING,
              "        if args.dry_run:\n            return _dry_run(",
              "        if False:\n            return _dry_run(")
script_mutant("fetch_kalshi: a draft calendar may have any name", ING,
              'and "draft" not in args.calendar_draft.name.lower():',
              "and False:")


# ---- the recovery study (experiments/recovery.py)
REC = "tests/test_recovery.py"
mutant("recovery: oracle truncation pre", REC, "experiments/recovery.py",
       ', post[b - 1])',
       ', pre[b - 1])')
mutant("recovery: oracle first trade after right", REC, "experiments/recovery.py",
       'ts[a:b] + horizon_s, side="left")',
       'ts[a:b] + horizon_s, side="right")')
mutant("recovery: oracle horizon ignored", REC, "experiments/recovery.py",
       'ts[a:b] + horizon_s, side',
       'ts[a:b] + HORIZON_S, side')
mutant("recovery: oracle side dropped", REC, "experiments/recovery.py",
       'out[a:b] = side[a:b] * (after - pre[a:b])',
       'out[a:b] = after - pre[a:b]')
mutant("recovery: oracle one market", REC, "experiments/recovery.py",
       'starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])',
       'starts = np.array([0])')
mutant("recovery: impact sign", REC, "experiments/recovery.py",
       'return side * (tape["latent_mid_post"].to_numpy() - tape["latent_mid_pre"].to_numpy())',
       'return side * (tape["latent_mid_pre"].to_numpy() - tape["latent_mid_post"].to_numpy())')
mutant("recovery: saturation first trade", REC, "experiments/recovery.py",
       '["latent_mid_post"].last()',
       '["latent_mid_post"].first()')
mutant("recovery: saturation pre mid", REC, "experiments/recovery.py",
       'groupby("event", sort=False)["latent_mid_post"]',
       'groupby("event", sort=False)["latent_mid_pre"]')
mutant("recovery: saturation closed band", REC, "experiments/recovery.py",
       '((last < lo) | (last > hi))',
       '((last <= lo) | (last >= hi))')
mutant("recovery: near upper bound open", REC, "experiments/recovery.py",
       '("near (0,30]", (tau > 0.0) & (tau <= 30.0))',
       '("near (0,30]", (tau > 0.0) & (tau < 30.0))')
mutant("recovery: quiet bucket previous", REC, "experiments/recovery.py",
       'rows[len(BUCKET_LABELS)], rows[len(BUCKET_LABELS) - 1]',
       'rows[len(BUCKET_LABELS)], rows[len(BUCKET_LABELS) - 2]')
mutant("recovery: d row sign", REC, "experiments/recovery.py",
       'near[key] - quiet[key] if key not in',
       'quiet[key] - near[key] if key not in')
mutant("recovery: bias sign", REC, "experiments/recovery.py",
       'row["bias_cents"] = row["tape_as_cents"] - row["oracle_as_cents"]',
       'row["bias_cents"] = row["oracle_as_cents"] - row["tape_as_cents"]')
mutant("recovery: scaled over tape defined", REC, "experiments/recovery.py",
       'has_scaled = in_window & ~np.isnan(scaled)',
       'has_scaled = in_window & defined')
mutant("recovery: informed share in minutes", REC, "experiments/recovery.py",
       'informed_share(mean_tau / 60.0, params)',
       'informed_share(mean_tau, params)')
mutant("recovery: oracle not in cents", REC, "experiments/recovery.py",
       '"oracle_as_cents": CENTS * float(oracle[use].mean())',
       '"oracle_as_cents": float(oracle[use].mean())')
mutant("recovery: undefined proxy kept", REC, "experiments/recovery.py",
       'use = in_window & defined',
       'use = in_window')
mutant("recovery: interval sign closed", REC, "experiments/recovery.py",
       '"positive" if lo > 0.0 else "negative" if hi < 0.0 else "zero"',
       '"positive" if lo >= 0.0 else "negative" if hi <= 0.0 else "zero"')
mutant("recovery: surge share inverted", REC, "experiments/recovery.py",
       'point = spec_null.effects["raw_0_30"]["D"] / planted.effects["raw_0_30"]["D"]',
       'point = planted.effects["raw_0_30"]["D"] / spec_null.effects["raw_0_30"]["D"]')
mutant("recovery: surge bootstrap unpaired", REC, "experiments/recovery.py",
       'share = effect(b) / effect(a)',
       'share = effect(b) / effect(a)[::-1]')
mutant("recovery: surge undefined counted wrong", REC, "experiments/recovery.py",
       '"B_undefined": int(B - usable.sum())',
       '"B_undefined": int(usable.sum())')
mutant("recovery: surge ci keeps undefined", REC, "experiments/recovery.py",
       'percentile_interval(share[usable])',
       'percentile_interval(np.nan_to_num(share))')
mutant("recovery: surge effect quiet dropped", REC, "experiments/recovery.py",
       '- ratio_of_sums(s["quiet_sum"], s["quiet_n"], weights))',
       '- 0.0)')
mutant("recovery: naive one sided z", REC, "experiments/recovery.py",
       'NormalDist().inv_cdf(0.5 + level / 2.0)',
       'NormalDist().inv_cdf(level)')
mutant("recovery: naive variance ddof0", REC, "experiments/recovery.py",
       'a.var(ddof=1) / a.size + b.var(ddof=1) / b.size',
       'a.var() / a.size + b.var() / b.size')
mutant("recovery: naive quiet lower closed", REC, "experiments/recovery.py",
       '(tau > quiet[0]) & (tau <= quiet[1])',
       '(tau >= quiet[0]) & (tau <= quiet[1])')
mutant("recovery: naive near lower closed", REC, "experiments/recovery.py",
       '(tau > near[0]) & (tau <= near[1])',
       '(tau >= near[0]) & (tau <= near[1])')
mutant("recovery: naive difference sign", REC, "experiments/recovery.py",
       'd = float(a.mean() - b.mean())',
       'd = float(b.mean() - a.mean())')
mutant("recovery: naive nan dropped not", REC, "experiments/recovery.py",
       'present = ~np.isnan(values)',
       'present = np.ones(len(values), dtype=bool)')
mutant("recovery: bootstrap seed without rep", REC, "experiments/recovery.py",
       'np.random.SeedSequence([BASE_SEED, 2, rep])',
       'np.random.SeedSequence([BASE_SEED, 2])')
mutant("recovery: world seed without rep", REC, "experiments/recovery.py",
       'generate_world(world_params(world, max(sizes)), BASE_SEED + rep)',
       'generate_world(world_params(world, max(sizes)), BASE_SEED)')
mutant("recovery: nested prefix inclusive", REC, "experiments/recovery.py",
       'first = slim[slim["event"] < n]',
       'first = slim[slim["event"] <= n]')
mutant("recovery: replication column ignored", REC, "experiments/recovery.py",
       'column=e.column, B=B, seed=seed)',
       'column="adverse_selection", B=B, seed=seed)')
mutant("recovery: replication B ignored", REC, "experiments/recovery.py",
       'column=e.column, B=B, seed=seed)',
       'column=e.column, B=BOOTSTRAP_B, seed=seed)')
mutant("recovery: naive column ignored", REC, "experiments/recovery.py",
       'naive_interval(first, e.column, e.near)',
       'naive_interval(first, "adverse_selection", e.near)')
mutant("recovery: interval ends swapped", REC, "experiments/recovery.py",
       '"lo": r["D_ci"][0], "hi": r["D_ci"][1], "p": r["p_one_sided"],',
       '"lo": r["D_ci"][1], "hi": r["D_ci"][0], "p": r["p_one_sided"],')
mutant("recovery: event no shifted", REC, "experiments/recovery.py",
       'get_indexer(markouts["event"]),',
       'get_indexer(markouts["event"]) + 1,')
mutant("recovery: coverage strict inequalities", REC, "experiments/recovery.py",
       '((cell["lo"] <= truth) & (truth <= cell["hi"]))',
       '((cell["lo"] < truth) & (truth < cell["hi"]))')
mutant("recovery: coverage naive uses cluster", REC, "experiments/recovery.py",
       'naive = float(((cell["naive_lo"] <= truth) & (truth <= cell["naive_hi"])).mean())',
       'naive = float(((cell["lo"] <= truth) & (truth <= cell["hi"])).mean())')
mutant("recovery: coverage sd population", REC, "experiments/recovery.py",
       'cell["D"].std(ddof=1)',
       'cell["D"].std(ddof=0)')
mutant("recovery: coverage width naive cluster", REC, "experiments/recovery.py",
       '"width_naive_cents": CENTS * float((cell["naive_hi"] - cell["naive_lo"]).mean()),',
       '"width_naive_cents": CENTS * float((cell["hi"] - cell["lo"]).mean()),')
mutant("recovery: power at or below alpha", REC, "experiments/recovery.py",
       'reject = float((cell["p"] < ALPHA).mean())',
       'reject = float((cell["p"] <= ALPHA).mean())')
mutant("recovery: power population of planted", REC, "experiments/recovery.py",
       'float(truth.loc[(world, key), "D_cents"])',
       'float(truth.loc[("planted", key), "D_cents"])')
mutant("recovery: sample size at least", REC, "experiments/recovery.py",
       'rows.loc[rows["reject"] > level, "n_events"]',
       'rows.loc[rows["reject"] >= level, "n_events"]')
mutant("recovery: sample size largest", REC, "experiments/recovery.py",
       'out[e.key] = int(enough.min()) if len(enough) else None',
       'out[e.key] = int(enough.max()) if len(enough) else None')
mutant("recovery: share se denominator", REC, "experiments/recovery.py",
       'np.sqrt(share * (1.0 - share) / n)',
       'np.sqrt(share * (1.0 - share) / (n - 1))')
mutant("recovery: cells keep empty cells", REC, "experiments/recovery.py",
       ('                if mask.any():\n                    yield world, e.key, int(n), '
        'replications[mask]'),
       '                yield world, e.key, int(n), replications[mask]')
mutant("recovery: truth of planted for all", REC, "experiments/recovery.py",
       ('truth = {(w, e.key): p.effects[e.key]["D"] for w, p in populations.items() for e in '
        'ESTIMANDS}'),
       ('truth = {(w, e.key): populations["planted"].effects[e.key]["D"] for w, p in '
        'populations.items() for e in ESTIMANDS}'))
mutant("recovery: replication numbers from one", REC, "experiments/recovery.py",
       'for w in WORLDS for r in range(reps)]',
       'for w in WORLDS for r in range(1, reps + 1)]')
mutant("recovery: design reps constant", REC, "experiments/recovery.py",
       '"reps": reps, "reps_design"',
       '"reps": REPS, "reps_design"')
mutant("recovery: population seed shifted", REC, "experiments/recovery.py",
       'population_world(w, pop_events, POP_SEED, B)',
       'population_world(w, pop_events, POP_SEED + 1, B)')
mutant("recovery: spec null without surge", REC, "experiments/recovery.py",
       '"spec_null": {"pi1": 0.0},',
       '"spec_null": {"pi1": 0.0, "surge": 0.0},')
mutant("recovery: flat null keeps surge", REC, "experiments/recovery.py",
       '"flat_null": {"pi1": 0.0, "surge": 0.0},',
       '"flat_null": {"pi1": 0.0},')
mutant("recovery: public markouts see latent", REC, "experiments/recovery.py",
       'tape = world.tape[[c for c in world.tape.columns if not str(c).startswith("latent_")]]',
       'tape = world.tape')
mutant("recovery: cut note at design", REC, "experiments/recovery.py",
       'if d["reps"] >= d["reps_design"]:',
       'if d["reps"] > d["reps_design"]:')
mutant("recovery: lost coverage ratio", REC, "experiments/recovery.py",
       'np.sqrt(1.0 + big / d["pop_events"])',
       'np.sqrt(1.0 + d["pop_events"] / big)')
mutant("recovery: mc se half", REC, "experiments/recovery.py",
       'se = 100.0 * np.sqrt(0.25 / reps)',
       'se = 100.0 * np.sqrt(0.5 / reps)')
mutant("recovery: effect sentence and", REC, "experiments/recovery.py",
       'if spec["sign"] != "zero" or flat["sign"] != "zero":',
       'if spec["sign"] != "zero" and flat["sign"] != "zero":')
mutant("recovery: effect sentence raw clean", REC, "experiments/recovery.py",
       'clean = _effect(res, "flat_null", "scaled_5_30")',
       'clean = _effect(res, "flat_null", "raw_5_30")')
mutant("recovery: coverage reference smallest", REC, "experiments/recovery.py",
       'ref = 200 if 200 in d["sizes"] else max(d["sizes"])',
       'ref = 200 if 200 in d["sizes"] else min(d["sizes"])')
mutant("recovery: table ids duplicate", REC, "experiments/recovery.py",
       '"### M6. Mean width',
       '"### M5. Mean width')
mutant("recovery: share without percent", REC, "experiments/recovery.py",
       'return f"{100.0 * x:.{digits}f}"',
       'return f"{x:.{digits}f}"')
mutant("recovery: sample next smaller includes self", REC, "experiments/recovery.py",
       'smaller = sorted(s for s in d["sizes"] if s < n)',
       'smaller = sorted(s for s in d["sizes"] if s <= n)')
mutant("recovery: main reps ignored", REC, "experiments/recovery.py",
       'res = run_study(args.reps, args.B, args.pop_events, args.sizes)',
       'res = run_study(REPS, args.B, args.pop_events, args.sizes)')
mutant("recovery: main B ignored", REC, "experiments/recovery.py",
       'res = run_study(args.reps, args.B, args.pop_events, args.sizes)',
       'res = run_study(args.reps, BOOTSTRAP_B, args.pop_events, args.sizes)')
mutant("recovery: main sizes ignored", REC, "experiments/recovery.py",
       'res = run_study(args.reps, args.B, args.pop_events, args.sizes)',
       'res = run_study(args.reps, args.B, args.pop_events, SIZES)')
mutant("recovery: main pop events ignored", REC, "experiments/recovery.py",
       'res = run_study(args.reps, args.B, args.pop_events, args.sizes)',
       'res = run_study(args.reps, args.B, POP_EVENTS, args.sizes)')
mutant("recovery: budget reps design changed", REC, "experiments/recovery.py",
       'REPS_DESIGN = 200',
       'REPS_DESIGN = 100')
mutant("recovery: bootstrap B constant", REC, "experiments/recovery.py",
       'BOOTSTRAP_B = 1000',
       'BOOTSTRAP_B = 500')
mutant("recovery: alpha constant", REC, "experiments/recovery.py",
       'ALPHA = 0.05',
       'ALPHA = 0.1')
mutant("recovery: horizon constant", REC, "experiments/recovery.py",
       'HORIZON_S = 300.0',
       'HORIZON_S = 600.0')
mutant("recovery: quiet constant", REC, "experiments/recovery.py",
       'QUIET = (360.0, np.inf)',
       'QUIET = (180.0, np.inf)')
mutant("recovery: sizes constant", REC, "experiments/recovery.py",
       'SIZES = (25, 50, 100, 200, 400)',
       'SIZES = (25, 50, 100, 200, 300)')

# ---- the calibration study (experiments/calibration_study.py)
CST = "tests/test_calibration_study.py"
mutant("calibration study: world x sd ignored", CST, "experiments/calibration_study.py",
       'x = rng.normal(0.0, x_sd, n_events)',
       'x = rng.normal(0.0, 1.5, n_events)')
mutant("calibration study: outcome uses price slope", CST, "experiments/calibration_study.py",
       '< sigmoid(x)).astype(np.int64)',
       '< sigmoid(beta * x)).astype(np.int64)')
mutant("calibration study: price without noise", CST, "experiments/calibration_study.py",
       'price = sigmoid(beta * x + rng.normal(0.0, noise_sd, n_events))',
       'price = sigmoid(beta * x)')
mutant("calibration study: spacing two weeks", CST, "experiments/calibration_study.py",
       'SPACING_S = 168.0 * 3600.0',
       'SPACING_S = 336.0 * 3600.0')
mutant("calibration study: pop no clip", CST, "experiments/calibration_study.py",
       'zc = np.clip(z, logit(clip), logit(1.0 - clip))',
       'zc = z')
mutant("calibration study: pop weight norm", CST, "experiments/calibration_study.py",
       'weight = np.outer(wx, we) / (2.0 * np.pi)',
       'weight = np.outer(wx, we) / np.pi')
mutant("calibration study: pop noise dropped", CST, "experiments/calibration_study.py",
       'z = beta * x + noise_sd * grid_e',
       'z = beta * x')
mutant("calibration study: pop brier raw swapped", CST, "experiments/calibration_study.py",
       'brier_raw = float(np.sum(weight * (q * (1.0 - p) ** 2 + (1.0 - q) * p**2)))',
       'brier_raw = float(np.sum(weight * (q * p**2 + (1.0 - q) * (1.0 - p) ** 2)))')
mutant("calibration study: pop bss direction", CST, "experiments/calibration_study.py",
       '"bss": 1.0 - brier_recal / brier_raw}',
       '"bss": brier_recal / brier_raw - 1.0}')
mutant("calibration study: seed without rep", CST, "experiments/calibration_study.py",
       'key = [BASE_SEED, 1, round(100 * beta), n_events, rep]',
       'key = [BASE_SEED, 1, round(100 * beta), n_events]')
mutant("calibration study: boot seed same stream", CST, "experiments/calibration_study.py",
       'np.random.SeedSequence([BASE_SEED, 2, *key[2:]])',
       'np.random.SeedSequence([BASE_SEED, 1, *key[2:]])')
mutant("calibration study: min events off by one", CST, "experiments/calibration_study.py",
       'world.price, world.outcome, world.release_ts, min_events=MIN_EVENTS,',
       'world.price, world.outcome, world.release_ts, min_events=MIN_EVENTS + 1,')
mutant("calibration study: clusters are pairs", CST, "experiments/calibration_study.py",
       'refit_every=REFIT_EVERY, cluster=np.arange(n_events),',
       'refit_every=REFIT_EVERY, cluster=np.arange(n_events) // 2,')
mutant("calibration study: slope last not mean", CST, "experiments/calibration_study.py",
       '"slope": float(frame["b"].mean()),',
       '"slope": float(frame["b"].iloc[-1]),')
mutant("calibration study: supported at or below", CST, "experiments/calibration_study.py",
       'supported = float((ok["p_one_sided"] < ALPHA).mean())',
       'supported = float((ok["p_one_sided"] <= ALPHA).mean())')
mutant("calibration study: failed rows kept", CST, "experiments/calibration_study.py",
       'ok = replications[~replications["failed"].astype(bool)]',
       'ok = replications')
mutant("calibration study: supported se divisor", CST, "experiments/calibration_study.py",
       'np.sqrt(supported * (1.0 - supported) / n)',
       'np.sqrt(supported * (1.0 - supported) / (n - 1))')
mutant("calibration study: n scored wrong", CST, "experiments/calibration_study.py",
       '"n_scored": n_events - MIN_EVENTS,',
       '"n_scored": n_events,')
mutant("calibration study: inverse beta wrong", CST, "experiments/calibration_study.py",
       'out["slope_inverse_beta"] = 1.0 / beta',
       'out["slope_inverse_beta"] = beta')
mutant("calibration study: table null rows wrong", CST, "experiments/calibration_study.py",
       'null = cells[cells["beta"] == 1.0]\n    null_text',
       'null = cells[cells["beta"] < 1.0]\n    null_text')
mutant("calibration study: table bss not percent", CST, "experiments/calibration_study.py",
       '| {_pct(c.bss_mean, 2)} "\n            f"({_pct(c.bss_se, 2)})',
       '| {c.bss_mean:.2f} "\n            f"({_pct(c.bss_se, 2)})')
mutant("calibration study: table slope column swapped", CST, "experiments/calibration_study.py",
       'f"{c.slope_inverse_beta:.3f} | {c.slope_pop:.3f} | "',
       'f"{c.slope_pop:.3f} | {c.slope_inverse_beta:.3f} | "')
mutant("calibration study: main wrong population", CST, "experiments/calibration_study.py",
       'rows.append(summarise_cell(reps, beta, n_events, args.B, population[beta]))',
       'rows.append(summarise_cell(reps, beta, n_events, args.B, population[BETAS[0]]))')
mutant("calibration study: main false positive beta", CST, "experiments/calibration_study.py",
       'for r in cells.itertuples() if r.beta == 1.0',
       'for r in cells.itertuples() if r.beta == 0.7')
mutant("calibration study: main reps ignored", CST, "experiments/calibration_study.py",
       'reps = run_cell(beta, n_events, args.reps, args.B)',
       'reps = run_cell(beta, n_events, REPS, args.B)')
mutant("calibration study: main B ignored", CST, "experiments/calibration_study.py",
       'reps = run_cell(beta, n_events, args.reps, args.B)',
       'reps = run_cell(beta, n_events, args.reps, BOOTSTRAP_B)')
mutant("calibration study: reps constant", CST, "experiments/calibration_study.py",
       'REPS = 200',
       'REPS = 100')
mutant("calibration study: failed row not marked", CST, "experiments/calibration_study.py",
       'rows.append({**row, "failed": True})',
       'rows.append({**row, "failed": False})')

# ---- more recalibration mutants (calibration.py)
mutant("calibration: clip not applied in fit", CAL, "calibration.py",
       '    x = _clipped_logit(p, clip)\n    if x.max()',
       '    x = logit(p)\n    if x.max()')
mutant("calibration: refit every off by one", CAL, "calibration.py",
       'stop = int(first[k + refit_every]) if k + refit_every < k_total else n',
       'stop = int(first[k + refit_every - 1]) if k + refit_every - 1 < k_total else n')
mutant("calibration: refit every range step", CAL, "calibration.py",
       'for k in range(min_events, k_total, refit_every):',
       'for k in range(min_events, k_total, refit_every + 1):')
mutant("calibration: rows not releases", CAL, "calibration.py",
       'releases, first = np.unique(ts, return_index=True)',
       'releases, first = np.arange(n, dtype=float), np.arange(n)')
mutant("calibration: ordering ignores time", CAL, "calibration.py",
       'keys = [y, p, ts]  # np.lexsort sorts by the last key first',
       'keys = [y, ts, p]  # np.lexsort sorts by the last key first')
mutant("calibration: ratio inverted", CAL, "calibration.py",
       'boot = 1.0 - ratio',
       'boot = ratio - 1.0')
mutant("calibration: brier absolute error", CAL, "calibration.py",
       'return float(np.mean((p - y) ** 2))',
       'return float(np.mean(np.abs(p - y)))')
mutant("calibration: max iter returns garbage", CAL, "calibration.py",
       '        a, b, ll = a + step * da, b + step * db, ll_new\n    raise not_converged',
       '        a, b, ll = a + step * da, b + step * db, ll_new\n    return a, b')
mutant("calibration: cluster codes unsorted", CAL, "calibration.py",
       'codes, uniques = pd.factorize(arr, sort=True)',
       'codes, uniques = pd.factorize(arr)')
mutant("calibration: n fit wrong", CAL, "calibration.py",
       'columns["n_fit"][start:stop] = start',
       'columns["n_fit"][start:stop] = stop')
mutant("calibration: min obs constant", CAL, "calibration.py",
       'MIN_OBSERVATIONS = 10',
       'MIN_OBSERVATIONS = 5')
mutant("calibration: scored cut wrong", CAL, "calibration.py",
       'scored = int(first[min_events]) if k_total > min_events else n',
       'scored = int(first[min_events]) + 1 if k_total > min_events else n')
mutant("calibration: skill ignores seed", CAL, "calibration.py",
       'np.random.default_rng(seed))',
       'np.random.default_rng(0))')
mutant("calibration: level ignored", CAL, "calibration.py",
       '"ci": percentile_interval(boot, level),',
       '"ci": percentile_interval(boot, 0.95),')


# ---- the scripts that paste the tables into the README and check the numbers there
RDM = "tests/test_readme_scripts.py"
file_mutant("make_tables: markers that do not pair up are accepted", RDM,
            "scripts/make_tables.py", "if opens != closes:", "if False:")
file_mutant("make_tables: a table that no file defines is not reported", RDM,
            "scripts/make_tables.py", "if table_id not in tables:", "if False:")
file_mutant("make_tables: an ID that two files define is accepted", RDM,
            "scripts/make_tables.py", "if table_id in tables:", "if False:")
file_mutant("make_tables: check mode never fails", RDM, "scripts/make_tables.py",
            "    if args.check:\n        if new != old:", "    if args.check:\n        if False:")
file_mutant("check_readme_numbers: a number that left the results file is accepted", RDM,
            "scripts/check_readme_numbers.py", "if in_file not in cache[path]:", "if False:")
file_mutant("check_readme_numbers: a number that left the README is accepted", RDM,
            "scripts/check_readme_numbers.py", "if in_readme not in readme:", "if False:")
file_mutant("check_readme_numbers: problems do not change the exit status", RDM,
            "scripts/check_readme_numbers.py", "return 1 if problems else 0", "return 0")
file_mutant("check_readme_numbers: a pasted table that differs is accepted", RDM,
            "scripts/check_readme_numbers.py",
            "make_tables.read_tables(ROOT / \"results\")) != readme:",
            "make_tables.read_tables(ROOT / \"results\")) == readme:")
file_mutant("check_readme_numbers: web links are looked up as files", RDM,
            "scripts/check_readme_numbers.py",
            'if target.startswith(("http://", "https://", "#", "mailto:")):', "if False:")
file_mutant("check_readme_numbers: the anchor of a link is not removed", RDM,
            "scripts/check_readme_numbers.py", 'target.split("#", 1)[0]', "target")


def run_pytest(work: Path, targets: list[str], timeout: float):
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    t0 = time.monotonic()
    try:
        p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-rf", "--tb=no",
                            "-p", "no:cacheprovider", *targets], cwd=work, env=env,
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, ["(timeout)"], time.monotonic() - t0, "timeout"
    failed = [ln.split(" - ")[0].replace("FAILED ", "") for ln in p.stdout.splitlines()
              if ln.startswith("FAILED") or ln.startswith("ERROR")]
    summary = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-300:]
    return p.returncode, failed, time.monotonic() - t0, summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Mutation check for the kalshi_mm test suite.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per pytest run")
    args = ap.parse_args()
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    def git(*a: str) -> str:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True,
                              text=True).stdout.strip()

    import numpy

    dirty = git("status", "--porcelain", "--", "src", "tests")
    log("# Generated by tests/mutation_check.py; do not edit by hand.")
    log("# command: python3 tests/mutation_check.py " + " ".join(
        shlex.quote(a) for a in sys.argv[1:]))
    log(f"# git:     {git('rev-parse', 'HEAD') or 'unknown'} "
        f"({'uncommitted changes in src/ or tests/' if dirty else 'clean'})")
    log(f"# date:    {datetime.datetime.now(datetime.UTC):%Y-%m-%dT%H:%M:%SZ}")
    log(f"# python:  {platform.python_version()}; numpy {numpy.__version__}")
    log("# For each mutant: the pytest summary and the first failing test.")
    log("")
    survived = 0
    with tempfile.TemporaryDirectory(prefix="kalshi_mutation_") as tmp:
        work = Path(tmp)
        for item in COPY:
            src = REPO / item
            if src.is_dir():
                shutil.copytree(src, work / item, ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".pytest_cache", "*.egg-info"))
            else:
                shutil.copy2(src, work / item)
        rc, _, secs, summary = run_pytest(work, ["tests"], args.timeout)
        log(f"baseline (no mutant, whole suite): {summary} [{secs:.0f} s]")
        if rc != 0:
            log("baseline must pass; aborting")
            return 1
        log("")
        originals: dict[str, str] = {}
        for name, targets, edits in MUTANTS:
            touched: dict[str, str] = {}
            try:
                for rel, old, new in edits:
                    if rel not in originals:
                        originals[rel] = (work / rel).read_text()
                    text = touched.get(rel, originals[rel])
                    if text.count(old) != 1:
                        log(f"{name}: SETUP ERROR, pattern found {text.count(old)} times in {rel}")
                        return 1
                    touched[rel] = text.replace(old, new)
                for rel, text in touched.items():
                    (work / rel).write_text(text)
                rc, failed, secs, summary = run_pytest(work, targets, args.timeout)
                caught = rc != 0
                survived += not caught
                log(f"{name}: {'caught' if caught else 'SURVIVED'} - {summary} [{secs:.0f} s]")
                for f in failed[:1]:
                    log(f"    {f}")
            finally:
                for rel in touched:
                    (work / rel).write_text(originals[rel])
    log("")
    log(f"{len(MUTANTS) - survived} of {len(MUTANTS)} mutants caught.")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines) + "\n")
        print(f"wrote {args.out}")
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
