#!/usr/bin/env python3
"""Mutation check: does the test suite notice deliberate bugs?

Copies src/, tests/ and pyproject.toml into a temporary directory, then applies ONE deliberate
bug ("mutant") at a time to the copy and runs pytest there. A mutant is "caught" if at least one
test fails. The working tree is never modified. The mutants are bugs that would silently produce
plausible numbers: a sign, an off-by-one, a wrong tick direction, look-ahead, a wrong horizon or a
unit error. To save time a mutant runs only the test files that exercise the changed module
(listed with it); the baseline runs the whole suite. Exits with status 1 if a mutant survives or
the baseline fails.

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
COPY = ["src", "tests", "pyproject.toml", "docs"]
S = "src/kalshi_mm/"

# (name, test files to run, [(file, text that must occur exactly once, replacement), ...])
MUTANTS: list[tuple[str, list[str], list[tuple[str, str, str]]]] = []


def mutant(name: str, tests: str, path: str, old: str, new: str) -> None:
    """Register a mutant that replaces ``old`` by ``new`` in ``src/kalshi_mm/<path>``."""
    MUTANTS.append((name, tests.split(), [(S + path, old, new)]))


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
