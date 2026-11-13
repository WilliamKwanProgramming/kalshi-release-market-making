"""Run the protocol of docs/protocol.md on a database of real Kalshi data.

    python3 scripts/run_real.py --db data/kalshi.sqlite [--out results/real]

The database is opened read-only and must say that it holds rows from Kalshi's public API (a
database of synthetic rows is refused: use ``kalshi_mm.experiments.rehearsal`` for those). The run
prints, before any statistic, the counts of series, events, markets and trades, the gap between
each release and its markets' close times, and three events to compare with the Kalshi website
(docs/protocol.md, "Checks before reading any result"), and the checks of the data
(:func:`data_checks`: trades off the 0.01 USD tick, trades read from both trade endpoints, the
price grids of the markets). It then refuses to run unless the files
that decide a result (``src``, ``scripts``, ``docs/protocol.md``, ``docs/model.md``,
``pyproject.toml``) are those of the commit tagged ``real-run``; ``--no-tag-check`` runs anyway and
writes the departure to ``DEVIATIONS.md``, as every ``--set`` that changes a setting does. The
record of the fetches (``panel.fetch_runs``) is printed and written; a database without one, a
fetch run from a repository with uncommitted changes, a fetch whose commit differs from the tag
in the guarded files, a tag that is not annotated (it has no creation time) and a fetch that
started before the tag was created are departures too (:func:`provenance_problems`). P2 is
labelled exploratory when more than ``Settings.max_off_tick_share`` of the kept trades are off the
0.01 USD tick (:func:`off_tick_share`). Results go to ``results/real`` with the table letter K.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import sqlite3
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kalshi_mm.experiments.common import REPO, Manifest
from kalshi_mm.ingest import lookalike_groups
from kalshi_mm.panel import (
    FETCH_FIELDS,
    fetch_runs,
    get_source,
    read_api_fields,
    read_dataset,
    schema_problem,
)
from kalshi_mm.protocol import DEFAULT, data_label, run_protocol, with_overrides
from kalshi_mm.protocol_report import write_results

__all__ = [
    "data_checks",
    "describe_checks",
    "describe_dataset",
    "main",
    "off_tick",
    "off_tick_share",
    "provenance_problems",
    "tag_created",
    "tag_problem",
]

TAG = "real-run"
GUARDED = ("src", "scripts", "docs/protocol.md", "docs/model.md", "pyproject.toml")
TICK = 0.01  # USD: the grid the backtest quotes on (QuoteParams.tick, WorldParams.tick)
TICK_TOLERANCE = 1e-9  # USD
NOT_RECORDED = "not recorded"


def _utc(ts: float) -> str:
    return datetime.datetime.fromtimestamp(float(ts), datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")


def describe_dataset(tape: pd.DataFrame, markets: pd.DataFrame, events: pd.DataFrame) -> str:
    """Counts by series, the gap of close time to release time and three events to spot-check.

    The gap is ``close_ts - release_ts`` of each market in hours, with its quantiles over all
    markets. The three events are the first, the middle and the last by release time.
    """
    series_of = events.set_index("event")["series"]
    release_of = events.set_index("event")["release_ts"]
    by = pd.DataFrame(
        {
            "events": events.groupby("series").size(),
            "markets": markets.groupby(markets["event"].map(series_of)).size(),
            "settled": markets[markets["outcome"].notna()]
            .groupby(markets["event"].map(series_of)).size(),
            "trades": tape.groupby(tape["event"].map(series_of)).size(),
        }
    ).fillna(0).astype(int)
    by.loc["all"] = by.sum()
    lines = ["Counts by series:", by.to_string(), ""]
    gap = (markets["close_ts"] - markets["event"].map(release_of)) / 3600.0
    gap = gap.dropna()
    if len(gap):
        q = np.quantile(gap, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
        names = ("min", "5%", "25%", "median", "75%", "95%", "max")
        lines.append("Market close time minus release time, hours:")
        lines.append("  " + "  ".join(f"{n} {v:.2f}" for n, v in zip(names, q, strict=True)))
        lines.append(f"  markets closing before the release: {int((gap < 0).sum())} of {len(gap)}")
        lines.append("")
    if len(events):
        order = events.sort_values(["release_ts", "event"]).reset_index(drop=True)
        picks = sorted({0, len(order) // 2, len(order) - 1})
        n_markets = markets.groupby("event").size()
        n_trades = tape.groupby("event").size()
        lines.append("Events to compare with the Kalshi website:")
        for i in picks:
            e = order.loc[i]
            lines.append(
                f"  {e['event']} ({e['series']}), release {_utc(e['release_ts'])}: "
                f"{int(n_markets.get(e['event'], 0))} markets, "
                f"{int(n_trades.get(e['event'], 0))} trades"
            )
    return "\n".join(lines)


def off_tick(prices: np.ndarray | pd.Series, tick: float = TICK,
             tolerance: float = TICK_TOLERANCE) -> np.ndarray:
    """True where a price is not a whole multiple of ``tick``: ``|p - tick * round(p / tick)|``
    is above ``tolerance`` (USD)."""
    p = np.asarray(prices, dtype=np.float64)
    return np.abs(p - tick * np.round(p / tick)) > tolerance


def data_checks(markets: pd.DataFrame, events: pd.DataFrame, api_trades: pd.DataFrame,
                api_markets: pd.DataFrame) -> dict[str, Any]:
    """Counts of the stored trades and markets that the analysis does not see (docs/data.md).

    ``api_trades`` and ``api_markets`` are the frames of ``panel.read_api_fields``. Per series and
    for ``all``: the kept trades; those whose price is off the tick (:func:`off_tick`); those
    whose ``trade_id`` was returned by both trade endpoints (``endpoint`` is ``both``); and the
    groups of trades alike across the two endpoints (``ingest.lookalike_groups``). Also the number
    of markets by ``price_level_structure``.
    """
    series_of = markets["event"].map(events.set_index("event")["series"])
    series_of.index = markets["market"]
    trades = api_trades.assign(series=api_trades["market"].map(series_of))
    rows: dict[str, dict[str, int]] = {}
    for name, part in [*trades.groupby("series", sort=True), ("all", trades)]:
        rows[str(name)] = {
            "trades kept": int(len(part)),
            "off the 0.01 tick": int(off_tick(part["price"]).sum()),
            "trade_id from both endpoints": int((part["endpoint"] == "both").sum()),
            "groups alike across endpoints": lookalike_groups(part.to_dict("records")),
        }
    structures = api_markets["price_level_structure"].fillna(NOT_RECORDED).value_counts()
    return {
        "tick_usd": TICK,
        "tolerance_usd": TICK_TOLERANCE,
        "by_series": rows,
        "markets by price_level_structure": {str(k): int(structures[k])
                                             for k in sorted(structures.index)},
    }


def off_tick_share(checks: dict[str, Any]) -> float | None:
    """The share of the kept trades that are off the tick, from :func:`data_checks`, or None when
    there is no kept trade."""
    total = checks["by_series"]["all"]
    if total["trades kept"] == 0:
        return None
    return total["off the 0.01 tick"] / total["trades kept"]


def describe_checks(checks: dict[str, Any]) -> str:
    """The checks of :func:`data_checks` as text."""
    rows = checks["by_series"]
    total = rows["all"]
    structures = checks["markets by price_level_structure"]
    return "\n".join([
        "Checks of the data (kept trades):",
        pd.DataFrame(rows).T.to_string(),
        f"The backtest quotes on a {TICK} USD tick; {total['off the 0.01 tick']} of the "
        f"{total['trades kept']} kept trades are at prices off it.",
        "Markets by price_level_structure: "
        + (", ".join(f"{k} {v}" for k, v in structures.items()) or "none"),
    ])


def tag_problem(repo: Path = REPO) -> str | None:
    """Why the files that decide a result differ from the tag ``real-run``, or None."""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)

    if git("rev-parse", "-q", "--verify", f"refs/tags/{TAG}^{{commit}}").returncode != 0:
        return f"the tag {TAG} does not exist"
    changed = git("diff", "--name-only", TAG, "--", *GUARDED).stdout.split()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *GUARDED).stdout.split()
    files = sorted(set(changed) | set(untracked))
    if files:
        shown = ", ".join(files[:5]) + (", ..." if len(files) > 5 else "")
        return f"{len(files)} file(s) differ from the tag {TAG}: {shown}"
    return None


def tag_created(repo: Path = REPO) -> datetime.datetime | None:
    """When the tag ``real-run`` was created, or None.

    The time is the tagger date of an annotated tag (``git tag -a``). A lightweight tag has none,
    and a tag that does not exist has none either.
    """
    shown = subprocess.run(
        ["git", "-C", str(repo), "for-each-ref", "--format=%(taggerdate:iso-strict)",
         f"refs/tags/{TAG}"], capture_output=True, text=True,
    ).stdout.strip()
    return datetime.datetime.fromisoformat(shown) if shown else None


def provenance_problems(runs: Sequence[dict[str, Any]], repo: Path = REPO) -> list[str]:
    """Departures from docs/protocol.md that the record of the fetches shows, as sentences.

    ``runs`` is ``panel.fetch_runs`` of the database. A departure is: no record at all; a fetch
    with ``git_dirty`` true; a fetch whose ``git_commit`` is unknown, not in ``repo`` or differs
    from the tag ``real-run`` in the guarded files (``git diff --quiet <commit> real-run --
    <guarded>``; the comparison is made once per commit); a tag that exists but is not annotated,
    so that its creation time is unknown (:func:`tag_created`); a fetch that started before the
    tag was created.
    """
    if not runs:
        return ["the database has no record of its fetch (fetch_started_utc, git_commit): it was "
                "not written by scripts/fetch_kalshi.py with a record"]
    problems = []
    dirty = [r for r in runs if r["git_dirty"] == "true"]
    if dirty:
        problems.append(f"{len(dirty)} of {len(runs)} fetches ran from a repository with "
                        f"uncommitted changes (git_dirty true), the first started "
                        f"{dirty[0]['fetch_started_utc']}")
    for commit in dict.fromkeys(r["git_commit"] for r in runs):
        if commit == "unknown":
            problems.append("a fetch has no recorded commit (git was unavailable)")
            continue
        diff = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", commit, TAG, "--",
                               *GUARDED], capture_output=True, text=True)
        if diff.returncode == 1:
            problems.append(f"a fetch ran from commit {commit}, whose guarded files differ "
                            f"from the tag {TAG}")
        elif diff.returncode != 0:
            why = (diff.stderr.strip().splitlines() or ["git failed"])[0]
            problems.append(f"the commit {commit} of a fetch cannot be compared with the tag "
                            f"{TAG}: {why}")
    problems += _tag_date_problems(runs, repo)
    return problems


def _tag_date_problems(runs: Sequence[dict[str, Any]], repo: Path) -> list[str]:
    """The departures about the creation time of the tag ``real-run`` (see
    :func:`provenance_problems`); none when the tag does not exist (``tag_problem`` says so)."""
    exists = subprocess.run(["git", "-C", str(repo), "rev-parse", "-q", "--verify",
                             f"refs/tags/{TAG}"], capture_output=True, text=True).returncode == 0
    if not exists:
        return []
    created = tag_created(repo)
    if created is None:
        return [f"the tag {TAG} is not annotated (git tag -a), so it has no creation time and it "
                "cannot be shown that it was made before the fetch"]
    started = []
    for r in runs:
        try:
            when = datetime.datetime.fromisoformat(r["fetch_started_utc"])
        except (TypeError, ValueError):
            return [f"a fetch has no readable start time ({r['fetch_started_utc']!r}), so it "
                    f"cannot be compared with the creation time of the tag {TAG}"]
        # a time without an offset is read as UTC, the zone in which the fetch records its start
        started.append(when if when.tzinfo else when.replace(tzinfo=datetime.UTC))
    first = min(started)
    if first < created:
        return [f"a fetch started {first.isoformat()}, before the tag {TAG} was created "
                f"({created.isoformat()})"]
    return []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the protocol on a database of real data; the exit status is 0, or 2 if it refuses."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--db", type=Path, default=REPO / "data" / "kalshi.sqlite")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "real")
    ap.add_argument("--no-tag-check", action="store_true",
                    help=f"run although the guarded files differ from the tag {TAG}")
    ap.add_argument("--set", dest="assign", action="append", default=[], metavar="NAME=VALUE",
                    help="change a setting (quote.<name> for a quote parameter); every change "
                         "is listed in DEVIATIONS.md and makes the run exploratory")
    args = ap.parse_args(argv)
    manifest = Manifest("kalshi_mm.run_real", DEFAULT.seed, args=argv)

    if not args.db.is_file():
        print(f"run_real: {args.db} does not exist; fetch the data first (docs/data.md)")
        return 2
    conn = sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True)
    try:
        source = get_source(conn)
        if data_label(source) != "real data":
            print(f"run_real: the database holds {data_label(source)}, not real data; "
                  "synthetic rows belong to kalshi_mm.experiments.rehearsal")
            return 2
        old = schema_problem(conn)
        if old:
            print(f"run_real: {old}")
            return 2
        tape, markets, events = read_dataset(conn)
        api_trades, api_markets = read_api_fields(conn)
        runs = fetch_runs(conn)
    finally:
        conn.close()
    print(describe_dataset(tape, markets, events))
    checks = data_checks(markets, events, api_trades, api_markets)
    checks["fetch_runs"] = runs
    print("\n" + describe_checks(checks))
    print("\nFetches recorded in the database:")
    for r in runs:
        print(f"  started {r['fetch_started_utc']}, finished {r['fetch_finished_utc']}, "
              f"commit {r['git_commit']}, uncommitted changes {r['git_dirty']}")
    extra: list[str] = provenance_problems(runs)
    for problem in extra:
        print(f"  departure: {problem}")

    problem = tag_problem()
    if problem:
        if not args.no_tag_check:
            print(f"\nrun_real: {problem}. Tag the commit that holds the code and the protocol "
                  f"with an annotated tag (git tag -a {TAG} -m 'before the fetch') before the "
                  "data are fetched, or pass --no-tag-check.")
            return 2
        extra.append(f"run without a clean tag: {problem}")
    settings = with_overrides(DEFAULT, args.assign)
    share = off_tick_share(checks)
    results = run_protocol(tape, markets, events, source, settings, share)
    names = write_results(results, args.out, extra_deviations=extra, data_checks=checks)
    manifest.extra += [
        f"database: {args.db.name}, sha256 {_sha256(args.db)}",
        f"source: {source}",
        *[f"count, {k}: {v}" for k, v in results.dataset.counts.items()],
        *[f"check, {k}: {v}" for k, v in checks["by_series"]["all"].items()],
        f"tag {TAG}: {'matches' if problem is None else problem}",
        *[f"fetch {i}: " + ", ".join(f"{k} {r[k]}" for k in FETCH_FIELDS)
          for i, r in enumerate(runs, start=1)],
    ]
    manifest.write(args.out, "run_real", [*names, "run_real.manifest.txt"])
    print("\nTests of the family (Holm-adjusted, one-sided):")
    for name, v in results.family.items():
        print(f"  {name}: p = {v['p_one_sided']}, adjusted {v['p_holm']:.4f}, {v['verdict']}")
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
