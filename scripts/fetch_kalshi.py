#!/usr/bin/env python3
"""Fetch Kalshi markets and trades into SQLite (docs/data.md; docs/protocol.md, "Data").

Run from the repository root with ``PYTHONPATH=src``. Three modes:

* ``--calendar-draft PATH`` lists the markets of the series and writes a draft release calendar
  (the file name must contain "draft"). The times are Kalshi's expected expiration times, not
  release times: check each one against the official calendars of the BLS, the BEA and the Federal
  Reserve and save the result as ``data/release_calendar.csv``. The analysis refuses a draft.
* the default fetches the series from the live and the historical endpoints into ``--db``, for the
  events of ``--calendar`` whose release is in ``--start`` to ``--end`` (both days included), and
  applies the protocol filters. A market is stored as soon as it is read, so running the same
  command again after a failure or an interrupt continues where it stopped. Each run records in
  the ``meta`` table when it started and finished and the commit of the repository it ran from,
  with whether the repository had uncommitted changes (``panel.begin_fetch_run``); a later run is
  added to the record and does not replace the first.
* ``--dry-run`` lists what a run would fetch, from the market listing and the calendar, and
  requests no trade and writes nothing.

The client was tested on canned responses shaped like the documentation and never run against the
live service. It sends no credentials.
"""

from __future__ import annotations

import argparse
import datetime
import sqlite3
import subprocess
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path

import pandas as pd

from kalshi_mm import ingest, panel
from kalshi_mm.kalshi_api import KalshiApiError, KalshiClient

REPO = Path(__file__).resolve().parents[1]
UNKNOWN = "unknown"

NOTICE = """\
Kalshi's Developer Agreement, v1.1:
https://kalshi-public-docs.s3.amazonaws.com/Kalshi-Developer-Agreement.pdf
Section 3 limits use of the API to facilitating a member's own trading. Section 3.1 prohibits
collecting, caching, aggregating or storing data accessed via the API except to facilitate your own
trading on Kalshi, and sharing such data with third parties without Kalshi's prior written
authorization. This script sends no credentials, and the agreement does not say whether it covers
the public endpoints. Read it before you run this. The data go to the database you name; keep them
out of version control (data/ is ignored by git)."""


def git_state(repo: Path) -> tuple[str, str]:
    """``(commit, dirty)`` of the repository at ``repo``: the output of ``git rev-parse HEAD``, and
    ``"true"`` if ``git status --porcelain`` lists anything, else ``"false"``; ``("unknown",
    "unknown")`` if git is missing or ``repo`` is not a repository."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                capture_output=True, text=True)
    except OSError:
        return UNKNOWN, UNKNOWN
    if head.returncode != 0 or status.returncode != 0 or not head.stdout.strip():
        return UNKNOWN, UNKNOWN
    return head.stdout.strip(), "true" if status.stdout.strip() else "false"


def utc_now() -> str:
    """The current time as ISO 8601 UTC with ``Z``, to the second."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Kalshi markets and trades into SQLite (see the module docstring).",
    )
    parser.add_argument("--series", nargs="+", default=list(ingest.PROTOCOL_SERIES),
                        metavar="TICKER", help="series tickers (default: those of the protocol)")
    parser.add_argument("--start", default=ingest.PROTOCOL_START, metavar="YYYY-MM-DD",
                        help="first day of the release window, UTC (default: %(default)s)")
    parser.add_argument("--end", default=ingest.PROTOCOL_END, metavar="YYYY-MM-DD",
                        help="last day of the release window, UTC (default: %(default)s)")
    parser.add_argument("--db", type=Path, default=Path("data/kalshi.sqlite"),
                        help="SQLite database to write (default: %(default)s)")
    parser.add_argument("--calendar", type=Path, default=Path("data/release_calendar.csv"),
                        help="the hand-checked release calendar (default: %(default)s)")
    parser.add_argument("--calendar-draft", type=Path, default=None, metavar="PATH",
                        help="write a draft calendar to PATH (its name needs 'draft') and stop")
    parser.add_argument("--min-trades", type=int, default=ingest.MIN_TRADES,
                        help="fewest trades before the release for a market to be kept "
                             "(default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="list what would be fetched; request no trade, write nothing")
    return parser


def _write_draft(args: argparse.Namespace, window: ingest.Window, client: KalshiClient,
                 out: Callable[[str], None]) -> int:
    records = []
    for series in args.series:
        listed = ingest.list_series_markets(client, series, window=window)
        out(f"{series}: {len(listed)} market records with an event time within "
            f"{ingest.MARKET_MARGIN_S / 86400.0:g} days of the window")
        records += listed
    if args.dry_run:
        out(f"dry run: a draft calendar would be written to {args.calendar_draft}")
        return 0
    summary = ingest.write_calendar_draft(records, args.calendar_draft)
    out(f"wrote {summary.written} events to {args.calendar_draft}; {summary.unreadable} market "
        f"records could not be read; {len(summary.omitted)} events have no expected expiration "
        f"time and are left out: {', '.join(summary.omitted) or 'none'}")
    out("The draft is not a calendar. Check every release time against the official calendars, "
        "fill in source_url, and save the file without 'draft' in its name.")
    return 0


def _loaded_markets_readonly(path: Path) -> frozenset[str]:
    """The markets of an existing database, without creating or changing anything."""
    if not path.exists():
        return frozenset()
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        try:
            return frozenset(panel.loaded_markets(conn))
        except sqlite3.OperationalError:  # a file without the tables
            return frozenset()


def _dry_run(args: argparse.Namespace, window: ingest.Window, calendar: pd.DataFrame,
             client: KalshiClient, out: Callable[[str], None]) -> int:
    cutoff = client.historical_cutoff()
    loaded = _loaded_markets_readonly(args.db)
    for series in args.series:
        classified = ingest.classify_markets(
            ingest.list_series_markets(client, series), calendar, window, loaded=loaded)
        wanted = [c for c in classified if c.state == ingest.WANTED]
        both = sum(len(ingest.trade_queries(c.release_ts, cutoff.trades_created_ts)) == 2
                   for c in wanted if c.release_ts is not None)
        events = sorted({c.market["event_ticker"] for c in wanted})
        stored = sum(c.state == ingest.ALREADY_LOADED for c in classified)
        skipped = sum(c.state == ingest.OUT_OF_SCOPE for c in classified)
        out(f"{series}: {len(classified)} market records. Markets to fetch: {len(wanted)}, in "
            f"{len(events)} events ({len(wanted) - both} on the historical endpoint only, "
            f"{both} on both). Already in {args.db}: {stored}. Unreadable, outside the calendar "
            f"or outside the window: {skipped}.")
    out(f"dry run: trades created before {ingest.utc_text(cutoff.trades_created_ts)} would be "
        "read from the historical endpoint. No trade was requested and nothing was written.")
    return 0


def _fetch(args: argparse.Namespace, window: ingest.Window, calendar: pd.DataFrame,
           client: KalshiClient, out: Callable[[str], None], repo: Path) -> int:
    args.db.parent.mkdir(parents=True, exist_ok=True)
    total = ingest.BuildReport(min_trades=args.min_trades)
    with closing(sqlite3.connect(args.db)) as conn:
        panel.create_schema(conn)
        commit, dirty = git_state(repo)
        panel.begin_fetch_run(conn, started_utc=utc_now(), git_commit=commit, git_dirty=dirty)
        out(f"fetch recorded: commit {commit}, uncommitted changes: {dirty}")
        try:
            total = _fetch_series(args, window, calendar, client, out, conn, total)
        finally:
            panel.end_fetch_run(conn, finished_utc=utc_now())
    out(f"all series ({len(args.series)}), read in this run:")
    for line in total.summary():
        out(f"  {line}")
    out(f"done; the database is {args.db}")
    return 0


def _fetch_series(args: argparse.Namespace, window: ingest.Window, calendar: pd.DataFrame,
                  client: KalshiClient, out: Callable[[str], None], conn: sqlite3.Connection,
                  total: ingest.BuildReport) -> ingest.BuildReport:
    """Fetch each series in turn and print its counts; returns ``total`` plus their reports."""
    for series in args.series:
        summary = ingest.ingest_series(conn, client, series, calendar, window,
                                       min_trades=args.min_trades, log=out)
        total += summary.report
        out(f"{series}: {summary.listed} market records: {summary.out_of_scope} out of "
            f"scope, {summary.already_loaded} already in the database, {summary.fetched} "
            f"fetched, {summary.written} stored")
        for line in summary.report.summary():
            out(f"  {line}")
        if summary.written + summary.already_loaded == 0:
            out(f"{series}: no market was stored; the series is reported as empty "
                "(docs/protocol.md) and is not replaced")
        if summary.cutoff_moved:
            out(f"warning: the historical cutoff moved during the run (trades created "
                f"before {ingest.utc_text(summary.cutoff_before.trades_created_ts)} became "
                f"{ingest.utc_text(summary.cutoff_after.trades_created_ts)}): trades created "
                "between the two may be missing for markets read after it moved; delete "
                "those markets, or the database, and run again")
    return total


def main(argv: Sequence[str] | None = None, *, client: KalshiClient | None = None,
         out: Callable[[str], None] = print, repo: Path = REPO) -> int:
    """Run the command line; ``client``, ``out`` and ``repo`` (whose git state a fetch records)
    are replaced in the tests. Returns the exit
    status: 0 done, 1 the API failed (what was stored is kept), 2 the arguments, the calendar or
    the database (written by an earlier version of the schema) are not usable."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        window = ingest.Window.from_dates(args.start, args.end)
    except ValueError as error:
        parser.error(f"--start and --end: {error}")
    if args.min_trades < 1:
        parser.error("--min-trades must be at least 1")
    if args.calendar_draft is not None and "draft" not in args.calendar_draft.name.lower():
        parser.error("the file name given to --calendar-draft must contain 'draft'")
    out(NOTICE)
    client = client if client is not None else KalshiClient()
    try:
        if args.calendar_draft is not None:
            return _write_draft(args, window, client, out)
        try:
            calendar = ingest.load_release_calendar(args.calendar)
        except (ValueError, OSError) as error:
            out(f"calendar: {error}")
            return 2
        if args.dry_run:
            return _dry_run(args, window, calendar, client, out)
        return _fetch(args, window, calendar, client, out, repo)
    except (ingest.CalendarChangedError, panel.OldSchemaError) as error:
        out(f"stopped: {error}")
        return 2
    except KalshiApiError as error:
        out(f"stopped: {error}. What was stored is kept; run the same command again to resume.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
