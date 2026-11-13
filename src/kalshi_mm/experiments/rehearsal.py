"""Rehearsal of the real-data run on a synthetic world.

The world of ``kalshi_mm.world`` is written into a SQLite database in the schema of
``kalshi_mm.panel`` (labelled ``synthetic``), read back from it and run through
``kalshi_mm.protocol.run_protocol`` and ``kalshi_mm.protocol_report.write_results``, the same
steps ``kalshi_mm.run_real`` takes for real data. Every table carries the letter R and the label
"synthetic data". The rehearsal shows that the pipeline runs end to end, what its output looks
like and how long it takes. It says nothing about Kalshi: the world has the effects that the
generator plants, nothing else.

    python -m kalshi_mm.experiments.rehearsal --out results/rehearsal
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections.abc import Sequence
from pathlib import Path

from kalshi_mm.experiments.common import REPO, Manifest
from kalshi_mm.panel import create_schema, get_source, read_dataset, write_dataset
from kalshi_mm.protocol import DEFAULT, run_protocol, with_overrides
from kalshi_mm.protocol_report import write_results
from kalshi_mm.world import WorldParams, generate_world

__all__ = ["N_EVENTS", "main"]

N_EVENTS = 150


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "results" / "rehearsal")
    ap.add_argument("--n-events", type=int, default=N_EVENTS)
    ap.add_argument("--seed", type=int, default=DEFAULT.seed)
    ap.add_argument("--set", dest="assign", action="append", default=[], metavar="NAME=VALUE",
                    help="change a setting (quote.<name> for a quote parameter)")
    args = ap.parse_args(argv)
    manifest = Manifest("kalshi_mm.experiments.rehearsal", args.seed, args=argv)

    world = generate_world(WorldParams(n_events=args.n_events), args.seed)
    conn = sqlite3.connect(":memory:")
    try:
        create_schema(conn)
        write_dataset(conn, world.tape, world.markets, world.events, source="synthetic")
        source = get_source(conn)
        tape, markets, events = read_dataset(conn)
    finally:
        conn.close()
    results = run_protocol(tape, markets, events, source, with_overrides(DEFAULT, args.assign))
    names = write_results(results, args.out)
    manifest.extra += [
        f"world: {args.n_events} events, seed {args.seed}, default WorldParams otherwise",
        *[f"count, {k}: {v}" for k, v in results.dataset.counts.items()],
    ]
    manifest.write(args.out, "rehearsal", [*names, "rehearsal.manifest.txt"],
                   notes="synthetic data; read back from SQLite like the real run")
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
