"""Files and tables for one run of the protocol analysis (``kalshi_mm.protocol.Results``).

``write_results`` writes, into one directory, ``summary.json`` (every number), ``p1_buckets.csv``,
``p2_releases.csv`` (per release: the sums over its markets, no trade-level data),
``p3_fits.csv`` (one row per refit of the recalibration) and ``tables.md``; ``DEVIATIONS.md`` as
well when a setting differs from docs/protocol.md. Every table starts with
``### <letter><n>. <title> (<data label>)``, where the data label is "synthetic data"
or "real data" from the source stored with the dataset and the letter follows from it (R for
synthetic, K for real), so a table cannot be mistaken for the other kind; a synthetic result is
refused a directory named ``real`` and a real result one named ``rehearsal``. The reading line
under each table only restates numbers of the table.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from kalshi_mm.protocol import PAIRS, PRIMARY_PAIR, Results, data_label

__all__ = ["TABLE_LETTER", "tables_markdown", "write_results"]

TABLE_LETTER = {"synthetic data": "R", "real data": "K"}


def _num(x: Any, digits: int = 2, scale: float = 1.0) -> str:
    """``x * scale`` with thousands separators, or "n/a" for None and non-finite values."""
    if x is None or not math.isfinite(float(x)):
        return "n/a"
    return f"{scale * float(x):,.{digits}f}"


def _interval(ci: Sequence[float] | None, digits: int = 2, scale: float = 1.0) -> str:
    if ci is None:
        return "n/a"
    return f"[{_num(ci[0], digits, scale)}, {_num(ci[1], digits, scale)}]"


def _p(p: float | None) -> str:
    return "n/a" if p is None else f"{p:.4f}"


def _markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def _block(table_id: str, title: str, label: str, reading: str, table: str, caveat: str) -> str:
    return (
        f"### {table_id}. {title} ({label})\n\nReading: {reading}\n\n{table}\n\n"
        f"Does not show: {caveat}\n"
    )


def _effect_cells(effect: Any) -> list[str]:
    """Row cells for one ``release_effect`` result; a string is why it could not be estimated."""
    if not isinstance(effect, dict):
        reason = str(effect).replace("|", "/")
        return [f"not estimable ({reason})", "n/a", "n/a", "n/a"]
    return [
        f"{_num(effect['D'], 2, 100)} {_interval(effect['D_ci'], 2, 100)}",
        _num(effect["mean_near"], 2, 100),
        _num(effect["mean_quiet"], 2, 100),
        f"{effect['n_near']:,} / {effect['n_quiet']:,}",
    ]


def _diff_row(name: str, diff: dict[str, Any] | None, note: str = "") -> list[str]:
    """Row of a paired difference (USD per release); None means no release was evaluated."""
    if diff is None:
        return [name, "n/a", "n/a", "n/a", "0"]
    return [name, _num(diff["mean"], 3), _interval(diff["ci"], 3),
            _p(diff["p_one_sided"]) + note, f"{diff['n_releases']:,}"]


def _sample_table(results: Results, prefix: str, label: str, deviations: Sequence[str]) -> str:
    s, counts = results.settings, results.dataset.counts
    n_rel = counts["releases kept"]
    note = ""
    if n_rel < s.min_releases:
        note += f" With fewer than {s.min_releases} releases, P1 and P2 are exploratory."
    if deviations:
        note += " Departures from docs/protocol.md: " + "; ".join(deviations) + "."
    rows = [[k.capitalize(), f"{v:,}"] for k, v in counts.items()]
    return _block(
        f"{prefix}1", "Sample", label,
        f"{n_rel:,} releases, {counts['markets kept']:,} markets and {counts['trades kept']:,} "
        f"trades enter the analysis.{note}",
        _markdown(["Step", "Count"], rows),
        "whether the markets that the rules removed differ from the ones kept.",
    )


def _p1_tables(results: Results, prefix: str, label: str, boot: str) -> list[str]:
    fam = results.family
    p1 = results.p1["primary"]
    ratio = ("not reported (quiet mean below 0.2 cent)" if p1["ratio"] is None
             else f"{_num(p1['ratio'])} {_interval(p1['ratio_ci'])}")
    rows = [
        ["Mean adverse selection, near (0, 30] minutes (cents per contract)",
         f"{_num(p1['mean_near'], 2, 100)} ({p1['n_near']:,} trades)"],
        ["Mean adverse selection, quiet (over 360 minutes) (cents per contract)",
         f"{_num(p1['mean_quiet'], 2, 100)} ({p1['n_quiet']:,} trades)"],
        ["D = near minus quiet (cents per contract), 95% interval",
         f"{_num(p1['D'], 2, 100)} {_interval(p1['D_ci'], 2, 100)}"],
        ["Ratio near / quiet, 95% interval", ratio],
        ["One-sided p (share of replications with D at or below 0)", _p(p1["p_one_sided"])],
        ["Holm-adjusted p (family P1, P2, P3)", _p(fam["P1"]["p_holm"])],
        ["Verdict", fam["P1"]["verdict"]],
    ]
    first = _block(
        f"{prefix}2", "P1: adverse selection near the release", label,
        f"the 5-minute adverse selection is {_num(p1['mean_near'], 2, 100)} cents per contract "
        f"near the release and {_num(p1['mean_quiet'], 2, 100)} when quiet, so D = "
        f"{_num(p1['D'], 2, 100)} cents {_interval(p1['D_ci'], 2, 100)} ({boot}; "
        f"{p1['n_events']:,} releases).",
        _markdown(["Quantity", "Value"], rows),
        "that D comes from informed trading: it is a difference of markouts measured on the tape.",
    )

    buckets = results.p1["buckets"]
    rows = [[r.label, f"{r.n:,}", f"{r.n_events:,}", _num(r.mean, 2, 100), _num(r.se, 2, 100)]
            for r in buckets.itertuples()]
    second = _block(
        f"{prefix}3", "P1: adverse selection by minutes to release", label,
        "mean 5-minute adverse selection (cents per contract, signed for the taker) in each "
        "bucket of minutes to the release; the standard error treats releases as clusters.",
        _markdown(["Minutes to release", "Trades", "Releases", "Mean (cents)", "SE (cents)"], rows),
        "the jump at the release itself: the markout of a trade in the last 5 minutes is cut at "
        "the release, so it covers less than 5 minutes.",
    )

    rows = [[f"series {k}", *_effect_cells(v)] for k, v in results.p1["by_series"].items()]
    rows += [[f"horizon {int(h // 60)} minute", *_effect_cells(v)]
             for h, v in results.p1["horizons"].items()]
    rows.append(["settlement markout", *_effect_cells(results.p1["settlement"])])
    q = results.settings.quote
    rows.append([f"scaled to a price of 50 cents (mid proxy in [{q.min_mid:g}, {q.max_mid:g}])",
                 *_effect_cells(results.p1["scaled"])])
    near_lo = results.settings.horizon_s / 60.0
    rows.append([f"near window ({near_lo:g}, {results.settings.near[1]:g}] minutes",
                 *_effect_cells(results.p1["uncut"])])
    third = _block(
        f"{prefix}4", "P1: reported, not tested", label,
        "D (cents per contract, 95% interval) for each series, at the other horizons, for the "
        "settlement markout, for the markout divided by 4 m (1 - m) (m is the mid proxy before "
        "the trade, so the unit is cents per contract at a price of 50 cents) and with a near "
        "window that never cuts the horizon; none of these is part of the tested family.",
        _markdown(["Slice", "D [interval]", "Near mean", "Quiet mean", "Trades near / quiet"],
                  rows),
        "a test: the slices are listed in docs/protocol.md, but only the 5-minute D is tested. "
        "A level of the price and a surge of the trade rate both change D without a change of the "
        "informed share, which is why the scaled slice is shown.",
    )
    return [first, second, third]


def _p2_tables(results: Results, prefix: str, label: str, boot: str) -> list[str]:
    p2 = results.p2
    rows = []
    for policy, v in p2["totals"].items():
        per_contract = "n/a" if not v["contracts"] else _num(v["pnl"] / v["contracts"], 2, 100)
        rows.append([policy, _num(v["pnl"], 3), _num(v["gross"], 3), _num(v["fees"], 3),
                     _num(v["fills"], 1), _num(v["contracts"], 1), per_contract,
                     _num(v["quotes_live_share"], 2)])
    first = _block(
        f"{prefix}5", "P2: the four quoting policies", label,
        "mean over the evaluated releases of the sum over each release's markets; net P&L is "
        "settled at the outcome and after maker fees.",
        _markdown(["Policy", "Net P&L (USD)", "Gross (USD)", "Fees (USD)", "Fills", "Contracts",
                   "Net cents per contract filled", "Share of intervals quoted"], rows),
        "queue position, market impact and the rest of the book: a fill is a strict trade-through "
        "of a tape that does not react to the quotes.",
    )

    primary = p2["primary"]
    rows = [
        _diff_row(f"{a} minus {b}" + (" (primary)" if (a, b) == PRIMARY_PAIR else ""),
                  p2["pairs"][f"{a} - {b}"], " (tested)" if (a, b) == PRIMARY_PAIR else "")
        for a, b in PAIRS
    ]
    head = f"{PRIMARY_PAIR[0]} minus {PRIMARY_PAIR[1]}"
    for source in (p2["sensitivity"], p2["halves"]):
        rows += [_diff_row(f"{head}, {k}", v) for k, v in source.items()]
    rows += [_diff_row(f"{head}, series {k}", v) for k, v in p2["by_series"].items()]
    reading = (
        "no release was evaluated, so the primary difference has no value." if primary is None
        else f"{head} is {_num(primary['mean'], 3)} USD per release "
             f"{_interval(primary['ci'], 3)} over {primary['n_releases']:,} evaluated releases "
             f"({boot})."
    )
    second = _block(
        f"{prefix}6", "P2: net P&L difference per release (USD)", label, reading,
        _markdown(["Difference", "Mean (USD)", "95% interval", "One-sided p", "Releases"], rows),
        "which policy earns money: a difference can be positive while both policies lose.",
    )
    return [first, second]


def _p3_table(results: Results, prefix: str, label: str, boot: str) -> str:
    s, p3, fam = results.settings, results.p3, results.family
    r = p3["primary"]
    missing = f"Markets without a price at {s.recal_hours:g} hours before the release"
    if r is None:
        why = (f"it failed: {p3['error']}" if p3["error"]
               else f"it needs more than {s.recal_min_releases} releases with a price before the "
                    "first prediction")
        reading = f"the recalibration was not computed: {why}."
        rows = [["Markets", f"{p3['n_markets']:,}"], [missing, f"{p3['n_without_price']:,}"]]
    else:
        reading = (f"BSS = {_num(r['bss'], 4)} {_interval(r['ci'], 4)} over {r['n']:,} markets "
                   f"({boot}); positive means that the recalibration helps out of sample.")
        rows = [
            ["BSS", f"{_num(r['bss'], 4)} {_interval(r['ci'], 4)}"],
            ["Brier score, raw / recalibrated",
             f"{_num(r['brier_raw'], 4)} / {_num(r['brier_recal'], 4)}"],
            ["One-sided p (share of replications with BSS at or below 0)",
             _p(r["p_one_sided"])],
            ["Holm-adjusted p (family P1, P2, P3)", _p(fam["P3"]["p_holm"])],
            ["Verdict", fam["P3"]["verdict"]],
            ["Markets predicted / releases", f"{r['n']:,} / {r['n_clusters']:,}"],
            [missing, f"{p3['n_without_price']:,}"],
        ]
        walk = p3["walk"]
        first, last = walk.iloc[0], walk.iloc[-1]
        rows += [
            ["Coefficients (a, b) of the first fit / the last fit",
             f"({_num(first['a'], 3)}, {_num(first['b'], 3)}) / "
             f"({_num(last['a'], 3)}, {_num(last['b'], 3)})"],
            ["Markets in the first fit / the last fit", f"{int(first['n_fit']):,} / "
             f"{int(last['n_fit']):,}"],
        ]
        for k, v in p3["by_series"].items():
            cell = (f"{_num(v['bss'], 4)} {_interval(v['ci'], 4)}" if isinstance(v, dict)
                    else f"not estimable ({str(v).replace('|', '/')})")
            rows.append([f"BSS, series {k}", cell])
    return _block(
        f"{prefix}7", "P3: walk-forward recalibration", label, reading,
        _markdown(["Quantity", "Value"], rows),
        "whether a trader could have used the recalibration: the prices are mid proxies from "
        "prints, not prices that anyone could have traded at.",
    )


def tables_markdown(results: Results, extra_deviations: Sequence[str] = ()) -> str:
    """The tables of one run as Markdown blocks with the IDs ``<letter>1`` to ``<letter>8``.

    ``extra_deviations`` are departures from docs/protocol.md that are not settings (for
    example a run without the ``real-run`` tag); they are named in the sample table.
    """
    s = results.settings
    label = data_label(results.source)
    prefix = TABLE_LETTER[label]
    boot = f"cluster bootstrap over releases, {s.n_boot:,} replications, seed {s.seed}"
    rows = [[k, _p(v["p_one_sided"]), _p(v["p_holm"]), v["verdict"]]
            for k, v in results.family.items()]
    family = _block(
        f"{prefix}8", "The tested family", label,
        "P1, P2 and P3 are one family; a test is supported only if its Holm-adjusted p-value is "
        f"below {s.alpha:g}.",
        _markdown(["Test", "One-sided p", "Holm-adjusted p", "Verdict"], rows),
        "the size of any effect: see the intervals in the tables above.",
    )
    parts = [_sample_table(results, prefix, label, [*results.deviations, *extra_deviations])]
    parts += _p1_tables(results, prefix, label, boot)
    parts += _p2_tables(results, prefix, label, boot)
    parts += [_p3_table(results, prefix, label, boot), family]
    return "\n".join(parts)


def _jsonable(x: Any) -> Any:
    """Plain-Python version of ``x`` for json: numpy scalars, tuples and non-finite floats."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, float | np.floating):
        v = float(x)
        return v if math.isfinite(v) else str(v)
    return x


def write_results(
    results: Results, out_dir: Path, *, extra_deviations: Sequence[str] = ()
) -> list[str]:
    """Write the files of the module docstring into ``out_dir``; returns their names.

    ``extra_deviations`` are departures from docs/protocol.md that are not settings; they go to
    ``DEVIATIONS.md`` and into the sample table, like the changed settings do.
    """
    label = data_label(results.source)
    out_dir = Path(out_dir)
    folders = {part.lower() for part in out_dir.resolve().parts}
    if label == "synthetic data" and "real" in folders:
        raise ValueError(f"synthetic results must not go under a directory named real: {out_dir}")
    if label == "real data" and "rehearsal" in folders:
        raise ValueError(f"real results must not go under a directory named rehearsal: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    deviations = [*results.deviations, *extra_deviations]
    summary = {
        "source": results.source,
        "data": data_label(results.source),
        "counts": results.dataset.counts,
        "settings": dataclasses.asdict(results.settings),
        "deviations": deviations,
        "p1": {k: v for k, v in results.p1.items() if k != "buckets"},
        "p2": {k: v for k, v in results.p2.items() if k != "frame"},
        "p3": {k: v for k, v in results.p3.items() if k != "walk"},
        "family": results.family,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True) + "\n"
    )
    results.p1["buckets"].to_csv(out_dir / "p1_buckets.csv", index=False)
    frame = results.p2["frame"]
    keys = ["event", "policy"]
    per_release = (
        frame.groupby(keys)[["pnl", "gross", "fees", "fills", "contracts", "marked_pnl"]].sum()
        .join(frame.groupby(keys)["evaluated"].first())
        .reset_index()
    )
    per_release.to_csv(out_dir / "p2_releases.csv", index=False)
    (out_dir / "tables.md").write_text(tables_markdown(results, extra_deviations))
    names = ["summary.json", "p1_buckets.csv", "p2_releases.csv", "tables.md"]
    walk = results.p3["walk"]
    if walk is not None:
        fits = (
            walk.groupby(["a", "b", "n_fit"], sort=False)
            .agg(first_release_ts=("release_ts", "min"), markets_scored=("price", "size"))
            .reset_index()
            .sort_values("first_release_ts")
        )
        fits.to_csv(out_dir / "p3_fits.csv", index=False)
        names.insert(3, "p3_fits.csv")
    if deviations:
        (out_dir / "DEVIATIONS.md").write_text(
            "# Departures from docs/protocol.md\n\n"
            "Written by the run itself. Every result in this directory is exploratory.\n\n"
            + "".join(f"- {d}\n" for d in deviations)
        )
        names.append("DEVIATIONS.md")
    return names
