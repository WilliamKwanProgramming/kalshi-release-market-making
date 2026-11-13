"""The wording of the protocol report about P2 (tables R5, R6 and R8)."""

from __future__ import annotations

import dataclasses
import re

import pytest

from kalshi_mm.protocol import DEFAULT, PAIRS, PRIMARY_PAIR, run_protocol, with_overrides
from kalshi_mm.protocol_report import tables_markdown
from kalshi_mm.quoting import POLICIES
from kalshi_mm.world import WorldParams, generate_world

SMALL = [
    "n_boot=200",
    "recal_min_releases=12",
    "recal_refit_every=4",
    "recal_hours=2",
    "quote.min_history_events=5",
]


@pytest.fixture(scope="module")
def results():
    world = generate_world(WorldParams(n_events=26, window_hours=8.0), seed=3)
    settings = with_overrides(DEFAULT, SMALL)
    return run_protocol(world.tape, world.markets, world.events, "synthetic", settings)


@pytest.fixture(scope="module")
def text(results):
    return tables_markdown(results)


def block(text, table_id):
    """The block ``### <table_id>. ...`` of the tables, as its four paragraphs: heading, Reading,
    table and Does not show."""
    found = re.search(rf"(?ms)^### {table_id}\. .*?(?=^### |\Z)", text)
    assert found, table_id
    heading, reading, table, not_shown = found.group(0).strip().split("\n\n")
    assert reading.startswith("Reading: ") and not_shown.startswith("Does not show: ")
    assert "\n" not in heading and "\n" not in reading and "\n" not in not_shown
    return heading, reading, table, not_shown


def test_every_table_is_a_block_of_four_paragraphs(text):
    for i in range(1, 9):
        block(text, f"R{i}")
    assert text.count("\nReading: ") == 8 and text.count("\nDoes not show: ") == 8


def test_r5_lists_the_five_policies(text):
    heading, _, table, _ = block(text, "R5")
    assert heading == "### R5. P2: the five quoting policies (synthetic data)"
    names = [line.split(" | ")[0].removeprefix("| ") for line in table.splitlines()[2:]]
    assert names == list(POLICIES) and len(names) == 5


def test_r5_gives_the_premium_at_the_fills_of_each_policy(text, results):
    _, reading, table, _ = block(text, "R5")
    lines = table.splitlines()
    assert lines[0].endswith("| Share of intervals quoted | "
                             "Premium at the fills, cents per unit of 4m(1 - m) |")
    assert ("the premium at the fills is the mean over all the contracts filled in the evaluated "
            "releases of the premium per unit of 4m(1 - m) of the quote that was hit, in cents"
            ) in reading
    cells = {line.split(" | ")[0].removeprefix("| "): line.rstrip(" |").split(" | ")[-1]
             for line in lines[2:]}
    for policy, v in results.p2["totals"].items():
        assert cells[policy] == f"{100.0 * v['fill_premium']:,.2f}"
    assert cells["always_on"] == cells["pull_only"] == "0.00"  # no premium, none at the fills
    # the premium policies quote a premium wherever they are filled
    assert all(float(cells[p]) > 0.0 for p in ("pooled_premium", "flat_matched", "release_aware"))


def test_r5_and_r6_show_n_a_where_a_policy_has_no_premium_at_its_fills(results):
    totals = {policy: dict(v) for policy, v in results.p2["totals"].items()}
    totals["flat_matched"]["fill_premium"] = float("nan")
    name = "flat premium scaled to the premium at the fills of release_aware (undefined)"
    sensitivity = {k: v for k, v in results.p2["sensitivity"].items()
                   if not k.startswith("flat premium scaled")} | {name: None}
    changed = dataclasses.replace(
        results, p2={**results.p2, "totals": totals, "sensitivity": sensitivity})
    text = tables_markdown(changed)
    _, _, table, _ = block(text, "R5")
    row = next(line for line in table.splitlines() if line.startswith("| flat_matched | "))
    assert row.rstrip(" |").split(" | ")[-1] == "n/a"
    _, _, table, _ = block(text, "R6")
    row = next(line for line in table.splitlines() if "(undefined)" in line)
    assert row == f"| release_aware minus flat_matched, {name} | n/a | n/a | n/a | 0 |"


def test_r6_names_the_primary_pair_and_says_what_the_flat_policy_is(text):
    heading, reading, table, not_shown = block(text, "R6")
    assert heading == "### R6. P2: net P&L difference per release (USD) (synthetic data)"
    assert PRIMARY_PAIR == ("release_aware", "flat_matched")
    assert reading.startswith("Reading: release_aware minus flat_matched is ")
    assert reading.endswith(
        "flat_matched quotes one flat premium equal to the average of the bucket premiums over "
        "the clock of the quoting window. That is the same premium averaged over the clock and "
        "not the same premium at the fills, where the quotes are hit: the table above has the "
        "premium at the fills of each policy, and one row below scales the flat premium to the "
        "premium at the fills of release_aware.")
    assert "so the difference is the effect of when" not in reading
    rows = [line.split(" | ")[0].removeprefix("| ") for line in table.splitlines()[2:]]
    assert rows[:len(PAIRS)] == [
        f"{a} minus {b}" + (" (primary)" if (a, b) == PRIMARY_PAIR else "") for a, b in PAIRS]
    assert "release_aware minus pooled_premium" in rows  # a reported pair, not the primary one
    scaled = [r for r in rows if "flat premium scaled to the premium at the fills of" in r]
    assert len(scaled) == 1 and re.fullmatch(
        r"release_aware minus flat_matched, flat premium scaled to the premium at the fills of "
        r"release_aware \(F = [0-9.]+\)", scaled[0])
    assert not_shown == (
        "Does not show: which policy earns money: a difference can be positive while both "
        "policies lose; whether the primary difference comes from when the premium is applied "
        "or from the premium at the fills: read it next to the row with the flat premium scaled "
        "to the premium at the fills; and the pairs with pooled_premium: the pooled premium is "
        "a flat premium of another size, so the sign of release_aware minus pooled_premium "
        "follows the level of the pooled premium.")


def test_r8_says_that_p2_is_the_primary_pair_and_the_pooled_comparison_is_not_tested(text):
    _, reading, table, not_shown = block(text, "R8")
    assert reading.startswith("Reading: P1, P2 and P3 are one family; a test is supported only "
                              "if its Holm-adjusted p-value is below 0.05. ")
    assert reading.endswith(
        "P2 is release_aware minus flat_matched; release_aware minus pooled_premium is one of "
        "the other pairs of the P2 table and is not tested.")
    assert [line.split(" | ")[0].removeprefix("| ") for line in table.splitlines()[2:]] == [
        "P1", "P2", "P3"]
    assert not_shown == (
        "Does not show: the size of any effect: see the intervals in the tables above; that a "
        "supported P2 shows informed trading or that a premium by minutes to the release pays: "
        "on the synthetic worlds the primary difference is positive with no informed share "
        "(table B1 of the README) and not positive at an equal premium at the fills (table B10); "
        "and the sign of the pooled comparison: it follows the level of the pooled premium.")


def test_no_table_calls_the_pooled_comparison_the_primary_one(text):
    assert "against the pooled premium" not in text
    assert "(primary)" in text and text.count("(primary)") == 1
    assert "pooled_premium (primary)" not in text and "minus pooled_premium (tested)" not in text
    assert re.search(r"\bfour\b", text.lower()) is None  # the policies are five
