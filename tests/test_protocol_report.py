"""The wording of the protocol report about P2 (tables R5, R6 and R8)."""

from __future__ import annotations

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
def text():
    world = generate_world(WorldParams(n_events=26, window_hours=8.0), seed=3)
    settings = with_overrides(DEFAULT, SMALL)
    results = run_protocol(world.tape, world.markets, world.events, "synthetic", settings)
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


def test_r6_names_the_primary_pair_and_says_what_the_flat_policy_is(text):
    heading, reading, table, not_shown = block(text, "R6")
    assert heading == "### R6. P2: net P&L difference per release (USD) (synthetic data)"
    assert PRIMARY_PAIR == ("release_aware", "flat_matched")
    assert reading.startswith("Reading: release_aware minus flat_matched is ")
    assert reading.endswith(
        "flat_matched quotes one flat premium equal to the average of the bucket premiums over "
        "the quoting window, so the difference is the effect of when a premium is applied and "
        "not of its average size.")
    rows = [line.split(" | ")[0].removeprefix("| ") for line in table.splitlines()[2:]]
    assert rows[:len(PAIRS)] == [
        f"{a} minus {b}" + (" (primary)" if (a, b) == PRIMARY_PAIR else "") for a, b in PAIRS]
    assert "release_aware minus pooled_premium" in rows  # a reported pair, not the primary one
    assert not_shown == (
        "Does not show: which policy earns money: a difference can be positive while both "
        "policies lose; and the pairs with pooled_premium: the pooled premium is a flat premium "
        "of another size, so the sign of release_aware minus pooled_premium follows the level of "
        "the pooled premium.")


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
        "Does not show: the size of any effect: see the intervals in the tables above; and the "
        "sign of the pooled comparison: it follows the level of the pooled premium.")


def test_no_table_calls_the_pooled_comparison_the_primary_one(text):
    assert "against the pooled premium" not in text
    assert "(primary)" in text and text.count("(primary)") == 1
    assert "pooled_premium (primary)" not in text and "minus pooled_premium (tested)" not in text
    assert re.search(r"\bfour\b", text.lower()) is None  # the policies are five
