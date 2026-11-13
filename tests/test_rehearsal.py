"""The rehearsal runner (kalshi_mm.experiments.rehearsal) on a very small world."""

from __future__ import annotations

import re

import pytest

from kalshi_mm.experiments import rehearsal

SMALL = ["n_boot=100", "recal_min_releases=10", "recal_refit_every=4", "quote.min_history_events=3"]


def test_the_rehearsal_writes_tables_for_synthetic_data_and_a_manifest(tmp_path):
    out = tmp_path / "rehearsal"
    code = rehearsal.main(["--out", str(out), "--n-events", "18", "--window-hours", "30",
                           *[f"--set={a}" for a in SMALL]])
    assert code == 0
    text = (out / "tables.md").read_text()
    assert re.findall(r"^### ([A-Z]\d+)\.", text, flags=re.MULTILINE) == [
        f"R{i}" for i in range(1, 9)]
    assert text.count("(synthetic data)") == 8
    manifest = (out / "rehearsal.manifest.txt").read_text()
    assert "world: 18 events of 30.0 hours, seed 20261113" in manifest
    assert "count, releases kept: 18" in manifest
    assert (out / "p3_fits.csv").exists()  # the world is long enough for a price a day ahead


def test_the_rehearsal_world_is_the_one_the_protocol_needs():
    assert rehearsal.N_EVENTS >= 100  # P1 and P2 are not exploratory below 100 releases
    assert rehearsal.WINDOW_HOURS > 24.0  # P3 prices a market a day before its release


def test_the_rehearsal_refuses_an_unknown_setting(tmp_path):
    with pytest.raises(ValueError, match="unknown setting"):
        rehearsal.main(["--out", str(tmp_path / "x"), "--n-events", "5", "--set=nonsense=1"])
