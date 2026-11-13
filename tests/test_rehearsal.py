"""The rehearsal runner (kalshi_mm.experiments.rehearsal) on a very small world."""

from __future__ import annotations

import re

import pytest

from kalshi_mm.experiments import rehearsal
from kalshi_mm.protocol import DEFAULT
from kalshi_mm.world import WorldParams

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


class SpyStopError(Exception):
    """Ends a run as soon as a spy has seen what it needs."""


def stop_at(monkeypatch, name, seen):
    """Replace ``rehearsal.<name>`` by a spy that records its arguments and then raises."""
    def spy(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        raise SpyStopError

    monkeypatch.setattr(rehearsal, name, spy)


def test_the_world_is_built_from_the_options(monkeypatch, tmp_path):
    seen = {}
    stop_at(monkeypatch, "generate_world", seen)
    with pytest.raises(SpyStopError):
        rehearsal.main(["--out", str(tmp_path / "x"), "--n-events", "7", "--window-hours", "26",
                        "--seed", "5"])
    params, seed = seen["args"]
    assert params == WorldParams(n_events=7, window_hours=26.0) and seed == 5


def test_without_options_the_world_is_the_one_of_the_constants(monkeypatch, tmp_path):
    seen = {}
    stop_at(monkeypatch, "generate_world", seen)
    with pytest.raises(SpyStopError):
        rehearsal.main(["--out", str(tmp_path / "x")])
    params, seed = seen["args"]
    assert (rehearsal.N_EVENTS, rehearsal.WINDOW_HOURS) == (200, 30.0)
    assert params == WorldParams(n_events=200, window_hours=30.0)
    assert seed == DEFAULT.seed == 20261113


def test_the_protocol_gets_the_frames_read_back_from_the_database_and_the_settings(
        monkeypatch, tmp_path):
    read, seen = {}, {}
    real_read = rehearsal.read_dataset

    def read_spy(conn):
        read["frames"] = real_read(conn)
        return read["frames"]

    monkeypatch.setattr(rehearsal, "read_dataset", read_spy)
    stop_at(monkeypatch, "run_protocol", seen)
    with pytest.raises(SpyStopError):
        rehearsal.main(["--out", str(tmp_path / "x"), "--n-events", "6", "--set=n_boot=50",
                        "--set=quote.min_history_events=3"])
    tape, markets, events, source, settings = seen["args"]
    assert "frames" in read, "the dataset was not read back from the database"
    assert tape is read["frames"][0] and markets is read["frames"][1]
    assert events is read["frames"][2]  # not the frames of the world: they went through SQLite
    assert not any(str(c).startswith("latent_") for c in tape.columns)
    assert source == "synthetic"  # the label stored with the dataset
    assert settings.n_boot == 50 and settings.quote.min_history_events == 3


def test_the_manifest_and_the_tables_follow_the_seed_and_the_output_directory(tmp_path):
    out = tmp_path / "rehearsal"
    code = rehearsal.main(["--out", str(out), "--n-events", "18", "--window-hours", "30",
                           "--seed", "7", *[f"--set={a}" for a in SMALL]])
    manifest = (out / "rehearsal.manifest.txt").read_text()
    assert code == 0 and "seeds: 7" in manifest
    assert "world: 18 events of 30.0 hours, seed 7, default WorldParams otherwise" in manifest
    assert "--n-events 18 --window-hours 30 --seed 7 --set=n_boot=100" in manifest  # the call
    assert "notes: synthetic data; read back from SQLite" in manifest
    assert (out / "tables.md").exists() and not (tmp_path / "tables.md").exists()
