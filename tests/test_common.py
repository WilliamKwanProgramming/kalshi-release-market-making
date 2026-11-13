"""Run manifests and Monte Carlo summaries of the experiments (kalshi_mm.experiments.common)."""

import datetime
import platform
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest
import scipy
from scipy import stats
from scipy.special import gammaln

from kalshi_mm.experiments.common import REPO, Manifest, git_state, mc_mean_se, mean_se


def test_manifest_has_every_field_in_order(tmp_path):
    before = datetime.datetime.now(datetime.UTC).replace(microsecond=0)
    m = Manifest("kalshi_mm.experiments.demo", seeds=20261113,
                 args=["--n", "5", "--out", str(REPO / "results" / "demo")])
    m.extra.append("events: 200")
    path = m.write(tmp_path, "demo", ["a.csv", "b.json"], notes="synthetic data only")
    assert path == tmp_path / "demo.manifest.txt"
    lines = path.read_text().splitlines()
    assert lines[0] == "experiment: demo"
    assert lines[1] == "command: python -m kalshi_mm.experiments.demo --n 5 --out results/demo"
    assert lines[2] == "seeds: 20261113"
    assert re.fullmatch(
        r"git commit: ([0-9a-f]{40}|unknown) \(working tree (clean|dirty|unknown)\)", lines[3]
    )
    assert lines[4] == (f"python {platform.python_version()}, numpy {np.__version__}, "
                        f"scipy {scipy.__version__}, pandas {pd.__version__}")
    assert re.fullmatch(rf"machine: {platform.system()} \S+, \S+ logical CPUs, "
                        r"load average at start \S+", lines[5])
    started = re.fullmatch(r"started: (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z", lines[6])
    assert started
    when = datetime.datetime.fromisoformat(started.group(1)).replace(tzinfo=datetime.UTC)
    assert before <= when <= datetime.datetime.now(datetime.UTC)
    assert re.fullmatch(r"runtime: \d+\.\d s", lines[7])
    assert lines[8:] == ["events: 200", "outputs: a.csv, b.json", "notes: synthetic data only"]


def test_manifest_without_notes_ends_with_outputs(tmp_path):
    m = Manifest("kalshi_mm.experiments.demo", seeds="base 7; replication r uses [7, r]", args=[])
    lines = m.write(tmp_path, "x", ["t.csv"]).read_text().splitlines()
    assert lines[1] == "command: python -m kalshi_mm.experiments.demo"
    assert lines[2] == "seeds: base 7; replication r uses [7, r]"
    assert lines[-1] == "outputs: t.csv"


def test_runtime_is_the_time_since_construction(tmp_path):
    m = Manifest("kalshi_mm.experiments.demo", seeds=1, args=[])
    m._t0 -= 12.3  # as if the run had started 12.3 s ago
    text = m.write(tmp_path, "x", []).read_text()
    runtime = float(re.search(r"runtime: (\S+) s", text).group(1))
    assert 12.3 <= runtime < 14.0


def test_manifest_defaults_to_the_process_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["prog", "--reps", "3", "--out", "results/y"])
    manifest = Manifest("kalshi_mm.experiments.demo", seeds=1)
    text = manifest.write(tmp_path, "y", ["a.csv"]).read_text()
    assert "command: python -m kalshi_mm.experiments.demo --reps 3 --out results/y\n" in text


def test_no_absolute_path_of_this_machine_in_the_manifest(tmp_path):
    outside = tmp_path / "cache" / "dir"
    args = [
        "--out", str(REPO / "results" / "demo"),  # absolute, inside the repository
        f"--data={REPO / 'data' / 'x.csv'}",  # flag=value form
        "--scratch", str(outside),  # absolute, outside the repository
        "--up", str(REPO / "results" / ".." / "src"),  # not normalised
        "--root", str(REPO),
        "--relative", "results/z",  # relative paths are kept as they are
        "plain",
    ]
    m = Manifest("kalshi_mm.experiments.demo", seeds=1, args=args)
    m.extra.append(f"input: {REPO}/data/y.csv")
    notes = f"cwd {REPO} and {REPO}/results"
    text = m.write(tmp_path / "out", "demo", ["a.csv"], notes=notes).read_text()
    command = (
        "command: python -m kalshi_mm.experiments.demo --out results/demo --data=data/x.csv "
        "--scratch .../dir --up src --root . --relative results/z plain\n"
    )
    assert command in text
    assert "input: data/y.csv\n" in text
    assert "notes: cwd . and results\n" in text
    assert str(REPO) not in text and str(tmp_path) not in text
    assert "/home/" not in text and "/tmp/" not in text


def test_manifest_creates_the_output_directory(tmp_path):
    target = tmp_path / "a" / "b"
    path = Manifest("kalshi_mm.experiments.demo", seeds=1, args=[]).write(target, "n", [])
    assert path.exists() and path.parent == target


# ------------------------------------------------------------------------------------- git state
def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
                    "-c", "commit.gpgsign=false", *args], cwd=cwd, check=True, capture_output=True)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_state_in_a_temporary_repository(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "README.md").write_text("r\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "first")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert git_state(tmp_path) == (head, "clean")
    (tmp_path / "README.md").write_text("changed\n")  # outside src, scripts, pyproject.toml
    assert git_state(tmp_path) == (head, "clean")
    (tmp_path / "src" / "a.py").write_text("x = 2\n")  # a tracked file under src
    assert git_state(tmp_path) == (head, "dirty")
    _git(tmp_path, "checkout", "--", "src/a.py")
    assert git_state(tmp_path) == (head, "clean")
    (tmp_path / "src" / "new.py").write_text("y = 1\n")  # an untracked file under src
    assert git_state(tmp_path) == (head, "dirty")


def test_git_state_unknown_outside_a_repository(tmp_path):
    assert git_state(tmp_path / "missing") == ("unknown", "unknown")


def test_git_state_of_this_checkout_is_consistent():
    commit, tree = git_state()
    assert (commit == "unknown") == (tree == "unknown")
    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit)
    assert tree in ("clean", "dirty", "unknown")


# ------------------------------------------------------------------------------- MC summaries
def test_mc_mean_se_by_hand_and_against_scipy():
    mean, se = mc_mean_se([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5
    assert se == pytest.approx(np.sqrt(5.0 / 3.0) / 2.0, rel=1e-14)  # sd = sqrt(5/3), n = 4
    x = np.random.default_rng(5).standard_normal(77)
    mean, se = mc_mean_se(x)
    assert mean == pytest.approx(x.mean(), rel=1e-14)
    assert se == pytest.approx(stats.sem(x, ddof=1), rel=1e-12)


def test_mc_mean_se_edge_cases():
    mean, se = mc_mean_se([3.0])
    assert mean == 3.0 and np.isnan(se)
    mean, se = mc_mean_se([])
    assert np.isnan(mean) and np.isnan(se)
    mean, se = mc_mean_se([1.0, np.nan, 3.0])
    assert np.isnan(mean) and np.isnan(se)  # undefined replications are not silently dropped
    assert isinstance(mc_mean_se([1.0, 2.0])[0], float)
    with pytest.raises(ValueError):
        mc_mean_se(np.ones((2, 2)))


def test_mc_mean_se_estimates_the_standard_error_of_the_mean():
    """For n = 25 draws of N(0, 1) the estimate sd / sqrt(n) has expectation c4(n) / sqrt(n), with
    c4(n) = sqrt(2 / (n - 1)) Gamma(n / 2) / Gamma((n - 1) / 2) (the bias of a sample sd); the
    average of 3000 estimates must be within 4 standard errors of it."""
    n, R = 25, 3000
    rng = np.random.default_rng(17)
    ses = np.array([mc_mean_se(rng.standard_normal(n))[1] for _ in range(R)])
    c4 = np.sqrt(2.0 / (n - 1)) * np.exp(gammaln(n / 2) - gammaln((n - 1) / 2))
    assert abs(ses.mean() - c4 / np.sqrt(n)) < 4 * ses.std(ddof=1) / np.sqrt(R)
    # and it is on the scale of the true standard error 1 / 5 (a bug in the divisor would not be)
    assert 0.19 < ses.mean() < 0.205


def test_mean_se_by_group():
    df = pd.DataFrame({"g": ["b", "a", "b", "a", "b"], "v": [1.0, 10.0, 3.0, 14.0, 5.0],
                       "w": [0.0, 1.0, 0.0, 1.0, 3.0]})
    out = mean_se(df, ["g"], ["v", "w"])
    assert list(out["g"]) == ["b", "a"]  # order of first appearance
    b, a = out.iloc[0], out.iloc[1]
    assert b["v"] == pytest.approx(3.0) and b["v_se"] == pytest.approx(2.0 / np.sqrt(3))
    assert a["v"] == pytest.approx(12.0) and a["v_se"] == pytest.approx(np.sqrt(8.0) / np.sqrt(2))
    assert b["w"] == pytest.approx(1.0) and a["w"] == pytest.approx(1.0)
    assert list(out["n_rep"]) == [3, 2]
