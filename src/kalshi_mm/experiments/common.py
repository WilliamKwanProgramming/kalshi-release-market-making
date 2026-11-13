"""Shared helpers for the experiments: run manifests and Monte Carlo summaries.

Every experiment writes ``<name>.manifest.txt`` next to its CSV files: the command, the seeds, the
git commit and whether the code was modified, the Python and library versions and the runtime.
The command is stored with paths relative to the repository root, and no absolute path of the
machine that ran it appears in the file.
"""

from __future__ import annotations

import datetime
import os
import platform
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from numpy.typing import ArrayLike

__all__ = ["REPO", "Manifest", "git_state", "mc_mean_se", "mean_se"]

REPO = Path(__file__).resolve().parents[3]


def git_state(repo: Path = REPO) -> tuple[str, str]:
    """``(commit hash, "clean" or "dirty")`` of the repository, or ``("unknown", "unknown")``.

    "dirty" means that a file under ``src``, ``scripts`` or ``pyproject.toml`` differs from the
    commit (an untracked file there counts): results written from such a tree cannot be
    reproduced from the commit alone.
    """
    git = ["git", "-C", str(repo)]
    try:
        commit = subprocess.run(
            [*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        changed = subprocess.run(
            [*git, "status", "--porcelain", "--", "src", "scripts", "pyproject.toml"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "unknown"
    return commit, "dirty" if changed else "clean"


def _relative_path(text: str, repo: Path) -> str:
    """An absolute path inside ``repo`` becomes relative to it; any other absolute path is
    shortened to ``.../<last component>``; a relative path is returned unchanged."""
    if not os.path.isabs(text):
        return text
    path = Path(os.path.normpath(text))
    try:
        return path.relative_to(repo).as_posix()
    except ValueError:
        return f".../{path.name}"


def _relative_argument(arg: str, repo: Path) -> str:
    """Apply :func:`_relative_path` to a command-line token, also as ``--flag=/abs/path``."""
    if arg.startswith("-") and "=" in arg:
        flag, _, value = arg.partition("=")
        return f"{flag}={_relative_path(value, repo)}"
    return _relative_path(arg, repo)


class Manifest:
    """Collects what is needed to reproduce one experiment run and writes it next to the CSVs.

    The command is recorded as ``python -m <module> <arguments>``; the arguments are those of the
    process (``sys.argv[1:]``) unless ``args`` is given, with absolute paths made relative to the
    repository root. ``extra`` holds further ``"key: value"`` lines, written before ``outputs``.
    """

    def __init__(self, module: str, seeds: str | int, args: Sequence[str] | None = None):
        self.module = module
        self.seeds = str(seeds)
        self.extra: list[str] = []
        self.args = list(sys.argv[1:] if args is None else args)
        self._t0 = time.perf_counter()
        self.started = datetime.datetime.now(datetime.UTC)
        self.load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")

    def command(self) -> str:
        """``python -m <module> <arguments>`` with paths relative to the repository root."""
        args = [_relative_argument(a, REPO) for a in self.args]
        return " ".join(["python", "-m", self.module, *args])

    def write(self, out_dir: Path, name: str, outputs: Iterable[str], notes: str = "") -> Path:
        """Write ``<out_dir>/<name>.manifest.txt`` and return its path."""
        commit, tree = git_state()
        lines = [
            f"experiment: {name}",
            f"command: {self.command()}",
            f"seeds: {self.seeds}",
            f"git commit: {commit} (working tree {tree})",
            f"python {platform.python_version()}, numpy {np.__version__}, "
            f"scipy {scipy.__version__}, pandas {pd.__version__}",
            f"machine: {platform.system()} {platform.machine()}, "
            f"{os.cpu_count() or 'unknown'} logical CPUs, "
            f"load average at start {self.load:.2f}",
            f"started: {self.started:%Y-%m-%dT%H:%M:%SZ}",
            f"runtime: {time.perf_counter() - self._t0:.1f} s",
            *self.extra,
            "outputs: " + ", ".join(outputs),
        ]
        if notes:
            lines.append(f"notes: {notes}")
        text = "\n".join(lines) + "\n"
        root = str(REPO)  # a path of this checkout that slipped into a note or an extra line
        text = text.replace(root + os.sep, "").replace(root, ".")
        path = Path(out_dir) / f"{name}.manifest.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path


def mean_se(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    """Mean and Monte Carlo standard error (sd / sqrt(count)) of ``cols`` within groups.

    The standard errors are in columns ``<col>_se``; ``n_rep`` is the number of rows per group.
    Groups keep the order of their first appearance.
    """
    g = df.groupby(by, sort=False)[cols]
    mean = g.mean()
    se = g.std(ddof=1) / np.sqrt(g.count())
    out = mean.join(se.add_suffix("_se"))
    out["n_rep"] = g.size()
    return out.reset_index()


def mc_mean_se(x: ArrayLike) -> tuple[float, float]:
    """Mean of replications and its Monte Carlo standard error ``sd / sqrt(n)`` (``ddof = 1``).

    The standard error is NaN for a single replication; NaN replications give NaN (they are not
    dropped, so the caller has to decide what an undefined replication means).
    """
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if a.size == 0:
        return float("nan"), float("nan")
    se = float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan")
    return float(np.mean(a)), se
