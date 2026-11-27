"""Provenance and summary helpers shared by everything that writes a JSON record.

Every record names the code it was produced with: ``code_commit`` (``git rev-parse HEAD``)
and ``code_dirty`` (True when ``git status --porcelain`` lists any change, tracked or
untracked, so that the commit alone does not describe the code that ran). Outside a Git
checkout they are reported as ``"unknown"`` and None.

``data_sha256`` fingerprints a training array: the SHA-256 of its values as C-ordered
float32 bytes, taken after standardisation (the array is cast to float32, the precision
the models are trained in). Two runs with the same fingerprint trained on bit-identical
data.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def _git(*args: str) -> str:
    out = subprocess.run(  # noqa: S603 (fixed arguments, no shell)
        ["git", *args],  # noqa: S607 (git is looked up on PATH on purpose)
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def git_state() -> tuple[str, bool | None]:
    """Current commit and whether the working tree has changes.

    ``("unknown", None)`` when Git or the repository is unavailable.
    """
    try:
        commit = _git("rev-parse", "HEAD").strip()
        dirty = bool(_git("status", "--porcelain").strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return commit, dirty


def git_commit() -> str:
    """Current commit, or ``"unknown"``."""
    return git_state()[0]


def provenance() -> dict[str, str | bool | None]:
    """``{"code_commit": ..., "code_dirty": ...}`` for a record (see the module docstring)."""
    commit, dirty = git_state()
    return {"code_commit": commit, "code_dirty": dirty}


def array_sha256(array: NDArray[np.floating]) -> str:
    """SHA-256 of ``array`` as C-ordered float32 bytes (see the module docstring)."""
    data = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(data.tobytes()).hexdigest()


def summary(values: Sequence[float]) -> dict[str, float]:
    """Mean, minimum and maximum of ``values``."""
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}
