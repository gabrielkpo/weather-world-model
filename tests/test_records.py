import subprocess

import numpy as np
import pytest

from weather_world_model import records


@pytest.mark.skipif(
    records.git_state()[1] is None, reason="needs a Git checkout (not a source archive)"
)
def test_provenance_in_a_checkout() -> None:
    prov = records.provenance()
    assert set(prov) == {"code_commit", "code_dirty"}
    assert isinstance(prov["code_commit"], str) and len(prov["code_commit"]) == 40
    assert isinstance(prov["code_dirty"], bool)
    assert records.git_commit() == prov["code_commit"]


def test_provenance_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subprocess, "run", fail)
    assert records.provenance() == {"code_commit": "unknown", "code_dirty": None}


def test_dirty_flag_follows_git_status(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = {"rev-parse": "abc\n", "status": " M file.py\n"}
    monkeypatch.setattr(records, "_git", lambda *args: outputs[args[0]])
    assert records.git_state() == ("abc", True)
    outputs["status"] = ""
    assert records.git_state() == ("abc", False)


def test_array_sha256_hashes_float32_values() -> None:
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert records.array_sha256(x) == records.array_sha256(x.astype(np.float32))
    assert records.array_sha256(x) == records.array_sha256(np.asfortranarray(x))
    assert records.array_sha256(x) != records.array_sha256(x + 1e-3)
    assert len(records.array_sha256(x)) == 64


def test_summary() -> None:
    assert records.summary([1.0, 2.0, 6.0]) == {"mean": 3.0, "min": 1.0, "max": 6.0}
