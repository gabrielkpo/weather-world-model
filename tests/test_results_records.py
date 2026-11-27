"""Checks on the versioned run records in ``results/`` and the numbers quoted from them.

Two guarantees:

1. Every record ``results/**/*.json`` (except ``*.partial.json``, which are
   in-progress checkpoints) carries its provenance: the code commit (40 hex
   characters), an ISO date, the configuration or profile it ran with, and the
   seeds when it trained models. A top-level ``exploratory: true`` is allowed.
   ``code_dirty`` and ``data_sha256`` are required only for records dated on or
   after ``PROVENANCE_V2_FROM``: older records predate these fields and are
   kept as they were written rather than edited after the fact.
2. A handful of headline numbers quoted in README.md and docs/RESULTS.md still
   match the record they come from. The formatted value is searched in the
   whole file, so rewording or moving a table does not break the check, but
   changing a record without updating the text does.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROVENANCE_V2_FROM = date(2026, 11, 28)

RECORDS = sorted(p for p in RESULTS.rglob("*.json") if not p.name.endswith(".partial.json"))


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict), f"{path}: a record is a JSON object"
    return data


def _trains_models(record: dict[str, Any]) -> bool:
    config = record.get("config", {})
    return "train_steps" in record or (isinstance(config, dict) and "train_steps" in config)


def _has_seeds(record: dict[str, Any]) -> bool:
    config = record.get("config", {})
    keys = list(record) + (list(config) if isinstance(config, dict) else [])
    return any("seed" in key for key in keys)


def test_there_are_records() -> None:
    assert RECORDS, "no record found under results/"


@pytest.mark.parametrize("path", RECORDS, ids=lambda p: str(p.relative_to(RESULTS)))
def test_record_provenance(path: Path) -> None:
    record = _load(path)
    assert re.fullmatch(r"[0-9a-f]{40}", str(record.get("code_commit", ""))), "code_commit"
    when = datetime.fromisoformat(str(record.get("date"))).date()
    assert any(key == "profile" or key.endswith("config") for key in record), "config or profile"
    if _trains_models(record):
        assert _has_seeds(record), "a record that trains models lists its seeds"
    if "exploratory" in record:
        assert record["exploratory"] is True
    if when >= PROVENANCE_V2_FROM:
        assert isinstance(record.get("code_dirty"), bool), "code_dirty"
        assert re.fullmatch(r"[0-9a-f]{64}", str(record.get("data_sha256", ""))), "data_sha256"


def _fmt(x: float, digits: int = 2) -> str:
    # The docs write negative numbers with a Unicode minus sign
    return f"{x:.{digits}f}".replace("-", chr(0x2212))


def _range(values: list[float]) -> str:
    return f"{_fmt(min(values))} to {_fmt(max(values))}"


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def calibration() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = _load(RESULTS / "calibration" / "calibration-full.json")["runs"][
        "predictability"
    ]
    return runs


def test_step1_reference_lyapunov(calibration: list[dict[str, Any]]) -> None:
    refs = [run["reference_lyapunov"] for run in calibration]
    assert _range(refs) == "1.68 to 1.70"
    assert "1.68 to 1.70" in _text("docs/RESULTS.md")


@pytest.mark.parametrize("doc", ["README.md", "docs/RESULTS.md"])
def test_step1_oracle_range(calibration: list[dict[str, Any]], doc: str) -> None:
    oracle = _range([run["oracle_eps0.001"]["growth_rate"] for run in calibration])
    assert oracle in _text(doc), f"oracle range {oracle} not found in {doc}"


@pytest.mark.parametrize("doc", ["README.md", "docs/RESULTS.md"])
def test_step1_jepa_range(calibration: list[dict[str, Any]], doc: str) -> None:
    jepa = _range([run["jepa_eps0.001"]["growth_rate"] for run in calibration])
    assert jepa in _text(doc), f"JEPA range {jepa} not found in {doc}"


def test_step2_wa_truth() -> None:
    truth = _load(RESULTS / "qg" / "qg-stationary.json")["truth"]["predictability"]
    value = _fmt(truth["eps0.001"]["growth_rate"])
    assert f"{value} per time unit" in _text("docs/RESULTS.md")
