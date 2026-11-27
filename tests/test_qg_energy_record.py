import importlib.util
import json
from pathlib import Path

_PATH = Path(__file__).parents[1] / "scripts" / "qg_energy_record.py"
_SPEC = importlib.util.spec_from_file_location("qg_energy_record", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
qg_energy_record = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qg_energy_record)


def test_summary_uses_times_from_start() -> None:
    out = qg_energy_record.summary([0.0, 5.0, 10.0, 15.0], [9.0, 1.0, 2.0, 3.0], 5.0)
    assert out == {"mean": 2.0, "min": 1.0, "max": 3.0}
    assert qg_energy_record.summary([0.0, 5.0], [1.0, 2.0], 60.0) is None


def test_short_record_is_written(tmp_path: Path) -> None:
    qg_energy_record.main(["--out", str(tmp_path), "--t-end", "5"])
    record = json.loads((tmp_path / "qg-energy-long.json").read_text())
    assert record["times"] == [0.0, 5.0]
    assert [run["seed"] for run in record["runs"]] == [0, 1]
    assert all(len(run["kinetic_energy"]) == 2 for run in record["runs"])
    assert all(v >= 0.0 for run in record["runs"] for v in run["kinetic_energy"])
