import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).parents[1] / "scripts" / "export_demo.py"
_SPEC = importlib.util.spec_from_file_location("export_demo", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
export_demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_demo)


def test_round_sig() -> None:
    assert export_demo.round_sig(123456.0) == 123500.0
    assert export_demo.round_sig(-0.00123456) == -0.001235
    assert export_demo.round_sig(0.0) == 0.0
    assert np.isnan(export_demo.round_sig(float("nan")))
    assert export_demo.rounded(np.array([1.23456, 2.0])) == [1.235, 2.0]


@pytest.mark.parametrize(("dtype", "low", "levels"), [("uint8", 0.0, 255), ("int8", -1.0, 127)])
def test_encode_decode_round_trip(dtype: str, low: float, levels: int) -> None:
    rng = np.random.default_rng(0)
    scale = 3.5
    values = rng.uniform(low * scale, scale, size=(4, 2, 5, 5))
    text = export_demo.encode_array(values, scale, dtype)
    assert isinstance(text, str)
    back = export_demo.decode_array(text, values.shape, scale, dtype)
    assert back.shape == values.shape
    assert np.abs(back - values).max() <= 0.5 * scale / levels + 1e-12
    # Extremes map to the ends of the range, and out-of-range values are clipped.
    ends = export_demo.decode_array(
        export_demo.encode_array(np.array([low * scale, scale, 2 * scale]), scale, dtype),
        (3,),
        scale,
        dtype,
    )
    np.testing.assert_allclose(ends, [low * scale, scale, scale])
    with pytest.raises(ValueError, match="positive"):
        export_demo.encode_array(values, 0.0, dtype)


def test_maps_and_ranges() -> None:
    rng = np.random.default_rng(1)
    winds = {k: rng.standard_normal((3, 2, 8, 8)) for k in ("A", "B")}
    out = export_demo.maps(winds, background=0.5)
    speed = export_demo.decode_array(
        out["worlds"]["A"]["speed_b64"], (3, 8, 8), out["speed_max"], "uint8"
    )
    expected = np.hypot(winds["A"][:, 0] + 0.5, winds["A"][:, 1])
    assert np.abs(speed - expected).max() < out["speed_max"] / 255
    uv = export_demo.decode_array(out["worlds"]["B"]["uv_b64"], (3, 2, 2, 2), out["uv_max"], "int8")
    assert np.abs(uv[:, 1] - winds["B"][:, 1, ::4, ::4]).max() < out["uv_max"] / 127
    record = {
        "physics": {"kinetic_energy": [1.0, 2.0]},
        "divergence": {"truth": [0.001, 0.5]},
        "readout": {"enstrophy": {"truth": [3.0, 4.0], "k1": [2.0, 9.0]}},
        "ensemble": {"kinetic_energy": {"truth": [1.0, 2.0], "members": [[0.5, 1.0], [1.5, 3.0]]}},
    }
    assert export_demo.ranges(record) == {
        "physics.kinetic_energy": [1.0, 2.0],
        "divergence.truth": [0.001, 0.5],
        "readout.enstrophy": [2.0, 9.0],
        "ensemble.kinetic_energy": [0.5, 3.0],
    }
