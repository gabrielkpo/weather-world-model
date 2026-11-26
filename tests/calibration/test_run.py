import json
from pathlib import Path

import numpy as np
import pytest

from weather_world_model.calibration import integrate_lorenz96, simulate_lorenz96_batch
from weather_world_model.calibration.run import growth_window, main
from weather_world_model.diagnostics.arrow import arrow_of_time_dataset, time_asymmetry_features


def test_integrate_starts_from_given_state() -> None:
    x0 = simulate_lorenz96_batch(2, 1, rng=np.random.default_rng(0))[:, 0]
    traj = integrate_lorenz96(x0, 3)
    assert traj.shape == (2, 4, 40)
    assert np.array_equal(traj[:, 0], x0)
    assert not np.allclose(traj[:, 1], x0)


def test_asymmetry_features_flip_sign_under_reversal() -> None:
    seqs = np.random.default_rng(1).standard_normal((3, 10, 4))
    fwd = time_asymmetry_features(seqs)
    rev = time_asymmetry_features(seqs[:, ::-1])
    assert fwd.shape == (3, 4 * 3 // 2 + 4)
    assert np.allclose(fwd, -rev)


def test_asymmetry_dataset_labels_match_orientation() -> None:
    seqs = np.random.default_rng(2).standard_normal((50, 6, 3))
    x, y = arrow_of_time_dataset(seqs, np.random.default_rng(3), features="asymmetry")
    base = time_asymmetry_features(seqs)
    assert np.allclose(x[y == 1], base[y == 1])
    assert np.allclose(x[y == 0], -base[y == 0])


def test_growth_window_stops_before_ceiling() -> None:
    curve = np.array([0.001, 0.002, 0.01, 0.05, 0.2, 0.5, 0.9])
    assert growth_window(curve) == (1, 4)
    assert growth_window(np.full(5, 0.01)) == (1, 4)
    with pytest.raises(ValueError, match="smaller epsilon"):
        growth_window(np.array([0.01, 0.2, 0.5, 0.9]))


def test_smoke_run_writes_a_record(tmp_path: Path) -> None:
    main(["--profile", "smoke", "--out", str(tmp_path)])
    record = json.loads((tmp_path / "calibration-smoke.json").read_text())
    assert set(record["verdicts"]) == {"H0-A", "H0-B", "H0-C"}
    assert len(record["runs"]["predictability"]) == 1
    assert record["runs"]["predictability"][0]["reference_lyapunov"] > 0
    assert "growth_preserved_all_seeds" in record["verdicts"]["H0-B"]


def test_stochastic_smoke_run_writes_a_record(tmp_path: Path) -> None:
    from weather_world_model.calibration.run import PROFILES, calibrate_predictability

    main(["--profile", "smoke", "--stochastic", "--out", str(tmp_path)])
    record = json.loads((tmp_path / "calibration-stochastic-smoke.json").read_text())
    verdict = record["verdicts"]["L-S"]
    assert isinstance(verdict["supported"], bool)
    assert record["noise_dim"] == 8
    run = record["runs"]["predictability"][0]
    assert {"sensitivity_eps0.001", "sensitivity_eps0.01", "spread"} <= set(run["stochastic"])
    assert run["stochastic"]["spread"]["normalised_curve"][0] == 0.0
    assert run["stochastic"]["sensitivity_eps0.001"]["normalised_curve"][0] > 0.0
    # The deterministic reference is the H0-A measurement, value for value.
    plain = calibrate_predictability(PROFILES["smoke"], 0)
    assert {k: v for k, v in run.items() if k != "stochastic"} == plain


def test_stochastic_verdict_rule() -> None:
    from weather_world_model.calibration.run import verdict_stochastic

    def one(det: float | None, sens: float | None) -> dict[str, object]:
        return {
            "reference_lyapunov": 1.0,
            "jepa_eps0.001": {"growth_rate": det},
            "stochastic": {
                "sensitivity_eps0.001": {"growth_rate": sens},
                "spread": {"growth_rate": None},
            },
        }

    assert verdict_stochastic([one(0.1, 0.8), one(None, 1.2)])["supported"]
    assert not verdict_stochastic([one(0.1, 0.8), one(0.1, 0.5)])["supported"]  # beyond 30 %
    assert not verdict_stochastic([one(0.9, 0.8)])["supported"]  # not closer than deterministic
    assert not verdict_stochastic([one(0.1, None)])["supported"]
