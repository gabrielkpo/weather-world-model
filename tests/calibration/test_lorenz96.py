import numpy as np
import pytest

from weather_world_model.calibration import (
    leading_lyapunov_lorenz96,
    lorenz96_rhs,
    lyapunov_spectrum_lorenz96,
    simulate_lorenz96,
    simulate_lorenz96_batch,
)


def test_rhs_fixed_point_and_known_value() -> None:
    x = np.full(40, 8.0)
    np.testing.assert_allclose(lorenz96_rhs(x, 8.0), np.zeros(40))
    x = np.arange(5, dtype=np.float64)
    # i = 0: (x1 - x3) * x4 - x0 + F = (1 - 3) * 4 - 0 + 8 = 0
    # i = 2: (x3 - x0) * x1 - x2 + F = 3 * 1 - 2 + 8 = 9
    out = lorenz96_rhs(x, 8.0)
    assert out.shape == (5,)
    assert out[0] == pytest.approx(0.0)
    assert out[2] == pytest.approx(9.0)


def test_rhs_batched_matches_single() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 40))
    np.testing.assert_allclose(lorenz96_rhs(x, 8.0)[1], lorenz96_rhs(x[1], 8.0))


def test_trajectory_shape_bounded_and_chaotic() -> None:
    traj = simulate_lorenz96(400, burn_in=1000, rng=np.random.default_rng(1))
    assert traj.shape == (400, 40)
    assert np.all(np.isfinite(traj))
    energy = 0.5 * np.sum(traj**2, axis=1)
    assert energy.max() < 40 * 15.0**2
    assert traj.std() > 1.0  # left the fixed point


def test_reproducible_with_same_seed() -> None:
    a = simulate_lorenz96(50, burn_in=200, rng=np.random.default_rng(7))
    b = simulate_lorenz96(50, burn_in=200, rng=np.random.default_rng(7))
    c = simulate_lorenz96(50, burn_in=200, rng=np.random.default_rng(8))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_batch_shape_and_independence() -> None:
    batch = simulate_lorenz96_batch(3, 20, n=10, burn_in=500, rng=np.random.default_rng(2))
    assert batch.shape == (3, 20, 10)
    assert not np.allclose(batch[0], batch[1])


@pytest.mark.parametrize(
    "kwargs",
    [{"n_steps": 0}, {"n": 3}, {"burn_in": -1}, {"dt": 0.0}, {"sample_every": 0}],
)
def test_invalid_arguments(kwargs: dict[str, float]) -> None:
    args: dict[str, object] = {"n_steps": 5, "rng": np.random.default_rng(0), **kwargs}
    with pytest.raises(ValueError, match="must be"):
        simulate_lorenz96(**args)  # type: ignore[arg-type]


def test_spectrum_argument_checks() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="cannot exceed"):
        lyapunov_spectrum_lorenz96(n=5, n_steps=10, n_exponents=6, burn_in=0, rng=rng)
    with pytest.raises(ValueError, match="at least 4"):
        lyapunov_spectrum_lorenz96(n=3, n_steps=10, n_exponents=1, burn_in=0, rng=rng)


def test_leading_exponent_short_run_in_literature_band() -> None:
    lam = leading_lyapunov_lorenz96(n_steps=4000, burn_in=1000, rng=np.random.default_rng(3))
    assert 1.2 < lam < 2.2


def test_spectrum_sorted_and_dissipative_tail() -> None:
    spec = lyapunov_spectrum_lorenz96(
        n=10, n_steps=3000, n_exponents=10, burn_in=1000, rng=np.random.default_rng(4), qr_every=5
    )
    assert np.all(np.diff(spec) <= 0)
    assert spec[0] > 0
    # Phase-space contraction: sum of all exponents equals the divergence, -n.
    assert spec.sum() == pytest.approx(-10.0, abs=0.5)


@pytest.mark.slow
def test_leading_exponent_long_run() -> None:
    lam = leading_lyapunov_lorenz96(n_steps=100_000, burn_in=5000, rng=np.random.default_rng(5))
    assert 1.5 < lam < 1.9
