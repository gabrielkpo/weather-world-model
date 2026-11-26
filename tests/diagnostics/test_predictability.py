import numpy as np
import pytest

from weather_world_model.diagnostics.predictability import (
    growth_rate,
    normalised,
    predictability_horizon,
    saturation_distance,
    separation_curve,
)


def test_growth_rate_exact_exponential() -> None:
    curve = 1e-3 * np.exp(0.7 * 0.05 * np.arange(20))
    assert growth_rate(curve, dt=0.05, window=(0, 10)) == pytest.approx(0.7, rel=1e-9)


def test_growth_rate_from_linear_rollout() -> None:
    rng = np.random.default_rng(0)
    d, lam, dt = 6, 1.2, 0.1
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    a = lam * q  # all eigenvalues of modulus 1.2

    def rollout(z0: np.ndarray, k: int) -> np.ndarray:
        out = [z0]
        for _ in range(k):
            out.append(out[-1] @ a.T)
        return np.stack(out, axis=1)

    curve = separation_curve(
        lambda x: x, rollout, rng.standard_normal((16, d)), epsilon=1e-3, k=10, rng=rng
    )
    assert curve.shape == (11,)
    assert growth_rate(curve, dt=dt, window=(0, 10)) == pytest.approx(np.log(lam) / dt, rel=1e-6)


def test_saturation_normalisation_horizon() -> None:
    rng = np.random.default_rng(1)
    s = saturation_distance(rng.standard_normal((200, 3)), rng)
    assert 1.5 < s < 3.0
    n = normalised(np.array([0.1, 1.0, 2.0]), 2.0)
    np.testing.assert_allclose(n, [0.05, 0.5, 1.0])
    assert predictability_horizon(n) == 1
    assert predictability_horizon(n, threshold=2.0) is None


def test_validation() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        saturation_distance(np.zeros((1, 3)), rng)
    with pytest.raises(ValueError):
        normalised(np.ones(3), 0.0)
    with pytest.raises(ValueError):
        growth_rate(np.ones(3), dt=1.0, window=(2, 1))
    with pytest.raises(ValueError):
        growth_rate(np.zeros(3), dt=1.0, window=(0, 2))


def test_ensemble_spread_of_a_random_walk() -> None:
    from weather_world_model.diagnostics.predictability import ensemble_spread_curve

    rng = np.random.default_rng(0)

    def walk(z0: np.ndarray, k: int) -> np.ndarray:
        steps = rng.standard_normal((z0.shape[0], k, z0.shape[1]))
        return np.concatenate([z0[:, None], z0[:, None] + np.cumsum(steps, axis=1)], axis=1)

    curve = ensemble_spread_curve(walk, np.zeros((64, 16)), k=9, n_members=8)
    assert curve.shape == (10,)
    assert curve[0] == 0.0
    # Two independent walks in d dims are sqrt(2 k d) apart after k steps.
    np.testing.assert_allclose(curve[1:], np.sqrt(2 * 16 * np.arange(1, 10)), rtol=0.05)

    def frozen(z0: np.ndarray, k: int) -> np.ndarray:
        return np.repeat(z0[:, None], k + 1, axis=1)

    assert np.all(ensemble_spread_curve(frozen, np.ones((4, 3)), k=5) == 0.0)
    with pytest.raises(ValueError, match="two ensemble members"):
        ensemble_spread_curve(frozen, np.ones((4, 3)), k=5, n_members=1)
