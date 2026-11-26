import numpy as np
import pytest

from weather_world_model.diagnostics.modes import fit_dmd, growth_modes, project_on_mode


def _rot(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _system(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    block = np.zeros((4, 4))
    block[:2, :2] = 1.05 * _rot(0.3)
    block[2:, 2:] = np.diag([0.9, 0.5])
    p = rng.standard_normal((4, 4))
    a = p @ block @ np.linalg.inv(p)
    expected = np.array([1.05 * np.exp(0.3j), 1.05 * np.exp(-0.3j), 0.9, 0.5], dtype=np.complex128)
    return a, expected


def test_recovers_eigenvalues() -> None:
    a, expected = _system(0)
    rng = np.random.default_rng(1)
    z = rng.standard_normal((200, 4))
    res = fit_dmd(z, z @ a.T, rank=None)
    assert res.r2 == pytest.approx(1.0, abs=1e-10)
    got = res.eigenvalues
    assert np.all(np.diff(np.abs(got)) <= 1e-12)
    for lam in expected:
        assert np.min(np.abs(got - lam)) < 1e-6
    np.testing.assert_array_equal(np.sort(growth_modes(res)), [0, 1])


def test_rank_projection_and_mode_amplitude() -> None:
    rng = np.random.default_rng(2)
    # 2-D dynamics embedded in 6-D latents.
    emb, _ = np.linalg.qr(rng.standard_normal((6, 2)))
    a = 1.1 * _rot(0.2)
    y = [rng.standard_normal(2)]
    for _ in range(60):
        y.append(a @ y[-1])
    z = np.array(y) @ emb.T
    res = fit_dmd(z[:-1], z[1:], rank=2)
    assert res.modes.shape == (6, 2)
    np.testing.assert_allclose(np.abs(res.eigenvalues), [1.1, 1.1], rtol=1e-6)
    amp = project_on_mode(res, z, 0)
    assert amp.shape == (61,)
    ratio = np.abs(amp[-1]) / np.abs(amp[-11])
    assert ratio == pytest.approx(1.1**10, rel=1e-3)


def test_validation() -> None:
    with pytest.raises(ValueError):
        fit_dmd(np.zeros((5, 2)), np.zeros((5, 3)), rank=None)
