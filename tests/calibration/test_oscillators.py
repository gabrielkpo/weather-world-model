import numpy as np
import pytest

from weather_world_model.calibration import (
    DEFAULT_EIGENVALUES,
    damped_linear_sequences,
    harmonic_oscillator_sequences,
)


def _sorted(ev: np.ndarray) -> np.ndarray:
    return np.sort_complex(np.round(ev, 9))


def test_default_eigenvalues_cover_growth_decay_and_oscillation() -> None:
    moduli = sorted(abs(e) for e in DEFAULT_EIGENVALUES)
    assert moduli[0] < 0.95 and moduli[-1] > 1.0
    assert any(0.98 < m < 1.0 for m in moduli)


def test_least_squares_recovers_true_eigenvalues() -> None:
    eigs = [*DEFAULT_EIGENVALUES, 0.5]
    x, true = damped_linear_sequences(8, 30, eigenvalues=eigs, dim=16, rng=np.random.default_rng(0))
    assert x.shape == (8, 30, 16)
    assert true.shape == (7,)
    x0 = x[:, :-1].reshape(-1, 16)
    x1 = x[:, 1:].reshape(-1, 16)
    a_hat = np.linalg.lstsq(x0, x1, rcond=None)[0].T
    fitted = np.linalg.eigvals(a_hat)
    fitted = fitted[np.abs(fitted) > 1e-6]
    assert fitted.shape == true.shape
    np.testing.assert_allclose(_sorted(fitted), _sorted(true), atol=1e-6)


def test_noise_and_reproducibility() -> None:
    a, _ = damped_linear_sequences(2, 10, dim=8, noise_std=0.1, rng=np.random.default_rng(1))
    b, _ = damped_linear_sequences(2, 10, dim=8, noise_std=0.1, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a, b)


def test_damped_argument_checks() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="dim"):
        damped_linear_sequences(1, 5, dim=5, rng=rng)
    with pytest.raises(ValueError, match="noise_std"):
        damped_linear_sequences(1, 5, dim=6, noise_std=-1.0, rng=rng)
    with pytest.raises(ValueError, match="at least one"):
        damped_linear_sequences(1, 5, eigenvalues=[], dim=6, rng=rng)
    with pytest.raises(ValueError, match="positive"):
        damped_linear_sequences(0, 5, dim=6, rng=rng)


def test_harmonic_preserves_norms_and_is_linear_rotation() -> None:
    x = harmonic_oscillator_sequences(5, 40, n_oscillators=3, dim=10, rng=np.random.default_rng(2))
    assert x.shape == (5, 40, 10)
    norms = np.linalg.norm(x, axis=-1)
    np.testing.assert_allclose(norms, np.broadcast_to(norms[:, :1], norms.shape), rtol=1e-10)
    # Each sequence follows x_{t+1} = A x_t with an orthogonal A; the reverse follows A^T.
    for seq in x:
        a_hat = np.linalg.lstsq(seq[:-1], seq[1:], rcond=None)[0].T
        np.testing.assert_allclose(seq[1:], seq[:-1] @ a_hat.T, atol=1e-8)
        np.testing.assert_allclose(seq[:-1], seq[1:] @ a_hat, atol=1e-8)


def test_harmonic_forward_and_reversed_have_same_statistics() -> None:
    x = harmonic_oscillator_sequences(
        4000, 16, n_oscillators=2, dim=6, rng=np.random.default_rng(3)
    )
    rev = x[:, ::-1]

    def stats(seq: np.ndarray) -> np.ndarray:
        inc = np.diff(seq, axis=1).reshape(-1, seq.shape[-1])
        lag = np.einsum("ntd,nte->de", seq[:, :-1], seq[:, 1:]) / (seq.shape[0] * 15)
        return np.concatenate([inc.mean(0), (inc**2).mean(0), (inc**3).mean(0), lag.ravel()])

    fwd, bwd = stats(x), stats(rev)
    scale = np.abs(fwd).max()
    np.testing.assert_allclose(fwd, bwd, atol=0.05 * scale)
    # The antisymmetric (rotation-direction) part of the lag-1 covariance averages out.
    lag = fwd[-36:].reshape(6, 6)
    assert np.abs(lag - lag.T).max() < 0.1 * np.abs(lag).max()
