"""The wave worlds of amendment 8: exact wind, the linear Rossby equation, reversibility."""

import numpy as np
import torch

from weather_world_model.worlds.vjepa2_qg import mirror_fields
from weather_world_model.worlds.waves import _modes, stream_function, wave_runs, wind

NX = 32


def _spectral(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = np.fft.fftfreq(NX, d=1.0 / NX)
    kx, ky = k[None, :], k[:, None]
    h = np.fft.fft2(psi)
    dx = np.real(np.fft.ifft2(1j * kx * h))
    dy = np.real(np.fft.ifft2(1j * ky * h))
    lap = np.real(np.fft.ifft2(-(kx**2 + ky**2) * h))
    return dx, dy, lap


def _setup(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    modes = _modes(3)
    return rng.standard_normal(len(modes)), rng.uniform(0, 2 * np.pi, len(modes)), modes


def test_wind_is_the_derivative_of_the_stream_function() -> None:
    amp, ph, modes = _setup()
    t = np.array([0.0, 0.7])
    psi = stream_function(amp, ph, modes, t, nx=NX)
    uv = wind(amp, ph, modes, t, nx=NX)
    for i in range(2):
        dx, dy, _ = _spectral(psi[i])
        assert np.allclose(uv[i, 0], -dy, atol=1e-8)
        assert np.allclose(uv[i, 1], dx, atol=1e-8)


def _residual(psi_fn, t: float, h: float = 1e-4) -> float:  # type: ignore[no-untyped-def]
    """Max of |d/dt (lap psi - F psi) + beta dpsi/dx| relative to |beta dpsi/dx|."""
    a, b = psi_fn(t - h), psi_fn(t + h)
    _, _, lap_a = _spectral(a)
    _, _, lap_b = _spectral(b)
    dq = ((lap_b - 16.0 * b) - (lap_a - 16.0 * a)) / (2 * h)
    dx, _, _ = _spectral(psi_fn(t))
    return float(np.abs(dq + 6.0 * dx).max() / np.abs(6.0 * dx).max())


def test_waves_solve_the_linear_rossby_equation_and_its_mirrored_reversal_too() -> None:
    amp, ph, modes = _setup(1)

    def forward(t: float) -> np.ndarray:
        out: np.ndarray = stream_function(amp, ph, modes, np.array([t]), nx=NX)[0]
        return out

    def mirrored_reversed(t: float) -> np.ndarray:  # psi(-x, y, -t)
        return np.roll(forward(-t)[:, ::-1], 1, axis=1)

    def reversed_only(t: float) -> np.ndarray:  # psi(x, y, -t)
        return forward(-t)

    assert _residual(forward, 0.3) < 1e-6
    assert _residual(mirrored_reversed, 0.3) < 1e-6  # reversible up to a reflection
    assert _residual(reversed_only, 0.3) > 0.1  # plain reversal is not a solution


def test_mirrored_reversed_wind_is_mirror_fields_of_the_reversed_clip() -> None:
    amp, ph, modes = _setup(2)
    t = np.linspace(0, 1, 4)
    psi_mr = np.stack(
        [
            np.roll(stream_function(amp, ph, modes, np.array([-s]), nx=NX)[0][:, ::-1], 1, 1)
            for s in t
        ]
    )
    uv_rev = wind(amp, ph, modes, -t, nx=NX)
    via_mirror = mirror_fields(torch.as_tensor(uv_rev)).numpy()
    for i in range(len(t)):
        dx, dy, _ = _spectral(psi_mr[i])
        mirrored_back = np.roll(via_mirror[i], 1, axis=-1)
        assert np.allclose(mirrored_back[0], -dy, atol=1e-8)
        assert np.allclose(mirrored_back[1], dx, atol=1e-8)


def test_wave_runs_shape_scale_and_damping() -> None:
    rng = np.random.default_rng(0)
    runs = wave_runs(3, 5, rng=rng, nx=NX)
    assert runs.shape == (3, 5, 2, NX, NX) and runs.dtype == np.float32
    assert abs(float(runs[:, 0].std()) - 1.3) < 1e-4
    damped = wave_runs(3, 21, rng=np.random.default_rng(0), nx=NX, damping=0.5)
    energy = (damped.astype(np.float64) ** 2).mean(axis=(0, 2, 3, 4))
    assert np.isclose(energy[-1] / energy[0], np.exp(-2 * 0.5 * 2.0), rtol=1e-6)
    steady = (runs.astype(np.float64) ** 2).mean(axis=(2, 3, 4))
    assert np.allclose(steady, steady[:, :1], rtol=1e-6)  # undamped waves keep their energy
