"""Linear Rossby waves on the QG domain: a reversible and a damped control world (amendment 8).

The wind of a superposition of free Rossby waves of the equivalent-barotropic equation

    d/dt (lap(psi) - F psi) + beta d(psi)/dx = 0,

with ``psi = sum_k A_k cos(k.x - omega(k) t + phi_k)`` and
``omega(k) = -beta k_x / (|k|^2 + F)``. The equation has no dissipation and is linear, so it is
time-reversible up to a reflection: if ``psi(x, y, t)`` is a solution, so is
``psi(-x, y, -t)``, whose wind is the x-mirrored wind with ``v`` negated
(:func:`weather_world_model.worlds.vjepa2_qg.mirror_fields`). A time-reversed clip is therefore
a valid forward clip of the mirrored world. The damped world multiplies the field by
``exp(-damping t)``, which no reflection undoes.

Defaults match the QG world: a 64 x 64 periodic grid on [0, 2 pi), ``beta = 6``,
``F = 1 / rd^2 = 16``, one snapshot every 0.1 time units, wind standard deviation about 1.3.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _modes(kmax: int) -> NDArray[np.int64]:
    """Integer wavevectors with ``1 <= |k|_inf <= kmax`` and one of each ``(k, -k)`` pair."""
    ks = [
        (kx, ky)
        for kx in range(-kmax, kmax + 1)
        for ky in range(-kmax, kmax + 1)
        if (kx, ky) != (0, 0) and (kx > 0 or (kx == 0 and ky > 0))
    ]
    return np.asarray(ks, dtype=np.int64)


def stream_function(
    amplitudes: FloatArray,
    phases: FloatArray,
    modes: NDArray[np.int64],
    times: FloatArray,
    *,
    nx: int = 64,
    beta: float = 6.0,
    deformation: float = 16.0,
    damping: float = 0.0,
) -> FloatArray:
    """``psi`` of one run, shape ``(T, nx, nx)`` (rows ``y``, columns ``x``)."""
    grid = np.arange(nx) * 2 * np.pi / nx
    x, y = grid[None, :], grid[:, None]
    k2 = (modes**2).sum(axis=1)
    omega = -beta * modes[:, 0] / (k2 + deformation)
    out = np.zeros((len(times), nx, nx))
    for (kx, ky), a, w, p in zip(modes, amplitudes, omega, phases, strict=True):
        spatial = kx * x + ky * y
        for i, t in enumerate(times):
            out[i] += a * np.cos(spatial - w * t + p)
    return out * np.exp(-damping * np.asarray(times))[:, None, None]


def wind(
    amplitudes: FloatArray,
    phases: FloatArray,
    modes: NDArray[np.int64],
    times: FloatArray,
    *,
    nx: int = 64,
    beta: float = 6.0,
    deformation: float = 16.0,
    damping: float = 0.0,
) -> FloatArray:
    """Wind ``(u, v) = (-dpsi/dy, dpsi/dx)`` of one run, shape ``(T, 2, nx, nx)``, exact."""
    grid = np.arange(nx) * 2 * np.pi / nx
    x, y = grid[None, :], grid[:, None]
    k2 = (modes**2).sum(axis=1)
    omega = -beta * modes[:, 0] / (k2 + deformation)
    out = np.zeros((len(times), 2, nx, nx))
    for (kx, ky), a, w, p in zip(modes, amplitudes, omega, phases, strict=True):
        spatial = kx * x + ky * y
        for i, t in enumerate(times):
            s = np.sin(spatial - w * t + p)
            out[i, 0] += a * ky * s  # u = -d/dy cos(.) = ky sin(.)
            out[i, 1] -= a * kx * s  # v = d/dx cos(.) = -kx sin(.)
    return out * np.exp(-damping * np.asarray(times))[:, None, None, None]


def wave_runs(
    n_runs: int,
    snapshots: int,
    *,
    rng: np.random.Generator,
    damping: float = 0.0,
    dt: float = 0.1,
    nx: int = 64,
    kmax: int = 4,
    wind_std: float = 1.3,
) -> NDArray[np.float32]:
    """Independent runs ``(n_runs, snapshots, 2, nx, nx)`` of random wave superpositions.

    Each run draws amplitudes proportional to ``1 / |k|`` times a standard normal and uniform
    phases; the start time is uniform in [0, 100) so runs are at different stages. The wind
    is scaled so its standard deviation at the first snapshot is ``wind_std`` on average.
    """
    modes = _modes(kmax)
    k = np.sqrt((modes**2).sum(axis=1))
    out = np.empty((n_runs, snapshots, 2, nx, nx), dtype=np.float32)
    for r in range(n_runs):
        amp = rng.standard_normal(len(modes)) / k
        phase = rng.uniform(0, 2 * np.pi, len(modes))
        t0 = rng.uniform(0, 100)
        times: FloatArray = t0 + dt * np.arange(snapshots, dtype=np.float64)
        field = (
            wind(amp, phase, modes, times, nx=nx)
            * np.exp(-damping * (times - t0))[:, None, None, None]
        )
        out[r] = field
    scale = wind_std / float(out[:, 0].std())
    return np.asarray(out * scale, dtype=np.float32)
