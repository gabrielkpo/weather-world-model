"""Lorenz-96 system: trajectories and Lyapunov spectrum from the tangent-linear model.

dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F, with periodic indices. The default
configuration (N = 40, F = 8, RK4 with dt = 0.01, sampled every 0.05 time units) is
chaotic and dissipative; its leading Lyapunov exponent is about 1.7 per time unit.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def lorenz96_rhs(x: FloatArray, forcing: float) -> FloatArray:
    """Time derivative of the Lorenz-96 state along the last axis (batched states allowed)."""
    xp1 = np.roll(x, -1, axis=-1)
    xm1 = np.roll(x, 1, axis=-1)
    xm2 = np.roll(x, 2, axis=-1)
    out: FloatArray = (xp1 - xm2) * xm1 - x + forcing
    return out


def _rk4_step(x: FloatArray, forcing: float, dt: float) -> FloatArray:
    k1 = lorenz96_rhs(x, forcing)
    k2 = lorenz96_rhs(x + 0.5 * dt * k1, forcing)
    k3 = lorenz96_rhs(x + 0.5 * dt * k2, forcing)
    k4 = lorenz96_rhs(x + dt * k3, forcing)
    out: FloatArray = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return out


def _check_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")


def _initial_states(rng: np.random.Generator, shape: tuple[int, ...], forcing: float) -> FloatArray:
    # Small perturbation of the unstable fixed point x_i = F.
    out: FloatArray = forcing + 0.01 * rng.standard_normal(shape)
    return out


def _simulate(
    n_trajectories: int,
    n_steps: int,
    *,
    n: int,
    forcing: float,
    dt: float,
    sample_every: int,
    burn_in: int,
    rng: np.random.Generator,
) -> FloatArray:
    _check_positive(
        n_trajectories=n_trajectories, n_steps=n_steps, dt=dt, sample_every=sample_every
    )
    if n < 4:
        raise ValueError(f"n must be at least 4, got {n}")
    if burn_in < 0:
        raise ValueError(f"burn_in must be non-negative, got {burn_in}")
    x = _initial_states(rng, (n_trajectories, n), forcing)
    for _ in range(burn_in):
        x = _rk4_step(x, forcing, dt)
    out = np.empty((n_trajectories, n_steps, n), dtype=np.float64)
    for t in range(n_steps):
        out[:, t] = x
        for _ in range(sample_every):
            x = _rk4_step(x, forcing, dt)
    return out


def simulate_lorenz96(
    n_steps: int,
    *,
    n: int = 40,
    forcing: float = 8.0,
    dt: float = 0.01,
    sample_every: int = 5,
    burn_in: int = 5000,
    rng: np.random.Generator,
) -> FloatArray:
    """One Lorenz-96 trajectory of shape (n_steps, n).

    Integrated with RK4 at step ``dt``, after ``burn_in`` integration steps onto the
    attractor, and sampled every ``sample_every`` integration steps (0.05 time units
    by default).
    """
    traj: FloatArray = _simulate(
        1,
        n_steps,
        n=n,
        forcing=forcing,
        dt=dt,
        sample_every=sample_every,
        burn_in=burn_in,
        rng=rng,
    )[0]
    return traj


def simulate_lorenz96_batch(
    n_trajectories: int,
    n_steps: int,
    *,
    n: int = 40,
    forcing: float = 8.0,
    dt: float = 0.01,
    sample_every: int = 5,
    burn_in: int = 5000,
    rng: np.random.Generator,
) -> FloatArray:
    """Independent Lorenz-96 trajectories of shape (n_trajectories, n_steps, n).

    Same integration as :func:`simulate_lorenz96`; all trajectories are integrated
    together, each from its own random initial condition.
    """
    return _simulate(
        n_trajectories,
        n_steps,
        n=n,
        forcing=forcing,
        dt=dt,
        sample_every=sample_every,
        burn_in=burn_in,
        rng=rng,
    )


def _tangent(x: FloatArray, v: FloatArray) -> FloatArray:
    """Jacobian of the Lorenz-96 RHS at state ``x`` (shape (n,)) applied to ``v`` (n, k)."""
    xc = x[:, None]
    xp1 = np.roll(xc, -1, axis=0)
    xm1 = np.roll(xc, 1, axis=0)
    xm2 = np.roll(xc, 2, axis=0)
    vp1 = np.roll(v, -1, axis=0)
    vm1 = np.roll(v, 1, axis=0)
    vm2 = np.roll(v, 2, axis=0)
    out: FloatArray = (vp1 - vm2) * xm1 + (xp1 - xm2) * vm1 - v
    return out


def _rk4_tangent_step(
    x: FloatArray, v: FloatArray, forcing: float, dt: float
) -> tuple[FloatArray, FloatArray]:
    """One RK4 step of the state and of the tangent vectors, integrated jointly."""
    k1x = lorenz96_rhs(x, forcing)
    k1v = _tangent(x, v)
    x2 = x + 0.5 * dt * k1x
    k2x = lorenz96_rhs(x2, forcing)
    k2v = _tangent(x2, v + 0.5 * dt * k1v)
    x3 = x + 0.5 * dt * k2x
    k3x = lorenz96_rhs(x3, forcing)
    k3v = _tangent(x3, v + 0.5 * dt * k2v)
    x4 = x + dt * k3x
    k4x = lorenz96_rhs(x4, forcing)
    k4v = _tangent(x4, v + dt * k3v)
    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_new = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return x_new, v_new


def lyapunov_spectrum_lorenz96(
    *,
    n: int = 40,
    forcing: float = 8.0,
    dt: float = 0.01,
    n_steps: int,
    n_exponents: int,
    burn_in: int,
    rng: np.random.Generator,
    qr_every: int = 1,
) -> FloatArray:
    """Leading Lyapunov exponents (1/time units, sorted descending), Benettin method.

    The tangent-linear model is integrated alongside the trajectory for ``n_steps``
    integration steps of size ``dt`` (after ``burn_in`` steps onto the attractor);
    the tangent vectors are re-orthonormalised by QR every ``qr_every`` steps and the
    exponents are the time averages of the logarithms of the diagonal of R.
    """
    _check_positive(n_steps=n_steps, n_exponents=n_exponents, dt=dt, qr_every=qr_every)
    if n < 4:
        raise ValueError(f"n must be at least 4, got {n}")
    if n_exponents > n:
        raise ValueError(f"n_exponents ({n_exponents}) cannot exceed n ({n})")
    x = _simulate(1, 1, n=n, forcing=forcing, dt=dt, sample_every=1, burn_in=burn_in, rng=rng)[0, 0]
    q, _ = np.linalg.qr(rng.standard_normal((n, n_exponents)))
    log_growth = np.zeros(n_exponents)
    for step in range(1, n_steps + 1):
        x, q = _rk4_tangent_step(x, q, forcing, dt)
        if step % qr_every == 0 or step == n_steps:
            q, r = np.linalg.qr(q)
            log_growth += np.log(np.abs(np.diag(r)))
    exponents: FloatArray = np.sort(log_growth / (n_steps * dt))[::-1].copy()
    return exponents


def leading_lyapunov_lorenz96(
    *,
    n: int = 40,
    forcing: float = 8.0,
    dt: float = 0.01,
    n_steps: int,
    burn_in: int,
    rng: np.random.Generator,
    qr_every: int = 1,
) -> float:
    """Largest Lyapunov exponent (1/time units); see :func:`lyapunov_spectrum_lorenz96`."""
    return float(
        lyapunov_spectrum_lorenz96(
            n=n,
            forcing=forcing,
            dt=dt,
            n_steps=n_steps,
            n_exponents=1,
            burn_in=burn_in,
            rng=rng,
            qr_every=qr_every,
        )[0]
    )
