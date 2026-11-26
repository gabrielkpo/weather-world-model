"""Linear calibration systems: damped/growing oscillators and reversible harmonic oscillators.

Both generators evolve a low-dimensional latent state by a known real linear map built from
rotation-scaling blocks, then embed it in ``dim`` observed dimensions with a random
orthonormal map, so the observed dynamics are ``x_{t+1} = A x_t`` with known eigenvalues.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

DEFAULT_EIGENVALUES: tuple[complex, ...] = (
    1.02 * np.exp(0.30j),  # growing pair, modulus 1.02
    0.90 * np.exp(0.80j),  # decaying pair, modulus 0.90
    0.995 * np.exp(1.50j),  # oscillating near-unit pair, modulus 0.995
)
"""Default calibration eigenvalues; each complex entry stands for itself and its conjugate."""


def _block(eigenvalue: complex) -> FloatArray:
    if eigenvalue.imag == 0:
        return np.array([[eigenvalue.real]])
    r, theta = abs(eigenvalue), np.angle(eigenvalue)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[r * c, -r * s], [r * s, r * c]])


def _block_diagonal(blocks: Sequence[FloatArray]) -> FloatArray:
    k = sum(b.shape[0] for b in blocks)
    out = np.zeros((k, k))
    i = 0
    for b in blocks:
        m = b.shape[0]
        out[i : i + m, i : i + m] = b
        i += m
    return out


def _orthonormal(rng: np.random.Generator, rows: int, cols: int) -> FloatArray:
    """Random matrix with orthonormal columns (Haar-distributed via sign-fixed QR)."""
    q, r = np.linalg.qr(rng.standard_normal((rows, cols)))
    out: FloatArray = q * np.sign(np.diag(r))
    return out


def _check_sizes(n_sequences: int, n_steps: int, latent_dim: int, dim: int) -> None:
    if n_sequences <= 0 or n_steps <= 0:
        raise ValueError("n_sequences and n_steps must be positive")
    if latent_dim == 0:
        raise ValueError("at least one eigenvalue or oscillator is required")
    if dim < latent_dim:
        raise ValueError(f"dim ({dim}) must be at least the latent dimension ({latent_dim})")


def damped_linear_sequences(
    n_sequences: int,
    n_steps: int,
    *,
    eigenvalues: Sequence[complex] = DEFAULT_EIGENVALUES,
    dim: int,
    noise_std: float = 0.0,
    rng: np.random.Generator,
) -> tuple[FloatArray, ComplexArray]:
    """Sequences ``x_{t+1} = A x_t (+ noise)`` from a real linear map with known eigenvalues.

    Each non-real entry of ``eigenvalues`` gives a 2x2 rotation-scaling block (the pair
    ``lambda, conj(lambda)``); each real entry gives a 1x1 block. The block-diagonal map is
    conjugated by a random orthogonal matrix and lifted to ``dim`` dimensions with a random
    orthonormal embedding ``E``: ``A = E Q B Q^T E^T`` acts on the embedded subspace.
    ``dim`` must be at least the latent dimension (2 per pair, 1 per real eigenvalue).
    Optional Gaussian process noise of standard deviation ``noise_std`` is added in the
    latent space at every step, so the observations stay in the embedded subspace.

    Returns the sequences, shape (n_sequences, n_steps, dim), and the true eigenvalues of
    the generating map on that subspace (conjugate pairs listed explicitly).
    """
    blocks = [_block(complex(ev)) for ev in eigenvalues]
    latent_dim = sum(b.shape[0] for b in blocks)
    _check_sizes(n_sequences, n_steps, latent_dim, dim)
    if noise_std < 0:
        raise ValueError(f"noise_std must be non-negative, got {noise_std}")
    q = _orthonormal(rng, latent_dim, latent_dim)
    a_latent = q @ _block_diagonal(blocks) @ q.T
    embed = _orthonormal(rng, dim, latent_dim)

    y = np.empty((n_sequences, n_steps, latent_dim))
    y[:, 0] = rng.standard_normal((n_sequences, latent_dim))
    for t in range(1, n_steps):
        y[:, t] = y[:, t - 1] @ a_latent.T
        if noise_std > 0:
            y[:, t] += noise_std * rng.standard_normal((n_sequences, latent_dim))
    x: FloatArray = y @ embed.T

    true: list[complex] = []
    for ev in eigenvalues:
        c = complex(ev)
        true.extend([c, c.conjugate()] if c.imag != 0 else [c])
    return x, np.asarray(true, dtype=np.complex128)


def harmonic_oscillator_sequences(
    n_sequences: int,
    n_steps: int,
    *,
    n_oscillators: int,
    dim: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Undamped harmonic oscillators (pure rotations, modulus 1), embedded in ``dim`` dims.

    Each sequence has ``n_oscillators`` independent planar rotations with random angular
    step ``s * omega`` (``omega`` uniform in [0.1, 1.0] rad/step, sign ``s`` = +/-1 with
    equal probability) and a random initial state (isotropic Gaussian, hence uniform phase).
    The latent state is embedded with one fixed random orthonormal map shared by all
    sequences, so norms are preserved exactly.

    The dynamics are exactly time-reversible and the ensemble is reversal-invariant:
    reading a sequence backwards gives rotations by ``-s * omega`` from an isotropic state,
    which is a sequence from the same distribution. An arrow-of-time probe trained on this
    data should therefore stay at chance.

    Returns an array of shape (n_sequences, n_steps, dim).
    """
    latent_dim = 2 * n_oscillators
    _check_sizes(n_sequences, n_steps, latent_dim, dim)
    embed = _orthonormal(rng, dim, latent_dim)
    omega = rng.uniform(0.1, 1.0, size=(n_sequences, n_oscillators))
    omega *= rng.choice(np.array([-1.0, 1.0]), size=(n_sequences, n_oscillators))
    z0 = rng.standard_normal((n_sequences, n_oscillators)) + 1j * rng.standard_normal(
        (n_sequences, n_oscillators)
    )
    t = np.arange(n_steps, dtype=np.float64)
    z = z0[:, None, :] * np.exp(1j * omega[:, None, :] * t[None, :, None])
    y = np.stack([z.real, z.imag], axis=-1).reshape(n_sequences, n_steps, latent_dim)
    x: FloatArray = y @ embed.T
    return x
