"""Diagnostic B: growth and decay modes from a linear (DMD) fit of latent dynamics.

The data are centred with the mean of ``z_t`` and optionally projected on the leading
``rank`` principal components. An affine map ``y_next = A y + b`` is then fitted by least
squares; the intercept absorbs the centring so that an exactly linear system is recovered
exactly. Eigenvalues and modes are those of ``A``; mode amplitudes are measured relative to
the fixed point of the affine map, so that each evolves as ``lambda**t``.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DMDResult:
    """Linear operator in reduced coordinates and its spectral decomposition."""

    operator: NDArray[np.float64]  # (r, r)
    eigenvalues: NDArray[np.complex128]  # (r,), sorted by modulus descending
    modes: NDArray[np.complex128]  # (d, r), eigenvectors mapped back to latent space
    r2: float
    mean: NDArray[np.float64]  # (d,)
    basis: NDArray[np.float64]  # (d, r), orthonormal columns
    eigenvectors: NDArray[np.complex128]  # (r, r), in reduced coordinates
    fixed_point: NDArray[np.float64]  # (r,), fixed point of the affine map (reduced coords)


def fit_dmd(
    z_t: NDArray[np.floating], z_next: NDArray[np.floating], *, rank: int | None
) -> DMDResult:
    """Fit ``z_next ~ A z_t`` on pairs of shape ``(M, d)``; see module docstring."""
    x = np.asarray(z_t, dtype=np.float64)
    y = np.asarray(z_next, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"z_t and z_next must share a (M, d) shape, got {x.shape}, {y.shape}")
    d = x.shape[1]
    mean = x.mean(axis=0)
    if rank is None or rank >= d:
        basis = np.eye(d)
    else:
        _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
        basis = vt[:rank].T
    px = (x - mean) @ basis
    py = (y - mean) @ basis
    design = np.hstack([px, np.ones((px.shape[0], 1))])
    coef, *_ = np.linalg.lstsq(design, py, rcond=None)
    operator = coef[:-1].T
    intercept = coef[-1]
    try:
        fixed_point = np.linalg.solve(np.eye(operator.shape[0]) - operator, intercept)
    except np.linalg.LinAlgError:
        fixed_point = np.zeros(operator.shape[0])
    pred = design @ coef
    ss_res = float(np.sum((py - pred) ** 2))
    ss_tot = float(np.sum((py - py.mean(axis=0)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    eigvals, eigvecs = np.linalg.eig(operator)
    order = np.argsort(-np.abs(eigvals), kind="stable")
    eigvals = eigvals[order].astype(np.complex128)
    eigvecs = eigvecs[:, order].astype(np.complex128)
    return DMDResult(
        operator=operator,
        eigenvalues=eigvals,
        modes=basis @ eigvecs,
        r2=r2,
        mean=mean,
        basis=basis,
        eigenvectors=eigvecs,
        fixed_point=fixed_point,
    )


def growth_modes(result: DMDResult, *, tol: float = 0.0) -> NDArray[np.intp]:
    """Indices of eigenvalues with modulus > 1 + tol."""
    return np.flatnonzero(np.abs(result.eigenvalues) > 1.0 + tol)


def project_on_mode(
    result: DMDResult, z: NDArray[np.floating], index: int
) -> NDArray[np.complex128]:
    """Complex amplitude of mode ``index`` for each latent of ``z`` ``(T, d)``, shape ``(T,)``."""
    y = (np.asarray(z, dtype=np.float64) - result.mean) @ result.basis - result.fixed_point
    coeffs = np.linalg.solve(result.eigenvectors, y.T.astype(np.complex128))
    return np.asarray(coeffs[index], dtype=np.complex128)
