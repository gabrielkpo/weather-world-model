"""Diagnostic A: divergence of nearby latent trajectories."""

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
GROWTH_CEILING = 0.3  # growth window ends before the normalised separation reaches this
ENSEMBLE_MEMBERS = 8  # members of the ensemble whose spread is reported (protocol amendment 3)


def saturation_distance(
    latents: NDArray[np.floating], rng: np.random.Generator, *, n_pairs: int = 1000
) -> float:
    """Median L2 distance between latents of random unrelated pairs; ``latents`` is ``(M, d)``."""
    m = latents.shape[0]
    if m < 2:
        raise ValueError("need at least two latents")
    i = rng.integers(0, m, size=n_pairs)
    j = (i + rng.integers(1, m, size=n_pairs)) % m  # j != i
    return float(np.median(np.linalg.norm(latents[i] - latents[j], axis=-1)))


def separation_curve(
    encode_fn: Callable[[FloatArray], FloatArray],
    rollout_fn: Callable[[FloatArray, int], FloatArray],
    frames: NDArray[np.floating],
    *,
    epsilon: float,
    k: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Median over the batch of the L2 latent separation at each step, shape ``(k+1,)``.

    ``frames`` is ``(B, D)``. Each frame is perturbed with Gaussian noise of std
    ``epsilon * RMS(frame)``; ``encode_fn`` maps ``(B, D) -> (B, d)`` and ``rollout_fn`` maps
    ``((B, d), k) -> (B, k+1, d)``.
    """
    x = np.asarray(frames, dtype=np.float64)
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True))
    x_pert = x + epsilon * rms * rng.standard_normal(x.shape)
    traj = rollout_fn(encode_fn(x), k)
    traj_pert = rollout_fn(encode_fn(x_pert), k)
    sep = np.linalg.norm(traj - traj_pert, axis=-1)
    return np.asarray(np.median(sep, axis=0), dtype=np.float64)


def normalised(curve: NDArray[np.floating], saturation: float) -> FloatArray:
    """Separation divided by the saturation distance."""
    if saturation <= 0:
        raise ValueError("saturation must be positive")
    return np.asarray(curve, dtype=np.float64) / saturation


def growth_rate(curve: NDArray[np.floating], *, dt: float, window: tuple[int, int]) -> float:
    """Slope of ``log(curve)`` over steps ``window[0]..window[1]`` (inclusive), divided by dt."""
    a, b = window
    if not 0 <= a < b < len(curve):
        raise ValueError(f"invalid window {window} for a curve of length {len(curve)}")
    steps = np.arange(a, b + 1, dtype=np.float64)
    values = np.asarray(curve[a : b + 1], dtype=np.float64)
    if np.any(values <= 0):
        raise ValueError("curve must be positive over the window")
    slope = np.polyfit(steps, np.log(values), 1)[0]
    return float(slope / dt)


def ensemble_spread_curve(
    sample_rollout_fn: Callable[[FloatArray, int], FloatArray],
    z0: NDArray[np.floating],
    *,
    k: int,
    n_members: int = 8,
) -> FloatArray:
    """Spread of an ensemble started from identical states, shape ``(k+1,)``.

    ``sample_rollout_fn`` maps ``((B, d), k) -> (B, k+1, d)`` and draws fresh noise at each
    call. The spread at a step is the mean L2 distance between pairs of members, so that it
    compares directly with the two-trajectory separation of :func:`separation_curve`; the
    curve is its median over the batch. It is zero at step 0.
    """
    if n_members < 2:
        raise ValueError("need at least two ensemble members")
    start = np.asarray(z0, dtype=np.float64)
    members = np.stack([sample_rollout_fn(start, k) for _ in range(n_members)])
    total = np.zeros(members.shape[1:3])
    for i in range(n_members):
        for j in range(i + 1, n_members):
            total += np.linalg.norm(members[i] - members[j], axis=-1)
    pairs = n_members * (n_members - 1) / 2
    return np.asarray(np.median(total / pairs, axis=0), dtype=np.float64)


def growth_window(norm_curve: NDArray[np.floating]) -> tuple[int, int]:
    """Steps 1 .. last step before the normalised separation reaches ``GROWTH_CEILING``.

    Raises ``ValueError`` when fewer than two steps stay below the ceiling: the separation
    saturates too early for a slope to be fitted, and a smaller perturbation is needed.
    """
    above = np.flatnonzero(norm_curve >= GROWTH_CEILING)
    end = int(above[0]) - 1 if above.size else len(norm_curve) - 1
    if end < 2:
        raise ValueError("separation saturates before step 3; use a smaller epsilon")
    return 1, end


def growth_fit(
    curve: NDArray[np.floating],
    saturation: float,
    *,
    dt: float,
    safe: bool = False,
    stored_curve: Callable[[FloatArray], list[float]] | None = None,
) -> dict[str, Any]:
    """Growth rate of ``curve`` over :func:`growth_window`, as a record entry.

    Returns ``growth_rate``, ``window`` and the normalised curve: under
    ``"normalised_curve"`` (every step) by default, or under
    ``"normalised_curve_subsampled"`` as returned by ``stored_curve``. With ``safe``, a
    curve with no fittable window (or not positive over it) gives a null rate and window
    and an ``"error"`` message instead of raising ``ValueError``.
    """
    norm = normalised(curve, saturation)
    if stored_curve is None:
        stored: dict[str, Any] = {"normalised_curve": [float(v) for v in norm]}
    else:
        stored = {"normalised_curve_subsampled": stored_curve(norm)}
    try:
        window = growth_window(norm)
        rate = growth_rate(curve, dt=dt, window=window)
        fit: dict[str, Any] = {"growth_rate": rate, "window": list(window)}
    except ValueError as err:
        if not safe:
            raise
        fit = {"growth_rate": None, "window": None, "error": str(err)}
    # Key order of the existing records: a subsampled curve first, a full curve last.
    return {**stored, **fit} if stored_curve is not None else {**fit, **stored}
