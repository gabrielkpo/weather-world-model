"""Data of the step-2 QG runs: profiles, simulation, splits, standardisation, mirroring.

Independent QG runs are split by run 70/15/15 (train/val/test); the field is ``"uv"``
(upper-layer eddy winds, ``(2, 64, 64)``), snapshots are taken after spin-up and
standardised per channel with the training statistics. See
:mod:`weather_world_model.worlds.run_qg` for the protocol.
"""

from __future__ import annotations

import resource
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

from weather_world_model.worlds.qg import QGConfig, QGModel, simulate_with_diagnostics

FloatArray = NDArray[np.float64]
STATIONARITY_TOLERANCE = 0.10


def log(message: str) -> None:
    """Progress line on stderr, flushed at once (long runs are watched from a log file)."""
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (macOS reports bytes, Linux kilobytes)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


@dataclass(frozen=True)
class Profile:
    """Sizes of one QG run."""

    n_runs: int
    snapshots: int
    sample_every: int
    spin_up: int
    train_steps: int
    batch_size: int
    conv_channels: tuple[int, ...]
    separation_k: int
    starts_per_test_run: int
    clip_len: int
    clip_stride: int
    seeds: tuple[int, ...]
    data_seed: int = 1000
    # Maximum relative change of the mean kinetic energy between the first and the last third
    # of the training snapshots (protocol amendment 5); None = reported but not enforced.
    stationarity_tolerance: float | None = STATIONARITY_TOLERANCE


PROFILES: dict[str, Profile] = {
    "full": Profile(
        n_runs=80,
        snapshots=100,
        sample_every=20,
        spin_up=12000,
        train_steps=3000,
        batch_size=64,
        conv_channels=(32, 64, 96, 128),
        separation_k=150,
        starts_per_test_run=2,
        clip_len=16,
        clip_stride=8,
        seeds=(0, 1, 2),
    ),
    "smoke": Profile(
        n_runs=7,
        snapshots=24,
        sample_every=10,
        spin_up=300,
        train_steps=4,
        batch_size=8,
        conv_channels=(4, 8),
        separation_k=12,
        starts_per_test_run=2,
        clip_len=8,
        clip_stride=8,
        seeds=(0,),
        stationarity_tolerance=None,  # too short to be stationary: the check is only reported
    ),
}


@dataclass
class QGData:
    """Snapshots ``(N, T, 2, H, W)`` per split, aligned diagnostics and W-A start states."""

    frames: dict[str, NDArray[np.float32]]
    diags: dict[str, FloatArray]
    starts: list[QGModel]
    start_frames: NDArray[np.float32]
    mean: NDArray[np.float32]
    std: NDArray[np.float32]
    sample_dt: float


def split_sizes(n_runs: int) -> tuple[int, int, int]:
    """70/15/15 split by run, each split non-empty."""
    if n_runs < 3:
        raise ValueError("need at least three runs")
    n_val = max(1, round(0.15 * n_runs))
    n_test = max(1, round(0.15 * n_runs))
    return n_runs - n_val - n_test, n_val, n_test


def simulate(profile: Profile, config: QGConfig | None = None) -> QGData:
    """Independent runs, split by run; keeps simulator copies at the W-A start states.

    Snapshots are standardised per channel with the statistics of the training runs.
    """
    cfg = config or QGConfig()
    rng = np.random.default_rng(profile.data_seed)
    sizes = split_sizes(profile.n_runs)
    names = ("train", "val", "test")
    start_idx = np.linspace(0, profile.snapshots - 1, profile.starts_per_test_run + 1)[:-1]
    start_idx = start_idx.astype(int)
    frames: dict[str, NDArray[np.float32]] = {}
    diags: dict[str, FloatArray] = {}
    starts: list[QGModel] = []
    start_frames: list[NDArray[np.float32]] = []
    for name, size in zip(names, sizes, strict=True):

        def keep_start(i: int, t: int, model: QGModel, test: bool = name == "test") -> None:
            if test and t in start_idx:
                starts.append(model.copy())
                start_frames.append(_upper_winds(model))

        frames[name], diags[name], _ = simulate_with_diagnostics(
            cfg,
            n_trajectories=size,
            n_steps=profile.snapshots * profile.sample_every,
            sample_every=profile.sample_every,
            spin_up=profile.spin_up,
            rng=rng,
            field="uv",
            on_snapshot=keep_start,
        )
    mean = frames["train"].mean(axis=(0, 1, 3, 4), keepdims=True)[0, 0]
    std = frames["train"].std(axis=(0, 1, 3, 4), keepdims=True)[0, 0]
    for name in names:
        frames[name] = ((frames[name] - mean) / std).astype(np.float32)
    return QGData(
        frames=frames,
        diags=diags,
        starts=starts,
        start_frames=((np.stack(start_frames) - mean) / std).astype(np.float32),
        mean=mean,
        std=std,
        sample_dt=profile.sample_every * cfg.dt,
    )


def _upper_winds(model: QGModel) -> NDArray[np.float32]:
    u, v = model.velocity()
    return np.stack([u[0], v[0]]).astype(np.float32)


def stationarity_check(
    kinetic_energy: NDArray[np.floating], tolerance: float | None = STATIONARITY_TOLERANCE
) -> dict[str, Any]:
    """Protocol amendment 5: is the recorded flow statistically stationary?

    ``kinetic_energy`` is ``(N, T)``, one row per training run. The mean over runs and
    snapshots of the last third of each run is compared with that of the first third; the
    check passes when the relative difference is below ``tolerance``. With ``tolerance``
    None the difference is reported and ``passes`` is None.
    """
    ke = np.asarray(kinetic_energy, dtype=np.float64)
    if ke.ndim != 2 or ke.shape[1] < 3:
        raise ValueError(f"expected (N, T>=3) kinetic energies, got {ke.shape}")
    third = ke.shape[1] // 3
    first, last = float(ke[:, :third].mean()), float(ke[:, -third:].mean())
    difference = abs(last - first) / abs(first) if first != 0 else float("inf")
    return {
        "quantity": "kinetic_energy",
        "runs": "train",
        "first_third_mean": first,
        "last_third_mean": last,
        "relative_difference": difference,
        "tolerance": tolerance,
        "passes": None if tolerance is None else bool(difference < tolerance),
    }


def mirror_x(frames: NDArray[np.float32], data: QGData) -> NDArray[np.float32]:
    """Standardised ``(u, v)`` frames mirrored in x, with the sign of the physical v flipped.

    The inviscid QG equations are invariant under ``(t, x, v) -> (-t, -x, -v)``, so a clip
    that is mirrored and then reversed keeps the direction of wave propagation (protocol
    amendment 4). ``v`` is flipped in physical units: for a standardised value ``s`` this is
    ``-s - 2 * mean / std``.
    """
    out = frames[..., ::-1].copy()
    shift = 2.0 * float(data.mean[1].item()) / float(data.std[1].item())
    out[..., 1, :, :] = -out[..., 1, :, :] - shift
    return out
