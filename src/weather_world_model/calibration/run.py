"""Calibration of the three diagnostics on systems with known answers (protocol 3.1 and H0).

Run: ``uv run python -m weather_world_model.calibration.run --out results/calibration``.
Each run writes one JSON record (code commit, date, configuration, per-seed values, verdicts).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from weather_world_model.calibration.lorenz96 import (
    integrate_lorenz96,
    leading_lyapunov_lorenz96,
    simulate_lorenz96_batch,
)
from weather_world_model.calibration.oscillators import (
    DEFAULT_EIGENVALUES,
    damped_linear_sequences,
    harmonic_oscillator_sequences,
)
from weather_world_model.diagnostics.arrow import (
    FeatureSet,
    arrow_of_time_dataset,
    probe_accuracy,
    train_probe,
)
from weather_world_model.diagnostics.modes import fit_dmd
from weather_world_model.diagnostics.predictability import (
    growth_rate,
    normalised,
    saturation_distance,
    separation_curve,
)
from weather_world_model.jepa import JEPAConfig, SequenceJEPA, train_jepa
from weather_world_model.seeding import seed_everything

FloatArray = NDArray[np.float64]
SAMPLE_DT = 0.05  # Lorenz-96 time units between two frames (protocol 3.1)
GROWTH_CEILING = 0.3  # growth window ends before the normalised separation reaches this


@dataclass(frozen=True)
class Profile:
    """Sizes of one calibration run."""

    n_trajectories: int
    n_steps: int
    train_steps: int
    rollout_k: int
    n_test: int
    reference_steps: int
    n_clips: int
    clip_len: int
    linear_sequences: int
    linear_steps: int
    seeds: tuple[int, ...]


PROFILES: dict[str, Profile] = {
    "full": Profile(
        n_trajectories=200,
        n_steps=200,
        train_steps=4000,
        rollout_k=120,
        n_test=256,
        reference_steps=100_000,
        n_clips=2000,
        clip_len=16,
        linear_sequences=400,
        linear_steps=40,
        seeds=(0, 1, 2),
    ),
    "smoke": Profile(
        n_trajectories=8,
        n_steps=40,
        train_steps=30,
        rollout_k=20,
        n_test=16,
        reference_steps=2_000,
        n_clips=80,
        clip_len=8,
        linear_sequences=20,
        linear_steps=12,
        seeds=(0,),
    ),
}


def growth_window(norm_curve: FloatArray) -> tuple[int, int]:
    """Steps 1 .. last step before the normalised separation reaches ``GROWTH_CEILING``.

    Raises ``ValueError`` when fewer than two steps stay below the ceiling: the separation
    saturates too early for a slope to be fitted, and a smaller perturbation is needed.
    """
    above = np.flatnonzero(norm_curve >= GROWTH_CEILING)
    end = int(above[0]) - 1 if above.size else len(norm_curve) - 1
    if end < 2:
        raise ValueError("separation saturates before step 3; use a smaller epsilon")
    return 1, end


def _encoder(model: SequenceJEPA, mean: float, std: float) -> Any:
    def encode(x: FloatArray) -> FloatArray:
        z = model.encode(torch.as_tensor((x - mean) / std, dtype=torch.float32))
        return np.asarray(z.numpy(), dtype=np.float64)

    return encode


def _latent_rollout(model: SequenceJEPA) -> Any:
    def rollout(z0: FloatArray, k: int) -> FloatArray:
        out = model.rollout(torch.as_tensor(z0, dtype=torch.float32), k)
        return np.asarray(out.numpy(), dtype=np.float64)

    return rollout


def _growth(curve: FloatArray, saturation: float) -> dict[str, Any]:
    norm = normalised(curve, saturation)
    window = growth_window(norm)
    return {
        "growth_rate": growth_rate(curve, dt=SAMPLE_DT, window=window),
        "window": list(window),
        "normalised_curve": [float(v) for v in norm],
    }


def calibrate_predictability(profile: Profile, seed: int) -> dict[str, Any]:
    """H0-A: growth rate of latent separation vs the leading Lyapunov exponent of Lorenz-96."""
    rng = seed_everything(seed)
    reference = leading_lyapunov_lorenz96(n_steps=profile.reference_steps, burn_in=5000, rng=rng)
    train = simulate_lorenz96_batch(profile.n_trajectories, profile.n_steps, rng=rng)
    mean, std = float(train.mean()), float(train.std())
    cfg = JEPAConfig(input_dim=train.shape[2])
    model, history = train_jepa((train - mean) / std, cfg, n_steps=profile.train_steps, seed=seed)
    frames = simulate_lorenz96_batch(profile.n_test, 1, rng=rng)[:, 0]
    encode = _encoder(model, mean, std)
    result: dict[str, Any] = {
        "reference_lyapunov": reference,
        "collapsed": history.collapsed,
        "final_latent_std": history.latent_std[-1],
    }
    for eps in (1e-3, 1e-2):
        jepa_curve = separation_curve(
            encode, _latent_rollout(model), frames, epsilon=eps, k=profile.rollout_k, rng=rng
        )
        oracle_curve = separation_curve(
            lambda x: x,
            lambda x0, k: integrate_lorenz96(x0, k),
            frames,
            epsilon=eps,
            k=profile.rollout_k,
            rng=rng,
        )
        result[f"jepa_eps{eps:g}"] = _growth(jepa_curve, saturation_distance(encode(frames), rng))
        result[f"oracle_eps{eps:g}"] = _growth(oracle_curve, saturation_distance(frames, rng))
    return result


def _matched(estimated: NDArray[np.complex128], true: list[complex]) -> dict[str, list[Any]]:
    """For each true eigenvalue, the closest estimate's relative modulus error and whether it
    stays on the same side of 1 (growth kept as growth, decay as decay)."""
    errors: list[float] = []
    same_side: list[bool] = []
    for lam in true:
        closest = estimated[np.argmin(np.abs(estimated - lam))]
        errors.append(float(abs(abs(closest) - abs(lam)) / abs(lam)))
        same_side.append(bool((abs(closest) > 1) == (abs(lam) > 1)))
    return {"relative_errors": errors, "growth_preserved": same_side}


def calibrate_modes(profile: Profile, seed: int) -> dict[str, Any]:
    """H0-B: DMD eigenvalues of latent dynamics vs the eigenvalues of the generating map."""
    rng = seed_everything(seed)
    seqs, true_eigs = damped_linear_sequences(
        profile.linear_sequences, profile.linear_steps, dim=20, rng=rng
    )
    true = [complex(v) for v in true_eigs]
    rank = len(true)
    scale = float(seqs.std())
    cfg = JEPAConfig(input_dim=seqs.shape[2])
    model, history = train_jepa(seqs / scale, cfg, n_steps=profile.train_steps, seed=seed)
    z = _encoder(model, 0.0, scale)(seqs.reshape(-1, seqs.shape[2])).reshape(
        seqs.shape[0], seqs.shape[1], -1
    )
    jepa = fit_dmd(z[:, :-1].reshape(-1, z.shape[2]), z[:, 1:].reshape(-1, z.shape[2]), rank=rank)
    oracle = fit_dmd(
        seqs[:, :-1].reshape(-1, seqs.shape[2]), seqs[:, 1:].reshape(-1, seqs.shape[2]), rank=rank
    )
    return {
        "true_moduli": [abs(v) for v in true],
        "collapsed": history.collapsed,
        "jepa": {
            "moduli": [float(abs(v)) for v in jepa.eigenvalues],
            "r2": jepa.r2,
            **_matched(jepa.eigenvalues, true),
        },
        "oracle": {
            "moduli": [float(abs(v)) for v in oracle.eigenvalues],
            "r2": oracle.r2,
            **_matched(oracle.eigenvalues, true),
        },
    }


def _pca(train: FloatArray, dims: int) -> tuple[FloatArray, FloatArray]:
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    return mean, vt[:dims].T


def _probe(
    sequences: FloatArray, rng: np.random.Generator, seed: int, features: FeatureSet
) -> float:
    n_train = int(0.7 * sequences.shape[0])
    mean, basis = _pca(sequences[:n_train], min(8, sequences.shape[-1]))
    reduced = (sequences - mean) @ basis
    x, y = arrow_of_time_dataset(reduced, rng, features=features)
    probe = train_probe(x[:n_train], y[:n_train], l2=1e-3, seed=seed)
    return probe_accuracy(probe, x[n_train:], y[n_train:])


def _clips(trajectories: FloatArray, clip_len: int) -> FloatArray:
    n, t, d = trajectories.shape
    per = t // clip_len
    return trajectories[:, : per * clip_len].reshape(n * per, clip_len, d)


def calibrate_arrow(profile: Profile, seed: int) -> dict[str, Any]:
    """H0-C: arrow-of-time probe on irreversible (Lorenz-96) vs reversible (harmonic) dynamics."""
    rng = seed_everything(seed)
    per_traj = max(1, profile.n_steps // profile.clip_len)
    n_traj = max(2, profile.n_clips // per_traj)
    l96 = _clips(simulate_lorenz96_batch(n_traj, profile.n_steps, rng=rng), profile.clip_len)
    harm = harmonic_oscillator_sequences(
        l96.shape[0], profile.clip_len, n_oscillators=4, dim=l96.shape[2], rng=rng
    )
    result: dict[str, Any] = {}
    for name, data in (("lorenz96", l96), ("harmonic", harm)):
        mean, std = float(data.mean()), float(data.std())
        cfg = JEPAConfig(input_dim=data.shape[2])
        model, _ = train_jepa((data - mean) / std, cfg, n_steps=profile.train_steps, seed=seed)
        latents = _encoder(model, mean, std)(data.reshape(-1, data.shape[2])).reshape(
            data.shape[0], data.shape[1], -1
        )
        result[name] = {
            "jepa_asymmetry": _probe(latents, rng, seed, "asymmetry"),
            "jepa_differences": _probe(latents, rng, seed, "differences"),
            "oracle_asymmetry": _probe(data, rng, seed, "asymmetry"),
        }
    return result


def _summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


def verdicts(runs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Apply the H0 criteria of the protocol to the per-seed results."""
    a_err = [
        abs(r["jepa_eps0.001"]["growth_rate"] - r["reference_lyapunov"]) / r["reference_lyapunov"]
        for r in runs["predictability"]
    ]
    b_err = [max(r["jepa"]["relative_errors"]) for r in runs["modes"]]
    c_l96 = [r["lorenz96"]["jepa_asymmetry"] for r in runs["arrow"]]
    c_harm = [r["harmonic"]["jepa_asymmetry"] for r in runs["arrow"]]
    return {
        "H0-A": {"relative_error": _summary(a_err), "supported": max(a_err) <= 0.30},
        "H0-B": {
            "max_relative_error": _summary(b_err),
            "supported": max(b_err) <= 0.10,
            "growth_preserved_all_seeds": all(
                all(r["jepa"]["growth_preserved"]) for r in runs["modes"]
            ),
        },
        "H0-C": {
            "lorenz96_accuracy": _summary(c_l96),
            "harmonic_accuracy": _summary(c_harm),
            "supported": min(c_l96) > 0.80 and max(c_harm) < 0.60,
        },
    }


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def run(profile_name: str) -> dict[str, Any]:
    """Run the three calibrations for every seed of the profile and apply the criteria."""
    profile = PROFILES[profile_name]
    runs: dict[str, list[dict[str, Any]]] = {"predictability": [], "modes": [], "arrow": []}
    for seed in profile.seeds:
        runs["predictability"].append(calibrate_predictability(profile, seed))
        runs["modes"].append(calibrate_modes(profile, seed))
        runs["arrow"].append(calibrate_arrow(profile, seed))
    return {
        "code_commit": _git_commit(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "profile": profile_name,
        "config": asdict(profile),
        "default_eigenvalues": [[v.real, v.imag] for v in DEFAULT_EIGENVALUES],
        "torch_version": torch.__version__,
        "runs": runs,
        "verdicts": verdicts(runs),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = run(args.profile)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"calibration-{args.profile}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["verdicts"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
