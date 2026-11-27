"""Calibration of the three diagnostics on systems with known answers (protocol 3.1 and H0).

Run: ``uv run python -m weather_world_model.calibration.run --out results/calibration``.
Each run writes one JSON record (code commit and dirty flag, date, configuration, per-seed
values, verdicts). Every per-seed entry that trains a model carries ``data_sha256``, the
fingerprint of its standardised training array (see :mod:`weather_world_model.records`).

With ``--stochastic`` the run measures hypothesis L-S of protocol amendment 3 instead (a
stochastic predictor on Lorenz-96 next to the deterministic one-step model) and writes
``calibration-stochastic[-<profile>].json``.
"""

from __future__ import annotations

import argparse
import json
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
    ENSEMBLE_MEMBERS,
    ensemble_spread_curve,
    growth_fit,
    saturation_distance,
    separation_curve,
)
from weather_world_model.jepa import (
    NOISE_DIM,
    JEPAConfig,
    SequenceJEPA,
    latent_rollout,
    standardised_encoder,
    train_jepa,
)
from weather_world_model.linalg import pca
from weather_world_model.records import array_sha256, provenance, summary
from weather_world_model.seeding import seed_everything

FloatArray = NDArray[np.float64]
SAMPLE_DT = 0.05  # Lorenz-96 time units between two frames (protocol 3.1)


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


def _growth(curve: FloatArray, saturation: float) -> dict[str, Any]:
    return growth_fit(curve, saturation, dt=SAMPLE_DT)


def _safe_growth(curve: FloatArray, saturation: float) -> dict[str, Any]:
    """As :func:`_growth`, but a curve with no fittable window gives a null rate, not an error."""
    return growth_fit(curve, saturation, dt=SAMPLE_DT, safe=True)


def stochastic_predictability(
    model: SequenceJEPA, frames: FloatArray, mean: float, std: float, profile: Profile, seed: int
) -> dict[str, Any]:
    """Sensitivity (twin rollouts sharing one noise sequence) and ensemble spread of a
    stochastic predictor, with the growth-window rule of diagnostic A."""
    rng = np.random.default_rng([seed, 1])
    generator = torch.Generator().manual_seed(seed)
    k, noise_dim = profile.rollout_k, model.config.noise_dim
    encode = standardised_encoder(model, mean, std)
    shared = torch.randn((frames.shape[0], k, noise_dim), generator=generator)

    def shared_rollout(z0: FloatArray, steps: int) -> FloatArray:
        out = model.rollout(torch.as_tensor(z0, dtype=torch.float32), steps, noise=shared)
        return np.asarray(out.numpy(), dtype=np.float64)

    def sampled_rollout(z0: FloatArray, steps: int) -> FloatArray:
        out = model.rollout(torch.as_tensor(z0, dtype=torch.float32), steps, generator=generator)
        return np.asarray(out.numpy(), dtype=np.float64)

    z0 = encode(frames)
    saturation = saturation_distance(z0, rng)
    result: dict[str, Any] = {"saturation_distance": saturation}
    for eps in (1e-3, 1e-2):
        curve = separation_curve(encode, shared_rollout, frames, epsilon=eps, k=k, rng=rng)
        result[f"sensitivity_eps{eps:g}"] = _safe_growth(curve, saturation)
    spread = ensemble_spread_curve(sampled_rollout, z0, k=k, n_members=ENSEMBLE_MEMBERS)
    result["spread"] = _safe_growth(spread, saturation)
    return result


def calibrate_predictability(
    profile: Profile, seed: int, *, stochastic: bool = False
) -> dict[str, Any]:
    """H0-A: growth rate of latent separation vs the leading Lyapunov exponent of Lorenz-96.

    With ``stochastic``, a second model with a stochastic predictor (same encoder
    architecture, data, steps and seed) is trained and measured under ``"stochastic"``
    (hypothesis L-S); the deterministic values are unchanged by it.
    """
    rng = seed_everything(seed)
    reference = leading_lyapunov_lorenz96(n_steps=profile.reference_steps, burn_in=5000, rng=rng)
    train = simulate_lorenz96_batch(profile.n_trajectories, profile.n_steps, rng=rng)
    mean, std = float(train.mean()), float(train.std())
    cfg = JEPAConfig(input_dim=train.shape[2])
    standardised = (train - mean) / std
    model, history = train_jepa(standardised, cfg, n_steps=profile.train_steps, seed=seed)
    frames = simulate_lorenz96_batch(profile.n_test, 1, rng=rng)[:, 0]
    encode = standardised_encoder(model, mean, std)
    result: dict[str, Any] = {
        "reference_lyapunov": reference,
        "data_sha256": array_sha256(standardised),
        "collapsed": history.collapsed,
        "final_latent_std": history.latent_std[-1],
    }
    for eps in (1e-3, 1e-2):
        jepa_curve = separation_curve(
            encode, latent_rollout(model), frames, epsilon=eps, k=profile.rollout_k, rng=rng
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
    if stochastic:
        noisy_cfg = JEPAConfig(input_dim=train.shape[2], noise_dim=NOISE_DIM)
        noisy, noisy_history = train_jepa(
            standardised, noisy_cfg, n_steps=profile.train_steps, seed=seed
        )
        result["stochastic"] = {
            "collapsed": noisy_history.collapsed,
            "final_latent_std": noisy_history.latent_std[-1],
            **stochastic_predictability(noisy, frames, mean, std, profile, seed),
        }
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
    standardised = seqs / scale
    model, history = train_jepa(standardised, cfg, n_steps=profile.train_steps, seed=seed)
    z = standardised_encoder(model, 0.0, scale)(seqs.reshape(-1, seqs.shape[2])).reshape(
        seqs.shape[0], seqs.shape[1], -1
    )
    jepa = fit_dmd(z[:, :-1].reshape(-1, z.shape[2]), z[:, 1:].reshape(-1, z.shape[2]), rank=rank)
    oracle = fit_dmd(
        seqs[:, :-1].reshape(-1, seqs.shape[2]), seqs[:, 1:].reshape(-1, seqs.shape[2]), rank=rank
    )
    return {
        "true_moduli": [abs(v) for v in true],
        "data_sha256": array_sha256(standardised),
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


def _probe(
    sequences: FloatArray, rng: np.random.Generator, seed: int, features: FeatureSet
) -> float:
    n_train = int(0.7 * sequences.shape[0])
    mean, basis = pca(sequences[:n_train], min(8, sequences.shape[-1]))
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
        standardised = (data - mean) / std
        model, _ = train_jepa(standardised, cfg, n_steps=profile.train_steps, seed=seed)
        latents = standardised_encoder(model, mean, std)(data.reshape(-1, data.shape[2])).reshape(
            data.shape[0], data.shape[1], -1
        )
        result[name] = {
            "data_sha256": array_sha256(standardised),
            "jepa_asymmetry": _probe(latents, rng, seed, "asymmetry"),
            "jepa_differences": _probe(latents, rng, seed, "differences"),
            "oracle_asymmetry": _probe(data, rng, seed, "asymmetry"),
        }
    return result


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
        "H0-A": {"relative_error": summary(a_err), "supported": max(a_err) <= 0.30},
        "H0-B": {
            "max_relative_error": summary(b_err),
            "supported": max(b_err) <= 0.10,
            "growth_preserved_all_seeds": all(
                all(r["jepa"]["growth_preserved"]) for r in runs["modes"]
            ),
        },
        "H0-C": {
            "lorenz96_accuracy": summary(c_l96),
            "harmonic_accuracy": summary(c_harm),
            "supported": min(c_l96) > 0.80 and max(c_harm) < 0.60,
        },
    }


def _rel_err(value: float | None, truth: float) -> float | None:
    return None if value is None else abs(value - truth) / abs(truth)


def verdict_stochastic(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """L-S (protocol amendment 3), at epsilon = 1e-3 as for H0-A, required for every seed:
    the sensitivity growth rate of the stochastic model is within 30 % of the leading
    Lyapunov exponent and closer to it than the deterministic one-step model."""
    key = "eps0.001"
    reference = [r["reference_lyapunov"] for r in runs]
    det = [r[f"jepa_{key}"]["growth_rate"] for r in runs]
    sens = [r["stochastic"][f"sensitivity_{key}"]["growth_rate"] for r in runs]
    det_err = [_rel_err(d, ref) for d, ref in zip(det, reference, strict=True)]
    sens_err = [_rel_err(s, ref) for s, ref in zip(sens, reference, strict=True)]
    supported = all(
        se is not None and se <= 0.30 and (de is None or se < de)
        for se, de in zip(sens_err, det_err, strict=True)
    )
    return {
        "epsilon": 1e-3,
        "reference_lyapunov": reference,
        "deterministic_growth_rate": det,
        "stochastic_sensitivity_growth_rate": sens,
        "stochastic_spread_growth_rate": [r["stochastic"]["spread"]["growth_rate"] for r in runs],
        "deterministic_relative_error": det_err,
        "stochastic_relative_error": sens_err,
        "supported": supported,
    }


def run_stochastic(profile_name: str) -> dict[str, Any]:
    """L-S: deterministic one-step and stochastic models on Lorenz-96 for every seed."""
    profile = PROFILES[profile_name]
    runs = [calibrate_predictability(profile, seed, stochastic=True) for seed in profile.seeds]
    return {
        **provenance(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "profile": profile_name,
        "config": asdict(profile),
        "noise_dim": NOISE_DIM,
        "ensemble_members": ENSEMBLE_MEMBERS,
        "sensitivity": "twin rollouts from clean and perturbed frames, same noise sequence",
        "spread": "mean pairwise distance of members from one state, independent noise",
        "torch_version": torch.__version__,
        "runs": {"predictability": runs},
        "verdicts": {"L-S": verdict_stochastic(runs)},
    }


def run(profile_name: str) -> dict[str, Any]:
    """Run the three calibrations for every seed of the profile and apply the criteria."""
    profile = PROFILES[profile_name]
    runs: dict[str, list[dict[str, Any]]] = {"predictability": [], "modes": [], "arrow": []}
    for seed in profile.seeds:
        runs["predictability"].append(calibrate_predictability(profile, seed))
        runs["modes"].append(calibrate_modes(profile, seed))
        runs["arrow"].append(calibrate_arrow(profile, seed))
    return {
        **provenance(),
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
    parser.add_argument(
        "--stochastic", action="store_true", help="measure L-S (protocol amendment 3) instead"
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stochastic:
        record = run_stochastic(args.profile)
        suffix = "" if args.profile == "full" else f"-{args.profile}"
        path = args.out / f"calibration-stochastic{suffix}.json"
    else:
        record = run(args.profile)
        path = args.out / f"calibration-{args.profile}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["verdicts"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
