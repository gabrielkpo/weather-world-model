"""Evaluation of the step-2 QG models: W-A, W-C, W-P and W-R per model and per seed.

See :mod:`weather_world_model.worlds.run_qg` for the protocol and the record.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import MISSING, asdict, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from weather_world_model.diagnostics.arrow import (
    arrow_of_time_dataset,
    clips,
    probe_accuracy,
    train_probe,
)
from weather_world_model.diagnostics.predictability import (
    ENSEMBLE_MEMBERS,
    ensemble_spread_curve,
    growth_fit,
    saturation_distance,
    separation_curve,
)
from weather_world_model.diagnostics.readouts import fit_readout_probes, rollout_readouts
from weather_world_model.jepa import NOISE_DIM, JEPAConfig, SequenceJEPA, encode_numpy, train_jepa
from weather_world_model.linalg import pca
from weather_world_model.seeding import seed_everything
from weather_world_model.worlds.qg import (
    DIAGNOSTIC_NAMES,
    twin_divergence,
    velocity_saturation,
)
from weather_world_model.worlds.qg_data import Profile, QGData, log, mirror_x, peak_rss_mb

FloatArray = NDArray[np.float64]
ConvNorm = Literal["group", "none"]
EPSILONS = (1e-3, 1e-2)
CURVE_POINTS = 40  # curves are stored subsampled to keep the record small
HORIZONS = (1, 5, 10)  # rollout-readout horizons, in snapshots (W-R)
READOUT_SAMPLES = 8  # sampled rollouts averaged for the readouts of a stochastic model
BASE_MODELS: tuple[tuple[str, int, int], ...] = (("k1", 1, 0), ("k8", 8, 0))
AMENDMENT3_MODELS = (*BASE_MODELS, ("stoch", 1, NOISE_DIM))  # (name, rollout_steps, noise_dim)

__all__ = ["ENSEMBLE_MEMBERS", "clips"]


# --------------------------------------------------------------------- W-A
def _subsample(curve: FloatArray) -> list[float]:
    idx = np.unique(np.linspace(0, len(curve) - 1, CURVE_POINTS).astype(int))
    return [float(curve[i]) for i in idx]


def _growth(curve: FloatArray, saturation: float, dt: float) -> dict[str, Any]:
    return growth_fit(curve, saturation, dt=dt, safe=True, stored_curve=_subsample)


def true_predictability(data: QGData, profile: Profile, seed: int) -> dict[str, Any]:
    """Simulator twin divergence from the W-A start states (median curve, as for latents)."""
    rng = np.random.default_rng(seed)
    saturation = float(np.median([velocity_saturation(m) for m in data.starts]))
    result: dict[str, Any] = {"n_states": len(data.starts)}
    for eps in EPSILONS:
        curves = np.stack(
            [
                twin_divergence(
                    m,
                    epsilon=eps,
                    n_samples=profile.separation_k,
                    sample_every=profile.sample_every,
                    rng=rng,
                )
                for m in data.starts
            ]
        )
        result[f"eps{eps:g}"] = _growth(np.median(curves, axis=0), saturation, data.sample_dt)
    return result


def latent_predictability(
    model: SequenceJEPA, data: QGData, profile: Profile, seed: int
) -> dict[str, Any]:
    """Latent separation of perturbed start frames under the model's own rollout.

    For a stochastic model the clean and perturbed rollouts are driven by the same noise
    sequence (sensitivity to initial conditions), and the spread of an ensemble started from
    the clean latents with independent noise is added under ``"spread"``.
    """
    rng = np.random.default_rng(seed)
    shape = model.config.frame_shape
    stochastic = model.config.noise_dim > 0
    generator = torch.Generator().manual_seed(seed)
    shared = (
        torch.randn(
            (len(data.start_frames), profile.separation_k, model.config.noise_dim),
            generator=generator,
        )
        if stochastic
        else None
    )

    def encode(x: FloatArray) -> FloatArray:
        return encode_numpy(model, x.reshape(-1, *shape))

    def rollout(z0: FloatArray, k: int) -> FloatArray:
        out = model.rollout(torch.as_tensor(z0, dtype=torch.float32), k, noise=shared)
        return np.asarray(out.numpy(), dtype=np.float64)

    def sampled_rollout(z0: FloatArray, k: int) -> FloatArray:
        out = model.rollout(torch.as_tensor(z0, dtype=torch.float32), k, generator=generator)
        return np.asarray(out.numpy(), dtype=np.float64)

    test_latents = encode_numpy(model, data.frames["test"]).reshape(-1, model.config.latent_dim)
    saturation = saturation_distance(test_latents, rng)
    flat = data.start_frames.reshape(len(data.start_frames), -1).astype(np.float64)
    result: dict[str, Any] = {"saturation_distance": saturation}
    for eps in EPSILONS:
        curve = separation_curve(
            encode, rollout, flat, epsilon=eps, k=profile.separation_k, rng=rng
        )
        result[f"eps{eps:g}"] = _growth(curve, saturation, data.sample_dt)
    if stochastic:
        result["noise"] = "same noise sequence for the clean and the perturbed rollout"
        spread = ensemble_spread_curve(
            sampled_rollout, encode(flat), k=profile.separation_k, n_members=ENSEMBLE_MEMBERS
        )
        result["spread"] = _growth(spread, saturation, data.sample_dt)
    return result


# --------------------------------------------------------------------- W-C
def arrow_probe(
    train: NDArray[np.floating],
    test: NDArray[np.floating],
    profile: Profile,
    seed: int,
    *,
    train_mirrored: NDArray[np.floating] | None = None,
    test_mirrored: NDArray[np.floating] | None = None,
) -> float:
    """Asymmetry-feature probe: PCA 8 and logistic probe fitted on train clips, test accuracy.

    ``train`` and ``test`` are per-run sequences ``(N, T, d)``. The PCA is fitted on the
    training frames that the training clips are cut from (each frame once, which keeps the
    memory of the 8192-dimensional oracle fields bounded), then the clips are cut from the
    projected sequences.

    With ``train_mirrored`` and ``test_mirrored`` (the same sequences computed from x-mirrored
    fields, see :func:`mirror_x`), the "reversed" class is the reversed mirrored clip: the
    control of protocol amendment 4, which keeps the direction of wave propagation. The PCA
    is still fitted on the unmirrored training sequences.
    """
    if (train_mirrored is None) != (test_mirrored is None):
        raise ValueError("give both mirrored sequences or neither")
    rng = np.random.default_rng(seed)
    mean, basis = pca(np.asarray(train), min(8, train.shape[-1]))

    def cut(seqs: NDArray[np.floating] | None) -> FloatArray | None:
        if seqs is None:
            return None
        return clips((seqs - mean) @ basis, profile.clip_len, profile.clip_stride)

    tr, te = cut(train), cut(test)
    if tr is None or te is None:  # pragma: no cover (narrowing for the type checker)
        raise TypeError("train and test sequences are required")
    x_tr, y_tr = arrow_of_time_dataset(
        tr, rng, features="asymmetry", reversed_from=cut(train_mirrored)
    )
    x_te, y_te = arrow_of_time_dataset(
        te, rng, features="asymmetry", reversed_from=cut(test_mirrored)
    )
    probe = train_probe(x_tr, y_tr, l2=1e-3, seed=seed)
    return probe_accuracy(probe, x_te, y_te)


# --------------------------------------------------------------------- per seed
def _latents(model: SequenceJEPA, data: QGData) -> dict[str, FloatArray]:
    return {k: encode_numpy(model, v, batch_size=256) for k, v in data.frames.items()}


def _flat(seqs: dict[str, FloatArray]) -> dict[str, FloatArray]:
    return {k: v.reshape(-1, v.shape[-1]) for k, v in seqs.items()}


def evaluate_model(
    model: SequenceJEPA,
    data: QGData,
    profile: Profile,
    seed: int,
    *,
    dynamics: bool,
    rollouts: bool = False,
) -> dict[str, Any]:
    """W-P readouts for any encoder; W-A and W-C for trained models (``dynamics``); W-R
    rollout readouts with ``rollouts`` (mean of ``READOUT_SAMPLES`` sampled rollouts for a
    stochastic model)."""
    lat = _latents(model, data)
    probes = fit_readout_probes(
        _flat(lat), {k: v.reshape(-1, v.shape[-1]) for k, v in data.diags.items()}, DIAGNOSTIC_NAMES
    )
    result: dict[str, Any] = {
        "readouts": probes.scores,
        "test_latent_std": float(lat["test"].reshape(-1, lat["test"].shape[-1]).std(axis=0).mean()),
    }
    if dynamics:
        result["predictability"] = latent_predictability(model, data, profile, seed)
        result["arrow_accuracy"] = arrow_probe(lat["train"], lat["test"], profile, seed)
        result["arrow_accuracy_mirrored"] = arrow_probe(
            lat["train"],
            lat["test"],
            profile,
            seed,
            train_mirrored=encode_numpy(
                model, mirror_x(data.frames["train"], data), batch_size=256
            ),
            test_mirrored=encode_numpy(model, mirror_x(data.frames["test"], data), batch_size=256),
        )
    if rollouts:
        generator = torch.Generator().manual_seed(seed)

        def rollout(z0: FloatArray, k: int) -> FloatArray:
            out = model.rollout(torch.as_tensor(z0, dtype=torch.float32), k, generator=generator)
            return np.asarray(out.numpy(), dtype=np.float64)

        n_samples = READOUT_SAMPLES if model.config.noise_dim > 0 else 1
        result["rollout_readouts"] = {
            "n_sampled_rollouts": n_samples,
            **rollout_readouts(
                probes,
                rollout,
                lat["test"],
                data.diags["test"],
                horizons=HORIZONS,
                n_samples=n_samples,
            ),
        }
    return result


def _plain(value: Any) -> Any:
    """JSON round trip, so that tuples compare equal to the lists read back from disk."""
    return json.loads(json.dumps(value))


def checkpoint_matches(saved: dict[str, Any], cfg: JEPAConfig, profile: Profile, seed: int) -> bool:
    """True if a checkpoint was written for this model configuration, profile and seed.

    A configuration saved before a field existed is read with that field's default, so that
    older checkpoints stay usable when the default keeps the earlier behaviour.
    """
    saved_cfg = dict(saved.get("config") or {})
    for f in fields(JEPAConfig):
        if f.name not in saved_cfg and f.default is not MISSING:
            saved_cfg[f.name] = _plain(f.default)
    return (
        saved_cfg == _plain(asdict(cfg))
        and saved.get("profile") == _plain(asdict(profile))
        and saved.get("seed") == seed
    )


def train_or_load(
    cfg: JEPAConfig, data: QGData, profile: Profile, seed: int, name: str, checkpoint: Path | None
) -> tuple[SequenceJEPA, dict[str, Any]]:
    """Train one model, or load it from ``checkpoint`` if that file holds the same
    configuration, profile and seed. A freshly trained model is saved there."""
    meta = {"config": _plain(asdict(cfg)), "profile": _plain(asdict(profile)), "seed": seed}
    if checkpoint is not None and checkpoint.exists():
        saved = torch.load(checkpoint, weights_only=True)
        if checkpoint_matches(saved, cfg, profile, seed):
            model = SequenceJEPA(cfg)
            model.load_state_dict(saved["state_dict"])
            log(f"seed {seed} {name} loaded from {checkpoint}")
            return model, {**saved["info"], "from_checkpoint": True}
        log(f"seed {seed} {name}: {checkpoint} has another configuration, retraining")
    t0 = time.perf_counter()

    def progress(step: int, loss: float, std: float) -> None:
        done = step + 1
        elapsed = time.perf_counter() - t0
        eta = elapsed / done * (profile.train_steps - done)
        log(
            f"seed {seed} {name} step {done}/{profile.train_steps} loss {loss:.4f} "
            f"latent_std {std:.3f} elapsed {elapsed:.0f}s eta {eta:.0f}s"
        )

    model, history = train_jepa(
        data.frames["train"], cfg, n_steps=profile.train_steps, seed=seed, progress=progress
    )
    train_s = time.perf_counter() - t0
    info: dict[str, Any] = {
        "train_seconds": train_s,
        "seconds_per_step": train_s / max(1, profile.train_steps),
        "final_latent_std": history.latent_std[-1],
        "collapsed": history.collapsed,
        "loss_subsampled": _subsample(np.asarray(history.loss)),
    }
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint.with_suffix(".tmp")
        torch.save({**meta, "info": info, "state_dict": model.state_dict()}, tmp)
        tmp.replace(checkpoint)
        log(f"seed {seed} {name} saved to {checkpoint}")
    return model, info


def run_seed(
    data: QGData,
    profile: Profile,
    seed: int,
    *,
    amendment3: bool = False,
    checkpoints: Path | None = None,
    conv_norm: ConvNorm = "group",
) -> dict[str, Any]:
    """Train k = 1 and k = 8 models, build the random-weight control, evaluate all three.

    ``conv_norm`` is the normalisation of the convolutional encoder, for every trained model
    and for the random-weight control (protocol amendment 6).

    With ``amendment3`` a stochastic one-step model is trained as well and every trained
    model also gets rollout readouts.
    """
    out: dict[str, Any] = {"seed": seed}
    base = JEPAConfig(
        encoder="conv",
        input_shape=(2, *data.frames["train"].shape[-2:]),
        batch_size=profile.batch_size,
        conv_channels=profile.conv_channels,
        log_every=max(1, profile.train_steps // 50),
        conv_norm=conv_norm,
    )
    for name, k, noise_dim in AMENDMENT3_MODELS if amendment3 else BASE_MODELS:
        cfg = replace(base, rollout_steps=k, noise_dim=noise_dim)
        path = checkpoints / f"seed{seed}-{name}.pt" if checkpoints is not None else None
        model, info = train_or_load(cfg, data, profile, seed, name, path)
        out[name] = {
            **info,
            **evaluate_model(model, data, profile, seed, dynamics=True, rollouts=amendment3),
        }
        log(f"seed {seed} {name} evaluated, peak RSS {peak_rss_mb():.0f} MB")
        del model
        gc.collect()
    seed_everything(seed)
    random_model = SequenceJEPA(base)  # same architecture as the trained models, untrained
    out["random"] = evaluate_model(random_model, data, profile, seed, dynamics=False)
    out["peak_rss_mb"] = peak_rss_mb()
    return out
