"""Step-2 measurements on the two-layer QG world (protocol amendment 2: W-A, W-C, W-P).

Run: ``uv run python -m weather_world_model.worlds.run_qg --profile full --out results/qg``.
Writes one JSON record (code commit, date, configuration, per-seed values, verdicts).

Data: independent QG runs split by run 70/15/15 (train/val/test), field ``"uv"`` (upper
layer eddy winds, ``(2, 64, 64)``), snapshots after spin-up, standardised per channel with
train statistics. Models per seed: the conv JEPA trained with a one-step loss (k = 1) and
with an 8-step rollout loss (k = 8) on the same data for the same number of steps, and the
same encoder with random weights (control for W-P).

Protocol amendment 5: snapshots are taken after a spin-up of 12 000 steps, and the run stops
before any training unless the training data pass the stationarity check (mean kinetic
energy of the last third of the snapshots within 10 % of the first third). Amendment 4: the
arrow-of-time probe is reported against plain reversal and against reversal combined with
the x-mirror, on true fields and on latents. Records are named ``qg-stationary...`` so that
the records of the earlier runs (spinning-up flow) stay untouched.

With ``--amendment3`` the run measures protocol amendment 3 and writes
``qg-stationary[-<profile>].json`` (without it: ``qg-stationary-base[-<profile>].json``):
the k = 1 and k = 8 models are retrained with the same
configuration and seeds, a stochastic one-step model (noise dimension 8, energy score) is
added, and every model also gets rollout readouts (W-R); the stochastic model's sensitivity
to initial conditions under a shared noise sequence decides W-S. Trained weights are saved
under ``--checkpoints`` (ignored by Git) and reused by a restarted run.

Protocol amendment 6: ``--conv-norm none`` builds every model and the random-weight control
with the convolutional encoder without normalisation; the record and the checkpoint
directory then carry the suffix ``-nonorm``. The default (``group``) is the encoder of the
earlier runs.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from dataclasses import MISSING, asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from numpy.typing import NDArray

from weather_world_model.calibration.run import _git_commit, _pca, _summary, growth_window
from weather_world_model.diagnostics.arrow import (
    arrow_of_time_dataset,
    probe_accuracy,
    train_probe,
)
from weather_world_model.diagnostics.predictability import (
    ensemble_spread_curve,
    growth_rate,
    normalised,
    saturation_distance,
    separation_curve,
)
from weather_world_model.diagnostics.readouts import fit_readout_probes, rollout_readouts
from weather_world_model.jepa import JEPAConfig, SequenceJEPA, encode_numpy, train_jepa
from weather_world_model.seeding import seed_everything
from weather_world_model.worlds.qg import (
    DIAGNOSTIC_NAMES,
    QGConfig,
    QGModel,
    diagnostics,
    twin_divergence,
    velocity_saturation,
)

FloatArray = NDArray[np.float64]
ConvNorm = Literal["group", "none"]
EPSILONS = (1e-3, 1e-2)
THRESHOLDED = ("kinetic_energy", "available_potential_energy", "enstrophy", "jet_speed_max")
CURVE_POINTS = 40  # curves are stored subsampled to keep the record small
TORCH_THREADS = 4
STATIONARITY_TOLERANCE = 0.10
NOISE_DIM = 8  # noise dimension of the stochastic predictor (protocol amendment 3)
HORIZONS = (1, 5, 10)  # rollout-readout horizons, in snapshots (W-R)
READOUT_SAMPLES = 8  # sampled rollouts averaged for the readouts of a stochastic model
ENSEMBLE_MEMBERS = 8  # members of the ensemble whose spread is reported
BASE_MODELS: tuple[tuple[str, int, int], ...] = (("k1", 1, 0), ("k8", 8, 0))
AMENDMENT3_MODELS = (*BASE_MODELS, ("stoch", 1, NOISE_DIM))  # (name, rollout_steps, noise_dim)


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


# --------------------------------------------------------------------- data
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
    """Independent runs, split by run; keeps simulator copies at the W-A start states."""
    cfg = config or QGConfig()
    rng = np.random.default_rng(profile.data_seed)
    sizes = split_sizes(profile.n_runs)
    names = ("train", "val", "test")
    n, t_len = cfg.nx, profile.snapshots
    start_idx = np.linspace(0, t_len - 1, profile.starts_per_test_run + 1)[:-1].astype(int)
    frames: dict[str, NDArray[np.float32]] = {}
    diags: dict[str, FloatArray] = {}
    starts: list[QGModel] = []
    start_frames: list[NDArray[np.float32]] = []
    for name, size in zip(names, sizes, strict=True):
        snaps = np.empty((size, t_len, 2, n, n), dtype=np.float32)
        dg = np.empty((size, t_len, len(DIAGNOSTIC_NAMES)))
        for i in range(size):
            model = QGModel(cfg, rng)
            model.spin_up(profile.spin_up)
            for t in range(t_len):
                model.step(profile.sample_every)
                u, v = model.velocity()
                snaps[i, t] = np.stack([u[0], v[0]])
                d = diagnostics(model)
                dg[i, t] = [d[k] for k in DIAGNOSTIC_NAMES]
                if name == "test" and t in start_idx:
                    starts.append(model.copy())
                    start_frames.append(snaps[i, t].copy())
        frames[name], diags[name] = snaps, dg
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


# --------------------------------------------------------------------- W-A
def _subsample(curve: FloatArray) -> list[float]:
    idx = np.unique(np.linspace(0, len(curve) - 1, CURVE_POINTS).astype(int))
    return [float(curve[i]) for i in idx]


def _growth(curve: FloatArray, saturation: float, dt: float) -> dict[str, Any]:
    norm = normalised(curve, saturation)
    out: dict[str, Any] = {"normalised_curve_subsampled": _subsample(norm)}
    try:
        window = growth_window(norm)
    except ValueError as err:
        return {**out, "growth_rate": None, "window": None, "error": str(err)}
    return {**out, "growth_rate": growth_rate(curve, dt=dt, window=window), "window": list(window)}


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
def clips(sequences: NDArray[np.floating], clip_len: int, stride: int) -> FloatArray:
    """Clips ``(N * per, clip_len, d)`` cut from ``(N, T, d)`` with the given stride."""
    seqs = np.asarray(sequences, dtype=np.float64)
    starts = range(0, seqs.shape[1] - clip_len + 1, stride)
    return np.concatenate([seqs[:, s : s + clip_len] for s in starts], axis=0)


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
    mean, basis = _pca(np.asarray(train), min(8, train.shape[-1]))

    def cut(seqs: NDArray[np.floating] | None) -> FloatArray | None:
        if seqs is None:
            return None
        return clips((seqs - mean) @ basis, profile.clip_len, profile.clip_stride)

    tr, te = cut(train), cut(test)
    assert tr is not None and te is not None  # noqa: S101 (narrowing for the type checker)
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


# --------------------------------------------------------------------- verdicts
def _rel_err(value: float | None, truth: float | None) -> float | None:
    if value is None or truth is None:
        return None
    return abs(value - truth) / abs(truth)


def verdicts(truth: dict[str, Any], seeds: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the step-2 criteria (protocol amendment 2) to the per-seed results.

    W-A uses epsilon = 1e-3 (as in the calibration) and must hold for every seed; W-C and
    W-P use the worst seed (minimum accuracy, minimum R² and minimum margin over random).
    """
    key = "eps0.001"
    true_rate = truth[key]["growth_rate"]
    rates = {m: [s[m]["predictability"][key]["growth_rate"] for s in seeds] for m in ("k1", "k8")}
    err1 = [_rel_err(r, true_rate) for r in rates["k1"]]
    err8 = [_rel_err(r, true_rate) for r in rates["k8"]]
    wa_ok = all(
        e8 is not None and e8 <= 0.30 and (e1 is None or e8 < e1)
        for e1, e8 in zip(err1, err8, strict=True)
    )
    arrow = {m: [s[m]["arrow_accuracy"] for s in seeds] for m in ("k1", "k8")}
    mirrored = {m: [s[m]["arrow_accuracy_mirrored"] for s in seeds] for m in ("k1", "k8")}
    wp: dict[str, Any] = {}
    for m in ("k1", "k8"):
        per_q: dict[str, Any] = {}
        for name in DIAGNOSTIC_NAMES:
            r2 = [s[m]["readouts"][name]["test_r2"] for s in seeds]
            rnd = [s["random"]["readouts"][name]["test_r2"] for s in seeds]
            margin = [a - b for a, b in zip(r2, rnd, strict=True)]
            entry: dict[str, Any] = {
                "r2": _summary(r2),
                "random_r2": _summary(rnd),
                "margin": _summary(margin),
            }
            if name in THRESHOLDED:
                entry["passes"] = min(r2) >= 0.7 and min(margin) >= 0.1
            per_q[name] = entry
        wp[m] = {
            "quantities": per_q,
            "supported": all(per_q[q]["passes"] for q in THRESHOLDED),
        }
    return {
        "W-A": {
            "epsilon": 1e-3,
            "true_growth_rate": true_rate,
            "k1_growth_rate": rates["k1"],
            "k8_growth_rate": rates["k8"],
            "k1_relative_error": err1,
            "k8_relative_error": err8,
            "supported": wa_ok,
        },
        "W-C": {
            m: {
                "accuracy": _summary(arrow[m]),
                "accuracy_mirrored_control": _summary(mirrored[m]),
                "supported": min(arrow[m]) > 0.80,
            }
            for m in ("k1", "k8")
        },
        "W-P": wp,
        "any_collapse": any(s[m]["collapsed"] for s in seeds for m in ("k1", "k8")),
    }


def _gain(readouts: dict[str, Any], horizon: int) -> float:
    """Mean over the thresholded quantities of rollout R² minus persistence R²."""
    at_h = readouts[f"h{horizon}"]
    return float(np.mean([at_h[q]["rollout_r2"] - at_h[q]["persistence_r2"] for q in THRESHOLDED]))


def verdicts_amendment3(truth: dict[str, Any], seeds: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the criteria of protocol amendment 3 to the per-seed results.

    W-R: at h = 10, the mean gain of the rollout over persistence on the four thresholded
    quantities is at least 0.05 for the one-step and the multi-step model, on every seed.
    W-S (epsilon = 1e-3, every seed): the stochastic model's sensitivity growth rate is
    within 30 % of the simulator's and closer to it than the deterministic one-step model.
    The stochastic model's W-C, W-P and W-R scores are reported without thresholds.
    """
    models = [name for name, _, _ in AMENDMENT3_MODELS]
    wr: dict[str, Any] = {}
    for m in models:
        gains = {f"h{h}": [_gain(s[m]["rollout_readouts"], h) for s in seeds] for h in HORIZONS}
        wr[m] = {
            "mean_gain_over_persistence": gains,
            "passes_h10": all(g >= 0.05 for g in gains["h10"]),
            "quantities": {
                f"h{h}": {
                    name: {
                        key: _summary([s[m]["rollout_readouts"][f"h{h}"][name][key] for s in seeds])
                        for key in ("rollout_r2", "persistence_r2")
                    }
                    for name in DIAGNOSTIC_NAMES
                }
                for h in HORIZONS
            },
        }
    key = "eps0.001"
    true_rate = truth[key]["growth_rate"]
    det = [s["k1"]["predictability"][key]["growth_rate"] for s in seeds]
    sens = [s["stoch"]["predictability"][key]["growth_rate"] for s in seeds]
    det_err = [_rel_err(r, true_rate) for r in det]
    sens_err = [_rel_err(r, true_rate) for r in sens]
    ws_ok = all(
        se is not None and se <= 0.30 and (de is None or se < de)
        for se, de in zip(sens_err, det_err, strict=True)
    )
    arrow = [s["stoch"]["arrow_accuracy"] for s in seeds]
    return {
        "W-R": {
            "horizons": list(HORIZONS),
            "criterion_quantities": list(THRESHOLDED),
            "models": wr,
            "supported": bool(wr["k1"]["passes_h10"] and wr["k8"]["passes_h10"]),
        },
        "W-S": {
            "epsilon": 1e-3,
            "true_growth_rate": true_rate,
            "deterministic_k1_growth_rate": det,
            "stochastic_sensitivity_growth_rate": sens,
            "stochastic_spread_growth_rate": [
                s["stoch"]["predictability"]["spread"]["growth_rate"] for s in seeds
            ],
            "deterministic_k1_relative_error": det_err,
            "stochastic_relative_error": sens_err,
            "supported": ws_ok,
        },
        "stochastic_reported": {
            "W-C_accuracy": _summary(arrow),
            "W-C_accuracy_mirrored_control": _summary(
                [s["stoch"]["arrow_accuracy_mirrored"] for s in seeds]
            ),
            "W-P_r2": {
                name: _summary([s["stoch"]["readouts"][name]["test_r2"] for s in seeds])
                for name in DIAGNOSTIC_NAMES
            },
            "W-R": "see W-R.models.stoch (mean of sampled rollouts, no threshold)",
        },
        "retrained_k1_k8": verdicts(truth, seeds),
        "any_collapse": any(s[m]["collapsed"] for s in seeds for m in models),
    }


def run(
    profile_name: str,
    partial: Path | None = None,
    *,
    amendment3: bool = False,
    checkpoints: Path | None = None,
    conv_norm: ConvNorm = "group",
) -> dict[str, Any]:
    """Simulate the data once, then train and evaluate every seed, and apply the criteria.

    With ``partial``, the record is rewritten there after every seed, and a run restarted
    with the same file skips the seeds (and the truth) it already holds, provided the
    profile matches. ``amendment3`` switches to the measurements of protocol amendment 3
    (see :func:`run_seed` and :func:`verdicts_amendment3`); ``checkpoints`` is the directory
    where trained weights are saved and looked up.
    """
    profile = PROFILES[profile_name]
    torch.set_num_threads(TORCH_THREADS)
    t0 = time.perf_counter()
    done: dict[str, Any] = {}
    if partial is not None and partial.exists():
        previous = json.loads(partial.read_text())
        same_encoder = previous.get("conv_norm", "group") == conv_norm
        if same_encoder and previous.get("config") == json.loads(json.dumps(asdict(profile))):
            done = previous
            log(f"resuming from {partial}: seeds {[s['seed'] for s in done['seeds']]} done")
    data = simulate(profile)
    simulate_s = time.perf_counter() - t0
    log(f"simulated {profile.n_runs} runs in {simulate_s:.0f}s, peak RSS {peak_rss_mb():.0f} MB")
    stationarity = stationarity_check(
        data.diags["train"][..., DIAGNOSTIC_NAMES.index("kinetic_energy")],
        profile.stationarity_tolerance,
    )
    log(
        f"stationarity: kinetic energy {stationarity['first_third_mean']:.4f} (first third) -> "
        f"{stationarity['last_third_mean']:.4f} (last third), relative difference "
        f"{stationarity['relative_difference']:.3f}, tolerance {stationarity['tolerance']}"
    )
    if stationarity["passes"] is False:
        raise RuntimeError(
            "the training data are not stationary (protocol amendment 5): mean kinetic energy "
            f"goes from {stationarity['first_third_mean']:.4f} in the first third of the "
            f"snapshots to {stationarity['last_third_mean']:.4f} in the last third, a relative "
            f"difference of {stationarity['relative_difference']:.3f} (tolerance "
            f"{stationarity['tolerance']}); increase the spin-up. Nothing was trained."
        )
    if "truth" in done:
        truth = done["truth"]["predictability"]
        oracle_arrow = done["truth"]["arrow_accuracy"]
        oracle_mirrored = done["truth"]["arrow_accuracy_mirrored"]
    else:
        truth = true_predictability(data, profile, profile.data_seed)

        def fields(split: str, mirrored: bool) -> NDArray[np.float32]:
            frames = data.frames[split]
            frames = mirror_x(frames, data) if mirrored else frames
            return frames.reshape(*frames.shape[:2], -1)

        oracle_arrow = arrow_probe(
            fields("train", False), fields("test", False), profile, profile.data_seed
        )
        oracle_mirrored = arrow_probe(
            fields("train", False),
            fields("test", False),
            profile,
            profile.data_seed,
            train_mirrored=fields("train", True),
            test_mirrored=fields("test", True),
        )
        gc.collect()
        log(f"truth done, peak RSS {peak_rss_mb():.0f} MB")
    seeds: list[dict[str, Any]] = list(done.get("seeds", []))
    previous_wall = float(done.get("wall_seconds", 0.0))

    def record(final: bool) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "code_commit": _git_commit(),
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "profile": profile_name,
            "config": asdict(profile),
            "conv_norm": conv_norm,
            "qg_config": asdict(QGConfig()),
            "split_runs": dict(
                zip(("train", "val", "test"), split_sizes(profile.n_runs), strict=True)
            ),
            "sample_dt": data.sample_dt,
            "field": "uv",
            "diagnostic_names": list(DIAGNOSTIC_NAMES),
            "torch_version": torch.__version__,
            "torch_threads": TORCH_THREADS,
            "simulate_seconds": simulate_s,
            "wall_seconds": previous_wall + time.perf_counter() - t0,
            "resumed": bool(done),
            "stationarity": stationarity,
            "truth": {
                "predictability": truth,
                "arrow_accuracy": oracle_arrow,
                "arrow_accuracy_mirrored": oracle_mirrored,
            },
            "seeds": seeds,
        }
        if amendment3:
            rec["amendment3"] = {
                "noise_dim": NOISE_DIM,
                "horizons": list(HORIZONS),
                "readout_sampled_rollouts": READOUT_SAMPLES,
                "ensemble_members": ENSEMBLE_MEMBERS,
                "stochastic_readout": "mean of the probe outputs over the sampled rollouts",
                "sensitivity": "twin rollouts from clean and perturbed frames, same noise sequence",
                "spread": "mean pairwise distance of members from one state, independent noise",
            }
        if final and amendment3:
            rec["verdicts"] = verdicts_amendment3(truth, seeds)
            wc = rec["verdicts"]["retrained_k1_k8"]["W-C"]
        elif final:
            rec["verdicts"] = verdicts(truth, seeds)
            wc = rec["verdicts"]["W-C"]
        if final:
            wc["oracle_accuracy"] = oracle_arrow
            wc["oracle_accuracy_mirrored_control"] = oracle_mirrored
        return rec

    for s in profile.seeds:
        if any(prev["seed"] == s for prev in seeds):
            continue
        seeds.append(
            run_seed(
                data,
                profile,
                s,
                amendment3=amendment3,
                checkpoints=checkpoints,
                conv_norm=conv_norm,
            )
        )
        seeds.sort(key=lambda r: r["seed"])
        if partial is not None:
            partial.write_text(json.dumps(record(final=False), indent=2) + "\n")
            log(f"seed {s} written to {partial}")
    return record(final=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--amendment3", action="store_true", help="measure protocol amendment 3 (W-R, W-S)"
    )
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="directory for trained weights (saved, and reused on restart); keep it out of Git",
    )
    parser.add_argument(
        "--conv-norm",
        choices=("group", "none"),
        default="group",
        help="normalisation of the convolutional encoder, for every model and the random control",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = "qg-stationary" if args.amendment3 else "qg-stationary-base"
    if args.conv_norm == "none":
        stem = f"{stem}-nonorm"
    if args.profile != "full":
        stem = f"{stem}-{args.profile}"
    partial = args.out / f"{stem}.partial.json"
    checkpoints = args.checkpoints / stem if args.checkpoints is not None else None
    record = run(
        args.profile,
        partial=partial,
        amendment3=args.amendment3,
        checkpoints=checkpoints,
        conv_norm=args.conv_norm,
    )
    path = args.out / f"{stem}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    partial.unlink(missing_ok=True)
    print(json.dumps(record["verdicts"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
