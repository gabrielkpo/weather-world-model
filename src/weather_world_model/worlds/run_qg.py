"""Step-2 measurements on the two-layer QG world (protocol amendment 2: W-A, W-C, W-P).

Run: ``uv run python -m weather_world_model.worlds.run_qg --profile full --out results/qg``.
Writes one JSON record (code commit, date, configuration, per-seed values, verdicts).

Data: independent QG runs split by run 70/15/15 (train/val/test), field ``"uv"`` (upper
layer eddy winds, ``(2, 64, 64)``), snapshots after spin-up, standardised per channel with
train statistics. Models per seed: the conv JEPA trained with a one-step loss (k = 1) and
with an 8-step rollout loss (k = 8) on the same data for the same number of steps, and the
same encoder with random weights (control for W-P).
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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
    growth_rate,
    normalised,
    saturation_distance,
    separation_curve,
)
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
EPSILONS = (1e-3, 1e-2)
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
THRESHOLDED = ("kinetic_energy", "available_potential_energy", "enstrophy", "jet_speed_max")
CURVE_POINTS = 40  # curves are stored subsampled to keep the record small
TORCH_THREADS = 4


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


PROFILES: dict[str, Profile] = {
    "full": Profile(
        n_runs=80,
        snapshots=100,
        sample_every=20,
        spin_up=4000,
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
    """Latent separation of perturbed start frames under the model's own rollout."""
    rng = np.random.default_rng(seed)
    shape = model.config.frame_shape

    def encode(x: FloatArray) -> FloatArray:
        return encode_numpy(model, x.reshape(-1, *shape))

    def rollout(z0: FloatArray, k: int) -> FloatArray:
        out = model.rollout(torch.as_tensor(z0, dtype=torch.float32), k)
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
    return result


# --------------------------------------------------------------------- W-C
def clips(sequences: NDArray[np.floating], clip_len: int, stride: int) -> FloatArray:
    """Clips ``(N * per, clip_len, d)`` cut from ``(N, T, d)`` with the given stride."""
    seqs = np.asarray(sequences, dtype=np.float64)
    starts = range(0, seqs.shape[1] - clip_len + 1, stride)
    return np.concatenate([seqs[:, s : s + clip_len] for s in starts], axis=0)


def arrow_probe(
    train: NDArray[np.floating], test: NDArray[np.floating], profile: Profile, seed: int
) -> float:
    """Asymmetry-feature probe: PCA 8 and logistic probe fitted on train clips, test accuracy.

    ``train`` and ``test`` are per-run sequences ``(N, T, d)``. The PCA is fitted on the
    training frames that the training clips are cut from (each frame once, which keeps the
    memory of the 8192-dimensional oracle fields bounded), then the clips are cut from the
    projected sequences.
    """
    rng = np.random.default_rng(seed)
    mean, basis = _pca(np.asarray(train), min(8, train.shape[-1]))
    tr = clips((train - mean) @ basis, profile.clip_len, profile.clip_stride)
    te = clips((test - mean) @ basis, profile.clip_len, profile.clip_stride)
    x_tr, y_tr = arrow_of_time_dataset(tr, rng, features="asymmetry")
    x_te, y_te = arrow_of_time_dataset(te, rng, features="asymmetry")
    probe = train_probe(x_tr, y_tr, l2=1e-3, seed=seed)
    return probe_accuracy(probe, x_te, y_te)


# --------------------------------------------------------------------- W-P
def _ridge(x: FloatArray, y: FloatArray, alpha: float) -> FloatArray:
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    reg = alpha * np.eye(xb.shape[1])
    reg[-1, -1] = 0.0  # no penalty on the intercept
    return np.asarray(np.linalg.solve(xb.T @ xb + reg, xb.T @ y), dtype=np.float64)


def _predict(x: FloatArray, w: FloatArray) -> FloatArray:
    return np.asarray(np.hstack([x, np.ones((x.shape[0], 1))]) @ w, dtype=np.float64)


def r2_score(y: FloatArray, pred: FloatArray) -> float:
    """Coefficient of determination of one target."""
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def readout_probes(latents: dict[str, FloatArray], diags: dict[str, FloatArray]) -> dict[str, Any]:
    """Ridge probe per quantity; alpha chosen on val R², scored once on test."""
    z_mean = latents["train"].mean(axis=0)
    z_std = latents["train"].std(axis=0)
    z_std[z_std == 0] = 1.0
    y_mean = diags["train"].mean(axis=0)
    y_std = diags["train"].std(axis=0)
    y_std[y_std == 0] = 1.0
    x = {k: (v - z_mean) / z_std for k, v in latents.items()}
    y = {k: (v - y_mean) / y_std for k, v in diags.items()}
    out: dict[str, Any] = {}
    for q, name in enumerate(DIAGNOSTIC_NAMES):
        val_r2 = [
            r2_score(y["val"][:, q], _predict(x["val"], _ridge(x["train"], y["train"][:, q], a)))
            for a in RIDGE_ALPHAS
        ]
        best = RIDGE_ALPHAS[int(np.nanargmax(val_r2))]
        w = _ridge(x["train"], y["train"][:, q], best)
        out[name] = {
            "alpha": best,
            "val_r2": float(np.nanmax(val_r2)),
            "test_r2": r2_score(y["test"][:, q], _predict(x["test"], w)),
        }
    return out


# --------------------------------------------------------------------- per seed
def _latents(model: SequenceJEPA, data: QGData) -> dict[str, FloatArray]:
    return {k: encode_numpy(model, v, batch_size=256) for k, v in data.frames.items()}


def _flat(seqs: dict[str, FloatArray]) -> dict[str, FloatArray]:
    return {k: v.reshape(-1, v.shape[-1]) for k, v in seqs.items()}


def evaluate_model(
    model: SequenceJEPA, data: QGData, profile: Profile, seed: int, *, dynamics: bool
) -> dict[str, Any]:
    """W-P readouts for any encoder; W-A and W-C for trained models (``dynamics``)."""
    lat = _latents(model, data)
    result: dict[str, Any] = {
        "readouts": readout_probes(
            _flat(lat), {k: v.reshape(-1, v.shape[-1]) for k, v in data.diags.items()}
        ),
        "test_latent_std": float(lat["test"].reshape(-1, lat["test"].shape[-1]).std(axis=0).mean()),
    }
    if dynamics:
        result["predictability"] = latent_predictability(model, data, profile, seed)
        result["arrow_accuracy"] = arrow_probe(lat["train"], lat["test"], profile, seed)
    return result


def run_seed(data: QGData, profile: Profile, seed: int) -> dict[str, Any]:
    """Train k = 1 and k = 8 models, build the random-weight control, evaluate all three."""
    out: dict[str, Any] = {"seed": seed}
    for name, k in (("k1", 1), ("k8", 8)):
        cfg = JEPAConfig(
            encoder="conv",
            input_shape=(2, *data.frames["train"].shape[-2:]),
            rollout_steps=k,
            batch_size=profile.batch_size,
            conv_channels=profile.conv_channels,
            log_every=max(1, profile.train_steps // 50),
        )
        t0 = time.perf_counter()

        def progress(step: int, loss: float, std: float, name: str = name, t0: float = t0) -> None:
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
        out[name] = {
            "train_seconds": train_s,
            "seconds_per_step": train_s / max(1, profile.train_steps),
            "final_latent_std": history.latent_std[-1],
            "collapsed": history.collapsed,
            "loss_subsampled": _subsample(np.asarray(history.loss)),
            **evaluate_model(model, data, profile, seed, dynamics=True),
        }
        log(f"seed {seed} {name} evaluated, peak RSS {peak_rss_mb():.0f} MB")
        del model, history
        gc.collect()
    seed_everything(seed)
    random_model = SequenceJEPA(cfg)  # same architecture as the trained models, untrained
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
            m: {"accuracy": _summary(arrow[m]), "supported": min(arrow[m]) > 0.80}
            for m in ("k1", "k8")
        },
        "W-P": wp,
        "any_collapse": any(s[m]["collapsed"] for s in seeds for m in ("k1", "k8")),
    }


def run(profile_name: str, partial: Path | None = None) -> dict[str, Any]:
    """Simulate the data once, then train and evaluate every seed, and apply the criteria.

    With ``partial``, the record is rewritten there after every seed, and a run restarted
    with the same file skips the seeds (and the truth) it already holds, provided the
    profile matches.
    """
    profile = PROFILES[profile_name]
    torch.set_num_threads(TORCH_THREADS)
    t0 = time.perf_counter()
    done: dict[str, Any] = {}
    if partial is not None and partial.exists():
        previous = json.loads(partial.read_text())
        if previous.get("config") == json.loads(json.dumps(asdict(profile))):
            done = previous
            log(f"resuming from {partial}: seeds {[s['seed'] for s in done['seeds']]} done")
    data = simulate(profile)
    simulate_s = time.perf_counter() - t0
    log(f"simulated {profile.n_runs} runs in {simulate_s:.0f}s, peak RSS {peak_rss_mb():.0f} MB")
    if "truth" in done:
        truth = done["truth"]["predictability"]
        oracle_arrow = done["truth"]["arrow_accuracy"]
    else:
        truth = true_predictability(data, profile, profile.data_seed)
        oracle_arrow = arrow_probe(
            data.frames["train"].reshape(*data.frames["train"].shape[:2], -1),
            data.frames["test"].reshape(*data.frames["test"].shape[:2], -1),
            profile,
            profile.data_seed,
        )
        log(f"truth done, peak RSS {peak_rss_mb():.0f} MB")
    seeds: list[dict[str, Any]] = list(done.get("seeds", []))
    previous_wall = float(done.get("wall_seconds", 0.0))

    def record(final: bool) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "code_commit": _git_commit(),
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "profile": profile_name,
            "config": asdict(profile),
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
            "truth": {"predictability": truth, "arrow_accuracy": oracle_arrow},
            "seeds": seeds,
        }
        if final:
            rec["verdicts"] = verdicts(truth, seeds)
            rec["verdicts"]["W-C"]["oracle_accuracy"] = oracle_arrow
        return rec

    for s in profile.seeds:
        if any(prev["seed"] == s for prev in seeds):
            continue
        seeds.append(run_seed(data, profile, s))
        seeds.sort(key=lambda r: r["seed"])
        if partial is not None:
            partial.write_text(json.dumps(record(final=False), indent=2) + "\n")
            log(f"seed {s} written to {partial}")
    return record(final=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    partial = args.out / f"qg-{args.profile}.partial.json"
    record = run(args.profile, partial=partial)
    path = args.out / f"qg-{args.profile}.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    partial.unlink(missing_ok=True)
    print(json.dumps(record["verdicts"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
