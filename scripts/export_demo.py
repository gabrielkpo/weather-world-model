"""Export real data for an interactive demo page: simulator wind maps and model curves.

One test initial state of the QG world is run forward twice by the simulator (world A, and
world B started from the same state perturbed with epsilon = 1e-3), and the models of one
seed of the stationary QG run (one-step ``k1``, 8-step ``k8``, stochastic ``stoch``) are loaded
from their checkpoints and rolled out from the latent of the first frame. The dataset,
standardisation and probes are those of the run: they are rebuilt with the runner's own
functions.

Run: ``uv run python scripts/export_demo.py --checkpoints checkpoints/qg-stationary``.
Default output: ``runs/demo/demo.json`` (``runs/`` is git-ignored). Maps come from the
simulator; every curve except the ``truth`` ones comes from the models.
"""

import argparse
import base64
import json
import math
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from weather_world_model.calibration.run import _git_commit
from weather_world_model.diagnostics.predictability import saturation_distance
from weather_world_model.diagnostics.readouts import ReadoutProbes, fit_readout_probes
from weather_world_model.jepa import JEPAConfig, SequenceJEPA, encode_numpy
from weather_world_model.worlds.qg import (
    DIAGNOSTIC_NAMES,
    QGConfig,
    QGModel,
    diagnostics,
    velocity_saturation,
)
from weather_world_model.worlds.run_qg import (
    AMENDMENT3_MODELS,
    ENSEMBLE_MEMBERS,
    PROFILES,
    TORCH_THREADS,
    QGData,
    checkpoint_matches,
    log,
    simulate,
    train_or_load,
)

FloatArray = NDArray[np.float64]
EPSILON = 1e-3
N_FRAMES = 150
ARROW_STRIDE = 4
PHYSICS = ("kinetic_energy", "available_potential_energy", "enstrophy", "n_vortices")
READOUTS = ("kinetic_energy", "enstrophy")


# --------------------------------------------------------------------- pure helpers
def round_sig(value: float, digits: int = 4) -> float:
    """``value`` rounded to ``digits`` significant digits (zero and non-finite unchanged)."""
    if value == 0 or not math.isfinite(value):
        return float(value)
    return round(float(value), digits - 1 - math.floor(math.log10(abs(value))))


def rounded(values: Any) -> list[float]:
    """A 1D array as a list of floats with 4 significant digits."""
    return [round_sig(float(v)) for v in np.asarray(values).ravel()]


def encode_array(values: NDArray[np.floating], scale: float, dtype: str) -> str:
    """Quantise ``values / scale`` to ``uint8`` (range [0, 1]) or ``int8`` (range [-1, 1])
    and return the C-ordered bytes as base64 text."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    levels = {"uint8": 255, "int8": 127}[dtype]
    low = 0 if dtype == "uint8" else -levels
    q = np.clip(np.rint(np.asarray(values) / scale * levels), low, levels).astype(dtype)
    return base64.b64encode(q.tobytes()).decode("ascii")


def decode_array(text: str, shape: tuple[int, ...], scale: float, dtype: str) -> FloatArray:
    """Inverse of :func:`encode_array`, up to the quantisation step ``scale / levels``."""
    levels = {"uint8": 255, "int8": 127}[dtype]
    q = np.frombuffer(base64.b64decode(text), dtype=dtype).reshape(shape)
    return np.asarray(q, dtype=np.float64) * scale / levels


# --------------------------------------------------------------------- simulator
def run_worlds(
    start: QGModel, sample_every: int, n_frames: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Advance world A (``start``) and its perturbed twin B; same rule as ``twin_divergence``.

    Returns, per world, the upper-layer eddy winds ``(n_frames, 2, H, W)``; for A the true
    diagnostics per frame; and the RMS upper-layer velocity difference between A and B.
    """
    a, b = start.copy(), start.copy()
    q = a.state
    rms = np.sqrt(np.mean(q**2, axis=(1, 2), keepdims=True))
    b.state = q + EPSILON * rms * rng.standard_normal(q.shape)
    a.state = q  # same restart (no AB history) for both twins
    n = start.config.nx
    winds = {name: np.empty((n_frames, 2, n, n)) for name in ("A", "B")}
    diags = np.empty((n_frames, len(DIAGNOSTIC_NAMES)))
    separation = np.empty(n_frames)
    for t in range(n_frames):
        if t:
            a.step(sample_every)
            b.step(sample_every)
        for name, world in (("A", a), ("B", b)):
            u, v = world.velocity()
            winds[name][t] = np.stack([u[0], v[0]])
        d = diagnostics(a)
        diags[t] = [d[k] for k in DIAGNOSTIC_NAMES]
        diff = winds["A"][t] - winds["B"][t]
        separation[t] = np.sqrt(np.mean(np.sum(diff**2, axis=0)))
    return {"winds": winds, "diags": diags, "separation": separation}


def maps(winds: dict[str, FloatArray], background: float) -> dict[str, Any]:
    """Total upper-layer wind (eddy + background zonal flow) as compact byte arrays."""
    total = {k: w + np.array([background, 0.0])[None, :, None, None] for k, w in winds.items()}
    speed = {k: np.hypot(w[:, 0], w[:, 1]) for k, w in total.items()}
    speed_max = float(max(s.max() for s in speed.values()))
    sub = {k: w[:, :, ::ARROW_STRIDE, ::ARROW_STRIDE] for k, w in total.items()}
    uv_max = float(max(np.abs(s).max() for s in sub.values()))
    return {
        "speed_max": round_sig(speed_max, 6),
        "uv_max": round_sig(uv_max, 6),
        "worlds": {
            k: {
                "speed_b64": encode_array(speed[k], speed_max, "uint8"),
                "uv_b64": encode_array(sub[k], uv_max, "int8"),
            }
            for k in winds
        },
    }


# --------------------------------------------------------------------- models
def load_models(
    data: QGData, checkpoints: Path, seed: int, conv_norm: str = "group"
) -> dict[str, tuple[SequenceJEPA, ReadoutProbes, float]]:
    """Each model of the seed with its W-P probes and latent saturation distance, as in the run."""
    profile = PROFILES["full"]
    base = JEPAConfig(
        encoder="conv",
        input_shape=(2, *data.frames["train"].shape[-2:]),
        batch_size=profile.batch_size,
        conv_channels=profile.conv_channels,
        log_every=max(1, profile.train_steps // 50),
        conv_norm=conv_norm,  # type: ignore[arg-type]
    )
    flat_diags = {k: v.reshape(-1, v.shape[-1]) for k, v in data.diags.items()}
    out: dict[str, tuple[SequenceJEPA, ReadoutProbes, float]] = {}
    for name, k, noise_dim in AMENDMENT3_MODELS:
        path = checkpoints / f"seed{seed}-{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing: run run_qg --amendment3 first")
        cfg = replace(base, rollout_steps=k, noise_dim=noise_dim)
        # Check before train_or_load, which would retrain and overwrite a mismatched file.
        saved = torch.load(path, weights_only=True)
        if not checkpoint_matches(saved, cfg, profile, seed):
            raise RuntimeError(f"{path} does not match the run's configuration")
        del saved
        model, _ = train_or_load(cfg, data, profile, seed, name, path)
        lat = {s: encode_numpy(model, f, batch_size=256) for s, f in data.frames.items()}
        flat = {s: v.reshape(-1, v.shape[-1]) for s, v in lat.items()}
        probes = fit_readout_probes(flat, flat_diags, DIAGNOSTIC_NAMES)
        saturation = saturation_distance(flat["test"], np.random.default_rng(seed))
        out[name] = (model, probes, saturation)
    return out


def physical(probes: ReadoutProbes, latents: FloatArray, name: str) -> FloatArray:
    """Probe reading of quantity ``name`` in physical units, for latents ``(M, d)``."""
    q = probes.names.index(name)
    reading = probes.predict(latents)[:, q] * probes.y_std[q] + probes.y_mean[q]
    return np.asarray(reading, dtype=np.float64)


def rollout(
    model: SequenceJEPA, z0: FloatArray, k: int, noise: torch.Tensor | None = None
) -> FloatArray:
    """Latent rollout ``(B, k+1, d)`` as float64 (fresh noise at each call if ``noise`` is None)."""
    out = model.rollout(torch.as_tensor(z0, dtype=torch.float32), k, noise=noise)
    return np.asarray(out.numpy(), dtype=np.float64)


def build(
    checkpoints: Path, seed: int, start_index: int, n_frames: int, conv_norm: str = "group"
) -> dict[str, Any]:
    """Simulate, load the models and assemble the demo record."""
    torch.set_num_threads(TORCH_THREADS)
    profile = PROFILES["full"]
    data = simulate(profile)
    log(f"dataset rebuilt, {len(data.starts)} test start states")
    truth = run_worlds(
        data.starts[start_index], profile.sample_every, n_frames, np.random.default_rng(seed)
    )
    log("worlds A and B simulated")
    frames = {k: ((w - data.mean) / data.std).astype(np.float32) for k, w in truth["winds"].items()}
    diag = {name: truth["diags"][:, DIAGNOSTIC_NAMES.index(name)] for name in DIAGNOSTIC_NAMES}
    velocity_sat = float(np.median([velocity_saturation(m) for m in data.starts]))
    models = load_models(data, checkpoints, seed, conv_norm)
    steps = n_frames - 1
    generator = torch.Generator().manual_seed(seed)

    divergence: dict[str, list[float]] = {"truth": rounded(truth["separation"] / velocity_sat)}
    readout: dict[str, dict[str, list[float]]] = {q: {"truth": rounded(diag[q])} for q in READOUTS}
    saturations: dict[str, float] = {}
    ensemble: dict[str, Any] = {}
    for name, (model, probes, saturation) in models.items():
        saturations[name] = round_sig(saturation, 6)
        z_a = encode_numpy(model, frames["A"])
        z_b0 = encode_numpy(model, frames["B"][:1])
        stochastic = model.config.noise_dim > 0
        shared = (
            torch.randn((1, steps, model.config.noise_dim), generator=generator)
            if stochastic
            else None
        )
        roll_a = rollout(model, z_a[:1], steps, shared)[0]
        roll_b = rollout(model, z_b0, steps, shared)[0]
        key = "stoch_shared_noise" if stochastic else name
        divergence[key] = rounded(np.linalg.norm(roll_a - roll_b, axis=-1) / saturation)
        if stochastic:
            torch.manual_seed(seed)
            members = [
                rounded(physical(probes, rollout(model, z_a[:1], steps)[0], "kinetic_energy"))
                for _ in range(ENSEMBLE_MEMBERS)
            ]
            ensemble = {
                "kinetic_energy": {
                    "truth": rounded(diag["kinetic_energy"]),
                    "members": members,
                    "encoder": rounded(physical(probes, z_a, "kinetic_energy")),
                }
            }
            continue
        for q in READOUTS:
            suffix = "" if name == "k1" else f"_{name}"
            readout[q][name] = rounded(physical(probes, roll_a, q))
            readout[q][f"encoder{suffix}"] = rounded(physical(probes, z_a, q))
            readout[q][f"persistence{suffix}"] = rounded(
                np.repeat(physical(probes, z_a[:1], q), n_frames)
            )
        log(f"{name} curves done")
    return {
        "meta": {
            "code_commit": _git_commit(),
            "date": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": seed,
            "conv_norm": conv_norm,
            "models": [name for name, _, _ in AMENDMENT3_MODELS],
            "test_start_index": start_index,
            "epsilon": EPSILON,
            "note": (
                "Maps (worlds A and B) come from the simulator. Curves named truth come from "
                "the simulator; every other curve comes from the models: latent rollouts from "
                "the encoded first frame, read with the linear probes fitted in the run."
            ),
            "maps": (
                "speed = |(u + U1, v)| of the upper layer as uint8 * speed_max / 255, shape "
                "(n_frames, grid, grid), row = y; uv = (u + U1, v) every 4 points as int8 * "
                "uv_max / 127, shape (n_frames, 2, grid / 4, grid / 4)"
            ),
            "divergence": (
                "truth: RMS upper-layer velocity difference between A and B over the velocity "
                "saturation; models: latent distance between rollouts from the encoded first "
                "frames of A and B over the latent saturation distance (stochastic model: same "
                "noise sequence for both)"
            ),
            "readout": (
                "encoder and persistence without suffix use the k1 model; the _k8 suffix is "
                "the same reading with the k8 model's own encoder and probes"
            ),
            "latent_saturation_distance": saturations,
            "velocity_saturation": round_sig(velocity_sat, 6),
            "background_wind_U1": QGConfig().U1,
        },
        "dt": data.sample_dt,
        "n_frames": n_frames,
        "grid": int(truth["winds"]["A"].shape[-1]),
        **maps(truth["winds"], QGConfig().U1),
        "physics": {q: rounded(diag[q]) for q in PHYSICS},
        "divergence": divergence,
        "readout": readout,
        "ensemble": ensemble,
    }


def ranges(record: dict[str, Any]) -> dict[str, list[float]]:
    """Minimum and maximum of every curve family, for axis limits."""

    def span(*curves: list[float]) -> list[float]:
        flat = np.concatenate([np.asarray(c, dtype=np.float64) for c in curves])
        return [round_sig(float(flat.min())), round_sig(float(flat.max()))]

    out = {f"physics.{k}": span(v) for k, v in record["physics"].items()}
    out |= {f"divergence.{k}": span(v) for k, v in record["divergence"].items()}
    out |= {f"readout.{q}": span(*curves.values()) for q, curves in record["readout"].items()}
    ens = record["ensemble"]["kinetic_energy"]
    out["ensemble.kinetic_energy"] = span(ens["truth"], *ens["members"])
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        type=Path,
        default=None,
        help="default: checkpoints/qg-stationary, or checkpoints/qg-stationary-nonorm",
    )
    parser.add_argument("--conv-norm", choices=("group", "none"), default="group")
    parser.add_argument("--out", type=Path, default=Path("runs/demo/demo.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0, help="index of the test start state")
    parser.add_argument("--frames", type=int, default=N_FRAMES)
    args = parser.parse_args(argv)
    suffix = "-nonorm" if args.conv_norm == "none" else ""
    checkpoints = args.checkpoints or Path(f"checkpoints/qg-stationary{suffix}")
    record = build(checkpoints, args.seed, args.start, args.frames, args.conv_norm)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"saved {args.out.resolve()} ({args.out.stat().st_size / 1e6:.2f} MB)")
    print(json.dumps(ranges(record), indent=2))


if __name__ == "__main__":
    main()
