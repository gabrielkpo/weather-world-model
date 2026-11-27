"""Step-2 measurements on the two-layer QG world (protocol amendment 2: W-A, W-C, W-P).

Run: ``uv run python -m weather_world_model.worlds.run_qg --profile full --out results/qg``.
Writes one JSON record (code commit and dirty flag, date, configuration, fingerprint
``data_sha256`` of the standardised training array, per-seed values, verdicts). Each seed
entry also carries the commit and dirty flag of the code that produced it, so that a resumed
run keeps every seed's own provenance.

The code is split into :mod:`~weather_world_model.worlds.qg_data` (simulation, splits,
standardisation, mirroring), :mod:`~weather_world_model.worlds.qg_eval` (training and
per-model diagnostics), :mod:`~weather_world_model.worlds.qg_verdicts` (criteria) and this
command-line runner, which re-exports their public names.

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
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from weather_world_model.diagnostics.predictability import ENSEMBLE_MEMBERS
from weather_world_model.jepa import NOISE_DIM
from weather_world_model.records import array_sha256, provenance
from weather_world_model.worlds.qg import DIAGNOSTIC_NAMES, THRESHOLDED, QGConfig
from weather_world_model.worlds.qg_data import (
    PROFILES,
    STATIONARITY_TOLERANCE,
    Profile,
    QGData,
    log,
    mirror_x,
    peak_rss_mb,
    simulate,
    split_sizes,
    stationarity_check,
)
from weather_world_model.worlds.qg_eval import (
    AMENDMENT3_MODELS,
    BASE_MODELS,
    CURVE_POINTS,
    EPSILONS,
    HORIZONS,
    READOUT_SAMPLES,
    ConvNorm,
    arrow_probe,
    checkpoint_matches,
    clips,
    evaluate_model,
    latent_predictability,
    run_seed,
    train_or_load,
    true_predictability,
)
from weather_world_model.worlds.qg_verdicts import verdicts, verdicts_amendment3

__all__ = [
    "AMENDMENT3_MODELS",
    "BASE_MODELS",
    "CURVE_POINTS",
    "ENSEMBLE_MEMBERS",
    "EPSILONS",
    "HORIZONS",
    "NOISE_DIM",
    "PROFILES",
    "READOUT_SAMPLES",
    "STATIONARITY_TOLERANCE",
    "THRESHOLDED",
    "TORCH_THREADS",
    "ConvNorm",
    "Profile",
    "QGData",
    "arrow_probe",
    "checkpoint_matches",
    "clips",
    "evaluate_model",
    "latent_predictability",
    "log",
    "main",
    "mirror_x",
    "peak_rss_mb",
    "run",
    "run_seed",
    "simulate",
    "split_sizes",
    "stationarity_check",
    "train_or_load",
    "true_predictability",
    "verdicts",
    "verdicts_amendment3",
]

TORCH_THREADS = 4
AMENDMENT3_NOTES: dict[str, Any] = {
    "noise_dim": NOISE_DIM,
    "horizons": list(HORIZONS),
    "readout_sampled_rollouts": READOUT_SAMPLES,
    "ensemble_members": ENSEMBLE_MEMBERS,
    "stochastic_readout": "mean of the probe outputs over the sampled rollouts",
    "sensitivity": "twin rollouts from clean and perturbed frames, same noise sequence",
    "spread": "mean pairwise distance of members from one state, independent noise",
}


def load_resume(partial: Path | None, profile: Profile, conv_norm: ConvNorm) -> dict[str, Any]:
    """The partial record to resume from, or ``{}`` if there is none or if it was written
    for another profile or encoder normalisation."""
    if partial is None or not partial.exists():
        return {}
    previous: dict[str, Any] = json.loads(partial.read_text())
    same_encoder = previous.get("conv_norm", "group") == conv_norm
    if same_encoder and previous.get("config") == json.loads(json.dumps(asdict(profile))):
        log(f"resuming from {partial}: seeds {[s['seed'] for s in previous['seeds']]} done")
        return previous
    return {}


def checked_stationarity(data: QGData, profile: Profile) -> dict[str, Any]:
    """Stationarity check of the training data (protocol amendment 5); raises if it fails."""
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
    return stationarity


def oracle_truth(data: QGData, profile: Profile) -> dict[str, Any]:
    """Simulator twin divergence (W-A) and the arrow-of-time probe on the true fields (W-C)."""
    predictability = true_predictability(data, profile, profile.data_seed)

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
    return {
        "predictability": predictability,
        "arrow_accuracy": oracle_arrow,
        "arrow_accuracy_mirrored": oracle_mirrored,
    }


def with_verdicts(rec: dict[str, Any], amendment3: bool) -> dict[str, Any]:
    """Add the verdicts to a complete record, with the oracle's W-C accuracies."""
    truth, seeds = rec["truth"], rec["seeds"]
    if amendment3:
        rec["verdicts"] = verdicts_amendment3(truth["predictability"], seeds)
        wc = rec["verdicts"]["retrained_k1_k8"]["W-C"]
    else:
        rec["verdicts"] = verdicts(truth["predictability"], seeds)
        wc = rec["verdicts"]["W-C"]
    wc["oracle_accuracy"] = truth["arrow_accuracy"]
    wc["oracle_accuracy_mirrored_control"] = truth["arrow_accuracy_mirrored"]
    return rec


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
    done = load_resume(partial, profile, conv_norm)
    data = simulate(profile)
    simulate_s = time.perf_counter() - t0
    log(f"simulated {profile.n_runs} runs in {simulate_s:.0f}s, peak RSS {peak_rss_mb():.0f} MB")
    stationarity = checked_stationarity(data, profile)
    truth = done["truth"] if "truth" in done else oracle_truth(data, profile)
    seeds: list[dict[str, Any]] = list(done.get("seeds", []))
    previous_wall = float(done.get("wall_seconds", 0.0))

    def record() -> dict[str, Any]:
        rec: dict[str, Any] = {
            **provenance(),
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
            "data_sha256": array_sha256(data.frames["train"]),
            "diagnostic_names": list(DIAGNOSTIC_NAMES),
            "torch_version": torch.__version__,
            "torch_threads": TORCH_THREADS,
            "simulate_seconds": simulate_s,
            "wall_seconds": previous_wall + time.perf_counter() - t0,
            "resumed": bool(done),
            "stationarity": stationarity,
            "truth": truth,
            "seeds": seeds,
        }
        if amendment3:
            rec["amendment3"] = dict(AMENDMENT3_NOTES)
        return rec

    for s in profile.seeds:
        if any(prev["seed"] == s for prev in seeds):
            continue
        result = run_seed(
            data, profile, s, amendment3=amendment3, checkpoints=checkpoints, conv_norm=conv_norm
        )
        # The commit is stored per seed: a resumed run keeps each seed's own provenance.
        seeds.append({**result, **provenance()})
        seeds.sort(key=lambda r: r["seed"])
        if partial is not None:
            partial.write_text(json.dumps(record(), indent=2) + "\n")
            log(f"seed {s} written to {partial}")
    return with_verdicts(record(), amendment3)


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
