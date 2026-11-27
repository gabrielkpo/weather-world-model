"""Exploratory analysis: why are energies not readable from the latents on a stationary flow?

Not a protocol result. It tests one explanation: the GroupNorm layers of the convolutional
encoder rescale every sample and so remove the overall amplitude of the wind field, which is
what kinetic energy and enstrophy measure. On the stationary QG data of the full profile it
fits the W-P ridge probes on:

- two hand-made feature sets of the standardised fields (no model): the per-channel mean of
  squares, which is the amplitude itself, and the first 32 PCA components, which are linear
  in the field;
- the convolutional encoder with and without GroupNorm, with random weights and after
  one-step JEPA training (same steps and data as the full profile).

Run: ``uv run python scripts/explore_energy_readout.py --out results/exploratory``.
"""

import argparse
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from weather_world_model.diagnostics.readouts import fit_readout_probes
from weather_world_model.jepa import JEPAConfig, SequenceJEPA, encode_numpy, train_jepa
from weather_world_model.records import provenance
from weather_world_model.seeding import seed_everything
from weather_world_model.worlds.qg import DIAGNOSTIC_NAMES
from weather_world_model.worlds.run_qg import PROFILES, simulate

SEEDS = (0, 1)


def _scores(features: dict[str, np.ndarray], quantities: dict[str, np.ndarray]) -> dict[str, float]:
    probes = fit_readout_probes(features, quantities, DIAGNOSTIC_NAMES)
    return {name: float(probes.scores[name]["test_r2"]) for name in DIAGNOSTIC_NAMES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()
    profile = PROFILES["full"]
    data = simulate(profile)
    print(f"simulated in {time.time() - t0:.0f}s", flush=True)
    quantities = {k: v.reshape(-1, v.shape[-1]).astype(np.float64) for k, v in data.diags.items()}
    flat = {k: v.reshape(-1, *v.shape[2:]) for k, v in data.frames.items()}

    rows: list[dict[str, object]] = []

    def add(name: str, seed: int | None, features: dict[str, np.ndarray], **extra: object) -> None:
        rows.append(
            {"features": name, "seed": seed, **extra, "test_r2": _scores(features, quantities)}
        )
        print(rows[-1], flush=True)

    # hand-made features of the standardised fields, no model
    add(
        "mean of squares per channel",
        None,
        {k: (v.astype(np.float64) ** 2).mean(axis=(2, 3)) for k, v in flat.items()},
    )
    train2d = flat["train"][::4].reshape(-1, flat["train"][0].size).astype(np.float64)
    mean = train2d.mean(axis=0)
    _, _, vt = np.linalg.svd(train2d - mean, full_matrices=False)
    basis = vt[:32].T
    add(
        "first 32 PCA components (linear in the field)",
        None,
        {k: (v.reshape(len(v), -1) - mean) @ basis for k, v in flat.items()},
    )

    base = JEPAConfig(
        encoder="conv",
        input_shape=(2, *data.frames["train"].shape[-2:]),
        batch_size=profile.batch_size,
        conv_channels=profile.conv_channels,
        log_every=max(1, profile.train_steps // 50),
    )
    for norm in ("group", "none"):
        cfg = replace(base, conv_norm=norm)
        for seed in SEEDS:
            seed_everything(seed)
            random_model = SequenceJEPA(cfg)
            add(
                "conv encoder, random weights",
                seed,
                {k: encode_numpy(random_model, v, batch_size=256) for k, v in flat.items()},
                conv_norm=norm,
            )
            t1 = time.time()
            model, history = train_jepa(
                data.frames["train"], cfg, n_steps=profile.train_steps, seed=seed
            )
            add(
                "conv encoder, one-step JEPA",
                seed,
                {k: encode_numpy(model, v, batch_size=256) for k, v in flat.items()},
                conv_norm=norm,
                final_latent_std=float(history.latent_std[-1]),
                final_loss=float(history.loss[-1]),
                collapsed=bool(history.collapsed),
                train_seconds=round(time.time() - t1),
            )

    record = {
        "exploratory": True,
        **provenance(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "profile": "full",
        "train_steps": profile.train_steps,
        "seeds": list(SEEDS),
        "quantities": list(DIAGNOSTIC_NAMES),
        "wall_seconds": round(time.time() - t0),
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "energy-readout-qg.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
