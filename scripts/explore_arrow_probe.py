"""Exploratory analysis: what does the arrow-of-time probe detect on the true QG wind fields?

Not a protocol result. It runs the probe of diagnostic C on true fields only (no model) and
varies the PCA dimension and the clip length, with two kinds of "reversed" clips:

- plain reversal: the clip is read backwards;
- reversal + x-mirror: the clip is read backwards, mirrored in x, and v changes sign. The
  inviscid two-layer QG equations are invariant under (t, x, v) -> (-t, -x, -v), so this
  keeps the direction in which waves propagate and leaves only true irreversibility
  (drag, dissipation) to tell the two directions apart.

Run: ``uv run python scripts/explore_arrow_probe.py --out results/exploratory``.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from weather_world_model.diagnostics.arrow import (
    arrow_of_time_dataset,
    clips,
    probe_accuracy,
    train_probe,
)
from weather_world_model.records import provenance
from weather_world_model.worlds.qg import QGConfig, simulate_snapshots

FloatArray = NDArray[np.float64]
N_RUNS, N_SNAPSHOTS, SAMPLE_EVERY = 120, 256, 20
SPIN_UP = 12000  # 60 time units: the flow is stationary (protocol amendment 5)
N_TRAIN_RUNS = 84
L2 = 1e-3  # the protocol's probe regularisation
PROBE_SEEDS = (0, 1, 2)


def dataset(
    forward: FloatArray, alternate: FloatArray, rng: np.random.Generator
) -> tuple[FloatArray, NDArray[np.int64]]:
    """Label 1 = forward clip; label 0 = time-reversed clip taken from ``alternate``."""
    return arrow_of_time_dataset(forward, rng, features="asymmetry", reversed_from=alternate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    t0 = time.time()

    data = simulate_snapshots(
        QGConfig(),
        n_trajectories=N_RUNS,
        n_steps=N_SNAPSHOTS * SAMPLE_EVERY,
        sample_every=SAMPLE_EVERY,
        spin_up=SPIN_UP,
        rng=np.random.default_rng(1234),
    )
    if data.shape[:2] != (N_RUNS, N_SNAPSHOTS):
        raise RuntimeError(f"unexpected snapshot array shape {data.shape}")
    mirrored = data[..., ::-1].copy()
    mirrored[:, :, 1] *= -1.0  # v = dpsi/dx changes sign under x -> -x
    flat = data.reshape(N_RUNS, N_SNAPSHOTS, -1)
    flat_mirrored = mirrored.reshape(N_RUNS, N_SNAPSHOTS, -1)

    mean = flat[:N_TRAIN_RUNS].reshape(-1, flat.shape[-1]).mean(axis=0)
    sample = (flat[:N_TRAIN_RUNS, ::8].reshape(-1, flat.shape[-1]) - mean).astype(np.float64)
    _, singular, vt = np.linalg.svd(sample, full_matrices=False)
    variance = singular**2 / np.sum(singular**2)

    def project(x: NDArray[np.float32], dim: int) -> FloatArray:
        basis = vt[:dim].T.astype(np.float32)
        return np.stack([(run - mean) @ basis for run in x]).astype(np.float64)

    # (training runs, snapshots used, test runs): a large sample, and the protocol's sizes
    sizes = ((N_TRAIN_RUNS, N_SNAPSHOTS, N_RUNS - N_TRAIN_RUNS), (56, 100, 12))
    rows = []
    for dim in (8, 16, 32, 64):
        proj, proj_mirrored = project(flat, dim), project(flat_mirrored, dim)
        for clip_len in (16, 32, 64, 128):
            stride = clip_len // 2
            for mode, alt in (("plain", proj), ("mirror", proj_mirrored)):
                for n_train, n_snap, n_test in sizes:
                    if n_snap < clip_len:
                        continue
                    parts = [
                        clips(x[sl, :n_snap], clip_len, stride)
                        for sl in (slice(0, n_train), slice(N_RUNS - n_test, N_RUNS))
                        for x in (proj, alt)
                    ]
                    accuracies = []
                    for seed in PROBE_SEEDS:
                        rng = np.random.default_rng(seed)
                        x_tr, y_tr = dataset(parts[0], parts[1], rng)
                        x_te, y_te = dataset(parts[2], parts[3], rng)
                        probe = train_probe(x_tr, y_tr, l2=L2, seed=seed)
                        accuracies.append(probe_accuracy(probe, x_te, y_te))
                    rows.append(
                        {
                            "pca_dim": dim,
                            "clip_len": clip_len,
                            "reversal": mode,
                            "train_runs": n_train,
                            "snapshots_per_run": n_snap,
                            "n_train_clips": len(y_tr),
                            "n_test_clips": len(y_te),
                            "accuracy_mean": float(np.mean(accuracies)),
                            "accuracy_min": float(np.min(accuracies)),
                            "accuracy_max": float(np.max(accuracies)),
                        }
                    )
                    print(rows[-1], flush=True)

    record = {
        "exploratory": True,
        **provenance(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "n_runs": N_RUNS,
            "snapshots_per_run": N_SNAPSHOTS,
            "sample_every": SAMPLE_EVERY,
            "spin_up": SPIN_UP,
            "field": "uv",
            "probe_l2": L2,
            "probe_seeds": list(PROBE_SEEDS),
        },
        "variance_explained": {str(k): float(variance[:k].sum()) for k in (8, 16, 32, 64)},
        "wall_seconds": round(time.time() - t0),
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "arrow-probe-qg.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
