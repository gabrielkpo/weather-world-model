"""Exploratory check: does a multi-step rollout loss restore chaotic error growth?

Trains the MLP JEPA on Lorenz-96 (sizes of the "full" calibration profile, one seed) with
``rollout_steps`` 1 and 8 and prints the latent separation growth rate of each next to the
reference leading Lyapunov exponent. Not a protocol result.

Run: ``uv run python scripts/compare_rollout_loss.py [--seed 0]``.
"""

from __future__ import annotations

import argparse
import time

from weather_world_model.calibration.lorenz96 import (
    leading_lyapunov_lorenz96,
    simulate_lorenz96_batch,
)
from weather_world_model.calibration.run import (
    PROFILES,
    _encoder,
    _growth,
    _latent_rollout,
)
from weather_world_model.diagnostics.predictability import saturation_distance, separation_curve
from weather_world_model.jepa import JEPAConfig, train_jepa
from weather_world_model.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 8])
    args = parser.parse_args()
    profile = PROFILES["full"]
    rng = seed_everything(args.seed)
    reference = leading_lyapunov_lorenz96(n_steps=profile.reference_steps, burn_in=5000, rng=rng)
    train = simulate_lorenz96_batch(profile.n_trajectories, profile.n_steps, rng=rng)
    frames = simulate_lorenz96_batch(profile.n_test, 1, rng=rng)[:, 0]
    mean, std = float(train.mean()), float(train.std())
    print(f"reference Lyapunov exponent: {reference:.3f}")
    for k in args.steps:
        cfg = JEPAConfig(input_dim=train.shape[2], rollout_steps=k)
        t0 = time.perf_counter()
        model, history = train_jepa(
            (train - mean) / std, cfg, n_steps=profile.train_steps, seed=args.seed
        )
        elapsed = time.perf_counter() - t0
        encode = _encoder(model, mean, std)
        eval_rng = seed_everything(args.seed + 1000)
        for eps in (1e-3, 1e-2):
            curve = separation_curve(
                encode,
                _latent_rollout(model),
                frames,
                epsilon=eps,
                k=profile.rollout_k,
                rng=eval_rng,
            )
            try:
                g = _growth(curve, saturation_distance(encode(frames), eval_rng))
                rate = f"{g['growth_rate']:.3f} (window {g['window']})"
            except ValueError as err:
                rate = f"n/a ({err})"
            print(
                f"k={k} eps={eps:g}: latent growth rate {rate}, "
                f"collapsed={history.collapsed}, train time {elapsed:.1f} s"
            )


if __name__ == "__main__":
    main()
