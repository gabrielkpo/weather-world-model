"""Exploratory analysis: can a training change make the small JEPA keep Lorenz-96's chaos?

Not a protocol result. On Lorenz-96 the one-step JEPA's latent dynamics contract (growth rate
about 0 against a leading Lyapunov exponent of about 1.7) and the stochastic predictor of
amendment 3 contracts more. This script trains variants on the calibration data and measures the
same growth rate (diagnostic A), with the same windows and fits:

- ``det_k1``, ``stoch_k1``: the calibration models (references);
- ``det_k1_strong_reg``, ``stoch_k1_strong_reg``: variance and covariance terms ten times
  stronger, also applied to the predicted latents (against collapse and information loss);
- ``stoch_k8``: stochastic predictor trained with the energy score over 8-step rollouts, so a
  rollout is not penalised for diverging as long as its spread covers the truth;
- ``stoch_state_k1``, ``stoch_state_k8``: the noise perturbs the latent state before the
  deterministic dynamics, ``z' = z + W eps``, ``z_next = z' + f(z')``, so spread can only come
  from the dynamics;
- ``det_k1_forcing``: the deterministic model trained on trajectories with forcing 6 to 10
  (the regime changes between trajectories), measured at forcing 8; a ridge probe reads the
  forcing from the latents.

Stochastic models are measured by their sensitivity (twin rollouts sharing one noise sequence)
and by their ensemble spread, as in the calibration.

Run: ``uv run python scripts/explore_contraction_lorenz.py --out results/exploratory``.
"""

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from weather_world_model.calibration.lorenz96 import (
    leading_lyapunov_lorenz96,
    simulate_lorenz96_batch,
)
from weather_world_model.calibration.run import PROFILES, SAMPLE_DT
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
    _covariance_loss,
    _mlp,
    _variance_loss,
    energy_score,
    latent_rollout,
    mean_latent_std,
    standardised_encoder,
)
from weather_world_model.records import array_sha256, provenance
from weather_world_model.seeding import seed_everything

FloatArray = NDArray[np.float64]
FORCINGS = (6.0, 7.0, 8.0, 9.0, 10.0)


@dataclass(frozen=True)
class Variant:
    """One training variant."""

    stochastic: bool = False
    noise: str = "output"  # "output": noise is a predictor input; "state": it perturbs z
    k: int = 1  # rollout steps of the training loss
    var_weight: float = 1.0
    cov_weight: float = 0.04
    reg_on_predictions: bool = False
    varied_forcing: bool = False


VARIANTS: dict[str, Variant] = {
    "det_k1": Variant(),
    "stoch_k1": Variant(stochastic=True),
    "det_k1_strong_reg": Variant(var_weight=10.0, cov_weight=0.4, reg_on_predictions=True),
    "stoch_k1_strong_reg": Variant(
        stochastic=True, var_weight=10.0, cov_weight=0.4, reg_on_predictions=True
    ),
    "stoch_k8": Variant(stochastic=True, k=8),
    "stoch_state_k1": Variant(stochastic=True, noise="state"),
    "stoch_state_k8": Variant(stochastic=True, noise="state", k=8),
    "det_k1_forcing": Variant(varied_forcing=True),
}


class VariantJEPA(SequenceJEPA):
    """:class:`SequenceJEPA` with the losses and noise placement of a :class:`Variant`."""

    def __init__(self, config: JEPAConfig, variant: Variant) -> None:
        super().__init__(config)
        self.variant = variant
        if variant.stochastic and variant.noise == "state":
            self.predictor = _mlp(config.latent_dim, config.hidden_dim, config.latent_dim)
            self.noise_in = nn.Linear(config.noise_dim, config.latent_dim, bias=False)

    def _step(self, z: Tensor, noise: Tensor | None) -> Tensor:
        if self.variant.stochastic and self.variant.noise == "state":
            if noise is None:
                raise ValueError("a stochastic predictor needs a noise tensor")
            z = z + self.noise_in(noise.to(z))
            out: Tensor = z + self.predictor(z)
            return out
        return super()._step(z, noise)

    def _regulariser(self, z: Tensor) -> Tensor:
        v = self.variant
        return v.var_weight * _variance_loss(z) + v.cov_weight * _covariance_loss(z)

    def variant_loss(self, window: Tensor) -> tuple[Tensor, Tensor]:
        k = window.shape[1] - 1
        z = self.encoder(window[:, 0])
        with torch.no_grad():
            targets = self.target_encoder(window[:, 1:])
        total = self._regulariser(z)
        p1 = p2 = z
        for j in range(k):
            if self.variant.stochastic:
                p1 = self._step(p1, self._draw(p1, None))
                p2 = self._step(p2, self._draw(p2, None))
                total = total + energy_score(p1, p2, targets[:, j]) / k
            else:
                p1 = self._step(p1, None)
                total = total + nn.functional.l1_loss(p1, targets[:, j]) / k
            if self.variant.reg_on_predictions:
                total = total + self._regulariser(p1) / k
        return total, z


def train_variant(
    sequences: FloatArray, variant: Variant, *, n_steps: int, seed: int
) -> tuple[VariantJEPA, list[float]]:
    """As :func:`weather_world_model.jepa.train_jepa`, with the variant's loss."""
    cfg = JEPAConfig(input_dim=sequences.shape[2], noise_dim=NOISE_DIM if variant.stochastic else 0)
    rng = seed_everything(seed)
    data = torch.as_tensor(sequences, dtype=torch.float32)
    k = variant.k
    per_seq = data.shape[1] - k
    n_windows = data.shape[0] * per_seq
    offsets = torch.arange(k + 1)
    model = VariantJEPA(cfg, variant)
    params = [p for n, p in model.named_parameters() if not n.startswith("target_encoder")]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    stds: list[float] = []
    model.train()
    for step in range(n_steps):
        idx = torch.as_tensor(rng.integers(0, n_windows, size=min(cfg.batch_size, n_windows)))
        seq, start = idx // per_seq, idx % per_seq
        loss, z = model.variant_loss(data[seq[:, None], start[:, None] + offsets])
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        model.update_target()
        if step % 100 == 0 or step == n_steps - 1:
            stds.append(mean_latent_std(z))
    return model, stds


def ridge_r2(x_tr: FloatArray, y_tr: FloatArray, x_te: FloatArray, y_te: FloatArray) -> float:
    """Test R² of a ridge regression (alpha 1e-3, standardised inputs)."""
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-8
    a, b = (x_tr - mu) / sd, (x_te - mu) / sd
    a1 = np.hstack([a, np.ones((len(a), 1))])
    w = np.linalg.solve(a1.T @ a1 + 1e-3 * np.eye(a1.shape[1]), a1.T @ y_tr)
    pred = np.hstack([b, np.ones((len(b), 1))]) @ w
    return float(1 - ((pred - y_te) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum())


def measure(
    name: str, variant: Variant, profile_name: str, seed: int, reference: float
) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    rng = seed_everything(seed)
    if variant.varied_forcing:
        per = profile.n_trajectories // len(FORCINGS)
        parts = [
            simulate_lorenz96_batch(per, profile.n_steps, forcing=f, rng=rng) for f in FORCINGS
        ]
        train = np.concatenate(parts)
        forcing_of = np.repeat(FORCINGS, per)
    else:
        train = simulate_lorenz96_batch(profile.n_trajectories, profile.n_steps, rng=rng)
    mean, std = float(train.mean()), float(train.std())
    standardised = (train - mean) / std
    t0 = time.perf_counter()
    model, stds = train_variant(standardised, variant, n_steps=profile.train_steps, seed=seed)
    out: dict[str, Any] = {
        "seed": seed,
        "data_sha256": array_sha256(standardised),
        "train_seconds": time.perf_counter() - t0,
        "final_latent_std": stds[-1],
        "collapsed": stds[-1] < 1e-2,
    }
    frames = simulate_lorenz96_batch(profile.n_test, 1, rng=rng)[:, 0]
    encode = standardised_encoder(model, mean, std)
    z0 = encode(frames)
    saturation = saturation_distance(z0, rng)
    k = profile.rollout_k
    if variant.stochastic:
        generator = torch.Generator().manual_seed(seed)
        shared = torch.randn((frames.shape[0], k, NOISE_DIM), generator=generator)

        def shared_rollout(z: FloatArray, steps: int) -> FloatArray:
            r = model.rollout(torch.as_tensor(z, dtype=torch.float32), steps, noise=shared)
            return np.asarray(r.numpy(), dtype=np.float64)

        def sampled_rollout(z: FloatArray, steps: int) -> FloatArray:
            r = model.rollout(torch.as_tensor(z, dtype=torch.float32), steps, generator=generator)
            return np.asarray(r.numpy(), dtype=np.float64)

        curve = separation_curve(encode, shared_rollout, frames, epsilon=1e-3, k=k, rng=rng)
        spread = ensemble_spread_curve(sampled_rollout, z0, k=k, n_members=ENSEMBLE_MEMBERS)
        out["growth_rate"] = growth_fit(curve, saturation, dt=SAMPLE_DT, safe=True)["growth_rate"]
        out["spread_growth_rate"] = growth_fit(spread, saturation, dt=SAMPLE_DT, safe=True)[
            "growth_rate"
        ]
    else:
        curve = separation_curve(encode, latent_rollout(model), frames, epsilon=1e-3, k=k, rng=rng)
        out["growth_rate"] = growth_fit(curve, saturation, dt=SAMPLE_DT, safe=True)["growth_rate"]
    if variant.varied_forcing:
        test = np.concatenate(
            [simulate_lorenz96_batch(8, 20, forcing=f, rng=rng) for f in FORCINGS]
        )
        y_test = np.repeat(FORCINGS, 8 * 20).astype(np.float64)
        z_train = encode(train[:, ::10].reshape(-1, train.shape[2]))
        y_train = np.repeat(forcing_of, train[:, ::10].shape[1]).astype(np.float64)
        z_test = encode(test.reshape(-1, test.shape[2]))
        out["forcing_readout_r2"] = ridge_r2(z_train, y_train, z_test, y_test)
    out["relative_error"] = (
        None if out["growth_rate"] is None else abs(out["growth_rate"] - reference) / reference
    )
    print(f"{name} seed {seed}: {json.dumps({k: v for k, v in out.items() if k != 'data_sha256'})}",
          flush=True)  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results/exploratory"))
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = parser.parse_args(argv)
    profile = PROFILES[args.profile]
    t0 = time.perf_counter()
    references = {}
    for seed in profile.seeds:
        rng = seed_everything(seed)
        references[seed] = leading_lyapunov_lorenz96(
            n_steps=profile.reference_steps, burn_in=5000, rng=rng
        )
    print("reference Lyapunov", references, flush=True)
    runs = {
        name: [measure(name, VARIANTS[name], args.profile, s, references[s]) for s in profile.seeds]
        for name in args.variants
    }
    summary = {
        name: {
            "growth_rate": [r["growth_rate"] for r in rs],
            "spread_growth_rate": [r.get("spread_growth_rate") for r in rs],
            "collapsed": [r["collapsed"] for r in rs],
            "forcing_readout_r2": [r.get("forcing_readout_r2") for r in rs],
        }
        for name, rs in runs.items()
    }
    record = {
        **provenance(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "profile": args.profile,
        "seeds": list(profile.seeds),
        "reference_lyapunov": references,
        "variants": {n: VARIANTS[n].__dict__ for n in args.variants},
        "noise_dim": NOISE_DIM,
        "epsilon": 1e-3,
        "summary": summary,
        "runs": runs,
        "wall_seconds": time.perf_counter() - t0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.profile == "full" else f"-{args.profile}"
    path = args.out / f"contraction-lorenz{suffix}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=1))
    print(f"wrote {path} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
