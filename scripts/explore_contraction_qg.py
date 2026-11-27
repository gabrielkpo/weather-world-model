"""Exploratory analysis: does the Lorenz-96 fix for latent contraction carry over to the QG world?

Not a protocol result. On Lorenz-96 (``scripts/explore_contraction_lorenz.py``), variance and
covariance terms much stronger than the default, on the encoder latents only, brought the
one-step JEPA's growth rate from about 0.1 to 1.4 to 1.65 against 1.69. This script trains the
one-step model on the QG world of the step-2 rerun (profile ``full``, encoder without
normalisation, amendment 6) with those settings and measures W-A, W-P and W-C exactly as the
runner does. The reference is the one-step model of ``results/qg/qg-stationary-nonorm.json``
(same data seed and profile; default weights 1 and 0.04, latent 32).

Run: ``uv run python scripts/explore_contraction_qg.py --out results/exploratory``.
"""

import argparse
import json
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from weather_world_model.jepa import JEPAConfig
from weather_world_model.records import array_sha256, provenance
from weather_world_model.worlds.qg import DIAGNOSTIC_NAMES
from weather_world_model.worlds.qg_data import PROFILES, simulate, stationarity_check
from weather_world_model.worlds.qg_eval import evaluate_model, train_or_load, true_predictability

VARIANTS: dict[str, dict[str, float | int]] = {
    "k1_reg10": {"var_weight": 10.0, "cov_weight": 0.4, "latent_dim": 32},
    "k1_reg100": {"var_weight": 100.0, "cov_weight": 4.0, "latent_dim": 32},
    "k1_reg100_dim64": {"var_weight": 100.0, "cov_weight": 4.0, "latent_dim": 64},
    "k1_reg10_dim128": {"var_weight": 10.0, "cov_weight": 0.4, "latent_dim": 128},
    "k1_reg10_dim256": {"var_weight": 10.0, "cov_weight": 0.4, "latent_dim": 256},
}
LATENT_SWEEP = ("k1_reg10_dim128", "k1_reg10_dim256")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("results/exploratory"))
    parser.add_argument("--profile", choices=sorted(PROFILES), default="full")
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS))
    parser.add_argument("--latent-sweep", action="store_true", help="run LATENT_SWEEP only")
    args = parser.parse_args(argv)
    if args.latent_sweep:
        args.variants = list(LATENT_SWEEP)
    t0 = time.perf_counter()
    profile = PROFILES[args.profile]
    data = simulate(profile)
    ke = DIAGNOSTIC_NAMES.index("kinetic_energy")
    stationarity = stationarity_check(data.diags["train"][..., ke], profile.stationarity_tolerance)
    truth = true_predictability(data, profile, profile.seeds[0])
    print("truth", json.dumps(truth), flush=True)
    runs: dict[str, list[dict[str, Any]]] = {}
    for name in args.variants:
        runs[name] = []
        for seed in profile.seeds:
            cfg = JEPAConfig(
                encoder="conv",
                input_shape=(2, *data.frames["train"].shape[-2:]),
                batch_size=profile.batch_size,
                conv_channels=profile.conv_channels,
                log_every=max(1, profile.train_steps // 50),
                conv_norm="none",
                **VARIANTS[name],  # type: ignore[arg-type]
            )
            model, info = train_or_load(cfg, data, profile, seed, name, None)
            res = evaluate_model(model, data, profile, seed, dynamics=True)
            runs[name].append({"seed": seed, **info, **res})
            pred = res["predictability"]
            print(name, seed, json.dumps({k: v.get("growth_rate") for k, v in pred.items()
                                          if isinstance(v, dict)}), flush=True)  # fmt: skip
    record = {
        **provenance(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "profile": args.profile,
        "config": asdict(profile),
        "conv_norm": "none",
        "data_sha256": array_sha256(data.frames["train"]),
        "variants": {n: VARIANTS[n] for n in args.variants},
        "stationarity": stationarity,
        "truth": truth,
        "runs": runs,
        "wall_seconds": time.perf_counter() - t0,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = ("-latent" if args.latent_sweep else "") + (
        "" if args.profile == "full" else f"-{args.profile}"
    )
    path = args.out / f"contraction-qg{suffix}.json"
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"wrote {path} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
