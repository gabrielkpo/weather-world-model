"""Long-run kinetic energy of the two-layer QG world from a cold start.

The evidence behind protocol amendment 5: with the simulator defaults the kinetic energy
keeps rising for about 40 time units before it fluctuates around its equilibrium, so data
recorded after a spin-up of 20 time units come from a flow that is still spinning up.

Run: ``uv run python scripts/qg_energy_record.py --out results/qg``.
Writes ``qg-energy-long.json``: the kinetic energy (layer-weighted domain mean, as in
``diagnostics``) every 5 time units up to t = 300, for two seeds.
"""

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from weather_world_model.records import provenance
from weather_world_model.worlds.qg import QGConfig, QGModel, diagnostics

SEEDS = (0, 1)
T_END = 300.0
SAMPLE_TIME = 5.0
EQUILIBRIUM_FROM = 60.0  # the spin-up of protocol amendment 5, in time units


def energy_series(seed: int, config: QGConfig, t_end: float, sample_time: float) -> list[float]:
    """Kinetic energy at t = 0, sample_time, ..., t_end from a cold start."""
    model = QGModel(config, np.random.default_rng(seed))
    steps = round(sample_time / config.dt)
    series = [diagnostics(model)["kinetic_energy"]]
    for _ in range(round(t_end / sample_time)):
        model.step(steps)
        series.append(diagnostics(model)["kinetic_energy"])
    return series


def summary(times: list[float], series: list[float], start: float) -> dict[str, float] | None:
    """Mean, minimum and maximum of the series from time ``start`` on (None if it ends before)."""
    values = np.asarray([v for t, v in zip(times, series, strict=True) if t >= start])
    if values.size == 0:
        return None
    return {"mean": float(values.mean()), "min": float(values.min()), "max": float(values.max())}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--t-end", type=float, default=T_END)
    args = parser.parse_args(argv)
    config = QGConfig()
    times = [i * SAMPLE_TIME for i in range(round(args.t_end / SAMPLE_TIME) + 1)]
    runs = []
    for seed in SEEDS:
        series = energy_series(seed, config, args.t_end, SAMPLE_TIME)
        runs.append(
            {
                "seed": seed,
                "kinetic_energy": series,
                "from_t60": summary(times, series, EQUILIBRIUM_FROM),
            }
        )
        print(f"seed {seed}: " + " ".join(f"{v:.3f}" for v in series), flush=True)
    record = {
        **provenance(),
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "qg_config": asdict(config),
        "start": "cold start (the simulator's random initial state), no spin-up",
        "quantity": "kinetic_energy (layer-weighted domain mean)",
        "time_units_per_step": config.dt,
        "times": times,
        "runs": runs,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "qg-energy-long.json").write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
