"""Render a PNG strip of the upper-layer wind of the two-layer QG world.

Run with: uv run --with matplotlib python scripts/qg_preview.py [--seed 0] [--out PATH]
Default output: runs/qg_preview/qg_upper_wind.png (``runs/`` is git-ignored).
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from weather_world_model.worlds.qg import QGConfig, QGModel, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spin-up", type=int, default=10000)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--every", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("runs/qg_preview/qg_upper_wind.png"))
    args = parser.parse_args()

    config = QGConfig()
    model = QGModel(config, np.random.default_rng(args.seed))
    model.spin_up(args.spin_up)
    n = config.nx
    x = np.arange(n) * config.L / n
    stride = 4
    fig, axes = plt.subplots(1, args.frames, figsize=(3.2 * args.frames, 3.4))
    for ax in np.atleast_1d(axes):
        u, v = model.velocity()
        speed = np.hypot(u[0] + config.U1, v[0])
        ax.imshow(speed, origin="lower", extent=(0, config.L, 0, config.L), cmap="viridis")
        sl = (slice(None, None, stride), slice(None, None, stride))
        ax.quiver(x[::stride], x[::stride], (u[0] + config.U1)[sl], v[0][sl], color="white")
        d = diagnostics(model)
        ax.set_title(f"t={model.t:.0f}  vortices={d['n_vortices']:.0f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        model.step(args.every)
    fig.suptitle("Upper-layer total wind speed (U1 + eddy) and arrows")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
