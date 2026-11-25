# weather-world-model

Probing a weather world model with dynamical-systems diagnostics.

> **Status:** framing done, no code yet. Expect breaking changes.

## What this project does

1. **Model.** Adapt V-JEPA 2, a pretrained latent world model, to storm radar videos (SEVIR, vertically integrated liquid at 5-minute steps), with a small JEPA trained from scratch for comparison.
2. **Diagnostics.** Interrogate the latent predictor beyond forecast quality:
   - predictability (divergence of nearby latent trajectories, finite-time Lyapunov exponents);
   - growth and decay modes (linear / Koopman approximation of the latent dynamics);
   - arrow of time (can the latent tell forward from reversed sequences).
3. **Protocol.** Calibrate the diagnostics first on a simulated chaotic system with known answers (Lorenz-96), then confront them with severe storms reported in NOAA Storm Events and with a pixel-space forecaster (Earthformer).

## Limitations (by design)

- Not an operational forecasting or warning system.
- No image generation.
- The diagnostics measure the dynamics learned by the model, not the atmosphere directly.

## Development

Requires Python 3.12 and `pre-commit`.

```bash
pre-commit install
```

Pull requests use Conventional Commits titles and are squash-merged.
