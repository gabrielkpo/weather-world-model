# Results

Every number below is read from a versioned run record. Hypotheses are those of [PROTOCOL.md](PROTOCOL.md), including amendment 1.

## Step 2: calibration on simulated systems

Run record: [`results/calibration/calibration-full.json`](../results/calibration/calibration-full.json) (code commit `b183320`, profile `full`, seeds 0, 1, 2, CPU). Reproduce with:

```bash
uv run python -m weather_world_model.calibration.run --profile full --out results/calibration
```

The calibration model is the small frame-level JEPA (MLP encoder, residual MLP predictor, EMA target, variance-covariance regularisation, latent dimension 32, 4 000 training steps). No run collapsed. Each diagnostic is also run on the true system ("oracle": identity encoder and true dynamics) to separate errors of the diagnostic code from errors of the learned model; only the JEPA values decide the hypotheses.

| Hypothesis | Criterion | Oracle | Small JEPA (3 seeds) | Verdict |
|---|---|---|---|---|
| H0-A, predictability | growth rate within 30 % of the leading Lyapunov exponent of Lorenz-96 (reference 1.68 to 1.70 per time unit) | 1.70 to 1.74 | −0.03 to 0.13 (relative error 0.93 to 1.02) | **not supported** |
| H0-B, modes | DMD eigenvalue moduli within 10 % of the generating map | exact (error < 1e-4) | max relative error 0.055 to 0.074, one-step R² 0.91 to 0.92 | **supported**, with a caveat |
| H0-C, arrow of time | probe > 80 % on Lorenz-96 and < 60 % on a reversible oscillator | 100 % and 49 to 52 % | 91 to 95 % and 51 to 52 % | **supported** |

### What this says

- **The diagnostics themselves work.** On the true systems, diagnostic A recovers the Lyapunov exponent, diagnostic B recovers the eigenvalues exactly, and diagnostic C separates irreversible from reversible dynamics.
- **The small JEPA does not reproduce chaotic error growth.** Its latent rollouts barely diverge: after 120 steps (6 time units), the normalised separation stays below 0.002 of the saturation level, while the true system saturates after about 3 time units. The learned predictor is contractive. This matches what is reported for AI weather models and for learned Lorenz-96 emulators, whose error growth is too slow, and it is the question diagnostic A will ask of the radar models.
- **The modes criterion passes, but the growth is lost.** Moduli are within 7.4 % of the truth, yet the growing pair (modulus 1.02) is estimated at 0.96 to 0.97: below one, so the latent dynamics damp a mode that grows. A criterion on modulus error alone hides this, so the run record now also states, for each mode, whether growth stays growth and decay stays decay (`growth_preserved`): the growing pair is lost in all three seeds, the decaying and near-unit pairs are kept.
- **The arrow of time is readable in the latent space,** with clip statistics that change sign under time reversal. The same probe on raw latent differences stays at chance (46 to 49 % on Lorenz-96), as expected for a stationary process.

### Consequence for the protocol

H0-A is not supported, so, as written in the protocol, diagnostic A is not applied to radar in this form. Next options, to be decided before any radar measurement and logged as an amendment:

1. treat the contraction itself as the finding and measure it on the radar models with the oracle-validated code (the diagnostic is correct; the model is what fails);
2. train the calibration JEPA with a multi-step rollout loss and check whether chaotic growth appears;
3. drop diagnostic A.

Decision (2026-11-26, protocol amendment 2): option 2 is taken. Every later model is trained with both a one-step and a multi-step latent rollout loss, the multi-step model is rerun on Lorenz-96, and diagnostic B is withdrawn.

## Step 2: quasi-geostrophic world

Run record: [`results/qg/qg-full.json`](../results/qg/qg-full.json) (code commit `e18950d`, profile `full`, seeds 0, 1, 2, CPU, 1 h 49 min wall time). Reproduce with:

```bash
uv run python -m weather_world_model.worlds.run_qg --profile full --out results/qg
```

Hypotheses W-A, W-C and W-P of protocol amendment 2. W-V needs a GPU and is not measured here; W-R, L-S and W-S (amendment 3) are measured in a later run.

Setup: 80 independent runs of the two-layer QG model, split by run 56 / 12 / 12 (train / validation / test), 100 snapshots per run every 0.1 time units after a spin-up of 4 000 steps, field = upper-layer eddy winds `u, v` (2 × 64 × 64), standardised per channel with training statistics. Per seed, the convolutional JEPA (channels 32-64-96-128, latent dimension 32, batch 64, 3 000 steps) is trained with a one-step loss (k = 1) and with an 8-step latent rollout loss (k = 8) on the same data; the same encoder with random weights is the W-P control. No model collapsed (final latent std 0.98 to 1.04).

| Hypothesis | Criterion | Truth or oracle | k = 1 | k = 8 | Random weights | Verdict |
|---|---|---|---|---|---|---|
| W-A, error growth (ε = 1e-3) | k = 8 growth rate within 30 % of truth and closer than k = 1 | 0.71 per time unit | 0.20 to 0.26 (relative error 0.64 to 0.72) | −0.06 to 0.004 (relative error 0.99 to 1.09) | n/a | **not supported** |
| W-A, error growth (ε = 1e-2, reported) | same | 0.67 | 0.19 to 0.28 | −0.03 to −0.02 | n/a | (not supported) |
| W-C, arrow of time | probe accuracy > 80 % | 76.5 % (true fields) | 75.8 to 77.3 % | 75.0 to 78.8 % | n/a | **not supported** |
| W-P, kinetic energy | R² ≥ 0.7 and ≥ 0.1 above random | | 0.70 to 0.74 | 0.73 to 0.80 | 0.31 to 0.61 | k = 1 fails (margin), k = 8 passes |
| W-P, available potential energy | same | | 0.76 to 0.79 | 0.76 to 0.79 | 0.35 to 0.59 | passes |
| W-P, enstrophy | same | | 0.81 to 0.85 | 0.82 to 0.87 | 0.39 to 0.70 | passes |
| W-P, jet speed | same | | 0.34 to 0.53 | 0.39 to 0.46 | 0.32 to 0.54 | fails |
| W-P, max cyclonic vorticity | reported | | 0.69 to 0.73 | 0.67 to 0.73 | 0.37 to 0.62 | |
| W-P, max anticyclonic vorticity | reported | | 0.63 to 0.73 | 0.72 to 0.75 | 0.43 to 0.61 | |
| W-P, vortex count | reported | | 0.20 to 0.21 | 0.20 to 0.23 | 0.07 to 0.15 | |
| W-P overall | all four thresholded quantities pass | | | | | **not supported** |

Ranges are minimum to maximum over the three seeds; verdicts use the worst seed.

### What this says

- **Multi-step training does not restore chaotic error growth; it makes it worse.** The simulator's twin errors grow at about 0.7 per time unit and reach 30 % of saturation after about 10 time units. The one-step model's latent separation grows at less than a third of that rate, and the 8-step model's separation does not grow at all (rates around zero, normalised separation below 0.001 after 15 time units). Neither latent curve reaches the 0.3 ceiling within the 150 steps, so their rates are fitted over the whole rollout. This repeats the Lorenz-96 finding: the learned predictors are contractive, and a rollout loss pushes them further towards smooth, non-diverging forecasts.
- **The arrow of time is not readable at the 80 % level, but neither is it from the true fields.** The probe on the true wind fields (same PCA to 8 dimensions, same clips of 16 snapshots) reaches 76.5 %, and both models match it (75 to 79 %). The latents keep the time asymmetry that this probe can see in the data; the criterion fails because the probe itself tops out below 80 % on this world, not because the encoder loses information.
- **Energies and enstrophy are linearly readable; the jet is not.** Kinetic energy, available potential energy and enstrophy are read with R² 0.70 to 0.87, well above the random-weight encoder (which already reaches 0.3 to 0.7, since a random convolutional net keeps much of the field's amplitude). The one-step model misses the kinetic-energy margin on one seed. Maximum zonal-mean jet speed is no better than random (about 0.4), so W-P fails for both models. Vortex count is barely readable (0.2).
