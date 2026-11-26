# Results

Every number below is read from a versioned run record. Hypotheses are those of [PROTOCOL.md](PROTOCOL.md), including its amendments.

## Step 1: Lorenz-96 proof of concept

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
- *Caveat added on 2026-11-26 (protocol amendment 4):* on the QG world the same probe turned out to read the direction of wave propagation rather than irreversibility. Lorenz-96 waves also have a preferred direction and no mirrored control was run here, so the result below may reflect propagation only.
- **The arrow of time is readable in the latent space,** with clip statistics that change sign under time reversal. The same probe on raw latent differences stays at chance (46 to 49 % on Lorenz-96), as expected for a stationary process.

### Consequence for the protocol

H0-A is not supported, so, as written in the protocol, diagnostic A is not applied to radar in this form. Next options, to be decided before any radar measurement and logged as an amendment:

1. treat the contraction itself as the finding and measure it on the radar models with the oracle-validated code (the diagnostic is correct; the model is what fails);
2. train the calibration JEPA with a multi-step rollout loss and check whether chaotic growth appears;
3. drop diagnostic A.

Decision (2026-11-26, protocol amendment 2): option 2 is taken. Every later model is trained with both a one-step and a multi-step latent rollout loss, the multi-step model is rerun on Lorenz-96, and diagnostic B is withdrawn.

### Stochastic predictor (hypothesis L-S, protocol amendment 3)

Run record: [`results/calibration/calibration-stochastic.json`](../results/calibration/calibration-stochastic.json) (code commit `0907742`, profile `full`, seeds 0, 1, 2, CPU). Reproduce with:

```bash
uv run python -m weather_world_model.calibration.run --profile full --stochastic --out results/calibration
```

Per seed, two models are trained on the same Lorenz-96 data for the same 4 000 steps: the deterministic one-step JEPA of H0-A (its values in this record are identical to those of the calibration record) and a stochastic variant whose predictor also receives a noise vector of dimension 8 and is trained with the two-sample energy score. No run collapsed (final latent std 0.99 to 1.02). The stochastic model is measured in two ways: sensitivity (clean and perturbed rollouts driven by the same noise sequence, ε = 1e-3, growth-window rule of diagnostic A), which decides L-S, and ensemble spread (8 rollouts from the same state with independent noise, mean distance between pairs of members, same normalisation), which is reported.

| Seed | Lyapunov reference | Deterministic growth rate | Stochastic sensitivity growth rate | Stochastic relative error |
|---|---|---|---|---|
| 0 | 1.69 | 0.12 | −0.26 | 1.15 |
| 1 | 1.68 | −0.03 | −0.19 | 1.11 |
| 2 | 1.70 | 0.13 | −0.15 | 1.09 |

Rates are per time unit. **L-S is not supported**: the criterion asks for a sensitivity growth rate within 30 % of the Lyapunov exponent and closer to it than the deterministic model; the stochastic rate is negative on every seed and further from the reference than the deterministic one (relative error 1.09 to 1.15 against 0.93 to 1.02). At ε = 1e-2 the sensitivity rates are the same (−0.27 to −0.16).

Ensemble spread, as a fraction of the saturation distance (range over seeds): 0.20 to 0.21 after one step, 0.28 after two, 0.55 to 0.57 after 10 steps (0.5 time units), 0.81 to 0.90 after 40 steps, 0.96 to 1.11 after 120 steps. The spread passes the 0.3 ceiling of the growth-window rule at the third step, so the fitted rate (6.4 to 6.7 per time unit) rests on two points and is not an exponential growth rate.

#### What this says

- **Noise does not make the predictor sensitive to its initial state.** With the noise sequence held fixed, two rollouts started 0.1 % apart come closer over time, slightly faster than with the deterministic model: the stochastic predictor is still a contraction in its latent argument. A proper score for the next-step distribution does not, by itself, teach the map to amplify small differences.
- **The model does produce diverse futures, but through the noise, not through chaos.** Members of an ensemble started from the same state are a fifth of the saturation distance apart after a single step and reach the saturation distance after 2.5 to 6 time units. The true system needs about 4 time units to bring a 0.1 % perturbation to half of saturation, starting slowly and accelerating; the model's spread jumps at the first step and then grows ever more slowly. The uncertainty is injected at each step rather than grown from the initial error.
- **Consequence.** Neither the deterministic nor the stochastic one-step model reproduces chaotic error growth on Lorenz-96. The stochastic variant is still measured on the QG world (W-S), as planned.

## Step 2: quasi-geostrophic world

Run record: [`results/qg/qg-full.json`](../results/qg/qg-full.json) (code commit `e18950d`, profile `full`, seeds 0, 1, 2, CPU, 1 h 49 min wall time). Reproduce with:

```bash
uv run python -m weather_world_model.worlds.run_qg --profile full --out results/qg
```

Hypotheses W-A, W-C and W-P of protocol amendment 2. W-V needs a GPU and is not measured here; L-S (amendment 3) is reported in step 1 above; W-R and W-S (amendment 3) are reported in their own subsection below.

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

### Rollout readouts and stochastic predictor (hypotheses W-R and W-S, protocol amendment 3)

Run record: [`results/qg/qg-amendment3.json`](../results/qg/qg-amendment3.json) (code commit `311fafc`, profile `full`, seeds 0, 1, 2, CPU, 2 h 29 min of compute). Reproduce with:

```bash
uv run python -m weather_world_model.worlds.run_qg --profile full --amendment3 --out results/qg --checkpoints checkpoints
```

Same data and setup as above. Per seed, the k = 1 and k = 8 models are retrained with the same configuration and seed as in the first run (their W-A and W-P values in this record are identical to the first record, value for value), and a third model is added: the one-step model with a stochastic predictor (noise vector of dimension 8, two-sample energy score). Trained weights are saved outside Git. No model collapsed (stochastic model: final latent std 0.81 to 0.89).

**W-R, rollout readouts.** The W-P probes, fitted on latents of real frames, are applied to latents rolled out h snapshots by the predictor and scored against the simulator's true quantity at t + h (test runs). The control is persistence: the same probe on the latent of frame t. For the stochastic model the estimate is the mean of the probe outputs over 8 sampled rollouts. The criterion is on the gain, R² of the rollout minus R² of persistence, averaged over kinetic energy, available potential energy, enstrophy and jet speed, at h = 10 (one time unit): at least 0.05 for the k = 1 and the k = 8 model, on every seed.

| Mean gain over persistence (seeds 0, 1, 2) | h = 1 | h = 5 | h = 10 | Passes at h = 10 |
|---|---|---|---|---|
| k = 1 | 0.003, −0.013, 0.010 | −0.03, −0.35, −0.17 | −0.20, −1.57, −0.84 | no |
| k = 8 | −0.007, −0.003, −0.002 | −0.023, −0.004, 0.014 | 0.0495, 0.047, 0.101 | no (one seed of three) |
| stochastic (reported) | −0.003, −0.002, 0.012 | −0.23, −0.08, 0.04 | −1.18, −0.34, 0.08 | no threshold |

| R² at h = 10, rollout (persistence) | k = 1 | k = 8 | stochastic |
|---|---|---|---|
| kinetic energy | −0.62 to 0.51 (0.40 to 0.53) | 0.57 to 0.64 (0.53 to 0.58) | −0.43 to 0.52 (0.29 to 0.39) |
| available potential energy | −0.43 to 0.36 (0.52 to 0.60) | 0.59 to 0.64 (0.53 to 0.60) | 0.32 to 0.62 (0.51 to 0.54) |
| enstrophy | 0.08 to 0.67 (0.61 to 0.71) | 0.72 to 0.79 (0.64 to 0.70) | 0.34 to 0.68 (0.56 to 0.61) |
| jet speed | −3.93 to −0.57 (0.05 to 0.12) | 0.13 to 0.28 (0.11 to 0.16) | −3.64 to −0.02 (−0.05 to 0.13) |
| max cyclonic vorticity (reported) | −0.20 to 0.30 (0.54 to 0.58) | 0.62 to 0.67 (0.51 to 0.54) | 0.37 to 0.59 (0.50 to 0.61) |
| max anticyclonic vorticity (reported) | −2.40 to 0.04 (0.50 to 0.63) | 0.68 to 0.72 (0.58 to 0.64) | 0.49 to 0.59 (0.57 to 0.59) |
| vortex count (reported) | −0.29 to 0.00 (0.04 to 0.07) | 0.11 to 0.15 (0.09 to 0.12) | −0.25 to 0.09 (0.06 to 0.12) |

Ranges are minimum to maximum over the three seeds, each model against the persistence of its own encoder. Per-quantity values at h = 1 and h = 5 are in the record. **W-R is not supported**: the one-step model is far below persistence at h = 10 on every seed, and the 8-step model reaches the 0.05 gain on one seed only (0.0495 and 0.047 on the other two).

**W-S, stochastic predictor.** Sensitivity is the growth rate of the latent separation between rollouts from the clean and the perturbed start frame driven by the same noise sequence (ε = 1e-3, rule of diagnostic A); it decides W-S. Ensemble spread (8 rollouts from the same latent with independent noise, mean distance between pairs of members, same normalisation) is reported.

| Seed | Simulator twin divergence | Deterministic k = 1 | Stochastic sensitivity | Stochastic relative error |
|---|---|---|---|---|
| 0 | 0.71 | 0.20 | 0.12 | 0.83 |
| 1 | 0.71 | 0.23 | 0.08 | 0.89 |
| 2 | 0.71 | 0.26 | −0.004 | 1.01 |

Rates are per time unit. **W-S is not supported**: the criterion asks for a sensitivity growth rate within 30 % of the simulator's and closer to it than the deterministic one-step model; the stochastic rate is lower than the deterministic one on every seed (0.04 to 0.07 at ε = 1e-2). Ensemble spread, as a fraction of the saturation distance: 0.27 to 0.29 after 3 snapshots, 0.39 to 0.46 after 7, 0.77 to 1.13 after 50, and 1.8 to 3.9 after 150. The spread passes the 0.3 ceiling after about 3 snapshots, so its fitted rate (2.3 to 2.8 per time unit) rests on steps 1 to 3 and is not an exponential growth rate.

**Stochastic model on W-C and W-P (reported, no threshold).** Arrow-of-time probe: 76.5 %, 87.9 % and 78.0 % (true fields 76.5 %, deterministic models 75 to 79 %). As established by the exploratory analysis below (protocol amendment 4), this probe reads the direction of wave propagation, not irreversibility; the mirrored control of amendment 4 was not part of this run. Readouts on real frames (R²): kinetic energy 0.64 to 0.68, available potential energy 0.72 to 0.77, enstrophy 0.76 to 0.82, jet speed 0.37 to 0.42, maximum cyclonic vorticity 0.68 to 0.71, maximum anticyclonic vorticity 0.69 to 0.72, vortex count 0.20 to 0.22: slightly below the deterministic one-step model on kinetic energy and enstrophy, similar elsewhere.

The record's code commit is the last commit of the branch before it was squash-merged. The run was started one commit earlier (`595c191`); the only change in between is the form of the W-R pass test (`all(...)` instead of `min(...)`), which gives the same verdict on these values.

#### What this says

- **The one-step predictor does not carry the physics forward.** After one step its rollout reads as well as persistence, which only says that one step changes little. After 10 steps its readouts are much worse than assuming nothing changes, with negative R² on several quantities and seeds: iterating the predictor takes the latent to places where the probes no longer mean anything.
- **The 8-step predictor does slightly better than "no change", not enough for the criterion.** At h = 10 it beats persistence on every seed and on all seven quantities, by 0.05 to 0.10 on average over the four criterion quantities. Two seeds fall just under the 0.05 threshold (by 0.0005 and 0.003), so W-R fails as written. The honest reading is a small, consistent gain: this predictor moves energies and enstrophy in the right direction over one time unit, while being the model whose error growth is furthest from the truth (W-A).
- **Noise does not restore sensitivity to initial conditions on the QG world either.** With the noise held fixed, the stochastic model's latent separation grows at 0.12, 0.08 and −0.004 per time unit, against 0.71 for the simulator and 0.20 to 0.26 for the deterministic one-step model. As on Lorenz-96, the stochastic predictor is at least as contractive as the deterministic one.
- **Its ensembles spread, and then keep spreading past the data.** Members started from the same state are about 0.3 of the saturation distance apart after 3 snapshots, and 1.8 to 3.9 times the saturation distance apart after 150: further from each other than two unrelated real states. The diversity comes from the noise injected at each step and accumulates without bound, instead of growing from the initial error and saturating as in the simulator. Its mean rollout readouts are unstable across seeds at h = 10 (gain −1.18 to 0.08).
- **Taken together:** none of the three training variants (one-step, 8-step, stochastic) reproduces the simulator's error growth, and only the 8-step model shows any evidence of predicting physical quantities beyond persistence, by a small margin.

### Exploratory: what the arrow-of-time probe detects (protocol amendment 4)

Not a protocol result. Record: [`results/exploratory/arrow-probe-qg.json`](../results/exploratory/arrow-probe-qg.json) (code commit `a00bd99`, true fields only, 120 new runs of 256 snapshots, 84 training and 36 test runs). Reproduce with `uv run python scripts/explore_arrow_probe.py --out results/exploratory`.

| PCA components, clip length | Test clips | Plain reversal | Reversal + x-mirror |
|---|---|---|---|
| 8, 16 | 1 116 | 89.7 % | 49.0 % |
| 8, 64 | 252 | 93.4 % | 51.5 % |
| 32, 16 | 1 116 | 98.7 % | 47.0 % |
| 32, 64 | 252 | 100 % | 45.8 % |

- **The probe reads the direction of wave propagation, not irreversibility.** Keeping more components or longer clips lifts it from 90 % to 100 % against plain reversal, but it is at chance (43 to 55 % across the 16 settings of the record) once the reversed clip is mirrored in x (with v changing sign), which keeps the direction in which waves travel. The W-C numbers above therefore say that the latents keep the propagation direction, and nothing about irreversibility.
- **The first run's 76.5 % ceiling is not explained.** With the protocol's sizes (56 training runs, 100 snapshots, 132 test clips) this analysis gives 86.9 % on true fields. The difference with the first run is to be traced (protocol amendment 4).
- A diagnostic aimed at true irreversibility is to be defined in a later amendment.
