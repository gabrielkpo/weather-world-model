# Protocol

Version 1, 2026-11-26. This protocol is frozen once merged to `main`, before any measurement. Any later change goes through a pull request and is logged in [Amendments](#amendments) with its date and reason. Results that depart from the protocol are reported as such.

## 1. Question

Beyond forecast quality, what has a latent world model learned about the physics of storms? We adapt a pretrained latent world model to storm radar and interrogate its latent predictor with three dynamical-systems diagnostics:

| Diagnostic | Question asked to the latent predictor `z(t+1) = f(z(t))` |
|---|---|
| **A. Predictability** | Do nearby latent trajectories diverge, how fast, and up to which horizon? |
| **B. Growth and decay modes** | Does a linear (Koopman / DMD) approximation of the latent dynamics expose growth, decay and oscillation modes that match storm intensification? |
| **C. Arrow of time** | Can a linear probe on frozen latents tell a forward sequence from a time-reversed one? |

Each diagnostic is first calibrated on simulated systems with known answers, then applied to radar.

## 2. Models

| ID | Model | Training |
|---|---|---|
| `vjepa2-frozen` | V-JEPA 2 ViT-L/16, 256 px (`facebook/vjepa2-vitl-fpc64-256`, MIT) | none |
| `vjepa2-adapted` | same | self-supervised continued pretraining on the SEVIR training split (JEPA objective, no labels) |
| `jepa-scratch` | small JEPA (ViT-S or convolutional encoder, predictor, EMA target, variance-covariance regularisation) | from scratch on the SEVIR training split, same number of training steps as `vjepa2-adapted` |
| `earthformer` | Earthformer with its published SEVIR weights (Apache-2.0) | none; pixel-space reference for diagnostic A only |

No code is copied from the CC BY-NC I-JEPA or V-JEPA (v1) repositories.

## 3. Data

### 3.1 Calibration systems (simulated locally)

| System | Use | Ground truth |
|---|---|---|
| Lorenz-96, N = 40, F = 8, RK4, dt = 0.01, sampled every 0.05 | A and C | leading Lyapunov exponent computed on the true equations with the tangent-linear QR method; the system is dissipative |
| Damped linear oscillators (known eigenvalues), rendered as sequences | B | eigenvalues of the generating linear map |
| Undamped harmonic oscillator | C (negative control) | time-reversible: a probe should stay near chance |

The same small JEPA architecture as `jepa-scratch` is trained on each calibration system.

### 3.2 Radar: SEVIR

- Source: SEVIR public bucket (`s3://sevir`), vertically integrated liquid (VIL) only, 384 × 384 px at 1 km, 49 frames at 5-minute steps per event.
- Severe storms: `STORMEVENTS` entries whose NOAA Storm Events `event_type` is `Hail`, `Tornado` or `Thunderstorm Wind`, joined on `event_id`.
- Ordinary storms: `RANDOMEVENTS` entries whose maximum VIL exceeds a convective threshold (value fixed on the training split before any diagnostic is run) and that match no Storm Events report.
- Splits by date, never by random sampling: train 2017-01-01 to 2018-12-31, validation 2019-01-01 to 2019-06-30, test 2019-07-01 onwards. The split boundaries are checked against the catalogue before download; if a split holds fewer than 200 severe events, the boundaries are moved once, before any measurement, and the change is logged.
- Download budget: at most 50 GB. The exact size of the selected subset is computed from the catalogue and announced before downloading.
- Preprocessing for V-JEPA 2: VIL decoded to physical units then scaled to [0, 1] with fixed bounds, resized to 256 × 256, replicated to three channels, clips of 16 frames.
- Data, weights and raw outputs never enter Git. A `DATA.md` lists every source with its licence.

## 4. Diagnostics

### A. Predictability

1. For each test clip, encode the observed frames, then perturb the input with additive Gaussian noise of relative amplitude ε ∈ {1e-3, 1e-2}.
2. Roll the latent predictor forward autoregressively for K = 12 steps (1 hour) from the clean and perturbed contexts.
3. Separation `d(k)` = L2 distance between the two latent trajectories, normalised by the median distance between latents of unrelated clips (saturation level).
4. Finite-time growth rate = slope of `log d(k)` over the initial growth window (fixed on validation); predictability horizon = first step where `d(k) ≥ 0.5`.
5. Earthformer: same perturbation, separation measured in pixel space with the same normalisation.

### B. Growth and decay modes

1. Pool each frame latent to one vector, project to r = 64 dimensions with PCA fitted on the training split.
2. Fit a linear operator `A` by least squares on pairs `(z(t), z(t+1))` from the training split (DMD).
3. Report the one-step R² of the linear fit next to the nonlinear predictor; report eigenvalues of `A`.
4. Growth modes: eigenvalues with modulus > 1. Project test trajectories on the leading growth mode and correlate its amplitude with the VIL growth rate and with the time to the next severe report.

### C. Arrow of time

1. Frozen latents of 16-frame clips, forward or reversed with equal probability.
2. Logistic-regression probe trained on the training split, evaluated on the test split.
3. Accuracy reported overall and by phase: growth (VIL increasing) and decay (VIL decreasing).

## 5. Hypotheses

Stated before any measurement. Each is reported as supported, not supported or not tested.

| ID | Hypothesis | Criterion |
|---|---|---|
| H0-A | Diagnostic A recovers the leading Lyapunov exponent of Lorenz-96 | within 30 % of the reference; otherwise A is not applied to radar and the failure is reported |
| H0-B | Diagnostic B recovers the eigenvalues of the linear calibration system | within 10 % in modulus |
| H0-C | Diagnostic C separates irreversible from reversible dynamics | > 80 % on Lorenz-96 and < 60 % on the harmonic oscillator |
| H1 | Severe storms have a shorter latent predictability horizon than ordinary storms | one-sided Mann-Whitney test, p < 0.05 |
| H2 | The latent model does not smooth out fast initial error growth relative to Earthformer | initial growth rate of `vjepa2-adapted` ≥ that of Earthformer at equal ε |
| H3 | The amplitude of the leading growth mode tracks intensification | Spearman correlation with the VIL growth rate ≥ 0.3 |
| H4 | The arrow of time is linearly readable on radar | probe accuracy > 80 % for `vjepa2-adapted` |
| H5 | Pretraining on natural video helps | `vjepa2-adapted` beats `jepa-scratch` on H4 accuracy by ≥ 5 points |

## 6. Statistics and reporting

- Uncertainty: 95 % bootstrap intervals resampled by event; three seeds for every trained model and probe.
- Tuning (windows, thresholds, probe regularisation) uses the validation split only; the test split is evaluated once per model.
- Every reported number comes from a versioned run record (code commit, data fingerprint, model and revision, number of events, date) written by the code under `results/`.
- Negative results are reported with the same detail as positive ones.

## 7. Compute

Calibration runs on a laptop CPU. Radar runs use a free Kaggle GPU (T4). No paid compute.

## 8. Out of scope

Operational forecasting or warning, image generation, agents and planning, additional sensors (infrared, lightning) and other physics datasets. They may become later work, recorded as amendments.

## Amendments

None yet.
