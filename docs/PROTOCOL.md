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

### 1. 2026-11-26, before the first calibration run

- **Diagnostic C features for frame-level encoders.** The calibration JEPA encodes one frame at a time, so a clip has no single latent. A linear probe on raw latent sequences or on their differences cannot separate the two time directions of a statistically stationary process (both directions have the same mean). Clip features for frame-level encoders are therefore the clip averages of statistics that change sign exactly under time reversal: the antisymmetric lag-one covariance `z_t^i z_{t+1}^j - z_{t+1}^i z_t^j` (`i < j`) and the third-order asymmetry `z_t^i² z_{t+1}^i - z_t^i z_{t+1}^i²`, after a PCA to 8 dimensions fitted on the training clips. The difference features are reported alongside for transparency. Clip-level encoders (V-JEPA 2) keep the protocol as written.
- **Diagnostic A growth window in calibration.** Steps 1 to the last step before the normalised separation reaches 0.3 (a fixed rule replaces the validation-tuned window on simulated data).
- **Oracle baselines.** Each calibration also runs the diagnostic on the true system (identity encoder and true dynamics), to separate errors of the diagnostic code from errors of the learned model. Only the JEPA values decide H0.

### 2. 2026-11-26, after the calibration and before any run on the new world

Decision, after the calibration results: the project proceeds in three steps of increasing realism.

1. **Lorenz-96, proof of concept** (done): the diagnostics recover known answers.
2. **A two-layer quasi-geostrophic (QG) atmosphere**, a 2D "wind map" world whose physics is fully known, simulated locally (64 × 64, doubly periodic, baroclinic jet with cyclones and anticyclones).
3. **SEVIR radar**, as in section 3.2, if time allows before the release; otherwise reported as not done.

Changes:

- **Diagnostic B (growth and decay modes) is withdrawn.** Its calibration record stays in the results; H3 is dropped.
- **Training objective.** Every JEPA in steps 2 and 3 is trained twice, with a one-step loss and with a multi-step latent rollout loss (k = 8 steps, autoregressive in latent space against EMA-target latents), because the one-step model contracted on Lorenz-96. The multi-step model is also rerun on Lorenz-96 and reported.
- **Readouts (new).** Ridge-regression probes from frozen frame latents to physical quantities computed exactly by the simulator: kinetic energy, available potential energy, enstrophy, maximum zonal-mean jet speed, maximum cyclonic and anticyclonic vorticity, and the number of coherent vortices. Probes are fitted on training trajectories, regularisation chosen on validation trajectories, scored once on test trajectories (R²). Control: the same encoder architecture with random weights.
- **QG data.** Independent simulator runs, split by run: 70 % train, 15 % validation, 15 % test. Snapshots are taken after spin-up. The predictability ground truth is the simulator's own twin divergence (same perturbation, same normalisation rule as diagnostic A).
- **Models on the QG world:** the small JEPA with a convolutional encoder (one-step and multi-step), and V-JEPA 2 (frozen, then adapted) on the upper-layer wind speed rendered as video, when a GPU session is available.

Hypotheses for step 2, stated before any run:

| ID | Hypothesis | Criterion |
|---|---|---|
| W-A | Multi-step training restores chaotic error growth | latent growth rate of the multi-step model within 30 % of the simulator's twin-divergence growth rate, and closer to it than the one-step model |
| W-C | The arrow of time is linearly readable on the QG world | probe accuracy > 80 % (asymmetry features, amendment 1) |
| W-P | Physical quantities are linearly readable from the latent | test R² ≥ 0.7 for kinetic energy, available potential energy, enstrophy and jet speed, and ≥ 0.1 above the random-weight control; vortex count and vorticity extremes are reported without a threshold |
| W-V | Pretraining on natural video helps on the QG world | V-JEPA 2 beats the small JEPA trained from scratch on W-C accuracy and on mean W-P R² by ≥ 0.05 |

Radar hypotheses H1, H2, H4 and H5 are unchanged.

### 3. 2026-11-26, after the first step-2 run finished and before its results were read

Timing, stated plainly: the first run of step 2 (hypotheses W-A, W-C, W-P) had been launched and its run record was on disk when this amendment was written, but nobody had opened the record. The two additions below come from reasoning about the method, not from those results. The results of W-A, W-C and W-P are read and reported first, unchanged.

**Rollout readouts (new hypothesis W-R).** W-P probes latents of real frames, so it evaluates the encoder only. To evaluate the learned dynamics, the probes fitted for W-P (on real-frame latents, unchanged) are applied to latents predicted by rolling the predictor out h = 1, 5 and 10 steps from the latent of frame t, and scored (R², test runs) against the simulator's true quantity at t + h. Control: persistence, the same probe applied to the latent of frame t as the estimate for t + h, because slowly varying quantities are well predicted by "no change".

**Stochastic predictor (new hypotheses L-S and W-S).** A deterministic predictor trained on a distance to the target is pulled toward the centre of the possible futures, which contracts nearby trajectories. A stochastic variant is trained on Lorenz-96 and on the QG world: the predictor receives the latent and a noise vector ε ~ N(0, I) of dimension 8; each training step draws two predictions per input and minimises the energy score, mean_i ‖ẑ_i − z*‖ − ½ ‖ẑ_1 − ẑ_2‖ (Euclidean norms), plus the existing variance and covariance terms; one-step loss, same encoder, data, steps and seeds as the deterministic one-step model.

Predictability of a stochastic model is measured in two ways, and only the first decides the hypotheses:

1. *Sensitivity to initial conditions:* twin rollouts from the clean and perturbed initial states, driven by the same noise sequence; growth rate with the rule of diagnostic A.
2. *Ensemble spread:* rollouts from the same initial state with independent noise sequences; spread growth, reported.

| ID | Hypothesis | Criterion |
|---|---|---|
| W-R | The predictor carries the physics forward | at h = 10, mean over kinetic energy, available potential energy, enstrophy and jet speed of (R² of the rollout − R² of persistence) ≥ 0.05, for the one-step and the multi-step model; per-quantity values at h = 1, 5, 10 are reported |
| L-S | A stochastic predictor restores chaotic error growth on Lorenz-96 | sensitivity growth rate within 30 % of the leading Lyapunov exponent, and closer to it than the deterministic one-step model |
| W-S | Same on the QG world | sensitivity growth rate within 30 % of the simulator's twin-divergence growth rate, and closer to it than the deterministic one-step model |

The stochastic models are also scored on W-C, W-P and W-R, reported without thresholds. The models of the first run were not saved, so W-R is measured on models retrained with the same configuration and seeds; trained weights are stored outside Git.

### 4. 2026-11-26, after an exploratory analysis of the arrow-of-time probe

An exploratory analysis on true QG fields only, with no model involved (`scripts/explore_arrow_probe.py`, record `results/exploratory/arrow-probe-qg.json`), showed what the probe of diagnostic C detects. Against plain time reversal it reaches 90 % (8 PCA components, clips of 16 snapshots) to 100 % (32 components, clips of 64) on 1 116 test clips. When the reversed clip is also mirrored in x with the sign of v flipped, a transformation under which the inviscid QG equations are invariant and which therefore keeps the direction of wave propagation, the same probe falls to chance (43 to 55 % across the 16 settings of the record). With its current features, the probe reads the direction in which waves propagate, not physical irreversibility.

Changes:

- **Interpretation.** The H0-C and W-C verdicts stand as measured. Their interpretation is corrected in the results: the latents keep the direction of wave propagation; nothing is claimed about irreversibility.
- **Mirrored control.** Every later arrow-of-time result on the QG world is reported twice, against plain reversal and against reversal combined with the x-mirror, on true fields and on latents (for latents, the mirrored clip is encoded after mirroring the fields).
- **Lorenz-96.** No equivalent control was run on Lorenz-96, whose waves also have a preferred direction of propagation; its H0-C result is reported with that caveat.
- **Irreversibility.** A diagnostic aimed at true irreversibility (one that stays above chance under the mirrored control on true fields) will be defined in a later amendment before it is applied to any model.
- **Open point.** With the protocol's sizes (56 training runs, 100 snapshots, 8 components, clips of 16) the exploratory analysis gives 87 % on true fields, against 76.5 % in the first step-2 run. The difference is not explained yet; the two computations differ at least in the standardisation of the fields and in the probe's training clips, and this is to be traced before diagnostic C is used again.

### 5. 2026-11-27, the step-2 data were recorded before the flow was stationary

**Finding.** With the simulator defaults, the kinetic energy of the QG world keeps rising for about 40 time units after a cold start (about 0.05 at t = 15, 0.45 at t = 20, 0.75 at t = 25) and only then fluctuates around its equilibrium (about 0.9, between 0.7 and 1.2). The step-2 runs used a spin-up of 4 000 steps (20 time units) and recorded snapshots from t = 20 to t = 30, so every training and test sequence was taken while the flow was still spinning up. The statement in amendment 2 that snapshots are taken "after spin-up" was not met in the sense intended. This was noticed on 2026-11-27 while exporting a demonstration sequence in which kinetic energy rose from 0.18 to 0.83.

**Consequences.** All step-2 results obtained so far (W-A, W-C, W-P, W-R, W-S, and the exploratory analysis of amendment 4, whose data used the same spin-up) are superseded. They stay in the results, marked as measured on a spinning-up flow. In particular, with energy rising in every sequence, readouts of energy and the comparison with persistence can reflect the elapsed time since the cold start rather than learned dynamics. Step 1 is not affected: Lorenz-96 runs use a burn-in of 50 time units.

**Changes.**

- Spin-up of 12 000 steps (60 time units) for every QG dataset, including the twin-divergence ground truth and the exploratory analysis.
- A stationarity check stored in every QG run record and required to pass before any hypothesis is evaluated: over the training runs, the mean kinetic energy of the last third of the recorded snapshots differs from that of the first third by less than 10 %. A long-run energy record of the simulator (two seeds, 300 time units) is versioned with the rerun.
- Step 2 is rerun in full on the new data, with the hypotheses and criteria of amendments 2 and 3 unchanged, and with the mirrored control of amendment 4 on W-C.

### 6. 2026-11-27, the encoder's normalisation erased the quantities that W-P reads

**Finding.** On the stationary flow (amendment 5), energies and enstrophy were not linearly readable from the latents of any model, including the random-weight control. An exploratory analysis (`scripts/explore_energy_readout.py`, record `results/exploratory/energy-readout-qg.json`) showed that the GroupNorm layers of the convolutional encoder rescale every sample and remove the overall amplitude of the field: kinetic energy is read at R² −0.03 to −0.02 by a random encoder with GroupNorm and at 0.70 to 0.83 by the same encoder without it. With this architecture, W-P could not separate learning from not learning for the amplitude-based quantities.

**Changes.**

- Step 2 is rerun with the convolutional encoder without normalisation (`conv_norm="none"`) for every model (one-step, 8-step, stochastic) and for the random-weight control, with the data, hypotheses and criteria of amendments 2, 3 and 5 unchanged and the mirrored control of amendment 4.
- The rerun of amendment 5 (encoder with GroupNorm) is kept and reported next to it, so that both architectures are compared on the same stationary data.
- V-JEPA 2 (hypothesis W-V) is compared with the small JEPA without normalisation.

### 7. 2026-11-27, reporting promises that the records did not keep

**Finding.** A review of the run records against this protocol found three promises that were not kept.

- **Uncertainty and data fingerprint (section 6).** Section 6 promised 95 % bootstrap intervals and a data fingerprint in every run record. The records report per-seed values and the minimum to maximum over three seeds, and none carries a hash of its data.
- **Multi-step model on Lorenz-96 (amendment 2).** Amendment 2 announced that the multi-step model would be rerun on Lorenz-96. This was not done as a recorded run: only an exploratory script (`scripts/compare_rollout_loss.py`, one seed, printed output, no record) was run; its numbers are reported as exploratory in the results.
- **`DATA.md` (section 3.2).** The file does not exist.

**Changes.**

- From now on, every run record carries a hash of the training array and the dirty state of the code (whether the working tree had uncommitted changes when the run started).
- Uncertainty is reported as per-seed values and the minimum to maximum over seeds, and verdicts use the worst seed. Bootstrap intervals are dropped: three seeds are what the compute budget allows, and an interval over three values would suggest a precision the runs do not have.
- Data sources and their licences are listed in the Data section of the README. A `DATA.md` will be added with step 3, when real data enter the project.
- Records written before this amendment are kept as they are.

### 8. 2026-11-27, the mirrored control of W-C failed for a large video encoder

**Finding.** The frozen V-JEPA 2 scored 98 to 100 % on the mirrored control of amendment 4, where every small JEPA is at chance. Exploratory symmetry controls (`scripts/explore_arrow_symmetry_vjepa2.py`, record `results/exploratory/arrow-symmetry-vjepa2.json`) showed why: only the reversed class is mirrored in that control, and both the pretrained and the random-weight encoder tell a mirrored clip from a plain one with time unchanged (94 to 100 %). With every clip of both classes turned by an independent random rotation or reflection of the square (grid and wind vector), the pretrained encoder still reads the time direction at 80 to 88 %, the random-weight encoder and frozen clips are at chance. Whether this signal comes from irreversibility or from how structures move is not tested.

**Changes.**

- For a clip-level encoder, W-C is reported with the random-symmetry control (`d4` in the records) next to the plain and mirrored probes; the mirrored control alone is not used to conclude.
- Two synthetic control worlds, rendered and probed exactly as the QG world (`worlds/waves.py`, same grid, `beta = 6`, `F = 16`, one snapshot every 0.1 time units, wind standard deviation 1.3, 56 training and 12 test runs of 100 snapshots, data seed 1000):
  - **reversible waves**: free linear Rossby waves, no dissipation. A time-reversed clip is a valid clip of the mirrored world, so under random symmetries both classes have the same distribution and no probe can beat chance except through an artefact of the pipeline;
  - **damped waves**: the same waves with the field multiplied by `exp(-0.2 t)`, which no reflection undoes.
- Criteria, fixed before the run, for the frozen V-JEPA 2 under random symmetries (three probe seeds): the pipeline is **clean** if every seed is at most 55 % on reversible waves; the encoder **detects dissipation** if every seed is at least 60 % on damped waves. The QG result (80 to 88 %) is read as compatible with irreversibility only if both hold; if the pipeline is not clean, the QG result is not interpreted. The random-weight encoder is reported for both worlds.
- These controls are exploratory: they qualify the reading of W-C and W-V and change no verdict of the records already published.

