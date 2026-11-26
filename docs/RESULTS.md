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
