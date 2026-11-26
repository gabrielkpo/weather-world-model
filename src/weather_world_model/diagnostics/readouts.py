"""Linear readouts of physical quantities from latents, on real frames and on rollouts."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def ridge(x: FloatArray, y: FloatArray, alpha: float) -> FloatArray:
    """Ridge weights with an unpenalised intercept (last entry); ``x`` is ``(M, d)``."""
    xb = np.hstack([x, np.ones((x.shape[0], 1))])
    reg = alpha * np.eye(xb.shape[1])
    reg[-1, -1] = 0.0  # no penalty on the intercept
    return np.asarray(np.linalg.solve(xb.T @ xb + reg, xb.T @ y), dtype=np.float64)


def _predict(x: FloatArray, w: FloatArray) -> FloatArray:
    return np.asarray(np.hstack([x, np.ones((x.shape[0], 1))]) @ w, dtype=np.float64)


def r2_score(y: FloatArray, pred: FloatArray) -> float:
    """Coefficient of determination of one target."""
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


@dataclass(frozen=True)
class ReadoutProbes:
    """One fitted ridge probe per quantity, with the standardisation it was fitted under."""

    names: tuple[str, ...]
    z_mean: FloatArray
    z_std: FloatArray
    y_mean: FloatArray
    y_std: FloatArray
    weights: FloatArray  # (d + 1, Q)
    scores: dict[str, Any]  # per quantity: alpha, val_r2, test_r2

    def predict(self, latents: NDArray[np.floating]) -> FloatArray:
        """Standardised estimates ``(M, Q)`` of every quantity from latents ``(M, d)``."""
        return _predict((np.asarray(latents) - self.z_mean) / self.z_std, self.weights)

    def standardise(self, quantities: NDArray[np.floating]) -> FloatArray:
        """True quantities ``(M, Q)`` in the units the probes predict."""
        return np.asarray((np.asarray(quantities) - self.y_mean) / self.y_std, dtype=np.float64)


def fit_readout_probes(
    latents: dict[str, FloatArray], quantities: dict[str, FloatArray], names: Sequence[str]
) -> ReadoutProbes:
    """Ridge probe per quantity from ``train``/``val``/``test`` latents ``(M, d)``.

    Inputs and targets are standardised with training statistics; the regularisation is the
    value of ``RIDGE_ALPHAS`` with the best validation R²; the test R² is computed once.
    """
    z_mean = latents["train"].mean(axis=0)
    z_std = latents["train"].std(axis=0)
    z_std[z_std == 0] = 1.0
    y_mean = quantities["train"].mean(axis=0)
    y_std = quantities["train"].std(axis=0)
    y_std[y_std == 0] = 1.0
    x = {k: (v - z_mean) / z_std for k, v in latents.items()}
    y = {k: (v - y_mean) / y_std for k, v in quantities.items()}
    scores: dict[str, Any] = {}
    weights = np.empty((x["train"].shape[1] + 1, len(names)))
    for q, name in enumerate(names):
        val_r2 = [
            r2_score(y["val"][:, q], _predict(x["val"], ridge(x["train"], y["train"][:, q], a)))
            for a in RIDGE_ALPHAS
        ]
        best = RIDGE_ALPHAS[int(np.nanargmax(val_r2))]
        w = ridge(x["train"], y["train"][:, q], best)
        weights[:, q] = w
        scores[name] = {
            "alpha": best,
            "val_r2": float(np.nanmax(val_r2)),
            "test_r2": r2_score(y["test"][:, q], _predict(x["test"], w)),
        }
    return ReadoutProbes(tuple(names), z_mean, z_std, y_mean, y_std, weights, scores)


def rollout_readouts(
    probes: ReadoutProbes,
    rollout_fn: Callable[[FloatArray, int], FloatArray],
    latents: NDArray[np.floating],
    quantities: NDArray[np.floating],
    *,
    horizons: Sequence[int],
    n_samples: int = 1,
) -> dict[str, Any]:
    """Score fitted probes on rolled-out latents against the true quantity ``h`` steps later.

    ``latents`` is ``(N, T, d)`` (real-frame latents of N runs) and ``quantities`` is
    ``(N, T, Q)``. For each horizon ``h``, every latent ``z_t`` with ``t + h < T`` is rolled
    out ``h`` steps with ``rollout_fn`` (``((B, d), k) -> (B, k+1, d)``), the probes are
    applied to the last latent and scored (R²) against the quantity at ``t + h``. The
    persistence control applies the same probes to ``z_t`` itself, on the same pairs. With
    ``n_samples > 1`` (stochastic predictors, ``rollout_fn`` drawing fresh noise at each call)
    the estimate is the mean of the probe outputs over that many sampled rollouts.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    lat = np.asarray(latents, dtype=np.float64)
    t_len, d = lat.shape[1], lat.shape[2]
    out: dict[str, Any] = {}
    for h in horizons:
        if not 1 <= h < t_len:
            raise ValueError(f"horizon {h} does not fit sequences of {t_len} frames")
        z0 = lat[:, : t_len - h].reshape(-1, d)
        truth = probes.standardise(quantities[:, h:].reshape(-1, quantities.shape[-1]))
        estimate = np.zeros_like(truth)
        for _ in range(n_samples):
            estimate += probes.predict(rollout_fn(z0, h)[:, -1])
        estimate /= n_samples
        persistence = probes.predict(z0)
        out[f"h{h}"] = {
            name: {
                "rollout_r2": r2_score(truth[:, q], estimate[:, q]),
                "persistence_r2": r2_score(truth[:, q], persistence[:, q]),
            }
            for q, name in enumerate(probes.names)
        }
    return out
