"""Diagnostic C: arrow of time read by a linear probe.

Two feature sets are available:

- ``"differences"``: the concatenated consecutive differences ``z_{t+1} - z_t``, shape
  ``((T-1) * d,)``. They suit processes with a systematic drift (decay to a fixed point).
- ``"asymmetry"``: clip averages of statistics that change sign exactly under time reversal,
  the antisymmetric lag-one covariance ``z_t^i z_{t+1}^j - z_{t+1}^i z_t^j`` (``i < j``) and the
  third-order asymmetry ``z_t^i**2 z_{t+1}^i - z_t^i z_{t+1}^i**2``. A linear probe on raw
  sequences or differences cannot separate the two directions of a statistically
  stationary process (both directions have the same mean), while these statistics can
  when the dynamics are irreversible.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

FeatureSet = Literal["differences", "asymmetry"]


def clips(sequences: NDArray[np.floating], clip_len: int, stride: int) -> NDArray[np.float64]:
    """Clips ``(N * per, clip_len, d)`` cut from ``(N, T, d)`` with the given stride.

    Clips are ordered start-major: all sequences at the first start, then the next start.
    """
    seqs = np.asarray(sequences, dtype=np.float64)
    starts = range(0, seqs.shape[1] - clip_len + 1, stride)
    return np.concatenate([seqs[:, s : s + clip_len] for s in starts], axis=0)


def time_asymmetry_features(sequences: NDArray[np.floating]) -> NDArray[np.float64]:
    """Clip statistics that flip sign under time reversal; ``(N, T, d) -> (N, d(d-1)/2 + d)``."""
    seqs = np.asarray(sequences, dtype=np.float64)
    a, b = seqs[:, :-1], seqs[:, 1:]
    lag = np.einsum("nti,ntj->nij", a, b) / a.shape[1]
    iu = np.triu_indices(seqs.shape[2], k=1)
    antisym = (lag - lag.transpose(0, 2, 1))[:, iu[0], iu[1]]
    third = np.mean(a**2 * b - a * b**2, axis=1)
    return np.concatenate([antisym, third], axis=1)


def arrow_of_time_dataset(
    latent_sequences: NDArray[np.floating],
    rng: np.random.Generator,
    *,
    features: FeatureSet = "differences",
    reversed_from: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Reverse each ``(T, d)`` sequence with probability 1/2; label 1 = forward, 0 = reversed.

    With ``reversed_from`` (same shape), the reversed version of sequence ``i`` is read from
    that array instead: this is how a transformed copy (for instance the x-mirrored fields)
    is used as the "reversed" class while the forward class stays untouched.
    """
    seqs = np.asarray(latent_sequences, dtype=np.float64)
    if seqs.ndim != 3 or seqs.shape[1] < 2:
        raise ValueError(f"expected (N, T>=2, d), got {seqs.shape}")
    source = seqs if reversed_from is None else np.asarray(reversed_from, dtype=np.float64)
    if source.shape != seqs.shape:
        raise ValueError(f"reversed_from must have shape {seqs.shape}, got {source.shape}")
    labels = rng.integers(0, 2, size=seqs.shape[0]).astype(np.int64)
    oriented = np.where(labels[:, None, None] == 1, seqs, source[:, ::-1])
    if features == "asymmetry":
        return time_asymmetry_features(oriented), labels
    return np.diff(oriented, axis=1).reshape(seqs.shape[0], -1), labels


@dataclass
class LinearProbe:
    """Logistic regression on standardised features."""

    weight: NDArray[np.float64]
    bias: float
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]

    def logits(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        x = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        return np.asarray(x @ self.weight + self.bias, dtype=np.float64)

    def predict(self, features: NDArray[np.floating]) -> NDArray[np.int64]:
        return (self.logits(features) > 0).astype(np.int64)


def train_probe(
    features: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    l2: float,
    seed: int,
    max_iter: int = 200,
) -> LinearProbe:
    """Full-batch L-BFGS logistic regression with L2 penalty ``l2 * ||w||^2``."""
    if not np.isin(np.asarray(labels), (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    x_np = np.asarray(features, dtype=np.float64)
    mean = x_np.mean(axis=0)
    scale = x_np.std(axis=0)
    scale[scale == 0] = 1.0
    x = torch.as_tensor((x_np - mean) / scale)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.float64)
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(x @ w + b, y)
        loss = loss + l2 * w.pow(2).sum()
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    opt.step(closure)  # type: ignore[no-untyped-call]
    return LinearProbe(
        weight=w.detach().numpy().copy(), bias=float(b.detach()), mean=mean, scale=scale
    )


def probe_accuracy(
    probe: LinearProbe, features: NDArray[np.floating], labels: NDArray[np.integer]
) -> float:
    """Fraction of correctly classified sequences."""
    return float(np.mean(probe.predict(features) == np.asarray(labels)))
