"""Diagnostic C: arrow of time read by a linear probe.

Features are the concatenated consecutive differences ``z_{t+1} - z_t`` of a sequence,
shape ``((T-1) * d,)``. Differences remove the static content of a clip, which is identical
in both directions, and keep what changes with the direction of time.
"""

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from weather_world_model.seeding import seed_everything


def arrow_of_time_dataset(
    latent_sequences: NDArray[np.floating], rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Reverse each ``(T, d)`` sequence with probability 1/2; label 1 = forward, 0 = reversed."""
    seqs = np.asarray(latent_sequences, dtype=np.float64)
    if seqs.ndim != 3 or seqs.shape[1] < 2:
        raise ValueError(f"expected (N, T>=2, d), got {seqs.shape}")
    labels = rng.integers(0, 2, size=seqs.shape[0]).astype(np.int64)
    oriented = np.where(labels[:, None, None] == 1, seqs, seqs[:, ::-1])
    features = np.diff(oriented, axis=1).reshape(seqs.shape[0], -1)
    return features, labels


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
    seed_everything(seed)
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
