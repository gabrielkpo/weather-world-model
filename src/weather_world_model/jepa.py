"""A small JEPA for sequences of state vectors.

The online encoder maps a frame ``x_t`` of shape ``(D,)`` to a latent ``z_t``. A residual
predictor maps ``z_t`` to a prediction of the latent of the next frame, which is compared
(L1) with the output of an exponential-moving-average target encoder on ``x_{t+1}``, with
gradients stopped. VICReg-style variance and covariance terms on the online latents guard
against representational collapse.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from weather_world_model.seeding import seed_everything


@dataclass(frozen=True)
class JEPAConfig:
    """Hyperparameters of :class:`SequenceJEPA` and :func:`train_jepa`."""

    input_dim: int
    latent_dim: int = 32
    hidden_dim: int = 128
    ema_momentum: float = 0.99
    var_weight: float = 1.0
    cov_weight: float = 0.04
    lr: float = 1e-3
    batch_size: int = 128
    context_len: int = 1
    log_every: int = 10
    collapse_threshold: float = 1e-2

    def __post_init__(self) -> None:
        if self.context_len != 1:
            raise ValueError("only context_len=1 (one frame per latent) is supported")
        if self.log_every < 1 or self.batch_size < 1:
            raise ValueError("log_every and batch_size must be positive")
        if not 0.0 <= self.ema_momentum <= 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1], got {self.ema_momentum}")


@dataclass
class TrainHistory:
    """Training log: loss and mean per-dimension latent std, every ``log_every`` steps."""

    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    latent_std: list[float] = field(default_factory=list)
    collapse_threshold: float = 1e-2

    @property
    def collapsed(self) -> bool:
        """True if the last recorded latent std fell under the collapse threshold."""
        return bool(self.latent_std) and is_collapsed(
            self.latent_std[-1], threshold=self.collapse_threshold
        )


def mean_latent_std(z: Tensor) -> float:
    """Mean over latent dimensions of the std over the batch; ``z`` has shape ``(B, d)``."""
    return float(z.detach().reshape(-1, z.shape[-1]).std(dim=0).mean())


def is_collapsed(latent_std: float, *, threshold: float = 1e-2) -> bool:
    """Collapse detector: the latents barely vary across inputs (non-finite counts as collapse)."""
    return not np.isfinite(latent_std) or latent_std < threshold


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _variance_loss(z: Tensor) -> Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.relu(1.0 - std).mean()


def _covariance_loss(z: Tensor) -> Tensor:
    n, d = z.shape
    zc = z - z.mean(dim=0)
    cov = zc.T @ zc / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return off.pow(2).sum() / d


class SequenceJEPA(nn.Module):
    """Online encoder, EMA target encoder and residual latent predictor."""

    def __init__(self, config: JEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _mlp(config.input_dim, config.hidden_dim, config.latent_dim)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = _mlp(config.latent_dim, config.hidden_dim, config.latent_dim)

    def loss(self, x_t: Tensor, x_next: Tensor) -> tuple[Tensor, Tensor]:
        """Return the total loss and the online latents of ``x_t`` (shape ``(B, d)``)."""
        z = self.encoder(x_t)
        pred = z + self.predictor(z)
        with torch.no_grad():
            target = self.target_encoder(x_next)
        total = nn.functional.l1_loss(pred, target)
        if self.config.var_weight:
            total = total + self.config.var_weight * _variance_loss(z)
        if self.config.cov_weight:
            total = total + self.config.cov_weight * _covariance_loss(z)
        return total, z

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of the target encoder from the online encoder."""
        m = self.config.ema_momentum
        for pt, po in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            pt.mul_(m).add_(po.detach(), alpha=1.0 - m)

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Online-encoder latents of frames ``x`` of shape ``(..., D)``."""
        was_training = self.training
        self.eval()
        z: Tensor = self.encoder(x)
        self.train(was_training)
        return z

    @torch.no_grad()
    def predict_step(self, z: Tensor) -> Tensor:
        """One latent step ``z_t -> z_{t+1}``."""
        out: Tensor = z + self.predictor(z)
        return out

    @torch.no_grad()
    def rollout(self, z0: Tensor, k: int) -> Tensor:
        """Autoregressive latent rollout; returns ``(B, k+1, d)`` with ``z0`` first."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        zs = [z0]
        for _ in range(k):
            zs.append(self.predict_step(zs[-1]))
        return torch.stack(zs, dim=1)


def train_jepa(
    sequences: NDArray[np.floating],
    config: JEPAConfig,
    *,
    n_steps: int,
    seed: int,
) -> tuple[SequenceJEPA, TrainHistory]:
    """Train a :class:`SequenceJEPA` on consecutive-frame pairs of ``sequences`` ``(N, T, D)``."""
    if sequences.ndim != 3 or sequences.shape[2] != config.input_dim:
        raise ValueError(f"sequences must be (N, T, {config.input_dim}), got {sequences.shape}")
    if sequences.shape[1] < 2:
        raise ValueError("sequences need at least two frames")
    rng = seed_everything(seed)
    data = torch.as_tensor(sequences, dtype=torch.float32)
    x_t = data[:, :-1].reshape(-1, config.input_dim)
    x_next = data[:, 1:].reshape(-1, config.input_dim)
    n_pairs = x_t.shape[0]

    model = SequenceJEPA(config)
    params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=config.lr)
    history = TrainHistory(collapse_threshold=config.collapse_threshold)
    model.train()
    for step in range(n_steps):
        idx = torch.as_tensor(rng.integers(0, n_pairs, size=min(config.batch_size, n_pairs)))
        loss, z = model.loss(x_t[idx], x_next[idx])
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        model.update_target()
        if step % config.log_every == 0 or step == n_steps - 1:
            history.steps.append(step)
            history.loss.append(float(loss.detach()))
            history.latent_std.append(mean_latent_std(z))
    return model, history
