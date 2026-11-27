"""A small JEPA for sequences of frames (state vectors or 2D fields).

The online encoder maps a frame ``x_t`` of shape ``input_shape`` (``(D,)`` for a state
vector, ``(C, H, W)`` for a doubly periodic field) to a latent ``z_t``. A residual MLP
predictor maps ``z_t`` to a prediction of the latent of the next frame, which is compared
(L1) with the output of an exponential-moving-average target encoder on ``x_{t+1}``, with
gradients stopped. With ``rollout_steps = k > 1`` the predictor is applied autoregressively
``k`` times in latent space and the loss averages the L1 errors against the target latents
of ``x_{t+1} .. x_{t+k}``, with gradients flowing through the whole rollout. VICReg-style
variance and covariance terms on the online latents guard against representational collapse.

With ``noise_dim > 0`` the predictor is stochastic: it receives ``concat(z_t, eps)`` with
``eps ~ N(0, I)`` and is trained (one-step only) with the two-sample energy score
``mean_i ||z_hat_i - z*|| - 0.5 ||z_hat_1 - z_hat_2||`` (Euclidean norms) in place of the L1
error. ``noise_dim = 0`` (default) is the deterministic model, unchanged.
"""

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from weather_world_model.seeding import seed_everything


@dataclass(frozen=True)
class JEPAConfig:
    """Hyperparameters of :class:`SequenceJEPA` and :func:`train_jepa`."""

    input_dim: int | None = None
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
    rollout_steps: int = 1
    rollout_weighting: Literal["uniform", "linear"] = "uniform"
    encoder: Literal["mlp", "conv"] = "mlp"
    input_shape: tuple[int, ...] | None = None
    conv_channels: tuple[int, ...] = (32, 64, 96, 128)
    conv_norm: Literal["group", "none"] = "group"
    noise_dim: int = 0

    def __post_init__(self) -> None:
        if self.input_dim is None and self.input_shape is None:
            raise ValueError("give input_dim or input_shape")
        if self.input_shape is not None:
            if not self.input_shape or min(self.input_shape) < 1:
                raise ValueError(f"input_shape must be non-empty and positive: {self.input_shape}")
            if self.input_dim is not None and self.input_dim != math.prod(self.input_shape):
                raise ValueError("input_dim must equal the product of input_shape")
        if self.rollout_steps < 1:
            raise ValueError(f"rollout_steps must be >= 1, got {self.rollout_steps}")
        if self.noise_dim < 0:
            raise ValueError(f"noise_dim must be >= 0, got {self.noise_dim}")
        if self.noise_dim > 0 and self.rollout_steps != 1:
            raise ValueError("the stochastic predictor (noise_dim > 0) is trained one-step only")
        if self.rollout_weighting not in ("uniform", "linear"):
            raise ValueError(f"unknown rollout_weighting {self.rollout_weighting!r}")
        if self.encoder not in ("mlp", "conv"):
            raise ValueError(f"unknown encoder {self.encoder!r}")
        if self.encoder == "conv" and len(self.frame_shape) != 3:
            raise ValueError("the conv encoder needs input_shape = (C, H, W)")
        if not self.conv_channels or min(self.conv_channels) < 1:
            raise ValueError("conv_channels must be non-empty and positive")
        if self.context_len != 1:
            raise ValueError("only context_len=1 (one frame per latent) is supported")
        if self.log_every < 1 or self.batch_size < 1:
            raise ValueError("log_every and batch_size must be positive")
        if not 0.0 <= self.ema_momentum <= 1.0:
            raise ValueError(f"ema_momentum must be in [0, 1], got {self.ema_momentum}")

    @property
    def frame_shape(self) -> tuple[int, ...]:
        """Shape of one frame: ``input_shape`` if given, else ``(input_dim,)``."""
        if self.input_shape is not None:
            return tuple(self.input_shape)
        if self.input_dim is None:  # pragma: no cover - rejected by __post_init__
            raise ValueError("give input_dim or input_shape")
        return (self.input_dim,)

    def step_weights(self) -> Tensor:
        """Weights of the ``rollout_steps`` L1 terms; they sum to one."""
        k = self.rollout_steps
        if self.rollout_weighting == "linear":
            w = torch.arange(1, k + 1, dtype=torch.float32)
        else:
            w = torch.ones(k)
        return w / w.sum()


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


def _group_count(channels: int, max_groups: int = 8) -> int:
    return math.gcd(channels, max_groups)


class ConvEncoder(nn.Module):
    """Small CNN for doubly periodic fields ``(..., C, H, W) -> (..., latent_dim)``.

    Strided 3x3 convolutions with circular padding (the fields are periodic in both
    directions), GroupNorm and GELU, then a global average pool and a linear head. Because the
    padding is circular and the pool is global, shifting the input periodically by a multiple
    of ``2 ** len(channels)`` pixels leaves the latent unchanged.
    """

    def __init__(
        self,
        in_channels: int,
        channels: tuple[int, ...],
        latent_dim: int,
        norm: Literal["group", "none"] = "group",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            layers.append(nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, padding_mode="circular"))
            if norm == "group":
                # GroupNorm rescales each sample, which removes the overall field amplitude
                layers.append(nn.GroupNorm(_group_count(c_out), c_out))
            layers.append(nn.GELU())
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c_in, latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-3]
        h = self.features(x.reshape(-1, *x.shape[-3:]))
        out: Tensor = self.head(h.mean(dim=(-2, -1)))
        return out.reshape(*lead, -1)


def _build_encoder(config: JEPAConfig) -> nn.Module:
    shape = config.frame_shape
    if config.encoder == "conv":
        return ConvEncoder(shape[0], config.conv_channels, config.latent_dim, norm=config.conv_norm)
    mlp = _mlp(math.prod(shape), config.hidden_dim, config.latent_dim)
    if len(shape) == 1:
        return mlp
    return nn.Sequential(nn.Flatten(start_dim=-len(shape)), *mlp)


def _variance_loss(z: Tensor) -> Tensor:
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.relu(1.0 - std).mean()


def _covariance_loss(z: Tensor) -> Tensor:
    n, d = z.shape
    zc = z - z.mean(dim=0)
    cov = zc.T @ zc / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return off.pow(2).sum() / d


def energy_score(pred_1: Tensor, pred_2: Tensor, target: Tensor) -> Tensor:
    """Two-sample energy score, averaged over the batch; inputs have shape ``(B, d)``.

    ``mean_i ||pred_i - target|| - 0.5 ||pred_1 - pred_2||`` with Euclidean norms. It is a
    proper score: its expectation is minimal when the predictions are distributed as the
    target. A tiny constant under the square roots keeps the gradient finite at zero distance.
    """

    def norm(x: Tensor) -> Tensor:
        return torch.sqrt(x.pow(2).sum(dim=-1) + 1e-12)

    accuracy = 0.5 * (norm(pred_1 - target) + norm(pred_2 - target))
    return (accuracy - 0.5 * norm(pred_1 - pred_2)).mean()


class SequenceJEPA(nn.Module):
    """Online encoder, EMA target encoder and residual latent predictor."""

    def __init__(self, config: JEPAConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = _build_encoder(config)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = _mlp(
            config.latent_dim + config.noise_dim, config.hidden_dim, config.latent_dim
        )

    def _step(self, z: Tensor, noise: Tensor | None) -> Tensor:
        if self.config.noise_dim == 0:
            if noise is not None:
                raise ValueError("a deterministic predictor (noise_dim = 0) takes no noise")
            out: Tensor = z + self.predictor(z)
            return out
        if noise is None:
            raise ValueError("a stochastic predictor needs a noise tensor")
        expected = (*z.shape[:-1], self.config.noise_dim)
        if tuple(noise.shape) != expected:
            raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}")
        out = z + self.predictor(torch.cat([z, noise.to(z)], dim=-1))
        return out

    def _draw(self, z: Tensor, generator: torch.Generator | None) -> Tensor | None:
        if self.config.noise_dim == 0:
            return None
        shape = (*z.shape[:-1], self.config.noise_dim)
        return torch.randn(shape, generator=generator).to(z)

    def loss(self, x_t: Tensor, x_next: Tensor) -> tuple[Tensor, Tensor]:
        """Return the total loss and the online latents of ``x_t`` (shape ``(B, d)``)."""
        z = self.encoder(x_t)
        with torch.no_grad():
            target = self.target_encoder(x_next)
        if self.config.noise_dim > 0:
            total = energy_score(
                self._step(z, self._draw(z, None)), self._step(z, self._draw(z, None)), target
            )
        else:
            total = nn.functional.l1_loss(self._step(z, None), target)
        if self.config.var_weight:
            total = total + self.config.var_weight * _variance_loss(z)
        if self.config.cov_weight:
            total = total + self.config.cov_weight * _covariance_loss(z)
        return total, z

    def rollout_loss(self, window: Tensor) -> tuple[Tensor, Tensor]:
        """Multi-step loss on windows ``(B, k+1, *frame_shape)`` of consecutive frames.

        The online latent of the first frame is rolled out ``k`` steps with the predictor
        (gradients flow through the rollout) and step ``j`` is compared (L1) with the target
        latent of frame ``j``; the ``k`` terms are averaged with ``config.step_weights()``.
        For ``k = 1`` this is exactly :meth:`loss`. Returns the total loss and the online
        latents of the first frames.
        """
        k = window.shape[1] - 1
        if k < 1:
            raise ValueError("windows need at least two frames")
        if k == 1:
            return self.loss(window[:, 0], window[:, 1])
        if k != self.config.rollout_steps:
            raise ValueError(
                f"windows have {k} steps, config.rollout_steps is {self.config.rollout_steps}"
            )
        z = self.encoder(window[:, 0])
        with torch.no_grad():
            targets = self.target_encoder(window[:, 1:])
        weights = self.config.step_weights().to(z)
        pred = z
        total = z.new_zeros(())
        for j in range(k):
            pred = self._step(pred, None)
            total = total + weights[j] * nn.functional.l1_loss(pred, targets[:, j])
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
        """Online-encoder latents of frames ``x`` of shape ``(..., *frame_shape)``."""
        was_training = self.training
        self.eval()
        z: Tensor = self.encoder(x)
        self.train(was_training)
        return z

    @torch.no_grad()
    def predict_step(
        self, z: Tensor, noise: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        """One latent step ``z_t -> z_{t+1}``.

        For a stochastic predictor, ``noise`` (shape ``(..., noise_dim)``) is used if given;
        otherwise it is drawn from ``generator`` (the global generator if None).
        """
        if noise is None:
            noise = self._draw(z, generator)
        return self._step(z, noise)

    @torch.no_grad()
    def rollout(
        self,
        z0: Tensor,
        k: int,
        noise: Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Autoregressive latent rollout; returns ``(B, k+1, d)`` with ``z0`` first.

        For a stochastic predictor, ``noise`` of shape ``(B, k, noise_dim)`` fixes the noise
        of every step, so that two rollouts given the same tensor share their noise sequence;
        otherwise each step draws from ``generator`` (the global generator if None).
        """
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if noise is not None and tuple(noise.shape[:2]) != (z0.shape[0], k):
            raise ValueError(f"noise must start with shape {(z0.shape[0], k)}")
        zs = [z0]
        for j in range(k):
            step_noise = noise[:, j] if noise is not None else None
            zs.append(self.predict_step(zs[-1], step_noise, generator))
        return torch.stack(zs, dim=1)


def encode_numpy(
    model: SequenceJEPA, frames: NDArray[np.floating], *, batch_size: int = 1024
) -> NDArray[np.float64]:
    """Online-encoder latents of ``frames`` ``(..., *frame_shape)`` as float64, in batches.

    Frames are converted to float32 one batch at a time, which bounds memory on large arrays.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    shape = model.config.frame_shape
    n = len(shape)
    if frames.ndim < n or tuple(frames.shape[frames.ndim - n :]) != shape:
        raise ValueError(f"frames must end with shape {shape}, got {frames.shape}")
    lead = frames.shape[: frames.ndim - n]
    flat = frames.reshape(-1, *shape)
    out = np.empty((flat.shape[0], model.config.latent_dim), dtype=np.float64)
    for start in range(0, flat.shape[0], batch_size):
        chunk = torch.as_tensor(flat[start : start + batch_size], dtype=torch.float32)
        out[start : start + batch_size] = model.encode(chunk).numpy()
    return out.reshape(*lead, model.config.latent_dim)


def train_jepa(
    sequences: NDArray[np.floating],
    config: JEPAConfig,
    *,
    n_steps: int,
    seed: int,
    progress: Callable[[int, float, float], None] | None = None,
) -> tuple[SequenceJEPA, TrainHistory]:
    """Train a :class:`SequenceJEPA` on ``sequences`` of shape ``(N, T, *frame_shape)``.

    Each step samples windows of ``rollout_steps + 1`` consecutive frames (pairs by default)
    uniformly over all sequences and start times, and minimises :meth:`SequenceJEPA.rollout_loss`.
    ``progress(step, loss, latent_std)``, if given, is called at every logged step.
    """
    shape = config.frame_shape
    k = config.rollout_steps
    if sequences.ndim != 2 + len(shape) or tuple(sequences.shape[2:]) != shape:
        raise ValueError(
            f"sequences must be (N, T, {', '.join(map(str, shape))}), got {sequences.shape}"
        )
    if sequences.shape[1] < k + 1:
        raise ValueError(f"sequences need at least rollout_steps + 1 = {k + 1} frames")
    rng = seed_everything(seed)
    data = torch.as_tensor(sequences, dtype=torch.float32)
    per_seq = data.shape[1] - k
    n_windows = data.shape[0] * per_seq
    offsets = torch.arange(k + 1)

    model = SequenceJEPA(config)
    params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    opt = torch.optim.AdamW(params, lr=config.lr)
    history = TrainHistory(collapse_threshold=config.collapse_threshold)
    model.train()
    for step in range(n_steps):
        idx = torch.as_tensor(rng.integers(0, n_windows, size=min(config.batch_size, n_windows)))
        seq, start = idx // per_seq, idx % per_seq
        window = data[seq[:, None], start[:, None] + offsets]
        loss, z = model.rollout_loss(window)
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        model.update_target()
        if step % config.log_every == 0 or step == n_steps - 1:
            history.steps.append(step)
            history.loss.append(float(loss.detach()))
            history.latent_std.append(mean_latent_std(z))
            if progress is not None:
                progress(step, history.loss[-1], history.latent_std[-1])
    return model, history
