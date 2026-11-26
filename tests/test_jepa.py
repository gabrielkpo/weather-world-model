import numpy as np
import pytest
import torch

from weather_world_model.jepa import (
    JEPAConfig,
    SequenceJEPA,
    TrainHistory,
    is_collapsed,
    mean_latent_std,
    train_jepa,
)


def _linear_sequences(n: int, t: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    a = 0.95 * q
    x = np.empty((n, t, d))
    x[:, 0] = rng.standard_normal((n, d))
    for i in range(1, t):
        x[:, i] = x[:, i - 1] @ a.T
    return x


def test_training_reduces_loss_without_collapse() -> None:
    seqs = _linear_sequences(32, 10, 4, seed=0)
    cfg = JEPAConfig(input_dim=4, latent_dim=8, hidden_dim=32, batch_size=64)
    model, hist = train_jepa(seqs, cfg, n_steps=200, seed=0)
    assert np.mean(hist.loss[-3:]) < hist.loss[0]
    assert not hist.collapsed
    assert hist.latent_std[-1] > cfg.collapse_threshold
    assert isinstance(model, SequenceJEPA)


def test_shapes() -> None:
    cfg = JEPAConfig(input_dim=4, latent_dim=8, hidden_dim=16)
    model = SequenceJEPA(cfg)
    z = model.encode(torch.zeros(3, 5, 4))
    assert z.shape == (3, 5, 8)
    assert model.predict_step(z[:, 0]).shape == (3, 8)
    assert model.rollout(z[:, 0], 6).shape == (3, 7, 8)
    torch.testing.assert_close(model.rollout(z[:, 0], 6)[:, 0], z[:, 0])
    with pytest.raises(ValueError):
        model.rollout(z[:, 0], -1)


def test_collapse_detector() -> None:
    const = torch.ones(50, 8)
    assert mean_latent_std(const) == 0.0
    assert is_collapsed(mean_latent_std(const))
    assert not is_collapsed(mean_latent_std(torch.randn(50, 8)))
    hist = TrainHistory(latent_std=[1.0, 0.0])
    assert hist.collapsed
    assert not TrainHistory().collapsed


def test_config_and_input_validation() -> None:
    with pytest.raises(ValueError):
        JEPAConfig(input_dim=4, context_len=2)
    with pytest.raises(ValueError):
        JEPAConfig(input_dim=4, ema_momentum=1.5)
    cfg = JEPAConfig(input_dim=4)
    with pytest.raises(ValueError):
        train_jepa(np.zeros((2, 5, 3)), cfg, n_steps=1, seed=0)
    with pytest.raises(ValueError):
        train_jepa(np.zeros((2, 1, 4)), cfg, n_steps=1, seed=0)
