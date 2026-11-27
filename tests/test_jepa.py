from typing import Any

import numpy as np
import pytest
import torch

from weather_world_model.jepa import (
    JEPAConfig,
    SequenceJEPA,
    TrainHistory,
    encode_numpy,
    energy_score,
    is_collapsed,
    mean_latent_std,
    train_jepa,
)
from weather_world_model.seeding import seed_everything


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
    assert is_collapsed(float("nan"))


def test_config_and_input_validation() -> None:
    with pytest.raises(ValueError):
        JEPAConfig(input_dim=4, context_len=2)
    with pytest.raises(ValueError):
        JEPAConfig(input_dim=4, ema_momentum=1.5)
    with pytest.raises(ValueError):
        JEPAConfig(input_dim=4, log_every=0)
    cfg = JEPAConfig(input_dim=4)
    with pytest.raises(ValueError):
        train_jepa(np.zeros((2, 5, 3)), cfg, n_steps=1, seed=0)
    with pytest.raises(ValueError):
        train_jepa(np.zeros((2, 1, 4)), cfg, n_steps=1, seed=0)


def test_rollout_loss_k1_equals_one_step_loss() -> None:
    torch.manual_seed(0)
    cfg = JEPAConfig(input_dim=4, latent_dim=8, hidden_dim=16)
    model = SequenceJEPA(cfg)
    window = torch.randn(16, 2, 4)
    expected, z_expected = model.loss(window[:, 0], window[:, 1])
    got, z = model.rollout_loss(window)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(z, z_expected)


def test_k1_training_matches_previous_pair_sampling() -> None:
    # Reference: the pre-rollout implementation of train_jepa (pairs flattened over N and T).
    seqs = _linear_sequences(6, 7, 4, seed=1)
    cfg = JEPAConfig(input_dim=4, latent_dim=8, hidden_dim=16, batch_size=8, log_every=1)
    _, hist = train_jepa(seqs, cfg, n_steps=5, seed=3)

    rng = seed_everything(3)
    data = torch.as_tensor(seqs, dtype=torch.float32)
    x_t, x_next = data[:, :-1].reshape(-1, 4), data[:, 1:].reshape(-1, 4)
    model = SequenceJEPA(cfg)
    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=cfg.lr
    )
    losses = []
    for _ in range(5):
        idx = torch.as_tensor(rng.integers(0, x_t.shape[0], size=8))
        loss, _ = model.loss(x_t[idx], x_next[idx])
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
        model.update_target()
        losses.append(float(loss.detach()))
    np.testing.assert_allclose(hist.loss, losses, rtol=1e-6)


@pytest.mark.parametrize("weighting", ["uniform", "linear"])
def test_multistep_training_reduces_loss(weighting: str) -> None:
    seqs = _linear_sequences(32, 12, 4, seed=0)
    cfg = JEPAConfig(
        input_dim=4,
        latent_dim=8,
        hidden_dim=32,
        batch_size=64,
        rollout_steps=5,
        rollout_weighting=weighting,  # type: ignore[arg-type]
    )
    _, hist = train_jepa(seqs, cfg, n_steps=150, seed=0)
    assert np.mean(hist.loss[-3:]) < hist.loss[0]
    assert not hist.collapsed
    w = cfg.step_weights()
    assert w.shape == (5,)
    torch.testing.assert_close(w.sum(), torch.tensor(1.0))


def test_rollout_loss_gradient_flows_through_rollout() -> None:
    cfg = JEPAConfig(input_dim=4, latent_dim=8, hidden_dim=16, rollout_steps=3)
    model = SequenceJEPA(cfg)
    loss, _ = model.rollout_loss(torch.randn(8, 4, 4))
    loss.backward()  # type: ignore[no-untyped-call]
    assert all(p.grad is not None for p in model.predictor.parameters())
    assert all(p.grad is None for p in model.target_encoder.parameters())
    with pytest.raises(ValueError):
        model.rollout_loss(torch.randn(8, 3, 4))
    with pytest.raises(ValueError):
        model.rollout_loss(torch.randn(8, 1, 4))


def test_conv_encoder_shapes_and_circular_padding() -> None:
    torch.manual_seed(0)
    cfg = JEPAConfig(
        encoder="conv", input_shape=(2, 32, 32), latent_dim=8, conv_channels=(4, 8, 8, 8)
    )
    model = SequenceJEPA(cfg)
    convs = [m for m in model.encoder.modules() if isinstance(m, torch.nn.Conv2d)]
    assert len(convs) == 4
    assert all(c.padding_mode == "circular" for c in convs)
    x = torch.randn(3, 5, 2, 32, 32)
    z = model.encode(x)
    assert z.shape == (3, 5, 8)
    # Shifting by 2**4 pixels is an exact symmetry with circular padding and a global pool.
    shifted = torch.roll(x, shifts=(16, 16), dims=(-2, -1))
    torch.testing.assert_close(model.encode(shifted), z, rtol=1e-4, atol=1e-5)
    assert (model.encode(torch.randn_like(x)) - z).abs().mean() > 1e-3


def test_conv_training_on_tiny_fields() -> None:
    rng = np.random.default_rng(0)
    base = rng.standard_normal((6, 1, 2, 16, 16))
    seqs = np.concatenate([np.roll(base, t, axis=-1) for t in range(6)], axis=1)
    cfg = JEPAConfig(
        encoder="conv",
        input_shape=(2, 16, 16),
        latent_dim=8,
        hidden_dim=16,
        conv_channels=(8, 8),
        batch_size=16,
        rollout_steps=3,
    )
    model, hist = train_jepa(seqs, cfg, n_steps=30, seed=0)
    assert not hist.collapsed
    assert np.isfinite(hist.loss).all()
    assert model.encode(torch.as_tensor(seqs[:2], dtype=torch.float32)).shape == (2, 6, 8)


def test_mlp_encoder_on_fields_and_encode_numpy() -> None:
    cfg = JEPAConfig(input_shape=(2, 4, 4), latent_dim=8, hidden_dim=16)
    assert cfg.frame_shape == (2, 4, 4)
    model = SequenceJEPA(cfg)
    frames = np.random.default_rng(0).standard_normal((5, 7, 2, 4, 4))
    expected = model.encode(torch.as_tensor(frames, dtype=torch.float32)).numpy()
    got = encode_numpy(model, frames, batch_size=4)
    assert got.shape == (5, 7, 8)
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        encode_numpy(model, frames[..., :3])
    with pytest.raises(ValueError):
        encode_numpy(model, frames, batch_size=0)


def test_new_config_validation() -> None:
    bad: list[dict[str, Any]] = [
        {},
        {"input_shape": ()},
        {"input_dim": 5, "input_shape": (2, 2)},
        {"input_dim": 4, "rollout_steps": 0},
        {"input_dim": 4, "rollout_weighting": "exp"},
        {"input_dim": 4, "encoder": "rnn"},
        {"input_dim": 4, "encoder": "conv"},
        {"input_shape": (1, 4, 4), "conv_channels": ()},
    ]
    for kwargs in bad:
        with pytest.raises(ValueError):
            JEPAConfig(**kwargs)
    assert JEPAConfig(input_dim=4, input_shape=(2, 2)).frame_shape == (2, 2)
    cfg = JEPAConfig(input_dim=4, rollout_steps=4)
    with pytest.raises(ValueError):
        train_jepa(np.zeros((2, 4, 4)), cfg, n_steps=1, seed=0)


def test_deterministic_default_is_unchanged() -> None:
    torch.manual_seed(0)
    cfg = JEPAConfig(input_dim=4, latent_dim=8, hidden_dim=16)
    assert cfg.noise_dim == 0
    model = SequenceJEPA(cfg)
    assert model.predictor[0].in_features == 8
    x_t, x_next = torch.randn(16, 4), torch.randn(16, 4)
    got, z = model.loss(x_t, x_next)
    # The pre-noise formula, written out.
    zz = model.encoder(x_t)
    expected = torch.nn.functional.l1_loss(zz + model.predictor(zz), model.target_encoder(x_next))
    std = torch.sqrt(zz.var(dim=0) + 1e-4)
    expected = expected + cfg.var_weight * torch.relu(1.0 - std).mean()
    zc = zz - zz.mean(dim=0)
    cov = zc.T @ zc / 15
    off = cov - torch.diag(torch.diag(cov))
    expected = expected + cfg.cov_weight * off.pow(2).sum() / 8
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    torch.testing.assert_close(z, zz, rtol=0, atol=0)
    with pytest.raises(ValueError, match="takes no noise"):
        model.predict_step(z, noise=torch.zeros(16, 8))


def test_energy_score_hand_computed() -> None:
    p1 = torch.tensor([[1.0, 0.0]])
    p2 = torch.tensor([[0.0, 2.0]])
    target = torch.zeros(1, 2)
    # (1 + 2) / 2 - sqrt(5) / 2
    torch.testing.assert_close(energy_score(p1, p2, target), torch.tensor(1.5 - 5**0.5 / 2))
    # Identical predictions: the score is the plain distance to the target, with a finite gradient.
    p = torch.tensor([[3.0, 4.0]], requires_grad=True)
    score = energy_score(p, p, target)
    torch.testing.assert_close(score, torch.tensor(5.0))
    score.backward()  # type: ignore[no-untyped-call]
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_stochastic_config_and_noise_validation() -> None:
    with pytest.raises(ValueError):
        JEPAConfig(input_dim=4, noise_dim=-1)
    with pytest.raises(ValueError, match="one-step"):
        JEPAConfig(input_dim=4, noise_dim=8, rollout_steps=2)
    model = SequenceJEPA(JEPAConfig(input_dim=4, latent_dim=8, hidden_dim=16, noise_dim=3))
    assert model.predictor[0].in_features == 11
    z = torch.randn(5, 8)
    assert model.predict_step(z).shape == (5, 8)
    with pytest.raises(ValueError, match="shape"):
        model.predict_step(z, noise=torch.zeros(5, 2))
    with pytest.raises(ValueError, match="needs a noise"):
        model._step(z, None)
    with pytest.raises(ValueError, match="noise must start"):
        model.rollout(z, 4, noise=torch.zeros(5, 3, 3))
    loss, _ = model.loss(torch.randn(5, 4), torch.randn(5, 4))
    loss.backward()  # type: ignore[no-untyped-call]
    assert all(p.grad is not None for p in model.predictor.parameters())


def test_shared_noise_rollouts_are_identical() -> None:
    torch.manual_seed(0)
    model = SequenceJEPA(JEPAConfig(input_dim=4, latent_dim=8, hidden_dim=16, noise_dim=8))
    z0 = torch.randn(6, 8)
    noise = torch.randn(6, 10, 8)
    a = model.rollout(z0, 10, noise=noise)
    torch.testing.assert_close(a, model.rollout(z0.clone(), 10, noise=noise), rtol=0, atol=0)
    g1, g2 = torch.Generator().manual_seed(5), torch.Generator().manual_seed(5)
    b = model.rollout(z0, 10, generator=g1)
    torch.testing.assert_close(b, model.rollout(z0, 10, generator=g2), rtol=0, atol=0)
    # Independent noise gives different trajectories from the same start.
    assert (model.rollout(z0, 10) - model.rollout(z0, 10)).abs().max() > 1e-4
    torch.testing.assert_close(b[:, 0], z0)


def test_stochastic_predictor_keeps_bimodal_spread() -> None:
    # Next state = current state + s * a with s = +1 or -1 at random: two equally likely futures.
    rng = np.random.default_rng(0)
    d = 4
    a = np.zeros(d)
    a[0] = 3.0
    x0 = rng.standard_normal((4000, d))
    sign = rng.choice([-1.0, 1.0], size=(4000, 1))
    seqs = np.stack([x0, x0 + sign * a], axis=1)
    test = torch.as_tensor(rng.standard_normal((256, d)), dtype=torch.float32)
    shift = torch.as_tensor(a, dtype=torch.float32)
    ratios = {}
    offsets = {}
    for name, noise_dim in (("det", 0), ("stoch", 8)):
        cfg = JEPAConfig(input_dim=d, latent_dim=8, hidden_dim=64, noise_dim=noise_dim)
        model, hist = train_jepa(seqs, cfg, n_steps=1500, seed=0)
        assert not hist.collapsed
        with torch.no_grad():
            t_plus = model.target_encoder(test + shift)
            t_minus = model.target_encoder(test - shift)
        half = (t_plus - t_minus).norm(dim=-1, keepdim=True) / 2
        unit = (t_plus - t_minus) / (2 * half)
        mid = (t_plus + t_minus) / 2
        z = model.encode(test)
        samples = torch.stack([model.predict_step(z) for _ in range(64)])  # (64, B, d)
        proj = ((samples - mid) * unit).sum(dim=-1) / half.squeeze(-1)  # in half-distances
        ratios[name] = float(proj.std(dim=0).mean()) if noise_dim else 0.0
        offsets[name] = float(proj.mean(dim=0).abs().mean())
    # Truth along the axis joining the two futures: mean 0, std 1 (in half-distances).
    assert 0.6 < ratios["stoch"] < 1.4, ratios
    assert offsets["stoch"] < 0.4, offsets
    # The deterministic model returns one point between the two futures (any such point
    # minimises the L1 error), so it has no spread at all.
    assert offsets["det"] < 0.8, offsets


def test_conv_norm_none_keeps_field_amplitude() -> None:
    # GroupNorm rescales each sample: doubling the field barely moves the latent.
    # Without normalisation the latent moves with the amplitude.
    torch.manual_seed(0)
    x = torch.randn(4, 2, 16, 16)
    shifts = {}
    for norm in ("group", "none"):
        torch.manual_seed(1)
        cfg = JEPAConfig(
            encoder="conv", input_shape=(2, 16, 16), conv_channels=(8, 16), conv_norm=norm
        )
        model = SequenceJEPA(cfg)
        has_norm = any(isinstance(m, torch.nn.GroupNorm) for m in model.encoder.modules())
        assert has_norm == (norm == "group")
        z1, z2 = model.encode(x), model.encode(2.0 * x)
        shifts[norm] = float((z2 - z1).norm() / z1.norm())
    assert shifts["none"] > 5 * shifts["group"]
