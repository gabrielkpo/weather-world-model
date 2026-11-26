import numpy as np
import pytest

from weather_world_model.diagnostics.readouts import (
    fit_readout_probes,
    r2_score,
    ridge,
    rollout_readouts,
)


def test_ridge_recovers_linear_map() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 5))
    y = x @ np.arange(1.0, 6.0) + 2.0
    w = ridge(x, y, 1e-6)
    pred = np.hstack([x, np.ones((200, 1))]) @ w
    assert r2_score(y, pred) > 0.999
    assert np.isnan(r2_score(np.ones(3), np.ones(3)))


def _rotation_world(n: int, t: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Latents rotating in a plane by a known map; the quantity is the first coordinate."""
    rng = np.random.default_rng(seed)
    c, s = np.cos(0.3), np.sin(0.3)
    step = np.array([[c, -s], [s, c]])
    z = np.empty((n, t, 2))
    z[:, 0] = rng.standard_normal((n, 2))
    for i in range(1, t):
        z[:, i] = z[:, i - 1] @ step.T
    return z, z[..., :1] * 2.0 + 1.0, step


def test_rollout_readouts_beat_persistence_with_true_dynamics() -> None:
    z, q, step = _rotation_world(60, 12, seed=0)
    split = {"train": z[:40], "val": z[40:50], "test": z[50:]}
    quant = {"train": q[:40], "val": q[40:50], "test": q[50:]}
    probes = fit_readout_probes(
        {k: v.reshape(-1, 2) for k, v in split.items()},
        {k: v.reshape(-1, 1) for k, v in quant.items()},
        ["first"],
    )
    assert probes.scores["first"]["test_r2"] > 0.999

    def true_rollout(z0: np.ndarray, k: int) -> np.ndarray:
        zs = [z0]
        for _ in range(k):
            zs.append(zs[-1] @ step.T)
        return np.stack(zs, axis=1)

    def frozen(z0: np.ndarray, k: int) -> np.ndarray:
        return np.repeat(z0[:, None], k + 1, axis=1)

    good = rollout_readouts(probes, true_rollout, split["test"], quant["test"], horizons=(1, 5))
    assert good["h5"]["first"]["rollout_r2"] > 0.999
    assert good["h5"]["first"]["persistence_r2"] < 0.5
    assert good["h1"]["first"]["persistence_r2"] > good["h5"]["first"]["persistence_r2"]
    # A predictor that does nothing scores exactly as persistence.
    same = rollout_readouts(probes, frozen, split["test"], quant["test"], horizons=(5,))
    assert same["h5"]["first"]["rollout_r2"] == pytest.approx(same["h5"]["first"]["persistence_r2"])


def test_rollout_readouts_average_sampled_rollouts() -> None:
    z, q, step = _rotation_world(40, 8, seed=1)
    flat = {k: z.reshape(-1, 2) for k in ("train", "val", "test")}
    probes = fit_readout_probes(flat, {k: q.reshape(-1, 1) for k in flat}, ["first"])
    rng = np.random.default_rng(0)

    def noisy(z0: np.ndarray, k: int) -> np.ndarray:
        zs = [z0]
        for _ in range(k):
            zs.append(zs[-1] @ step.T + 0.5 * rng.standard_normal(z0.shape))
        return np.stack(zs, axis=1)

    one = rollout_readouts(probes, noisy, z, q, horizons=(3,), n_samples=1)
    many = rollout_readouts(probes, noisy, z, q, horizons=(3,), n_samples=32)
    assert many["h3"]["first"]["rollout_r2"] > one["h3"]["first"]["rollout_r2"]
    with pytest.raises(ValueError, match="does not fit"):
        rollout_readouts(probes, noisy, z, q, horizons=(8,))
    with pytest.raises(ValueError, match="positive"):
        rollout_readouts(probes, noisy, z, q, horizons=(1,), n_samples=0)
