import numpy as np
import pytest

from weather_world_model.diagnostics.arrow import (
    arrow_of_time_dataset,
    probe_accuracy,
    train_probe,
)

N, T, D = 400, 8, 3


def _relaxation(rng: np.random.Generator) -> np.ndarray:
    target = 3.0 * np.ones(D)
    x = np.empty((N, T, D))
    x[:, 0] = 0.3 * rng.standard_normal((N, D))
    for t in range(1, T):
        x[:, t] = target + 0.6 * (x[:, t - 1] - target) + 0.05 * rng.standard_normal((N, D))
    return x


def _rotations(rng: np.random.Generator) -> np.ndarray:
    x = np.empty((N, T, D))
    for n in range(N):
        q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        x[n, 0] = rng.standard_normal(D)
        for t in range(1, T):
            x[n, t] = q @ x[n, t - 1]
    return x


def _accuracy(x: np.ndarray, rng: np.random.Generator) -> float:
    f, y = arrow_of_time_dataset(x, rng)
    half = len(y) // 2
    probe = train_probe(f[:half], y[:half], l2=1e-3, seed=0)
    return probe_accuracy(probe, f[half:], y[half:])


def test_irreversible_is_readable() -> None:
    rng = np.random.default_rng(0)
    assert _accuracy(_relaxation(rng), rng) > 0.95


def test_reversible_is_chance() -> None:
    rng = np.random.default_rng(0)
    assert _accuracy(_rotations(rng), rng) < 0.6


def test_dataset_shapes_and_validation() -> None:
    rng = np.random.default_rng(0)
    f, y = arrow_of_time_dataset(np.zeros((5, 4, 2)), rng)
    assert f.shape == (5, 6) and y.shape == (5,)
    with pytest.raises(ValueError):
        arrow_of_time_dataset(np.zeros((5, 1, 2)), rng)
    with pytest.raises(ValueError):
        train_probe(f, 2 * y - 1, l2=0.0, seed=0)


def test_reversed_class_can_come_from_another_array() -> None:
    import numpy as np
    import pytest

    from weather_world_model.diagnostics.arrow import arrow_of_time_dataset, time_asymmetry_features

    rng = np.random.default_rng(0)
    seqs = rng.standard_normal((40, 6, 3))
    other = rng.standard_normal((40, 6, 3))
    x, y = arrow_of_time_dataset(
        seqs, np.random.default_rng(1), features="asymmetry", reversed_from=other
    )
    assert np.allclose(x[y == 1], time_asymmetry_features(seqs)[y == 1])
    assert np.allclose(x[y == 0], -time_asymmetry_features(other)[y == 0])
    # Same labels and features as the plain call when the alternate array is the input itself.
    x_plain, y_plain = arrow_of_time_dataset(seqs, np.random.default_rng(1), features="asymmetry")
    x_same, y_same = arrow_of_time_dataset(
        seqs, np.random.default_rng(1), features="asymmetry", reversed_from=seqs
    )
    assert np.array_equal(y_plain, y_same) and np.array_equal(x_plain, x_same)
    with pytest.raises(ValueError, match="reversed_from"):
        arrow_of_time_dataset(seqs, rng, reversed_from=other[:, :5])
