import numpy as np
import pytest

from weather_world_model.worlds import (
    DIAGNOSTIC_NAMES,
    QGConfig,
    QGModel,
    diagnostics,
    simulate_snapshots,
    simulate_with_diagnostics,
    snapshot_diagnostics,
    true_growth_rate,
    twin_divergence,
)
from weather_world_model.worlds.qg import count_regions, velocity_saturation


@pytest.fixture(scope="module")
def spun_up() -> QGModel:
    model = QGModel(QGConfig(), np.random.default_rng(0))
    model.spin_up(8000)
    return model


def _grid(model: QGModel) -> tuple[np.ndarray, np.ndarray]:
    n = model.config.nx
    x = np.arange(n) * model.dx
    return np.meshgrid(x, x)  # X varies along axis 1, Y along axis 0


def test_spectral_derivatives_of_analytic_streamfunction() -> None:
    config = QGConfig(nx=32)
    model = QGModel(config, np.random.default_rng(0))
    xx, yy = _grid(model)
    psi1 = np.sin(2 * xx) * np.cos(3 * yy)
    psi2 = np.cos(xx + yy)
    lap1, lap2 = -13 * psi1, -2 * psi2
    q = np.stack([lap1 + config.F1 * (psi2 - psi1), lap2 + config.F2 * (psi1 - psi2)])
    model.state = q
    np.testing.assert_allclose(model.streamfunction(), np.stack([psi1, psi2]), atol=1e-10)
    u, v = model.velocity()
    np.testing.assert_allclose(u[0], 3 * np.sin(2 * xx) * np.sin(3 * yy), atol=1e-10)
    np.testing.assert_allclose(v[0], 2 * np.cos(2 * xx) * np.cos(3 * yy), atol=1e-10)
    np.testing.assert_allclose(u[1], np.sin(xx + yy), atol=1e-10)
    np.testing.assert_allclose(model.vorticity(), np.stack([lap1, lap2]), atol=1e-9)


def test_velocity_is_divergence_free(spun_up: QGModel) -> None:
    u, v = spun_up.velocity()
    uh, vh = np.fft.rfft2(u), np.fft.rfft2(v)
    div = np.fft.irfft2(spun_up.ik * uh + spun_up.il * vh, s=u.shape[1:])
    assert np.max(np.abs(div)) < 1e-10 * np.max(np.abs(u))


def test_energy_bounded_and_finite(spun_up: QGModel) -> None:
    model = spun_up.copy()
    energies = []
    for _ in range(10):
        model.step(200)
        d = diagnostics(model)
        assert all(np.isfinite(val) for val in d.values())
        energies.append(d["kinetic_energy"] + d["available_potential_energy"])
    e = np.array(energies)
    assert np.all(e > 0.2)
    assert np.all(e < 5.0)
    assert e.max() / e.min() < 3.0  # statistically stationary, no runaway


def test_copy_is_independent_and_reproducible() -> None:
    a = QGModel(QGConfig(nx=32), np.random.default_rng(3))
    b = QGModel(QGConfig(nx=32), np.random.default_rng(3))
    a.step(50)
    b.step(50)
    np.testing.assert_array_equal(a.state, b.state)
    c = a.copy()
    c.step(5)
    a.step(5)
    np.testing.assert_array_equal(a.state, c.state)
    c.step(1)
    assert not np.array_equal(a.state, c.state)
    assert c.n_steps == a.n_steps + 1


def test_diagnostics_names_and_signs(spun_up: QGModel) -> None:
    d = diagnostics(spun_up)
    assert tuple(d) == DIAGNOSTIC_NAMES
    assert d["max_cyclonic_vorticity"] > 0
    assert d["max_anticyclonic_vorticity"] > 0
    assert d["n_vortices"] >= 1


def test_count_regions_periodic() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, :3] = mask[9, :3] = True  # wraps across the y boundary: one region of 6
    mask[5, 5] = True  # too small
    assert count_regions(mask, min_cells=4) == 1
    assert count_regions(mask, min_cells=1) == 2


def test_snapshots_shapes_and_alignment() -> None:
    config = QGConfig(nx=16)
    kw = {"n_trajectories": 2, "n_steps": 30, "sample_every": 10, "spin_up": 20}
    snaps, diags, names = simulate_with_diagnostics(
        config, rng=np.random.default_rng(1), field="q", **kw
    )
    assert snaps.shape == (2, 3, 2, 16, 16)
    assert snaps.dtype == np.float32
    assert diags.shape == (2, 3, len(DIAGNOSTIC_NAMES))
    assert names == DIAGNOSTIC_NAMES
    d2, _ = snapshot_diagnostics(config, rng=np.random.default_rng(1), **kw)
    np.testing.assert_array_equal(diags, d2)
    for field in ("psi", "uv"):
        out = simulate_snapshots(config, rng=np.random.default_rng(1), field=field, **kw)
        assert out.shape == (2, 3, 2, 16, 16)
    with pytest.raises(ValueError, match="field"):
        simulate_snapshots(config, rng=np.random.default_rng(1), field="w", **kw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sample_every"):
        simulate_snapshots(
            config,
            n_trajectories=1,
            n_steps=10,
            sample_every=0,
            spin_up=0,
            rng=np.random.default_rng(1),
        )


def test_config_validation_and_state_shape() -> None:
    with pytest.raises(ValueError):
        QGConfig(nx=7)
    with pytest.raises(ValueError):
        QGConfig(dt=0.0)
    with pytest.raises(ValueError):
        QGConfig(rek=-1.0)
    model = QGModel(QGConfig(nx=16), np.random.default_rng(0))
    with pytest.raises(ValueError):
        model.state = np.zeros((2, 8, 8))


def test_twin_divergence_grows(spun_up: QGModel) -> None:
    before = spun_up.state.copy()
    curve = twin_divergence(
        spun_up, epsilon=1e-6, n_samples=40, sample_every=20, rng=np.random.default_rng(2)
    )
    np.testing.assert_array_equal(spun_up.state, before)
    assert curve.shape == (41,)
    assert curve[0] > 0
    assert curve[-1] > 10 * curve[1]
    assert curve[-1] < velocity_saturation(spun_up)
    rate = true_growth_rate(
        spun_up, epsilon=1e-6, n_samples=40, sample_every=20, rng=np.random.default_rng(2)
    )
    assert rate > 0


@pytest.mark.slow
def test_spun_up_flow_is_turbulent() -> None:
    model = QGModel(QGConfig(), np.random.default_rng(5))
    model.spin_up(10_000)
    d = diagnostics(model)
    assert d["n_vortices"] >= 2
    assert 0.3 < d["jet_speed_max"] < 5.0


def _total_energy_and_enstrophy(model: QGModel) -> tuple[float, float]:
    d = diagnostics(model)
    return d["kinetic_energy"] + d["available_potential_energy"], d["enstrophy"]


@pytest.mark.parametrize("beta", [0.0, 6.0])
def test_inviscid_unforced_flow_conserves_energy_and_enstrophy(beta: float) -> None:
    # Without background shear, bottom drag and spectral filter, the two-layer QG equations
    # conserve total energy and potential enstrophy; the spectral Galerkin truncation keeps
    # both exactly, so only the time-stepping error remains. The initial state is truncated
    # to the dealiased wavenumbers, otherwise aliasing of the unresolved modes breaks it.
    cfg = QGConfig(U1=0.0, U2=0.0, rek=0.0, beta=beta, filter_cutoff=100.0, dt=0.002)
    model = QGModel(cfg, np.random.default_rng(0))
    model._qh = model._qh * model._dealias
    e0, z0 = _total_energy_and_enstrophy(model)
    model.step(1000)
    e1, z1 = _total_energy_and_enstrophy(model)
    assert abs(e1 - e0) / e0 < 1e-4
    assert abs(z1 - z0) / z0 < 1e-5


def test_time_stepping_error_shrinks_with_dt() -> None:
    drifts = []
    for dt in (0.002, 0.001):
        cfg = QGConfig(U1=0.0, U2=0.0, rek=0.0, filter_cutoff=100.0, dt=dt)
        model = QGModel(cfg, np.random.default_rng(1))
        model._qh = model._qh * model._dealias
        e0, _ = _total_energy_and_enstrophy(model)
        model.step(round(2.0 / dt))
        e1, _ = _total_energy_and_enstrophy(model)
        drifts.append(abs(e1 - e0) / e0)
    # third-order scheme in theory, at least second order observed
    assert drifts[1] < drifts[0] / 3
