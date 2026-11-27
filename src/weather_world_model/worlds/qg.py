"""Two-layer quasi-geostrophic (Phillips) atmosphere on a doubly periodic square.

Model
-----
The standard two-layer model (Phillips 1956; Vallis, *Atmospheric and Oceanic Fluid
Dynamics*, ch. 5-6 and 9) in the formulation used by pyqg (``pyqg.QGModel``). For layer
``k`` (1 = upper, 2 = lower) the potential-vorticity anomaly is

    q_1 = ∇²ψ_1 + F_1 (ψ_2 - ψ_1),    q_2 = ∇²ψ_2 + F_2 (ψ_1 - ψ_2),

i.e. ``q_k = ∇²ψ_k + (-1)^k F_k (ψ_1 - ψ_2)``; the full PV adds ``βy`` and the PV of the
uniform background zonal flow ``U_k``. With ``δ = H_1 / H_2`` and deformation radius ``r_d``,
``F_1 = 1 / (r_d² (1 + δ))`` and ``F_2 = δ F_1``. The anomaly evolves as

    ∂q_k/∂t + J(ψ_k, q_k) + U_k ∂q_k/∂x + Q_{k,y} ∂ψ_k/∂x = -δ_{k2} r_ek ∇²ψ_2 + filter,

with background PV gradients ``Q_{1,y} = β + F_1 (U_1 - U_2)`` and
``Q_{2,y} = β - F_2 (U_1 - U_2)``. The vertical shear ``U_1 - U_2`` makes the flow
baroclinically unstable; linear bottom drag ``r_ek`` on the lower layer and the small-scale
filter remove the energy the instability releases, so the flow becomes statistically
stationary turbulence with a meandering eddy-driven jet and coherent vortices.

Numerics
--------
Pseudo-spectral on an ``nx x nx`` grid (real FFTs), Jacobian computed on the grid in
advective form with 2/3-rule dealiasing, third-order Adams-Bashforth time stepping (first
two steps Euler and AB2) and the exponential spectral filter of pyqg
(``exp(-23.6 (κΔx - 0.65π)^4)`` above ``κΔx = 0.65π``), applied every step. The bottom drag
is integrated explicitly with the other tendencies.

Units and defaults
------------------
Non-dimensional: lengths in units of ``L / 2π`` (domain ``2π``), time in units of
``1 / (U_1 - U_2)`` per unit length. The defaults keep pyqg's default regime in terms of
the dimensionless groups that set the dynamics, adjusted to a coarse 64² grid so that the
deformation radius is resolved by ~2.5 grid points:

* ``r_d = 0.25`` (``L / r_d ≈ 25``, pyqg default ≈ 67 at 64² where ``r_d ≈ Δx``),
* ``δ = 1`` (equal layer depths, Vallis' symmetric Phillips model),
* ``U_1 - U_2 = 1`` with ``U_2 = 0``,
* ``β = 6``: supercriticality ``F_1 (U_1 - U_2) / β ≈ 1.3``. pyqg's default (≈ 5.9) lets
  energy run away to the box scale on this smaller domain (tested: ``β ≤ 2`` with weak drag
  blows up); a stronger β arrests the inverse cascade into zonal jets,
* ``r_ek = 1``, i.e. ``r_ek r_d / ΔU = 0.25`` (pyqg default ≈ 0.35),
* ``dt = 0.005``: peak winds reach ~7, so the advective CFL number is ≈ 0.35, inside the
  AB3 stability range,
* ``init_amplitude = 0.5`` (white noise on q) skips the slow linear phase; the flow is
  statistically stationary after ~2000 steps (``t ≈ 10``), spin-ups of 10 000 steps are
  used in the tests and preview. 10 000 steps take ~3 s on an M1 CPU.

Velocities returned by :meth:`QGModel.velocity` are the eddy (anomaly) winds
``u = -∂ψ/∂y``, ``v = ∂ψ/∂x``; the uniform background ``U_k`` is not added.
"""

from __future__ import annotations

import copy as _copy
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from weather_world_model.diagnostics.predictability import growth_rate, growth_window, normalised

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
Field = Literal["q", "psi", "uv"]

DIAGNOSTIC_NAMES: tuple[str, ...] = (
    "kinetic_energy",
    "available_potential_energy",
    "enstrophy",
    "jet_speed_max",
    "max_cyclonic_vorticity",
    "max_anticyclonic_vorticity",
    "n_vortices",
)
# The four diagnostics with a pass threshold in the step-2 criteria (W-P, W-R, W-V).
THRESHOLDED: tuple[str, ...] = (
    "kinetic_energy",
    "available_potential_energy",
    "enstrophy",
    "jet_speed_max",
)

# Coherent-vortex rule (Isern-Fontanet et al. 2003, Okubo-Weiss): a vortex is a connected
# (periodic, 4-neighbour) region of the upper layer where W = s_n² + s_s² - ζ² < -0.2 std(W),
# containing at least VORTEX_MIN_CELLS grid cells.
OKUBO_WEISS_FACTOR = 0.2
VORTEX_MIN_CELLS = 16


@dataclass(frozen=True)
class QGConfig:
    """Parameters of the two-layer QG model (non-dimensional, see the module docstring)."""

    nx: int = 64
    L: float = 2 * np.pi
    beta: float = 6.0
    rd: float = 0.25
    delta: float = 1.0
    U1: float = 1.0
    U2: float = 0.0
    rek: float = 1.0
    dt: float = 0.005
    filterfac: float = 23.6
    filter_cutoff: float = 0.65 * np.pi
    init_amplitude: float = 0.5

    def __post_init__(self) -> None:
        if self.nx < 8 or self.nx % 2:
            raise ValueError("nx must be an even integer >= 8")
        for name in ("L", "rd", "delta", "dt"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.rek < 0:
            raise ValueError("rek must be non-negative")

    @property
    def F1(self) -> float:  # noqa: N802 (physics notation)
        return 1.0 / (self.rd**2 * (1.0 + self.delta))

    @property
    def F2(self) -> float:  # noqa: N802 (physics notation)
        return self.delta * self.F1

    @property
    def layer_weights(self) -> tuple[float, float]:
        """Layer depth fractions ``H_1 / H`` and ``H_2 / H``."""
        return self.delta / (1.0 + self.delta), 1.0 / (1.0 + self.delta)


class QGModel:
    """Two-layer QG model; ``state`` is the PV anomaly ``q`` of shape ``(2, nx, nx)``."""

    def __init__(self, config: QGConfig, rng: np.random.Generator) -> None:
        self.config = config
        c = config
        n = c.nx
        self.dx = c.L / n
        k = 2 * np.pi * np.fft.fftfreq(n, d=self.dx)  # y wavenumbers (full axis)
        l_ = 2 * np.pi * np.fft.rfftfreq(n, d=self.dx)  # x wavenumbers (half axis)
        self.ky = k[:, None] * np.ones_like(l_)[None, :]
        self.kx = np.ones_like(k)[:, None] * l_[None, :]
        self.ik = 1j * self.kx
        self.il = 1j * self.ky
        self.wv2 = self.kx**2 + self.ky**2
        wv = np.sqrt(self.wv2)

        # PV inversion: [q1, q2] = M [psi1, psi2] per wavenumber; det = K² (K² + F1 + F2).
        f1, f2 = c.F1, c.F2
        det = self.wv2 * (self.wv2 + f1 + f2)
        det_inv = np.zeros_like(det)
        det_inv[det > 0] = 1.0 / det[det > 0]
        self._inv = np.empty((2, 2, *det.shape))
        self._inv[0, 0] = -(self.wv2 + f2) * det_inv
        self._inv[0, 1] = -f1 * det_inv
        self._inv[1, 0] = -f2 * det_inv
        self._inv[1, 1] = -(self.wv2 + f1) * det_inv

        self._ubg = np.array([c.U1, c.U2])[:, None, None]
        self._qy = np.array([c.beta + f1 * (c.U1 - c.U2), c.beta - f2 * (c.U1 - c.U2)])[
            :, None, None
        ]
        kmax = n // 2 * 2 * np.pi / c.L
        self._dealias = (np.abs(self.kx) < 2 / 3 * kmax) & (np.abs(self.ky) < 2 / 3 * kmax)
        wvdx = wv * self.dx
        self._filter = np.where(
            wvdx > c.filter_cutoff, np.exp(-c.filterfac * (wvdx - c.filter_cutoff) ** 4), 1.0
        )

        q0 = c.init_amplitude * rng.standard_normal((2, n, n))
        self._qh: ComplexArray = self._filter * np.fft.rfft2(
            q0 - q0.mean(axis=(1, 2), keepdims=True)
        )
        self._qh[:, 0, 0] = 0.0
        self._tend_hist: deque[ComplexArray] = deque(maxlen=2)
        self.t = 0.0
        self.n_steps = 0

    # ----------------------------------------------------------------- state and fields
    @property
    def state(self) -> FloatArray:
        """PV anomaly ``q``, shape ``(2, ny, nx)`` (axis 1 is y, axis 2 is x)."""
        return np.asarray(np.fft.irfft2(self._qh, s=(self.config.nx, self.config.nx)))

    @state.setter
    def state(self, q: FloatArray) -> None:
        n = self.config.nx
        if q.shape != (2, n, n):
            raise ValueError(f"state must have shape (2, {n}, {n})")
        self._qh = np.fft.rfft2(np.asarray(q, dtype=np.float64))
        self._qh[:, 0, 0] = 0.0
        self._tend_hist.clear()

    def _psih(self, qh: ComplexArray) -> ComplexArray:
        inv = self._inv
        return np.stack(
            [inv[0, 0] * qh[0] + inv[0, 1] * qh[1], inv[1, 0] * qh[0] + inv[1, 1] * qh[1]]
        )

    def _irfft(self, fh: ComplexArray) -> FloatArray:
        n = self.config.nx
        return np.asarray(np.fft.irfft2(fh, s=(n, n)))

    def streamfunction(self) -> FloatArray:
        """Streamfunction ``ψ``, shape ``(2, ny, nx)``."""
        return self._irfft(self._psih(self._qh))

    def velocity(self) -> tuple[FloatArray, FloatArray]:
        """Eddy winds ``u = -∂ψ/∂y`` and ``v = ∂ψ/∂x``, each ``(2, ny, nx)``."""
        ph = self._psih(self._qh)
        return self._irfft(-self.il * ph), self._irfft(self.ik * ph)

    def vorticity(self) -> FloatArray:
        """Relative vorticity ``ζ = ∇²ψ``, shape ``(2, ny, nx)``."""
        return self._irfft(-self.wv2 * self._psih(self._qh))

    # ----------------------------------------------------------------- dynamics
    def _tendency(self, qh: ComplexArray) -> ComplexArray:
        ph = self._psih(qh)
        u = self._irfft(-self.il * ph)
        v = self._irfft(self.ik * ph)
        qx = self._irfft(self.ik * qh)
        qy = self._irfft(self.il * qh)
        adv = np.fft.rfft2(u * qx + v * qy) * self._dealias  # J(ψ, q)
        tend = -adv - self.ik * (self._ubg * qh + self._qy * ph)
        tend[1] += self.config.rek * self.wv2 * ph[1]  # -r_ek ∇²ψ_2
        return tend

    def step(self, n: int = 1) -> None:
        """Advance ``n`` time steps (AB3, spectral filter after each step)."""
        dt = self.config.dt
        for _ in range(n):
            tend = self._tendency(self._qh)
            h = self._tend_hist
            if len(h) == 0:
                incr = dt * tend
            elif len(h) == 1:
                incr = dt * (1.5 * tend - 0.5 * h[-1])
            else:
                incr = dt * (23 * tend - 16 * h[-1] + 5 * h[-2]) / 12
            h.append(tend)
            self._qh = self._filter * (self._qh + incr)
            self.t += dt
            self.n_steps += 1

    def spin_up(self, n_steps: int) -> None:
        """Run ``n_steps`` steps to reach the statistically stationary regime."""
        self.step(n_steps)

    def copy(self) -> QGModel:
        """Independent deep copy (state, time-stepping history and time)."""
        return _copy.deepcopy(self)


# --------------------------------------------------------------------- diagnostics
def _okubo_weiss(model: QGModel) -> FloatArray:
    ph = model._psih(model._qh)[0]
    psi_xx = model._irfft(-(model.kx**2) * ph)
    psi_yy = model._irfft(-(model.ky**2) * ph)
    psi_xy = model._irfft(-(model.kx * model.ky) * ph)
    s_n = -2 * psi_xy  # u_x - v_y
    s_s = psi_xx - psi_yy  # v_x + u_y
    zeta = psi_xx + psi_yy
    return np.asarray(s_n**2 + s_s**2 - zeta**2)


def count_regions(mask: NDArray[np.bool_], *, min_cells: int = VORTEX_MIN_CELLS) -> int:
    """Number of 4-connected components of ``mask`` (doubly periodic) with ≥ ``min_cells``."""
    ny, nx = mask.shape
    seen = np.zeros_like(mask)
    count = 0
    for j0, i0 in zip(*np.nonzero(mask), strict=True):
        if seen[j0, i0]:
            continue
        seen[j0, i0] = True
        stack = [(int(j0), int(i0))]
        size = 0
        while stack:
            j, i = stack.pop()
            size += 1
            for jj, ii in (
                ((j + 1) % ny, i),
                ((j - 1) % ny, i),
                (j, (i + 1) % nx),
                (j, (i - 1) % nx),
            ):
                if mask[jj, ii] and not seen[jj, ii]:
                    seen[jj, ii] = True
                    stack.append((jj, ii))
        count += size >= min_cells
    return count


def diagnostics(model: QGModel) -> dict[str, float]:
    """Physical ground-truth quantities of the current state (keys: ``DIAGNOSTIC_NAMES``).

    Energies and enstrophy are domain means weighted by the layer depth fractions
    ``H_k / H``. ``jet_speed_max`` is the maximum over y of the absolute zonal-mean eddy
    zonal wind of the upper layer. Vorticity extremes are for the upper layer, cyclonic
    meaning ``ζ > 0`` (northern hemisphere, f > 0); both are reported as positive numbers.
    ``n_vortices`` follows the Okubo-Weiss rule documented at ``OKUBO_WEISS_FACTOR``.
    """
    c = model.config
    h1, h2 = c.layer_weights
    w = np.array([h1, h2])
    psi = model.streamfunction()
    u, v = model.velocity()
    zeta = model.vorticity()
    q = model.state
    ke = 0.5 * float(np.sum(w * np.mean(u**2 + v**2, axis=(1, 2))))
    ape = 0.5 * c.F1 * h1 * float(np.mean((psi[0] - psi[1]) ** 2))
    ens = 0.5 * float(np.sum(w * np.mean(q**2, axis=(1, 2))))
    jet = float(np.max(np.abs(u[0].mean(axis=1))))
    ow = _okubo_weiss(model)
    n_vort = count_regions(ow < -OKUBO_WEISS_FACTOR * float(np.std(ow)))
    return {
        "kinetic_energy": ke,
        "available_potential_energy": ape,
        "enstrophy": ens,
        "jet_speed_max": jet,
        "max_cyclonic_vorticity": float(np.max(zeta[0])),
        "max_anticyclonic_vorticity": float(-np.min(zeta[0])),
        "n_vortices": float(n_vort),
    }


# --------------------------------------------------------------------- datasets
def _field(model: QGModel, field: Field) -> FloatArray:
    if field == "q":
        return model.state
    if field == "psi":
        return model.streamfunction()
    if field == "uv":
        u, v = model.velocity()
        return np.stack([u[0], v[0]])
    raise ValueError(f"unknown field {field!r}; expected 'q', 'psi' or 'uv'")


def simulate_with_diagnostics(
    config: QGConfig,
    *,
    n_trajectories: int,
    n_steps: int,
    sample_every: int,
    spin_up: int,
    rng: np.random.Generator,
    field: Field = "uv",
    on_snapshot: Callable[[int, int, QGModel], None] | None = None,
) -> tuple[NDArray[np.float32], FloatArray, tuple[str, ...]]:
    """Snapshots ``(N, T, 2, ny, nx)`` (float32), aligned diagnostics ``(N, T, Q)`` and names.

    Each trajectory starts from its own random initial condition (drawn from ``rng``), is
    spun up for ``spin_up`` steps, then run ``n_steps`` steps, recording one snapshot every
    ``sample_every`` steps: ``T = n_steps // sample_every``. ``field`` is ``"q"`` (PV
    anomaly of both layers), ``"psi"`` (streamfunction of both layers) or ``"uv"`` (upper
    layer eddy ``u`` and ``v``, the wind map; default). ``on_snapshot(i, t, model)``, if
    given, is called after snapshot ``t`` of trajectory ``i`` is recorded (it must not
    change ``model``).
    """
    if field not in ("q", "psi", "uv"):
        raise ValueError(f"unknown field {field!r}; expected 'q', 'psi' or 'uv'")
    if sample_every < 1 or n_steps < sample_every or n_trajectories < 1:
        raise ValueError("need n_trajectories >= 1 and n_steps >= sample_every >= 1")
    t_len = n_steps // sample_every
    n = config.nx
    snaps = np.empty((n_trajectories, t_len, 2, n, n), dtype=np.float32)
    diags = np.empty((n_trajectories, t_len, len(DIAGNOSTIC_NAMES)))
    for i in range(n_trajectories):
        model = QGModel(config, rng)
        model.spin_up(spin_up)
        for t in range(t_len):
            model.step(sample_every)
            snaps[i, t] = _field(model, field)
            d = diagnostics(model)
            diags[i, t] = [d[name] for name in DIAGNOSTIC_NAMES]
            if on_snapshot is not None:
                on_snapshot(i, t, model)
    return snaps, diags, DIAGNOSTIC_NAMES


def simulate_snapshots(
    config: QGConfig,
    *,
    n_trajectories: int,
    n_steps: int,
    sample_every: int,
    spin_up: int,
    rng: np.random.Generator,
    field: Field = "uv",
) -> NDArray[np.float32]:
    """Snapshots ``(N, T, 2, ny, nx)``; see :func:`simulate_with_diagnostics`."""
    return simulate_with_diagnostics(
        config,
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        sample_every=sample_every,
        spin_up=spin_up,
        rng=rng,
        field=field,
    )[0]


def snapshot_diagnostics(
    config: QGConfig,
    *,
    n_trajectories: int,
    n_steps: int,
    sample_every: int,
    spin_up: int,
    rng: np.random.Generator,
) -> tuple[FloatArray, tuple[str, ...]]:
    """Diagnostics ``(N, T, Q)`` and their names.

    Aligned with :func:`simulate_snapshots` called with the same arguments and an ``rng``
    in the same state (e.g. the same seed); use :func:`simulate_with_diagnostics` to get
    both in one pass.
    """
    _, diags, names = simulate_with_diagnostics(
        config,
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        sample_every=sample_every,
        spin_up=spin_up,
        rng=rng,
    )
    return diags, names


# --------------------------------------------------------------------- predictability
def _upper_velocity(model: QGModel) -> FloatArray:
    u, v = model.velocity()
    return np.stack([u[0], v[0]])


def twin_runs(
    model: QGModel,
    *,
    epsilon: float,
    n_samples: int,
    sample_every: int,
    rng: np.random.Generator,
) -> Iterator[tuple[QGModel, QGModel]]:
    """Twin runs ``(a, b)`` at ``n_samples + 1`` sample times (the first is the start).

    ``a`` restarts from the current state of ``model``; ``b`` from the same state with ``q``
    perturbed by Gaussian noise of std ``epsilon * RMS(q_k)`` in each layer (one draw from
    ``rng``). Both advance ``sample_every`` steps between samples; ``model`` itself is not
    modified, and the yielded models must not be modified by the caller.
    """
    a = model.copy()
    b = model.copy()
    q = a.state
    rms = np.sqrt(np.mean(q**2, axis=(1, 2), keepdims=True))
    b.state = q + epsilon * rms * rng.standard_normal(q.shape)
    a.state = q  # same restart (no AB history) for both twins
    for s in range(n_samples + 1):
        if s:
            a.step(sample_every)
            b.step(sample_every)
        yield a, b


def twin_divergence(
    model: QGModel,
    *,
    epsilon: float,
    n_samples: int,
    sample_every: int,
    rng: np.random.Generator,
) -> FloatArray:
    """RMS upper-layer velocity difference between twin runs, shape ``(n_samples + 1,)``.

    The twins are those of :func:`twin_runs`. ``model`` itself is not modified.
    """
    out = np.empty(n_samples + 1)
    twins = twin_runs(
        model, epsilon=epsilon, n_samples=n_samples, sample_every=sample_every, rng=rng
    )
    for s, (a, b) in enumerate(twins):
        d = _upper_velocity(a) - _upper_velocity(b)
        out[s] = np.sqrt(np.mean(np.sum(d**2, axis=0)))
    return out


def velocity_saturation(model: QGModel) -> float:
    """Expected RMS velocity difference of two unrelated states: ``sqrt(2) * RMS(|u_1|)``."""
    vel = _upper_velocity(model)
    return float(np.sqrt(2.0 * np.mean(np.sum(vel**2, axis=0))))


def true_growth_rate(
    model: QGModel,
    *,
    epsilon: float = 1e-6,
    n_samples: int = 200,
    sample_every: int = 20,
    rng: np.random.Generator,
) -> float:
    """Leading growth rate (per model time unit) of the true twin divergence.

    Log-slope of :func:`twin_divergence` over ``growth_window`` (step 1 up to the last
    sample before the separation, normalised by :func:`velocity_saturation`, reaches 0.3).
    Multiply by ``sample_every * config.dt`` for the rate per sample.
    """
    curve = twin_divergence(
        model, epsilon=epsilon, n_samples=n_samples, sample_every=sample_every, rng=rng
    )
    window = growth_window(normalised(curve, velocity_saturation(model)))
    return growth_rate(curve, dt=sample_every * model.config.dt, window=window)
