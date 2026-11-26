"""Simulated systems with known answers, used to calibrate the diagnostics (protocol 3.1)."""

from weather_world_model.calibration.lorenz96 import (
    leading_lyapunov_lorenz96,
    lorenz96_rhs,
    lyapunov_spectrum_lorenz96,
    simulate_lorenz96,
    simulate_lorenz96_batch,
)
from weather_world_model.calibration.oscillators import (
    DEFAULT_EIGENVALUES,
    damped_linear_sequences,
    harmonic_oscillator_sequences,
)

__all__ = [
    "DEFAULT_EIGENVALUES",
    "damped_linear_sequences",
    "harmonic_oscillator_sequences",
    "leading_lyapunov_lorenz96",
    "lorenz96_rhs",
    "lyapunov_spectrum_lorenz96",
    "simulate_lorenz96",
    "simulate_lorenz96_batch",
]
