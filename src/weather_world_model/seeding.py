"""Reproducibility helpers."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python's ``random`` and PyTorch's global generators; return a NumPy generator.

    NumPy's global generator (``np.random.seed``) is NOT seeded: code that needs NumPy
    randomness must draw from the returned ``np.random.default_rng(seed)``.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)
