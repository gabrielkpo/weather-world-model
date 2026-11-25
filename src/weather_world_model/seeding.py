"""Reproducibility helpers."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and PyTorch, and return a NumPy generator for explicit use."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)
