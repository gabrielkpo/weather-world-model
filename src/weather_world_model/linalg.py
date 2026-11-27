"""Small linear-algebra helpers shared by the diagnostics."""

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def pca(train: FloatArray, dims: int) -> tuple[FloatArray, FloatArray]:
    """Mean ``(d,)`` and the leading ``dims`` principal directions ``(d, dims)`` of ``train``.

    ``train`` is ``(..., d)``; every leading index is one sample. Project with
    ``(x - mean) @ basis``.
    """
    flat = train.reshape(-1, train.shape[-1])
    mean = flat.mean(axis=0)
    _, _, vt = np.linalg.svd(flat - mean, full_matrices=False)
    return mean, vt[:dims].T
