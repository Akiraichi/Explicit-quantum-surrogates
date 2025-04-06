from typing import Callable

import numpy as np
from jaxtyping import Complex128, Float64

from misc.util import calc_fidelity_numpy


def fidelity_kernel(
    x: Complex128[np.ndarray, "_"],
    y: Complex128[np.ndarray, "_"],
    noisy=False,
    n_shots=None,
) -> np.float64:
    """Fidelity kernel function."""
    return calc_fidelity_numpy(state1=x, state2=y, noisy=noisy, n_shots=n_shots)


def get_kernel_matrix(
    x1: Complex128[np.ndarray, "n_samples n_features"],
    x2: Complex128[np.ndarray, "n_samples n_features"],
    kernel: Callable[
        [Complex128[np.ndarray, "n_features"], Complex128[np.ndarray, "n_features"]],
        Float64[np.ndarray, ""],
    ],
) -> Float64[np.ndarray, "n_samples n_samples"]:
    """
    Get the kernel matrix using the specified kernel function.
    """
    if x1.shape[1] != x2.shape[1]:
        raise ValueError("Input data dimensions must match")
    else:
        # Note that kernel matrices can be calculated between different datasets, such as training and test data
        n_samples1 = x1.shape[0]
        n_samples2 = x2.shape[0]
        kernel_matrix = np.empty((n_samples1, n_samples2))
        for i in range(n_samples1):
            for j in range(n_samples2):
                kernel_matrix[i, j] = kernel(x1[i], x2[j])
        return kernel_matrix
