from typing import TypedDict

import numpy as np
from jaxtyping import Float64, Complex128

from eigenvalue_decompose_util.EigenvalueDecomposeConditionClass import EigenValueDecomposeCondition


class EigenValueDecomposeDataDict(TypedDict, total=False):
    """Type definition for eigenvalue decomposition data dictionary.

    Some fields are optional to accommodate different implementations.
    """

    # Required fields (present in both implementations)
    eigenvalues: Float64[np.ndarray, "n_support_vectors"]
    eigenvectors: Complex128[np.ndarray, "n_support_vectors n_features"]
    matrix: Complex128[np.ndarray, "n_support_vectors n_support_vectors"]

    # Optional fields (present only in the main implementation)
    compact_eigenvectors: Complex128[np.ndarray, "n_support_vectors n_support_vectors"]
    e_array: Complex128[np.ndarray, "n_support_vectors n_features"]
    coef_array: Complex128[np.ndarray, "n_support_vectors n_support_vectors"]
    psi_list: Complex128[np.ndarray, "n_support_vectors n_features"]
    alpha_array: Float64[np.ndarray, "n_support_vectors"]
    G: Complex128[np.ndarray, "n_support_vectors n_support_vectors"]
    eigen_coef_list: Complex128[np.ndarray, "n_support_vectors n_support_vectors"]
    condition: EigenValueDecomposeCondition
