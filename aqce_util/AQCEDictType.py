from typing import List, TypedDict

import numpy as np
from jaxtyping import Complex128

from misc.util import AQCECounter


class MatrixCDict(TypedDict):
    i: int
    j: int
    U: Complex128[np.ndarray, "2 2"]


class AQCESaveDataDict(TypedDict):
    """Type definition for AQCE save data dictionary."""

    fidelity2target_list_list: List[
        tuple[int, int, List[float]]
    ]  # self.Cによる量子状態とtarget_stateとのFidelity
    matrixC_list: List[MatrixCDict]
    t_list: List[tuple[int, int, float]]
    target_state_list: List[
        Complex128[np.ndarray, "n_features"]
    ]  # Converted from QuantumState to vector
    unitary_fidelity_list: List[tuple[int, int, List[float]]]
    k: int
    max_fidelity: float
    time_list: List[float]
    today: str  # Format: "%Y-%m-%d-%H-%M-%S", e.g., '2024-06-18-14-40-24'
    noisy: bool
    n_shots: int | None
    coef_array_array: Complex128[np.ndarray, "k n_support_vectors"] | None
    data_list: (
        List[Complex128[np.ndarray, "n_features"]] | None
    )  # Converted from QuantumState to vector
    counter_list: List[AQCECounter]
