import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import joblib  # type: ignore
import numpy as np
from jaxtyping import Complex128, Int64
from qulacs import QuantumState
from qulacs.state import inner_product
from qulacs_core import QuantumGateMatrix  # type: ignore


def data_shuffle(
    X: Complex128[np.ndarray, "n_samples n_features"],
    y: Int64[np.ndarray, "n_samples"],
    seed:int | None=None,
) -> Tuple[
    Complex128[np.ndarray, "n_samples n_features"], Int64[np.ndarray, "n_samples"]
]:
    """
    Shuffle a dataset.

    Args:
        X: Feature data
        y: Label data
        seed: Random seed for reproducibility

    Returns:
        Tuple of shuffled (X, y)
    """
    combined = list(zip(X, y))
    random.shuffle(
        combined
    )  # TODO: Note that seed value is not specified for this shuffle. Fix this.
    X_, y_ = zip(*combined)  # Extract shuffled X and y

    X_shuffled = np.array(X_, dtype=np.complex128)
    y_shuffled = np.array(y_, dtype=np.int64)
    return X_shuffled, y_shuffled


def calc_fidelity_numpy(
    state1: Complex128[np.ndarray, "n_features"],
    state2: Complex128[np.ndarray, "n_features"],
    noisy: bool = False,
    n_shots: Optional[int] = None,
) -> np.float64:
    """Calculate the fidelity between two quantum states."""
    inner_product_value = np.vdot(state1, state2)
    fidelity = np.abs(inner_product_value) ** 2

    if noisy:
        assert n_shots is not None, "n_shots is None"
        p0 = fidelity
        n0 = np.random.binomial(n_shots, p0)
        fidelity = n0 / n_shots
    return fidelity


def timer_decorator(func):
    """Function to measure the execution time of a function.

    Use as a decorator with @timer_decorator.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.5f} seconds to run.")
        return result

    return wrapper


def calc_fidelity_qulacs(
    target_state: QuantumState,
    C: List[QuantumGateMatrix],
    i: int,
    noisy: bool = False,
    n_shots: Optional[int] = None,
) -> float:
    """Calculate the fidelity between the quantum state C|i> and target_state."""
    n_qubits = target_state.get_qubit_count()
    state = QuantumState(n_qubits)
    state.set_computational_basis(i)  # Initialize to |i>
    # Calculate state = C|i>
    for k in range(len(C)):
        C[k].update_quantum_state(state)
    fidelity = np.abs(inner_product(state, target_state)) ** 2

    if noisy:
        assert n_shots is not None, "n_shots is None"

        p0 = fidelity
        n0 = np.random.binomial(n_shots, p0)
        fidelity = n0 / n_shots

    return fidelity


def list2str(input_list: list) -> str:
    """Convert a list like [1, 2, 3, 4] to a string like '1_2_3_4'."""
    input_txt = str(input_list)
    return input_txt.replace(", ", "_").replace("[", "").replace("]", "")


def my_clip(
    value: np.float64, min_value: np.float64, max_value: np.float64
) -> np.float64:
    """Faster alternative to np.clip when applied to scalar values."""
    if value <= min_value:
        return min_value
    elif value >= max_value:
        return max_value
    else:
        return value


def inner_product_exp(
    vec1: Complex128[np.ndarray, "n_features"],
    vec2: Complex128[np.ndarray, "n_features"],
    noisy: bool,
    n_shots: int | None,
) -> np.complex128:
    """
    Function to calculate the inner product between two quantum states.

    The inner product is defined as:
    $$
    \langle \text{vec1}|\text{vec2}\rangle
    =\sum_i \overline{(\text{vec1}_i)}\,\text{vec2}_i
    $$
    This function calculates the inner product between two quantum states $|\psi\rangle$.

    Args:
      - vec1, vec2: Normalized quantum states represented as complex vectors (numpy arrays).
      - noisy: If True, returns an estimated inner product with statistical errors simulating actual measurements.
      - n_shots: Number of measurement shots. More shots result in smaller statistical errors.

    Simulation approach:
    On actual quantum hardware, the real and imaginary parts of the inner product can be estimated using methods like the Hadamard test.
    Here, we estimate them as follows:

    - For the real part:
      The probability of measuring 0 is
      $$
      p_{0,\mathrm{real}}=\frac{1+\Re\langle \psi_1|\psi_2\rangle}{2}
      $$
      So the estimated real part is
      $$
      \Re\langle \psi_1|\psi_2\rangle \approx 2\frac{n_{0,\mathrm{real}}}{\text{shots}} - 1
      $$

    - For the imaginary part:
      The probability of measuring 0 is
      $$
      p_{0,\mathrm{imag}}=\frac{1-\Im\langle \psi_1|\psi_2\rangle}{2}
      $$
      So the estimated imaginary part is
      $$
      \Im\langle \psi_1|\psi_2\rangle \approx 1-2\frac{n_{0,\mathrm{imag}}}{\text{shots}}
      $$

    This approach reflects the statistical errors associated with actual quantum measurements.
    """
    # Ideal inner product (true value for simulation)
    ip_exact = np.vdot(vec1, vec2)

    if not noisy:
        return ip_exact
    else:
        assert n_shots is not None, "n_shots is None"
        """Approximation using normal distribution. Thought it would be faster, but it wasn't."""
        # # Estimation of the real part
        # p0_real = (1 + ip_exact.real) / 2  # Probability of measuring 0 (real part)
        # p0_real = my_clip(p0_real,min_value=0,max_value=1)
        # mean_real = n_shots * p0_real
        # std_real = np.sqrt(n_shots * p0_real * (1 - p0_real))
        # # Sample from normal distribution (rounding or integer conversion is not required, but keep within the range of measurement counts)
        # n0_real = np.random.normal(loc=mean_real, scale=std_real)
        # n0_real = my_clip(n0_real,min_value=0,max_value=n_shots)
        #
        # re_est = 2 * (n0_real / n_shots) - 1  # Estimated real part
        #
        # # Estimation of the imaginary part
        # p0_imag = (1 + ip_exact.imag) / 2  # Probability of measuring 0 (imaginary part)
        # p0_imag = my_clip(p0_imag,min_value=0,max_value=1)
        # mean_imag = n_shots * p0_imag
        # std_imag = np.sqrt(n_shots * p0_imag * (1 - p0_imag))
        # n0_imag = np.random.normal(loc=mean_imag, scale=std_imag)
        # n0_imag = my_clip(n0_imag,min_value=0,max_value=n_shots)
        # im_est = 2 * (n0_imag / n_shots) - 1  # Estimated imaginary part

        """Using binomial distribution"""
        # Estimation of the real part
        p0_real = (1 + ip_exact.real) / 2  # Probability of measuring 0 (real part)
        p0_real = my_clip(
            np.float64(p0_real), min_value=np.float64(0), max_value=np.float64(1)
        )
        n0_real = np.random.binomial(n_shots, p0_real)
        re_est = 2 * (n0_real / n_shots) - 1  # Estimated real part

        # Estimation of the imaginary part
        p0_imag = (1 + ip_exact.imag) / 2  # Probability of measuring 0 (imaginary part)
        p0_imag = my_clip(
            np.float64(p0_imag), min_value=np.float64(0), max_value=np.float64(1)
        )
        n0_imag = np.random.binomial(n_shots, p0_imag)
        im_est = 2 * (n0_imag / n_shots) - 1  # Estimated imaginary part

        return np.complex128(re_est + 1j * im_est)


@dataclass
class AQCECounter:
    # Ideally this would be in AQCEutil, but it's placed here for compatibility reasons
    n_sweeps: int = 0  # Total number of sweeps at the current point
    n_hadamard_test: int = 0  # Total number of Hadamard tests executed at the current point
    n_gates: int = 0  # Number of gates in the circuit being constructed by AQCE
    n_qubits: int = 0  # Number of qubits in the circuit being constructed by AQCE

    fidelity_list: list = field(default_factory=list)


def debug_print(message: str, debug_print: bool):
    if debug_print:
        print(message)


def calc_fidelity_between_state_and_k_state(
    state: Complex128[np.ndarray, "n_features"],
    k: int,
    K: int,
    noisy: bool,
    n_shots: int | None,
) -> float:
    """Function to calculate |<\psi|k>|^2. k is in the range [1, K]."""
    if noisy:
        assert n_shots is not None, "n_shots is None"

        # Calculate theoretical probabilities for each basis state
        probs: NDArray[np.float64] = np.abs(state) ** 2  # type: ignore
        probs.clip(min=0.0, max=1.0, out=probs)

        if len(probs) > K:
            # For computational efficiency, handle the first K items individually and group the rest together
            grouped_probs = np.zeros(K + 1, dtype=np.float64)
            grouped_probs[:K] = probs[:K]
            grouped_probs[K] = np.sum(probs[K:])
        else:
            # If the dimension of the state is less than or equal to K, use it as is
            grouped_probs = probs.astype(np.float64)

        # Normalize grouped_probs (to handle rounding errors)
        grouped_probs = grouped_probs / np.sum(grouped_probs)
        # Get the counts of n_shots measurement results at once using a multinomial distribution
        counts = np.random.multinomial(n_shots, grouped_probs)
        return counts[k] / n_shots
    else:
        return np.abs(state[k]) ** 2


def update_state_from_dagger_gate_matrix_list(
    state: QuantumState, C_dagger: list[QuantumGateMatrix]
) -> QuantumState:
    state_copy: QuantumState = state.copy()

    for c_dagger in reversed(C_dagger):
        c_dagger.update_quantum_state(state_copy)
    return state_copy


def repeat_save_until_success(data, path, retry_count=10):
    """Before refactoring, there were cases where saving failed because hundreds of files were being saved to the same folder.
    Therefore, this function checks if the save was successful and retries up to 10 times if it failed.
    """
    for _ in range(retry_count):
        try:
            # Save the data
            joblib.dump(data, path)
            # Load the data immediately after saving
            loaded_data = joblib.load(path)
            # Return True if the data was loaded successfully
            return True
        except Exception as e:
            print(f"Error during save/load attempt: {e}")
        # Return False if all retry attempts fail
    return False
