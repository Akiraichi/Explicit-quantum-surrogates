import numpy as np
from jax import numpy as jnp
from jaxtyping import Complex128, Float64, Int
from qulacs import QuantumState
from qulacs.state import inner_product

from qnn_util.LearningCircuitClass import MyLearningCircuit
from qnn_util.helper import FirstKHelper


def calculate_cross_entropy_loss(
        y_array: Int[jnp.ndarray, "1 n_samples"],
        probs: Float64[jnp.ndarray, "n_samples"],
        w0: Float64[jnp.ndarray, ""],
        w1: Float64[jnp.ndarray, ""],
        balanced: bool,
) -> Float64[jnp.ndarray, ""]:
    """
    Calculate cross-entropy loss.

    Args:
        y_array: Array of target labels (0 or 1)
        probs: Predicted probabilities
        w0: Weight for class 0
        w1: Weight for class 1
        balanced: Boolean indicating whether to use class weights for balanced loss

    Returns:
        Loss value as a JAX scalar (0-dimensional array)
    """
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10
    safe_probs = jnp.clip(probs, epsilon, 1.0 - epsilon)

    if balanced:
        # Apply class weights to balance the loss
        return -jnp.mean(
            y_array * jnp.log(safe_probs) * w1
            + (1 - y_array) * jnp.log(1 - safe_probs) * w0
        )
    else:
        # Standard binary cross-entropy
        return -jnp.mean(
            y_array * jnp.log(safe_probs) + (1 - y_array) * jnp.log(1 - safe_probs)
        )


def calculate_hinge_loss(
        y_array: Int[jnp.ndarray, "1 n_samples"],
        z: Float64[jnp.ndarray, "n_samples"],
        w0: Float64[jnp.ndarray, ""],
        w1: Float64[jnp.ndarray, ""],
        balanced: bool,
) -> Float64[jnp.ndarray, ""]:
    """
    Calculate hinge loss.

    Args:
        y_array: Array of target labels (0 or 1)
        z: Values of the decision function
        w0: Weight for class 0
        w1: Weight for class 1
        balanced: Boolean indicating whether to use class weights for balanced loss

    Returns:
        Loss value as a JAX scalar (0-dimensional array)
    """
    # Convert labels from {0,1} to {-1,1}
    relabeled_y_array = jnp.where(
        y_array == 0, -1, 1
    )  # If label is 0, set to -1; otherwise set to 1

    # Calculate hinge loss: max(0, 1 - y*z)
    loss = jnp.maximum(0, 1 - z * relabeled_y_array)

    if balanced:
        # Apply class weights to balance the loss
        weights = jnp.where(relabeled_y_array == -1, w0, w1)
        return jnp.mean(loss * weights)
    else:
        # Standard hinge loss
        return jnp.mean(loss)


def sigmoid(z: Float64[jnp.ndarray, "n_samples"]) -> Float64[jnp.ndarray, "n_samples"]:
    """
    Apply the sigmoid function element-wise to the input array.

    Args:
        z: Input array

    Returns:
        Array with sigmoid function applied element-wise
    """
    return 1 / (1 + jnp.exp(-1 * z))


def get_state_from_parametric_circuit(
        theta_jax: Float64[jnp.ndarray, "n_theta"], U: MyLearningCircuit, k: int
) -> tuple[QuantumState, MyLearningCircuit]:
    """
    Get a quantum state from a parametric circuit with the given parameters.

    Args:
        theta_jax: Circuit parameters
        U: Quantum circuit
        k: Index of the computational basis state to prepare

    Returns:
        Tuple containing:
        - The quantum state after applying the circuit to |0>
        - The updated circuit
    """
    n_qubits = U._circuit.get_qubit_count()

    assert theta_jax.shape[0] == len(
        U.get_parameters(exclude_first_k=True)
    ), "Number of parameters does not match"

    # Update circuit parameters. Adding X gates to change state_ket to |k>
    U = update_circuit_from_k_induced_params(theta_jax=theta_jax, U=U, k=k)

    # Create initial state
    state_ket = QuantumState(qubit_count=n_qubits)
    state_ket.set_zero_state()
    U._circuit.update_quantum_state(state_ket)
    return state_ket, U


def translate_qulacs_quantum_state(
        x: Complex128[np.ndarray, "n_features"], n_qubits: int, conj: bool
) -> QuantumState:
    """
    Convert a numpy array to a qulacs quantum state.

    Args:
        x: Input array representing a quantum state
        n_qubits: Number of qubits in the quantum state
        conj: If True, take the complex conjugate of the input (for bra states)

    Returns:
        Qulacs QuantumState object
    """
    # Load input data into a quantum state
    if conj:
        psi_bra = QuantumState(qubit_count=n_qubits)
        # Take the conjugate for bra states
        x_conj = np.conj(x)
        psi_bra.load(x_conj)
        return psi_bra
    else:
        psi_ket = QuantumState(qubit_count=n_qubits)
        psi_ket.load(x.tolist())
        return psi_ket


def _generate_k_params(k: int) -> list[float]:
    """
    Generate parameters for a circuit that prepares the computational basis state |k>.

    This function converts the integer k to its binary representation and sets
    appropriate rotation parameters to prepare the state |k> from |0>.

    Args:
        k: Index of the computational basis state to prepare

    Returns:
        List of rotation parameters for the circuit
    """
    k_params = [0.0] * FirstKHelper.param_count
    k_bin = str(bin(k))[2:]

    for __i, __k in enumerate(reversed(k_bin)):
        if __k == "1":
            k_params[__i * 3] = -np.pi / 2  # RZ
            k_params[__i * 3 + 1] = np.pi  # RX
            k_params[__i * 3 + 2] = np.pi / 2  # RZ

    return k_params


def update_circuit_from_k_induced_params(
        theta_jax: Float64[jnp.ndarray, "len_theta"], U: MyLearningCircuit, k: int
) -> MyLearningCircuit:
    """
    Construct U(θ')|0> that reproduces U(θ)|k>.

    This means adding parameters that effectively add X gates at the beginning of the circuit.
    The circuit U is assumed to already have RX gates added.

    Args:
        theta_jax: Circuit parameters
        U: Quantum circuit
        k: Index of the computational basis state to prepare

    Returns:
        Updated quantum circuit
    """
    k_params = _generate_k_params(k)
    U.update_parameters(k_params + theta_jax.tolist())
    return U


def util_calc_inner_product(
        theta_jax: Float64[jnp.ndarray, "n_theta"],
        U: MyLearningCircuit,
        k: int,
        x: Complex128[np.ndarray, "n_features"],
) -> complex:
    """
    Calculate the inner product <x|U|k>.

    Args:
        theta_jax: Circuit parameters
        U: Quantum circuit
        k: Index of the computational basis state
        x: Input quantum state vector

    Returns:
        Complex value of the inner product
    """
    # Load input data into a quantum state. Take conjugate for bra state
    psi_bra = translate_qulacs_quantum_state(
        x=x, n_qubits=U._circuit.get_qubit_count(), conj=True
    )
    # Update circuit parameters and get the quantum state by applying the updated circuit to |k>
    state_ket, U = get_state_from_parametric_circuit(theta_jax=theta_jax, U=U, k=k)
    # Calculate inner product
    inner_value = inner_product(
        psi_bra, state_ket
    )  # Use qulacs.state.inner_product to calculate the inner product
    return inner_value
