"""
Inner Product Calculation Module with JAX Support

This module provides functions for calculating inner products between quantum states
and their gradients using JAX for automatic differentiation. It implements custom
vector-Jacobian products (VJPs) for the inner product calculation to enable efficient
backpropagation through quantum circuits.

The module includes parallel processing capabilities to speed up calculations for
large datasets and supports both forward and backward passes through the quantum circuit.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from typing import TypedDict

import numpy as np
from jax import custom_vjp
from jax import numpy as jnp
from jaxlib.xla_extension import ArrayImpl
from jaxtyping import Complex128, Float64
from qulacs import QuantumState


class ResDict(TypedDict):
    """
    Type definition for the result dictionary of inner product calculations.

    This dictionary contains the circuit parameters and the results of inner product
    calculations between quantum states, including real part, imaginary part, and
    absolute value.
    """

    # When you want to treat theta_jax as jax.Array
    theta_jax: Float64[jnp.ndarray, "len_theta"]

    # When you want to treat all of the following as np.ndarray
    real_inner_product_vector: Float64[jnp.ndarray, "n_samples K"]
    imag_inner_product_vector: Float64[jnp.ndarray, "n_samples K"]
    abs_inner_product_vector: Float64[jnp.ndarray, "n_samples K"]


from misc.util import debug_print
from qnn_util.cost_func_util import (
    translate_qulacs_quantum_state,
    update_circuit_from_k_induced_params,
    util_calc_inner_product,
)
from qnn_util.LearningCircuitClass import MyLearningCircuit



def _abs_inner_product_fn(
    theta_jax: Float64[jnp.ndarray, "len_theta"],
    U: MyLearningCircuit,
    setting_dict: dict,
) -> Float64[jnp.ndarray, "n_samples K"]:
    """
    Calculate the absolute value of inner products between quantum states.

    This function computes the inner products between quantum states and circuit outputs,
    then takes the absolute value of each inner product. It's the core function that
    will be wrapped with custom differentiation rules.

    Args:
        theta_jax: Circuit parameters
        U: Quantum circuit
        setting_dict: Dictionary containing calculation settings

    Returns:
        Array of absolute inner product values with shape (n_samples, K)
    """
    _t = time.time()
    # Calculate inner products
    inner_product_vector = parallelize_calculate_inner_product(
        theta_jax=theta_jax, U=U, setting_dict=setting_dict
    )

    # Apply abs to all elements
    abs_inner_product_vector = jnp.abs(inner_product_vector)

    if setting_dict["debug_print"]:
        debug_print(
            f"fwd: Calculation time for inner_product_fn: {time.time() - _t}",
            debug_print=setting_dict["debug_print"],
        )

    return abs_inner_product_vector


abs_inner_product_fn = custom_vjp(
    _abs_inner_product_fn, nondiff_argnums=(1, 2)
)  # ここでラップ


def parallelize_calculate_inner_product(
    theta_jax: Float64[jnp.ndarray, "len_theta"],
    U: MyLearningCircuit,
    setting_dict: dict,
) -> Complex128[jnp.ndarray, "n_samples K"]:
    """
    Calculate inner products between quantum states and circuit outputs with parallel processing.

    This function computes the inner products between input quantum states and the outputs
    of a parameterized quantum circuit. It can use parallel processing to speed up the
    calculation for large datasets.

    Args:
        theta_jax: Circuit parameters
        U: Quantum circuit
        setting_dict: Dictionary containing calculation settings including input data

    Returns:
        2D JAX array of inner product values with shape (n_samples, K)
    """
    inner_product_ndarray = np.zeros(
        (setting_dict["n_samples"], setting_dict["K"]), dtype=np.complex128
    )
    # inner_product_vector = [[0] * setting_dict["K"] for _ in range(setting_dict["n_samples"])]

    if setting_dict["fwd_parallel"]:
        max_workers = setting_dict["n_jobs"]
        chunksize = (
            (len(setting_dict["x_list"]) * setting_dict["K"]) // max_workers
        ) + 1

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            args = [
                {"index_x": index_x, "k": k, "x": x, "theta_jax": theta_jax, "U": U}
                for k in range(setting_dict["K"])
                for index_x, x in enumerate(setting_dict["x_list"])
            ]
            results = executor.map(compute_inner_product, args, chunksize=chunksize)

            for index_x, k, _inner_value in results:
                inner_product_ndarray[index_x, k] = _inner_value

    else:
        for k in range(setting_dict["K"]):
            # Create a circuit for each k. Act directly on U.
            U = update_circuit_from_k_induced_params(theta_jax=theta_jax, U=U, k=k)

            for index_x, x in enumerate(setting_dict["x_list"]):
                # Calculate inner product for each data point
                inner_value: complex = util_calc_inner_product(
                    theta_jax=theta_jax, U=U, k=k, x=x
                )
                inner_product_ndarray[index_x, k] = inner_value
    return jnp.array(inner_product_ndarray)


def compute_inner_product(args: dict) -> tuple[int, int, complex]:
    """
    Compute inner product for a single data point and circuit configuration.

    This function is designed to be used with parallel processing. It takes a dictionary
    of arguments and calculates the inner product between a quantum state and a circuit output.

    Args:
        args: Dictionary containing the following keys:
            - theta_jax: Circuit parameters
            - U: Quantum circuit
            - k: Index for computational basis state
            - x: Input quantum state
            - index_x: Index of the data point

    Returns:
        Tuple containing (index_x, k, inner_product_value)
    """
    inner_value = util_calc_inner_product(
        theta_jax=args["theta_jax"], U=args["U"], k=args["k"], x=args["x"]
    )

    return args["index_x"], args["k"], inner_value


def abs_inner_product_fn_fwd(
    theta_jax: Float64[jnp.ndarray, "len_theta"],
    U: MyLearningCircuit,
    setting_dict: dict,
) -> tuple[Float64[jnp.ndarray, "n_samples K"], ResDict]:
    """
    Forward pass for custom vector-Jacobian product (VJP) of abs_inner_product_fn.

    This function implements the forward pass for the custom differentiation rule
    of the absolute inner product function. It calculates the inner products and
    stores intermediate values needed for the backward pass.

    Args:
        theta_jax: Circuit parameters
        U: Quantum circuit
        setting_dict: Dictionary containing calculation settings

    Returns:
        Tuple containing:
        - Array of absolute inner product values
        - Dictionary of intermediate values for the backward pass
    """
    inner_product_vector = parallelize_calculate_inner_product(
        theta_jax=theta_jax, U=U, setting_dict=setting_dict
    )
    abs_inner_product_vector = jnp.abs(inner_product_vector)

    res: ResDict = {
        "theta_jax": theta_jax,
        "real_inner_product_vector": jnp.real(inner_product_vector),
        "imag_inner_product_vector": jnp.imag(inner_product_vector),
        "abs_inner_product_vector": abs_inner_product_vector,
    }

    return abs_inner_product_vector, res


def abs_inner_product_fn_bwd(
    U: MyLearningCircuit, setting_dict: dict, res: ResDict, g: ArrayImpl
) -> tuple[Float64[jnp.ndarray, "len_theta"]]:
    """
    Backward pass for custom vector-Jacobian product (VJP) of abs_inner_product_fn.

    This function implements the backward pass for the custom differentiation rule
    of the absolute inner product function. It calculates the gradients with respect
    to the circuit parameters using the intermediate values from the forward pass.

    Args:
        U: Quantum circuit
        setting_dict: Dictionary containing calculation settings
        res: Dictionary of intermediate values from the forward pass
        g: Upstream gradient

    Returns:
        Tuple containing the gradient with respect to theta_jax
    """
    _t_bwd = time.time()

    theta_jax = res["theta_jax"]
    abs_inner_product_vector = res["abs_inner_product_vector"]
    real_inner_product_vector = res["real_inner_product_vector"]
    imag_inner_product_vector = res["imag_inner_product_vector"]

    gradient_ndarray = np.zeros(
        shape=(
            setting_dict["n_samples"],
            setting_dict["K"],
            setting_dict["len_theta"],
        ),
        dtype=np.float64,
    )

    if setting_dict["bwd_parallel"]:
        max_workers = setting_dict["n_jobs"]
        chunksize = (len(setting_dict["x_list"]) // max_workers) + 1

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            args = [
                {
                    "index_x": index_x,
                    "k": k,
                    "x": x,
                    "theta_jax": theta_jax,
                    "U": U,
                    "abs_inner_product": abs_inner_product_vector[index_x, k],
                    "real_inner_product": real_inner_product_vector[index_x, k],
                    "imag_inner_product": imag_inner_product_vector[index_x, k],
                    "g": g,
                }
                for k in range(setting_dict["K"])
                for index_x, x in enumerate(setting_dict["x_list"])
            ]
            results = executor.map(compute_gradient, args, chunksize=chunksize)

            for index_x, k, grad_params in results:
                gradient_ndarray[index_x, k] = grad_params
    else:
        for k in range(setting_dict["K"]):
            for index_x, x in enumerate(setting_dict["x_list"]):
                grad_params, U = calculate_gradient_for_single_pair(
                    k=k,
                    x=x,
                    index_x=index_x,
                    theta_jax=theta_jax,
                    U=U,
                    abs_inner_product=abs_inner_product_vector[index_x, k],
                    real_inner_product=real_inner_product_vector[index_x, k],
                    imag_inner_product=imag_inner_product_vector[index_x, k],
                    g=g,
                )
                # print(f"grad_params.shape]: {grad_params.shape}")
                gradient_ndarray[index_x, k] = grad_params

    if setting_dict["debug_print"]:
        debug_print(
            f"Calculation time for bwd: {time.time() - _t_bwd}",
            debug_print=setting_dict["debug_print"],
        )
    theta_gradient_jax = gradient_ndarray.sum(axis=(0, 1))
    return (jnp.array(theta_gradient_jax),)


def calculate_gradient_for_single_pair(
    k: int,
    x: Complex128[np.ndarray, "n_features"],
    index_x: int,
    theta_jax: Float64[jnp.ndarray, "len_theta"],
    U: MyLearningCircuit,
    abs_inner_product: Float64[jnp.ndarray, ""],
    real_inner_product: Float64[jnp.ndarray, ""],
    imag_inner_product: Float64[jnp.ndarray, ""],
    g: ArrayImpl,
) -> tuple[Float64[jnp.ndarray, "len_theta"], MyLearningCircuit]:
    """
    Calculate gradient for a single (k, x) pair.

    This function computes the gradient of the absolute inner product with respect
    to the circuit parameters for a single data point and computational basis state.

    Args:
        k: Index k for computational basis state |k>
        x: Data point (quantum state)
        index_x: Index of the data point
        theta_jax: Circuit parameters
        U: Quantum circuit
        abs_inner_product: Absolute value of the inner product
        real_inner_product: Real part of the inner product
        imag_inner_product: Imaginary part of the inner product
        g: Upstream gradient

    Returns:
        Tuple containing:
        - Gradient parameters
        - Updated quantum circuit
    """
    # Update circuit for specific k
    U = update_circuit_from_k_induced_params(theta_jax=theta_jax, U=U, k=k)

    # Load quantum state
    psi_bra = translate_qulacs_quantum_state(
        x=x, n_qubits=U._circuit.get_qubit_count(), conj=True
    )

    # Calculate gradient
    grad_abs = __calculate_one_gradient_for_abs_inner_product(
        abs_inner_product=abs_inner_product,
        real_inner_product=real_inner_product,
        imag_inner_product=imag_inner_product,
        theta_jax=theta_jax,
        state_bra=psi_bra,
        U=U,
    )  # (len_theta)
    # print("grad_abs.shape", grad_abs.shape)
    # print(type(g))
    # print(g.shape)  # (n_samples, K)

    # Apply gradient
    # g_expanded = g[
    #     :, :, None
    # ]  # (n_samples, K) -> (n_samples, K, 1) # TODO Make corrections here
    g_target = g[index_x, k]
    grad_params = grad_abs * g_target

    return grad_params, U


def compute_gradient(args: dict) -> tuple[int, int, Float64[jnp.ndarray, "len_theta"]]:
    """
    Compute gradient for a single data point and circuit configuration.

    This function is designed to be used with parallel processing. It takes a dictionary
    of arguments and calculates the gradient of the absolute inner product with respect
    to the circuit parameters for a single data point and computational basis state.

    Args:
        args: Dictionary containing the following keys:
            - index_x: Index of the data point
            - k: Index for computational basis state
            - x: Data point (quantum state)
            - theta_jax: Circuit parameters
            - U: Quantum circuit
            - abs_inner_product: Absolute value of the inner product
            - real_inner_product: Real part of the inner product
            - imag_inner_product: Imaginary part of the inner product
            - g: Upstream gradient

    Returns:
        Tuple containing (index_x, k, gradient_parameters)
    """
    index_x: int = args["index_x"]
    k: int = args["k"]

    grad_params, U = calculate_gradient_for_single_pair(
        k=k,
        x=args["x"],
        theta_jax=args["theta_jax"],
        U=args["U"],
        abs_inner_product=args["abs_inner_product"],
        real_inner_product=args["real_inner_product"],
        imag_inner_product=args["imag_inner_product"],
        g=args["g"],
        index_x=args["index_x"],
    )

    return index_x, k, grad_params


def __calculate_one_gradient_for_abs_inner_product(
    abs_inner_product: Float64[jnp.ndarray, ""],  # 1-dimensional vector
    real_inner_product: Float64[jnp.ndarray, ""],
    imag_inner_product: Float64[jnp.ndarray, ""],
    theta_jax: Float64[jnp.ndarray, "len_theta"],
    state_bra: QuantumState,
    U: MyLearningCircuit,
) -> Float64[jnp.ndarray, "len_theta"]:
    """
    Calculate the gradient of the absolute inner product with respect to circuit parameters.

    This function computes the gradient of |<x|U|k>| with respect to the circuit parameters
    by calculating the gradients of the real and imaginary parts separately and then
    combining them according to the chain rule.

    Args:
        abs_inner_product: Absolute value of the inner product
        real_inner_product: Real part of the inner product
        imag_inner_product: Imaginary part of the inner product
        theta_jax: Circuit parameters
        state_bra: Quantum state <x|
        U: Quantum circuit

    Returns:
        Gradient of the absolute inner product with respect to circuit parameters
    """
    if abs_inner_product < 1e-14:
        # In extremely small cases, the derivative becomes unstable, so return zero
        return jnp.zeros_like(theta_jax)

    # Gradient calculation for the real component of the inner product
    state_bra_copy = state_bra.copy()
    grad_real_list = U.backprop_inner_product(
        state_bra_copy, exclude_first_k=True
    )  # Note that backprop_inner_product performs destructive operations on state_bra. This is probably a bug in backprop. Also, it returns the real component of the inner product, which is confusing.
    grad_real: Float64[jnp.ndarray, "len_theta"] = jnp.array(
        grad_real_list, dtype=jnp.float64
    )

    # Calculate imaginary part gradient
    state_bra_imag_copy = state_bra.copy()
    state_bra_imag_copy.multiply_coef(1.0j)
    grad_imag_list = U.backprop_inner_product(state_bra_imag_copy, exclude_first_k=True)

    grad_imag: Float64[jnp.ndarray, "len_theta"] = jnp.array(
        grad_imag_list, dtype=jnp.float64
    )

    # 4. Synthesis: d|z|/dθ_j = [r * dr/dθ_j + i_ * di/dθ_j] / |z|
    grad_abs: Float64[jnp.ndarray, "len_theta"] = (
        real_inner_product * grad_real + imag_inner_product * grad_imag
    ) / abs_inner_product
    return grad_abs


abs_inner_product_fn.defvjp(abs_inner_product_fn_fwd, abs_inner_product_fn_bwd)
