from jax import numpy as jnp
from jaxtyping import Float64
from jax import config

config.update("jax_enable_x64", True)

from qnn_util.cost_func_util import (
    calculate_cross_entropy_loss,
    calculate_hinge_loss,
    sigmoid,
)
from misc.util import debug_print
from qnn_util.inner_product_apply_jax import abs_inner_product_fn
from qnn_util.LearningCircuitClass import MyLearningCircuit


def loss_fn(
    theta_jax: Float64[jnp.ndarray, "len_theta"], U: MyLearningCircuit, setting_dict: dict
) -> Float64[jnp.ndarray, ""]:
    """
    Calculate cross-entropy or hinge loss for a given set of parameters.

    Args:
        theta_jax: List of circuit parameters
        U: Quantum circuit
        setting_dict: Dictionary containing calculation settings:
            - y_list: List of target labels (0 or 1)
            - use_log_loss: Boolean indicating whether to use log loss (cross-entropy)
            - use_hinge_loss: Boolean indicating whether to use hinge loss
            - balanced: Boolean indicating whether to use class weights for balanced loss
            - debug_print: Boolean indicating whether to display debug information

    Returns:
        Loss value as a JAX scalar (0-dimensional array)
    """
    # Reshape y_list to (1, n) for broadcasting
    y_array = jnp.array(setting_dict["y_list"]).reshape(1, -1)
    # Get the values of the decision function
    z = svm_decision_fn(theta_jax=theta_jax, U=U, setting_dict=setting_dict)

    probs = sigmoid(z)
    # Number of data points for each label
    n_target = y_array.sum()  # Only target_label is set to 1
    n0 = y_array.shape[1] - n_target

    # Calculate weights
    w0 = n_target / (n0 + n_target)
    w1 = n0 / (n0 + n_target)
    if setting_dict["debug_print"]:
        debug_print(
            f"n_target: {n_target}, n0: {n0}", debug_print=setting_dict["debug_print"]
        )
        debug_print(f"w0: {w0}, w1: {w1}", debug_print=setting_dict["debug_print"])
        debug_print(
            f'use_log_loss: {setting_dict["use_log_loss"]}',
            debug_print=setting_dict["debug_print"],
        )
        debug_print(
            f'use_hinge_loss: {setting_dict["use_hinge_loss"]}',
            debug_print=setting_dict["debug_print"],
        )

    if setting_dict["use_log_loss"]:
        return calculate_cross_entropy_loss(
            y_array, probs, w0, w1, setting_dict["balanced"]
        )
    elif setting_dict["use_hinge_loss"]:
        return calculate_hinge_loss(y_array, z, w0, w1, setting_dict["balanced"])
    else:
        cost_func_str = setting_dict.get("cost_func_str", "unknown")
        raise RuntimeError(
            f"Invalid cost function name: {cost_func_str}. Please specify 'use_log_loss' or 'use_hinge_loss'."
        )


def svm_decision_fn(
    theta_jax: Float64[jnp.ndarray, "len_theta"], U: MyLearningCircuit, setting_dict: dict
) -> Float64[jnp.ndarray, "n_samples"]:
    """
    Calculate the values of the SVM decision function.

    Args:
        theta_jax: List of circuit parameters
        U: Quantum circuit
        setting_dict: Dictionary containing calculation settings:
            - lambda_list: List of lambda values for each eigenvector
            - K: Number of eigenvectors
            - b: Bias term of the decision function

    Returns:
        JAX array with decision function values of shape (n_samples,)
    """
    # Reshape lambda_list to (1, K) for broadcasting
    lambda_jax = jnp.array(setting_dict["lambda_list"]).reshape(1, setting_dict["K"])

    # Calculate the absolute value of the inner product. Custom differentiation is defined only for this line
    abs_inner_product_vector = abs_inner_product_fn(
        theta_jax=theta_jax, U=U, setting_dict=setting_dict
    )

    # Square the absolute value of the inner product element-wise
    abs_inner_product_squared = abs_inner_product_vector**2

    # Calculate the decision function: sum(lambda * inner_product^2) + b
    # This performs a weighted sum across the K dimension (axis=1)
    value_vector = (
        jnp.sum(lambda_jax * abs_inner_product_squared, axis=1) + setting_dict["b"]
    )

    return value_vector
