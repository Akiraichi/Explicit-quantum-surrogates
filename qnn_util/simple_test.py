"""
Simple Test Script for QNN Cost Function

This script tests the quantum neural network (QNN) cost function and its gradient calculation.
It creates a simple quantum circuit, generates dummy input data and labels, and then:
1. Calculates the loss value using the cost function
2. Computes gradients using JAX automatic differentiation
3. Verifies the gradients using finite difference checks

This is useful for testing the correctness of the cost function implementation
and ensuring that automatic differentiation works properly with quantum circuits.
"""

import numpy as np
from jax import config, grad
from jax import numpy as jnp
from jax._src.test_util import check_grads
from qulacs import QuantumState

from qnn_util.cost_func_revise import loss_fn
from qnn_util.LearningCircuitClass import MyLearningCircuit

config.update("jax_enable_x64", True)

if __name__ == "__main__":
    # Constants
    n_qubits = 10
    K = 10
    n_samples = 100

    # Create a dummy circuit
    U = MyLearningCircuit(n_qubit=n_qubits)

    # Circuit for |k⟩ state preparation
    for i in range(n_qubits):
        U.add_parametric_RZ_gate(index=i, parameter=0.0)
        U.add_parametric_RX_gate(index=i, parameter=0.0)
        U.add_parametric_RZ_gate(index=i, parameter=0.0)
    # AQCE circuit follows...
    for i in range(n_qubits):
        U.add_parametric_RY_gate(index=i, parameter=1.0)
        U.add_parametric_RZ_gate(index=i, parameter=2.0)
        U.add_parametric_RX_gate(index=i, parameter=3.0)

    len_theta = len(U.get_parameters(exclude_first_k=True))

    # Initialize parameter list
    # theta_list = [0.0]*n_circuit_params
    # np.random.seed(0)
    theta_ndarray = np.random.rand(len_theta)
    theta_jax = jnp.array(theta_ndarray)

    # List of dummy input data
    x_list = []
    for _ in range(n_samples):
        state = QuantumState(qubit_count=n_qubits)
        # state.set_Haar_random_state()
        _x = state.get_vector()
        # _x = np.random.rand(2 ** n_qubits)
        # _x /= np.linalg.norm(_x)
        x_list.append(_x)
    x_ndarray = np.array(x_list)

    # List of dummy output labels
    y_ndarray = np.random.randint(0, 2, size=n_samples)  # Binary labels

    # Initialize lambda_list
    lambda_list = np.random.rand(K).tolist()
    # lambda_list = np.random.rand(K).tolist()
    #
    # Initialize b_list
    b = 0
    # Test
    setting_dict = {
        "balanced": True,
        "fwd_parallel": False,
        "bwd_parallel": False,
        "fwd_chunk_size": 1,
        "bwd_chunk_size": 1,
        "n_jobs": 10,
        "x_list": x_ndarray,
        "y_list": y_ndarray,
        "lambda_list": lambda_list,
        "b": b,
        "n_qubits": n_qubits,
        "K": len(lambda_list),
        "n_samples": len(x_list),
        "debug_print": False,
        "use_log_loss": True,
        "use_hinge_loss": False,
        "len_theta": len_theta,
    }

    loss_value = loss_fn(theta_jax=theta_jax, U=U, setting_dict=setting_dict)
    print("Loss value:", loss_value)

    # Calculate gradients
    grad_fn = grad(loss_fn)
    grads = grad_fn(theta_jax, U, setting_dict)

    print("JAX automatic differentiation:", grads)

    # Finite difference check
    # Fix circuit and state_bra in advance
    def loss_fn_wrapped(theta_jax):
        """
        Wrapper function for loss_fn that fixes the circuit and settings.

        This function is used for gradient checking with finite differences.

        Args:
            theta_jax: Circuit parameters

        Returns:
            Loss value for the given parameters
        """
        return loss_fn(theta_jax, U, setting_dict)

    check_grads(loss_fn_wrapped, args=(theta_jax,), order=1, modes=["rev"], atol=1e-20)
