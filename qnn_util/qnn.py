import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import jax.numpy as jnp
import numpy as np
import optax  # type: ignore
from jax import grad
from jaxtyping import Complex128, Float64, Int
from optax._src.base import GradientTransformation  # type: ignore
from skqulacs.qnn import QNNClassifier  # type: ignore

from misc.util import debug_print
from qnn_util.cost_func_revise import loss_fn, svm_decision_fn
from qnn_util.helper import FirstKHelper
from qnn_util.LearningCircuitClass import MyLearningCircuit
from qnn_util.OptimizerCallbackClass import OptimizerCallback
from qnn_util.QNNConditionClass import QNNCondition


@dataclass(eq=False)
class MyQNNClassifier:
    """
    This class implements a quantum neural network for classification tasks that supports
    various optimization methods and cost functions.
    """

    def __init__(
        self,
        circuit: MyLearningCircuit,
        lambda_k_list_list: List[List[float]],
        b_list: List[float],
        qnn_condition: QNNCondition,
        callback_obj: OptimizerCallback | None,
    ):
        """
        Initialize MyQNNClassifier.

        Args:
            lambda_k_list_list: List of lists containing lambda values for each model
            b_list: List of bias terms for each binary SVM model
            qnn_condition: Data class describing QNN settings
            callback_obj: Callback object for tracking optimization progress
        """
        self.circuit = circuit
        self.solver = MyQNNClassifier.get_solver(qnn_condition=qnn_condition)
        self.lambda_k_list_list = lambda_k_list_list  # Lambda values for each model
        self.b_list = b_list  # Bias terms for each binary SVM model
        self.qnn_condition = qnn_condition
        self.callback_obj = callback_obj
        self.grad_fn = grad(
            loss_fn, argnums=0
        )  # Get the gradient function for the cost function, differentiating with respect to the first positional argument
        self.setting_dict = {
            #
            "x_list": Complex128[np.ndarray, "_n_samples n_features"] | None,
            "y_list": Int[np.ndarray, "_n_samples"] | None,
            "lambda_list": self.lambda_k_list_list[self.qnn_condition.target_label],
            # TODO: Verify that the indices are ordered by target_label
            "b": self.b_list[self.qnn_condition.target_label],
            "n_qubits": self.qnn_condition.n_qubits,
            #
            "K": len(self.lambda_k_list_list[self.qnn_condition.target_label]),
            "n_samples": Optional[int],
            "len_theta": len(self.circuit.get_parameters()) - FirstKHelper.param_count,
            # Parameters excluding those used to create |k>. These are the parameters subject to differentiation
            # Cost function settings
            "use_log_loss": qnn_condition.use_log_loss,
            "use_hinge_loss": qnn_condition.use_hinge_loss,
            # Parallel processing settings
            "balanced": self.qnn_condition.balanced,
            "bwd_parallel": self.qnn_condition.bwd_parallel,
            "fwd_parallel": self.qnn_condition.fwd_parallel,
            "bwd_chunk_size": self.qnn_condition.bwd_chunk_size,
            "fwd_chunk_size": self.qnn_condition.fwd_chunk_size,
            "n_jobs": self.qnn_condition.n_jobs,
            # Debug settings
            "debug_print": self.qnn_condition.debug_print,
        }

    @staticmethod
    def get_solver(qnn_condition: QNNCondition) -> GradientTransformation:
        if qnn_condition.solver_name == "adam":
            optimizer = optax.adam(
                learning_rate=qnn_condition.start_learning_rate,
                b1=qnn_condition.b1,
                b2=qnn_condition.b2,
                eps=qnn_condition.epsilon,
                eps_root=qnn_condition.eps_root,
            )
            return optimizer
        elif qnn_condition.solver_name == "radam":
            optimizer = optax.radam(
                learning_rate=qnn_condition.start_learning_rate,
                b1=qnn_condition.b1,
                b2=qnn_condition.b2,
                eps=qnn_condition.epsilon,
                eps_root=qnn_condition.eps_root,
            )
            return optimizer
        else:
            raise RuntimeError(f"{qnn_condition.solver_name} is not defined")

    def predict(
        self, x: Complex128[np.ndarray, "n_samples n_features"]
    ) -> Int[np.ndarray, "n_samples"]:
        """
        Predict class labels for each input data point in `x`.

        This method applies the trained quantum neural network to predict binary class labels
        (0 or 1) for each input data point.

        Args:
            x: Input data array of shape (n_samples, n_features).

        Returns:
            y_pred: Predicted class labels of shape (n_samples,).
                   Takes values 0 or 1 based on the sign of the decision function.
        """
        if x.ndim == 1:
            x = x.reshape((-1, 1))

        # Calculate the decision function and convert to binary predictions
        decision_vector = self._predict_inner(x)
        y_pred = np.where(decision_vector >= 0, 1, 0)

        return y_pred

    def get_setted_dict(
        self,
        x: Complex128[np.ndarray, "n_samples n_features"] | None,
        y: Int[np.ndarray, "n_samples"] | None = None,
    ) -> dict:
        """Prepare the settings dictionary"""
        setting_dict = self.setting_dict.copy()
        if x is not None:
            setting_dict["x_list"] = x
            setting_dict["n_samples"] = len(x)
        if y is not None:
            setting_dict["y_list"] = y

        return setting_dict

    def _predict_inner(
        self, x: Complex128[np.ndarray, "n_samples n_features"]
    ) -> Float64[np.ndarray, "n_samples"]:
        """
        Internal prediction method that calculates the decision function values.

        Args:
            x: Input data array. For example, with 10 qubits and 5000 data points,
               the shape would be (5000, 1024).

        Returns:
            Array of decision function values

        Raises:
            AssertionError: If the number of classes is not 2, or if manyclass is True
        """

        # Initialize the first RX gates to 0.0 to make them identity gates
        # TODO: Consider processing them as input gates for improved efficiency
        theta_helper: list[float] = [0.0] * FirstKHelper.param_count
        self.circuit.set_parameters4k_state(theta=theta_helper)

        # Get circuit parameters (excluding the first k parameters)
        theta_list: list[float] = self.circuit.get_parameters(exclude_first_k=True)
        theta_jax = jnp.array(theta_list)

        # Prepare the settings dictionary including input data
        setting_dict = self.get_setted_dict(x=x)

        # Calculate the decision function values
        svm_decision_vector = svm_decision_fn(
            theta_jax=theta_jax, U=self.circuit, setting_dict=setting_dict
        )

        return np.asarray(svm_decision_vector)

    def fit(
        self,
        x_train: Complex128[np.ndarray, "n_samples n_features"],
        y_train: Int[np.ndarray, "n_samples"],
        maxiter: int | None = None,
    ) -> tuple[Float64[np.ndarray, ""], Float64[np.ndarray, "len_theta"]]:
        """
        Train the quantum neural network with the provided data.

        Args:
            x_train: Training data input with shape (n_samples, n_features).
                    If a 1D array is provided, it will be reshaped to a 2D array (i.e., for a single data point).
            y_train: Training data labels with shape (n_samples,).
                    Labels must be represented as integers.
            maxiter: Maximum number of optimization iterations.
                    If None, the value from qnn_condition will be used.

        Returns:
            loss: Loss value after training (may be None for some optimizers)
            theta: Optimized circuit parameters

        Raises:
            ValueError: If the solver name is not recognized
        """
        # Prepare training data
        y_scaled = y_train

        if x_train.ndim == 1:
            x_train = x_train.reshape((-1, 1))

        # Get initial parameters (excluding the first k parameters)
        theta_init: list = self.circuit.get_parameters(exclude_first_k=True)
        theta_init_jax = jnp.array(theta_init)

        # Select optimization method based on solver name
        if self.qnn_condition.solver_name in ["adam", "radam"]:
            cost_value, params = self._optimize_with_jax_optimizer(
                theta_init_jax, x_train, y_scaled, with_schedule=False
            )

        elif self.qnn_condition.solver_name in ["adamw_schedule"]:
            cost_value, params = self._optimize_with_jax_optimizer(
                theta_init_jax, x_train, y_scaled, with_schedule=True
            )

        else:
            raise RuntimeError("ソルバーを確認してください")
        return cost_value, params

    def _optimize_with_jax_optimizer(
        self,
        theta_jax: Float64[jnp.ndarray, "len_theta"],
        x_scaled: Complex128[np.ndarray, "n_samples n_features"],
        y_scaled: Int[np.ndarray, "n_samples"],
        with_schedule: bool = False,
    ) -> tuple[Float64[np.ndarray, ""], Float64[np.ndarray, "len_theta"]]:
        """
        Optimize model parameters using a JAX-based optimizer.

        Args:
            theta_jax: Initial parameters for optimization
            x_scaled: Scaled input data
            y_scaled: Target labels
            with_schedule: Whether to use a learning rate schedule (for adamw_schedule)

        Returns:
            Tuple of (loss, optimized parameters)
        """

        # Define update function based on whether to use schedule
        # @jit
        def __update_fn(theta, opt_state, grads):
            if with_schedule:
                updates, new_opt_state = self.solver.update(grads, opt_state, theta)
            else:
                updates, new_opt_state = self.solver.update(grads, opt_state)
            new_theta = optax.apply_updates(theta, updates)
            return new_theta, new_opt_state

        # Initialize optimizer state
        opt_state = self.solver.init(theta_jax)

        # Optimization loop
        start_time = time.time()
        total_iterations = self.qnn_condition.maxiter * len(x_scaled)
        batch_size = self.qnn_condition.batch_size

        for iteration in range(0, total_iterations, batch_size):
            # Get batch data
            batch_start = iteration % len(x_scaled)
            batch_end = batch_start + batch_size
            batch_x = x_scaled[batch_start:batch_end]
            batch_y = y_scaled[batch_start:batch_end]

            # Log batch information if debug is enabled
            debug_print(
                f"Batch size: x={len(batch_x)}, y={len(batch_y)}",
                debug_print=self.qnn_condition.debug_print,
            )

            # Calculate gradients
            grads_jax = self._cost_func_grad(theta_jax, batch_x, batch_y)
            gradient_norm = np.sum(grads_jax**2)  # TODO: Is float() needed?
            debug_print(
                f"Gradient norm: {gradient_norm}",
                debug_print=self.qnn_condition.debug_print,
            )

            # Update parameters
            theta_jax, opt_state = __update_fn(
                theta=theta_jax, opt_state=opt_state, grads=grads_jax
            )

            # Display estimated remaining time
            if iteration > batch_size:
                elapsed_time = time.time() - start_time
                remaining_iterations = total_iterations - iteration
                estimated_time_hours = (
                    elapsed_time * remaining_iterations / batch_size / 3600
                )
                print(
                    f"Iteration: {iteration}/{total_iterations}: "
                    f"Estimated remaining time: {estimated_time_hours:.2f} hours"
                )
                start_time = time.time()

            # Calculate cost at epoch boundaries
            if iteration % len(x_scaled) == 0:
                current_cost = self.cost_func(theta_jax, x_scaled, y_scaled)
                print(f"Cost function value: {current_cost}")
                if self.callback_obj is not None:
                    self.callback_obj.call_back_adam(
                        np.asarray(theta_jax),
                        gradient_norm,
                        iteration,
                        current_cost,
                        np.asarray(grads_jax),
                    )

        # Update circuit with final parameters
        self.circuit.update_parameters(
            [0.0] * FirstKHelper.param_count + theta_jax.tolist()
        )
        current_cost = self.cost_func(theta_jax, x_scaled, y_scaled)
        return current_cost, np.asarray(theta_jax)

    def cost_func(
        self,
        theta_jax: Float64[jnp.ndarray, "len_theta"],
        x_scaled: Complex128[np.ndarray, "n_samples n_features"],
        y_scaled: Int[np.ndarray, "n_samples"],
    ) -> Float64[np.ndarray, ""]:
        """
        Calculate the cost function value for the given parameters and data.

        Args:
            theta_jax: List of circuit parameters
            x_scaled: Scaled input data
            y_scaled: Target labels

        Returns:
            Cost function value (floating point number)

        Raises:
            NotImplementedError: If the specified cost function is not implemented
        """

        # Prepare settings dictionary including input data
        setting_dict = self.setting_dict.copy()
        setting_dict["x_list"] = x_scaled
        setting_dict["y_list"] = y_scaled
        setting_dict["n_samples"] = len(x_scaled)

        loss_value = loss_fn(
            theta_jax=theta_jax, U=self.circuit, setting_dict=setting_dict
        )  # Should be a scalar
        return np.asarray(loss_value)

    def _cost_func_grad(
        self,
        theta_jax: Float64[jnp.ndarray, "len_theta"],
        x_scaled: Complex128[np.ndarray, "n_samples n_features"],
        y_scaled: Int[np.ndarray, "n_samples"],
    ) -> jnp.ndarray:
        """
        Calculate the gradient of the cost function with respect to parameters.

        Args:
            theta_jax: List of circuit parameters
            x_scaled: Scaled input data
            y_scaled: Target labels

        Returns:
            Gradient of the cost function with respect to parameters
        """
        # Update circuit parameters (add FirstKHelper.param_count zeros for RX gates)
        self.circuit.update_parameters(
            [0.0] * FirstKHelper.param_count + theta_jax.tolist()
        )

        # Get gradient function for cost loss
        # grad_fn = grad(loss_fn, argnums=0)  # Differentiate with respect to the first positional argument

        # Prepare settings dictionary including input data
        setting_dict = self.get_setted_dict(x=x_scaled, y=y_scaled)

        # Calculate gradients
        grads_jax = self.grad_fn(theta_jax, self.circuit, setting_dict)

        return grads_jax

    def return_svm_decision_vector(
        self, x: Complex128[np.ndarray, "n_samples n_features"]
    ) -> Float64[np.ndarray, "n_samples"]:
        """
        Get the values of the decision function using the current parameters.

        This method returns the raw decision function values without converting them to predictions
        like the predict method does.

        Args:
            x: Input data array

        Returns:
            Array of decision function values
        """

        if x.ndim == 1:
            x = x.reshape((-1, 1))

        decision_vector = self._predict_inner(x)

        return decision_vector
