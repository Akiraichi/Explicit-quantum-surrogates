import datetime
import os

import joblib  # type: ignore
import numpy as np
from jax import numpy as jnp
from jaxtyping import Float64

from qnn_util.QNNConditionClass import QNNCondition


class OptimizerCallback:
    """
    Callback class for tracking optimization progress during QNN training.

    This class stores the history of parameters, costs, gradients, and iteration numbers
    during the optimization process. It provides methods to save this data for later analysis.

    Attributes:
        qnn_condition: Configuration for the quantum neural network
        theta_history: List of parameter values at each callback point
        cost_history: List of cost function values at each callback point
        iter_history: List of iteration numbers at each callback point
        grad_norm_history: List of gradient norm values at each callback point
        grads_history: List of gradient vectors at each callback point
    """
    def __init__(self, qnn_condition: QNNCondition):
        """
        Initialize the OptimizerCallback with a QNN condition.

        Args:
            qnn_condition: Configuration for the quantum neural network
        """
        self.qnn_condition: QNNCondition = qnn_condition
        self.theta_history: list[Float64[np.ndarray, "len_theta"]] = []
        self.cost_history: list[Float64[np.ndarray, ""]] = []
        self.iter_history: list[int] = []
        self.grad_norm_history: list[Float64[jnp.ndarray, ""]] = []
        self.grads_history: list[Float64[np.ndarray, "len_theta"]] = []

    def call_back_adam(
        self,
        theta_now: Float64[np.ndarray, "len_theta"],
        grad_norm: Float64[jnp.ndarray, ""],
        iter_: int,
        now_cost: Float64[np.ndarray, ""],
        grads: Float64[np.ndarray, "len_theta"],
    ):
        """
        Callback function for the Adam optimizer.

        This method is called during optimization to record the current state of the training process.
        It appends the current parameter values, cost, gradient norm, iteration number, and gradients
        to their respective history lists, then saves the updated data.

        Args:
            theta_now: Current parameter values
            grad_norm: Norm of the current gradient
            iter_: Current iteration number
            now_cost: Current cost function value
            grads: Current gradient vector
        """
        self.theta_history.append(theta_now)
        self.cost_history.append(now_cost)
        self.grad_norm_history.append(grad_norm)
        self.iter_history.append(iter_)
        self.grads_history.append(grads)
        self.save()  # TODO: Could improve memory usage and save speed by only saving the newly added data.

    def save(self) -> None:
        """
        Save the current optimization history to a file.

        This method creates a directory structure based on the QNN condition and
        iteration number, then saves the current state of the optimization process
        to a file with a timestamp in the filename.
        """
        # Specify data to save
        datas = {
            "iter": iter,
            "iter_history": self.iter_history,
            "theta_history": self.theta_history,
            "cost_history": self.cost_history,
            "grad_norm_history": self.grad_norm_history,
            "grads_history": self.grads_history,
        }
        # Directory structure
        third_folder_name = self.get_third_folder_name()
        folder_path = os.path.join(
            self.qnn_condition.get_first_folder_name(),
            self.qnn_condition.get_second_folder_name(),
            third_folder_name,
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        # File
        today = datetime.datetime.today().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )  # Example: '2024-06-18-14-40-24'
        file_name = self.get_file_name(today)

        # Save
        folder_path = os.path.join(folder_path, file_name)
        joblib.dump(value=datas, filename=folder_path)

    def get_third_folder_name(self) -> str:
        """
        Generate the third level folder name based on the current iteration number.

        This method creates a folder name by dividing the current iteration number by 100
        and zero-padding the result to 6 digits. This helps organize saved files into
        groups of 100 iterations.

        Returns:
            A string representing the folder name
        """
        return f"{str(self.iter_history[-1] // 100).zfill(6)}"

    def get_file_name(self, today: str) -> str:
        """
        Generate a filename for saving the current optimization state.

        This method creates a filename that includes the QNN condition name,
        the current iteration number, and a timestamp.

        Args:
            today: A string representing the current date and time

        Returns:
            A string representing the filename
        """
        return f"callback_{self.qnn_condition.get_second_folder_name()}_iter{str(self.iter_history[-1]).zfill(5)}_{today}.jb"
