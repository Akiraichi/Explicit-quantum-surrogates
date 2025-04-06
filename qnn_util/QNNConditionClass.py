import datetime
import glob
import os
from dataclasses import dataclass

import joblib  # type: ignore
import optax  # type: ignore
from optax._src.base import GradientTransformation  # type: ignore

from qnn_util.errors import ConfigurationError


@dataclass(frozen=True)
class QNNCondition:
    """
    Data class that summarizes the main parameters of quantum neural network experiments.

    This class stores configuration parameters for quantum neural network (QNN) experiments,
    including dataset settings, circuit configuration, optimization parameters, and
    parallel processing options.

    Attributes are organized into several categories:
    - General settings (seed, experiment name, etc.)
    - Experiment conditions (circuit type, target label, etc.)
    - Dataset configuration
    - Parallel processing options
    - Cost function settings
    - Optimization parameters
    """

    # General settings
    seed: int  # Random seed for reproducibility, not used
    exp_name: str  # Name of the experiment
    debug_print: bool  # Whether to print debug information
    n_qubits: int  # Number of qubits in the quantum circuit

    # Experiment conditions
    circuit_type: str  # Type of quantum circuit to use
    k: int  # Number of eigenvectors to embed in EQS
    target_label: int  # Target label for binary classification

    # Dataset settings
    ds_name: str  # Dataset name
    fidelity: (
        float | None
    )  # Dataset fidelity setting. Only effective when MNISQ dataset is selected
    n_data_by_label: (
        int  # Number of data points per label. Currently only effective for MNISQ dataset
    )
    svm_trained_label: list[int]  # Specify labels to load from the dataset
    test_size: float  # Proportion of test data
    start_index: int  # Starting index in the dataset to begin using data

    # Parallel processing options
    fwd_parallel: bool  # Whether to parallelize forward prediction calculations
    bwd_parallel: bool  # Whether to parallelize backward gradient calculations
    fwd_chunk_size: int  # Number of items to process in parallel for forward pass, not used
    bwd_chunk_size: int  # Number of items to process in parallel for backward pass, not used
    n_jobs: int  # Number of parallel processes

    # Cost function settings
    balanced: bool  # Whether to use class weights
    # Set only one of these to True (str cannot be used in JAX)
    use_log_loss: bool  # Whether to use cross-entropy loss
    use_hinge_loss: bool  # Whether to use hinge loss

    # Optimization settings
    maxiter: int  # Maximum number of iterations. More iterations increase accuracy but take more time
    solver_name: str  # Optimization algorithm. Others like `Adam()` or `NelderMead()` can be used
    batch_size: int  # Batch size for optimization processing
    start_learning_rate: float  # Learning rate for Adam optimizer
    b1: float  # Adam beta1 parameter
    b2: float  # Adam beta2 parameter
    epsilon: float  # Small constant for numerical stability
    eps_root: float  # Small constant for root operations

    def get_save_path(self) -> str:
        """
        Generate the full path for saving QNN experiment data.

        This method creates a directory structure based on the experiment name and
        configuration, then generates a filename with the current timestamp.

        Returns:
            str: Full path for saving QNN experiment data
        """
        # Directory
        folder_name = os.path.join(
            self.get_first_folder_name(), self.get_second_folder_name()
        )
        if not os.path.exists(folder_name):
            os.makedirs(folder_name, exist_ok=True)

        # File
        today = datetime.datetime.today().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )  # Example: '2024-06-18-14-40-24'
        file_name = self.get_file_name(today)

        path = os.path.join(folder_name, file_name)
        return path

    def get_file_name(self, today) -> str:
        """
        Generate a filename for saving QNN experiment data.

        Args:
            today: String representing the current date and time

        Returns:
            str: Formatted filename for QNN experiment data
        """
        return f"qnn_{self.get_second_folder_name()}_{today}.jb"

    def get_first_folder_name(self) -> str:
        """
        Get the top-level folder name for QNN experiment data storage.

        This method creates a folder name based on the experiment name and circuit type.

        Returns:
            str: Path to the top-level folder
        """
        top_folder_name = "exp_result"
        return os.path.join(
            top_folder_name, self.exp_name + f"qnn_data_{self.circuit_type}"
        )

    def get_second_folder_name(self) -> str:
        """
        Generate a subfolder name based on experiment parameters.

        This method creates a detailed folder name that includes all relevant
        experiment parameters such as dataset name, fidelity, number of qubits,
        and optimization settings.

        Returns:
            str: Formatted subfolder name
        """
        if self.fidelity is None:
            fidelity_str = "None"
        else:
            fidelity_str = str(int(self.fidelity * 100))
        return f"ds_name={self.ds_name}-fidelity={fidelity_str}_n_qubits={str(self.n_qubits).zfill(2)}-n_label={str(len(self.svm_trained_label)).zfill(2)}-seed={self.seed}-test_size={self.test_size}-solvername={self.solver_name}-maxiter={self.maxiter}-balanced={self.balanced}-batchsize={self.batch_size}-targetlabel={self.target_label}"

    def load_saved_data(self, add_path: str = ""):
        """
        Load saved experiment data.

        This method searches for saved experiment data files matching the current
        configuration and loads the data from the file.

        Args:
            add_path: Optional additional path prefix to search in

        Returns:
            The loaded experiment data

        Raises:
            ConfigurationError: If no file or multiple files match the search criteria
        """
        # Directory
        folder_name = os.path.join(
            self.get_first_folder_name(), self.get_second_folder_name()
        )
        # Path to file
        if add_path:
            file_paths = glob.glob(f"{add_path}/{folder_name}/*.jb")
        else:
            file_paths = glob.glob(f"{folder_name}/*.jb")
        file_paths.sort()

        if len(file_paths) != 1:
            raise ConfigurationError(
                f"Error in number of data files: {file_paths}. Expected: 1, Actual: {len(file_paths)}"
            )

        # Load data
        data = joblib.load(file_paths[0])
        return data
